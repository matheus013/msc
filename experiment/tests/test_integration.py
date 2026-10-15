from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from simulation.core.device import get_device, get_device_for_batch
from simulation.core.inventory_env import InventoryEnv
from simulation.core.inventory_env_torch import BatchInventoryEnv
from simulation.preprocessing.sales import prepare_sales


def small_config(device="auto"):
    return {
        "SIMULATION": {"lead_time": 2, "initial_inventory": 20, "device": device},
        "COST": {"holding_cost_per_unit": 1, "stockout_cost_per_unit": 5,
                 "ordering_cost_per_order": 3, "ordering_cost_per_unit": 0.5},
        "GENETIC_ALGORITHM": {"population_size": 8, "n_generations": 2, "seed": 42},
    }


def test_device_environment_changes_after_cached_resolution(monkeypatch):
    get_device("auto")
    monkeypatch.setenv("AIPE_DEVICE", "cpu")
    assert get_device().type == "cpu"
    monkeypatch.delenv("AIPE_DEVICE")
    if torch.cuda.is_available():
        assert get_device_for_batch(1).type == "cuda"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_threshold_matches_scalar_environment(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA indisponivel")
    demand = np.array([0, 5, 18, 0, 35, 10, 4, 28, 0, 9], dtype=float)
    config = small_config(device)
    rop, quantity, safety = [10, 25], [15, 30], [0, 3]
    env = BatchInventoryEnv(demand, config, batch_size=2, device=torch.device(device))
    tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
    batched = env.run_threshold(tensor(rop), tensor(quantity), tensor(safety))
    for index in range(2):
        scalar = InventoryEnv(demand, config).run_policy(
            lambda state, e: quantity[index] if e.inventory + sum(e.pipeline) <= rop[index] + safety[index] else 0,
            n_reps=1,
        )["kpis"]
        for metric in ["TIC", "ServiceLevel", "StockoutRate", "BullwhipEffect", "OrderFrequency"]:
            assert batched[metric][index].item() == pytest.approx(scalar[metric], rel=1e-5, abs=1e-5)


def test_sales_combines_files_keeps_warehouse_and_commercial_cycle(tmp_path):
    source = tmp_path / "raw"
    pb, ba = source / "uf=PB", source / "uf=BA"
    pb.mkdir(parents=True)
    ba.mkdir()
    row = dict(produto_cod=[1], revendedor_cod=[2], venda_ciclo=[202313],
               venda_qtd=[3.0], venda_vlr_receita_liquida=[12.0])
    pd.DataFrame(row).to_parquet(pb / "a.parquet")
    pd.DataFrame(row).to_parquet(pb / "b.parquet")
    pd.DataFrame(row).to_parquet(ba / "a.parquet")
    output = tmp_path / "prepared.csv"
    prepare_sales(source, output, states=["PB"])
    result = pd.read_csv(output)
    assert result["warehouse"].tolist() == ["PB"]
    assert result.loc[0, "sales"] == 6
    assert result.loc[0, "revenue"] == 24
    assert result.loc[0, "avg_price"] == 4
    assert result.loc[0, "date"] == "2023-09-10"
    all_states = tmp_path / "all.csv"
    prepare_sales(source, all_states)
    assert set(pd.read_csv(all_states)["warehouse"]) == {"BA", "PB"}
    with pytest.raises(FileExistsError):
        prepare_sales(source, output)
    with pytest.raises(ValueError):
        prepare_sales(source, source / "new.csv")


def test_adapter_carries_device_into_optimizers():
    from simulation.pipelines.inventory_simulation.nodes import _build_cfg, _n_workers
    assert _build_cfg({"device": "cpu"})["SIMULATION"]["device"] == "cpu"
    assert _build_cfg({})["DQN"]["device"] == "auto"
    if torch.cuda.is_available():
        assert _n_workers({"device": "auto", "n_workers": 7}) == 1


def test_shared_ga_and_rl_train_on_selected_device():
    from simulation.core.policies import GAPolicyOptimizer, DQNPolicy, PPOPolicy
    config = small_config()
    config["DQN"] = dict(episodes=2, batch_size=2, n_actions=3, hidden_layers=[8],
                         memory_size=20, max_order_qty=20, device="auto")
    config["PPO"] = dict(episodes=2, n_actions=3, hidden_layers=[8],
                         max_order_qty=20, minibatch_size=4, update_epochs=1, device="auto")
    demand = np.array([0, 4, 10, 2, 15, 0, 6, 9], dtype=float)
    ga = GAPolicyOptimizer(demand, config)
    assert np.isfinite(ga.optimize(verbose=False)["cost"])
    assert ga.device == get_device()
    for cls in (DQNPolicy, PPOPolicy):
        agent = cls(InventoryEnv.STATE_DIM, config)
        agent.train(demand, config, verbose=False)
        assert agent.device == get_device()


def test_forecasting_backends_and_gpu_training():
    from simulation.core.forecasting import create_lstm, LSTMNumpy, XGBoostForecaster
    assert isinstance(create_lstm(3, {"backend": "numpy"}), LSTMNumpy)
    features = np.arange(60, dtype=float).reshape(20, 3) / 10
    target = features[:, -1] + 2
    model = create_lstm(3, {"hidden_size": 4})
    model.fit(features, target, epochs=2, batch_size=8, verbose=False)
    assert model.device == get_device()
    assert np.isfinite(model.predict(features[:3])).all()
    regressor = XGBoostForecaster(n_estimators=2, max_depth=2)
    regressor.fit(features, target)
    assert np.isfinite(regressor.predict(features[:3])).all()


def test_kedro_configuration_and_shared_raw_source(monkeypatch):
    from kedro.framework.startup import bootstrap_project
    from kedro.framework.session import KedroSession
    from simulation.paths import EXPERIMENT_ROOT, RAW_DATA_ROOT
    from simulation.pipeline_registry import register_pipelines
    monkeypatch.setenv("KEDRO_DISABLE_TELEMETRY", "1")
    monkeypatch.chdir(EXPERIMENT_ROOT)
    bootstrap_project(EXPERIMENT_ROOT)
    with KedroSession.create(project_path=EXPERIMENT_ROOT, save_on_close=False) as session:
        context = session.load_context()
        dataset = context.catalog["vendas_partitioned"]
        assert dataset is not None
        raw_path = context.config_loader["catalog"]["vendas_partitioned"]["path"]
        assert Path(raw_path).resolve() == RAW_DATA_ROOT / "source/vendas"
        assert context.params["simulation"]["device"] == "auto"
    assert len(register_pipelines()["__default__"].nodes) > 10
