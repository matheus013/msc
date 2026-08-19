"""
inventory_simulation/nodes.py — Execução das 12 políticas de inventário.

Fluxo:
  scenarios + scenarios_meta + params -> scale_parameters_per_store -> scaled_params
  scenarios + scaled_params + params -> run_classical_policies    -> kpis_classical
  scenarios + scaled_params + params -> run_metaheuristic_policies -> kpis_metaheuristic
  scenarios + scaled_params + params -> run_rl_policies            -> kpis_rl
  scenarios + scaled_params + params -> run_proposed_architecture  -> kpis_proposed
  kpis_classical + kpis_metaheuristic + kpis_rl + kpis_proposed -> aggregate_kpis -> kpis

Paralelizacao (2026-08-18)
---------------------------
Cada serie (warehouse,store_id,item_id) e 100% independente das demais --
nao ha estado compartilhado entre elas em nenhum dos 6 nos de politica
(cada uma le seu proprio cfg escalado e sua propria demanda). Ate aqui,
porem, o loop era sequencial num unico processo (1 nucleo de 16 usados).
`_run_parallel_policies` despacha as series para um `ProcessPoolExecutor`
(processos, nao threads -- evita o conflito de libomp entre torch e
xgboost/sklearn documentado em REIMPLEMENTACAO_SOTA.md, que e especifico
de MULTIPLAS THREADS num MESMO processo; processos separados nao
compartilham essa instancia). Cada serie usa sementes (`cfg[...]["seed"]`,
`params["random_seed"]`) fixas e locais -- nao ha RNG global acumulado
entre series mesmo no codigo sequencial original, entao o resultado
numerico e IDENTICO independente de paralelizar ou nao (validado
empiricamente, ver AJUSTES_INFRA_2026-08-18.md item 19).
"""
import concurrent.futures
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

log = logging.getLogger(__name__)


def _n_workers(params: dict) -> int:
    configured = params.get("n_workers")
    if configured is not None:
        return max(1, int(configured))
    return max(1, (os.cpu_count() or 4) - 2)


def _run_parallel_policies(scenarios: pd.DataFrame, scenarios_meta: pd.DataFrame,
                            scaled_params: dict, params: dict,
                            worker_fn, worker_extra_args: tuple, log_tag: str) -> pd.DataFrame:
    """
    Executa `worker_fn(key, cfg, demand, meta_row, params, *worker_extra_args)
    -> (rows, logs)` para cada serie, em paralelo quando n_workers > 1.

    `worker_fn` precisa ser uma funcao de MODULO (nao aninhada/lambda) para
    ser picklable pelo ProcessPoolExecutor no Windows (metodo spawn).
    Extrai demand/meta_row no processo principal (barato, vetorizado) para
    nao repetir esse trabalho -- e nao precisar pickled `scenarios`
    inteiro -- em cada tarefa.
    """
    meta_idx = scenarios_meta.set_index(["warehouse", "store_id", "item_id"])
    tasks = []
    for key, cfg in scaled_params.items():
        demand = _get_series(scenarios, key)
        if len(demand) < 5:
            continue
        meta_row = meta_idx.loc[key].to_dict() if key in meta_idx.index else {}
        tasks.append((key, cfg, demand, meta_row))

    if not tasks:
        return pd.DataFrame()

    n_workers = min(_n_workers(params), len(tasks))
    rows_all: list = []

    def _emit(logs):
        for m in logs:
            if "failed" in m:
                log.warning(m)
            else:
                log.info(m)

    if n_workers <= 1:
        for key, cfg, demand, meta_row in tasks:
            rows, logs = worker_fn(key, cfg, demand, meta_row, params, *worker_extra_args)
            _emit(logs)
            rows_all.extend(rows)
        return pd.DataFrame(rows_all)

    log.info("[%s] paralelizando %d series em %d processos", log_tag, len(tasks), n_workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(worker_fn, key, cfg, demand, meta_row, params, *worker_extra_args): key
            for key, cfg, demand, meta_row in tasks
        }
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            done += 1
            try:
                rows, logs = fut.result()
                _emit(logs)
                rows_all.extend(rows)
            except Exception as e:
                log.warning("[%s] série %s falhou no worker: %s", log_tag, key, e)
            if done % 200 == 0 or done == len(tasks):
                log.info("[%s] %d/%d séries concluídas", log_tag, done, len(tasks))

    return pd.DataFrame(rows_all)


def _worker_generic(key, cfg, demand, meta_row, params, policy_specs, log_tag):
    """
    Worker de processo para politicas "construct" (classicas/SOTA/Zabraoui),
    "optimize" (meta-heuristicas) e "rl_train" (RL). Mesma logica dos loops
    sequenciais originais, isolada por serie -- roda dentro de UM worker do
    ProcessPoolExecutor (ou inline, se n_workers=1).
    """
    from simulation.core.inventory_env import InventoryEnv
    w, s, i = key
    rows = []
    logs = [f"[{log_tag}] ({w}, {s}, {i})"]
    n_reps = params.get("n_replications", 5)
    seed = params.get("random_seed", 42)
    demand_train, demand_eval = _split_demand(demand, params)

    for spec in policy_specs:
        name = spec["name"]
        try:
            if spec["kind"] == "construct":
                policy_fn = spec["cls"](demand_train, cfg, **spec.get("kw", {}))
            elif spec["kind"] == "optimize":
                opt = spec["cls"](demand_train, cfg)
                opt.optimize(verbose=False)
                policy_fn = opt.make_policy()
            elif spec["kind"] == "rl_train":
                agent = spec["cls"](InventoryEnv.STATE_DIM, cfg)
                agent.train(demand_train, cfg, verbose=False)
                policy_fn = agent.make_policy()
            else:
                raise ValueError(f"kind desconhecido: {spec['kind']}")
            kpis = _run_episode(demand_eval, cfg, policy_fn, n_reps, seed)
            rows.append(_kpi_row(w, s, i, name, kpis, meta_row, 0))
        except Exception as e:
            logs.append(f"[{log_tag}] {name} failed for {key}: {e}")

    return rows, logs


def _worker_proposed(key, cfg, demand, meta_row, params):
    """Worker de processo para GA-DQN/GA-PPO — mesma logica de 2 fases do
    loop sequencial original (GA gera limiares, RL parte deles)."""
    from simulation.core.policies import (
        GAPolicyOptimizer, DQNPolicy, PPOPolicy, HybridGADQN, HybridGAPPO,
    )
    from simulation.core.inventory_env import InventoryEnv

    w, s, i = key
    rows = []
    logs = [f"[Proposed] ({w}, {s}, {i})"]
    n_reps = params.get("n_replications", 5)
    seed = params.get("random_seed", 42)
    state_dim = InventoryEnv.STATE_DIM
    hybrid_params = params.get("hybrid", {})
    demand_train, demand_eval = _split_demand(demand, params)

    try:
        ga_cfg = dict(cfg)
        ga_cfg["GENETIC_ALGORITHM"] = dict(cfg["GENETIC_ALGORITHM"])
        ga_cfg["GENETIC_ALGORITHM"]["n_generations"] = hybrid_params.get(
            "ga_generations", cfg["GENETIC_ALGORITHM"]["n_generations"])
        ga = GAPolicyOptimizer(demand_train, ga_cfg)
        ga.optimize(verbose=False)
        ga_params = ga.best
    except Exception as e:
        logs.append(f"[Proposed] GA phase failed for {key}: {e}")
        return rows, logs

    buffer_size = hybrid_params.get("buffer_size", 1000)

    try:
        rl_cfg = dict(cfg)
        rl_cfg["DQN"] = dict(cfg["DQN"])
        rl_cfg["DQN"]["episodes"] = hybrid_params.get("rl_episodes", cfg["DQN"]["episodes"])
        dqn = DQNPolicy(state_dim, rl_cfg)
        dqn.prepopulate_from_ga(demand_train, cfg, ga_params, n_transitions=buffer_size)
        dqn.train(demand_train, rl_cfg, verbose=False)
        hybrid_dqn = HybridGADQN(ga_params, dqn)
        policy_fn = hybrid_dqn.make_policy()
        kpis = _run_episode(demand_eval, cfg, policy_fn, n_reps, seed)
        rows.append(_kpi_row(w, s, i, "GA-DQN", kpis, meta_row, 0))
    except Exception as e:
        logs.append(f"[Proposed] GA-DQN failed for {key}: {e}")

    try:
        rl_cfg = dict(cfg)
        rl_cfg["PPO"] = dict(cfg["PPO"])
        rl_cfg["PPO"]["episodes"] = hybrid_params.get("rl_episodes", cfg["PPO"]["episodes"])
        ppo = PPOPolicy(state_dim, rl_cfg)
        ppo.warmstart_from_ga(demand_train, cfg, ga_params, n_episodes=20)
        ppo.train(demand_train, rl_cfg, verbose=False)
        hybrid_ppo = HybridGAPPO(ga_params, ppo)
        policy_fn = hybrid_ppo.make_policy()
        kpis = _run_episode(demand_eval, cfg, policy_fn, n_reps, seed)
        rows.append(_kpi_row(w, s, i, "GA-PPO", kpis, meta_row, 0))
    except Exception as e:
        logs.append(f"[Proposed] GA-PPO failed for {key}: {e}")

    return rows, logs

POLICY_NAMES = {
    "classical":      ["EOQ", "sS", "Newsvendor"],
    "sota_classical": ["PIL", "CappedBaseStock", "BigDataNewsvendor"],
    "zabraoui":       ["MinMax", "FixedInterval", "VendorResponsive"],
    "metaheuristics": ["GA", "SA", "PSO", "DE"],
    "rl":             ["DQN", "PPO", "SARSA"],
    "proposed":       ["GA-DQN", "GA-PPO"],
}

# Famílias das políticas, para estratificar relatórios e testes estatísticos
POLICY_FAMILY = {
    "EOQ": "classical", "sS": "classical", "Newsvendor": "classical",
    "PIL": "sota_classical", "CappedBaseStock": "sota_classical",
    "BigDataNewsvendor": "sota_classical",
    "MinMax": "zabraoui", "FixedInterval": "zabraoui",
    "VendorResponsive": "zabraoui",
    "GA": "metaheuristic", "SA": "metaheuristic",
    "PSO": "metaheuristic", "DE": "metaheuristic",
    "DQN": "rl", "PPO": "rl", "SARSA": "rl",
    "GA-DQN": "hybrid", "GA-PPO": "hybrid",
}


def _build_cfg(params: dict) -> dict:
    """Converte parâmetros Kedro (lowercase) para o formato cfg legado (UPPERCASE)."""
    cost = params.get("cost", {})
    return {
        "SIMULATION": {
            "lead_time":          params.get("lead_time", 2),
            "initial_inventory":  params.get("initial_inventory", 100),
            "n_replications":     params.get("n_replications", 5),
        },
        "COST": {
            "holding_cost_per_unit":   cost.get("holding", 1.0),
            "stockout_cost_per_unit":  cost.get("stockout", 5.0),
            "ordering_cost_per_order": cost.get("order_fixed", 50.0),
            "ordering_cost_per_unit":  cost.get("order_unit", 0.5),
        },
        "HEURISTIC": {
            "z_score": params.get("z_score", 1.28),
        },
        # GA: hiperparametros de Zabraoui et al. (2025), Tabela 5, e a
        # funcao de aptidao da Eq.(3) do artigo (fitness_mode "zabraoui").
        "GENETIC_ALGORITHM": {
            "population_size":  params.get("ga", {}).get("population", 100),
            "n_generations":    params.get("ga", {}).get("generations", 50),
            "crossover_prob":   params.get("ga", {}).get("crossover_prob", 0.8),
            "mutation_rate":    params.get("ga", {}).get("mutation_prob", 0.05),
            # 2026-08-18: mutacao adaptativa (correcao de fidelidade, ver
            # TorchGA._mutation_rate) -- taxa final = mutation_rate * este fator.
            "mutation_final_ratio": params.get("ga", {}).get("mutation_final_ratio", 0.2),
            "tournament_size":  params.get("ga", {}).get("tournament_size", 3),
            "fitness_mode":     params.get("ga", {}).get("fitness_mode", "zabraoui"),
            "fitness_weights":  params.get("ga", {}).get("fitness_weights", [1.0, 0.0001]),
            "alpha_min":        params.get("alpha_min", 0.70),
            "penalty_weight":   params.get("ga", {}).get("penalty_weight", 10.0),
            # 2026-08-18: TR (risco de ruptura, ponderado por 1-NS) e BE
            # (punicao por compra excessiva, BE>1) somados a fitness_cost
            # -- ver _risk_terms em core/inventory_env_torch.py.
            "tr_weight":        params.get("ga", {}).get("tr_weight", 1.0),
            "be_weight":        params.get("ga", {}).get("be_weight", 1.0),
            "seed":             params.get("random_seed", 42),
            "search_space":     {"ROP": [0, 2000], "Q": [1, 2000], "SS": [0, 1000]},
        },
        "SA":  {
            "maxiter":      params.get("sa", {}).get("max_iter", 500),
            "initial_temp": params.get("sa", {}).get("initial_temp", 1000.0),
            "cooling_rate": params.get("sa", {}).get("cooling_rate", 0.95),
            "n_chains":     params.get("sa", {}).get("n_chains", 32),
            "seed":         params.get("random_seed", 42),
        },
        "PSO": {
            "n_particles":  params.get("pso", {}).get("n_particles", 40),
            "n_iterations": params.get("pso", {}).get("iterations", 80),
            "inertia":      params.get("pso", {}).get("inertia", 0.7),
            "cognitive":    params.get("pso", {}).get("cognitive", 1.5),
            "social":       params.get("pso", {}).get("social", 1.5),
            "use_constriction": params.get("pso", {}).get("use_constriction", True),
            "seed":         params.get("random_seed", 42),
        },
        "DE": {
            "maxiter":      params.get("de", {}).get("max_iter", 100),
            "population_size": params.get("de", {}).get("population_size", 15),
            "mutation":     params.get("de", {}).get("mutation", 0.8),
            "recombination":params.get("de", {}).get("recombination", 0.9),
            "seed":         params.get("random_seed", 42),
        },
        "DQN": {
            "episodes":          params.get("dqn", {}).get("episodes", 500),
            "gamma":             params.get("dqn", {}).get("gamma", 0.95),
            "epsilon_start":     params.get("dqn", {}).get("epsilon_start", 1.0),
            "epsilon_end":       params.get("dqn", {}).get("epsilon_end", 0.01),
            "epsilon_decay":     params.get("dqn", {}).get("epsilon_decay", 0.995),
            "batch_size":        params.get("dqn", {}).get("batch_size", 64),
            "memory_size":       params.get("dqn", {}).get("memory_size", 10000),
            "target_update_freq":params.get("dqn", {}).get("target_update_freq", 10),
            "n_actions":         params.get("dqn", {}).get("n_actions", 20),
            "max_order_qty":     200,
            "learning_rate":     0.001,
            "hidden_layers":     params.get("dqn", {}).get("hidden_layers", [64, 64]),
            "device":            params.get("dqn", {}).get("device", "cpu"),
            "seed":              params.get("dqn", {}).get("seed", 42),
            "grad_clip":         params.get("dqn", {}).get("grad_clip", 10.0),
            "soft_update_tau":   params.get("dqn", {}).get("soft_update_tau", 0.0),
        },
        "PPO": {
            "episodes":      params.get("ppo", {}).get("episodes", 300),
            "gamma":         params.get("ppo", {}).get("gamma", 0.99),
            "clip_epsilon":  params.get("ppo", {}).get("clip_epsilon", 0.2),
            "update_epochs": params.get("ppo", {}).get("update_epochs", 4),
            "learning_rate": params.get("ppo", {}).get("learning_rate", 0.0003),
            "n_actions":     params.get("ppo", {}).get("n_actions", 20),
            "max_order_qty": 200,
            "device":        params.get("ppo", {}).get("device", "cpu"),
            "seed":          params.get("ppo", {}).get("seed", 42),
            "gae_lambda":    params.get("ppo", {}).get("gae_lambda", 0.95),
            "entropy_coef":  params.get("ppo", {}).get("entropy_coef", 0.005),
            "value_coef":    params.get("ppo", {}).get("value_coef", 0.5),
            "minibatch_size":params.get("ppo", {}).get("minibatch_size", 64),
            "grad_clip":     params.get("ppo", {}).get("grad_clip", 0.5),
            "clip_value_loss": params.get("ppo", {}).get("clip_value_loss", True),
            "hidden_layers": params.get("ppo", {}).get("hidden_layers", [64, 64]),
        },
        # SARSA tabular: α, γ, ε fixo (seção 2.2.4 e tabela 5×10 da dissertação)
        "SARSA": {
            "episodes":      params.get("sarsa", {}).get("episodes", 500),
            "n_states":      params.get("sarsa", {}).get("n_states", 5),
            "n_actions":     params.get("sarsa", {}).get("n_actions", 10),
            "gamma":         params.get("sarsa", {}).get("gamma", 0.99),
            "learning_rate": params.get("sarsa", {}).get("learning_rate", 0.1),
            "epsilon":       params.get("sarsa", {}).get("epsilon", 0.1),
            "max_order_qty": 200,
            "n_pipeline_states": params.get("sarsa", {}).get("n_pipeline_states", 3),
            "seed":          params.get("sarsa", {}).get("seed", 42),
        },
    }


def scale_parameters_per_store(scenarios_meta: pd.DataFrame,
                                params: dict) -> dict:
    """
    Calcula parâmetros escalados por série (init_inv, max_order_qty).

    Returns:
        dict chave=(warehouse, store_id, item_id) -> cfg adaptado para essa série
    """
    base_cfg = _build_cfg(params)
    scaled = {}

    for _, row in scenarios_meta.iterrows():
        key = (row["warehouse"], row["store_id"], row["item_id"])
        mu = row["mu"]
        sigma = row["sigma"]
        z = params.get("z_score", 1.28)
        lt = params.get("lead_time", 2)

        # I₀ = ponto de reposição de referência (eq. 3.5): μL + zσ√L
        init_inv = rop_ref(mu, sigma, z, lt)
        max_ord = max(50.0, mu * 12)  # ~12 ciclos de demanda média

        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base_cfg.items()}
        cfg["SIMULATION"] = dict(base_cfg["SIMULATION"])
        cfg["SIMULATION"]["initial_inventory"] = float(init_inv)
        cfg["GENETIC_ALGORITHM"] = dict(base_cfg["GENETIC_ALGORITHM"])
        cfg["GENETIC_ALGORITHM"]["search_space"] = {
            "ROP": [0, max(2000, rop_ref(mu, sigma, z, lt) * 3)],
            "Q":   [1, max(2000, max_ord * 2)],
            "SS":  [0, max(1000, z * sigma * np.sqrt(lt) * 3)],
        }
        for rl_key in ["DQN", "PPO", "SARSA"]:
            cfg[rl_key] = dict(base_cfg[rl_key])
            cfg[rl_key]["max_order_qty"] = float(max_ord)

        scaled[key] = cfg

    log.info("Parâmetros escalados para %d séries", len(scaled))
    return scaled


def rop_ref(mu, sigma, z, lt):
    return mu * lt + z * sigma * np.sqrt(lt)


def _get_series(scenarios: pd.DataFrame, key: tuple) -> np.ndarray:
    w, s, i = key
    mask = ((scenarios["warehouse"] == w) &
            (scenarios["store_id"] == s) &
            (scenarios["item_id"] == i))
    return scenarios[mask].sort_values("venda_ciclo")["demand"].values.astype(float)


def _split_demand(demand: np.ndarray, params: dict) -> tuple:
    """
    Retorna (demand_train, demand_eval) respeitando a linha do tempo.

    evaluation_mode="walkforward": parâmetros das políticas estimados APENAS em
    demand[:train_split_cycles]; avaliação de KPIs em demand[train_split_cycles:].
    evaluation_mode="full": usa a série completa para treino E avaliação
    (comportamento anterior — válido para benchmark sem preocupação com leakage).
    """
    if params.get("evaluation_mode", "full") == "walkforward":
        n = params.get("train_split_cycles", 17)
        if len(demand) > n:
            return demand[:n], demand[n:]
    return demand, demand


def _run_episode(demand: np.ndarray, cfg: dict, policy_fn,
                 n_reps: int, seed: int) -> dict:
    """Executa a política n_reps vezes e retorna KPIs agregados."""
    from simulation.core.inventory_env import InventoryEnv
    np.random.seed(seed)
    env = InventoryEnv(demand, cfg)
    result = env.run_policy(policy_fn, n_reps=n_reps, base_seed=seed)
    return result["kpis"]


def _kpi_row(w, s, i, policy, kpis_agg: dict, meta_row: pd.Series,
             replication: int) -> dict:
    row = {
        "warehouse": w, "store_id": s, "item_id": i,
        "policy": policy,
        "TIC":     kpis_agg.get("TIC", np.nan),
        "NS":      kpis_agg.get("ServiceLevel", np.nan),
        "TR":      kpis_agg.get("StockoutRate", np.nan),
        "BE":      kpis_agg.get("BullwhipEffect", np.nan),
        "FP":      kpis_agg.get("OrderFrequency", np.nan),
        "TIC_std": kpis_agg.get("TIC_std", np.nan),
        "NS_std":  kpis_agg.get("ServiceLevel_std", np.nan),
        # Decomposição do custo (2026-08-18, base p/ CTI_ajustado — ver
        # AJUSTES_INFRA item #33): holding/stockout/pedido separados, antes
        # só existiam somados dentro de TIC.
        "HoldingCost":  kpis_agg.get("HoldingCost", np.nan),
        "StockoutCost": kpis_agg.get("StockoutCost", np.nan),
        "OrderCost":    kpis_agg.get("OrderCost", np.nan),
        "AvgInventory": kpis_agg.get("AvgInventory", np.nan),
        # Estatísticas da série
        "group":      meta_row.get("group", "?"),
        "cv":         meta_row.get("cv", np.nan),
        "mu":         meta_row.get("mu", np.nan),
        "n_periods":  meta_row.get("n_periods", np.nan),
        "mu_revenue": meta_row.get("mu_revenue", np.nan),
        # Perfil da revendedora (propagados de scenarios_meta)
        "segmento":         meta_row.get("segmento", None),
        "genero":           meta_row.get("genero", None),
        "filial":           meta_row.get("filial", None),
        "praca":            meta_row.get("praca", None),
        "gerente_regional": meta_row.get("gerente_regional", None),
        "ci_status":        meta_row.get("ci_status", None),
    }
    return row


def run_classical_policies(scenarios: pd.DataFrame,
                           scenarios_meta: pd.DataFrame,
                           scaled_params: dict,
                           params: dict) -> pd.DataFrame:
    """EOQ, (s,S), Newsvendor × todas as séries × n_replications."""
    from simulation.core.policies import EOQPolicy, SsPolicyClass, NewsvendorPolicy

    if not params.get("policies", {}).get("classical", True):
        log.info("Políticas clássicas desativadas — pulando")
        return pd.DataFrame()

    policy_specs = [
        {"kind": "construct", "cls": EOQPolicy, "name": "EOQ"},
        {"kind": "construct", "cls": SsPolicyClass, "name": "sS"},
        {"kind": "construct", "cls": NewsvendorPolicy, "name": "Newsvendor"},
    ]
    return _run_parallel_policies(scenarios, scenarios_meta, scaled_params, params,
                                   _worker_generic, (policy_specs, "Classical"), "Classical")


def run_sota_classical_policies(scenarios: pd.DataFrame,
                                scenarios_meta: pd.DataFrame,
                                scaled_params: dict,
                                params: dict) -> pd.DataFrame:
    """
    Políticas clássicas em versão estado da arte, adicionadas ao portfólio
    sem substituir as de referência:

      PIL                 van Jaarsveld & Arts, Operations Research 72(5), 2024
      CappedBaseStock     Xin, Operations Research 69(1), 2021
      BigDataNewsvendor   Ban & Rudin, Operations Research 67(1), 2019

    O `CappedBaseStock` é o benchmark que a literatura de DRL em lost sales
    não supera de forma consistente. Sem ele, os agentes de RL do portfólio
    seriam comparados apenas contra heurísticas subdimensionadas, o que
    inflaria artificialmente qualquer ganho reportado.
    """
    from simulation.core.policies_sota import (
        PILPolicy, CappedBaseStockPolicy, BigDataNewsvendorPolicy
    )

    if not params.get("policies", {}).get("sota_classical", True):
        log.info("Políticas clássicas SOTA desativadas — pulando")
        return pd.DataFrame()

    alpha_min = params.get("alpha_min", 0.70)
    policy_specs = [
        {"kind": "construct", "cls": PILPolicy, "name": "PIL", "kw": {"alpha_min": alpha_min}},
        {"kind": "construct", "cls": CappedBaseStockPolicy, "name": "CappedBaseStock",
         "kw": {"alpha_min": alpha_min}},
        {"kind": "construct", "cls": BigDataNewsvendorPolicy, "name": "BigDataNewsvendor"},
    ]
    return _run_parallel_policies(scenarios, scenarios_meta, scaled_params, params,
                                   _worker_generic, (policy_specs, "SOTA-Classical"), "SOTA-Classical")


def run_zabraoui_policies(scenarios: pd.DataFrame,
                          scenarios_meta: pd.DataFrame,
                          scaled_params: dict,
                          params: dict) -> pd.DataFrame:
    """
    Heuristicas de referencia de Zabraoui et al. (2025), Secao 3.5.

    O artigo define quatro heuristicas baseline: (s,S), Min-Max, Fixed
    Interval Replenishment e Vendor-Responsive. A primeira ja existe no
    portfolio como politica classica; as outras tres entram aqui.

    Adotadas por decisao explicita: para as politicas do portfolio sem estado
    da arte publicado, usa-se a versao do artigo-base. Ver
    docs/references/estado_da_arte_politicas.md.
    """
    from simulation.core.policies_zabraoui import (
        MinMaxPolicy, FixedIntervalPolicy, VendorResponsivePolicy
    )

    if not params.get("policies", {}).get("zabraoui", True):
        log.info("Politicas Zabraoui desativadas - pulando")
        return pd.DataFrame()

    policy_specs = [
        {"kind": "construct", "cls": MinMaxPolicy, "name": "MinMax"},
        {"kind": "construct", "cls": FixedIntervalPolicy, "name": "FixedInterval"},
        {"kind": "construct", "cls": VendorResponsivePolicy, "name": "VendorResponsive"},
    ]
    return _run_parallel_policies(scenarios, scenarios_meta, scaled_params, params,
                                   _worker_generic, (policy_specs, "Zabraoui"), "Zabraoui")


def run_metaheuristic_policies(scenarios: pd.DataFrame,
                               scenarios_meta: pd.DataFrame,
                               scaled_params: dict,
                               params: dict) -> pd.DataFrame:
    """GA, SA, PSO, DE × todas as séries × n_replications."""
    from simulation.core.policies import (
        GAPolicyOptimizer, SimulatedAnnealingPolicy, PSOPolicy, DEPolicy
    )

    if not params.get("policies", {}).get("metaheuristics", True):
        log.info("Metaheurísticas desativadas — pulando")
        return pd.DataFrame()

    policy_specs = [
        {"kind": "optimize", "cls": GAPolicyOptimizer, "name": "GA"},
        {"kind": "optimize", "cls": SimulatedAnnealingPolicy, "name": "SA"},
        {"kind": "optimize", "cls": PSOPolicy, "name": "PSO"},
        {"kind": "optimize", "cls": DEPolicy, "name": "DE"},
    ]
    return _run_parallel_policies(scenarios, scenarios_meta, scaled_params, params,
                                   _worker_generic, (policy_specs, "Metaheuristic"), "Metaheuristic")


def run_rl_policies(scenarios: pd.DataFrame,
                    scenarios_meta: pd.DataFrame,
                    scaled_params: dict,
                    params: dict) -> pd.DataFrame:
    """DQN, PPO, SARSA × todas as séries × n_replications."""
    from simulation.core.policies import DQNPolicy, PPOPolicy, SARSAPolicy

    if not params.get("policies", {}).get("reinforcement_learning", True):
        log.info("Políticas RL desativadas — pulando")
        return pd.DataFrame()

    policy_specs = [
        {"kind": "rl_train", "cls": DQNPolicy, "name": "DQN"},
        {"kind": "rl_train", "cls": PPOPolicy, "name": "PPO"},
        {"kind": "rl_train", "cls": SARSAPolicy, "name": "SARSA"},
    ]
    return _run_parallel_policies(scenarios, scenarios_meta, scaled_params, params,
                                   _worker_generic, (policy_specs, "RL"), "RL")


def run_proposed_architecture(scenarios: pd.DataFrame,
                              scenarios_meta: pd.DataFrame,
                              scaled_params: dict,
                              params: dict) -> pd.DataFrame:
    """
    GA-DQN e GA-PPO: arquitetura proposta da dissertação.
    GA inicializa os limiares (ROP, Q, SS); RL ajusta a quantidade pedida.
    """
    if not params.get("policies", {}).get("proposed_architecture", True):
        log.info("Arquitetura proposta desativada — pulando")
        return pd.DataFrame()

    return _run_parallel_policies(scenarios, scenarios_meta, scaled_params, params,
                                   _worker_proposed, (), "Proposed")


def aggregate_kpis(kpis_classical: pd.DataFrame,
                   kpis_sota_classical: pd.DataFrame,
                   kpis_zabraoui: pd.DataFrame,
                   kpis_metaheuristic: pd.DataFrame,
                   kpis_rl: pd.DataFrame,
                   kpis_proposed: pd.DataFrame) -> pd.DataFrame:
    """Concatena todos os KPIs em um único DataFrame."""
    frames = [df for df in [kpis_classical, kpis_sota_classical, kpis_zabraoui,
                            kpis_metaheuristic, kpis_rl, kpis_proposed]
              if df is not None and not df.empty]

    if not frames:
        raise RuntimeError("Nenhum KPI gerado — todas as políticas falharam ou estão desativadas")

    kpis = pd.concat(frames, ignore_index=True)
    kpis["policy_family"] = kpis["policy"].map(POLICY_FAMILY).fillna("other")
    log.info("KPIs agregados: %d linhas, %d políticas, %d séries | famílias: %s",
             len(kpis),
             kpis["policy"].nunique(),
             kpis.groupby(["warehouse", "store_id", "item_id"]).ngroups,
             kpis["policy_family"].value_counts().to_dict())
    return kpis
