"""
inventory_simulation/nodes.py — Execução das 12 políticas de inventário.

Fluxo:
  scenarios + scenarios_meta + params -> scale_parameters_per_store -> scaled_params
  scenarios + scaled_params + params -> run_classical_policies    -> kpis_classical
  scenarios + scaled_params + params -> run_metaheuristic_policies -> kpis_metaheuristic
  scenarios + scaled_params + params -> run_rl_policies            -> kpis_rl
  scenarios + scaled_params + params -> run_proposed_architecture  -> kpis_proposed
  kpis_classical + kpis_metaheuristic + kpis_rl + kpis_proposed -> aggregate_kpis -> kpis
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

log = logging.getLogger(__name__)

POLICY_NAMES = {
    "classical":      ["EOQ", "sS", "Newsvendor"],
    "metaheuristics": ["GA", "SA", "PSO", "DE"],
    "rl":             ["DQN", "PPO", "SARSA"],
    "proposed":       ["GA-DQN", "GA-PPO"],
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
        "GENETIC_ALGORITHM": {
            "population_size":  params.get("ga", {}).get("population", 100),
            "n_generations":    params.get("ga", {}).get("generations", 50),
            "crossover_prob":   params.get("ga", {}).get("crossover_prob", 0.7),
            "mutation_rate":    params.get("ga", {}).get("mutation_prob", 0.05),
            "fitness_weights":  [1.0, 0.0001],
            "search_space":     {"ROP": [0, 2000], "Q": [1, 2000], "SS": [0, 1000]},
        },
        "SA":  {"maxiter": params.get("sa", {}).get("max_iter", 500)},
        "PSO": {
            "n_particles":  params.get("pso", {}).get("n_particles", 40),
            "n_iterations": params.get("pso", {}).get("iterations", 80),
            "inertia":      params.get("pso", {}).get("inertia", 0.7),
            "cognitive":    params.get("pso", {}).get("cognitive", 1.5),
            "social":       params.get("pso", {}).get("social", 1.5),
        },
        "DE": {
            "maxiter":      params.get("de", {}).get("max_iter", 100),
            "popsize":      params.get("de", {}).get("population_size", 15),
            "mutation":     params.get("de", {}).get("mutation", [0.5, 1.0]),
            "recombination":params.get("de", {}).get("recombination", 0.7),
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
        },
        "PPO": {
            "episodes":      params.get("ppo", {}).get("episodes", 300),
            "gamma":         params.get("ppo", {}).get("gamma", 0.99),
            "clip_epsilon":  params.get("ppo", {}).get("clip_epsilon", 0.2),
            "update_epochs": params.get("ppo", {}).get("update_epochs", 4),
            "learning_rate": params.get("ppo", {}).get("learning_rate", 0.0003),
            "n_actions":     params.get("ppo", {}).get("n_actions", 20),
            "max_order_qty": 200,
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

    n_reps = params.get("n_replications", 5)
    seed = params.get("random_seed", 42)
    rows = []
    meta_idx = scenarios_meta.set_index(["warehouse", "store_id", "item_id"])

    for key, cfg in scaled_params.items():
        demand = _get_series(scenarios, key)
        if len(demand) < 5:
            continue
        w, s, i = key
        meta_row = meta_idx.loc[key].to_dict() if key in meta_idx.index else {}
        log.info("[Classical] (%s, %s, %s)", w, s, i)

        demand_train, demand_eval = _split_demand(demand, params)

        for PolicyClass, name in [
            (EOQPolicy, "EOQ"), (SsPolicyClass, "sS"), (NewsvendorPolicy, "Newsvendor")
        ]:
            try:
                # Parâmetros estimados só com dados de treino
                pol = PolicyClass(demand_train, cfg)
                # KPIs avaliados no período de teste
                kpis = _run_episode(demand_eval, cfg, pol, n_reps, seed)
                rows.append(_kpi_row(w, s, i, name, kpis, meta_row, 0))
            except Exception as e:
                log.warning("[Classical] %s failed for %s: %s", name, key, e)

    return pd.DataFrame(rows)


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

    n_reps = params.get("n_replications", 5)
    seed = params.get("random_seed", 42)
    rows = []
    meta_idx = scenarios_meta.set_index(["warehouse", "store_id", "item_id"])

    for key, cfg in scaled_params.items():
        demand = _get_series(scenarios, key)
        if len(demand) < 5:
            continue
        w, s, i = key
        meta_row = meta_idx.loc[key].to_dict() if key in meta_idx.index else {}
        log.info("[Metaheuristic] (%s, %s, %s)", w, s, i)

        demand_train, demand_eval = _split_demand(demand, params)

        for OptimizerClass, name in [
            (GAPolicyOptimizer, "GA"),
            (SimulatedAnnealingPolicy, "SA"),
            (PSOPolicy, "PSO"),
            (DEPolicy, "DE"),
        ]:
            try:
                # Otimiza parâmetros (ROP, Q, SS) usando apenas dados históricos
                opt = OptimizerClass(demand_train, cfg)
                opt.optimize(verbose=False)
                policy_fn = opt.make_policy()
                # Avalia a política resultante no período de teste
                kpis = _run_episode(demand_eval, cfg, policy_fn, n_reps, seed)
                rows.append(_kpi_row(w, s, i, name, kpis, meta_row, 0))
            except Exception as e:
                log.warning("[Metaheuristic] %s failed for %s: %s", name, key, e)

    return pd.DataFrame(rows)


def run_rl_policies(scenarios: pd.DataFrame,
                    scenarios_meta: pd.DataFrame,
                    scaled_params: dict,
                    params: dict) -> pd.DataFrame:
    """DQN, PPO, SARSA × todas as séries × n_replications."""
    from simulation.core.policies import DQNPolicy, PPOPolicy, SARSAPolicy
    from simulation.core.inventory_env import InventoryEnv

    if not params.get("policies", {}).get("reinforcement_learning", True):
        log.info("Políticas RL desativadas — pulando")
        return pd.DataFrame()

    n_reps = params.get("n_replications", 5)
    seed = params.get("random_seed", 42)
    state_dim = InventoryEnv.STATE_DIM
    rows = []
    meta_idx = scenarios_meta.set_index(["warehouse", "store_id", "item_id"])

    for key, cfg in scaled_params.items():
        demand = _get_series(scenarios, key)
        if len(demand) < 5:
            continue
        w, s, i = key
        meta_row = meta_idx.loc[key].to_dict() if key in meta_idx.index else {}
        log.info("[RL] (%s, %s, %s)", w, s, i)

        demand_train, demand_eval = _split_demand(demand, params)

        for AgentClass, name in [
            (DQNPolicy, "DQN"), (PPOPolicy, "PPO"), (SARSAPolicy, "SARSA")
        ]:
            try:
                agent = AgentClass(state_dim, cfg)
                # Treina o agente apenas com dados históricos (sem ver o período de teste)
                agent.train(demand_train, cfg, verbose=False)
                policy_fn = agent.make_policy()
                # Avalia no período de teste
                kpis = _run_episode(demand_eval, cfg, policy_fn, n_reps, seed)
                rows.append(_kpi_row(w, s, i, name, kpis, meta_row, 0))
            except Exception as e:
                log.warning("[RL] %s failed for %s: %s", name, key, e)

    return pd.DataFrame(rows)


def run_proposed_architecture(scenarios: pd.DataFrame,
                              scenarios_meta: pd.DataFrame,
                              scaled_params: dict,
                              params: dict) -> pd.DataFrame:
    """
    GA-DQN e GA-PPO: arquitetura proposta da dissertação.
    GA inicializa os limiares (ROP, Q, SS); RL ajusta a quantidade pedida.
    """
    from simulation.core.policies import (
        GAPolicyOptimizer, DQNPolicy, PPOPolicy,
        HybridGADQN, HybridGAPPO,
    )
    from simulation.core.inventory_env import InventoryEnv

    if not params.get("policies", {}).get("proposed_architecture", True):
        log.info("Arquitetura proposta desativada — pulando")
        return pd.DataFrame()

    n_reps = params.get("n_replications", 5)
    seed = params.get("random_seed", 42)
    state_dim = InventoryEnv.STATE_DIM
    hybrid_params = params.get("hybrid", {})
    rows = []
    meta_idx = scenarios_meta.set_index(["warehouse", "store_id", "item_id"])

    for key, cfg in scaled_params.items():
        demand = _get_series(scenarios, key)
        if len(demand) < 5:
            continue
        w, s, i = key
        meta_row = meta_idx.loc[key].to_dict() if key in meta_idx.index else {}
        log.info("[Proposed] (%s, %s, %s)", w, s, i)

        demand_train, demand_eval = _split_demand(demand, params)

        # Fase 1: GA otimiza os limiares de reposição (apenas dados de treino)
        try:
            ga_cfg = dict(cfg)
            ga_cfg["GENETIC_ALGORITHM"] = dict(cfg["GENETIC_ALGORITHM"])
            ga_cfg["GENETIC_ALGORITHM"]["n_generations"] = hybrid_params.get(
                "ga_generations", cfg["GENETIC_ALGORITHM"]["n_generations"])
            ga = GAPolicyOptimizer(demand_train, ga_cfg)
            ga.optimize(verbose=False)
            ga_params = ga.best
        except Exception as e:
            log.warning("[Proposed] GA phase failed for %s: %s", key, e)
            continue

        buffer_size = hybrid_params.get("buffer_size", 1000)

        # Fase 2a: GA-DQN
        # (i) GA já executou; (ii) pré-popula buffer; (iii) DQN 200 ep — proposta seção 4.4
        try:
            rl_cfg = dict(cfg)
            rl_cfg["DQN"] = dict(cfg["DQN"])
            rl_cfg["DQN"]["episodes"] = hybrid_params.get(
                "rl_episodes", cfg["DQN"]["episodes"])
            dqn = DQNPolicy(state_dim, rl_cfg)
            dqn.prepopulate_from_ga(demand_train, cfg, ga_params,
                                    n_transitions=buffer_size)
            dqn.train(demand_train, rl_cfg, verbose=False)
            hybrid_dqn = HybridGADQN(ga_params, dqn)
            policy_fn = hybrid_dqn.make_policy()
            kpis = _run_episode(demand_eval, cfg, policy_fn, n_reps, seed)
            rows.append(_kpi_row(w, s, i, "GA-DQN", kpis, meta_row, 0))
        except Exception as e:
            log.warning("[Proposed] GA-DQN failed for %s: %s", key, e)

        # Fase 2b: GA-PPO
        # (i) GA já executou; (ii) warm-start PPO; (iii) PPO 200 ep — proposta seção 4.4
        try:
            rl_cfg = dict(cfg)
            rl_cfg["PPO"] = dict(cfg["PPO"])
            rl_cfg["PPO"]["episodes"] = hybrid_params.get(
                "rl_episodes", cfg["PPO"]["episodes"])
            ppo = PPOPolicy(state_dim, rl_cfg)
            ppo.warmstart_from_ga(demand_train, cfg, ga_params, n_episodes=20)
            ppo.train(demand_train, rl_cfg, verbose=False)
            hybrid_ppo = HybridGAPPO(ga_params, ppo)
            policy_fn = hybrid_ppo.make_policy()
            kpis = _run_episode(demand_eval, cfg, policy_fn, n_reps, seed)
            rows.append(_kpi_row(w, s, i, "GA-PPO", kpis, meta_row, 0))
        except Exception as e:
            log.warning("[Proposed] GA-PPO failed for %s: %s", key, e)

    return pd.DataFrame(rows)


def aggregate_kpis(kpis_classical: pd.DataFrame,
                   kpis_metaheuristic: pd.DataFrame,
                   kpis_rl: pd.DataFrame,
                   kpis_proposed: pd.DataFrame) -> pd.DataFrame:
    """Concatena todos os KPIs em um único DataFrame."""
    frames = [df for df in [kpis_classical, kpis_metaheuristic,
                             kpis_rl, kpis_proposed]
              if df is not None and not df.empty]

    if not frames:
        raise RuntimeError("Nenhum KPI gerado — todas as políticas falharam ou estão desativadas")

    kpis = pd.concat(frames, ignore_index=True)
    log.info("KPIs agregados: %d linhas, %d políticas, %d séries",
             len(kpis),
             kpis["policy"].nunique(),
             kpis.groupby(["warehouse", "store_id", "item_id"]).ngroups)
    return kpis
