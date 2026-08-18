"""
policies.py — 12 Políticas de Inventário para Comparativo Estendido
Todas operam no MESMO cenário fixo (InventoryEnv padronizado).

Políticas implementadas:
  Baseline clássico:
    1. EOQ          — Economic Order Quantity clássico
    2. (s,S)        — Banda min-max com reposição até S
    3. NewsvendorQ  — Newsvendor quantile otimizado por demanda histórica
  Metaheurísticas:
    4. GA           — Algoritmo Genético (DEAP) — ROP, Q, SS
    5. SA           — Simulated Annealing (scipy)
    6. PSO          — Particle Swarm Optimization (numpy puro)
    7. DE           — Differential Evolution (scipy)
  Aprendizado por Reforço:
    8. DQN          — Deep Q-Network
    9. PPO          — Proximal Policy Optimization
   10. SARSA        — On-policy TD learning
  Híbridos:
   11. GA-DQN       — GA inicializa limiar, DQN decide quantidade
   12. GA-PPO       — GA inicializa limiar, PPO decide quantidade
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def _get_search_bounds(cfg: dict, demand: np.ndarray) -> tuple:
    """
    Retorna (rop_range, q_range, ss_range) escalados pela demanda real.
    """
    sp  = cfg.get("GENETIC_ALGORITHM", {}).get("search_space", {})
    mu  = float(np.mean(demand))
    std = float(np.std(demand))
    lt  = cfg["SIMULATION"].get("lead_time", 2)
    z   = cfg.get("HEURISTIC", {}).get("z_score", 1.645)
    K   = cfg["COST"].get("ordering_cost_per_order", 50.0)
    h   = cfg["COST"].get("holding_cost_per_unit", 1.0)

    q_eoq   = float(np.sqrt(2 * mu * max(len(demand), 1) * K / max(h, 1e-6)))
    ss_ref  = z * std * np.sqrt(lt)
    rop_ref = mu * lt + ss_ref

    rop_r = (0, max(sp.get("ROP", [0, 2000])[1], rop_ref * 2))
    q_r   = (1, max(sp.get("Q",   [1, 2000])[1], q_eoq  * 2))
    ss_r  = (0, max(sp.get("SS",  [0, 1000])[1], ss_ref * 2))
    return rop_r, q_r, ss_r


def _eval_static(demand, cfg, ROP, Q, SS):
    """
    Aptidão de uma parametrização (ROP, Q, SS), a MAXIMIZAR.

    Implementa a formulação restrita da Equação (4.2) da proposta:

        min CTI(theta)   sujeito a   NS(theta) >= alpha_min

    A versão anterior usava soma ponderada (w0*NS - w1*CTI), que não é a
    mesma coisa: pesos fixos deixam o ponto de operação escolhido depender
    da escala do CTI, que varia por ordens de grandeza entre séries (a razão
    entre a maior e a menor loja chega a 124x no estudo piloto). Na prática
    isso fazia a mesma configuração de pesos privilegiar serviço em séries
    de baixo volume e custo em séries de alto volume, contaminando a
    comparação entre políticas.

    Aqui a restrição entra por penalidade proporcional ao déficit de serviço
    e relativa ao próprio custo, o que mantém a aptidão adimensional e
    comparável entre séries. Candidatos viáveis são ordenados por -CTI puro.

    Retrocompatibilidade: se `fitness_mode` for "weighted" na configuração,
    o comportamento antigo é preservado, para permitir reproduzir os
    resultados já reportados no Capítulo 5.
    """
    from simulation.core.inventory_env import InventoryEnv
    gcfg = cfg.get("GENETIC_ALGORITHM", {})
    env  = InventoryEnv(demand, cfg, seed=42)

    def policy(state, e):
        return Q if e.inventory + sum(e.pipeline) <= ROP + SS else 0.0

    state = env._state(); done = False
    while not done:
        state, _, done, _ = env.step(policy(state, env))

    k = env.kpis()

    if gcfg.get("fitness_mode", "constrained") == "weighted":
        w = gcfg.get("fitness_weights", [1.0, 0.0001])
        return w[0] * k["ServiceLevel"] - w[1] * k["TIC"], k

    alpha_min = float(gcfg.get("alpha_min", 0.70))
    penalty_w = float(gcfg.get("penalty_weight", 10.0))
    tic = float(k["TIC"])
    deficit = max(0.0, alpha_min - float(k["ServiceLevel"]))
    cost = tic + deficit * penalty_w * max(tic, 1.0)
    return -cost, k


# ══════════════════════════════════════════════════════════════
# 1. EOQ
# ══════════════════════════════════════════════════════════════
class EOQPolicy:
    def __init__(self, demand, cfg):
        hcfg = cfg.get("HEURISTIC", {})
        scfg = cfg["SIMULATION"]
        ccfg = cfg["COST"]
        mu   = np.mean(demand); std = np.std(demand)
        lt   = scfg.get("lead_time", 2)
        z    = hcfg.get("z_score", 1.645)
        D    = mu * len(demand)
        K    = ccfg.get("ordering_cost_per_order", 50.0)
        h    = ccfg.get("holding_cost_per_unit", 1.0)
        self.Q  = max(1.0, np.sqrt(2 * D * K / max(h, 0.01)))
        self.SS = z * std * np.sqrt(lt)
        self.ROP= mu * lt + self.SS
        print(f"  [EOQ] Q={self.Q:.1f} | ROP={self.ROP:.1f} | SS={self.SS:.1f}")

    def __call__(self, state, env):
        return self.Q if env.inventory + sum(env.pipeline) <= self.ROP else 0.0


# ══════════════════════════════════════════════════════════════
# 2. Política (s, S) — banda min-max
# ══════════════════════════════════════════════════════════════
class SsPolicyClass:
    def __init__(self, demand, cfg):
        scfg = cfg["SIMULATION"]; ccfg = cfg["COST"]
        mu  = np.mean(demand); std = np.std(demand)
        lt  = scfg.get("lead_time", 2)
        z   = cfg.get("HEURISTIC", {}).get("z_score", 1.645)
        self.s = mu * lt + z * std * np.sqrt(lt)
        self.S = self.s + max(1.0, np.sqrt(
            2 * mu * len(demand) *
            ccfg.get("ordering_cost_per_order", 50.0) /
            max(ccfg.get("holding_cost_per_unit", 1.0), 0.01)))
        print(f"  [(s,S)] s={self.s:.1f} | S={self.S:.1f}")

    def __call__(self, state, env):
        pos = env.inventory + sum(env.pipeline)
        return max(0.0, self.S - pos) if pos <= self.s else 0.0


# ══════════════════════════════════════════════════════════════
# 3. Newsvendor  — Q* = μ̂ + z·σ̂  (proposta eq. Jornaleiro)
# ══════════════════════════════════════════════════════════════
class NewsvendorPolicy:
    def __init__(self, demand, cfg):
        z    = cfg.get("HEURISTIC", {}).get("z_score", 1.28)
        mu   = float(np.mean(demand))
        std  = float(np.std(demand))
        lt   = cfg["SIMULATION"].get("lead_time", 2)
        self.Q_opt = max(0.0, mu + z * std)
        self.ROP   = mu * lt
        print(f"  [Newsvendor] Q*={self.Q_opt:.1f} | z={z:.2f} | ROP={self.ROP:.1f}")

    def __call__(self, state, env):
        pos = env.inventory + sum(env.pipeline)
        return self.Q_opt if pos <= self.ROP else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Meta-heurísticas
#
# GA, SA, PSO e DE migraram para `simulation.core.metaheuristics_torch`, com
# simulação vetorizada: a população inteira é avaliada em uma chamada, em vez
# de um laço Python por indivíduo. DEAP e scipy deixaram de ser necessários.
#
# Não se trata de upgrade para estado da arte: conforme
# docs/references/estado_da_arte_politicas.md (Secao 3.2), nao existe estado da
# arte publicado em veiculo relevante para meta-heuristicas em inventario.
# O que mudou foi (i) a funcao objetivo, agora a formulacao restrita da
# Eq. (4.2), (ii) elitismo no GA, (iii) fator de constricao no PSO, e
# (iv) cronograma de Metropolis explicito no SA, em vez de scipy.dual_annealing.
# ═══════════════════════════════════════════════════════════════════════════

from simulation.core.metaheuristics_torch import (   # noqa: E402
    TorchGA, TorchSA, TorchPSO, TorchDE,
    GAPolicyOptimizer, SimulatedAnnealingPolicy, PSOPolicy, DEPolicy,
)

# ═══════════════════════════════════════════════════════════════════════════
# Aprendizado por reforço e arquiteturas híbridas
#
# Os agentes DQN, PPO e SARSA implementados à mão em NumPy foram removidos e
# substituídos pelas versões em PyTorch de `simulation.core.rl_torch`:
#
#   DQNPolicy    -> DoubleDQNAgent     Double DQN (van Hasselt et al., 2016)
#                                      com arquitetura dueling (Wang et al.,
#                                      2016), perda de Huber e rede alvo.
#   PPOPolicy    -> PPOAgent           PPO (Schulman et al., 2017) com GAE
#                                      (Schulman et al., 2016).
#   SARSAPolicy  -> ExpectedSARSAAgent Expected SARSA (van Seijen et al., 2009).
#
# Motivo da substituição, além da atualização para o estado da arte: a
# atualização do ator no PPO antigo somava a vantagem diretamente aos logits
# alvo e treinava por erro quadrático. Isso NÃO é o gradiente da surrogate
# function recortada, de modo que o agente rotulado "PPO" nos resultados
# anteriores não otimizava o objetivo do PPO. A degeneração observada no
# Experimento 1 (FP = 0,98) é consistente com esse defeito.
#
# Os nomes antigos seguem exportados como aliases para não quebrar imports
# existentes no pipeline e nos notebooks.
# ═══════════════════════════════════════════════════════════════════════════

from simulation.core.rl_torch import (      # noqa: E402  (import tardio proposital)
    DoubleDQNAgent,
    PPOAgent,
    ExpectedSARSAAgent,
    HybridGARL,
    HybridGADQN,
    HybridGAPPO,
)

# Aliases de compatibilidade
DQNPolicy   = DoubleDQNAgent
PPOPolicy   = PPOAgent
SARSAPolicy = ExpectedSARSAAgent
_HybridGARL = HybridGARL

__all__ = [
    "EOQPolicy", "SsPolicyClass", "NewsvendorPolicy",
    "TorchGA", "TorchSA", "TorchPSO", "TorchDE",
    "GAPolicyOptimizer", "SimulatedAnnealingPolicy", "PSOPolicy", "DEPolicy",
    "DoubleDQNAgent", "PPOAgent", "ExpectedSARSAAgent",
    "DQNPolicy", "PPOPolicy", "SARSAPolicy",
    "HybridGARL", "HybridGADQN", "HybridGAPPO",
]
