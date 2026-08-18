"""
policies_sota.py — Políticas clássicas em versão estado da arte.

Complementam (não substituem) as políticas de referência de `policies.py`.
EOQ, (s,S) e Jornaleiro permanecem intactos como baseline do varejo, que é o
denominador das comparações já reportadas no Capítulo 5. As políticas deste
módulo entram como alternativas modernas do mesmo slot conceitual.

Critério de seleção: para cada slot, a referência publicada mais recente em
veículo relevante (ver docs/references/estado_da_arte_politicas.md).

  PILPolicy              Projected Inventory Level
                         van Jaarsveld & Arts, Operations Research 72(5):
                         1790-1805, 2024.  ← mais recente do slot (s,S)

  CappedBaseStockPolicy  Capped base-stock
                         Xin, Operations Research 69(1):61-70, 2021.
                         Mantida porque a literatura de DRL a usa como
                         benchmark que nenhum agente supera consistentemente
                         em lost sales; serve de piso de comparação honesto.

  BigDataNewsvendorPolicy  Newsvendor orientado a características
                           Ban & Rudin, Operations Research 67(1):90-108, 2019.
                           ← mais recente do slot Jornaleiro

Todas expõem a mesma interface das demais políticas do benchmark:
    pol = Policy(demand_train, cfg)
    qty = pol(state, env)
de modo que `InventoryEnv.run_policy` as avalia sob condições idênticas.

Observação sobre calibração: os parâmetros são ajustados por busca sobre a
JANELA DE TREINO apenas. O nó de simulação já separa treino e avaliação
(`_split_demand`), então nenhuma destas políticas observa o período de teste.
"""
from __future__ import annotations

import numpy as np
import torch

from simulation.core.device import get_device_for_batch
from simulation.core.inventory_env import InventoryEnv
from simulation.core.inventory_env_torch import BatchInventoryEnv, constrained_cost  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Calibração vetorizada
# ═══════════════════════════════════════════════════════════════════════════

def _calibrate_batch(demand, cfg: dict, order_fn, grid: torch.Tensor,
                     alpha_min: float = 0.70, penalty_weight: float = 10.0,
                     device=None, demand_pool=None) -> tuple:
    """
    Avalia toda a grade de parâmetros em UMA simulação vetorizada.

    `order_fn(env, theta) -> order (B,)` recebe o ambiente em lote e a matriz
    de parâmetros (B, k), devolvendo a quantidade pedida por candidato. Assim a
    busca em grade de PIL e CappedBaseStock deixa de ser um laço Python sobre
    candidatos (que era o trecho mais lento das políticas clássicas) e passa a
    ser um único avanço de B trajetórias paralelas.

    `demand_pool`: 2026-08-18, pedido do usuário -- treino "com perfil"
    (pooling). Quando dado (lista de séries do mesmo perfil), a MESMA
    grade é avaliada contra CADA série do pool e o custo usado pra
    escolher o candidato vencedor é a MÉDIA sobre o pool -- mesmo padrão
    usado em `TorchGA.evaluate` (metaheuristics_torch.py).

    Retorna (melhor_theta como tuple, melhor_custo).
    """
    B = int(grid.shape[0])
    dev = device if device is not None else get_device_for_batch(
        B, cfg.get("SIMULATION", {}).get("device", "auto"))

    def _run_one(d) -> torch.Tensor:
        env = BatchInventoryEnv(d, cfg, batch_size=B, device=dev)
        theta = grid.to(device=env.device, dtype=env.dtype)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(order_fn(env, theta))
        return constrained_cost(env.kpis(), alpha_min, penalty_weight)

    if demand_pool is None:
        cost = _run_one(demand)
    else:
        cost = torch.stack([_run_one(d) for d in demand_pool], dim=0).mean(dim=0)

    theta = grid.to(device=dev)
    i = int(torch.argmin(cost).item())
    return tuple(float(v) for v in theta[i].tolist()), float(cost[i].item())


# ═══════════════════════════════════════════════════════════════════════════
# Utilidades comuns
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_cost(demand: np.ndarray, cfg: dict, policy_fn,
                   alpha_min: float = 0.0, seed: int = 42) -> float:
    """
    Custo total de uma política na janela dada.

    Se `alpha_min` > 0 e o nível de serviço ficar abaixo do limiar, aplica
    penalidade proporcional à violação. Isso implementa a restrição
    NS >= alpha_min da Equação (4.2) da proposta dentro de uma busca escalar,
    em vez de descartar candidatos silenciosamente.
    """
    env = InventoryEnv(demand, cfg, seed=seed)
    state = env.reset()
    done = False
    while not done:
        state, _, done, _ = env.step(policy_fn(state, env))
    k = env.kpis()
    cost = float(k["TIC"])
    if alpha_min > 0.0:
        deficit = max(0.0, alpha_min - float(k["ServiceLevel"]))
        if deficit > 0.0:
            # penalidade proporcional; escala com o próprio custo para
            # permanecer comparável entre séries de volumes distintos
            cost += deficit * 10.0 * max(cost, 1.0)
    return cost


def _empirical_samples(demand: np.ndarray, horizon: int, n: int,
                       seed: int = 7) -> np.ndarray:
    """
    Amostra (n, horizon) da distribuição empírica da demanda de treino.

    Reamostragem com reposição preserva a massa em zero e a cauda de picos —
    ambas essenciais no regime Lumpy, onde um ajuste paramétrico (Normal ou
    Poisson) subestima sistematicamente a probabilidade de ruptura.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(demand, dtype=float)
    if pool.size == 0:
        pool = np.array([0.0])
    return rng.choice(pool, size=(n, horizon), replace=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PIL — Projected Inventory Level (van Jaarsveld & Arts, Oper. Res. 2024)
# ═══════════════════════════════════════════════════════════════════════════

class PILPolicy:
    """
    Política de nível de estoque projetado.

    Emite pedidos de forma que o nível ESPERADO de estoque no instante em que
    o pedido chega seja igual a um alvo fixo S:

        Q_t = max(0, S - PIL_t)

    onde PIL_t é o estoque projetado para t+L, obtido propagando estoque
    atual e pedidos em trânsito ao longo de L+1 ciclos.

    Ponto central sob lost sales: a projeção precisa aplicar max(0, .) a cada
    ciclo intermediário. Projetar linearmente (I_t + pipeline - L*mu) é
    enviesado para baixo, porque supõe que o estoque pode ficar negativo e
    "recuperar" com a chegada seguinte — o que não ocorre quando a venda é
    perdida. Esse viés é grande justamente em séries Lumpy, onde o estoque
    zera com frequência.

    A projeção usa reamostragem da demanda empírica de treino em vez de
    demanda média, porque em regime intermitente a média é um péssimo resumo
    da distribuição.

    Parâmetro S calibrado por busca unidimensional na janela de treino.
    """

    def __init__(self, demand: np.ndarray, cfg: dict, alpha_min: float = 0.70,
                 n_scenarios: int = 48, n_grid: int = 40, demand_pool=None):
        """
        `demand_pool`: 2026-08-18, pedido do usuário -- treino "com perfil"
        (pooling). Quando dado, a reamostragem empírica usa a demanda
        CONCATENADA do pool inteiro (distribuição do perfil, não de uma
        série só), e a calibração de S é feita contra a MÉDIA do custo
        simulado em cada série do pool (ver `_calibrate_batch`).
        """
        self.cfg = cfg
        self.L = int(cfg["SIMULATION"].get("lead_time", 2))
        self.n_scenarios = int(n_scenarios)
        demand = np.asarray(demand, dtype=float)
        resample_source = (np.concatenate([np.asarray(d, dtype=float) for d in demand_pool])
                           if demand_pool is not None else demand)
        self._samples = _empirical_samples(resample_source, self.L + 1, self.n_scenarios)

        mu = float(np.mean(resample_source)) if resample_source.size else 0.0
        sigma = float(np.std(resample_source)) if resample_source.size else 0.0
        hi = max(1.0, (mu * (self.L + 1)) + 3.0 * sigma * np.sqrt(self.L + 1))
        grid = torch.linspace(0.0, hi, int(n_grid)).unsqueeze(1)   # (B, 1)

        # Projeção vetorizada: os cenários de demanda entram como uma dimensão
        # extra, de modo que os B candidatos e os K cenários avançam juntos.
        samples_t = torch.as_tensor(self._samples, dtype=torch.float32)

        def order_fn(env, theta):
            S = theta[:, 0]                                  # (B,)
            x = env.inventory.unsqueeze(1).expand(-1, self.n_scenarios).clone()
            smp = samples_t.to(env.device, env.dtype)         # (K, L+1)
            for k in range(self.L + 1):
                arriving = env.pipeline[:, k].unsqueeze(1)    # (B,1)
                x = torch.clamp(x + arriving - smp[:, k].unsqueeze(0), min=0.0)
            pil = x.mean(dim=1)                               # (B,)
            return torch.clamp(S - pil, min=0.0)

        best, cost = _calibrate_batch(demand, cfg, order_fn, grid,
                                      alpha_min=alpha_min, demand_pool=demand_pool)
        self.S = float(best[0])
        self.calibration_cost = float(cost)

    def _project(self, inventory: float, pipeline: list) -> float:
        """Estoque esperado no ciclo de chegada do pedido emitido agora."""
        x = np.full(self.n_scenarios, float(inventory))
        for k in range(self.L + 1):
            arriving = float(pipeline[k]) if k < len(pipeline) else 0.0
            x = np.maximum(0.0, x + arriving - self._samples[:, k])
        return float(x.mean())

    def __call__(self, state, env) -> float:
        return max(0.0, self.S - self._project(env.inventory, env.pipeline))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Capped base-stock (Xin, Oper. Res. 2021)
# ═══════════════════════════════════════════════════════════════════════════

class CappedBaseStockPolicy:
    """
    Política base-stock com teto de pedido:

        Q_t = min(cap, max(0, S - IP_t)),   IP_t = estoque + pedidos em trânsito

    Híbrido entre base-stock (cap -> infinito) e constant-order (S -> infinito).
    Xin (2021) demonstra garantia de desempenho de pior caso de 1,79 sob
    penalidade e lead time grandes.

    Relevância para esta dissertação: é a política que a literatura de DRL em
    lost sales não supera de forma consistente. Incluí-la torna o benchmark
    honesto — sem ela, agentes de RL são comparados apenas contra heurísticas
    subdimensionadas.

    Calibração conjunta de (S, cap) por busca em grade na janela de treino.
    """

    def __init__(self, demand: np.ndarray, cfg: dict, alpha_min: float = 0.70,
                 n_grid_s: int = 20, n_grid_cap: int = 12, demand_pool=None):
        """
        `demand_pool`: mesma semântica de pooling de `PILPolicy` -- grade
        dimensionada pela média/desvio do pool inteiro, calibrada contra o
        custo médio simulado em cada série do pool.
        """
        demand = np.asarray(demand, dtype=float)
        pool_concat = (np.concatenate([np.asarray(d, dtype=float) for d in demand_pool])
                      if demand_pool is not None else demand)
        L = int(cfg["SIMULATION"].get("lead_time", 2))
        mu = float(np.mean(pool_concat)) if pool_concat.size else 0.0
        sigma = float(np.std(pool_concat)) if pool_concat.size else 0.0

        s_hi = max(1.0, mu * (L + 1) + 3.0 * sigma * np.sqrt(L + 1))
        cap_hi = max(1.0, mu * (L + 1) + 3.0 * sigma)
        s_grid = torch.linspace(0.0, s_hi, int(n_grid_s))
        cap_grid = torch.linspace(max(1.0, mu), cap_hi, int(n_grid_cap))
        # produto cartesiano das duas grades -> um único lote
        grid = torch.cartesian_prod(s_grid, cap_grid)          # (B, 2)

        def order_fn(env, theta):
            S, cap = theta[:, 0], theta[:, 1]
            ip = env.inventory + env.pipeline.sum(dim=1)
            return torch.minimum(cap, torch.clamp(S - ip, min=0.0))

        best, cost = _calibrate_batch(demand, cfg, order_fn, grid,
                                      alpha_min=alpha_min, demand_pool=demand_pool)
        self.S, self.cap = float(best[0]), float(best[1])
        self.calibration_cost = float(cost)

    def __call__(self, state, env) -> float:
        ip = env.inventory + sum(env.pipeline)
        return min(self.cap, max(0.0, self.S - ip))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Big Data Newsvendor (Ban & Rudin, Oper. Res. 2019)
# ═══════════════════════════════════════════════════════════════════════════

class BigDataNewsvendorPolicy:
    """
    Jornaleiro orientado a características, por minimização empírica do risco.

    O jornaleiro clássico usa Q* = mu + z*sigma, isto é, supõe demanda normal e
    ignora qualquer informação de contexto. Ban & Rudin (2019) substituem a
    estimação da distribuição pela otimização direta da decisão:

        min_beta  sum_i [ c_u (d_i - x_i'beta)^+ + c_o (x_i'beta - d_i)^+ ]

    que é exatamente regressão quantílica no quantil crítico
    tau = c_u / (c_u + c_o). Aqui c_u é o custo de ruptura e c_o o de posse,
    de modo que tau = c_s / (c_s + h).

    O problema é resolvido como programa linear (formulação padrão com
    variáveis de desvio u, v >= 0), com regularização L1 opcional sobre os
    coeficientes — necessária porque a janela de treino é curta (17 ciclos no
    protocolo walk-forward) frente ao número de características.

    O alvo é a demanda acumulada de lead time (L+1 ciclos), e não a demanda de
    um ciclo, porque a decisão precisa cobrir o intervalo até a chegada do
    próximo pedido.

    Fallback: se o LP não convergir (série curta ou degenerada), a política
    recai no quantil empírico da demanda de lead time — que é o próprio
    estimador de Ban & Rudin sem características.
    """

    N_FEATURES = 8

    def __init__(self, demand: np.ndarray, cfg: dict, l1_penalty: float = 0.01,
                 demand_pool=None):
        """
        `demand_pool`: 2026-08-18, pedido do usuário -- treino "com perfil"
        (pooling). Quando dado, o dataset (X, y) da regressão quantílica é
        CONCATENADO a partir de TODAS as séries do pool (mais linhas de
        treino, um único β compartilhado), e μ/σ/quantil de fallback usam
        a demanda concatenada do pool. Na avaliação, `_target()` continua
        usando o histórico REAL da série sendo simulada (`env.demand_history`)
        -- é o mesmo β aplicado a cada série, não um modelo por série.
        """
        demand = np.asarray(demand, dtype=float)
        self.L = int(cfg["SIMULATION"].get("lead_time", 2))
        ccfg = cfg["COST"]
        c_s = float(ccfg.get("stockout_cost_per_unit", 5.0))
        h = float(ccfg.get("holding_cost_per_unit", 1.0))
        self.tau = c_s / max(c_s + h, 1e-9)

        pool = ([np.asarray(d, dtype=float) for d in demand_pool]
               if demand_pool is not None else [demand])
        pool_concat = np.concatenate(pool)

        self._mu = float(np.mean(pool_concat)) if pool_concat.size else 0.0
        self._sigma = float(np.std(pool_concat)) if pool_concat.size else 0.0
        self._hist = list(pool_concat)

        Xs, ys = [], []
        for d in pool:
            Xd, yd = self._build_dataset(d)
            if Xd is not None:
                Xs.append(Xd)
                ys.append(yd)
        X = np.concatenate(Xs, axis=0) if Xs else None
        y = np.concatenate(ys, axis=0) if ys else np.array([])

        self.beta = None
        self._fallback_q = self._empirical_quantile(pool_concat)

        if X is not None and len(y) >= self.N_FEATURES:
            self.beta = self._fit_quantile_lp(X, y, self.tau, l1_penalty)

    # ── construção de características ────────────────────────────────────
    def _features(self, hist: list) -> np.ndarray:
        """
        Características disponíveis no instante da decisão, todas causais.

        Escolhidas para descrever intermitência, e não apenas nível: em regime
        Lumpy a fração de zeros recentes e o intervalo desde a última venda
        carregam mais sinal sobre a próxima ocorrência do que a média.
        """
        h = np.asarray(hist[-12:], dtype=float) if hist else np.array([0.0])
        last = h[-1] if h.size else 0.0
        lag2 = h[-2] if h.size >= 2 else last
        ma3 = float(np.mean(h[-3:])) if h.size else 0.0
        ma6 = float(np.mean(h[-6:])) if h.size else 0.0
        zeros6 = float(np.mean(h[-6:] == 0)) if h.size else 0.0
        sd6 = float(np.std(h[-6:])) if h.size else 0.0
        nz = np.nonzero(h)[0]
        since = float(h.size - 1 - nz[-1]) if nz.size else float(h.size)
        return np.array([1.0, last, lag2, ma3, ma6, zeros6, sd6, since],
                        dtype=float)

    def _build_dataset(self, demand: np.ndarray):
        """Pares (características em t, demanda acumulada de lead time)."""
        T = len(demand)
        win = self.L + 1
        if T < win + 3:
            return None, np.array([])
        X, y = [], []
        for t in range(2, T - win + 1):
            X.append(self._features(list(demand[:t])))
            y.append(float(np.sum(demand[t:t + win])))
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    def _empirical_quantile(self, demand: np.ndarray) -> float:
        win = self.L + 1
        if len(demand) < win:
            return float(self._mu * win)
        sums = np.array([np.sum(demand[i:i + win])
                         for i in range(len(demand) - win + 1)], dtype=float)
        return float(np.quantile(sums, self.tau))

    # ── ajuste por programação linear ────────────────────────────────────
    @staticmethod
    def _fit_quantile_lp(X: np.ndarray, y: np.ndarray, tau: float,
                         l1: float):
        """
        Regressão quantílica como LP.

        Variáveis: beta+ (p), beta- (p), u (n), v (n), todas >= 0.
        beta = beta+ - beta-, o que permite penalizar |beta| linearmente.

        min  tau*sum(u) + (1-tau)*sum(v) + l1*sum(beta+ + beta-)
        s.a. X(beta+ - beta-) + u - v = y
        """
        try:
            from scipy.optimize import linprog
        except Exception:
            return None

        n, p = X.shape
        c = np.concatenate([
            np.full(2 * p, l1, dtype=float),
            np.full(n, tau, dtype=float),
            np.full(n, 1.0 - tau, dtype=float),
        ])
        A_eq = np.hstack([X, -X, np.eye(n), -np.eye(n)])
        try:
            res = linprog(c, A_eq=A_eq, b_eq=y,
                          bounds=[(0, None)] * (2 * p + 2 * n),
                          method="highs")
        except Exception:
            return None
        if not res.success:
            return None
        bp = res.x[:p]
        bm = res.x[p:2 * p]
        return bp - bm

    # ── decisão ──────────────────────────────────────────────────────────
    def _target(self, env) -> float:
        hist = list(env.demand_history) if getattr(env, "demand_history", None) else self._hist
        if self.beta is None:
            return self._fallback_q
        q = float(self._features(hist) @ self.beta)
        if not np.isfinite(q):
            return self._fallback_q
        return max(0.0, q)

    def __call__(self, state, env) -> float:
        ip = env.inventory + sum(env.pipeline)
        return max(0.0, self._target(env) - ip)
