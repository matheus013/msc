"""
inventory_env_torch.py — Simulador de inventário vetorizado em PyTorch.

Equivalente numérico do `InventoryEnv` original, mas avança B simulações
independentes em paralelo no mesmo passo de tempo. É isso que torna a GPU
útil neste projeto: as meta-heurísticas avaliam centenas de parametrizações
candidatas por geração, e cada avaliação era antes um laço Python sobre T
ciclos. Aqui a população inteira avança junta.

Ganho esperado: o custo passa de O(B x T) chamadas Python para O(T) operações
tensoriais de largura B.

Fidelidade ao ambiente original (garantida por teste de equivalência em
`tests/test_env_equivalence.py`):
  - lost sales: demanda não atendida é perdida, não acumulada
  - pipeline de L+1 posições; pedido emitido em t chega em t+L
  - ordem das operações por ciclo: chegada -> demanda -> custo
  - custo = h*I_t + c_s*S_t + c_o^fix*1(Q_t>0) + c_o^var*Q_t
  - mesmos KPIs: TIC, ServiceLevel, StockoutRate, BullwhipEffect, OrderFrequency

Convenção de shapes:
    demand      (T,)      série compartilhada por todo o lote
    inventory   (B,)
    pipeline    (B, L+1)   coluna 0 chega no ciclo corrente
    order       (B,)
"""
from __future__ import annotations

import numpy as np
import torch

from simulation.core.device import default_dtype, get_device_for_batch


class BatchInventoryEnv:
    """
    Simulador vetorizado. Uma instância representa B trajetórias paralelas
    sobre a MESMA série de demanda, diferindo apenas na política/parâmetros —
    exatamente o que as meta-heurísticas precisam.

    O estado exposto a políticas segue o vetor de 6 componentes da Equação
    (2.17) da dissertação, agora com dimensão de lote:
        [I_t/I_0, D_hat_{t+1}/mu, OP_t/I_0, mu_6/mu, sigma_6/sigma, t/T]
    """

    STATE_DIM = 6

    def __init__(self, demand, cfg: dict, batch_size: int = 1,
                 device=None, forecast=None):
        self.device = device if device is not None else get_device_for_batch(
            int(batch_size), cfg.get("SIMULATION", {}).get("device", "auto"))
        self.dtype = default_dtype(self.device)
        d = torch.as_tensor(np.asarray(demand, dtype=np.float32),
                            dtype=self.dtype, device=self.device)
        self.demand = d
        self.T = int(d.numel())
        self.B = int(batch_size)

        self._mu = float(d.mean().item()) if self.T else 0.0
        self._sd = max(float(d.std(unbiased=False).item()) if self.T else 0.0, 1e-6)

        if forecast is not None:
            f = torch.as_tensor(np.asarray(forecast, dtype=np.float32),
                                dtype=self.dtype, device=self.device)
        else:
            # previsão ingênua D_hat_{t+1} = D_{t-1}, igual ao ambiente original
            f = torch.cat([torch.full((1,), self._mu, dtype=self.dtype,
                                      device=self.device), d[:-1]])
        self.forecast = f

        scfg = cfg["SIMULATION"]
        ccfg = cfg["COST"]
        self.L = int(scfg.get("lead_time", 2))
        self.init_inv = float(scfg.get("initial_inventory", 100))
        self.h = float(ccfg.get("holding_cost_per_unit", 1.0))
        self.cs = float(ccfg.get("stockout_cost_per_unit", 5.0))
        self.of = float(ccfg.get("ordering_cost_per_order", 50.0))
        self.ou = float(ccfg.get("ordering_cost_per_unit", 0.5))

        # médias/desvios móveis de 6 ciclos, pré-computados (idênticos para
        # todo o lote porque a série é compartilhada)
        self._mu6, self._sd6 = self._rolling_stats()
        self.reset()

    # ── pré-cálculo ──────────────────────────────────────────────────────
    def _rolling_stats(self):
        mu6 = torch.empty(self.T, dtype=self.dtype, device=self.device)
        sd6 = torch.empty(self.T, dtype=self.dtype, device=self.device)
        for t in range(self.T):
            lb = min(t, 6)
            if lb > 1:
                w = self.demand[max(0, t - lb):t]
                mu6[t] = w.mean()
                sd6[t] = w.std(unbiased=False)
            else:
                mu6[t] = self._mu
                sd6[t] = self._sd
        return mu6, sd6

    # ── ciclo de vida ────────────────────────────────────────────────────
    def reset(self):
        B, dev, dt = self.B, self.device, self.dtype
        self.t = 0
        self.inventory = torch.full((B,), self.init_inv, dtype=dt, device=dev)
        self.pipeline = torch.zeros((B, self.L + 1), dtype=dt, device=dev)
        self.total_cost = torch.zeros(B, dtype=dt, device=dev)
        self.n_orders = torch.zeros(B, dtype=dt, device=dev)
        self.served = torch.zeros(B, dtype=dt, device=dev)
        self.demand_total = torch.zeros(B, dtype=dt, device=dev)
        self.stockout_events = torch.zeros(B, dtype=dt, device=dev)
        self._order_sum = torch.zeros(B, dtype=dt, device=dev)
        self._order_sqsum = torch.zeros(B, dtype=dt, device=dev)
        self.holding_cost = torch.zeros(B, dtype=dt, device=dev)
        self.stockout_cost = torch.zeros(B, dtype=dt, device=dev)
        self.order_cost = torch.zeros(B, dtype=dt, device=dev)
        return self.state()

    def state(self) -> torch.Tensor:
        """(B, 6) — mesmo vetor de estado do ambiente original."""
        t = min(self.t, self.T - 1) if self.T else 0
        dhat = self.forecast[t] if self.t < self.T else torch.zeros(
            (), dtype=self.dtype, device=self.device)
        pend = self.pipeline.sum(dim=1)
        B = self.B
        col = lambda v: torch.full((B,), float(v), dtype=self.dtype, device=self.device)
        return torch.stack([
            self.inventory / (self.init_inv + 1e-6),
            col(dhat) / (self._mu + 1e-6),
            pend / (self.init_inv + 1e-6),
            col(self._mu6[t]) / (self._mu + 1e-6),
            col(self._sd6[t]) / (self._sd + 1e-6),
            col(self.t / max(self.T, 1)),
        ], dim=1)

    def step(self, order: torch.Tensor):
        """Avança um ciclo para todo o lote. `order` shape (B,)."""
        order = torch.clamp(order.to(self.dtype), min=0.0)

        # 1. chegada do pedido emitido L ciclos atrás
        arrived = self.pipeline[:, 0]
        self.inventory = self.inventory + arrived
        # desloca pipeline e insere o pedido novo na última posição
        self.pipeline = torch.cat([self.pipeline[:, 1:], order.unsqueeze(1)], dim=1)

        placed = (order > 0).to(self.dtype)
        self.n_orders = self.n_orders + placed

        # 2. realização da demanda (lost sales)
        d = self.demand[self.t] if self.t < self.T else torch.zeros(
            (), dtype=self.dtype, device=self.device)
        served = torch.minimum(self.inventory, d.expand(self.B))
        shortage = torch.clamp(d - served, min=0.0)
        self.inventory = torch.clamp(self.inventory - d, min=0.0)

        # 3. custos
        h_cost = self.h * self.inventory
        s_cost = self.cs * shortage
        o_cost = placed * (self.of + self.ou * order)
        cost = h_cost + s_cost + o_cost
        self.total_cost = self.total_cost + cost
        self.holding_cost = self.holding_cost + h_cost
        self.stockout_cost = self.stockout_cost + s_cost
        self.order_cost = self.order_cost + o_cost
        self.served = self.served + served
        self.demand_total = self.demand_total + d
        self.stockout_events = self.stockout_events + (shortage > 0).to(self.dtype)
        self._order_sum = self._order_sum + order
        self._order_sqsum = self._order_sqsum + order * order

        self.t += 1
        done = self.t >= self.T
        return self.state(), -cost, done, {"shortage": shortage}

    # ── execução de política ─────────────────────────────────────────────
    def run(self, policy_fn) -> dict:
        """
        Executa `policy_fn(state, env) -> order (B,)` até o fim do horizonte.
        Retorna KPIs, cada um tensor (B,).
        """
        state = self.reset()
        done = False
        while not done:
            state, _, done, _ = self.step(policy_fn(state, self))
        return self.kpis()

    def run_threshold(self, ROP: torch.Tensor, Q: torch.Tensor,
                      SS: torch.Tensor) -> dict:
        """
        Caso especializado e mais rápido: política (s,Q) com limiar.

            order = Q  se  (I_t + pipeline) <= ROP + SS,  senão 0

        É a regra induzida por GA, SA, PSO e DE (Seção 4.4 da proposta), e o
        caminho quente de todo o benchmark de meta-heurísticas.
        """
        thr = (ROP + SS).to(self.dtype)
        Q = Q.to(self.dtype)
        self.reset()
        done = False
        while not done:
            pos = self.inventory + self.pipeline.sum(dim=1)
            order = torch.where(pos <= thr, Q,
                                torch.zeros_like(Q))
            _, _, done, _ = self.step(order)
        return self.kpis()

    def kpis(self) -> dict:
        T = max(self.T, 1)
        sl = self.served / torch.clamp(self.demand_total, min=1e-9)
        sr = self.stockout_events / T
        of = self.n_orders / T
        var_o = self._order_sqsum / T - (self._order_sum / T) ** 2
        var_o = torch.clamp(var_o, min=0.0)
        var_d = float(self.demand.var(unbiased=False).item()) if self.T else 1.0
        bw = var_o / max(var_d, 1e-9)
        return {
            "TIC": self.total_cost,
            "ServiceLevel": sl,
            "StockoutRate": sr,
            "StockoutEvents": self.stockout_events,
            "N_Orders": self.n_orders,
            "OrderFrequency": of,
            "BullwhipEffect": bw,
            "HoldingCost": self.holding_cost,
            "StockoutCost": self.stockout_cost,
            "OrderCost": self.order_cost,
            "AvgInventory": self.holding_cost / max(self.h, 1e-9) / T,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Função objetivo compartilhada
# ═══════════════════════════════════════════════════════════════════════════

def _risk_terms(kpis: dict, tr_weight: float, be_weight: float,
                tic_ref: torch.Tensor) -> torch.Tensor:
    """
    Termos de risco somados a QUALQUER modo de fitness (2026-08-18, pedido
    explicito do usuario): TIC so registra o custo de ruptura REALIZADO na
    trajetoria de demanda simulada; TR (StockoutRate) e BE (BullwhipEffect)
    capturam duas formas de fragilidade que TIC sozinho nao penaliza.

    TR * (1-NS): TR e a PROBABILIDADE de ruptura (fracao de ciclos em que
    falta produto pro cliente -- independente da magnitude da falta, que e
    o que NS/ServiceLevel mede). Modulado por (1-NS): o risco composto
    cresce quando a politica ja tem servico ruim E rompe com frequencia --
    as duas coisas juntas sao pior sinal de fragilidade do que qualquer
    uma isolada. Escalado por tic_ref (o proprio TIC da trajetoria,
    piso 1.0) para ficar comparavel entre series de volumes muito
    diferentes -- mesmo truque ja usado na penalidade de deficit de NS.

    max(BE-1, 0): BE = var(pedidos)/var(demanda). BE<=1 significa que a
    politica pede de forma tao ou mais suave que a propria demanda --
    comportamento normal, sem punicao. BE>1 e amplificacao: a politica
    pede de forma mais erratica/excessiva do que a demanda justificaria
    ("compra excessiva") -- so esse excesso acima de 1 e punido.
    """
    tr = kpis.get("StockoutRate", 0.0)
    ns = kpis.get("ServiceLevel", 1.0)
    be = kpis.get("BullwhipEffect", 0.0)
    risk = tr_weight * tr * (1.0 - ns) * tic_ref
    excess = be_weight * torch.clamp(be - 1.0, min=0.0) * tic_ref \
        if torch.is_tensor(be) else be_weight * max(be - 1.0, 0.0) * tic_ref
    return risk + excess


def constrained_cost(kpis: dict, alpha_min: float = 0.70,
                     penalty_weight: float = 10.0,
                     tr_weight: float = 1.0, be_weight: float = 1.0) -> torch.Tensor:
    """
    Custo a MINIMIZAR, implementando a Equação (4.2):

        min CTI(theta)   sujeito a   NS(theta) >= alpha_min

    Idêntica à formulação escalar de `policies._eval_static`, aqui vetorizada:
    penalidade proporcional ao déficit de serviço e relativa ao próprio custo,
    o que mantém a grandeza comparável entre séries de volumes muito distintos.

    Acrescenta os termos de risco de `_risk_terms` (TR, BE) — ver docstring.
    """
    tic = kpis["TIC"]
    tic_ref = torch.clamp(tic, min=1.0)
    deficit = torch.clamp(alpha_min - kpis["ServiceLevel"], min=0.0)
    return tic + deficit * penalty_weight * tic_ref + _risk_terms(kpis, tr_weight, be_weight, tic_ref)


def zabraoui_fitness_cost(kpis: dict, lambda1: float = 1.0,
                          lambda2: float = 1e-4,
                          tr_weight: float = 1.0, be_weight: float = 1.0) -> torch.Tensor:
    """
    Custo a MINIMIZAR correspondente à aptidão do Zabraoui et al. (2025),
    Equação (3):

        Fitness = lambda1 * ServiceLevel - lambda2 * TotalInventoryCost

    Como o restante do código minimiza, devolve-se o negativo da aptidão.

    Ressalva metodológica, registrada uma vez: esta soma ponderada é
    sensível à escala do CTI. Como o custo varia por ordens de grandeza entre
    séries (razão de 124x entre a maior e a menor loja no estudo piloto), o
    mesmo par (lambda1, lambda2) privilegia serviço em séries de baixo volume
    e custo em séries de alto volume. No M5 do artigo, com itens de alto giro
    e escala homogênea, o efeito é pequeno; no recorte brasileiro, não é.
    A alternativa `constrained_cost` implementa a Eq. (4.2) da proposta e
    permanece disponível.

    Acrescenta os termos de risco de `_risk_terms` (TR, BE) — ver docstring.
    Nota: aqui a aptidao original do artigo NAO tem esses termos; a soma
    abaixo e uma extensao do portfolio desta dissertacao sobre o artigo-base,
    nao uma reproducao literal da Eq. (3).
    """
    tic_ref = torch.clamp(kpis["TIC"], min=1.0)
    return -(lambda1 * kpis["ServiceLevel"] - lambda2 * kpis["TIC"]) \
        + _risk_terms(kpis, tr_weight, be_weight, tic_ref)


def fitness_cost(kpis: dict, mode: str = "zabraoui", **kw) -> torch.Tensor:
    """Despacha a função objetivo conforme o modo configurado."""
    tr_weight = kw.get("tr_weight", 1.0)
    be_weight = kw.get("be_weight", 1.0)
    if mode in ("zabraoui", "weighted"):
        return zabraoui_fitness_cost(
            kpis, kw.get("lambda1", 1.0), kw.get("lambda2", 1e-4), tr_weight, be_weight)
    if mode == "constrained":
        return constrained_cost(
            kpis, kw.get("alpha_min", 0.70), kw.get("penalty_weight", 10.0), tr_weight, be_weight)
    raise ValueError(f"fitness_mode invalido: {mode!r}")


def evaluate_threshold_population(demand, cfg: dict, params: torch.Tensor,
                                  alpha_min: float = 0.70,
                                  penalty_weight: float = 10.0,
                                  device=None) -> torch.Tensor:
    """
    Avalia uma população inteira de parametrizações de uma vez.

    params: (B, 3) com colunas (ROP, Q, SS)
    retorna: (B,) custo restrito, a minimizar

    Este é o ponto de entrada usado por GA, SA, PSO e DE.
    """
    B = params.shape[0]
    env = BatchInventoryEnv(demand, cfg, batch_size=B, device=device)
    p = params.to(device=env.device, dtype=env.dtype)
    k = env.run_threshold(p[:, 0], torch.clamp(p[:, 1], min=1.0), p[:, 2])
    return constrained_cost(k, alpha_min, penalty_weight)
