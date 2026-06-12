"""
inventory_env.py — Ambiente de simulação padronizado
CENÁRIO FIXO: mesma sequência de demanda, mesmo número de períodos,
mesma semente para TODOS os experimentos — permite comparação justa.

MULTI-NÍVEL: Suporta rede de suprimento warehouse → store → produto
- demand_series pode ser agregada (warehouse+store) ou por PDV individual
- lead_time modela reabastecimento de warehouse para store
- Dinâmica Bullwhip effect entre níveis hierárquicos
"""
import numpy as np


class InventoryEnv:
    """
    Simulador (s,Q) com lead time e cenário completamente determinístico.
    Suporta análise single-echelon ou multi-level (warehouse network).

    Garantia de equivalência entre políticas:
    - demand_series é FIXA (não amostrada) para todos
    - lead_time, custos e estoque inicial são idênticos
    - pipeline de pedidos reseta ao mesmo estado

    State vector (6 dim) — eq. (2.17) da dissertação:
        [I_t/I_0, D̂_{t+1}/μ, OP_t/I_0, μ_6(D)/μ, σ_6(D)/σ, t/T]

    onde D̂_{t+1} é a previsão de demanda do próximo ciclo (saída do modelo
    de previsão). Se forecast_series não for fornecida, usa previsão ingênua
    D̂_{t+1} = D_{t-1}.

    Contexto Multi-Nível (opcional):
    - warehouse: identificador do estoque (ex: estado, código UF)
    - store_id: identificador do ponto de venda
    - product_id: identificador do produto
    """

    STATE_DIM = 6

    def __init__(self, demand_series: np.ndarray, cfg: dict, seed: int = None,
                 warehouse: str = None, store_id: str = None, product_id: str = None,
                 forecast_series: np.ndarray = None):
        """
        Args:
            demand_series: série temporal de demanda (agregada ou por PDV)
            cfg: configuração (SIMULATION + COST)
            seed: semente para reprodutibilidade
            warehouse: (opcional) identificador do warehouse/estoque
            store_id: (opcional) identificador da loja/PDV
            product_id: (opcional) identificador do produto
            forecast_series: (opcional) D̂_{t+1} pré-calculado pelo modelo de previsão
                             (mesmo comprimento de demand_series). Se None, usa previsão
                             ingênua D_{t-1}.
        """
        self.demand       = demand_series.astype(float)
        self.T            = len(demand_series)
        self._global_mu   = float(np.mean(demand_series))
        self._global_std  = max(float(np.std(demand_series)), 1e-6)

        # Previsão de demanda para o próximo ciclo (saída do modelo de previsão)
        if forecast_series is not None:
            self._forecast = np.asarray(forecast_series, dtype=float)
        else:
            # Previsão ingênua: D̂_{t+1} = D_{t-1}
            naive = np.concatenate([[self._global_mu], demand_series[:-1]])
            self._forecast = naive.astype(float)

        # Contexto multi-nível (para rastreabilidade)
        self.warehouse = warehouse
        self.store_id = store_id
        self.product_id = product_id

        scfg = cfg["SIMULATION"]
        ccfg = cfg["COST"]
        self.lead_time = int(scfg.get("lead_time", 2))
        self.init_inv  = float(scfg.get("initial_inventory", 100))
        self.h_cost    = float(ccfg.get("holding_cost_per_unit", 1.0))
        self.s_cost    = float(ccfg.get("stockout_cost_per_unit", 5.0))
        self.o_fixed   = float(ccfg.get("ordering_cost_per_order", 50.0))
        self.o_unit    = float(ccfg.get("ordering_cost_per_unit", 0.5))
        self.reset()

    def reset(self):
        self.t           = 0
        self.inventory   = self.init_inv
        self.pipeline    = [0.0] * (self.lead_time + 1)
        self.total_cost  = 0.0
        self.n_orders    = 0
        self._demand_served   = 0.0
        self._demand_total    = 0.0
        self._stockout_events = 0
        self.inv_history      = []
        self.order_history    = []
        self.demand_history   = []
        self.stockout_history = []
        return self._state()

    def _state(self):
        # μ_6(D) e σ_6(D): média e desvio padrão dos últimos 6 ciclos (seção 2.5.5)
        lb = min(self.t, 6)
        if lb > 1:
            recent = self.demand[max(0, self.t - lb): self.t]
            mu_r  = float(np.mean(recent))
            std_r = float(np.std(recent))
        else:
            mu_r  = self._global_mu
            std_r = self._global_std

        # D̂_{t+1}: previsão de demanda para o próximo ciclo (seção 2.5.5)
        d_hat_next = float(self._forecast[self.t]) if self.t < self.T else 0.0
        pending    = float(sum(self.pipeline))
        return np.array([
            self.inventory / (self.init_inv + 1e-6),
            d_hat_next     / (self._global_mu + 1e-6),
            pending        / (self.init_inv + 1e-6),
            mu_r           / (self._global_mu + 1e-6),
            std_r          / (self._global_std + 1e-6),
            self.t / max(self.T, 1),
        ], dtype=np.float32)

    def step(self, order_qty: float):
        order_qty = max(0.0, float(order_qty))
        arrived   = self.pipeline.pop(0)
        self.inventory += arrived
        self.pipeline.append(order_qty)
        if order_qty > 0:
            self.n_orders += 1

        d = float(self.demand[self.t]) if self.t < self.T else 0.0
        served   = min(self.inventory, d)
        shortage = max(0.0, d - served)
        self.inventory = max(0.0, self.inventory - d)

        hc = self.h_cost * self.inventory
        sc = self.s_cost * shortage
        oc = (self.o_fixed + self.o_unit * order_qty) if order_qty > 0 else 0.0
        cost = hc + sc + oc
        self.total_cost       += cost
        self._demand_served   += served
        self._demand_total    += d
        if shortage > 0:
            self._stockout_events += 1

        self.inv_history.append(self.inventory)
        self.order_history.append(order_qty)
        self.demand_history.append(d)
        self.stockout_history.append(shortage)

        # r_t = -(h·I_t + c_s·S_t + c_o^fix·1(Q_t>0) + c_o^var·Q_t)  — eq. (2.17)
        reward = -cost
        self.t += 1
        done   = (self.t >= self.T)
        ns     = self._state() if not done else np.zeros(self.STATE_DIM, np.float32)
        return ns, reward, done, {"shortage": shortage, "cost": cost}

    def kpis(self) -> dict:
        orders  = np.array(self.order_history, dtype=float)
        demands = np.array(self.demand_history, dtype=float)
        sl      = self._demand_served / max(self._demand_total, 1e-9)
        sr      = self._stockout_events / max(self.T, 1)
        var_d   = np.var(demands)
        var_o   = np.var(orders)
        bw      = var_o / max(var_d, 1e-9)
        of      = self.n_orders / max(self.T, 1)
        return {
            "TIC":            self.total_cost,
            "ServiceLevel":   sl,
            "StockoutRate":   sr,
            "StockoutEvents": self._stockout_events,
            "N_Orders":       self.n_orders,
            "OrderFrequency": of,
            "BullwhipEffect": bw,
        }

    def run_policy(self, policy_fn, n_reps: int = 5,
                   base_seed: int = 0) -> dict:
        """
        Roda n_reps vezes com a MESMA demand_series (determinística).
        As replicações diferem apenas no ruído interno (se houver),
        garantindo comparação justa entre políticas.
        """
        all_kpis    = []
        all_inv     = []
        all_orders  = []
        all_stocks  = []

        for rep in range(n_reps):
            state = self.reset()
            done  = False
            while not done:
                qty   = policy_fn(state, self)
                state, _, done, _ = self.step(qty)
            k = self.kpis()
            all_kpis.append(k)
            all_inv.append(list(self.inv_history))
            all_orders.append(list(self.order_history))
            all_stocks.append(list(self.stockout_history))

        agg = {}
        for key in all_kpis[0]:
            vals = [k[key] for k in all_kpis]
            agg[key]         = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))

        min_len = min(len(h) for h in all_inv)
        return {
            "kpis":            agg,
            "inv_history":     np.mean([h[:min_len] for h in all_inv], axis=0),
            "inv_std":         np.std([h[:min_len] for h in all_inv], axis=0),
            "order_history":   np.mean([h[:min_len] for h in all_orders], axis=0),
            "stockout_history":np.mean([h[:min_len] for h in all_stocks], axis=0),
            "demand_history":  list(self.demand_history[:min_len]),
        }
