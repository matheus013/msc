"""
policies_zabraoui.py — Políticas heurísticas conforme Zabraoui et al. (2025).

Referência
----------
Zabraoui, O., Hmamou, Y., Chafi, A., Kammouri Alami, S.
"A comparative study of multi-algorithm optimization for inventory analytics
in supply chains." Supply Chain Analytics 12 (2025) 100154. CC-BY 4.0.
PDF local: docs/references/1-s2.0-S2949863525000548-main.pdf

Por que este módulo existe
--------------------------
Para as políticas do portfólio em que NÃO existe estado da arte publicado em
veículo relevante, a decisão foi adotar a versão do artigo-base. A leitura do
texto completo mostra, porém, que o conjunto de heurísticas do Zabraoui é
diferente do nosso: a Seção 3.5 ("Heuristic models for baseline comparison")
define quatro políticas de referência —

    (s,S)  ·  Min-Max  ·  Fixed Interval Replenishment  ·  Vendor-Responsive

Destas, apenas (s,S) já existia no portfólio. As outras três são implementadas
aqui. EOQ aparece no artigo somente na revisão de literatura, não como política
avaliada; Jornaleiro, SA, PSO, DE e SARSA não aparecem em nenhuma seção.

Modelo de custo do artigo (Eq. 1)
--------------------------------
    J = sum_t ( c_h * I_t + c_p * S_t + c_o * R_t )

onde R_t é a FREQUÊNCIA de pedido (número de pedidos emitidos), não a
quantidade. O artigo é explícito: "c_o × R_t correctly reflects the total
ordering cost ... by distinguishing between the number of ordering actions and
the quantity purchased". Ou seja, **não há custo variável por unidade pedida**.

O ambiente desta dissertação inclui c_o^var * Q_t. Para reproduzir o custo do
artigo, use `zabraoui_cost(cfg)`, que zera esse termo. A diferença não é
cosmética: com c_o^var > 0, pedidos grandes são penalizados, o que desloca o
ótimo de Q para baixo e favorece políticas de reposição frequente.

Fidelidade
----------
Min-Max e Fixed Interval são descritas no artigo em uma frase cada, sem
formulação. As implementações abaixo seguem a definição clássica de cada uma e
estão comentadas onde houve escolha nossa. Vendor-Responsive é a mais
subespecificada das quatro ("adapts order timing based on supplier lead times
and historical service levels") — a formulação adotada está documentada na
própria classe e NÃO deve ser apresentada como sendo exatamente a do artigo.
"""
from __future__ import annotations

import copy

import numpy as np


def zabraoui_cost(cfg: dict) -> dict:
    """
    Cópia de `cfg` com o modelo de custo da Equação (1) do artigo: custo de
    pedido puramente fixo, sem componente por unidade.

    Use quando o objetivo for reproduzir os números do Zabraoui. Para os
    experimentos da dissertação, mantenha o custo variável, que reflete o
    contrato logístico da rede varejista estudada.
    """
    out = copy.deepcopy(cfg)
    out["COST"]["ordering_cost_per_unit"] = 0.0
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 1. Min-Max
# ═══════════════════════════════════════════════════════════════════════════

class MinMaxPolicy:
    """
    "The Min-Max policy, which maintains inventory within a predefined range."
    (Zabraoui et al., 2025, Seção 3.5)

    Quando a posição de estoque atinge ou cai abaixo de `min`, pede o
    suficiente para chegar a `max`:

        Q_t = max(0, max_level - IP_t)   se IP_t <= min_level,  senão 0

    Diferença operacional em relação a (s,S): no varejo, Min-Max usa faixas
    definidas pelo planejamento (cobertura de demanda), não parâmetros
    derivados de otimização de custo. É essa a leitura adotada aqui:

        min_level = demanda média durante o lead time
        max_level = min_level + cobertura de `coverage_cycles` ciclos

    O artigo não fornece a fórmula; esta é a definição clássica da política.
    """

    def __init__(self, demand, cfg: dict, coverage_cycles: float = 3.0):
        d = np.asarray(demand, dtype=float)
        mu = float(np.mean(d)) if d.size else 0.0
        L = int(cfg["SIMULATION"].get("lead_time", 2))
        self.min_level = mu * L
        self.max_level = self.min_level + mu * float(coverage_cycles)
        if self.max_level <= self.min_level:
            self.max_level = self.min_level + max(1.0, mu)

    def __call__(self, state, env) -> float:
        ip = env.inventory + sum(env.pipeline)
        return max(0.0, self.max_level - ip) if ip <= self.min_level else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Fixed Interval Replenishment
# ═══════════════════════════════════════════════════════════════════════════

class FixedIntervalPolicy:
    """
    "The Fixed Interval Replenishment policy, which schedules periodic reviews
    with consistent order quantities." (Zabraoui et al., 2025, Seção 3.5)

    Revisão a cada `review_period` ciclos; nos ciclos de revisão, repõe até o
    nível-alvo:

        S = mu * (R + L) + z * sigma * sqrt(R + L)

    que é o nível de reposição periódica clássico, cobrindo o intervalo de
    revisão mais o lead time. O artigo menciona "consistent order quantities";
    a implementação repõe até o alvo, o que é a leitura padrão da política de
    revisão periódica (R,S). A variante de lote estritamente constante está
    disponível por `fixed_quantity=True`.
    """

    def __init__(self, demand, cfg: dict, review_period: int = 3,
                 fixed_quantity: bool = False):
        d = np.asarray(demand, dtype=float)
        mu = float(np.mean(d)) if d.size else 0.0
        sd = float(np.std(d)) if d.size else 0.0
        L = int(cfg["SIMULATION"].get("lead_time", 2))
        z = float(cfg.get("HEURISTIC", {}).get("z_score", 1.28))

        self.R = max(1, int(review_period))
        self.fixed_quantity = bool(fixed_quantity)
        horizon = self.R + L
        self.S = mu * horizon + z * sd * np.sqrt(horizon)
        self.Q_fixed = max(1.0, mu * self.R)

    def __call__(self, state, env) -> float:
        if env.t % self.R != 0:
            return 0.0
        if self.fixed_quantity:
            return self.Q_fixed
        ip = env.inventory + sum(env.pipeline)
        return max(0.0, self.S - ip)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Vendor-Responsive
# ═══════════════════════════════════════════════════════════════════════════

class VendorResponsivePolicy:
    """
    "The Vendor-Responsive policy, which adapts order timing based on supplier
    lead times and historical service levels." (Zabraoui et al., 2025, Seção 3.5)

    ATENÇÃO À FIDELIDADE: esta é a política menos especificada do artigo — uma
    frase, sem equação, sem hiperparâmetros. A formulação abaixo é uma leitura
    razoável dessa descrição, NÃO uma transcrição. Não a apresente na
    dissertação como sendo exatamente a implementação do Zabraoui.

    Formulação adotada. O ponto de pedido parte do padrão de lead time e é
    corrigido a cada ciclo pelo nível de serviço realizado até ali:

        s_t = mu_L + z_t * sigma * sqrt(L)
        z_t = z_0 * (1 + kappa * (alpha_target - SL_realizado_t))

    Quando o serviço observado fica abaixo da meta, z_t sobe e a política passa
    a pedir mais cedo; quando fica acima, z_t desce e o estoque de segurança
    encolhe. É esse laço de realimentação que a descrição do artigo sugere ao
    falar em "historical service levels". O pedido repõe até um nível-alvo que
    cobre o lead time mais um ciclo.
    """

    def __init__(self, demand, cfg: dict, alpha_target: float = 0.95,
                 kappa: float = 2.0):
        d = np.asarray(demand, dtype=float)
        self.mu = float(np.mean(d)) if d.size else 0.0
        self.sd = float(np.std(d)) if d.size else 0.0
        self.L = int(cfg["SIMULATION"].get("lead_time", 2))
        self.z0 = float(cfg.get("HEURISTIC", {}).get("z_score", 1.28))
        self.alpha = float(alpha_target)
        self.kappa = float(kappa)

    def _realized_sl(self, env) -> float:
        total = getattr(env, "_demand_total", 0.0)
        if total <= 0:
            return 1.0
        return float(getattr(env, "_demand_served", 0.0)) / total

    def __call__(self, state, env) -> float:
        sl = self._realized_sl(env)
        z = self.z0 * (1.0 + self.kappa * (self.alpha - sl))
        z = float(np.clip(z, 0.0, 4.0))          # evita explodir em séries ruins
        s = self.mu * self.L + z * self.sd * np.sqrt(self.L)
        ip = env.inventory + sum(env.pipeline)
        if ip > s:
            return 0.0
        target = self.mu * (self.L + 1) + z * self.sd * np.sqrt(self.L + 1)
        return max(0.0, target - ip)
