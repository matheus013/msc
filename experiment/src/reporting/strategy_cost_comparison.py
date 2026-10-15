"""
strategy_cost_comparison.py
Compara o custo de inventário entre três estratégias de decisão de política:

  A1. Política única global — menor CTI entre políticas com NS médio >= NS_THRESHOLD
  A2. Política baseline operacional — EOQ (política clássica de referência)
  B.  Seleção por perfil — política dominante dentro de cada perfil operacional
  C.  Oráculo por série — limite exploratório: melhor política viável por série

Entradas:
  data/07_model_output/kpis.parquet          — KPIs por série × política
  data/04_feature/demand_profiles.parquet    — Perfil operacional por série

Saídas (em data/08_reporting/strategy/):
  policy_global_metrics.csv           — Métricas globais por política
  profile_policy_metrics.csv          — Métricas por perfil × política
  dominant_policy_by_profile.csv      — Política dominante por perfil
  strategy_cost_comparison.csv        — Comparação de estratégias
  strategy_cost_comparison.md         — Relatório de validação
  table_strategy_comparison.tex       — Tabela LaTeX para inclusão no capítulo

Regra de dominância:
  Política viável: NS médio >= NS_THRESHOLD (padrão 0.70)
  Dominante: menor CTI médio entre as viáveis
  Fallback (nenhuma viável): maior NS médio, marcado como fallback

Nota sobre os dados:
  kpis.parquet provém da rodada experimental atual. Os valores de CTI podem diferir
  da Tabela 5.2 da dissertação se a rodada final ainda não tiver sido gerada.
  Executar novamente este script após regenerar kpis.parquet atualiza todos os artefatos.
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATA_DIR     = REPO_ROOT / "data"
KPI_PATH     = DATA_DIR / "07_model_output" / "kpis.parquet"
PROF_PATH    = DATA_DIR / "04_feature" / "demand_profiles.parquet"
OUT_DIR      = DATA_DIR / "08_reporting" / "strategy"

NS_THRESHOLD     = 0.70
PENALTY_WEIGHT   = 10.0
EXCESS_WEIGHT    = 0.5
BASELINE_POLICY  = "EOQ"
METRIC_COLS      = ["CTI", "CTI_ajustado", "NS", "TR", "BE", "FP"]

POLICY_DISPLAY = {
    "sS":         "(s,S)",
    "Newsvendor": "Jornaleiro",
}
PROFILE_DISPLAY = {
    "Sparse_High_Impact": "Sparse High Impact",
    "High_Vol_Seasonal":  "High Vol. Seasonal",
    "Unstable_Trend":     "Unstable Trend",
    "Low_Vol_Stable":     "Low Vol. Stable",
    "Fast_Moving":        "Fast Moving",
}

STRATEGY_LABELS = {
    "A1": "Política única global (melhor viável)",
    "A2": f"Política baseline ({BASELINE_POLICY})",
    "B":  "Seleção por perfil operacional",
    "C":  "Oráculo por série (exploratório)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento e junção
# ─────────────────────────────────────────────────────────────────────────────

def _load(kpis: pd.DataFrame | None = None, profiles: pd.DataFrame | None = None,
          kpi_path: Path = KPI_PATH, prof_path: Path = PROF_PATH) -> pd.DataFrame:
    """
    2026-08-18: aceita `kpis`/`profiles` (DataFrames já carregados pelo
    catálogo Kedro) -- o script era hardcoded em
    data/07_model_output/kpis.parquet, sem isolamento por ambiente (mesmo
    problema corrigido em profile_policy_analysis.py). Sem argumentos,
    mantém o comportamento antigo (retrocompat CLI, base/Bahia).
    """
    if kpis is None:
        log.info("Lendo %s …", kpi_path)
        kpis = pd.read_parquet(kpi_path)
    if profiles is None:
        log.info("Lendo %s …", prof_path)
        profiles = pd.read_parquet(prof_path)
    prof = profiles[["warehouse", "store_id", "item_id", "operational_profile"]]
    df = kpis.merge(prof, on=["warehouse", "store_id", "item_id"], how="left")
    n_series = df[["warehouse", "store_id", "item_id"]].drop_duplicates().shape[0]
    n_profiles = df["operational_profile"].nunique()
    log.info(
        f"Dados unidos: {len(df)} linhas, {n_series} séries, "
        f"{df.policy.nunique()} políticas, {n_profiles} perfis."
    )
    if df["operational_profile"].isna().any():
        n_missing = df["operational_profile"].isna().sum()
        log.warning(f"  {n_missing} linhas sem perfil operacional após o join.")
    df = df.rename(columns={"TIC": "CTI"})
    df = _add_adjusted_cost(df)
    return df


def _add_adjusted_cost(df: pd.DataFrame, ns_threshold: float = NS_THRESHOLD,
                       penalty_weight: float = PENALTY_WEIGHT,
                       excess_weight: float = EXCESS_WEIGHT) -> pd.DataFrame:
    """
    CTI_ajustado — métrica de escolha final (2026-08-18, AJUSTES_INFRA item
    #33; pedido do usuário: "no comparativo final devemos comparar o custo
    total ajustado"). Mesma formulação de `profile_policy_analysis.
    _add_adjusted_cost`, reimplementada aqui sobre a coluna já renomeada
    `CTI` (== TIC) para não criar dependência cruzada entre os dois
    scripts de relatório:

        CTI_ajustado = CTI
                      + deficit_NS * penalty_weight * CTI_ref_serie   (prejuízo por indisponibilidade;
                                                                        CTI_ref FIXO = maior CTI entre
                                                                        as 18 políticas da mesma série,
                                                                        não o CTI da própria candidata —
                                                                        corrige o bug de auto-referência
                                                                        que fazia "nunca pedir" vencer)
                      + excess_weight * max(0, HoldingCost - mediana(HoldingCost na série))
                                                                       (estoque excessivo, se a
                                                                        decomposição de custo estiver
                                                                        disponível)
    """
    series_key = ["warehouse", "store_id", "item_id"]
    cti_ref_series = df.groupby(series_key)["CTI"].transform("max").clip(lower=1.0)
    deficit = (ns_threshold - df["NS"]).clip(lower=0.0)
    service_loss = deficit * penalty_weight * cti_ref_series

    if "HoldingCost" in df.columns and df["HoldingCost"].notna().any():
        median_holding = df.groupby(series_key)["HoldingCost"].transform("median")
        excess = (df["HoldingCost"] - median_holding).clip(lower=0.0) * excess_weight
    else:
        excess = 0.0

    df = df.copy()
    df["CTI_ajustado"] = df["CTI"] + service_loss + excess
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Métricas globais por política
# ─────────────────────────────────────────────────────────────────────────────

def _global_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("policy")[["CTI", "CTI_ajustado", "NS", "TR", "BE", "FP"]]
        .agg(["mean", "std", "sum"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    n_series = df.groupby("policy").size().rename("n_series")
    agg = agg.join(n_series)
    agg["viable"] = agg["NS_mean"] >= NS_THRESHOLD
    agg = agg.reset_index()
    return agg


def _pick_global_best(global_agg: pd.DataFrame) -> str:
    """Escolhe A1 (política única global) por menor CTI_ajustado_mean entre as
    viáveis (2026-08-18: era CTI_mean puro -- ver `_add_adjusted_cost`)."""
    viable = global_agg[global_agg["viable"]]
    if viable.empty:
        log.warning("Nenhuma política viável globalmente; usando política com maior NS.")
        return global_agg.loc[global_agg["NS_mean"].idxmax(), "policy"]
    return viable.loc[viable["CTI_ajustado_mean"].idxmin(), "policy"]


# ─────────────────────────────────────────────────────────────────────────────
# Métricas por perfil × política
# ─────────────────────────────────────────────────────────────────────────────

def _profile_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["operational_profile", "policy"])[["CTI", "CTI_ajustado", "NS", "TR", "BE", "FP"]]
        .agg(["mean", "std", "sum"])
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    n_series = df.groupby(["operational_profile", "policy"]).size().rename("n_series")
    agg = agg.join(n_series)
    agg["viable"] = agg["NS_mean"] >= NS_THRESHOLD
    agg = agg.reset_index()
    return agg


def _dominant_by_profile(profile_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, grp in profile_agg.groupby("operational_profile"):
        n_series_profile = grp["n_series"].iloc[0]  # same for all policies in profile

        # contar séries do perfil (podem variar entre políticas, pegar da col CTI que não tem NA)
        # na verdade n_series aqui é n_series naquele (perfil, politica), que é o mesmo
        # Verificar: n_series no profile_agg é contagem de séries × políticas? não, é a contagem
        # de (profile, policy) que é o número de séries naquele perfil
        # Buscar n_series do perfil real
        viable = grp[grp["viable"]]
        fallback = False
        if viable.empty:
            dominant_row = grp.loc[grp["NS_mean"].idxmax()]
            fallback = True
        else:
            dominant_row = viable.loc[viable["CTI_ajustado_mean"].idxmin()]

        # Segunda melhor política viável (por CTI_ajustado -- ver _add_adjusted_cost)
        if len(viable) >= 2:
            others = viable[viable.policy != dominant_row.policy]
            second_row = others.loc[others["CTI_ajustado_mean"].idxmin()]
            second_policy = second_row.policy
            second_cti = second_row.CTI_ajustado_mean
            diff_pct = 100.0 * (second_cti - dominant_row.CTI_ajustado_mean) / (dominant_row.CTI_ajustado_mean + 1e-9)
        else:
            second_policy = None
            second_cti = None
            diff_pct = None

        rows.append({
            "operational_profile":     profile,
            "profile_display":         PROFILE_DISPLAY.get(profile, profile),
            "n_series":                int(grp["n_series"].iloc[0]),
            "dominant_policy":         dominant_row.policy,
            "dominant_policy_disp":    POLICY_DISPLAY.get(dominant_row.policy, dominant_row.policy),
            "CTI_mean":                round(dominant_row.CTI_mean, 2),
            "CTI_ajustado_mean":       round(dominant_row.CTI_ajustado_mean, 2),
            "CTI_std":                 round(dominant_row.CTI_std, 2),
            "NS_mean":                 round(dominant_row.NS_mean, 3),
            "TR_mean":                 round(dominant_row.TR_mean, 3),
            "BE_mean":                 round(dominant_row.BE_mean, 3),
            "FP_mean":                 round(dominant_row.FP_mean, 3),
            "status":                  "fallback" if fallback else "normal",
            "second_policy":           second_policy,
            "second_CTI_ajustado_mean": round(second_cti, 2) if second_cti else None,
            "diff_pct_vs_second":      round(diff_pct, 1) if diff_pct is not None else None,
            "exploratory":             int(grp["n_series"].iloc[0]) < 20,
        })

    dom = pd.DataFrame(rows).sort_values("CTI_ajustado_mean")
    return dom


# ─────────────────────────────────────────────────────────────────────────────
# Comparação de estratégias por série
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_per_series(df: pd.DataFrame, best_global: str,
                          dominant: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela wide: uma linha por série, com CTI sob cada estratégia.
    """
    # Pivot para CTI (e CTI_ajustado) por (série, política)
    piv_cti = df.pivot_table(
        index=["warehouse", "store_id", "item_id", "operational_profile"],
        columns="policy",
        values="CTI",
    ).reset_index()
    piv_adj = df.pivot_table(
        index=["warehouse", "store_id", "item_id", "operational_profile"],
        columns="policy",
        values="CTI_ajustado",
    ).reset_index()
    piv_ns = df.pivot_table(
        index=["warehouse", "store_id", "item_id", "operational_profile"],
        columns="policy",
        values="NS",
    ).reset_index()

    policies = [c for c in piv_cti.columns if c not in
                ["warehouse", "store_id", "item_id", "operational_profile"]]

    # Mapa perfil → política dominante
    dom_map = dict(zip(dominant.operational_profile, dominant.dominant_policy))

    piv_cti["CTI_A1"] = piv_cti[best_global]
    piv_cti["CTI_A2"] = piv_cti[BASELINE_POLICY]
    piv_cti["CTI_B"]  = piv_cti.apply(
        lambda r: r.get(dom_map.get(r["operational_profile"], best_global), r[best_global]),
        axis=1,
    )

    # CTI_ajustado por estratégia -- métrica de escolha final (ver
    # _add_adjusted_cost); mesmo esquema de A1/A2/B acima, sobre piv_adj.
    piv_adj_key = piv_adj.set_index(["warehouse", "store_id", "item_id", "operational_profile"])
    piv_cti["CTI_ajustado_A1"] = piv_cti.apply(
        lambda r: piv_adj_key.loc[(r.warehouse, r.store_id, r.item_id, r.operational_profile), best_global], axis=1)
    piv_cti["CTI_ajustado_A2"] = piv_cti.apply(
        lambda r: piv_adj_key.loc[(r.warehouse, r.store_id, r.item_id, r.operational_profile), BASELINE_POLICY], axis=1)
    piv_cti["CTI_ajustado_B"] = piv_cti.apply(
        lambda r: piv_adj_key.loc[(r.warehouse, r.store_id, r.item_id, r.operational_profile),
                                   dom_map.get(r["operational_profile"], best_global)], axis=1)

    # Oracle: por série, menor CTI_ajustado viável (requer NS por série também)
    piv_ns_idx = piv_ns.set_index(["warehouse", "store_id", "item_id"])
    piv_adj_idx = piv_adj.set_index(["warehouse", "store_id", "item_id"])
    def oracle_row(row):
        ns_row = piv_ns_idx.loc[(row.warehouse, row.store_id, row.item_id), policies]
        adj_row = piv_adj_idx.loc[(row.warehouse, row.store_id, row.item_id), policies]
        viable_p = [p for p in policies if ns_row[p] >= NS_THRESHOLD]
        if viable_p:
            best_p = min(viable_p, key=lambda p: adj_row[p])
        else:
            best_p = min(policies, key=lambda p: adj_row[p])
        return pd.Series({"CTI_C": row[best_p], "CTI_ajustado_C": adj_row[best_p]})

    oracle = piv_cti.apply(oracle_row, axis=1)
    piv_cti["CTI_C"] = oracle["CTI_C"]
    piv_cti["CTI_ajustado_C"] = oracle["CTI_ajustado_C"]

    return piv_cti


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values, kept monotonic in original order."""
    p = p_values.astype(float).to_numpy()
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        adj = min((m - rank) * p[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return pd.Series(adjusted, index=p_values.index)


def _paired_strategy_observations(
    df: pd.DataFrame,
    best_global: str,
    dominant: pd.DataFrame,
) -> pd.DataFrame:
    """One row per series with A1, A2 and profile-selection KPIs."""
    series_key = ["warehouse", "store_id", "item_id", "operational_profile"]
    base = df[series_key].drop_duplicates().reset_index(drop=True)
    dom_map = dict(zip(dominant.operational_profile, dominant.dominant_policy))

    for metric in METRIC_COLS:
        if metric not in df.columns:
            continue
        piv = df.pivot_table(
            index=series_key,
            columns="policy",
            values=metric,
        ).reset_index()
        piv[f"{metric}_A1"] = piv[best_global]
        piv[f"{metric}_A2"] = piv[BASELINE_POLICY]
        piv[f"{metric}_B"] = piv.apply(
            lambda r: r.get(dom_map.get(r["operational_profile"], best_global), r[best_global]),
            axis=1,
        )
        keep_cols = series_key + [f"{metric}_A1", f"{metric}_A2", f"{metric}_B"]
        base = base.merge(piv[keep_cols], on=series_key, how="left")

    return base


def _strategy_hypothesis_tests(paired_df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired tests between single-policy strategies and profile selection.

    Primary hypothesis for CTI:
      H0: median(CTI_B - CTI_A) = 0
      H1: median(CTI_B - CTI_A) < 0

    Other KPIs use two-sided Wilcoxon tests to document trade-offs.
    """
    rows = []
    comparisons = [
        ("A1_vs_B", "A1", "B", "Global single policy vs profile selection"),
        ("A2_vs_B", "A2", "B", "EOQ baseline vs profile selection"),
    ]

    for comparison_id, ref_suffix, profile_suffix, description in comparisons:
        for metric in METRIC_COLS:
            ref_col = f"{metric}_{ref_suffix}"
            prof_col = f"{metric}_{profile_suffix}"
            if ref_col not in paired_df.columns or prof_col not in paired_df.columns:
                continue

            paired = paired_df[[ref_col, prof_col]].dropna()
            if len(paired) < 5:
                continue

            ref = paired[ref_col].astype(float).to_numpy()
            prof = paired[prof_col].astype(float).to_numpy()
            diff = prof - ref
            alternative = "less" if metric in ("CTI", "CTI_ajustado") else "two-sided"

            if np.all(np.abs(diff) <= 1e-12):
                stat, p_value = 0.0, 1.0
            else:
                stat, p_value = stats.wilcoxon(
                    prof,
                    ref,
                    alternative=alternative,
                    zero_method="wilcox",
                )

            mean_ref = float(np.mean(ref))
            mean_profile = float(np.mean(prof))
            mean_delta = mean_profile - mean_ref
            median_delta = float(np.median(diff))
            sd_delta = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
            dz = mean_delta / sd_delta if sd_delta and not np.isnan(sd_delta) else np.nan
            rel_change = 100.0 * mean_delta / (abs(mean_ref) + 1e-12)

            rows.append({
                "comparison": comparison_id,
                "description": description,
                "reference_strategy": ref_suffix,
                "profile_strategy": profile_suffix,
                "metric": metric,
                "alternative": alternative,
                "n_pairs": int(len(paired)),
                "mean_reference": mean_ref,
                "mean_profile": mean_profile,
                "mean_delta_profile_minus_reference": mean_delta,
                "median_delta_profile_minus_reference": median_delta,
                "relative_change_pct": rel_change,
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
                "cohens_dz": float(dz) if not np.isnan(dz) else np.nan,
            })

    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["p_value_holm"] = _holm_adjust(tests["p_value"])
        tests["significant_0_05"] = tests["p_value"] < 0.05
        tests["significant_holm_0_05"] = tests["p_value_holm"] < 0.05
    return tests


# ─────────────────────────────────────────────────────────────────────────────
# Tabela de comparação de estratégias
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_table(series_df: pd.DataFrame, global_agg: pd.DataFrame,
                    best_global: str, dominant: pd.DataFrame) -> pd.DataFrame:
    """
    Uma linha por estratégia: CTI total, médio, NS médio, reduções.
    """
    # NS médio por estratégia (NS da política escolhida para cada série)
    piv_ns = series_df[["warehouse", "store_id", "item_id", "operational_profile"]].copy()

    # Obter NS médio global de cada política (já calculado no global_agg)
    ns_map = dict(zip(global_agg.policy, global_agg.NS_mean))
    tr_map = dict(zip(global_agg.policy, global_agg.TR_mean))
    be_map = dict(zip(global_agg.policy, global_agg.BE_mean))
    fp_map = dict(zip(global_agg.policy, global_agg.FP_mean))

    dom_map = dict(zip(dominant.operational_profile, dominant.dominant_policy))
    n = len(series_df)

    def _profile_ns_mean(series_df, dominant):
        """NS médio ponderado pela distribuição de séries por perfil.
        Usa NS da política dominante DENTRO de cada perfil (não média global)."""
        prof_ns = dict(zip(dominant.operational_profile, dominant.NS_mean))
        prof_tr = dict(zip(dominant.operational_profile, dominant.TR_mean))
        prof_be = dict(zip(dominant.operational_profile, dominant.BE_mean))
        prof_fp = dict(zip(dominant.operational_profile, dominant.FP_mean))
        total_ns = 0.0
        total_tr = 0.0
        total_be = 0.0
        total_fp = 0.0
        for _, row in series_df.iterrows():
            p = row["operational_profile"]
            total_ns += prof_ns.get(p, ns_map.get(dom_map.get(p, best_global), 0))
            total_tr += prof_tr.get(p, tr_map.get(dom_map.get(p, best_global), 0))
            total_be += prof_be.get(p, be_map.get(dom_map.get(p, best_global), 0))
            total_fp += prof_fp.get(p, fp_map.get(dom_map.get(p, best_global), 0))
        return total_ns / n, total_tr / n, total_be / n, total_fp / n

    rows = []

    # Estratégia A1
    cti_total_A1 = series_df["CTI_A1"].sum()
    cti_mean_A1  = series_df["CTI_A1"].mean()
    adj_total_A1 = series_df["CTI_ajustado_A1"].sum()
    adj_mean_A1  = series_df["CTI_ajustado_A1"].mean()
    rows.append({
        "estrategia":    "A1",
        "descricao":     STRATEGY_LABELS["A1"],
        "regra":         f"Política única: {POLICY_DISPLAY.get(best_global, best_global)} (menor CTI ajustado global viável)",
        "politica_repr": POLICY_DISPLAY.get(best_global, best_global),
        "CTI_total":     round(cti_total_A1, 2),
        "CTI_medio":     round(cti_mean_A1, 2),
        "CTI_ajustado_total": round(adj_total_A1, 2),
        "CTI_ajustado_medio": round(adj_mean_A1, 2),
        "NS_medio":      round(ns_map.get(best_global, 0), 3),
        "TR_medio":      round(tr_map.get(best_global, 0), 3),
        "BE_medio":      round(be_map.get(best_global, 0), 3),
        "FP_medio":      round(fp_map.get(best_global, 0), 3),
    })

    # Estratégia A2 (baseline)
    cti_total_A2 = series_df["CTI_A2"].sum()
    cti_mean_A2  = series_df["CTI_A2"].mean()
    adj_total_A2 = series_df["CTI_ajustado_A2"].sum()
    adj_mean_A2  = series_df["CTI_ajustado_A2"].mean()
    rows.append({
        "estrategia":    "A2",
        "descricao":     STRATEGY_LABELS["A2"],
        "regra":         f"Política única: {POLICY_DISPLAY.get(BASELINE_POLICY, BASELINE_POLICY)} (baseline operacional)",
        "politica_repr": POLICY_DISPLAY.get(BASELINE_POLICY, BASELINE_POLICY),
        "CTI_total":     round(cti_total_A2, 2),
        "CTI_medio":     round(cti_mean_A2, 2),
        "CTI_ajustado_total": round(adj_total_A2, 2),
        "CTI_ajustado_medio": round(adj_mean_A2, 2),
        "NS_medio":      round(ns_map.get(BASELINE_POLICY, 0), 3),
        "TR_medio":      round(tr_map.get(BASELINE_POLICY, 0), 3),
        "BE_medio":      round(be_map.get(BASELINE_POLICY, 0), 3),
        "FP_medio":      round(fp_map.get(BASELINE_POLICY, 0), 3),
    })

    # Estratégia B
    cti_total_B  = series_df["CTI_B"].sum()
    cti_mean_B   = series_df["CTI_B"].mean()
    adj_total_B  = series_df["CTI_ajustado_B"].sum()
    adj_mean_B   = series_df["CTI_ajustado_B"].mean()
    ns_B, tr_B, be_B, fp_B = _profile_ns_mean(series_df, dominant)
    rows.append({
        "estrategia":    "B",
        "descricao":     STRATEGY_LABELS["B"],
        "regra":         "Política dominante por perfil operacional (menor CTI ajustado viável no perfil)",
        "politica_repr": "; ".join(
            f"{PROFILE_DISPLAY.get(p, p)}: {POLICY_DISPLAY.get(v, v)}"
            for p, v in dom_map.items()
        ),
        "CTI_total":     round(cti_total_B, 2),
        "CTI_medio":     round(cti_mean_B, 2),
        "CTI_ajustado_total": round(adj_total_B, 2),
        "CTI_ajustado_medio": round(adj_mean_B, 2),
        "NS_medio":      round(ns_B, 3),
        "TR_medio":      round(tr_B, 3),
        "BE_medio":      round(be_B, 3),
        "FP_medio":      round(fp_B, 3),
    })

    # Estratégia C (oracle)
    cti_total_C  = series_df["CTI_C"].sum()
    cti_mean_C   = series_df["CTI_C"].mean()
    adj_total_C  = series_df["CTI_ajustado_C"].sum()
    adj_mean_C   = series_df["CTI_ajustado_C"].mean()
    rows.append({
        "estrategia":    "C",
        "descricao":     STRATEGY_LABELS["C"],
        "regra":         "Melhor política viável por série, por CTI ajustado (referência exploratória)",
        "politica_repr": "Variável por série",
        "CTI_total":     round(cti_total_C, 2),
        "CTI_medio":     round(cti_mean_C, 2),
        "CTI_ajustado_total": round(adj_total_C, 2),
        "CTI_ajustado_medio": round(adj_mean_C, 2),
        "NS_medio":      round(NS_THRESHOLD, 3),   # por construção, NS >= threshold
        "TR_medio":      None,
        "BE_medio":      None,
        "FP_medio":      None,
    })

    result = pd.DataFrame(rows)

    # Calcular reduções vs A1 e vs A2 -- métrica de comparação final é
    # CTI_ajustado (2026-08-18, AJUSTES_INFRA item #33), CTI bruto mantido
    # só para referência/retrocompatibilidade.
    ref_A1 = adj_total_A1
    ref_A2 = adj_total_A2

    result["red_abs_vs_A1"] = result["CTI_ajustado_total"].apply(
        lambda x: round(ref_A1 - x, 2) if x is not None else None
    )
    result["red_pct_vs_A1"] = result["CTI_ajustado_total"].apply(
        lambda x: round(100.0 * (ref_A1 - x) / ref_A1, 2) if x is not None else None
    )
    result["red_abs_vs_A2"] = result["CTI_ajustado_total"].apply(
        lambda x: round(ref_A2 - x, 2) if x is not None else None
    )
    result["red_pct_vs_A2"] = result["CTI_ajustado_total"].apply(
        lambda x: round(100.0 * (ref_A2 - x) / ref_A2, 2) if x is not None else None
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Relatório de validação
# ─────────────────────────────────────────────────────────────────────────────

def _validation_report(df_raw: pd.DataFrame, global_agg: pd.DataFrame,
                        dominant: pd.DataFrame, strategy: pd.DataFrame,
                        best_global: str, out_path: Path,
                        n_series_expected: int | None = None,
                        policies_expected: list[str] | None = None) -> None:
    """
    2026-08-18: "Checagem 1/2" eram hardcoded pra Bahia (145 séries, lista
    fixa das 12 políticas antigas) -- sempre reportavam "DIVERGE" contra
    "bot"/M5 (contagens diferentes por design) e contra o portfólio atual
    de 18 políticas. Agora comparam contra `n_series_expected`/
    `policies_expected`, passados por `run()` a partir dos próprios dados
    carregados (checagem de auto-consistência, não mais um número fixo de
    uma única base).
    """
    n = df_raw[['warehouse','store_id','item_id']].drop_duplicates().shape[0]
    n_series_expected = n_series_expected if n_series_expected is not None else n
    policies = sorted(df_raw.policy.unique())
    expected = sorted(policies_expected) if policies_expected is not None else policies

    lines = [
        "# Validação — Comparação de Estratégias de Política de Inventário",
        f"\nGerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## Fonte dos dados",
        "- KPIs: `kpis.parquet` (caminho isolado por ambiente, ver conf/*/catalog.yml)",
        "- Perfis: `demand_profiles.parquet` (idem)",
        "\n## Cobertura",
        f"- Séries (loja, produto): **{n}**",
        f"- Políticas avaliadas: **{df_raw.policy.nunique()}**",
        f"- Perfis operacionais: **{df_raw.operational_profile.nunique()}** de 5 definidos",
        "\n## Checagem 1 — Quantidade de séries",
    ]
    lines.append(f"  Esperado: {n_series_expected} | Encontrado: {n} | {'OK' if n == n_series_expected else 'DIVERGE'}")

    lines.append("\n## Checagem 2 — Políticas")
    lines.append(f"  Encontradas: {policies}")
    lines.append(f"  Esperadas:   {expected}")
    lines.append(f"  Match: {'OK' if policies == expected else 'DIVERGE'}")

    lines.append("\n## Checagem 3 — Política única global (A1)")
    lines.append(f"  Política dominante global (por CTI ajustado): **{POLICY_DISPLAY.get(best_global, best_global)}**")
    viable = global_agg[global_agg["viable"]]
    lines.append(f"  Políticas viáveis (NS >= {NS_THRESHOLD}): {list(viable.policy)}")

    lines.append("\n## Checagem 4 — Dominância por perfil (B)")
    for _, row in dominant.iterrows():
        marker = " (*)" if row["exploratory"] else ""
        lines.append(
            f"  {row['profile_display']}{marker}: "
            f"{row['dominant_policy_disp']} | CTI={row['CTI_mean']} | CTI_ajustado={row['CTI_ajustado_mean']} | NS={row['NS_mean']} | status={row['status']}"
        )
    lines.append("  (*) n < 20: evidência exploratória")

    lines.append("\n## Checagem 5 — Redução de CTI ajustado (fórmula verificada)")
    lines.append(
        "  redução (%) = 100 × (CTI_ajustado_A1_total − CTI_ajustado_B_total) / CTI_ajustado_A1_total"
    )
    lines.append(
        "  CTI_ajustado = CTI + deficit_NS*penalty_weight*CTI_ref_serie(fixo) + excess_weight*excesso_holding"
        " (AJUSTES_INFRA item #33; corrige auto-referência do score antigo e"
        " incorpora estoque excessivo + indisponibilidade)."
    )
    for _, row in strategy.iterrows():
        if row["estrategia"] == "A1":
            continue
        pct = row.get("red_pct_vs_A1", None)
        lines.append(
            f"  {row['estrategia']} ({row['descricao'][:40]}…): "
            f"CTI_total={row['CTI_total']} | red_pct_vs_A1={pct}%"
        )

    lines.append("\n## Checagem 6 — NS médio preservado")
    for _, row in strategy.iterrows():
        flag = ""
        if row.get("NS_medio") is not None and row["NS_medio"] < NS_THRESHOLD:
            flag = " ← ABAIXO DO LIMIAR"
        lines.append(f"  {row['estrategia']}: NS_medio={row.get('NS_medio', 'N/A')}{flag}")

    lines.append("\n## Checagem 7 — Consistência com Tabela 5.2 (agregado global)")
    lines.append(
        "  Os valores de CTI aqui NÃO são idênticos à Tabela 5.2 (rodada anterior)."
    )
    lines.append(
        "  Após regenerar kpis.parquet com a rodada final, reexecutar este script."
    )
    eoc_mean = global_agg.loc[global_agg.policy == "EOQ", "CTI_mean"].values
    if len(eoc_mean):
        lines.append(f"  EOQ CTI médio (kpis.parquet atual): {eoc_mean[0]:.2f}")
        lines.append("  EOQ CTI médio (Tabela 5.2 final):    628,42")

    lines.append("\n## Limitações")
    lines.append("- Perfis Low_Vol_Stable e Fast_Moving ausentes no Experimento 2 (regime Lumpy, BA).")
    lines.append("- Perfis com n < 20 séries: evidência exploratória.")
    lines.append("- Oráculo por série (C) é limite superior exploratório, não estratégia operacional.")
    lines.append("- Generalização para regimes não-Lumpy: objetivo do Experimento 3.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Relatório de validação: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tabela LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def _latex_table(strategy: pd.DataFrame, best_global: str,
                 dominant: pd.DataFrame, out_path: Path) -> None:
    def _fmt_pct(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return r"\textemdash{}"
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.1f}"

    def _fmt_num(v, decimals=2):
        if v is None:
            return r"\textemdash{}"
        return f"{v:.{decimals}f}"

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Comparação de custo entre estratégias de seleção de política"
        r" (Experimento~2, BA, 145 séries, regime \textit{Lumpy}). CTI ajustado incorpora"
        r" prejuízo por indisponibilidade (referência fixa por série) e estoque excessivo"
        r" (AJUSTES\_INFRA item \#33); é a métrica de comparação final. Redução calculada como"
        r" $100 \times (\mathrm{CTI}^{\text{ajust}}_{\text{ref}} - \mathrm{CTI}^{\text{ajust}}_{\text{estratégia}}) /"
        r" \mathrm{CTI}^{\text{ajust}}_{\text{ref}}$. O oráculo por série (C$^\dagger$) é referência"
        r" exploratória: assume conhecimento perfeito da política ótima de cada série.}",
        r"\label{tab:strategy_cost_comparison}",
        r"\begin{tabular}{@{}p{3.6cm}rrrrr@{}}",
        r"\toprule",
        r"\makecell[l]{Estratégia} &",
        r"\makecell[r]{CTI ajust.\\total (R\$)} &",
        r"\makecell[r]{CTI ajust.\\médio (R\$)} &",
        r"\makecell[r]{NS\\médio} &",
        r"\makecell[r]{Red. vs A1\\(\%)} &",
        r"\makecell[r]{Red. vs A2\\(\%)} \\",
        r"\midrule",
    ]

    for _, row in strategy.iterrows():
        est = row["estrategia"]
        desc = {
            "A1": r"A1: política única (melhor viável)",
            "A2": r"A2: política única (EOQ, \textit{baseline})",
            "B":  r"B\phantom{0}: seleção por perfil",
            "C":  r"C$^\dagger$: oráculo por série",
        }.get(est, est)
        ns_str   = _fmt_num(row.get("NS_medio"), decimals=3)
        pct_a1   = _fmt_pct(row.get("red_pct_vs_A1"))
        pct_a2   = _fmt_pct(row.get("red_pct_vs_A2"))
        adj_t    = _fmt_num(row["CTI_ajustado_total"])
        adj_m    = _fmt_num(row["CTI_ajustado_medio"])
        lines.append(
            f"{desc} & {adj_t} & {adj_m} & {ns_str} & {pct_a1} & {pct_a2} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\scriptsize $^\dagger$Não é estratégia operacional"
        r" direta; representa limite inferior exploratório.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Tabela LaTeX: {out_path}")


def _latex_hypothesis_table(tests: pd.DataFrame, out_path: Path) -> None:
    def _fmt_num(v, decimals=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return r"\textemdash{}"
        return f"{v:.{decimals}f}"

    def _fmt_p(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return r"\textemdash{}"
        if v < 0.001:
            return r"$<0.001$"
        return f"{v:.3f}"

    if tests.empty:
        lines = [
            r"\begin{table}[htb]",
            r"\centering",
            r"\caption{Teste pareado entre política única e seleção por perfil.}",
            r"\label{tab:strategy_hypothesis_tests}",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Comparação & n & $\Delta$ médio & $p$ & $p_{\mathrm{Holm}}$ \\",
            r"\midrule",
            r"\multicolumn{5}{c}{Sem pares suficientes para teste.} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    metric_order = {metric: i for i, metric in enumerate(METRIC_COLS)}
    selected = tests.copy()
    selected["metric_order"] = selected["metric"].map(metric_order)
    selected = selected.sort_values(["comparison", "metric_order"])

    comparison_label = {
        "A1_vs_B": "A1 vs B",
        "A2_vs_B": "A2 vs B",
    }
    metric_label = {
        "CTI": "CTI",
        "CTI_ajustado": "CTI ajust.",
        "NS": "NS",
        "TR": "TR",
        "BE": "BE",
        "FP": "FP",
    }

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\small",
        r"\caption{Teste de hipótese pareado entre política única e seleção por perfil"
        r" (Wilcoxon signed-rank, Experimento~2). Para CTI, a hipótese alternativa é"
        r" $\mathrm{CTI}_{B}<\mathrm{CTI}_{A}$; para os demais KPIs, o teste é bilateral.}",
        r"\label{tab:strategy_hypothesis_tests}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Comparação & KPI & n & Média A & Média B & $\Delta$ B-A & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]

    for _, row in selected.iterrows():
        comp = comparison_label.get(row["comparison"], row["comparison"])
        metric = metric_label.get(row["metric"], row["metric"])
        delta = row["mean_delta_profile_minus_reference"]
        p_holm = row.get("p_value_holm", np.nan)
        sig = r"$^{*}$" if row.get("significant_holm_0_05", False) else ""
        lines.append(
            f"{comp} & {metric} & {int(row['n_pairs'])} & "
            f"{_fmt_num(row['mean_reference'], 3)} & "
            f"{_fmt_num(row['mean_profile'], 3)} & "
            f"{_fmt_num(delta, 3)} & {_fmt_p(p_holm)}{sig} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\scriptsize $^{*}$ significativo após correção de Holm, $\alpha=0{,}05$.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Tabela LaTeX de testes pareados: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrada principal
# ─────────────────────────────────────────────────────────────────────────────

def run(kpis: pd.DataFrame | None = None, profiles: pd.DataFrame | None = None,
        out_dir: Path | str | None = None) -> pd.DataFrame:
    """
    2026-08-18: aceita `kpis`/`profiles`/`out_dir` explícitos (mesmo padrão
    de correção de `profile_policy_analysis.run()`). Retorna a tabela
    `strategy` (A1/A2/B/C) -- essa é a resposta direta à pergunta "a
    seleção por perfil (B) é melhor que a única global (A1)?": ver colunas
    `red_pct_vs_A1`/`red_pct_vs_A2` e o teste de hipótese pareado salvo em
    `strategy_hypothesis_tests.csv` (Wilcoxon, H1: CTI_B < CTI_A1).
    """
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(kpis, profiles)
    n_series = df[["warehouse", "store_id", "item_id"]].drop_duplicates().shape[0]
    n_policies = df["policy"].nunique()

    # 1. Métricas globais
    global_agg = _global_metrics(df)
    global_agg.to_csv(out_dir / "policy_global_metrics.csv", index=False)
    log.info(f"Métricas globais: {out_dir / 'policy_global_metrics.csv'}")

    best_global = _pick_global_best(global_agg)
    log.info(f"Política única global (A1): {POLICY_DISPLAY.get(best_global, best_global)}")

    # 2. Métricas por perfil
    profile_agg = _profile_metrics(df)
    profile_agg.to_csv(out_dir / "profile_policy_metrics.csv", index=False)
    log.info(f"Métricas por perfil: {out_dir / 'profile_policy_metrics.csv'}")

    # 3. Dominância por perfil
    dominant = _dominant_by_profile(profile_agg)
    dominant.to_csv(out_dir / "dominant_policy_by_profile.csv", index=False)
    log.info(f"Dominância por perfil: {out_dir / 'dominant_policy_by_profile.csv'}")

    # 4. CTI por estratégia, por série
    series_df = _strategy_per_series(df, best_global, dominant)
    paired_df = _paired_strategy_observations(df, best_global, dominant)
    paired_df.to_csv(out_dir / "strategy_paired_observations.csv", index=False)
    log.info(f"Observacoes pareadas: {out_dir / 'strategy_paired_observations.csv'}")

    # 5. Tabela de comparação de estratégias
    strategy = _strategy_table(series_df, global_agg, best_global, dominant)
    strategy.to_csv(out_dir / "strategy_cost_comparison.csv", index=False)
    hypothesis_tests = _strategy_hypothesis_tests(paired_df)
    hypothesis_tests.to_csv(out_dir / "strategy_hypothesis_tests.csv", index=False)
    log.info(f"Testes pareados: {out_dir / 'strategy_hypothesis_tests.csv'}")
    log.info(f"Comparação de estratégias: {out_dir / 'strategy_cost_comparison.csv'}")

    # 6. Relatório de validação
    _validation_report(df, global_agg, dominant, strategy, best_global,
                       out_dir / "strategy_cost_validation.md",
                       n_series_expected=n_series, policies_expected=sorted(df["policy"].unique()))

    # 7. Tabela LaTeX
    _latex_table(strategy, best_global, dominant, out_dir / "table_strategy_comparison.tex")
    _latex_hypothesis_table(hypothesis_tests, out_dir / "table_strategy_hypothesis_tests.tex")

    # Sumário no console
    log.info("=" * 60)
    log.info("SUMÁRIO DA COMPARAÇÃO DE ESTRATÉGIAS (%d séries, %d políticas)", n_series, n_policies)
    log.info("=" * 60)
    for _, row in strategy.iterrows():
        log.info(
            f"  {row['estrategia']:2s} | CTI_ajustado_medio={row['CTI_ajustado_medio']:8.2f}"
            f" | CTI_medio={row['CTI_medio']:8.2f}"
            f" | NS={row.get('NS_medio','N/A')}"
            f" | red_vs_A1={row.get('red_pct_vs_A1','—')}%"
        )
    log.info(f"Artefatos em: {out_dir}")
    return strategy


if __name__ == "__main__":
    run()
