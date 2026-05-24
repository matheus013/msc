"""
statistical_validation/nodes.py — Testes estatísticos para comparação de políticas.

Rigor exigido pela dissertação:
  - Wilcoxon signed-rank: comparação pareada política proposta vs cada baseline
  - Friedman + Nemenyi post-hoc: comparação multi-política global
  - Cohen's d: tamanho de efeito por par de políticas
  - Análise estratificada por grupo de demanda (I/II/III) e estado
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

log = logging.getLogger(__name__)

PROPOSED_POLICIES = ["GA-DQN", "GA-PPO"]
KPI_COLS = ["TIC", "NS", "TR", "BE", "FP"]


def run_wilcoxon_tests(kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Wilcoxon signed-rank entre cada política proposta e cada baseline.

    Comparação pareada por (warehouse, store_id, item_id): cada par tem uma
    observação por loja, permitindo teste não-paramétrico pareado.

    Returns:
        DataFrame [policy_a, policy_b, metric, statistic, p_value, significant]
    """
    rows = []
    all_policies = kpis["policy"].unique().tolist()
    baselines = [p for p in all_policies if p not in PROPOSED_POLICIES]
    series_key = ["warehouse", "store_id", "item_id"]

    for proposed in PROPOSED_POLICIES:
        if proposed not in all_policies:
            continue
        df_prop = kpis[kpis["policy"] == proposed].set_index(series_key)

        for baseline in baselines:
            if baseline == proposed:
                continue
            df_base = kpis[kpis["policy"] == baseline].set_index(series_key)
            common_idx = df_prop.index.intersection(df_base.index)
            if len(common_idx) < 5:
                log.warning("Poucas observações para Wilcoxon: %s vs %s (%d)",
                            proposed, baseline, len(common_idx))
                continue

            for metric in KPI_COLS:
                if metric not in df_prop.columns:
                    continue
                a = df_prop.loc[common_idx, metric].values
                b = df_base.loc[common_idx, metric].values
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() < 5:
                    continue
                try:
                    stat, p = stats.wilcoxon(a[mask], b[mask], alternative="two-sided")
                    rows.append({
                        "policy_a": proposed, "policy_b": baseline,
                        "metric": metric,
                        "statistic": float(stat), "p_value": float(p),
                        "n_pairs": int(mask.sum()),
                        "significant": bool(p < 0.05),
                    })
                except Exception as e:
                    log.warning("Wilcoxon falhou: %s vs %s / %s: %s",
                                proposed, baseline, metric, e)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        n_tests = len(df_out)
        bonferroni_alpha = 0.05 / n_tests
        df_out["significant_bonferroni"] = df_out["p_value"] < bonferroni_alpha
        df_out["bonferroni_alpha"] = bonferroni_alpha
        log.info(
            "Wilcoxon: %d testes | %d sig. (p<0.05) | %d sig. Bonferroni (α=%.4f)",
            n_tests,
            df_out["significant"].sum(),
            df_out["significant_bonferroni"].sum(),
            bonferroni_alpha,
        )
    return df_out


def run_friedman_nemenyi(kpis: pd.DataFrame) -> dict:
    """
    Friedman test (multi-política) + Nemenyi post-hoc.

    Returns:
        dict com keys: friedman_stat, p_value, ranks, cd (critical difference),
                       posthoc_matrix (DataFrame)
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        log.warning("scikit-posthocs não instalado — Nemenyi pulado")
        sp = None

    results = {}
    series_key = ["warehouse", "store_id", "item_id"]

    for metric in KPI_COLS:
        if metric not in kpis.columns:
            continue
        pivot = kpis.pivot_table(index=series_key, columns="policy",
                                 values=metric, aggfunc="mean")
        pivot = pivot.dropna()
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            continue

        data_groups = [pivot[col].values for col in pivot.columns]
        try:
            stat, p = stats.friedmanchisquare(*data_groups)
        except Exception as e:
            log.warning("Friedman falhou para %s: %s", metric, e)
            continue

        # Rankings médios
        ranks = pivot.rank(axis=1).mean(axis=0).to_dict()

        posthoc = None
        if sp is not None and p < 0.05:
            try:
                posthoc = sp.posthoc_nemenyi_friedman(pivot.values)
                posthoc.index = pivot.columns
                posthoc.columns = pivot.columns
            except Exception as e:
                log.warning("Nemenyi falhou para %s: %s", metric, e)

        # Critical difference (alpha=0.05, Demsar 2006)
        k = pivot.shape[1]
        n = pivot.shape[0]
        cd = _critical_difference(k, n, alpha=0.05)

        results[metric] = {
            "friedman_stat": float(stat),
            "p_value": float(p),
            "ranks": ranks,
            "cd": cd,
            "posthoc_matrix": posthoc,
            "n_series": n,
            "n_policies": k,
        }
        log.info("Friedman [%s]: stat=%.3f p=%.4f | ranks: %s",
                 metric, stat, p,
                 {k: f"{v:.2f}" for k, v in sorted(ranks.items(), key=lambda x: x[1])})

    return results


def _critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Diferença crítica de Nemenyi (aproximação de Demsar 2006).
    CD = q_alpha * sqrt(k(k+1) / 6n)
    q_alpha para alpha=0.05: tabela para k políticas.
    """
    # Valores de q_alpha (Studentized Range / sqrt(2)) para alpha=0.05
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 12: 3.268}
    q = q_table.get(k, q_table.get(min(q_table.keys(), key=lambda x: abs(x - k))))
    return q * np.sqrt(k * (k + 1) / (6 * n))


def compute_effect_sizes(kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Cohen's d por par (política proposta, baseline) × métrica.

    d = (μ_a - μ_b) / σ_pooled
    |d|<0.2 trivial, 0.2-0.5 pequeno, 0.5-0.8 médio, >0.8 grande.
    """
    rows = []
    all_policies = kpis["policy"].unique().tolist()
    baselines = [p for p in all_policies if p not in PROPOSED_POLICIES]
    series_key = ["warehouse", "store_id", "item_id"]

    for proposed in PROPOSED_POLICIES:
        if proposed not in all_policies:
            continue
        df_prop = kpis[kpis["policy"] == proposed].set_index(series_key)

        for baseline in baselines:
            if baseline == proposed:
                continue
            df_base = kpis[kpis["policy"] == baseline].set_index(series_key)
            common_idx = df_prop.index.intersection(df_base.index)
            if len(common_idx) < 2:
                continue

            for metric in KPI_COLS:
                if metric not in df_prop.columns:
                    continue
                a = df_prop.loc[common_idx, metric].dropna().values
                b = df_base.loc[common_idx, metric].dropna().values
                n_common = min(len(a), len(b))
                if n_common < 2:
                    continue
                a, b = a[:n_common], b[:n_common]
                pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
                d = (np.mean(a) - np.mean(b)) / (pooled_std + 1e-9)
                magnitude = (
                    "trivial" if abs(d) < 0.2 else
                    "small"   if abs(d) < 0.5 else
                    "medium"  if abs(d) < 0.8 else "large"
                )
                rows.append({
                    "policy_a": proposed, "policy_b": baseline,
                    "metric": metric,
                    "cohens_d": float(d),
                    "magnitude": magnitude,
                    "n": n_common,
                })

    return pd.DataFrame(rows)


def stratified_analysis(kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Médias de KPIs estratificadas por múltiplas dimensões:
      - estado (warehouse) × grupo de demanda (I/II/III) × política
      - segmento (Bronze/Prata/Ouro/…) × política        — se disponível
      - filial × política                                  — se disponível

    Returns:
        DataFrame com colunas de agrupamento + KPI_mean/std + n_series
    """
    available_metrics = [c for c in KPI_COLS if c in kpis.columns]

    if "group" not in kpis.columns:
        kpis = kpis.copy()
        kpis["group"] = "?"

    # Dimensões de estratificação: obrigatórias + opcionais presentes
    base_dims = ["warehouse", "group", "policy"]
    optional_dims = [c for c in ["segmento", "filial", "gerente_regional"]
                     if c in kpis.columns and kpis[c].notna().any()]
    all_dims = base_dims + optional_dims

    frames = []
    # 1. Estratificação base (sempre)
    frames.append(_aggregate_kpis(kpis, base_dims, available_metrics, "base"))
    # 2. Com segmento (se presente)
    if "segmento" in optional_dims:
        dims = ["warehouse", "segmento", "policy"]
        frames.append(_aggregate_kpis(kpis, dims, available_metrics, "segmento"))
    # 3. Com filial (se presente)
    if "filial" in optional_dims:
        dims = ["warehouse", "filial", "policy"]
        frames.append(_aggregate_kpis(kpis, dims, available_metrics, "filial"))

    summary = pd.concat(frames, ignore_index=True)
    log.info("Análise estratificada: %d combinações (%s)",
             len(summary), ", ".join([f"level={f}" for f in ["base"] + optional_dims[:2]]))
    return summary


def _aggregate_kpis(kpis: pd.DataFrame, group_cols: list,
                     metrics: list, level: str) -> pd.DataFrame:
    """Agrega KPIs por group_cols, adiciona n_series e coluna 'level'."""
    sub = kpis.copy()
    # Garante que todas as colunas de agrupamento existem
    for c in group_cols:
        if c not in sub.columns:
            sub[c] = "?"

    summary = (sub.groupby(group_cols)[metrics]
               .agg(["mean", "std"])
               .reset_index())
    summary.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in summary.columns
    ]
    n_series = sub.groupby(group_cols).apply(
        lambda df: df.groupby(["store_id", "item_id"]).ngroups
    ).reset_index(name="n_series")
    summary = summary.merge(n_series, on=group_cols, how="left")
    summary["level"] = level
    return summary
