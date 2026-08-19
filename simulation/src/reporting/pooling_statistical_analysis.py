"""
pooling_statistical_analysis.py — Testes de hipótese pareados para a
pergunta "pooling por perfil operacional ajuda ou atrapalha?" (com_perfil
vs sem_perfil, AJUSTES_INFRA itens #34/#36/#37).

Os experimentos de pooling (`scratchpad/pooling_full_bahia.py`/
`pooling_full_m5.py`) já reportam ganho percentual agregado por política,
mas isso não diz se a diferença é estatisticamente significativa ou só
ruído de uma única execução estocástica (GA/SA/PSO/DE/RL têm
variabilidade própria). Este módulo aplica o MESMO padrão de teste pareado
já usado em `strategy_cost_comparison.py` (Wilcoxon signed-rank + correção
de Holm) à comparação com_perfil vs sem_perfil, série a série, por
política -- e agregado entre todas as políticas.

H0 (por política): median(CTI_ajustado_com_perfil - CTI_ajustado_sem_perfil) = 0
H1: bilateral (não se sabe a priori qual direção vence -- diferente de
`strategy_cost_comparison.py`, onde H1 é direcional porque B é desenhado
pra vencer A1/A2). NS e TIC reportados como diagnóstico complementar,
sempre bilaterais.

Uso:
    python pooling_statistical_analysis.py <caminho_do_csv> [--label NOME]

Ou programaticamente: `run(csv_path, label="Bahia")`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

NS_THRESHOLD = 0.70
PENALTY_WEIGHT = 10.0
EXCESS_WEIGHT = 0.5
METRICS = ["CTI_ajustado", "TIC", "NS"]
# direcao de "melhor" por metrica -- CTI_ajustado/TIC: menor eh melhor;
# NS: maior eh melhor. Usado pra decidir o "vencedor" a partir do sinal de
# mean_delta_com_menos_sem (bug corrigido 2026-08-19: a logica antiga
# tratava delta>0 como "sem_perfil vence" pra QUALQUER metrica, o que
# inverte a resposta certa pra NS -- lah delta>0 significa COM_perfil tem
# NS maior, ou seja, COM_perfil vence).
LOWER_IS_BETTER = {"CTI_ajustado": True, "TIC": True, "NS": False}


def _decide_winner(mean_delta: float, metric: str) -> str:
    lower_better = LOWER_IS_BETTER.get(metric, True)
    com_wins = (mean_delta < 0) if lower_better else (mean_delta > 0)
    return "com_perfil" if com_wins else "sem_perfil"


def _add_adjusted_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Mesma fórmula de `profile_policy_analysis._add_adjusted_cost`/
    `strategy_cost_comparison._add_adjusted_cost`, adaptada à chave de série
    (mode, store_id, item_id) -- ver dashboard.py::get_pooling_results."""
    key_cols = ["mode", "store_id", "item_id"]
    tic_ref = df.groupby(key_cols)["TIC"].transform("max").clip(lower=1.0)
    deficit = (NS_THRESHOLD - df["NS"]).clip(lower=0.0)
    service_loss = deficit * PENALTY_WEIGHT * tic_ref
    if "HoldingCost" in df.columns and df["HoldingCost"].notna().any():
        med_h = df.groupby(key_cols)["HoldingCost"].transform("median")
        excess = (df["HoldingCost"] - med_h).clip(lower=0.0) * EXCESS_WEIGHT
    else:
        excess = 0.0
    df = df.copy()
    df["CTI_ajustado"] = df["TIC"] + service_loss + excess
    return df


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down, mesma implementação de strategy_cost_comparison.py."""
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


def _paired_wide(df: pd.DataFrame, policy: str | None = None) -> pd.DataFrame:
    """
    Uma linha por série (dentro de uma política, ou agrupando todas), com
    colunas `<metrica>_com_perfil`/`<metrica>_sem_perfil` lado a lado.
    """
    sub = df if policy is None else df[df["policy"] == policy]
    piv = sub.pivot_table(index=["warehouse", "store_id", "item_id"],
                          columns="mode", values=METRICS)
    piv.columns = [f"{m}_{mode}" for m, mode in piv.columns]
    return piv.reset_index()


def hypothesis_tests_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Um teste de Wilcoxon pareado por (política, métrica) -- com_perfil
    vs sem_perfil, série a série. Holm aplicado DENTRO de cada métrica,
    entre as políticas (mesma lógica de m testes independentes por família
    de hipótese)."""
    rows = []
    for policy in sorted(df["policy"].unique()):
        wide = _paired_wide(df, policy)
        for metric in METRICS:
            col_com = f"{metric}_com_perfil"
            col_sem = f"{metric}_sem_perfil"
            if col_com not in wide.columns or col_sem not in wide.columns:
                continue
            paired = wide[[col_com, col_sem]].dropna()
            if len(paired) < 5:
                continue
            com = paired[col_com].to_numpy(dtype=float)
            sem = paired[col_sem].to_numpy(dtype=float)
            diff = com - sem
            if np.all(np.abs(diff) <= 1e-12):
                stat, p_value = 0.0, 1.0
            else:
                stat, p_value = stats.wilcoxon(com, sem, alternative="two-sided",
                                               zero_method="wilcox")
            mean_com, mean_sem = float(np.mean(com)), float(np.mean(sem))
            mean_delta = mean_com - mean_sem
            median_delta = float(np.median(diff))
            sd_delta = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
            dz = mean_delta / sd_delta if sd_delta and not np.isnan(sd_delta) else np.nan
            rel_pct = 100.0 * mean_delta / (abs(mean_sem) + 1e-12)
            rows.append({
                "policy": policy, "metric": metric, "n_pairs": int(len(paired)),
                "mean_com_perfil": mean_com, "mean_sem_perfil": mean_sem,
                "mean_delta_com_menos_sem": mean_delta,
                "median_delta_com_menos_sem": median_delta,
                "relative_change_pct": rel_pct,
                "wilcoxon_statistic": float(stat), "p_value": float(p_value),
                "cohens_dz": float(dz) if not np.isnan(dz) else np.nan,
                "vencedor": _decide_winner(mean_delta, metric),
            })
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    tests["p_value_holm"] = tests.groupby("metric")["p_value"].transform(_holm_adjust)
    tests["significant_0_05"] = tests["p_value"] < 0.05
    tests["significant_holm_0_05"] = tests["p_value_holm"] < 0.05
    return tests


def hypothesis_test_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Um teste por métrica, agregando TODAS as (série, política) como
    observações pareadas -- responde "no geral, através do portfólio
    inteiro de políticas, com_perfil difere de sem_perfil?", diferente do
    teste por política (que isola o efeito por família de política)."""
    rows = []
    wide = df.pivot_table(index=["warehouse", "store_id", "item_id", "policy"],
                          columns="mode", values=METRICS)
    wide.columns = [f"{m}_{mode}" for m, mode in wide.columns]
    wide = wide.reset_index()
    for metric in METRICS:
        col_com, col_sem = f"{metric}_com_perfil", f"{metric}_sem_perfil"
        if col_com not in wide.columns or col_sem not in wide.columns:
            continue
        paired = wide[[col_com, col_sem]].dropna()
        com = paired[col_com].to_numpy(dtype=float)
        sem = paired[col_sem].to_numpy(dtype=float)
        diff = com - sem
        stat, p_value = stats.wilcoxon(com, sem, alternative="two-sided", zero_method="wilcox")
        mean_delta = float(np.mean(diff))
        sd_delta = float(np.std(diff, ddof=1))
        rows.append({
            "metric": metric, "n_pairs": int(len(paired)),
            "mean_com_perfil": float(np.mean(com)), "mean_sem_perfil": float(np.mean(sem)),
            "mean_delta_com_menos_sem": mean_delta,
            "median_delta_com_menos_sem": float(np.median(diff)),
            "cohens_dz": mean_delta / sd_delta if sd_delta else np.nan,
            "wilcoxon_statistic": float(stat), "p_value": float(p_value),
            "significant_0_05": p_value < 0.05,
            "vencedor": _decide_winner(mean_delta, metric),
        })
    return pd.DataFrame(rows)


def run(csv_path: str | Path, label: str, out_dir: str | Path | None = None) -> dict:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    log.info("Lendo %s ...", csv_path)
    df = pd.read_csv(csv_path)
    df = _add_adjusted_cost(df)

    by_policy = hypothesis_tests_by_policy(df)
    pooled = hypothesis_test_pooled(df)

    stem = csv_path.stem
    by_policy_path = out_dir / f"{stem}_stat_by_policy.csv"
    pooled_path = out_dir / f"{stem}_stat_pooled.csv"
    by_policy.to_csv(by_policy_path, index=False)
    pooled.to_csv(pooled_path, index=False)

    log.info("=== %s: teste pareado agregado (todas as políticas juntas) ===", label)
    for _, row in pooled.iterrows():
        sig = "***" if row["significant_0_05"] else "n.s."
        log.info("  %-14s vencedor=%-11s delta=%+10.2f p=%.4f %s",
                 row["metric"], row["vencedor"], row["mean_delta_com_menos_sem"],
                 row["p_value"], sig)

    log.info("=== %s: por política (CTI_ajustado, Holm-corrigido) ===", label)
    cti = by_policy[by_policy["metric"] == "CTI_ajustado"].sort_values("p_value_holm")
    for _, row in cti.iterrows():
        sig = "***" if row["significant_holm_0_05"] else ("*" if row["significant_0_05"] else "n.s.")
        log.info("  %-20s vencedor=%-11s delta=%+10.2f p_holm=%.4f %s",
                 row["policy"], row["vencedor"], row["mean_delta_com_menos_sem"],
                 row["p_value_holm"], sig)

    log.info("Salvo: %s / %s", by_policy_path, pooled_path)
    return {"by_policy": by_policy, "pooled": pooled}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    run(args.csv_path, args.label or Path(args.csv_path).stem, args.out_dir)
