"""
profile_policy_analysis.py
Avaliação de políticas de inventário por Perfil Operacional de Demanda (POD).

Entrada:
  data/07_model_output/kpis.parquet          — KPIs por série × política
  data/04_feature/demand_profiles.parquet    — Perfil operacional por série

Saída (em data/08_reporting/profiles/):
  profile_policy_metrics.csv / .parquet      — Métricas por perfil × política
  dominant_policy_by_profile.csv / .parquet  — Política dominante por perfil
  profile_policy_heatmap_cti.pdf             — Heatmap CTI médio
  profile_policy_heatmap_ns.pdf              — Heatmap NS médio
  profile_policy_validation.md               — Relatório de validação

Regra de dominância:
  Políticas viáveis: NS médio >= NS_THRESHOLD (padrão 0.70)
  Política dominante: menor CTI médio entre as viáveis
  Fallback (nenhuma viável): política de maior NS, marcada como *fallback*

Uso:
  python simulation/src/reporting/profile_policy_analysis.py
  ou como módulo: from reporting.profile_policy_analysis import run
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_DIR    = REPO_ROOT / "data"
KPI_PATH    = DATA_DIR / "07_model_output" / "kpis.parquet"
PROF_PATH   = DATA_DIR / "04_feature" / "demand_profiles.parquet"
OUT_DIR     = DATA_DIR / "08_reporting" / "profiles"

NS_THRESHOLD = 0.70

POLICY_ORDER = ["EOQ", "sS", "Newsvendor", "GA", "SA", "PSO", "DE",
                "DQN", "PPO", "SARSA", "GA-DQN", "GA-PPO"]

POLICY_DISPLAY = {
    "sS": "(s,S)",
    "Newsvendor": "Jornaleiro",
}

PROFILE_DISPLAY = {
    "Sparse_High_Impact": "Sparse High Impact",
    "High_Vol_Seasonal":  "High Vol. Seasonal",
    "Unstable_Trend":     "Unstable Trend",
    "Low_Vol_Stable":     "Low Vol. Stable",
    "Fast_Moving":        "Fast Moving",
}

DEGENERATE_POLICIES = {"DQN", "PPO"}  # marcadas como degeneradas no Experimento 2

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "font.size":        9,
})


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Lendo kpis.parquet …")
    kpis = pd.read_parquet(KPI_PATH)
    log.info("Lendo demand_profiles.parquet …")
    profiles = pd.read_parquet(PROF_PATH)
    return kpis, profiles


def _merge(kpis: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["warehouse", "store_id", "item_id"]
    profile_cols = join_cols + ["operational_profile", "adi", "cv2", "mu", "burstiness"]
    missing = [c for c in profile_cols if c not in profiles.columns]
    if missing:
        raise ValueError(f"demand_profiles.parquet sem colunas: {missing}")
    merged = kpis.merge(profiles[profile_cols], on=join_cols, how="left",
                        suffixes=("", "_prof"))
    n_missing = merged["operational_profile"].isna().sum()
    if n_missing:
        log.warning(f"{n_missing} linhas sem operational_profile após join.")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_by_profile(merged: pd.DataFrame) -> pd.DataFrame:
    """Agrega KPIs por (operational_profile, policy)."""
    kpi_cols = [c for c in ["TIC", "NS", "TR", "BE", "FP"] if c in merged.columns]
    agg_fns  = {c: ["mean", "std"] for c in kpi_cols}
    agg_fns["store_id"] = "count"  # para contar n_series (sem duplicata por política)

    grp = merged.groupby(["operational_profile", "policy"])
    agg = grp[kpi_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]

    # n_series: número de séries distintas por perfil (independente de política)
    series_per_profile = (
        merged[["operational_profile", "store_id", "item_id"]]
        .drop_duplicates(subset=["operational_profile", "store_id", "item_id"])
        .groupby("operational_profile")["store_id"].count()
        .rename("n_series")
    )
    agg = agg.reset_index().merge(series_per_profile, on="operational_profile", how="left")
    agg["profile_display"] = agg["operational_profile"].map(
        lambda x: PROFILE_DISPLAY.get(x, x)
    )
    agg["policy_display"] = agg["policy"].map(
        lambda x: POLICY_DISPLAY.get(x, x)
    )
    return agg


def _dominant_policy_per_profile(agg: pd.DataFrame) -> pd.DataFrame:
    """Identifica política dominante por perfil: min CTI entre NS_mean >= NS_THRESHOLD."""
    records = []
    for profile, grp in agg.groupby("operational_profile"):
        n_series = grp["n_series"].iloc[0]
        viable = grp[grp["NS_mean"] >= NS_THRESHOLD].copy()
        fallback = False
        if viable.empty:
            viable = grp.copy()
            fallback = True
            log.warning(f"Perfil '{profile}': nenhuma política atinge NS>={NS_THRESHOLD}. Usando fallback (maior NS).")
            dominant_row = viable.loc[viable["NS_mean"].idxmax()]
        else:
            dominant_row = viable.loc[viable["TIC_mean"].idxmin()]

        note = "fallback: nenhuma política viável" if fallback else ""
        if dominant_row["policy"] in DEGENERATE_POLICIES:
            note = "dominante degenerado; interpretar com cautela"

        records.append({
            "operational_profile":  profile,
            "profile_display":      PROFILE_DISPLAY.get(profile, profile),
            "n_series":             int(n_series),
            "dominant_policy":      dominant_row["policy"],
            "dominant_policy_disp": POLICY_DISPLAY.get(dominant_row["policy"], dominant_row["policy"]),
            "CTI_mean":             round(dominant_row["TIC_mean"], 2),
            "NS_mean":              round(dominant_row["NS_mean"], 3),
            "TR_mean":              round(dominant_row.get("TR_mean", float("nan")), 3),
            "BE_mean":              round(dominant_row.get("BE_mean", float("nan")), 2),
            "fallback":             fallback,
            "note":                 note,
        })
    return pd.DataFrame(records).sort_values("operational_profile")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap(agg: pd.DataFrame, metric: str, title: str,
             out_path: Path, cmap: str = "RdYlGn_r", fmt: str = ".1f") -> None:
    profiles_order = [p for p in PROFILE_DISPLAY.keys() if p in agg["operational_profile"].unique()]
    policies_order = [p for p in POLICY_ORDER if p in agg["policy"].unique()]

    pivot = agg.pivot(index="operational_profile", columns="policy", values=metric)
    pivot = pivot.reindex(index=profiles_order, columns=policies_order)

    row_labels = [PROFILE_DISPLAY.get(p, p) for p in pivot.index]
    col_labels = [POLICY_DISPLAY.get(p, p) for p in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.9),
                                    max(2.5, len(row_labels) * 0.8)))
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10, pad=8)

    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = data[r, c]
            if not np.isnan(val):
                ax.text(c, r, format(val, fmt), ha="center", va="center",
                        fontsize=7, color="black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {out_path}")


def _dominance_barplot(dominant: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(dominant) * 0.7)))
    colors = plt.cm.tab10.colors
    policy_color: dict = {}
    ci = 0
    for pol in dominant["dominant_policy"]:
        if pol not in policy_color:
            policy_color[pol] = colors[ci % len(colors)]
            ci += 1

    bars = ax.barh(
        dominant["profile_display"],
        dominant["CTI_mean"],
        color=[policy_color[p] for p in dominant["dominant_policy"]],
    )
    for bar, row in zip(bars, dominant.itertuples()):
        ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{POLICY_DISPLAY.get(row.dominant_policy, row.dominant_policy)}"
                f"  NS={row.NS_mean:.2f}",
                va="center", fontsize=8, color="white",
                fontweight="bold")

    ax.set_xlabel("CTI médio (R$)")
    ax.set_title("Política dominante por Perfil Operacional\n"
                 f"(NS mínimo viável = {NS_THRESHOLD})", fontsize=10)
    patches = [mpatches.Patch(color=c, label=POLICY_DISPLAY.get(p, p))
               for p, c in policy_color.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation report
# ─────────────────────────────────────────────────────────────────────────────

def _validation_report(kpis: pd.DataFrame, merged: pd.DataFrame,
                        agg: pd.DataFrame, dominant: pd.DataFrame,
                        out_path: Path) -> None:
    n_series  = merged[["store_id", "item_id"]].drop_duplicates().shape[0]
    n_policies = merged["policy"].nunique()
    profiles  = merged["operational_profile"].dropna().unique()

    global_means = kpis.groupby("policy")["TIC"].mean().to_dict()

    lines = [
        f"# Validação — Avaliação por Perfil Operacional",
        f"",
        f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Fonte dos dados",
        f"- KPIs: `{KPI_PATH.relative_to(REPO_ROOT)}`",
        f"- Perfis: `{PROF_PATH.relative_to(REPO_ROOT)}`",
        f"",
        f"## Granularidade",
        f"- Uma linha por (série loja-produto, política) em kpis.parquet",
        f"- Resultados agregados sobre replicações na geração de kpis.parquet",
        f"",
        f"## Cobertura",
        f"- Séries (loja, produto): **{n_series}** (Experimento 2, BA)",
        f"- Políticas avaliadas: **{n_policies}**",
        f"- Perfis operacionais presentes: **{len(profiles)}** de 5 definidos",
        f"",
        f"## Distribuição por perfil",
    ]
    for _, row in dominant.iterrows():
        lines.append(
            f"- **{row['profile_display']}**: {row['n_series']} séries"
            f" | dominante: {row['dominant_policy_disp']}"
            f" | CTI={row['CTI_mean']:.1f} | NS={row['NS_mean']:.2f}"
            + (f" ⚠ {row['note']}" if row["note"] else "")
        )

    lines += [
        f"",
        f"## Regra de dominância",
        f"- Políticas viáveis: NS médio >= {NS_THRESHOLD}",
        f"- Política dominante: menor CTI médio entre viáveis",
        f"- Fallback: maior NS médio quando nenhuma política é viável",
        f"",
        f"## Consistência com Tabela 5.2 (agregado global)",
        f"",
        f"| Política | CTI médio (kpis.parquet) |",
        f"|---|---|",
    ]
    for pol in POLICY_ORDER:
        if pol in global_means:
            lines.append(f"| {POLICY_DISPLAY.get(pol, pol)} | {global_means[pol]:.2f} |")

    lines += [
        f"",
        f"## Limitações",
        f"- Análise concentrada no regime *Lumpy* (Experimentos 1 e 2).",
        f"- Perfis `Low_Vol_Stable` e `Fast_Moving` não têm séries no Experimento 2.",
        f"- Perfis com poucas séries (n < 20) devem ser interpretados de forma exploratória.",
        f"- Generalização para outros regimes é objetivo do Experimento 3.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Relatório de validação salvo: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table snippet
# ─────────────────────────────────────────────────────────────────────────────

def _latex_dominance_table(dominant: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\small",
        r"\caption{Política dominante por Perfil Operacional de Demanda (Experimento~2, BA,"
        r" regime \textit{Lumpy}). Para cada perfil, são reportados o número de séries,"
        r" a política dominante (menor CTI médio entre políticas com NS médio"
        r" $\geq 0{,}70$), o CTI médio e o NS médio da política dominante."
        r" Perfis com $n < 20$ séries devem ser interpretados de forma exploratória.}",
        r"\label{tab:dominancia_por_perfil}",
        r"\begin{tabular}{@{}p{3.2cm}clrrl@{}}",
        r"\toprule",
        r"Perfil & $n$ & Política dominante & CTI médio (R\$) & NS médio & Observação \\",
        r"\midrule",
    ]
    for _, row in dominant.iterrows():
        note = r"\dag" if row["note"] else ""
        n_marker = r"\textsuperscript{*}" if row["n_series"] < 20 else ""
        lines.append(
            f"{row['profile_display']} & "
            f"{row['n_series']}{n_marker} & "
            f"{row['dominant_policy_disp']} & "
            f"{row['CTI_mean']:.2f} & "
            f"{row['NS_mean']:.3f} & "
            f"{note} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\scriptsize{*}$n < 20$: evidência exploratória.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Tabela LaTeX salva: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kpis, profiles = _load_data()
    merged = _merge(kpis, profiles)

    log.info(f"Dados unidos: {len(merged)} linhas, "
             f"{merged[['store_id','item_id']].drop_duplicates().shape[0]} séries, "
             f"{merged['operational_profile'].nunique()} perfis.")

    agg = _aggregate_by_profile(merged)
    dominant = _dominant_policy_per_profile(agg)

    # Export CSVs and parquets
    agg_out = OUT_DIR / "profile_policy_metrics"
    agg.to_csv(str(agg_out) + ".csv", index=False)
    agg.to_parquet(str(agg_out) + ".parquet", index=False)
    log.info(f"Métricas por perfil: {agg_out}.csv / .parquet")

    dom_out = OUT_DIR / "dominant_policy_by_profile"
    dominant.to_csv(str(dom_out) + ".csv", index=False)
    dominant.to_parquet(str(dom_out) + ".parquet", index=False)
    log.info(f"Dominância por perfil: {dom_out}.csv / .parquet")

    # Figures
    _heatmap(agg, "TIC_mean", "CTI médio por Perfil e Política (R$)",
             OUT_DIR / "profile_policy_heatmap_cti.pdf", cmap="RdYlGn_r", fmt=".0f")
    _heatmap(agg, "NS_mean", "NS médio por Perfil e Política",
             OUT_DIR / "profile_policy_heatmap_ns.pdf", cmap="RdYlGn", fmt=".2f")
    _dominance_barplot(dominant, OUT_DIR / "profile_policy_dominance_barplot.pdf")

    # LaTeX snippet
    _latex_dominance_table(dominant, OUT_DIR / "table_dominancia_por_perfil.tex")

    # Validation report
    _validation_report(kpis, merged, agg, dominant,
                       OUT_DIR / "profile_policy_validation.md")

    log.info("Análise por perfil concluída.")
    log.info(f"Artefatos em: {OUT_DIR}")


if __name__ == "__main__":
    run()
