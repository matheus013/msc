"""
visualizations.py — Gráficos para relatórios e dissertação de mestrado.

Categorias:
  A. Comparação de políticas
  B. Fronteira de Pareto e sensibilidade
  C. Validação estatística
  D. Qualidade do forecast
  E. Caracterização da demanda
  F. Mapas geográficos do Brasil
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path

log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
})

POLICY_ORDER = ["EOQ", "sS", "Newsvendor", "GA", "SA", "PSO", "DE",
                "DQN", "PPO", "SARSA", "GA-DQN", "GA-PPO"]

POLICY_DISPLAY = {
    "sS": "(s,S)",
    "Newsvendor": "Jornaleiro",
}

POLICY_CATEGORIES = {
    "classical":     ("EOQ", "sS", "Newsvendor"),
    "metaheuristic": ("GA", "SA", "PSO", "DE"),
    "rl":            ("DQN", "PPO", "SARSA"),
    "hybrid":        ("GA-DQN", "GA-PPO"),
}

_CAT_COLORS = {
    "classical": "#1565C0",
    "metaheuristic": "#2E7D32",
    "rl": "#F57C00",
    "hybrid": "#C62828",
}

KPI_META = {
    "TIC": ("Custo Total (R$)", False),
    "NS":  ("Nível de Serviço",  True),
    "TR":  ("Taxa de Ruptura",   False),
    "BE":  ("Efeito Bullwhip",   False),
    "FP":  ("Freq. Pedidos",     True),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig, name, out_dir, fmt="pdf", dpi=150):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    log.info("[Plot] %s", p)
    return p


def _policy_colors():
    colors = {}
    for cat, pols in POLICY_CATEGORIES.items():
        for p in pols:
            colors[p] = _CAT_COLORS[cat]
    return colors


def _ordered_policies(kpis_df):
    present = set(kpis_df["policy"].unique())
    return [p for p in POLICY_ORDER if p in present]


def _display_name(policy):
    return POLICY_DISPLAY.get(policy, policy)


def _category_legend_handles():
    return [
        mpatches.Patch(color="#1565C0", label="Clássica"),
        mpatches.Patch(color="#2E7D32", label="Meta-heurística"),
        mpatches.Patch(color="#F57C00", label="RL"),
        mpatches.Patch(color="#C62828", label="Híbrida (proposta)"),
    ]


# ── A. COMPARAÇÃO DE POLÍTICAS ────────────────────────────────────────────────

def plot_comparison_bars(kpis_df, out_dir, params):
    """5-panel bar chart — one panel per KPI, all policies."""
    policies = _ordered_policies(kpis_df)
    avail_kpis = [k for k in KPI_META if k in kpis_df.columns]
    summary = kpis_df.groupby("policy")[avail_kpis].mean()
    pcolors = _policy_colors()

    fig, axes = plt.subplots(1, len(avail_kpis), figsize=(22, 5))
    for ax, col in zip(axes, avail_kpis):
        label, _ = KPI_META[col]
        vals = summary.reindex(policies)[col].fillna(0).values
        colors = [pcolors.get(p, "#888") for p in policies]
        ax.bar(range(len(policies)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policies, rotation=40, ha="right", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel(col)

    axes[0].legend(handles=_category_legend_handles(), fontsize=7, loc="upper right")
    fig.suptitle("Comparativo de KPIs — Todas as Políticas", fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "comparison_bars", out_dir, fmt, params.get("figure_dpi", 150))


def plot_radar_kpis(kpis_df, out_dir, params):
    """
    Spider/radar chart: every policy as a polygon over 5 normalized KPI axes.
    Each axis scaled to [0, 1] where 1 = best performance.
    """
    kpis = [k for k in KPI_META if k in kpis_df.columns]
    if not kpis:
        return None
    policies = _ordered_policies(kpis_df)
    summary = kpis_df.groupby("policy")[kpis].mean()

    normed = summary.copy()
    for col, (_, higher_better) in KPI_META.items():
        if col not in normed.columns:
            continue
        mn, mx = normed[col].min(), normed[col].max()
        rng = mx - mn if mx != mn else 1.0
        normed[col] = (normed[col] - mn) / rng if higher_better else (mx - normed[col]) / rng

    N = len(kpis)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    kpi_labels = [KPI_META[k][0] for k in kpis]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(kpi_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)

    pcolors = _policy_colors()
    for pol in policies:
        if pol not in normed.index:
            continue
        vals = [normed.loc[pol, k] if k in normed.columns else 0 for k in kpis] + \
               [normed.loc[pol, kpis[0]] if kpis[0] in normed.columns else 0]
        color = pcolors.get(pol, "#888")
        ax.plot(angles, vals, "o-", lw=1.5, color=color, label=pol, alpha=0.85)
        ax.fill(angles, vals, alpha=0.05, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8, ncol=2)
    ax.set_title("Perfil de Desempenho (KPIs Normalizados)", y=1.1, fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "radar_kpis", out_dir, fmt, params.get("figure_dpi", 150))


def plot_kpi_heatmap_all(kpis_df, out_dir, params):
    """Policy × KPI heatmap — color = normalized performance, cell text = raw mean."""
    kpis = [k for k in KPI_META if k in kpis_df.columns]
    policies = _ordered_policies(kpis_df)
    summary = kpis_df.groupby("policy")[kpis].mean().reindex(policies)

    normed = summary.copy()
    for col, (_, higher_better) in KPI_META.items():
        if col not in normed.columns:
            continue
        mn, mx = normed[col].min(), normed[col].max()
        rng = mx - mn if mx != mn else 1.0
        normed[col] = (normed[col] - mn) / rng if higher_better else (mx - normed[col]) / rng

    fig, ax = plt.subplots(figsize=(max(8, len(kpis) * 1.6), max(6, len(policies) * 0.55 + 2)))
    im = ax.imshow(normed.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(kpis)))
    ax.set_xticklabels([KPI_META[k][0] for k in kpis], fontsize=9)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies, fontsize=9)

    for r, pol in enumerate(policies):
        for c, kpi in enumerate(kpis):
            if pol in summary.index and not pd.isna(summary.loc[pol, kpi]):
                v = summary.loc[pol, kpi]
                nv = normed.loc[pol, kpi]
                tc = "white" if nv < 0.25 or nv > 0.75 else "black"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8, color=tc)

    plt.colorbar(im, ax=ax, label="Desempenho normalizado (verde = melhor)", shrink=0.6)
    ax.set_title("Heatmap de Desempenho: Políticas × KPIs", fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "kpi_heatmap_all", out_dir, fmt, params.get("figure_dpi", 150))


def plot_heatmap_service_level(kpis_df, out_dir, params):
    """Store × policy heatmap — cell = mean service level."""
    pivot = kpis_df.pivot_table(index="store_id", columns="policy", values="NS", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2),
                                    max(6, len(pivot) * 0.5 + 2)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    plt.colorbar(im, ax=ax, label="Nível de Serviço (NS)")
    ax.set_title("Heatmap: Nível de Serviço por Loja e Política", fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "heatmap_service_level", out_dir, fmt, params.get("figure_dpi", 150))


def plot_boxplot_tic(kpis_df, out_dir, params):
    """Boxplot of TIC distribution per policy."""
    policies = _ordered_policies(kpis_df)
    pcolors = _policy_colors()
    data = [kpis_df[kpis_df["policy"] == p]["TIC"].dropna().values for p in policies]
    fig, ax = plt.subplots(figsize=(max(10, len(policies) * 0.9), 5))
    display_labels = [_display_name(p) for p in policies]
    bp = ax.boxplot(data, labels=display_labels, patch_artist=True,
                    medianprops={"color": "black", "lw": 1.5})
    for patch, pol in zip(bp["boxes"], policies):
        patch.set_facecolor(pcolors.get(pol, "#888"))
        patch.set_alpha(0.7)
    ax.set_title("Distribuição do CTI por Política (todas as lojas × replicações)")
    ax.set_ylabel("CTI (R$)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(handles=_category_legend_handles(), fontsize=7)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "boxplot_tic", out_dir, fmt, params.get("figure_dpi", 150))


def plot_violin_bullwhip(kpis_df, out_dir, params):
    """Violin plot of Bullwhip Effect per policy."""
    policies = _ordered_policies(kpis_df)
    pcolors = _policy_colors()
    data = [kpis_df[kpis_df["policy"] == p]["BE"].dropna().values for p in policies]
    data = [d if len(d) > 1 else np.concatenate([d, d]) for d in data]
    fig, ax = plt.subplots(figsize=(max(10, len(policies) * 0.9), 5))
    parts = ax.violinplot(data, positions=range(len(policies)), showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(pcolors.get(policies[i], "#888"))
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=30, fontsize=8)
    ax.set_title("Distribuição do Efeito Bullwhip por Política")
    ax.set_ylabel("Bullwhip Effect")
    ax.legend(handles=_category_legend_handles(), fontsize=7)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "violin_bullwhip", out_dir, fmt, params.get("figure_dpi", 150))


def plot_tradeoff_scatter(kpis_df, out_dir, params):
    """TIC vs NS scatter — trade-off view with policy labels."""
    summary = kpis_df.groupby("policy").agg(TIC=("TIC", "mean"), NS=("NS", "mean")).reset_index()
    pcolors = _policy_colors()
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        color = pcolors.get(row["policy"], "#888")
        ax.scatter(row["TIC"], row["NS"], color=color, s=120, zorder=3)
        ax.annotate(row["policy"], (row["TIC"], row["NS"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xlabel("CTI médio (R$)")
    ax.set_ylabel("NS médio")
    ax.set_title("Trade-off Custo × Nível de Serviço por Política")
    ax.legend(handles=_category_legend_handles(), fontsize=8)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "tradeoff_scatter", out_dir, fmt, params.get("figure_dpi", 150))


# ── B. FRONTEIRA DE PARETO E SENSIBILIDADE ────────────────────────────────────

def plot_pareto_frontier(kpis_df, out_dir, params):
    """
    TIC vs NS scatter highlighting Pareto-optimal (non-dominated) policies.
    A policy is non-dominated if no other policy has both lower TIC and higher NS.
    """
    summary = kpis_df.groupby("policy").agg(
        TIC=("TIC", "mean"), NS=("NS", "mean")
    ).reset_index()
    pcolors = _policy_colors()

    def dominated(row, others):
        return any(
            (o["TIC"] <= row["TIC"] and o["NS"] >= row["NS"] and
             (o["TIC"] < row["TIC"] or o["NS"] > row["NS"]))
            for _, o in others.iterrows()
        )

    summary["dominated"] = [dominated(r, summary.drop(i)) for i, r in summary.iterrows()]
    pareto = summary[~summary["dominated"]].sort_values("TIC")

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in summary[summary["dominated"]].iterrows():
        ax.scatter(row["TIC"], row["NS"], color="lightgray", s=90, zorder=2)
        ax.annotate(_display_name(row["policy"]), (row["TIC"], row["NS"]),
                    textcoords="offset points", xytext=(5, 2), fontsize=8, color="gray")

    for _, row in pareto.iterrows():
        color = pcolors.get(row["policy"], "#C62828")
        ax.scatter(row["TIC"], row["NS"], color=color, s=160, zorder=4,
                   edgecolors="black", lw=0.8)
        ax.annotate(_display_name(row["policy"]), (row["TIC"], row["NS"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=9, fontweight="bold")

    if len(pareto) > 1:
        ax.plot(pareto["TIC"].values, pareto["NS"].values,
                "--", color="#333", lw=1.2, alpha=0.6, zorder=3)

    handles = [
        mpatches.Patch(color="lightgray", label="Dominado"),
        mpatches.Patch(color="#C62828", label="Não-dominado (Pareto)"),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel("CTI médio (R$)")
    ax.set_ylabel("NS médio")
    ax.set_title("Fronteira de Pareto: Custo × Nível de Serviço", fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "pareto_frontier", out_dir, fmt, params.get("figure_dpi", 150))


def plot_policy_sensitivity(kpis_df, out_dir, params):
    """
    Tornado chart: TIC range per policy across demand groups (I → III).
    Sorted by range width — shows which policies are most sensitive to demand type.
    """
    if "group" not in kpis_df.columns:
        return None
    if kpis_df["group"].nunique() < 2:
        return None

    policies = _ordered_policies(kpis_df)
    summary = kpis_df.groupby(["policy", "group"])["TIC"].mean().unstack()
    if summary.empty:
        return None

    baseline = kpis_df.groupby("policy")["TIC"].mean()
    width = (summary.max(axis=1) - summary.min(axis=1)).fillna(0)
    order = width.sort_values(ascending=True).index
    summary = summary.reindex(order)
    pcolors = _policy_colors()

    fig, ax = plt.subplots(figsize=(10, max(5, len(order) * 0.55 + 2)))
    for i, pol in enumerate(order):
        lo = summary.loc[pol].min()
        hi = summary.loc[pol].max()
        b = baseline.get(pol, (lo + hi) / 2)
        color = pcolors.get(pol, "#888")
        ax.barh(i, lo - b, left=b, height=0.5, color=color, alpha=0.4)
        ax.barh(i, hi - b, left=b, height=0.5, color=color, alpha=0.85)

    ax.axvline(baseline.mean(), color="gray", lw=1, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(order), fontsize=9)
    ax.set_xlabel("TIC (R$) — desvio da media por grupo de demanda")
    ax.set_title("Sensibilidade ao Grupo de Demanda (Syntetos-Boylan)\n"
                 "Barras mostram variacao do TIC entre grupos (ordenado por amplitude)", fontsize=10)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "policy_sensitivity", out_dir, fmt, params.get("figure_dpi", 150))


# ── C. VALIDAÇÃO ESTATÍSTICA ──────────────────────────────────────────────────

def plot_critical_difference(friedman_results, out_dir, params):
    """Critical difference diagram from Nemenyi post-hoc ranks."""
    try:
        ranks = friedman_results.get("ranks")
        cd = friedman_results.get("cd")
        if ranks is None:
            return None

        policies = list(ranks.keys())
        avg_ranks = [ranks[p] for p in policies]
        order = np.argsort(avg_ranks)
        sorted_pols = [policies[i] for i in order]
        sorted_ranks = [avg_ranks[i] for i in order]

        pcolors = _policy_colors()
        fig, ax = plt.subplots(figsize=(10, max(3, len(policies) * 0.45 + 2)))
        ax.set_xlim(0.5, len(policies) + 0.5)
        ax.set_ylim(0, 1)
        ax.axis("off")

        y = 0.7
        for pol, rank in zip(sorted_pols, sorted_ranks):
            ax.plot(rank, y, "o", color=pcolors.get(pol, "#888"), ms=10, zorder=3)
            ax.text(rank, y + 0.1, pol, ha="center", va="bottom", fontsize=8, rotation=45)

        ax.axhline(y, color="black", lw=1, alpha=0.4)

        if cd is not None and sorted_ranks:
            r0 = sorted_ranks[0]
            ax.annotate("", xy=(r0 + cd, y - 0.15), xytext=(r0, y - 0.15),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
            ax.text(r0 + cd / 2, y - 0.23, f"CD = {cd:.2f}",
                    ha="center", va="top", fontsize=8)
            title = f"Diagrama de Diferença Crítica (Nemenyi) | CD = {cd:.3f}"
        else:
            title = "Diagrama de Diferença Crítica (Nemenyi)"

        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fmt = params.get("figure_format", "pdf")
        return _save(fig, "critical_difference", out_dir, fmt, params.get("figure_dpi", 150))
    except Exception as e:
        log.warning("critical_difference skipped: %s", e)
        return None


def plot_wilcoxon_pvalue_heatmap(wilcoxon_results, out_dir, params):
    """
    Pairwise Wilcoxon p-value matrix — one subplot per KPI.
    Cells with p < 0.05 marked with *.
    """
    if wilcoxon_results is None or wilcoxon_results.empty:
        return None
    metrics = [m for m in wilcoxon_results["metric"].unique()] if "metric" in wilcoxon_results.columns else []
    if not metrics:
        return None

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(max(6, n * 5), 7))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = wilcoxon_results[wilcoxon_results["metric"] == metric]
        all_pols = sorted(set(sub["policy_a"].tolist() + sub["policy_b"].tolist()))
        mat = np.full((len(all_pols), len(all_pols)), np.nan)
        for _, row in sub.iterrows():
            if row["policy_a"] in all_pols and row["policy_b"] in all_pols:
                r, c = all_pols.index(row["policy_a"]), all_pols.index(row["policy_b"])
                mat[r, c] = mat[c, r] = row["p_value"]

        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")
        ax.set_xticks(range(len(all_pols)))
        ax.set_xticklabels(all_pols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_pols)))
        ax.set_yticklabels(all_pols, fontsize=8)
        ax.set_title(f"p-valores Wilcoxon\n({metric})", fontsize=9)
        for r in range(len(all_pols)):
            for c in range(len(all_pols)):
                if not np.isnan(mat[r, c]):
                    sig = "*" if mat[r, c] < 0.05 else ""
                    ax.text(c, r, f"{mat[r,c]:.3f}{sig}",
                            ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7, label="p-valor")

    fig.suptitle("Heatmap de p-valores (Wilcoxon) — * indica p < 0.05", fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "wilcoxon_pvalue_heatmap", out_dir, fmt, params.get("figure_dpi", 150))


def plot_cohens_d_heatmap(effect_sizes, out_dir, params):
    """
    Pairwise Cohen's d heatmap per KPI.
    G = grande (≥0.8), M = médio (≥0.5), P = pequeno (<0.5).
    """
    if effect_sizes is None or effect_sizes.empty:
        return None
    metrics = list(effect_sizes["metric"].unique()) if "metric" in effect_sizes.columns else []
    if not metrics:
        return None

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(max(6, n * 5), 7))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = effect_sizes[effect_sizes["metric"] == metric]
        all_pols = sorted(set(sub["policy_a"].tolist() + sub["policy_b"].tolist()))
        mat = np.full((len(all_pols), len(all_pols)), np.nan)
        for _, row in sub.iterrows():
            if row["policy_a"] in all_pols and row["policy_b"] in all_pols:
                r, c = all_pols.index(row["policy_a"]), all_pols.index(row["policy_b"])
                mat[r, c] = mat[c, r] = abs(row["cohens_d"])

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(all_pols)))
        ax.set_xticklabels(all_pols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_pols)))
        ax.set_yticklabels(all_pols, fontsize=8)
        ax.set_title(f"Cohen's d\n({metric})", fontsize=9)
        for r in range(len(all_pols)):
            for c in range(len(all_pols)):
                if not np.isnan(mat[r, c]):
                    v = mat[r, c]
                    mag = "G" if v >= 0.8 else ("M" if v >= 0.5 else "P")
                    ax.text(c, r, f"{v:.2f} ({mag})",
                            ha="center", va="center", fontsize=6.5)
        plt.colorbar(im, ax=ax, shrink=0.7, label="|Cohen's d|")

    fig.suptitle("Tamanho de Efeito (Cohen's d) — G=grande, M=médio, P=pequeno", fontsize=10)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "cohens_d_heatmap", out_dir, fmt, params.get("figure_dpi", 150))


# ── D. QUALIDADE DO FORECAST ──────────────────────────────────────────────────

def plot_forecast_vs_actual(forecast_predictions, out_dir, params):
    """
    Um arquivo por série (warehouse, store_id, item_id).
    Salvo em out_dir/forecast_vs_actual/<warehouse>/<storeid>_<itemid>.<fmt>

    Cada figura tem uma coluna por modelo ML. Tanto o dado real quanto as
    previsões são exibidos como linhas com marcadores nos valores observados.
    """
    if forecast_predictions is None or forecast_predictions.empty:
        return None

    key_cols     = ["warehouse", "store_id", "item_id"]
    model_colors = {"LSTM": "#1565C0", "ANN": "#2E7D32", "XGBoost": "#F57C00"}
    actual_color = "#424242"
    fmt = params.get("figure_format", "pdf")
    dpi = params.get("figure_dpi", 120)

    series_dir = os.path.join(out_dir, "forecast_vs_actual")
    os.makedirs(series_dir, exist_ok=True)

    keys = (
        forecast_predictions[key_cols]
        .drop_duplicates()
        .sort_values(key_cols)
        .values.tolist()
    )

    saved = {}
    for wh, sid, iid in keys:
        sub = forecast_predictions[
            (forecast_predictions["warehouse"] == wh) &
            (forecast_predictions["store_id"]  == sid) &
            (forecast_predictions["item_id"]   == iid)
        ].sort_values("cycle")

        models = sorted(sub["model"].unique())
        if not models:
            continue

        cycles  = sorted(sub["cycle"].unique())
        x       = np.arange(len(cycles))
        cyc_str = [str(c) for c in cycles]

        fig, axes = plt.subplots(
            1, len(models),
            figsize=(len(models) * 5, 4),
            sharey=True,
        )
        if len(models) == 1:
            axes = [axes]

        # Dado real — igual para todos os painéis, extraído do primeiro modelo
        actual_ref = sub[sub["model"] == models[0]].sort_values("cycle")["actual"].values

        for ax, model in zip(axes, models):
            msub      = sub[sub["model"] == model].sort_values("cycle")
            predicted = msub["predicted"].values
            n         = min(len(actual_ref), len(predicted))
            cyc_x     = x[:n]
            color     = model_colors.get(model, "#C62828")
            mae       = float(np.mean(np.abs(actual_ref[:n] - predicted[:n])))
            mase_vals = msub["MASE"].values if "MASE" in msub.columns else None

            # Dado real: linha preta + círculos preenchidos
            ax.plot(cyc_x, actual_ref[:n], "o-",
                    color=actual_color, lw=1.5, ms=4,
                    label="Real", zorder=3)

            # Previsão: linha colorida + diamantes
            ax.plot(cyc_x, predicted[:n], "D--",
                    color=color, lw=1.5, ms=3, alpha=0.85,
                    label=model, zorder=2)

            # Banda de incerteza ±MAE
            ax.fill_between(cyc_x,
                            predicted[:n] - mae,
                            predicted[:n] + mae,
                            alpha=0.12, color=color)

            title = f"{model}   MAE={mae:.1f}"
            if mase_vals is not None and len(mase_vals) > 0:
                title += f"   MASE={np.nanmean(mase_vals):.2f}"
            ax.set_title(title, fontsize=8)

            step = max(1, n // 7)
            ax.set_xticks(cyc_x[::step])
            ax.set_xticklabels(
                [cyc_str[i] for i in range(0, n, step)],
                rotation=45, ha="right", fontsize=6,
            )
            ax.legend(fontsize=6, loc="upper left")
            ax.set_ylabel("Demanda", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(axis="y", lw=0.4, alpha=0.4)

        fig.suptitle(f"{wh} | loja {sid} | item {iid}", fontsize=9)
        fig.tight_layout()

        # Salva em subpasta por warehouse
        wh_dir = os.path.join(series_dir, str(wh))
        os.makedirs(wh_dir, exist_ok=True)
        fname = f"{sid}__{iid}.{fmt}"
        fpath = os.path.join(wh_dir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved[f"forecast_vs_actual__{wh}__{sid}__{iid}"] = fpath

    log.info("[Plot] forecast_vs_actual: %d arquivos em %s", len(saved), series_dir)
    return saved


def plot_forecast_residuals(forecast_predictions, out_dir, params):
    """
    Residual analysis per model: scatter (predicted vs actual) + error histogram.
    """
    if forecast_predictions is None or forecast_predictions.empty:
        return None

    models = sorted(forecast_predictions["model"].unique())
    model_colors = {"LSTM": "#1565C0", "ANN": "#2E7D32", "XGBoost": "#F57C00"}

    fig, axes = plt.subplots(2, len(models),
                              figsize=(max(10, len(models) * 4), 8))
    if len(models) == 1:
        axes = axes.reshape(2, 1)

    for j, model in enumerate(models):
        msub = forecast_predictions[forecast_predictions["model"] == model]
        actual = msub["actual"].values
        predicted = msub["predicted"].values
        errors = predicted - actual
        color = model_colors.get(model, "#888")

        ax_sc = axes[0, j]
        lim = max(actual.max(), predicted.max()) * 1.05
        ax_sc.scatter(actual, predicted, alpha=0.35, s=12, color=color)
        ax_sc.plot([0, lim], [0, lim], "k--", lw=1, label="Ideal")
        ax_sc.set_xlabel("Real")
        ax_sc.set_ylabel("Previsto")
        ax_sc.set_title(f"{model}  (n={len(actual):,})", fontsize=9)
        ax_sc.legend(fontsize=7)

        ax_h = axes[1, j]
        ax_h.hist(errors, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax_h.axvline(0, color="black", lw=1, linestyle="--")
        ax_h.axvline(errors.mean(), color="red", lw=1.5,
                     label=f"Média={errors.mean():.1f}")
        ax_h.set_xlabel("Erro (Previsto − Real)")
        ax_h.set_ylabel("Frequência")
        ax_h.legend(fontsize=7)

    fig.suptitle("Análise de Resíduos por Modelo de Forecast", fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "forecast_residuals", out_dir, fmt, params.get("figure_dpi", 150))


# ── E. CARACTERIZAÇÃO DA DEMANDA ──────────────────────────────────────────────

def plot_adi_cv_scatter(scenarios_meta, out_dir, params):
    """
    Croston ADI × CV² scatter — classifies demand into 4 quadrants:
    Suave | Errática | Intermitente | Grumosa (Lumpy).
    Thresholds: ADI = 1.32, CV² = 0.49 (Syntetos et al. 2005).
    """
    df = scenarios_meta.copy()
    df["adi"] = df["n_periods"] / df["n_positive"].replace(0, 1)
    df["cv2"] = df["cv"] ** 2

    ADI_T, CV2_T = 1.32, 0.49
    group_colors = {
        "Smooth":       "#1565C0",
        "Erratic":      "#F57C00",
        "Intermittent": "#2E7D32",
        "Lumpy":        "#C62828",
        # legacy fallback
        "I": "#1565C0", "II": "#F57C00", "III": "#C62828",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    if "group" in df.columns:
        for grp, color in group_colors.items():
            sub = df[df["group"] == grp]
            if not sub.empty:
                ax.scatter(sub["cv2"], sub["adi"], c=color, alpha=0.6, s=28,
                           label=f"{grp} (n={len(sub)})", zorder=3)
    else:
        ax.scatter(df["cv2"], df["adi"], alpha=0.5, s=25, zorder=3)

    ax.axvline(CV2_T, color="gray", lw=1.5, linestyle="--", alpha=0.8)
    ax.axhline(ADI_T, color="gray", lw=1.5, linestyle="--", alpha=0.8)

    xlim = max(df["cv2"].quantile(0.97) * 1.1, CV2_T * 2.5)
    ylim = max(df["adi"].quantile(0.97) * 1.1, ADI_T * 3)
    ax.set_xlim(-0.02, xlim)
    ax.set_ylim(0.8, ylim)

    kw = dict(fontsize=9, color="gray", alpha=0.7, style="italic")
    ax.text(0.02, ADI_T * 0.9, "Suave", **kw, va="top")
    ax.text(CV2_T * 1.05, ADI_T * 0.9, "Errática", **kw, va="top")
    ax.text(0.02, ADI_T * 1.05, "Intermitente", **kw)
    ax.text(CV2_T * 1.05, ADI_T * 1.05, "Grumosa\n(Lumpy)", **kw)

    ax.set_xlabel("CV² (variabilidade da demanda não-nula)", fontsize=10)
    ax.set_ylabel("ADI — Intervalo Médio de Demanda", fontsize=10)
    ax.set_title("Classificação ADI × CV² (Syntetos et al. 2005)\n"
                 "Distribuição das séries por tipo de demanda", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "adi_cv_scatter", out_dir, fmt, params.get("figure_dpi", 150))


def plot_demand_distribution(scenarios, scenarios_meta, out_dir, params):
    """
    Demand histograms per group (I/II/III) — non-zero values only.
    Shows % of zero-demand cycles in subtitle.
    """
    if scenarios is None or scenarios_meta is None:
        return None

    merged = scenarios.merge(
        scenarios_meta[["warehouse", "store_id", "item_id", "group"]],
        on=["warehouse", "store_id", "item_id"], how="left"
    )
    groups = sorted(merged["group"].dropna().unique())
    if not groups:
        return None

    group_colors = {
        "Smooth": "#1565C0", "Erratic": "#F57C00",
        "Intermittent": "#2E7D32", "Lumpy": "#C62828",
        "I": "#1565C0", "II": "#F57C00", "III": "#C62828",
    }
    fig, axes = plt.subplots(1, len(groups),
                              figsize=(max(10, len(groups) * 4), 5))
    if len(groups) == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        sub = merged[merged["group"] == grp]["demand"].dropna()
        pct_zero = (sub == 0).mean() * 100
        positive = sub[sub > 0]
        color = group_colors.get(grp, "#888")
        ax.hist(positive, bins=30, color=color, alpha=0.8, edgecolor="white")
        if not positive.empty:
            ax.axvline(positive.mean(), color="black", lw=1.5, linestyle="--",
                       label=f"Media = {positive.mean():.1f}")
        ax.set_title(f"{grp}  (n={len(sub):,} | zeros = {pct_zero:.1f}%)", fontsize=9)
        ax.set_xlabel("Demanda por ciclo (valores positivos)")
        ax.set_ylabel("Frequencia")
        ax.legend(fontsize=8)

    fig.suptitle("Distribuicao da Demanda por Grupo (Syntetos-Boylan)", fontsize=11)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "demand_distribution", out_dir, fmt, params.get("figure_dpi", 150))


def plot_intermittency_calendar(scenarios, out_dir, params, max_series=40):
    """
    Calendar heatmap — rows = demand series, columns = bimonthly cycles.
    White/light = zero demand; blue gradient = positive demand magnitude.
    """
    if scenarios is None or scenarios.empty:
        return None

    key_cols = ["warehouse", "store_id", "item_id"]
    all_cycles = sorted(scenarios["venda_ciclo"].astype(str).unique())
    cycle_idx = {c: j for j, c in enumerate(all_cycles)}

    keys = (scenarios.groupby(key_cols)["demand"].count()
            .reset_index().head(max_series))
    mat = np.zeros((len(keys), len(all_cycles)))

    for i, krow in keys.iterrows():
        ser = scenarios[
            (scenarios["warehouse"] == krow["warehouse"]) &
            (scenarios["store_id"]  == krow["store_id"]) &
            (scenarios["item_id"]   == krow["item_id"])
        ]
        for _, r in ser.iterrows():
            c = str(r["venda_ciclo"])
            if c in cycle_idx:
                mat[i, cycle_idx[c]] = r["demand"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "demand_cal", ["#F5F5F5", "#BBDEFB", "#1565C0"], N=256)

    fig, ax = plt.subplots(figsize=(max(14, len(all_cycles) * 0.45),
                                    max(6, len(keys) * 0.35 + 2)))
    ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")

    step = max(1, len(all_cycles) // 20)
    ax.set_xticks(range(0, len(all_cycles), step))
    ax.set_xticklabels(all_cycles[::step], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel(f"{len(keys)} séries (warehouse × loja × item)", fontsize=9)
    ax.set_xlabel("Ciclo bimestral (YYYYCC)")
    ax.set_title("Calendário de Intermitência — Branco = demanda zero, Azul = demanda positiva",
                 fontsize=11)
    im = ax.images[0]
    plt.colorbar(im, ax=ax, label="Demanda", shrink=0.6)
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "intermittency_calendar", out_dir, fmt, params.get("figure_dpi", 150))


# ── F. MAPAS GEOGRÁFICOS DO BRASIL ────────────────────────────────────────────

def _load_brazil_states():
    """Load Brazil state boundaries via geobr (official IBGE data)."""
    try:
        import geobr
        return geobr.read_state(year=2020)   # API correta: read_state (singular)
    except ImportError:
        log.warning("geobr não instalado. Mapas do Brasil ignorados. "
                    "Instale com: pip install geobr")
        return None
    except Exception as e:
        log.warning("geobr.read_state falhou: %s", e)
        return None


def plot_brazil_store_map(scenarios_meta, out_dir, params):
    """
    Brazil choropleth: states colored by number of stores in the simulation.
    State centroids labeled with abbreviation + store count.
    Uses geobr for official IBGE state boundaries.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        log.warning("geopandas não instalado — mapa do Brasil ignorado")
        return None

    states_gdf = _load_brazil_states()
    if states_gdf is None:
        return None

    store_stats = (scenarios_meta
                   .groupby("warehouse")
                   .agg(n_stores=("store_id", "nunique"),
                        n_items=("item_id", "nunique"),
                        mean_cv=("cv", "mean"))
                   .reset_index())

    state_col = "abbrev_state" if "abbrev_state" in states_gdf.columns else \
                "name_state" if "name_state" in states_gdf.columns else \
                states_gdf.columns[0]
    merged = states_gdf.merge(store_stats, left_on=state_col,
                              right_on="warehouse", how="left")

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_facecolor("#E8F4F8")
    states_gdf.plot(ax=ax, color="#ECECEC", edgecolor="white", linewidth=0.5)

    has_data = merged[merged["n_stores"].notna()]
    if not has_data.empty:
        has_data.plot(ax=ax, column="n_stores", cmap="Blues",
                      legend=True,
                      legend_kwds={"label": "Número de Lojas",
                                   "shrink": 0.6, "orientation": "vertical"},
                      linewidth=0.5, edgecolor="white")

    for _, row in merged.iterrows():
        try:
            centroid = row.geometry.centroid
            label = str(row.get(state_col, ""))
            n = row.get("n_stores", None)
            if n is not None and not (isinstance(n, float) and np.isnan(n)):
                ax.annotate(f"{label}\n({int(n)})", (centroid.x, centroid.y),
                            ha="center", va="center", fontsize=7, fontweight="bold")
            else:
                ax.annotate(label, (centroid.x, centroid.y),
                            ha="center", va="center", fontsize=7, color="gray")
        except Exception:
            continue

    ax.set_title("Distribuição de Lojas por Estado (UF)\n"
                 "Cobertura do conjunto de simulação", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fmt = params.get("figure_format", "pdf")
    return _save(fig, "brazil_store_map", out_dir, fmt, params.get("figure_dpi", 150))


def plot_brazil_kpi_choropleth(kpis_df, out_dir, params):
    """
    One Brazil choropleth per KPI (TIC, NS, TR) — states colored by mean KPI.
    Green = better performance (direction-aware colormap per KPI).
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        log.warning("geopandas não instalado — mapa do Brasil ignorado")
        return None

    states_gdf = _load_brazil_states()
    if states_gdf is None:
        return None

    state_col = "abbrev_state" if "abbrev_state" in states_gdf.columns else \
                "name_state" if "name_state" in states_gdf.columns else \
                states_gdf.columns[0]

    metrics_to_plot = [m for m in ["TIC", "NS", "TR"] if m in kpis_df.columns]
    saved = {}

    for metric in metrics_to_plot:
        state_kpi = kpis_df.groupby("warehouse")[metric].mean().reset_index()
        merged = states_gdf.merge(state_kpi, left_on=state_col,
                                  right_on="warehouse", how="left")

        _, higher_better = KPI_META.get(metric, (metric, False))
        cmap = "RdYlGn" if higher_better else "RdYlGn_r"

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_facecolor("#E8F4F8")
        states_gdf.plot(ax=ax, color="#ECECEC", edgecolor="white", linewidth=0.5)

        has_data = merged[merged[metric].notna()]
        if not has_data.empty:
            has_data.plot(ax=ax, column=metric, cmap=cmap,
                          legend=True,
                          legend_kwds={"label": KPI_META.get(metric, (metric,))[0],
                                       "shrink": 0.6},
                          linewidth=0.5, edgecolor="white",
                          missing_kwds={"color": "#D0D0D0", "label": "Sem dados"})

        for _, row in merged.iterrows():
            try:
                centroid = row.geometry.centroid
                label = str(row.get(state_col, ""))
                val = row.get(metric, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    ax.annotate(f"{label}\n{val:.1f}", (centroid.x, centroid.y),
                                ha="center", va="center", fontsize=7, fontweight="bold")
                else:
                    ax.annotate(label, (centroid.x, centroid.y),
                                ha="center", va="center", fontsize=7, color="gray")
            except Exception:
                continue

        label, _ = KPI_META.get(metric, (metric, False))
        direction = "maior = melhor" if higher_better else "menor = melhor"
        ax.set_title(f"Distribuição de {label} por Estado\n({direction})", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fmt = params.get("figure_format", "pdf")
        p = _save(fig, f"brazil_{metric.lower()}_choropleth", out_dir, fmt,
                  params.get("figure_dpi", 150))
        saved[f"brazil_{metric.lower()}"] = p

    return saved or None


# ── G. METRICAS DE FORECAST AGREGADAS ────────────────────────────────────────

METRIC_DISPLAY = {
    "MASE":    ("MASE",    False, "scale-free (< 1 = melhor que naive)"),
    "RMSSE":   ("RMSSE",   False, "RMSE escalonado (M4)"),
    "sMAPE":   ("sMAPE (%)", False, "sMAPE — simetrico"),
    "MAPE":    ("MAPE (%)", False, "MAPE classico"),
    "MAE":     ("MAE",     False, "erro absoluto medio"),
    "MBE":     ("MBE",     None,  "vies (negativo = over-forecast)"),
    "TheilsU": ("Theil U", False, "< 1 = melhor que random walk"),
}
MODEL_COLORS = {"LSTM": "#1565C0", "ANN": "#2E7D32", "XGBoost": "#F57C00"}


def plot_forecast_metrics_distribution(forecast_metrics: pd.DataFrame,
                                        out_dir: str, params: dict) -> dict:
    """
    Boxplots of MASE, RMSSE, sMAPE, MBE per model — aggregated across all series.
    Only uses rows where cycle == 'ALL' (aggregate per-series metrics).
    Returns dict of saved paths.
    """
    if forecast_metrics is None or forecast_metrics.empty:
        return {}

    # aggregate rows (cycle='ALL') — one value per (warehouse,store,item,model)
    agg = forecast_metrics[forecast_metrics["cycle"].astype(str) == "ALL"].copy()
    if agg.empty:
        log.warning("forecast_metrics: nenhuma linha com cycle='ALL'")
        return {}

    metrics_to_plot = [m for m in ["MASE", "RMSSE", "sMAPE", "MBE", "TheilsU"] if m in agg.columns]
    if not metrics_to_plot:
        return {}

    models = sorted(agg["model"].unique())
    fmt = params.get("figure_format", "pdf")
    dpi = params.get("figure_dpi", 150)
    saved = {}

    # ── 1. Multi-metric boxplot (one panel per metric) ────────────────────
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(max(10, n * 3.5), 5))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        data  = [agg[agg["model"] == m][metric].dropna().values for m in models]
        colors = [MODEL_COLORS.get(m, "#888") for m in models]
        bp = ax.boxplot(data, labels=models, patch_artist=True,
                        medianprops={"color": "black", "lw": 1.5},
                        flierprops={"marker": ".", "ms": 3, "alpha": 0.4})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        label, _, note = METRIC_DISPLAY.get(metric, (metric, False, ""))
        ax.set_title(f"{label}\n({note})", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        if metric in ("MASE", "RMSSE", "TheilsU"):
            ax.axhline(1.0, color="red", lw=1, linestyle="--", alpha=0.6,
                       label="Baseline (=1)")
            ax.legend(fontsize=7)
        elif metric == "MBE":
            ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.7)

    fig.suptitle("Distribuicao das Metricas de Forecast por Modelo\n"
                 "(todas as series — ciclos de teste)", fontsize=10)
    fig.tight_layout()
    saved["forecast_metrics_boxplot"] = _save(
        fig, "forecast_metrics_boxplot", out_dir, fmt, dpi)

    # ── 2. MASE vs sMAPE scatter — model ranking ─────────────────────────
    if "MASE" in agg.columns and "sMAPE" in agg.columns:
        summary = agg.groupby("model").agg(
            MASE_med  = ("MASE",  "median"),
            sMAPE_med = ("sMAPE", "median"),
            MASE_q25  = ("MASE",  lambda x: x.quantile(0.25)),
            MASE_q75  = ("MASE",  lambda x: x.quantile(0.75)),
            n_series  = ("MASE",  "count"),
        ).reset_index()

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        for _, row in summary.iterrows():
            color = MODEL_COLORS.get(row["model"], "#888")
            ax2.scatter(row["sMAPE_med"], row["MASE_med"],
                        s=180, color=color, zorder=4, edgecolors="black", lw=0.7)
            ax2.errorbar(row["sMAPE_med"], row["MASE_med"],
                         yerr=[[row["MASE_med"] - row["MASE_q25"]],
                               [row["MASE_q75"]  - row["MASE_med"]]],
                         fmt="none", color=color, capsize=4, lw=1.2, alpha=0.7)
            ax2.annotate(f'{row["model"]} (n={int(row["n_series"])})',
                         (row["sMAPE_med"], row["MASE_med"]),
                         textcoords="offset points", xytext=(8, 4), fontsize=9)

        ax2.axhline(1.0, color="red", lw=1, linestyle="--", alpha=0.5,
                    label="MASE = 1 (naive baseline)")
        ax2.set_xlabel("sMAPE mediano (%)", fontsize=10)
        ax2.set_ylabel("MASE mediano (IQR error bars)", fontsize=10)
        ax2.set_title("Ranking de Modelos: MASE vs sMAPE\n"
                      "(canto inferior esquerdo = melhor)", fontsize=10)
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        saved["forecast_model_ranking"] = _save(
            fig2, "forecast_model_ranking", out_dir, fmt, dpi)

    # ── 3. MASE by demand group (if group column present) ─────────────────
    group_col = None
    for gc in ("group", "demand_group"):
        if gc in agg.columns:
            group_col = gc
            break

    if group_col is not None and agg[group_col].nunique() > 1:
        groups = sorted(agg[group_col].dropna().unique())
        group_clrs = {
            "Smooth": "#1565C0", "Erratic": "#F57C00",
            "Intermittent": "#2E7D32", "Lumpy": "#C62828",
            "I": "#1565C0", "II": "#F57C00", "III": "#C62828",
        }
        fig3, ax3 = plt.subplots(figsize=(max(8, len(groups) * 2), 5))
        positions = []
        tick_labels = []
        idx = 0
        for grp in groups:
            for mdl in models:
                sub = agg[(agg[group_col] == grp) & (agg["model"] == mdl)]["MASE"].dropna()
                if sub.empty:
                    continue
                bp = ax3.boxplot([sub.values], positions=[idx],
                                 patch_artist=True, widths=0.6,
                                 medianprops={"color": "black", "lw": 1.5},
                                 flierprops={"marker": ".", "ms": 3, "alpha": 0.4})
                for patch in bp["boxes"]:
                    patch.set_facecolor(MODEL_COLORS.get(mdl, "#888"))
                    patch.set_alpha(0.7)
                positions.append(idx)
                tick_labels.append(f"{grp}\n{mdl}")
                idx += 1
            idx += 0.5  # gap between groups

        ax3.axhline(1.0, color="red", lw=1, linestyle="--", alpha=0.5,
                    label="Baseline naive (MASE=1)")
        ax3.set_xticks(positions)
        ax3.set_xticklabels(tick_labels, fontsize=7)
        ax3.set_ylabel("MASE")
        ax3.set_title("MASE por Grupo de Demanda (Syntetos-Boylan) e Modelo", fontsize=10)
        ax3.legend(fontsize=8)
        fig3.tight_layout()
        saved["forecast_mase_by_group"] = _save(
            fig3, "forecast_mase_by_group", out_dir, fmt, dpi)

    log.info("[Forecast metrics] Figuras salvas: %s", list(saved.keys()))
    return saved
