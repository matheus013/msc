"""
presentation_visuals.py
Visualizacoes adicionais para a apresentacao de qualificacao, derivadas dos
artefatos JA EXISTENTES em data/08_reporting/ (profiles/, strategy/) e dos
dados de entrada (demand_profiles.parquet, scenarios.parquet).

Este script NAO reexecuta a simulacao nem recalcula KPIs: ele apenas le os
artefatos ja gerados pelos scripts profile_policy_analysis.py e
strategy_cost_comparison.py, e o demand_profiles.parquet/scenarios.parquet
para caracterizacao de demanda, produzindo figuras mais legiveis para
slides (uma mensagem central por figura, paleta sobria).

Entrada (somente leitura):
  data/04_feature/demand_profiles.parquet
  data/03_primary/scenarios.parquet                         (opcional, exemplos Lumpy)
  data/08_reporting/profiles/profile_policy_metrics.csv
  data/08_reporting/profiles/dominant_policy_by_profile.csv
  data/08_reporting/strategy/strategy_cost_comparison.csv

Saida (em data/08_reporting/presentation/):
  syntetos_boylan_scatter.png / .pdf / .csv
  strategy_tradeoff_cti_ns.png / .pdf / .csv
  profile_dominance_bars.png / .pdf / .csv
  profile_policy_heatmap_simplified.png / .pdf / .csv
  aipe_evidence_pipeline.png / .pdf
  lumpy_series_examples.png / .pdf / .csv
  manifest.json
  README.md
  figures_validation.md

Uso:
  python simulation/src/reporting/presentation_visuals.py
  ou como modulo: from reporting.presentation_visuals import run
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[2]                 # .../sbpo/simulation
DATA_DIR      = REPO_ROOT / "data"
PROF_PATH     = DATA_DIR / "04_feature" / "demand_profiles.parquet"
SCENARIOS_PATH = DATA_DIR / "03_primary" / "scenarios.parquet"
PROFILES_DIR  = DATA_DIR / "08_reporting" / "profiles"
STRATEGY_DIR  = DATA_DIR / "08_reporting" / "strategy"
OUT_DIR       = DATA_DIR / "08_reporting" / "presentation"

PROFILE_METRICS_PATH   = PROFILES_DIR / "profile_policy_metrics.csv"
DOMINANT_PROFILE_PATH  = PROFILES_DIR / "dominant_policy_by_profile.csv"
STRATEGY_COMPARISON_PATH = STRATEGY_DIR / "strategy_cost_comparison.csv"

NS_THRESHOLD = 0.70
ADI_CUT = 1.32
CV2_CUT = 0.49

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

# Paleta sobria, consistente com a apresentacao de qualificacao
# (docs/qualification_presentation/main.tex)
COLOR_DARKBLUE   = "#0F2F57"
COLOR_PETROL     = "#1E5D8C"
COLOR_TEAL       = "#0E8C95"
COLOR_GRAY       = "#4D6175"
COLOR_LIGHTGRAY  = "#EAF0F6"
COLOR_WARN       = "#9A3A3A"
COLOR_CARDBG     = "#F4F8FC"

SLIDE_FIGSIZE = (13.33, 7.5)
FIG_DPI = 220

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   COLOR_GRAY,
    "axes.labelcolor":  COLOR_DARKBLUE,
    "text.color":       COLOR_DARKBLUE,
    "xtick.color":      COLOR_GRAY,
    "ytick.color":      COLOR_GRAY,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.color":       COLOR_GRAY,
    "font.size":        13,
    "font.family":      "sans-serif",
})


@dataclass
class ManifestEntry:
    filename: str
    description: str
    source_data: str
    generator: str
    notes: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_reporting_data() -> dict:
    """Carrega os artefatos ja existentes necessarios as novas visualizacoes.
    Nao recalcula KPIs nem reexecuta simulacao."""
    data = {}

    log.info("Lendo demand_profiles.parquet …")
    data["profiles"] = pd.read_parquet(PROF_PATH)

    if SCENARIOS_PATH.exists():
        log.info("Lendo scenarios.parquet (series de demanda) …")
        data["scenarios"] = pd.read_parquet(SCENARIOS_PATH)
    else:
        log.warning("scenarios.parquet nao encontrado; exemplos Lumpy serao pulados.")
        data["scenarios"] = None

    for key, path in [
        ("profile_metrics", PROFILE_METRICS_PATH),
        ("dominant_by_profile", DOMINANT_PROFILE_PATH),
        ("strategy_comparison", STRATEGY_COMPARISON_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Artefato esperado nao encontrado: {path}. "
                "Execute antes: python simulation/src/reporting/profile_policy_analysis.py "
                "e strategy_cost_comparison.py"
            )
        log.info(f"Lendo {path.relative_to(REPO_ROOT)} …")
        data[key] = pd.read_csv(path)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plano Syntetos-Boylan (ADI x CV2)
# ─────────────────────────────────────────────────────────────────────────────

def _sb_quadrant(adi: float, cv2: float) -> str:
    if adi < ADI_CUT and cv2 < CV2_CUT:
        return "Smooth"
    if adi < ADI_CUT and cv2 >= CV2_CUT:
        return "Erratic"
    if adi >= ADI_CUT and cv2 < CV2_CUT:
        return "Intermittent"
    return "Lumpy"


def plot_syntetos_boylan_plane(profiles: pd.DataFrame, out_dir: Path) -> dict:
    """Scatter ADI x CV2 com os 4 quadrantes de Syntetos-Boylan, destacando Lumpy.
    Cada ponto = 1 serie (loja, produto). Quadrante recalculado diretamente de
    profiles['adi'] e profiles['cv2'] (sem usar a coluna 'group', que e apenas
    conferida para consistencia)."""
    df = profiles.copy()
    df["sb_quadrant"] = [
        _sb_quadrant(a, c) for a, c in zip(df["adi"], df["cv2"])
    ]

    # Conferência: a coluna pré-existente 'group' deve concordar com o quadrante recomputado
    if "group" in df.columns:
        mismatch = (df["group"] != df["sb_quadrant"]).sum()
        if mismatch:
            log.warning(
                f"{mismatch} série(s) com 'group' != quadrante recomputado de adi/cv2."
            )

    n_total = len(df)
    counts = df["sb_quadrant"].value_counts()
    summary = (
        counts.rename_axis("quadrante").reset_index(name="n_series")
        .assign(pct=lambda d: (d["n_series"] / n_total * 100).round(1))
        .sort_values("quadrante")
        .reset_index(drop=True)
    )

    quad_colors = {
        "Smooth": COLOR_GRAY,
        "Erratic": COLOR_PETROL,
        "Intermittent": COLOR_TEAL,
        "Lumpy": COLOR_WARN,
    }

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)

    x_max = max(df["adi"].max() * 1.15, ADI_CUT * 1.6)
    y_max = max(df["cv2"].max() * 1.15, CV2_CUT * 1.6)
    y_min = max(df["cv2"].min() * 0.85, 0.05)

    ax.set_yscale("log")
    # Sombreia apenas o quadrante Lumpy (ADI >= corte E CV2 >= corte); converte o
    # corte de CV2 para fração do eixo (escala log) dentro do intervalo plotado.
    cv2_cut_frac = np.clip(
        (np.log10(CV2_CUT) - np.log10(y_min)) / (np.log10(y_max) - np.log10(y_min)),
        0.0, 1.0,
    )
    ax.axvspan(ADI_CUT, x_max, cv2_cut_frac, 1, color=COLOR_WARN, alpha=0.06, zorder=0)

    for quad, color in quad_colors.items():
        sub = df[df["sb_quadrant"] == quad]
        if sub.empty:
            continue
        ax.scatter(
            sub["adi"], sub["cv2"],
            s=42, color=color, alpha=0.75,
            edgecolor="white", linewidth=0.5,
            label=f"{quad} (n={len(sub)})", zorder=3,
        )

    ax.axvline(ADI_CUT, color=COLOR_DARKBLUE, linestyle="--", linewidth=1.3, zorder=2)
    ax.axhline(CV2_CUT, color=COLOR_DARKBLUE, linestyle="--", linewidth=1.3, zorder=2)

    label_kwargs = dict(fontsize=12, fontweight="bold", color=COLOR_DARKBLUE, alpha=0.85)
    ax.text(ADI_CUT * 0.55, y_min * 1.15, "Smooth", ha="center", **label_kwargs)
    ax.text(ADI_CUT * 0.55, y_max * 0.75, "Erratic", ha="center", **label_kwargs)
    ax.text((ADI_CUT + x_max) / 2, y_min * 1.15, "Intermittent", ha="center", **label_kwargs)
    ax.text((ADI_CUT + x_max) / 2, y_max * 0.75, "Lumpy", ha="center",
            fontsize=13, fontweight="bold", color=COLOR_WARN)

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("ADI — intervalo médio entre demandas positivas (ciclos)")
    ax.set_ylabel(r"CV$^2$ — variabilidade normalizada (escala log)")
    ax.set_title(
        "Classificação Syntetos-Boylan das séries — Experimento 2 (BA)",
        fontsize=16, fontweight="bold", color=COLOR_DARKBLUE, pad=14,
    )

    lumpy_row = summary[summary["quadrante"] == "Lumpy"]
    lumpy_pct = float(lumpy_row["pct"].iloc[0]) if not lumpy_row.empty else 0.0
    lumpy_n = int(lumpy_row["n_series"].iloc[0]) if not lumpy_row.empty else 0

    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
    fig.text(
        0.5, 0.045,
        f"{lumpy_n}/{n_total} séries ({lumpy_pct:.1f}%) classificadas como Lumpy no recorte avaliado",
        ha="center", fontsize=12.5, color=COLOR_WARN, fontweight="bold",
    )
    fig.text(
        0.01, 0.005,
        f"Fonte: demand_profiles.parquet ({n_total} séries, loja×produto, BA, Experimento 2). "
        f"Cortes: ADI={ADI_CUT}, CV²={CV2_CUT} (Syntetos-Boylan, 2005).",
        fontsize=8.5, color=COLOR_GRAY,
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    png_path = out_dir / "syntetos_boylan_scatter.png"
    pdf_path = out_dir / "syntetos_boylan_scatter.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {png_path}")

    csv_path = out_dir / "syntetos_boylan_summary.csv"
    summary.to_csv(csv_path, index=False)
    log.info(f"Sumário salvo: {csv_path}")

    return {
        "png": png_path, "pdf": pdf_path, "csv": csv_path,
        "summary": summary, "n_total": n_total, "lumpy_pct": lumpy_pct,
        "mismatch_group_vs_quadrant": int(mismatch) if "group" in df.columns else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trade-off política única vs seleção por perfil
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_tradeoff(strategy_comparison: pd.DataFrame, out_dir: Path) -> dict:
    """Dois paineis (CTI medio e NS medio) por estrategia A1/A2/B/C, lidos
    diretamente de strategy/strategy_cost_comparison.csv (nao recalculado)."""
    df = strategy_comparison.copy()
    order = ["A1", "A2", "B", "C"]
    df["estrategia"] = df["estrategia"].astype(str)
    df = df.set_index("estrategia").reindex(order).reset_index()

    labels = {
        "A1": "A1: Única\n(melhor viável)",
        "A2": "A2: Única\n(baseline EOQ)",
        "B": "B: Por perfil",
        "C": "C: Oráculo †",
    }
    bar_colors = {
        "A1": COLOR_GRAY, "A2": COLOR_GRAY,
        "B": COLOR_TEAL, "C": COLOR_WARN,
    }
    hatches = {"A1": "", "A2": "", "B": "", "C": "//"}

    fig, axes = plt.subplots(1, 2, figsize=SLIDE_FIGSIZE)

    # Painel 1: CTI medio
    ax = axes[0]
    bars = ax.bar(
        [labels[e] for e in order], df["CTI_medio"],
        color=[bar_colors[e] for e in order],
        hatch=[hatches[e] for e in order],
        edgecolor=COLOR_DARKBLUE, linewidth=0.8,
    )
    for b, e in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + df["CTI_medio"].max() * 0.02,
                f"R$ {df.loc[df['estrategia']==e, 'CTI_medio'].iloc[0]:.2f}",
                ha="center", fontsize=11, fontweight="bold", color=COLOR_DARKBLUE)
    ax.set_ylabel("CTI médio (R$)")
    ax.set_title("Custo Total de Inventário médio", fontsize=14, fontweight="bold")
    ax.set_ylim(0, df["CTI_medio"].max() * 1.22)
    ax.tick_params(axis="x", labelsize=10.5)

    cti_a1 = df.loc[df["estrategia"] == "A1", "CTI_medio"].iloc[0]
    cti_b = df.loc[df["estrategia"] == "B", "CTI_medio"].iloc[0]
    red_pct_b = df.loc[df["estrategia"] == "B", "red_pct_vs_A1"].iloc[0]

    # Painel 2: NS medio
    ax = axes[1]
    bars = ax.bar(
        [labels[e] for e in order], df["NS_medio"],
        color=[bar_colors[e] for e in order],
        hatch=[hatches[e] for e in order],
        edgecolor=COLOR_DARKBLUE, linewidth=0.8,
    )
    for b, e in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{df.loc[df['estrategia']==e, 'NS_medio'].iloc[0]:.3f}",
                ha="center", fontsize=11, fontweight="bold", color=COLOR_DARKBLUE)
    ax.axhline(NS_THRESHOLD, color=COLOR_WARN, linestyle=":", linewidth=1.4)
    ax.text(3.4, NS_THRESHOLD - 0.06, f"limiar mínimo NS={NS_THRESHOLD:.2f}",
            fontsize=9.5, color=COLOR_WARN, ha="right")
    ax.set_ylabel("Nível de Serviço médio")
    ax.set_title("Nível de Serviço médio", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", labelsize=10.5)

    ns_a1 = df.loc[df["estrategia"] == "A1", "NS_medio"].iloc[0]
    ns_b = df.loc[df["estrategia"] == "B", "NS_medio"].iloc[0]

    fig.suptitle(
        "Política única vs. seleção por perfil — trade-off custo-serviço",
        fontsize=17, fontweight="bold", color=COLOR_DARKBLUE, y=1.04,
    )
    fig.text(
        0.5, 0.965,
        f"Seleção por perfil (B): CTI {red_pct_b:+.1f}% vs. política única "
        f"(R$ {cti_a1:.2f} → R$ {cti_b:.2f})  ·  NS {ns_a1:.3f} → {ns_b:.3f}",
        ha="center", fontsize=12.5, color=COLOR_TEAL, fontweight="bold",
    )
    fig.text(
        0.5, -0.03,
        "A seleção por perfil reduz custo médio, mas há queda de nível de serviço. "
        "Trata-se de trade-off custo-serviço.\n"
        "† C (oráculo por série) é referência exploratória, não estratégia operacional.",
        ha="center", fontsize=10.5, color=COLOR_GRAY,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    png_path = out_dir / "strategy_tradeoff_cti_ns.png"
    pdf_path = out_dir / "strategy_tradeoff_cti_ns.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {png_path}")

    csv_path = out_dir / "strategy_tradeoff_summary.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Sumário salvo: {csv_path}")

    return {
        "png": png_path, "pdf": pdf_path, "csv": csv_path,
        "cti_a1": cti_a1, "cti_b": cti_b, "red_pct_b": red_pct_b,
        "ns_a1": ns_a1, "ns_b": ns_b,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dominância por perfil
# ─────────────────────────────────────────────────────────────────────────────

def plot_profile_dominance(dominant_by_profile: pd.DataFrame, out_dir: Path) -> dict:
    """Barras horizontais: política dominante (menor CTI viável, NS>=0.70) por
    perfil, lida diretamente de profiles/dominant_policy_by_profile.csv."""
    df = dominant_by_profile.copy().sort_values("CTI_mean")

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)

    colors = [COLOR_TEAL if n >= 20 else COLOR_GRAY for n in df["n_series"]]
    bars = ax.barh(df["profile_display"], df["CTI_mean"], color=colors,
                    edgecolor=COLOR_DARKBLUE, linewidth=0.8, height=0.55)

    for bar, row in zip(bars, df.itertuples()):
        exploratory = " (n<20: evidência exploratória)" if row.n_series < 20 else ""
        ax.text(
            bar.get_width() * 0.015, bar.get_y() + bar.get_height() / 2,
            f"{row.dominant_policy_disp}  ·  CTI R$ {row.CTI_mean:.2f}  ·  NS {row.NS_mean:.3f}",
            va="center", fontsize=12, color="white", fontweight="bold",
        )
        ax.text(
            bar.get_width() + df["CTI_mean"].max() * 0.015, bar.get_y() + bar.get_height() / 2,
            f"n={row.n_series}{exploratory}",
            va="center", fontsize=10.5, color=COLOR_WARN if row.n_series < 20 else COLOR_GRAY,
        )

    ax.set_xlabel("CTI médio da política dominante (R$)")
    ax.set_title(
        "Dominância depende do Perfil Operacional de Demanda\n"
        f"(política viável de menor CTI, NS médio ≥ {NS_THRESHOLD:.2f})",
        fontsize=15, fontweight="bold", color=COLOR_DARKBLUE, pad=12,
    )
    ax.set_xlim(0, df["CTI_mean"].max() * 1.55)
    ax.invert_yaxis()

    handles = [
        mpatches.Patch(color=COLOR_TEAL, label="n ≥ 20 (evidência consolidada)"),
        mpatches.Patch(color=COLOR_GRAY, label="n < 20 (evidência exploratória)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10.5, framealpha=0.9)

    fig.text(
        0.01, 0.01,
        "Fonte: data/08_reporting/profiles/dominant_policy_by_profile.csv "
        "(Experimento 2, BA, regime Lumpy).",
        fontsize=8.5, color=COLOR_GRAY,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    png_path = out_dir / "profile_dominance_bars.png"
    pdf_path = out_dir / "profile_dominance_bars.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {png_path}")

    csv_path = out_dir / "profile_dominance_summary.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Sumário salvo: {csv_path}")

    return {"png": png_path, "pdf": pdf_path, "csv": csv_path, "table": df}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Heatmap simplificado CTI por perfil x política
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY_POLICIES = ["EOQ", "sS", "Newsvendor", "SA", "DQN", "PPO", "GA-DQN", "GA-PPO"]


def plot_policy_profile_heatmap_simplified(profile_metrics: pd.DataFrame,
                                            dominant_by_profile: pd.DataFrame,
                                            out_dir: Path) -> dict:
    """Heatmap de CTI medio (perfil x politica), restrito a um subconjunto
    legivel de politicas + a politica dominante de cada perfil. Celulas com
    NS medio < 0.70 (inviaveis) sao marcadas com hachuras, e a celula dominante
    de cada linha recebe contorno em destaque."""
    df = profile_metrics.copy()

    dominant_policies = set(dominant_by_profile["dominant_policy"].unique())
    policies = [p for p in PRIORITY_POLICIES if p in df["policy"].unique()]
    for p in dominant_policies:
        if p not in policies and p in df["policy"].unique():
            policies.append(p)

    profiles_order = [p for p in PROFILE_DISPLAY if p in df["operational_profile"].unique()]
    df = df[df["policy"].isin(policies) & df["operational_profile"].isin(profiles_order)]

    pivot_cti = df.pivot(index="operational_profile", columns="policy", values="TIC_mean") \
                  .reindex(index=profiles_order, columns=policies)
    pivot_ns = df.pivot(index="operational_profile", columns="policy", values="NS_mean") \
                 .reindex(index=profiles_order, columns=policies)

    row_labels = [PROFILE_DISPLAY.get(p, p) for p in pivot_cti.index]
    col_labels = [POLICY_DISPLAY.get(p, p) for p in pivot_cti.columns]

    dominant_lookup = dominant_by_profile.set_index("operational_profile")["dominant_policy"].to_dict()

    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    data = pivot_cti.values.astype(float)
    # Normaliza por linha (perfil) para destacar heterogeneidade relativa, mantendo
    # o valor absoluto anotado em cada célula.
    row_norm = (data - np.nanmin(data, axis=1, keepdims=True)) / (
        np.nanmax(data, axis=1, keepdims=True) - np.nanmin(data, axis=1, keepdims=True) + 1e-9
    )
    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")
    im = ax.imshow(row_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    for r, profile_key in enumerate(pivot_cti.index):
        for c, policy_key in enumerate(pivot_cti.columns):
            val = data[r, c]
            ns_val = pivot_ns.values[r, c]
            if np.isnan(val):
                continue
            viable = ns_val >= NS_THRESHOLD
            text_color = "black" if row_norm[r, c] < 0.6 else "white"
            cell_text = f"{val:.0f}" + ("" if viable else "\n(inviável)")
            ax.text(c, r, cell_text, ha="center", va="center",
                    fontsize=10.5 if viable else 9.5, color=text_color,
                    fontweight="bold" if policy_key == dominant_lookup.get(profile_key) else "normal")
            if not viable:
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False, hatch="////",
                    edgecolor=COLOR_GRAY, linewidth=0,
                ))
            if policy_key == dominant_lookup.get(profile_key):
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False,
                    edgecolor=COLOR_DARKBLUE, linewidth=3.2,
                ))

    ax.set_title(
        "CTI médio por perfil e política (subconjunto legível)\n"
        f"Hachura = inviável (NS médio < {NS_THRESHOLD:.2f}); contorno = dominante do perfil",
        fontsize=14.5, fontweight="bold", color=COLOR_DARKBLUE, pad=12,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="CTI relativo dentro do perfil (0=menor, 1=maior)")

    fig.text(
        0.01, 0.01,
        "Jornaleiro pode ter menor CTI bruto, mas é inviável quando NS < 0,70 "
        "(ex.: Sparse High Impact, NS=0,55). "
        "Fonte: data/08_reporting/profiles/profile_policy_metrics.csv.",
        fontsize=8.5, color=COLOR_GRAY, wrap=True,
    )

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    png_path = out_dir / "profile_policy_heatmap_simplified.png"
    pdf_path = out_dir / "profile_policy_heatmap_simplified.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {png_path}")

    out_table = df[["operational_profile", "profile_display", "policy", "policy_display",
                     "TIC_mean", "NS_mean", "n_series"]].copy()
    out_table["viable_ns70"] = out_table["NS_mean"] >= NS_THRESHOLD
    out_table["is_dominant"] = out_table.apply(
        lambda r: dominant_lookup.get(r["operational_profile"]) == r["policy"], axis=1
    )
    csv_path = out_dir / "profile_policy_heatmap_simplified.csv"
    out_table.to_csv(csv_path, index=False)
    log.info(f"Tabela salva: {csv_path}")

    return {"png": png_path, "pdf": pdf_path, "csv": csv_path, "policies_shown": policies}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Diagrama do pipeline de evidência do AIPE
# ─────────────────────────────────────────────────────────────────────────────

def plot_aipe_evidence_pipeline(out_dir: Path) -> dict:
    """Diagrama estatico (nao orientado a dados) da cadeia:
    Dados -> caracteristicas -> POD -> simulacao -> rotulo -> PSE -> recomendacao.
    Mensagem: 'POD classifica. Simulação avalia. Rótulo escolhe. PSE recomenda.'
    """
    steps = [
        ("Dados\npreparados", "Séries\nloja-produto"),
        ("Características +\nsinais de previsão", "ADI, CV²,\nburstiness; MASE/\nsMAPE (auxiliares)"),
        ("POD", "Perfil Operacional\nde Demanda\n(classifica)"),
        ("Ambiente de\nsimulação", "12 políticas\ncandidatas\n(avalia)"),
        ("Rótulo", "Política viável\nde menor CTI\n(escolhe)"),
        ("PSE", "Meta-modelo\nsupervisionado\n(recomenda)"),
        ("Recomendação", "Política sugerida\npara nova série"),
    ]

    n = len(steps)
    fig, ax = plt.subplots(figsize=SLIDE_FIGSIZE)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h = 0.80, 0.62
    y_center = 0.56

    for i, (title, subtitle) in enumerate(steps):
        x_center = i + 0.5
        is_core = title in ("POD", "Rótulo", "PSE")
        face = COLOR_DARKBLUE if is_core else "white"
        edge = COLOR_DARKBLUE
        text_color = "white" if is_core else COLOR_DARKBLUE

        box = FancyBboxPatch(
            (x_center - box_w / 2, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x_center, y_center + 0.18, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=text_color, zorder=4)
        ax.text(x_center, y_center - 0.10, subtitle, ha="center", va="center",
                fontsize=8.6, color=text_color, zorder=4, linespacing=1.5)

        if i < n - 1:
            arrow = FancyArrowPatch(
                (x_center + box_w / 2, y_center), (x_center + 1 - box_w / 2, y_center),
                arrowstyle="-|>", mutation_scale=18, linewidth=1.8,
                color=COLOR_TEAL, zorder=2,
            )
            ax.add_patch(arrow)

    # Realimentacao periodica (Δt ciclos), do final ao inicio do nucleo
    fb = FancyArrowPatch(
        (n - 0.5, y_center - box_h / 2 - 0.02), (2.5, y_center - box_h / 2 - 0.02),
        connectionstyle="arc3,rad=-0.32", arrowstyle="-|>", mutation_scale=16,
        linewidth=1.3, linestyle="--", color=COLOR_GRAY, zorder=1,
    )
    ax.add_patch(fb)
    ax.text((n - 0.5 + 2.5) / 2, y_center - box_h / 2 - 0.13,
            "reavaliação periódica (Δt ciclos): recalcula características e POD",
            ha="center", fontsize=9, color=COLOR_GRAY, style="italic")

    ax.text(
        0.5, 0.94, "Evidência empírica do AIPE: do dado à recomendação",
        transform=ax.transAxes, ha="center", fontsize=17, fontweight="bold",
        color=COLOR_DARKBLUE,
    )

    fig.text(
        0.5, 0.015,
        "POD classifica. Simulação avalia. Rótulo escolhe. PSE recomenda.",
        ha="center", fontsize=12.5, fontweight="bold",
        color=COLOR_TEAL, style="italic",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png_path = out_dir / "aipe_evidence_pipeline.png"
    pdf_path = out_dir / "aipe_evidence_pipeline.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {png_path}")

    return {"png": png_path, "pdf": pdf_path}


# ─────────────────────────────────────────────────────────────────────────────
# 6. (Opcional) Exemplos de séries Lumpy
# ─────────────────────────────────────────────────────────────────────────────

def plot_lumpy_series_examples(scenarios: pd.DataFrame, profiles: pd.DataFrame,
                                out_dir: Path, n_examples: int = 3) -> dict:
    """Seleciona series proximas da mediana conjunta de (ADI, CV2) do quadrante
    Lumpy (regra registrada, sem escolha manual) e plota demanda x ciclo."""
    if scenarios is None:
        log.warning("scenarios.parquet indisponível — pulando lumpy_series_examples.")
        return {}

    prof = profiles.copy()
    prof["sb_quadrant"] = [_sb_quadrant(a, c) for a, c in zip(prof["adi"], prof["cv2"])]
    lumpy = prof[prof["sb_quadrant"] == "Lumpy"].copy()
    if lumpy.empty:
        log.warning("Nenhuma série Lumpy encontrada — pulando lumpy_series_examples.")
        return {}

    med_adi, med_cv2 = lumpy["adi"].median(), lumpy["cv2"].median()
    adi_norm = (lumpy["adi"] - lumpy["adi"].min()) / (lumpy["adi"].max() - lumpy["adi"].min() + 1e-9)
    cv2_norm = (lumpy["cv2"] - lumpy["cv2"].min()) / (lumpy["cv2"].max() - lumpy["cv2"].min() + 1e-9)
    med_adi_n = (med_adi - lumpy["adi"].min()) / (lumpy["adi"].max() - lumpy["adi"].min() + 1e-9)
    med_cv2_n = (med_cv2 - lumpy["cv2"].min()) / (lumpy["cv2"].max() - lumpy["cv2"].min() + 1e-9)

    lumpy["dist_to_median"] = np.sqrt((adi_norm - med_adi_n) ** 2 + (cv2_norm - med_cv2_n) ** 2)
    selected = lumpy.sort_values(["dist_to_median", "store_id"]).head(n_examples)

    fig, axes = plt.subplots(1, len(selected), figsize=(SLIDE_FIGSIZE[0], 4.6), sharey=False)
    if len(selected) == 1:
        axes = [axes]

    for ax, row in zip(axes, selected.itertuples()):
        sub = scenarios[
            (scenarios["warehouse"] == row.warehouse)
            & (scenarios["store_id"] == row.store_id)
            & (scenarios["item_id"] == row.item_id)
        ].sort_values("venda_ciclo")
        cycles = np.arange(1, len(sub) + 1)
        ax.bar(cycles, sub["demand"].values, color=COLOR_PETROL, width=0.8)
        ax.set_title(
            f"Loja {row.store_id} · Produto {row.item_id}\n"
            f"ADI={row.adi:.2f}  CV²={row.cv2:.2f}",
            fontsize=11, color=COLOR_DARKBLUE,
        )
        ax.set_xlabel("Ciclo comercial")
        ax.set_ylabel("Demanda (un.)")

    fig.suptitle(
        "Exemplos de séries Lumpy: muitos ciclos sem demanda e picos irregulares",
        fontsize=15, fontweight="bold", color=COLOR_DARKBLUE, y=1.04,
    )
    fig.text(
        0.5, -0.04,
        "Critério: 3 séries Lumpy mais próximas da mediana conjunta normalizada de "
        "(ADI, CV²) — sem seleção manual.",
        ha="center", fontsize=9.5, color=COLOR_GRAY,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    png_path = out_dir / "lumpy_series_examples.png"
    pdf_path = out_dir / "lumpy_series_examples.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {png_path}")

    csv_path = out_dir / "lumpy_series_examples.csv"
    selected[["warehouse", "store_id", "item_id", "adi", "cv2", "dist_to_median"]].to_csv(
        csv_path, index=False
    )
    log.info(f"Sumário salvo: {csv_path}")

    return {"png": png_path, "pdf": pdf_path, "csv": csv_path, "selected": selected}


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(entries: list, out_dir: Path) -> Path:
    manifest = [
        {
            "filename": e.filename,
            "description": e.description,
            "source_data": e.source_data,
            "generator": e.generator,
            "generated_at": e.generated_at,
            "notes": e.notes,
        }
        for e in entries
    ]
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Manifest salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Validação
# ─────────────────────────────────────────────────────────────────────────────

def validate_outputs(data: dict, sb_result: dict, tradeoff_result: dict,
                      dominance_result: dict, out_dir: Path) -> Path:
    alerts = []
    checks = []

    profiles = data["profiles"]
    n_series = profiles[["store_id", "item_id"]].drop_duplicates().shape[0]
    checks.append(f"Total de séries (demand_profiles.parquet): **{n_series}**"
                  + (" ✅ (== 145)" if n_series == 145 else " ⚠ (esperado 145)"))
    if n_series != 145:
        alerts.append(f"Total de séries = {n_series}, divergente do esperado (145).")

    pod_counts = profiles["operational_profile"].value_counts()
    checks.append("Distribuição por POD: " + ", ".join(
        f"{PROFILE_DISPLAY.get(k, k)}={v}" for k, v in pod_counts.items()
    ))
    if pod_counts.sum() != n_series:
        alerts.append("Soma das séries por POD não bate com o total de séries.")

    lumpy_pct = sb_result["lumpy_pct"]
    checks.append(f"Séries no quadrante Lumpy (recomputado de ADI/CV² em "
                   f"demand_profiles.parquet): **{lumpy_pct:.1f}%** ({sb_result['summary'].loc[sb_result['summary']['quadrante']=='Lumpy','n_series'].iloc[0]}/{sb_result['n_total']})")
    if abs(lumpy_pct - 71.0) > 1.0:
        alerts.append(
            "ALERTA: a Tabela 'Características do dataset por experimento' "
            "(docs/master_proposal/capitulos/resultados.tex, Seção 'Visão Geral') "
            "reporta 71% de séries Lumpy para o Experimento 2 (BA). Os artefatos "
            "atuais (demand_profiles.parquet e scenarios_meta.parquet, ambos com "
            f"145 séries) mostram {lumpy_pct:.1f}% (coluna 'group' e quadrante "
            "recomputado de ADI/CV² concordam). Não foi encontrada nenhuma fonte "
            "de dados atual que sustente 71%. A própria Seção 'Experimento 2' do "
            "Capítulo de Resultados afirma que as 145 séries foram 'classificadas "
            "no quadrante Lumpy' (ou seja, 100%), o que é consistente com os "
            "dados, mas contradiz a tabela-resumo da Seção 'Visão Geral'. "
            "Recomenda-se revisar/corrigir essa tabela na dissertação; nenhuma "
            "alteração foi feita na dissertação ou na apresentação nesta tarefa."
        )

    if sb_result.get("mismatch_group_vs_quadrant"):
        alerts.append(
            f"{sb_result['mismatch_group_vs_quadrant']} série(s) com coluna "
            "'group' divergente do quadrante recomputado a partir de adi/cv2 "
            "em demand_profiles.parquet."
        )

    # Coluna 'dominant_policy' de demand_profiles.parquet é um lookup estático
    # (Tabela 4.1, política de referência por POD), NÃO o resultado empírico do
    # benchmark. Registrar para evitar uso indevido em trabalhos futuros.
    if "dominant_policy" in profiles.columns:
        ref_lookup = {"Sparse_High_Impact": "GA-DQN", "Unstable_Trend": "GA-PPO",
                      "High_Vol_Seasonal": "PPO"}
        matches_lookup = all(
            (profiles.loc[profiles["operational_profile"] == pod, "dominant_policy"] == pol).all()
            for pod, pol in ref_lookup.items()
            if pod in profiles["operational_profile"].unique()
        )
        if matches_lookup:
            alerts.append(
                "ALERTA (uso indevido evitado): a coluna 'dominant_policy' em "
                "demand_profiles.parquet é idêntica, por POD, à 'política de "
                "referência inicial' heurística da Tabela 4.1 da dissertação "
                "(GA-DQN/GA-PPO/PPO), e NÃO ao resultado empírico do benchmark "
                "(SA/EOQ/EOQ, ver profiles/dominant_policy_by_profile.csv). Esta "
                "coluna NÃO foi usada em nenhuma das novas visualizações; o "
                "'oráculo por série' usado no gráfico de trade-off vem de "
                "strategy/strategy_cost_comparison.csv (estratégia C), que aplica "
                "corretamente a restrição NS≥0,70 por série."
            )

    # Estratégia
    red_pct_b = tradeoff_result["red_pct_b"]
    checks.append(f"Redução de CTI da estratégia B vs. política única: **{red_pct_b:+.2f}%**"
                   + (" ✅ (≈6,2%)" if abs(red_pct_b - 6.2) < 0.5 else " ⚠ (esperado ≈6,2%)"))
    if abs(red_pct_b - 6.2) > 0.5:
        alerts.append(f"Redução de CTI da estratégia B = {red_pct_b:.2f}%, diverge de ≈6,2% citado na dissertação.")

    strategy_df = data["strategy_comparison"]
    c_row = strategy_df[strategy_df["estrategia"] == "C"]
    if not c_row.empty:
        checks.append(f"Estratégia C (oráculo por série) marcada como exploratória na "
                       f"figura e descrita como '{c_row['descricao'].iloc[0]}' no artefato fonte. ✅")

    # Jornaleiro nunca deve ser dominante com NS < 0.70
    dom = data["dominant_by_profile"]
    jorn_dominant = dom[dom["dominant_policy"] == "Newsvendor"]
    if not jorn_dominant.empty and (jorn_dominant["NS_mean"] < NS_THRESHOLD).any():
        alerts.append("Jornaleiro aparece como dominante com NS médio < 0,70 em algum perfil.")
    else:
        checks.append("Jornaleiro não aparece como política dominante em nenhum perfil "
                       "com NS médio < 0,70. ✅")

    profile_metrics = data["profile_metrics"]
    n_after_filters = profile_metrics["n_series"].drop_duplicates().sum() if False else None
    sum_n_profiles = dom["n_series"].sum()
    checks.append(f"Soma de séries nos perfis com representação no Experimento 2: "
                   f"**{sum_n_profiles}**"
                   + (" ✅ (== 145)" if sum_n_profiles == 145 else " ⚠"))

    lines = [
        "# Validação dos novos artefatos de apresentação",
        "",
        f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Fontes usadas",
        f"- `{PROF_PATH.relative_to(REPO_ROOT)}`",
        f"- `{PROFILE_METRICS_PATH.relative_to(REPO_ROOT)}`",
        f"- `{DOMINANT_PROFILE_PATH.relative_to(REPO_ROOT)}`",
        f"- `{STRATEGY_COMPARISON_PATH.relative_to(REPO_ROOT)}`",
        f"- `{SCENARIOS_PATH.relative_to(REPO_ROOT)}`" if data.get("scenarios") is not None else
        f"- `{SCENARIOS_PATH.relative_to(REPO_ROOT)}` (não encontrado — exemplos Lumpy pulados)",
        "",
        "## Checagens",
        "",
    ] + [f"- {c}" for c in checks] + [
        "",
        "## Alertas / divergências",
        "",
    ]

    if alerts:
        lines += [f"- {a}" for a in alerts]
    else:
        lines += ["- Nenhuma divergência encontrada."]

    lines += [
        "",
        "## Resultado científico",
        "",
        "Nenhum número de CTI, NS, TR, BE, FP ou estratégia foi recalculado ou "
        "alterado nesta tarefa. Todas as figuras leem diretamente os artefatos "
        "já existentes em `data/08_reporting/profiles/` e `data/08_reporting/strategy/`, "
        "ou recomputam apenas características já presentes em `demand_profiles.parquet` "
        "(ADI, CV², quadrante Syntetos-Boylan), sem reexecutar a simulação.",
    ]

    path = out_dir / "figures_validation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Validação salva: {path}")
    return path, alerts


# ─────────────────────────────────────────────────────────────────────────────
# README do diretório de saída
# ─────────────────────────────────────────────────────────────────────────────

def write_readme(out_dir: Path, alerts: list) -> Path:
    lines = [
        "# Artefatos visuais para a apresentação de qualificação",
        "",
        "Gerados por `simulation/src/reporting/presentation_visuals.py`, a partir "
        "dos artefatos já existentes em `data/08_reporting/profiles/` e "
        "`data/08_reporting/strategy/`, e de `data/04_feature/demand_profiles.parquet` "
        "/ `data/03_primary/scenarios.parquet`. Nenhum resultado de simulação foi "
        "recalculado ou alterado.",
        "",
        "## Como reproduzir",
        "",
        "```bash",
        "cd simulation",
        "python src/reporting/presentation_visuals.py",
        "# ou, via Kedro (node registrado no pipeline de reporting):",
        "kedro run --pipeline reporting --nodes generate_presentation_visuals",
        "```",
        "",
        "## Arquivos",
        "",
        "| Arquivo | Descrição |",
        "|---|---|",
        "| `syntetos_boylan_scatter.png/.pdf/.csv` | Plano ADI×CV² com os 4 quadrantes, destacando Lumpy |",
        "| `strategy_tradeoff_cti_ns.png/.pdf/.csv` | CTI médio e NS médio por estratégia (A1/A2/B/C) |",
        "| `profile_dominance_bars.png/.pdf/.csv` | Política dominante por perfil operacional |",
        "| `profile_policy_heatmap_simplified.png/.pdf/.csv` | Heatmap CTI restrito a políticas prioritárias + viabilidade NS≥0,70 |",
        "| `aipe_evidence_pipeline.png/.pdf` | Diagrama dados → características → POD → simulação → rótulo → PSE → recomendação |",
        "| `lumpy_series_examples.png/.pdf/.csv` | 3 séries Lumpy próximas da mediana ADI/CV² (opcional) |",
        "| `manifest.json` | Metadados de proveniência de cada artefato |",
        "| `figures_validation.md` | Checagens numéricas e alertas de divergência |",
        "",
        "## Pendências / observações",
        "",
    ]
    if alerts:
        lines += [f"- {a}" for a in alerts]
    else:
        lines.append("- Nenhuma pendência identificada.")
    lines += [
        "",
        "## Integração com a apresentação (etapa futura)",
        "",
        "Esta tarefa **não** alterou `docs/qualification_presentation/`. Sugestões "
        "de substituição/complemento de slides estão no relatório da tarefa que "
        "gerou estes artefatos (não persistido neste diretório).",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"README salvo: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_reporting_data()

    entries = []

    sb_result = plot_syntetos_boylan_plane(data["profiles"], OUT_DIR)
    entries.append(ManifestEntry(
        filename="syntetos_boylan_scatter.png/.pdf/.csv",
        description="Plano ADI x CV2 (Syntetos-Boylan) com os 4 quadrantes; "
                     "cada ponto é uma série loja-produto; quadrante Lumpy destacado.",
        source_data="data/04_feature/demand_profiles.parquet (colunas adi, cv2, group)",
        generator="plot_syntetos_boylan_plane",
        notes=f"{sb_result['lumpy_pct']:.1f}% das séries classificadas como Lumpy "
              "no recorte avaliado (recomputado de adi/cv2; ver figures_validation.md "
              "para divergência com a tabela da dissertação que cita 71%).",
    ))

    tradeoff_result = plot_strategy_tradeoff(data["strategy_comparison"], OUT_DIR)
    entries.append(ManifestEntry(
        filename="strategy_tradeoff_cti_ns.png/.pdf/.csv",
        description="CTI médio e NS médio por estratégia (A1/A2 política única, "
                     "B seleção por perfil, C oráculo por série exploratório).",
        source_data="data/08_reporting/strategy/strategy_cost_comparison.csv",
        generator="plot_strategy_tradeoff",
        notes="C (oráculo) explicitamente rotulado como referência exploratória, "
              "não estratégia operacional.",
    ))

    dominance_result = plot_profile_dominance(data["dominant_by_profile"], OUT_DIR)
    entries.append(ManifestEntry(
        filename="profile_dominance_bars.png/.pdf/.csv",
        description="Política dominante (menor CTI viável, NS>=0.70) por Perfil "
                     "Operacional de Demanda, com indicação de n e evidência exploratória (n<20).",
        source_data="data/08_reporting/profiles/dominant_policy_by_profile.csv",
        generator="plot_profile_dominance",
    ))

    heatmap_result = plot_policy_profile_heatmap_simplified(
        data["profile_metrics"], data["dominant_by_profile"], OUT_DIR
    )
    entries.append(ManifestEntry(
        filename="profile_policy_heatmap_simplified.png/.pdf/.csv",
        description="Heatmap de CTI médio (perfil x política), restrito a um "
                     "subconjunto legível de políticas + dominantes; hachura = "
                     "inviável (NS<0.70); contorno = política dominante do perfil.",
        source_data="data/08_reporting/profiles/profile_policy_metrics.csv, "
                     "data/08_reporting/profiles/dominant_policy_by_profile.csv",
        generator="plot_policy_profile_heatmap_simplified",
        notes=f"Políticas exibidas: {', '.join(heatmap_result.get('policies_shown', []))}.",
    ))

    aipe_result = plot_aipe_evidence_pipeline(OUT_DIR)
    entries.append(ManifestEntry(
        filename="aipe_evidence_pipeline.png/.pdf",
        description="Diagrama ilustrativo (não orientado a dados) da cadeia "
                     "Dados -> Características -> POD -> Simulação -> Rótulo -> PSE -> Recomendação.",
        source_data="N/A (diagrama conceitual)",
        generator="plot_aipe_evidence_pipeline",
        notes="POD classifica; simulação avalia; rótulo escolhe; PSE recomenda "
              "(não 'decide'). XGBoost não é citado no diagrama.",
    ))

    lumpy_result = plot_lumpy_series_examples(data.get("scenarios"), data["profiles"], OUT_DIR)
    if lumpy_result:
        entries.append(ManifestEntry(
            filename="lumpy_series_examples.png/.pdf/.csv",
            description="3 séries Lumpy mais próximas da mediana conjunta normalizada "
                         "de (ADI, CV²) — critério registrado, sem escolha manual.",
            source_data="data/03_primary/scenarios.parquet, "
                         "data/04_feature/demand_profiles.parquet",
            generator="plot_lumpy_series_examples",
        ))

    write_manifest(entries, OUT_DIR)
    _, alerts = validate_outputs(data, sb_result, tradeoff_result, dominance_result, OUT_DIR)
    write_readme(OUT_DIR, alerts)

    log.info("Visualizações de apresentação concluídas.")
    log.info(f"Artefatos em: {OUT_DIR}")
    return {
        "out_dir": OUT_DIR,
        "alerts": alerts,
        "sb_result": sb_result,
        "tradeoff_result": tradeoff_result,
        "dominance_result": dominance_result,
        "heatmap_result": heatmap_result,
        "aipe_result": aipe_result,
        "lumpy_result": lumpy_result,
    }


if __name__ == "__main__":
    run()
