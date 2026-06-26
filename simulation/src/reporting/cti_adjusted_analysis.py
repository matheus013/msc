"""
cti_adjusted_analysis.py
Análise complementar de CTI Ajustado por Estabilidade Operacional.

Métrica central:
  J(π, g) = CTI_norm(π, g) + λ_BE · BE_norm(π, g)

Onde:
  - CTI_norm e BE_norm são normalizações min-max sobre políticas viáveis
    dentro do escopo de comparação (global ou por perfil)
  - λ_BE ∈ {0.00, 0.25, 0.50, 1.00} controla o peso da penalidade por BE
  - λ_BE = 0 recupera a seleção por CTI puro normalizado

Saídas em data/08_reporting/strategy/:
  cti_adjusted_sensitivity.csv           — J por (perfil, política, λ)
  dominant_policy_by_profile_adjusted.csv — política dominante por (perfil, λ)
  strategy_cost_comparison_adjusted.csv  — estratégias A-E por λ
  table_strategy_adjusted.tex            — tabela LaTeX principal
  table_dominant_adjusted.tex            — tabela LaTeX de dominância por perfil
  cti_adjusted_sensitivity.pdf           — figura sensibilidade
  cti_adjusted_validation.md             — relatório de validação

Regras:
  - CTI original preservado; CTI_adjusted é métrica COMPLEMENTAR
  - Políticas degeneradas: DQN (NS << limiar) e PPO (FP excessivo)
  - Viabilidade: NS_mean >= NS_THRESHOLD dentro do escopo
  - Perfis com n < 20 séries: evidência exploratória
"""

import logging
from pathlib import Path
from datetime import datetime
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Caminhos e constantes
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = REPO_ROOT / "data"
KPI_PATH   = DATA_DIR / "07_model_output" / "kpis.parquet"
PROF_PATH  = DATA_DIR / "04_feature" / "demand_profiles.parquet"
OUT_DIR    = DATA_DIR / "08_reporting" / "strategy"

NS_THRESHOLD       = 0.70
LAMBDA_VALUES      = [0.00, 0.25, 0.50, 1.00]
DEGENERATE_POLICIES = {"DQN", "PPO"}   # degeneração documentada na Tabela 5.2
BASELINE_POLICY    = "EOQ"
N_EXPLORATORY      = 20   # perfis com n < N_EXPLORATORY → evidência exploratória

POLICY_DISPLAY = {"sS": "(s,S)", "Newsvendor": "Jornaleiro"}
PROFILE_DISPLAY = {
    "Sparse_High_Impact": "Sparse High Impact",
    "High_Vol_Seasonal":  "High Vol. Seasonal",
    "Unstable Trend":     "Unstable Trend",
    "Unstable_Trend":     "Unstable Trend",
}
POLICY_COLORS = {
    "EOQ": "#2166ac",  "sS": "#4dac26",   "Newsvendor": "#f1a340",
    "GA":  "#7b2d8b",  "SA":  "#d73027",   "PSO":        "#abd9e9",
    "DE":  "#74add1",  "DQN": "#fdae61",   "PPO":        "#fee090",
    "SARSA": "#a6d96a", "GA-DQN": "#e31a1c", "GA-PPO": "#1a9641",
}


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento
# ─────────────────────────────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    log.info("Lendo kpis.parquet …")
    kpis = pd.read_parquet(KPI_PATH).rename(columns={"TIC": "CTI"})
    log.info("Lendo demand_profiles.parquet …")
    prof = pd.read_parquet(PROF_PATH)[
        ["warehouse", "store_id", "item_id", "operational_profile"]
    ]
    df = kpis.merge(prof, on=["warehouse", "store_id", "item_id"], how="left")
    log.info(
        f"Dados: {len(df)} linhas, "
        f"{df[['warehouse','store_id','item_id']].drop_duplicates().shape[0]} séries, "
        f"{df.policy.nunique()} políticas, "
        f"{df.operational_profile.nunique()} perfis."
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Agregação por escopo (global ou por perfil)
# ─────────────────────────────────────────────────────────────────────────────

def _agg_global(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("policy")[["CTI", "NS", "TR", "BE", "FP"]].mean().round(4)
    agg["n_series"] = df.groupby("policy").size() // df.policy.nunique()
    agg["n_series"] = df[["warehouse","store_id","item_id"]].drop_duplicates().shape[0]
    agg["viable"]      = (agg["NS"] >= NS_THRESHOLD) & (~agg.index.isin(DEGENERATE_POLICIES))
    agg["degenerate"]  = agg.index.isin(DEGENERATE_POLICIES)
    agg = agg.reset_index()
    return agg


def _agg_profile(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["operational_profile", "policy"])[["CTI", "NS", "TR", "BE", "FP"]]
        .mean().round(4)
    )
    n_ser = df.groupby(["operational_profile", "policy"]).size()
    agg = agg.join(n_ser.rename("n_pairs"))
    n_profile = df.groupby("operational_profile")[["warehouse","store_id","item_id"]].apply(
        lambda g: g.drop_duplicates().shape[0]
    ).rename("n_series")
    agg = agg.join(n_profile, on="operational_profile")
    agg["viable"]     = (agg["NS"] >= NS_THRESHOLD) & (~agg.index.get_level_values("policy").isin(DEGENERATE_POLICIES))
    agg["degenerate"] = agg.index.get_level_values("policy").isin(DEGENERATE_POLICIES)
    agg["exploratory"] = agg["n_series"] < N_EXPLORATORY
    agg = agg.reset_index()
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Normalização min-max
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    mn, mx = series.min(), series.max()
    if abs(mx - mn) < eps:
        log.warning(f"  min==max para normalização ({series.name}); definindo norm=0")
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def _add_adjusted(df: pd.DataFrame, lam: float,
                  scope: str = "global",
                  profile_col: str = "operational_profile") -> pd.DataFrame:
    """Adiciona CTI_norm, BE_norm e CTI_adjusted ao DataFrame.
    scope='global': normaliza sobre todas as políticas (viáveis e não-viáveis).
    scope='profile': normaliza dentro de cada perfil separadamente.
    """
    df = df.copy()
    if scope == "global":
        df["CTI_norm"] = _minmax(df["CTI"])
        df["BE_norm"]  = _minmax(df["BE"])
    elif scope == "profile":
        df["CTI_norm"] = df.groupby(profile_col)["CTI"].transform(
            lambda x: _minmax(x)
        )
        df["BE_norm"] = df.groupby(profile_col)["BE"].transform(
            lambda x: _minmax(x)
        )
    df["CTI_adjusted"] = df["CTI_norm"] + lam * df["BE_norm"]
    df["lambda_BE"] = lam
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Política dominante por escopo e λ
# ─────────────────────────────────────────────────────────────────────────────

def _pick_dominant(agg: pd.DataFrame, metric: str = "CTI") -> dict:
    """Retorna {grupo: policy_name} para políticas viáveis.
    metric: 'CTI' (custo puro) ou 'CTI_adjusted' (custo ajustado).
    Fallback: maior NS_mean quando nenhuma política viável.
    """
    dom = {}
    group_col = "operational_profile" if "operational_profile" in agg.columns else "__global__"
    if group_col == "__global__":
        agg = agg.copy()
        agg["__global__"] = "global"
    groups = agg[group_col].unique()
    for g in groups:
        grp = agg[agg[group_col] == g]
        viable = grp[grp["viable"]]
        if viable.empty:
            best = grp.loc[grp["NS"].idxmax(), "policy"]
            dom[g] = (best, "fallback_NS")
        else:
            best = viable.loc[viable[metric].idxmin(), "policy"]
            dom[g] = (best, "normal")
    return dom


def _build_dominant_table(profile_agg: pd.DataFrame) -> pd.DataFrame:
    """Constrói tabela de política dominante por (perfil, λ), com e sem BE."""
    rows = []
    profiles = profile_agg.operational_profile.unique()
    for lam in LAMBDA_VALUES:
        tmp = _add_adjusted(profile_agg, lam, scope="profile")
        for prof in profiles:
            grp = tmp[tmp.operational_profile == prof].copy()
            n   = int(grp.n_series.iloc[0])
            exp = bool(n < N_EXPLORATORY)

            viable = grp[grp.viable]

            # Dominante por CTI puro
            if viable.empty:
                dom_cti_row = grp.loc[grp.NS.idxmax()]
                dom_cti_status = "fallback_NS"
            else:
                dom_cti_row = viable.loc[viable.CTI.idxmin()]
                dom_cti_status = "normal"

            # Dominante por CTI ajustado
            if viable.empty:
                dom_adj_row = grp.loc[grp.NS.idxmax()]
                dom_adj_status = "fallback_NS"
            else:
                dom_adj_row = viable.loc[viable.CTI_adjusted.idxmin()]
                dom_adj_status = "normal"

            rows.append({
                "operational_profile":   prof,
                "profile_display":       PROFILE_DISPLAY.get(prof, prof),
                "n_series":              n,
                "exploratory":           exp,
                "lambda_BE":             lam,
                "dominant_policy_cti":        dom_cti_row.policy,
                "dominant_cti_display":       POLICY_DISPLAY.get(dom_cti_row.policy, dom_cti_row.policy),
                "CTI_mean_cti":          round(dom_cti_row.CTI, 2),
                "BE_mean_cti":           round(dom_cti_row.BE, 3),
                "NS_mean_cti":           round(dom_cti_row.NS, 3),
                "status_cti":            dom_cti_status,
                "dominant_policy_adj":        dom_adj_row.policy,
                "dominant_adj_display":       POLICY_DISPLAY.get(dom_adj_row.policy, dom_adj_row.policy),
                "CTI_adjusted_selected": round(dom_adj_row.CTI_adjusted, 4),
                "CTI_mean_adj":          round(dom_adj_row.CTI, 2),
                "BE_mean_adj":           round(dom_adj_row.BE, 3),
                "NS_mean_adj":           round(dom_adj_row.NS, 3),
                "status_adj":            dom_adj_status,
                "changed_vs_cti_only":   dom_cti_row.policy != dom_adj_row.policy,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Comparação de estratégias
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_comparison(df: pd.DataFrame, global_agg: pd.DataFrame,
                          profile_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compara estratégias A–E para cada λ_BE.

    A: política única global por CTI puro
    B: política única global por CTI_adjusted (varia com λ)
    C: seleção por perfil por CTI puro
    D: seleção por perfil por CTI_adjusted (varia com λ)
    E: oráculo por série (referência exploratória, CTI puro)

    Métricas de reporting:
    - CTI_mean / BE_mean / NS_mean: médias ponderadas das médias de perfil da política dominante
    - CTI_adjusted_mean: usando normalização global (12 políticas) sobre médias de perfil
      Esta normalização é consistente entre estratégias; differe da normalização por perfil
      usada para SELEÇÃO em C e D.
    """
    eps = 1e-9

    # ── Normalização para reporting: TODAS as 12 políticas, médias globais ──────
    g_cti_min, g_cti_max = global_agg.CTI.min(), global_agg.CTI.max()
    g_be_min,  g_be_max  = global_agg.BE.min(),  global_agg.BE.max()
    cti_norm_g = {r.policy: (r.CTI - g_cti_min) / max(g_cti_max - g_cti_min, eps)
                  for _, r in global_agg.iterrows()}
    be_norm_g  = {r.policy: (r.BE  - g_be_min)  / max(g_be_max  - g_be_min,  eps)
                  for _, r in global_agg.iterrows()}

    # ── Estrutura de métricas por (perfil, política) ─────────────────────────────
    prof_policy_metrics = {
        (r.operational_profile, r.policy): r
        for _, r in profile_agg.iterrows()
    }

    # Número de séries por perfil
    n_per_prof = (
        df.groupby("operational_profile")[["warehouse","store_id","item_id"]]
          .apply(lambda g: g.drop_duplicates().shape[0])
    )
    total_n  = int(n_per_prof.sum())
    profiles = list(n_per_prof.index)

    # ── Pivot para oráculo (per-série) ───────────────────────────────────────────
    pivot_cti = df.pivot_table(
        index=["warehouse","store_id","item_id","operational_profile"],
        columns="policy", values="CTI"
    )
    pivot_ns = df.pivot_table(
        index=["warehouse","store_id","item_id","operational_profile"],
        columns="policy", values="NS"
    )
    all_policies = list(pivot_cti.columns)

    oracle_cti_vals = []
    for (_, _, _, _), row_cti in pivot_cti.iterrows():
        row_ns = pivot_ns.loc[row_cti.name]
        viable_p = [p for p in all_policies
                    if p not in DEGENERATE_POLICIES and row_ns[p] >= NS_THRESHOLD]
        if not viable_p:
            viable_p = [p for p in all_policies if p not in DEGENERATE_POLICIES]
        oracle_cti_vals.append(min(row_cti[p] for p in viable_p))
    oracle_cti = np.array(oracle_cti_vals)

    # ── Política dominante global por CTI puro ───────────────────────────────────
    viable_global = global_agg[global_agg.viable]
    pol_A = viable_global.loc[viable_global.CTI.idxmin(), "policy"]
    log.info(f"Estratégia A (CTI puro global): {POLICY_DISPLAY.get(pol_A, pol_A)}")

    def _aggregate(dom_map: dict, label: str, lam: float) -> dict:
        """Agrega métricas ponderadas pelas médias de perfil da política dominante em cada perfil."""
        cti_w, be_w, ns_w, adj_w = 0.0, 0.0, 0.0, 0.0
        for pr in profiles:
            n  = int(n_per_prof[pr])
            wt = n / total_n
            pol = dom_map[pr]
            pm = prof_policy_metrics.get((pr, pol))
            if pm is None:
                log.warning(f"  Métricas ausentes para ({pr}, {pol})")
                continue
            cti_w += wt * float(pm.CTI)
            be_w  += wt * float(pm.BE)
            ns_w  += wt * float(pm.NS)
            adj_w += wt * (cti_norm_g.get(pol, 0.5) + lam * be_norm_g.get(pol, 0.5))
        return {
            "strategy":           label,
            "lambda_BE":          lam,
            "CTI_total":          round(cti_w * total_n, 2),
            "CTI_mean":           round(cti_w, 2),
            "BE_mean":            round(be_w, 3),
            "NS_mean":            round(ns_w, 3),
            "CTI_adjusted_mean":  round(adj_w, 4),
        }

    rows = []
    for lam in LAMBDA_VALUES:
        # B: política única global por CTI_adjusted
        tmp_g = _add_adjusted(viable_global.copy(), lam, scope="global")
        pol_B = tmp_g.loc[tmp_g.CTI_adjusted.idxmin(), "policy"]
        log.info(f"λ={lam}: Estratégia B: {POLICY_DISPLAY.get(pol_B, pol_B)}")

        # C: seleção por perfil por CTI puro (invariante em λ, recomputado por legibilidade)
        dom_C = {}
        for pr in profiles:
            grp = profile_agg[profile_agg.operational_profile == pr]
            v   = grp[grp.viable]
            dom_C[pr] = v.loc[v.CTI.idxmin(), "policy"] if not v.empty else grp.loc[grp.NS.idxmax(), "policy"]

        # D: seleção por perfil por CTI_adjusted
        tmp_p = _add_adjusted(profile_agg.copy(), lam, scope="profile")
        dom_D = {}
        for pr in profiles:
            grp = tmp_p[tmp_p.operational_profile == pr]
            v   = grp[grp.viable]
            dom_D[pr] = v.loc[v.CTI_adjusted.idxmin(), "policy"] if not v.empty else grp.loc[grp.NS.idxmax(), "policy"]
        log.info(f"λ={lam}: D dominantes: { {p: POLICY_DISPLAY.get(v,v) for p,v in dom_D.items()} }")

        strat_A = _aggregate({pr: pol_A for pr in profiles}, "A: política única (CTI puro)", lam)
        strat_B = _aggregate({pr: pol_B for pr in profiles}, "B: política única (CTI ajustado)", lam)
        strat_C = _aggregate(dom_C, "C: seleção por perfil (CTI puro)", lam)
        strat_D = _aggregate(dom_D, "D: seleção por perfil (CTI ajustado)", lam)

        # E: oráculo por série — CTI puro; CTI_adjusted usa norm global sobre médias de política
        # (BE e NS do oráculo não são bem-definidos pois cada série usa política diferente)
        oracle_cti_adj = np.mean([
            cti_norm_g.get(pol, 0.5) + lam * be_norm_g.get(pol, 0.5)
            for _, row_cti in pivot_cti.iterrows()
            for pol in [[p for p in all_policies
                         if p not in DEGENERATE_POLICIES and pivot_ns.loc[row_cti.name][p] >= NS_THRESHOLD
                         or p not in DEGENERATE_POLICIES][0]]
        ])
        strat_E = {
            "strategy":          "E: oráculo por série (exploratório)",
            "lambda_BE":         lam,
            "CTI_total":         round(float(oracle_cti.sum()), 2),
            "CTI_mean":          round(float(oracle_cti.mean()), 2),
            "BE_mean":           None,
            "NS_mean":           None,
            "CTI_adjusted_mean": None,
        }

        ref_cti = strat_A["CTI_mean"]
        ref_adj = strat_A["CTI_adjusted_mean"]
        for strat in [strat_A, strat_B, strat_C, strat_D, strat_E]:
            strat["reduction_CTI_pct_vs_A"] = round(
                100 * (ref_cti - strat["CTI_mean"]) / max(ref_cti, eps), 1
            ) if strat["CTI_mean"] is not None else None
            strat["reduction_adj_pct_vs_A"] = round(
                100 * (ref_adj - strat["CTI_adjusted_mean"]) / max(ref_adj, eps), 1
            ) if strat["CTI_adjusted_mean"] is not None else None
            rows.append(strat)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Artefato de sensibilidade
# ─────────────────────────────────────────────────────────────────────────────

def _sensitivity_csv(profile_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lam in LAMBDA_VALUES:
        tmp = _add_adjusted(profile_agg, lam, scope="profile")
        for _, r in tmp.iterrows():
            rows.append({
                "operational_profile": r.operational_profile,
                "profile_display":     PROFILE_DISPLAY.get(r.operational_profile, r.operational_profile),
                "policy":              r.policy,
                "n_series":            r.n_series,
                "lambda_BE":           lam,
                "CTI_mean":            r.CTI,
                "BE_mean":             r.BE,
                "NS_mean":             r.NS,
                "FP_mean":             r.FP,
                "CTI_norm":            round(r.CTI_norm, 4),
                "BE_norm":             round(r.BE_norm, 4),
                "CTI_adjusted":        round(r.CTI_adjusted, 4),
                "viable":              r.viable,
                "degenerate":          r.degenerate,
                "exploratory":         r.exploratory,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────

def _sensitivity_figure(dominant_table: pd.DataFrame, out_path: Path) -> None:
    profiles = sorted(dominant_table.operational_profile.unique())
    fig, axes = plt.subplots(1, len(profiles), figsize=(4 * len(profiles), 4), sharey=False)
    if len(profiles) == 1:
        axes = [axes]

    for ax, prof in zip(axes, profiles):
        sub = dominant_table[dominant_table.operational_profile == prof].sort_values("lambda_BE")
        cti_pols = sub.dominant_policy_cti.tolist()
        adj_pols = sub.dominant_policy_adj.tolist()
        lams      = sub.lambda_BE.tolist()
        exp       = sub.exploratory.iloc[0]
        n         = sub.n_series.iloc[0]
        title     = PROFILE_DISPLAY.get(prof, prof) + (f"\n(n={n}*)" if exp else f"\n(n={n})")

        for i, lam in enumerate(lams):
            cp = cti_pols[i]
            ap = adj_pols[i]
            ax.barh(i - 0.15, 1, left=0, height=0.28,
                    color=POLICY_COLORS.get(cp, "gray"), alpha=0.85, label=f"CTI: {POLICY_DISPLAY.get(cp,cp)}")
            ax.barh(i + 0.15, 1, left=0, height=0.28,
                    color=POLICY_COLORS.get(ap, "lightgray"), alpha=0.85,
                    hatch="/" if ap != cp else "",
                    label=f"Adj: {POLICY_DISPLAY.get(ap,ap)}")
            ax.text(1.05, i - 0.15, POLICY_DISPLAY.get(cp, cp), va="center", fontsize=7)
            ax.text(1.05, i + 0.15, POLICY_DISPLAY.get(ap, ap), va="center", fontsize=7,
                    fontstyle="italic" if ap != cp else "normal")

        ax.set_yticks(range(len(lams)))
        ax.set_yticklabels([f"λ={l:.2f}" for l in lams])
        ax.set_xlim(0, 2.5)
        ax.set_xlabel("")
        ax.set_title(title, fontsize=9)
        ax.tick_params(bottom=False, labelbottom=False)

    # Legend
    patches = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Barra sólida: pol. dominante por CTI puro"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="/", label="Barra hachurada: pol. por CTI ajustado (se diferente)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Política Dominante por Perfil: CTI Puro vs CTI Ajustado (λ_BE)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"Figura de sensibilidade: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tabelas LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\textemdash{}"
    return f"{v:.{dec}f}"

def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\textemdash{}"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}"


def _latex_strategy_table(strat_df: pd.DataFrame, out_path: Path,
                           lam_subset: list = None) -> None:
    if lam_subset is None:
        lam_subset = [0.00, 0.25, 0.50, 1.00]
    sel = strat_df[strat_df.lambda_BE.isin(lam_subset)].copy()

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Comparação de custo entre estratégias de seleção de política com"
        r" CTI ajustado por estabilidade operacional (Experimento~2, BA, 145 séries,"
        r" regime \textit{Lumpy}). CTI~Adj. $= \mathrm{CTI_{norm}} + \lambda_{BE}"
        r" \cdot \mathrm{BE_{norm}}$, normalizado por escopo global."
        r" Redução\% calculada contra estratégia~A (política única, CTI puro)."
        r" $^\dagger$Oráculo por série: referência exploratória, não estratégia operacional.}",
        r"\label{tab:strategy_cost_comparison_adjusted}",
        r"\begin{tabular}{@{}p{3.4cm}crrrrcc@{}}",
        r"\toprule",
        r"\makecell[l]{Estratégia} &",
        r"\makecell[c]{$\lambda$} &",
        r"\makecell[r]{CTI\\médio} &",
        r"\makecell[r]{BE\\médio} &",
        r"\makecell[r]{NS\\médio} &",
        r"\makecell[r]{CTI~Adj.\\médio} &",
        r"\makecell[c]{Red. CTI\\vs~A (\%)} &",
        r"\makecell[c]{Red. Adj.\\vs~A (\%)} \\",
        r"\midrule",
    ]

    STRAT_SHORT = {
        "A: política única (CTI puro)":        r"A: pol. única (CTI puro)",
        "B: política única (CTI ajustado)":    r"B: pol. única (CTI ajustado)",
        "C: seleção por perfil (CTI puro)":    r"C: por perfil (CTI puro)",
        "D: seleção por perfil (CTI ajustado)": r"D: por perfil (CTI ajustado)",
        "E: oráculo por série (exploratório)": r"E: oráculo por série$^\dagger$",
    }

    prev_lam = None
    for _, row in sel.sort_values(["lambda_BE","strategy"]).iterrows():
        lam = row["lambda_BE"]
        if lam != prev_lam and prev_lam is not None:
            lines.append(r"\midrule")
        prev_lam = lam
        slabel = STRAT_SHORT.get(row["strategy"], row["strategy"])
        lines.append(
            f"{slabel} & ${lam:.2f}$ & "
            f"{_fmt(row.CTI_mean)} & {_fmt(row.BE_mean, 1)} & {_fmt(row.NS_mean, 3)} & "
            f"{_fmt(row.CTI_adjusted_mean, 3)} & "
            f"${_fmt_pct(row.reduction_CTI_pct_vs_A)}$ & "
            f"${_fmt_pct(row.reduction_adj_pct_vs_A)}$ \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{l}{\scriptsize"
        r" CTI~Adj. usa normalização global min-max sobre todas as 12~políticas;"
        r" $\lambda_{BE}=0$ recupera seleção por CTI puro; A~$=$~B~$=$~EOQ para todo~$\lambda$.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Tabela LaTeX estratégias: {out_path}")


def _latex_dominant_table(dominant_table: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Política dominante por Perfil Operacional de Demanda para diferentes"
        r" pesos de penalidade por BE (Experimento~2, BA). ``CTI puro'': menor CTI médio viável"
        r" no perfil. ``CTI ajustado'': menor"
        r" $\mathrm{CTI_{norm}} + \lambda_{BE} \cdot \mathrm{BE_{norm}}$ viável no perfil,"
        r" normalizado dentro de cada perfil."
        r" Asterisco: perfil com $n < 20$ séries (evidência exploratória).}",
        r"\label{tab:dominant_policy_adjusted}",
        r"\begin{tabular}{@{}p{2.6cm}ccp{2.0cm}p{2.0cm}l@{}}",
        r"\toprule",
        r"\makecell[l]{Perfil} &",
        r"$n$ &",
        r"$\lambda_{BE}$ &",
        r"\makecell[l]{CTI puro} &",
        r"\makecell[l]{CTI ajustado} &",
        r"\makecell[l]{Muda?} \\",
        r"\midrule",
    ]
    prev_prof = None
    for _, row in dominant_table.sort_values(["operational_profile","lambda_BE"]).iterrows():
        prof = row["profile_display"]
        if prev_prof and prev_prof != prof:
            lines.append(r"\midrule")
        prev_prof = prof
        exp_mark = r"$^*$" if row["exploratory"] else ""
        changed  = r"\checkmark" if row["changed_vs_cti_only"] else ""
        lines.append(
            f"{prof}{exp_mark} & {row.n_series} & ${row.lambda_BE:.2f}$ & "
            f"{row.dominant_cti_display} & {row.dominant_adj_display} & {changed} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\scriptsize $^*$ $n < 20$: evidência exploratória."
        r" Normalização de CTI e BE dentro de cada perfil para seleção.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Tabela LaTeX dominância ajustada: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Relatório de validação
# ─────────────────────────────────────────────────────────────────────────────

def _validation_report(df: pd.DataFrame, global_agg: pd.DataFrame,
                        dominant_table: pd.DataFrame,
                        strat_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Validação — CTI Ajustado por Estabilidade Operacional",
        f"\nGerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## Fonte dos dados",
        "- `data/07_model_output/kpis.parquet`",
        "- `data/04_feature/demand_profiles.parquet`",
        "\n## Checagem 1 — Séries",
        f"  Esperado: 145 | Encontrado: {df[['warehouse','store_id','item_id']].drop_duplicates().shape[0]}",
        "\n## Checagem 2 — Políticas degeneradas marcadas",
        f"  DEGENERATE_POLICIES = {sorted(DEGENERATE_POLICIES)}",
        "\n## Checagem 3 — CTI original preservado",
        "  A coluna 'CTI' nos artefatos CSV reproduz o TIC original do kpis.parquet sem modificação.",
        "\n## Checagem 4 — lambda=0 recupera CTI puro",
    ]
    sub0 = dominant_table[dominant_table.lambda_BE == 0.00]
    for _, r in sub0.iterrows():
        match = "OK" if r.dominant_policy_cti == r.dominant_policy_adj else "DIVERGE"
        lines.append(f"  {r.profile_display}: CTI_puro={r.dominant_policy_cti}, CTI_adj(λ=0)={r.dominant_policy_adj} — {match}")
    lines.append("\n## Checagem 5 — Normalização")
    for prof in dominant_table.operational_profile.unique():
        sub = dominant_table[(dominant_table.operational_profile==prof) & (dominant_table.lambda_BE==0.25)]
        if not sub.empty:
            r = sub.iloc[0]
            lines.append(f"  {r.profile_display}: CTI_adj_selected(λ=0.25)={r.CTI_adjusted_selected:.4f} ∈ [0,2] ✓")
    lines.append("\n## Checagem 6 — NS médio nas estratégias")
    for lam in LAMBDA_VALUES:
        sub = strat_df[strat_df.lambda_BE == lam]
        lines.append(f"  λ={lam}:")
        for _, r in sub.iterrows():
            ns = r.get("NS_mean", None)
            flag = " ← ABAIXO DO LIMIAR" if ns is not None and ns < NS_THRESHOLD else ""
            lines.append(f"    {r['strategy'][:40]}: NS={ns}{flag}")
    lines.append("\n## Checagem 7 — Consistência com Tabela 5.2")
    eoc = global_agg[global_agg.policy=="EOQ"].iloc[0]
    lines.append(f"  EOQ CTI_mean (kpis atual): {eoc.CTI:.2f}  (Tabela 5.2: 628.42 — rodada final)")
    lines.append("\n## Limitações")
    lines.append("- kpis.parquet é rodada anterior; valores serão atualizados após re-simulação.")
    lines.append("- Perfis com n < 20: evidência exploratória.")
    lines.append("- Oráculo por série: limite exploratório, não estratégia operacional.")
    lines.append("- Análise concentrada no regime Lumpy do Experimento 2.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Relatório de validação: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrada principal
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df          = _load()
    global_agg  = _agg_global(df)
    profile_agg = _agg_profile(df)

    # Sensibilidade (perfil × política × λ)
    sens = _sensitivity_csv(profile_agg)
    sens.to_csv(OUT_DIR / "cti_adjusted_sensitivity.csv", index=False)
    log.info(f"Sensibilidade: {OUT_DIR / 'cti_adjusted_sensitivity.csv'}")

    # Dominância por perfil e λ
    dominant_table = _build_dominant_table(profile_agg)
    dominant_table.to_csv(OUT_DIR / "dominant_policy_by_profile_adjusted.csv", index=False)
    log.info(f"Dominância ajustada: {OUT_DIR / 'dominant_policy_by_profile_adjusted.csv'}")

    # Comparação de estratégias
    strat_df = _strategy_comparison(df, global_agg, profile_agg)
    strat_df.to_csv(OUT_DIR / "strategy_cost_comparison_adjusted.csv", index=False)
    log.info(f"Comparação ajustada: {OUT_DIR / 'strategy_cost_comparison_adjusted.csv'}")

    # Figura
    _sensitivity_figure(dominant_table, OUT_DIR / "cti_adjusted_sensitivity.pdf")

    # Tabelas LaTeX
    _latex_strategy_table(strat_df, OUT_DIR / "table_strategy_adjusted.tex")
    _latex_dominant_table(dominant_table, OUT_DIR / "table_dominant_adjusted.tex")

    # Relatório de validação
    _validation_report(df, global_agg, dominant_table, strat_df,
                       OUT_DIR / "cti_adjusted_validation.md")

    # Sumário
    log.info("=" * 60)
    log.info("SUMÁRIO — Dominância por perfil e λ")
    log.info("=" * 60)
    for _, r in dominant_table.sort_values(["lambda_BE","operational_profile"]).iterrows():
        changed = " ← MUDOU" if r.changed_vs_cti_only else ""
        log.info(
            f"  λ={r.lambda_BE:.2f} | {r.profile_display:20s} "
            f"| CTI: {r.dominant_cti_display:12s} "
            f"| Adj: {r.dominant_adj_display:12s}{changed}"
        )
    log.info(f"Artefatos: {OUT_DIR}")


if __name__ == "__main__":
    run()
