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
PENALTY_WEIGHT = 10.0  # mesmo peso da restrição de NS usado em constrained_cost (Eq. 4.2)
EXCESS_WEIGHT = 0.5  # peso do termo de estoque excessivo em CTI_ajustado (ver _add_adjusted_cost)

# 2026-08-18: portfolio ampliado de 12 para 18 politicas (reimplementacao
# 66d5ad8 + adocao Zabraoui). Faltavam as 6 novas aqui -- sem elas, o
# heatmap (`_heatmap`, que filtra por `p in POLICY_ORDER`) as descartava
# silenciosamente das figuras, mesmo que a tabela de dominancia (que nao
# usa POLICY_ORDER) ja as considerasse corretamente.
POLICY_ORDER = ["EOQ", "sS", "Newsvendor",
                "PIL", "CappedBaseStock", "BigDataNewsvendor",
                "MinMax", "FixedInterval", "VendorResponsive",
                "GA", "SA", "PSO", "DE",
                "DQN", "PPO", "SARSA", "GA-DQN", "GA-PPO"]

POLICY_DISPLAY = {
    "sS": "(s,S)",
    "Newsvendor": "Jornaleiro",
    "CappedBaseStock": "Capped Base-Stock",
    "BigDataNewsvendor": "Big Data Newsvendor",
    "MinMax": "Min-Max",
    "FixedInterval": "Fixed Interval",
    "VendorResponsive": "Vendor-Responsive",
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

def _load_data(kpi_path: Path = KPI_PATH, prof_path: Path = PROF_PATH
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    2026-08-18: aceita `kpi_path`/`prof_path` explícitos -- o script era
    hardcoded em `data/07_model_output/kpis.parquet`, sem noção de ambiente
    (`m5`/`bot` isolam esses datasets em `data/*/m5/` e `data/*/bot/`, ver
    AJUSTES_INFRA item #8). Chamado sem argumentos, mantém o comportamento
    antigo (base/Bahia) por retrocompatibilidade com o uso via CLI.
    """
    log.info("Lendo %s …", kpi_path)
    kpis = pd.read_parquet(kpi_path)
    log.info("Lendo %s …", prof_path)
    profiles = pd.read_parquet(prof_path)
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

def _add_adjusted_cost(merged: pd.DataFrame, ns_threshold: float = NS_THRESHOLD,
                       penalty_weight: float = PENALTY_WEIGHT,
                       excess_weight: float = EXCESS_WEIGHT) -> pd.DataFrame:
    """
    CTI_ajustado por linha (série x política) -- ver AJUSTES_INFRA item #33.

    Motivação (2026-08-18, "esse calculo de score não ta legal"): o `score`
    antigo (e `constrained_cost`, usado no treino) escalam a penalidade de
    déficit de NS pelo TIC_ref = clip(TIC_DA_PRÓPRIA_CANDIDATA, min=1.0).
    Isso cria uma brecha auto-referencial: uma política que nunca pede
    (S=0) tem TIC quase zero, então mesmo um déficit de NS de 45pp vira
    uma penalidade pequena em valor absoluto -- comprovado numericamente
    numa série real da Bahia (S=0: custo=474 vs S=8 com NS=1.0: custo=587;
    o "nunca pedir" vence porque penaliza a si mesmo com sua própria régua
    minúscula). É a causa-raiz do colapso "nunca pedir" encontrado em PIL,
    CappedBaseStock, DQN e PPO.

    Correção: a régua de penalidade (`tic_ref`) passa a ser FIXA por série
    -- o maior TIC observado entre as 18 políticas para aquela mesma série
    (o "teto" de custo que já se paga por ali) -- em vez do TIC da própria
    candidata. Isso torna o déficit de NS caro em termos absolutos mesmo
    quando a candidata finge ser barata não pedindo nada.

    Além disso, soma-se um termo explícito de estoque excessivo (pedido do
    usuário: "aumento do custo por estoque excessivo e manter no armazém
    além de prejuízo por não ter produto disponível"):

        CTI_ajustado = TIC
                     + deficit_NS * penalty_weight * tic_ref_serie   (prejuízo por indisponibilidade,
                                                                       referência FIXA por série)
                     + excess_weight * max(0, HoldingCost - mediana(HoldingCost na série))
                                                                      (estoque excessivo vs. o que as
                                                                       outras políticas da mesma série
                                                                       precisaram manter)

    O termo de excesso só é calculado se `HoldingCost` estiver presente
    (kpis.parquet gerado após este item; rodadas antigas caem no termo
    apenas de indisponibilidade). `AvgInventory`/`HoldingCost`/
    `StockoutCost` vêm da decomposição nova de `.kpis()` em
    `inventory_env.py`/`inventory_env_torch.py`.
    """
    series_key = ["store_id", "item_id"]
    tic_ref_series = merged.groupby(series_key)["TIC"].transform("max").clip(lower=1.0)
    deficit = (ns_threshold - merged["NS"]).clip(lower=0.0)
    service_loss = deficit * penalty_weight * tic_ref_series

    if "HoldingCost" in merged.columns and merged["HoldingCost"].notna().any():
        median_holding = merged.groupby(series_key)["HoldingCost"].transform("median")
        excess = (merged["HoldingCost"] - median_holding).clip(lower=0.0) * excess_weight
    else:
        excess = 0.0

    merged = merged.copy()
    merged["CTI_ajustado"] = merged["TIC"] + service_loss + excess
    return merged


def _aggregate_by_profile(merged: pd.DataFrame, ns_threshold: float = NS_THRESHOLD,
                          penalty_weight: float = PENALTY_WEIGHT,
                          excess_weight: float = EXCESS_WEIGHT) -> pd.DataFrame:
    """
    Agrega KPIs por (operational_profile, policy) -- esta é a saída principal
    do AIPE pedida pelo usuário: uma tabela com o SCORE de cada uma das 18
    políticas para cada perfil operacional (não só a política vencedora).

    `score_ajustado`/`CTI_ajustado_mean`: métrica de escolha final -- ver
    `_add_adjusted_cost` para a formulação e a motivação (correção do bug
    de auto-referência + termos de estoque excessivo / indisponibilidade
    pedidos pelo usuário). `score` (formulação antiga, TIC_ref por linha)
    é mantido só por retrocompatibilidade de leitura dos relatórios já
    publicados; `score_ajustado` é o que deve ser usado daqui em diante.
    """
    merged = _add_adjusted_cost(merged, ns_threshold, penalty_weight, excess_weight)
    kpi_cols = [c for c in ["TIC", "NS", "TR", "BE", "FP", "CTI_ajustado",
                            "HoldingCost", "StockoutCost", "OrderCost", "AvgInventory"]
               if c in merged.columns]

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

    # score antigo (retrocompatibilidade) -- TIC_ref auto-referencial por linha
    tic_ref = agg["TIC_mean"].clip(lower=1.0)
    deficit = (ns_threshold - agg["NS_mean"]).clip(lower=0.0)
    agg["score"] = -(agg["TIC_mean"] + deficit * penalty_weight * tic_ref)
    agg["viable"] = agg["NS_mean"] >= ns_threshold

    # score_ajustado -- métrica de escolha final (ver _add_adjusted_cost)
    agg["score_ajustado"] = -agg["CTI_ajustado_mean"]
    return agg


def _dominant_policy_per_profile(agg: pd.DataFrame,
                                 ns_threshold: float = NS_THRESHOLD) -> pd.DataFrame:
    """Identifica política dominante por perfil: min CTI_ajustado entre NS_mean >= ns_threshold.

    2026-08-18 (AJUSTES_INFRA item #33): critério de desempate passou de
    `min(TIC_mean)` para `min(CTI_ajustado_mean)` -- pedido explícito do
    usuário ("no comparativo final devemos comparar o custo total
    ajustado"). Ver `_add_adjusted_cost` para a formulação.

    Resumo de 1 linha por perfil sobre a tabela completa de `score` (que
    tem as 18 políticas x todos os perfis) -- útil pra leitura rápida, mas
    a tabela completa (`agg`/`profile_policy_metrics.csv`) é a saída
    principal: tem o score de TODAS as políticas, não só a vencedora.
    """
    records = []
    for profile, grp in agg.groupby("operational_profile"):
        n_series = grp["n_series"].iloc[0]
        viable = grp[grp["NS_mean"] >= ns_threshold].copy()
        fallback = False
        if viable.empty:
            viable = grp.copy()
            fallback = True
            log.warning(f"Perfil '{profile}': nenhuma política atinge NS>={ns_threshold}. Usando fallback (maior NS).")
            dominant_row = viable.loc[viable["NS_mean"].idxmax()]
        else:
            dominant_row = viable.loc[viable["CTI_ajustado_mean"].idxmin()]

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
            "CTI_ajustado_mean":    round(dominant_row["CTI_ajustado_mean"], 2),
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
                        out_path: Path, kpi_path: Path = KPI_PATH,
                        prof_path: Path = PROF_PATH,
                        ns_threshold: float = NS_THRESHOLD) -> None:
    n_series  = merged[["store_id", "item_id"]].drop_duplicates().shape[0]
    n_policies = merged["policy"].nunique()
    profiles  = merged["operational_profile"].dropna().unique()

    global_means = kpis.groupby("policy")["TIC"].mean().to_dict()

    def _rel(p: Path) -> str:
        try:
            return str(Path(p).relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    lines = [
        f"# Validação — Avaliação por Perfil Operacional",
        f"",
        f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Fonte dos dados",
        f"- KPIs: `{_rel(kpi_path)}`",
        f"- Perfis: `{_rel(prof_path)}`",
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
        f"- Políticas viáveis: NS médio >= {ns_threshold}",
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
        r" a política dominante (menor CTI ajustado médio entre políticas com NS médio"
        r" $\geq 0{,}70$), o CTI médio, o CTI ajustado médio (Eq. custo total ajustado,"
        r" AJUSTES\_INFRA item \#33) e o NS médio da política dominante."
        r" Perfis com $n < 20$ séries devem ser interpretados de forma exploratória.}",
        r"\label{tab:dominancia_por_perfil}",
        r"\begin{tabular}{@{}p{3.0cm}clrrrl@{}}",
        r"\toprule",
        r"Perfil & $n$ & Política dominante & CTI médio (R\$) & CTI ajustado (R\$) & NS médio & Observação \\",
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
            f"{row['CTI_ajustado_mean']:.2f} & "
            f"{row['NS_mean']:.3f} & "
            f"{note} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\scriptsize{*}$n < 20$: evidência exploratória.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Tabela LaTeX salva: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(kpis: pd.DataFrame | None = None,
        profiles: pd.DataFrame | None = None,
        out_dir: Path | str | None = None,
        ns_threshold: float = NS_THRESHOLD,
        penalty_weight: float = PENALTY_WEIGHT,
        excess_weight: float = EXCESS_WEIGHT,
        kpi_path: Path = KPI_PATH,
        prof_path: Path = PROF_PATH) -> pd.DataFrame:
    """
    2026-08-18: `run()` passou a aceitar `kpis`/`profiles`/`out_dir`
    explícitos. Antes, sempre relia no `KPI_PATH`/`PROF_PATH`/`OUT_DIR`
    hardcoded (só a base/Bahia) -- rodar contra "bot"/M5 exigia isso.
    Chamado sem argumentos, mantém o comportamento antigo (retrocompat
    para uso via CLI: `python profile_policy_analysis.py`).

    Retorna a tabela completa `agg` (perfil x política x score), a mesma
    salva em `profile_policy_metrics.csv/.parquet` -- é essa a saída
    principal do AIPE: o score de cada uma das 18 políticas em cada
    perfil, não só a política vencedora.
    """
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if kpis is None or profiles is None:
        kpis, profiles = _load_data(kpi_path, prof_path)
    merged = _merge(kpis, profiles)

    log.info(f"Dados unidos: {len(merged)} linhas, "
             f"{merged[['store_id','item_id']].drop_duplicates().shape[0]} séries, "
             f"{merged['operational_profile'].nunique()} perfis.")

    agg = _aggregate_by_profile(merged, ns_threshold, penalty_weight, excess_weight)
    dominant = _dominant_policy_per_profile(agg, ns_threshold)

    # Export CSVs and parquets
    agg_out = out_dir / "profile_policy_metrics"
    agg.to_csv(str(agg_out) + ".csv", index=False)
    agg.to_parquet(str(agg_out) + ".parquet", index=False)
    log.info(f"Métricas por perfil (score de cada política x perfil): {agg_out}.csv / .parquet")

    dom_out = out_dir / "dominant_policy_by_profile"
    dominant.to_csv(str(dom_out) + ".csv", index=False)
    dominant.to_parquet(str(dom_out) + ".parquet", index=False)
    log.info(f"Dominância por perfil: {dom_out}.csv / .parquet")

    # Figures
    _heatmap(agg, "TIC_mean", "CTI médio por Perfil e Política (R$)",
             out_dir / "profile_policy_heatmap_cti.pdf", cmap="RdYlGn_r", fmt=".0f")
    _heatmap(agg, "NS_mean", "NS médio por Perfil e Política",
             out_dir / "profile_policy_heatmap_ns.pdf", cmap="RdYlGn", fmt=".2f")
    _heatmap(agg, "CTI_ajustado_mean", "CTI ajustado médio por Perfil e Política (R$, menor = melhor)",
             out_dir / "profile_policy_heatmap_cti_ajustado.pdf", cmap="RdYlGn_r", fmt=".0f")
    _heatmap(agg, "score_ajustado", "Score ajustado por Perfil e Política (maior = melhor)",
             out_dir / "profile_policy_heatmap_score.pdf", cmap="RdYlGn", fmt=".0f")
    _dominance_barplot(dominant, out_dir / "profile_policy_dominance_barplot.pdf")

    # LaTeX snippet
    _latex_dominance_table(dominant, out_dir / "table_dominancia_por_perfil.tex")

    # Validation report
    _validation_report(kpis, merged, agg, dominant,
                       out_dir / "profile_policy_validation.md",
                       kpi_path=kpi_path, prof_path=prof_path, ns_threshold=ns_threshold)

    log.info("Análise por perfil concluída.")
    log.info(f"Artefatos em: {out_dir}")
    return agg


if __name__ == "__main__":
    run()
