"""
reporting/nodes.py — Geracao de figuras e tabelas LaTeX para a dissertacao.

Estrutura de saida em data/08_reporting/:
  comparison/   — barras, radar, heatmap, boxplot, violin, scatter, pareto
  statistical/  — wilcoxon heatmap, cohen's d, critical difference
  forecast/     — forecast vs actual, residuos, metricas agregadas
  demand/       — adi-cv scatter, distribuicao demanda, calendario
  maps/         — mapas do Brasil
  latex/        — tabelas .tex para booktabs
"""
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

BASE_DIR = "data/08_reporting"

SUBDIRS = {
    "comparison":  os.path.join(BASE_DIR, "comparison"),
    "statistical": os.path.join(BASE_DIR, "statistical"),
    "forecast":    os.path.join(BASE_DIR, "forecast"),
    "demand":      os.path.join(BASE_DIR, "demand"),
    "maps":        os.path.join(BASE_DIR, "maps"),
    "latex":       os.path.join(BASE_DIR, "latex"),
}

KPI_DISPLAY = {
    "TIC": ("TIC (R$)", False),
    "NS":  ("NS",       True),
    "TR":  ("TR",       False),
    "BE":  ("BE",       False),
    "FP":  ("FP",       False),
}

POLICY_ORDER = ["EOQ", "sS", "Newsvendor", "GA", "SA", "PSO", "DE",
                "DQN", "PPO", "SARSA", "GA-DQN", "GA-PPO"]


def generate_comparison_plots(kpis: pd.DataFrame,
                               wilcoxon_results: pd.DataFrame,
                               friedman_results: dict,
                               effect_sizes: pd.DataFrame,
                               params: dict) -> dict:
    """
    Gera figuras de comparacao de politicas e validacao estatistica.
    Retorna dict {nome: caminho_salvo}.
    """
    from simulation.core.visualizations import (
        plot_comparison_bars, plot_heatmap_service_level,
        plot_boxplot_tic, plot_tradeoff_scatter, plot_violin_bullwhip,
        plot_critical_difference, plot_radar_kpis, plot_kpi_heatmap_all,
        plot_pareto_frontier, plot_policy_sensitivity,
        plot_wilcoxon_pvalue_heatmap, plot_cohens_d_heatmap,
    )

    comp_dir = SUBDIRS["comparison"]
    stat_dir = SUBDIRS["statistical"]
    cfg = params.get("plots", {})
    saved = {}

    _try_plot(cfg, "comparison_bars",       saved, plot_comparison_bars,       kpis, comp_dir, params)
    _try_plot(cfg, "radar_kpis",            saved, plot_radar_kpis,            kpis, comp_dir, params)
    _try_plot(cfg, "kpi_heatmap_all",       saved, plot_kpi_heatmap_all,       kpis, comp_dir, params)
    _try_plot(cfg, "heatmap_service_level", saved, plot_heatmap_service_level, kpis, comp_dir, params)
    _try_plot(cfg, "boxplot_tic",           saved, plot_boxplot_tic,           kpis, comp_dir, params)
    _try_plot(cfg, "violin_bullwhip",       saved, plot_violin_bullwhip,       kpis, comp_dir, params)
    _try_plot(cfg, "tradeoff_scatter",      saved, plot_tradeoff_scatter,      kpis, comp_dir, params)
    _try_plot(cfg, "pareto_frontier",       saved, plot_pareto_frontier,       kpis, comp_dir, params)
    _try_plot(cfg, "policy_sensitivity",    saved, plot_policy_sensitivity,    kpis, comp_dir, params)

    if cfg.get("wilcoxon_pvalue_heatmap", True) and wilcoxon_results is not None and not wilcoxon_results.empty:
        _try_plot(cfg, "wilcoxon_pvalue_heatmap", saved,
                  plot_wilcoxon_pvalue_heatmap, wilcoxon_results, stat_dir, params)

    if cfg.get("cohens_d_heatmap", True) and effect_sizes is not None and not effect_sizes.empty:
        _try_plot(cfg, "cohens_d_heatmap", saved,
                  plot_cohens_d_heatmap, effect_sizes, stat_dir, params)

    if cfg.get("critical_difference", True) and friedman_results:
        for metric, res in friedman_results.items():
            try:
                p = plot_critical_difference(res, stat_dir, params)
                if p:
                    saved[f"critical_difference_{metric}"] = p
            except Exception as e:
                log.warning("critical_difference [%s] falhou: %s", metric, e)

    log.info("Figuras de comparacao: %s", list(saved.keys()))
    return saved


def generate_demand_plots(scenarios: pd.DataFrame,
                           scenarios_meta: pd.DataFrame,
                           forecast_predictions: pd.DataFrame,
                           forecast_metrics: pd.DataFrame,
                           params: dict) -> dict:
    """
    Gera figuras de caracterizacao da demanda e qualidade do forecast.
    Demand plots -> demand/  |  Forecast plots -> forecast/
    """
    from simulation.core.visualizations import (
        plot_adi_cv_scatter, plot_demand_distribution,
        plot_intermittency_calendar, plot_forecast_vs_actual,
        plot_forecast_residuals, plot_forecast_metrics_distribution,
    )

    dem_dir  = SUBDIRS["demand"]
    fore_dir = SUBDIRS["forecast"]
    cfg = params.get("plots", {})
    saved = {}

    _try_plot(cfg, "adi_cv_scatter",         saved, plot_adi_cv_scatter,         scenarios_meta, dem_dir, params)
    _try_plot(cfg, "demand_distribution",    saved, plot_demand_distribution,    scenarios, scenarios_meta, dem_dir, params)
    _try_plot(cfg, "intermittency_calendar", saved, plot_intermittency_calendar, scenarios, dem_dir, params)

    if forecast_predictions is not None and not forecast_predictions.empty:
        _try_plot(cfg, "forecast_vs_actual", saved, plot_forecast_vs_actual, forecast_predictions, fore_dir, params)
        _try_plot(cfg, "forecast_residuals", saved, plot_forecast_residuals, forecast_predictions, fore_dir, params)

    if forecast_metrics is not None and not forecast_metrics.empty:
        _try_plot(cfg, "forecast_metrics_distribution", saved,
                  plot_forecast_metrics_distribution, forecast_metrics, fore_dir, params)

    log.info("Figuras de demanda/forecast: %s", list(saved.keys()))
    return saved


def generate_map_plots(scenarios_meta: pd.DataFrame,
                        kpis: pd.DataFrame,
                        params: dict) -> dict:
    """
    Gera mapas geograficos do Brasil (requer geopandas + geobr).
    """
    from simulation.core.visualizations import (
        plot_brazil_store_map, plot_brazil_kpi_choropleth,
    )

    maps_dir = SUBDIRS["maps"]
    cfg = params.get("plots", {})
    saved = {}

    _try_plot(cfg, "brazil_store_map", saved, plot_brazil_store_map, scenarios_meta, maps_dir, params)

    if cfg.get("brazil_kpi_choropleth", True):
        try:
            result = plot_brazil_kpi_choropleth(kpis, maps_dir, params)
            if result:
                saved.update(result)
        except Exception as e:
            log.warning("brazil_kpi_choropleth falhou: %s", e)

    log.info("Mapas gerados: %s", list(saved.keys()))
    return saved


def generate_dissertation_report(scenarios_meta: pd.DataFrame,
                                  forecast_metrics: pd.DataFrame,
                                  kpis: pd.DataFrame,
                                  params: dict) -> dict:
    """
    Gera relatorio de recomendacao de pares (loja, produto) para a dissertacao.

    Criterios de selecao:
      1. Representatividade: pelo menos um par por grupo Syntetos-Boylan
      2. Dificuldade de forecast: MASE alto (serie desafiadora, mais interessante)
      3. Volume: demanda media elevada (relevancia economica)
      4. Riqueza de dados: maximo de ciclos disponiveis

    Retorna dict com:
      - 'recommendations_df' : DataFrame ranqueado
      - 'report_text'        : texto markdown do relatorio
    """
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(BASE_DIR, "dissertation_report.md")

    # ── 1. Base de dados por serie ─────────────────────────────────────────
    meta_cols = ["warehouse", "store_id", "item_id", "group", "cv", "adi",
                 "n_periods", "n_positive", "mu", "mean_demand"]
    meta_cols = [c for c in meta_cols if c in scenarios_meta.columns]
    base = scenarios_meta[meta_cols].copy()
    # Normaliza nome da coluna de demanda media
    if "mu" in base.columns and "mean_demand" not in base.columns:
        base = base.rename(columns={"mu": "mean_demand"})

    # Adiciona ADI se nao calculado
    if "adi" not in base.columns and "n_periods" in base.columns and "n_positive" in base.columns:
        base["adi"] = base["n_periods"] / base["n_positive"].replace(0, 1)

    # ── 2. Metricas de forecast agregadas (cycle == 'ALL') ────────────────
    if forecast_metrics is not None and not forecast_metrics.empty:
        agg_metrics = forecast_metrics[
            forecast_metrics["cycle"].astype(str) == "ALL"
        ].copy()

        # Escolhe o modelo com melhor MASE mediano como referencia (mais robusto)
        if "MASE" in agg_metrics.columns:
            model_med = agg_metrics.groupby("model")["MASE"].median()
            best_model = model_med.idxmin() if not model_med.empty else None
        else:
            best_model = None

        if best_model:
            agg_metrics = agg_metrics[agg_metrics["model"] == best_model]
            metric_cols = [c for c in ["MAE", "RMSE", "MASE", "RMSSE", "sMAPE", "MBE", "TheilsU"]
                           if c in agg_metrics.columns]
            series_metrics = (
                agg_metrics.groupby(["warehouse", "store_id", "item_id"])[metric_cols]
                .mean()
                .reset_index()
            )
            base = base.merge(series_metrics, on=["warehouse", "store_id", "item_id"], how="left")
            base["_best_model"] = best_model
        else:
            best_model = "N/A"
    else:
        best_model = "N/A"

    # ── 3. KPIs de inventario (media da politica proposta) ────────────────
    if kpis is not None and not kpis.empty:
        proposed = kpis[kpis["policy"].isin(["GA-DQN", "GA-PPO"])]
        if proposed.empty:
            proposed = kpis
        kpi_cols = [c for c in ["TIC", "NS", "TR"] if c in proposed.columns]
        kpi_agg = (
            proposed.groupby(["warehouse", "store_id", "item_id"])[kpi_cols]
            .mean()
            .reset_index()
            .rename(columns={c: f"kpi_{c}" for c in kpi_cols})
        )
        base = base.merge(kpi_agg, on=["warehouse", "store_id", "item_id"], how="left")

    # ── 4. Score composto ─────────────────────────────────────────────────
    # Normaliza cada dimensao para [0,1] e combina com pesos
    def _norm(s, higher_is_better=True):
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.5, index=s.index)
        norm = (s - lo) / (hi - lo)
        return norm if higher_is_better else (1 - norm)

    score = pd.Series(0.0, index=base.index)
    w_total = 0.0

    if "MASE" in base.columns and base["MASE"].notna().any():
        score += 0.35 * _norm(base["MASE"].fillna(base["MASE"].median()), higher_is_better=True)
        w_total += 0.35

    if "mean_demand" in base.columns and base["mean_demand"].notna().any():
        score += 0.25 * _norm(base["mean_demand"].fillna(0), higher_is_better=True)
        w_total += 0.25

    if "n_periods" in base.columns and base["n_periods"].notna().any():
        score += 0.20 * _norm(base["n_periods"].fillna(0), higher_is_better=True)
        w_total += 0.20

    if "cv" in base.columns and base["cv"].notna().any():
        score += 0.20 * _norm(base["cv"].fillna(0), higher_is_better=True)
        w_total += 0.20

    if w_total > 0:
        score = score / w_total
    base["score"] = score

    # ── 5. Selecao por grupo (top-3 por grupo) ────────────────────────────
    recommendations = []
    groups = sorted(base["group"].dropna().unique()) if "group" in base.columns else ["N/A"]

    for grp in groups:
        sub = base[base["group"] == grp].sort_values("score", ascending=False)
        n_pick = min(3, len(sub))
        top = sub.head(n_pick).copy()
        top["rank_in_group"] = range(1, n_pick + 1)
        recommendations.append(top)

    if recommendations:
        rec_df = pd.concat(recommendations, ignore_index=True)
    else:
        rec_df = base.sort_values("score", ascending=False).head(10)

    rec_df = rec_df.sort_values(["group", "score"], ascending=[True, False])

    # ── 6. Texto markdown do relatorio ────────────────────────────────────
    lines = [
        "# Relatorio de Recomendacao — Pares (Loja, Produto) para Dissertacao",
        "",
        "**Objetivo**: Selecionar series temporais representativas de cada categoria",
        "de demanda intermitente (Syntetos-Boylan 2005) para analise aprofundada na",
        "dissertacao de mestrado.",
        "",
        "## Metodologia de Selecao",
        "",
        "| Criterio              | Peso | Justificativa |",
        "|------------------------|------|---------------|",
        "| MASE elevado (dificuldade) | 35% | Series desafiadoras sao mais informativas |",
        "| Demanda media (volume) | 25% | Relevancia economica do par loja-produto |",
        "| Riqueza de dados (n ciclos) | 20% | Mais ciclos = validacao mais confiavel |",
        "| CV elevado (variabilidade) | 20% | Variabilidade e central no estudo |",
        "",
        f"**Modelo de referencia para metricas de forecast**: {best_model}",
        "",
        "## Grupos Syntetos-Boylan",
        "",
        "| Grupo | ADI | CV² | Caracteristica |",
        "|-------|-----|-----|----------------|",
        "| Smooth | < 1.32 | < 0.49 | Demanda regular, facil de prever |",
        "| Erratic | < 1.32 | >= 0.49 | Demanda frequente mas tamanho erratico |",
        "| Intermittent | >= 1.32 | < 0.49 | Demanda infrequente, tamanhos estaveis |",
        "| Lumpy | >= 1.32 | >= 0.49 | Demanda infrequente e erratica (mais dificil) |",
        "",
        "## Series Recomendadas por Grupo",
        "",
    ]

    display_cols = ["warehouse", "store_id", "item_id", "group", "cv", "adi",
                    "n_periods", "mean_demand", "MASE", "sMAPE", "score"]
    display_cols = [c for c in display_cols if c in rec_df.columns]

    for grp in groups:
        sub = rec_df[rec_df["group"] == grp] if "group" in rec_df.columns else rec_df
        if sub.empty:
            continue
        lines.append(f"### {grp}")
        lines.append("")

        header = "| # | Estado | Loja | Produto | CV | ADI | Ciclos | Demanda Media | MASE | sMAPE | Score |"
        divider = "|---|--------|------|---------|-----|-----|--------|---------------|------|-------|-------|"
        lines.append(header)
        lines.append(divider)

        for j, (_, row) in enumerate(sub.iterrows(), 1):
            def _f(col, decimals=2):
                v = row.get(col, None)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "-"
                return f"{v:.{decimals}f}"

            lines.append(
                f"| {j} "
                f"| {row.get('warehouse', '-')} "
                f"| {row.get('store_id', '-')} "
                f"| {row.get('item_id', '-')} "
                f"| {_f('cv')} "
                f"| {_f('adi')} "
                f"| {int(row['n_periods']) if 'n_periods' in row and not pd.isna(row['n_periods']) else '-'} "
                f"| {_f('mean_demand', 1)} "
                f"| {_f('MASE', 3)} "
                f"| {_f('sMAPE', 1)} "
                f"| {_f('score', 3)} |"
            )
        lines.append("")

    # Sumario executivo
    lines += [
        "## Sumario Executivo",
        "",
        f"- **Total de series avaliadas**: {len(base):,}",
        f"- **Series recomendadas**: {len(rec_df)}",
        "",
        "### Por que estas series?",
        "",
        "As series selecionadas maximizam simultaneamente:",
        "1. **Dificuldade de previsao** (MASE > 1 indica que o modelo enfrenta mais",
        "   incerteza que a previsao naive — ideal para demonstrar ganho da arquitetura proposta)",
        "2. **Representatividade** de cada quadrante ADI-CV² para garantir que a",
        "   dissertacao cobre toda a gama de comportamentos de demanda intermitente",
        "3. **Relevancia economica** via volume de demanda (series com TIC mais alto",
        "   mostram maior impacto de uma politica de estoque superior)",
        "",
        "### Proximos passos",
        "",
        "1. Validar as series recomendadas visualmente (`forecast/forecast_vs_actual.pdf`)",
        "2. Confirmar que cada grupo tem pelo menos 3 ciclos de teste possiveis",
        "3. Para a dissertacao, usar as series com MASE > 1 como caso principal e",
        "   series com MASE < 1 como casos em que a previsao ja funciona bem",
        "",
        "---",
        "*Relatorio gerado automaticamente pelo pipeline Kedro (reporting pipeline)*",
    ]

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info("Relatorio da dissertacao salvo em: %s", report_path)

    # Salva CSV compacto das recomendacoes
    csv_path = os.path.join(BASE_DIR, "recommendations.parquet")
    rec_df.to_parquet(csv_path, index=False)
    log.info("Recomendacoes salvas em: %s", csv_path)

    return {
        "report_path":       report_path,
        "n_recommended":     len(rec_df),
        "groups_covered":    sorted(rec_df["group"].dropna().unique().tolist())
                             if "group" in rec_df.columns else [],
        "best_forecast_model": best_model,
    }


def _try_plot(cfg, key, saved, fn, *args):
    """Executa fn(*args) se cfg.get(key, True), captura excecoes."""
    if not cfg.get(key, True):
        return
    try:
        result = fn(*args)
        if result is not None:
            if isinstance(result, dict):
                saved.update(result)
            else:
                saved[key] = result
    except Exception as e:
        log.warning("%s falhou: %s", key, e)


def generate_latex_tables(kpis: pd.DataFrame,
                           wilcoxon_results: pd.DataFrame,
                           effect_sizes: pd.DataFrame,
                           stratified_summary: pd.DataFrame,
                           params: dict) -> dict:
    """
    Gera tabelas LaTeX para a dissertacao.
    Retorna dict {nome: string .tex}.
    """
    latex_cfg = params.get("latex", {})
    tables = {}

    if latex_cfg.get("global_table", True):
        tables["global_kpis"] = _latex_global_table(kpis, latex_cfg)

    if latex_cfg.get("stratified_table", True):
        tables["stratified_kpis"] = _latex_stratified_table(stratified_summary, latex_cfg)

    if latex_cfg.get("statistical_tests_table", True) and not wilcoxon_results.empty:
        tables["wilcoxon"] = _latex_wilcoxon_table(wilcoxon_results, latex_cfg)

    if latex_cfg.get("effect_sizes_table", True) and not effect_sizes.empty:
        tables["effect_sizes"] = _latex_effect_sizes_table(effect_sizes, latex_cfg)

    latex_dir = SUBDIRS["latex"]
    Path(latex_dir).mkdir(parents=True, exist_ok=True)
    for name, tex in tables.items():
        path = os.path.join(latex_dir, f"table_{name}.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        log.info("  [LaTeX] %s", path)

    return tables


def _fmt_num(val, decimals=2, decimal_sep="{,}"):
    """Formata numero com separador decimal configuravel."""
    if pd.isna(val):
        return "--"
    s = f"{val:.{decimals}f}"
    if decimal_sep != ".":
        s = s.replace(".", decimal_sep)
    return s


def _latex_global_table(kpis: pd.DataFrame, cfg: dict) -> str:
    """Tabela global: media +- desvio por politica x KPI."""
    sep = cfg.get("decimal_sep", "{,}")
    booktabs = cfg.get("booktabs", True)
    rule_top = "\\toprule" if booktabs else "\\hline"
    rule_mid = "\\midrule" if booktabs else "\\hline"
    rule_bot = "\\bottomrule" if booktabs else "\\hline"

    kpi_cols = [c for c in ["TIC", "NS", "TR", "BE", "FP"] if c in kpis.columns]
    policies = [p for p in POLICY_ORDER if p in kpis["policy"].unique()]

    header_cols = " & ".join(["Politica"] + kpi_cols)
    col_spec = "l" + "r" * len(kpi_cols)

    lines = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Resultados medios das politicas de inventario (todas as lojas)}",
        "\\label{tab:kpis_global}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        rule_top,
        header_cols + " \\\\",
        rule_mid,
    ]

    summary = kpis.groupby("policy")[kpi_cols].agg(["mean", "std"])
    for pol in policies:
        if pol not in summary.index:
            continue
        row_vals = [pol.replace("-", "\\text{-}")]
        for col in kpi_cols:
            mu = summary.loc[pol, (col, "mean")]
            sd = summary.loc[pol, (col, "std")]
            cell = f"${_fmt_num(mu, 1, sep)} \\pm {_fmt_num(sd, 1, sep)}$"
            row_vals.append(cell)
        lines.append(" & ".join(row_vals) + " \\\\")

    lines += [rule_bot, "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _latex_stratified_table(stratified: pd.DataFrame, cfg: dict) -> str:
    """Tabela estratificada por grupo de demanda."""
    sep = cfg.get("decimal_sep", "{,}")
    booktabs = cfg.get("booktabs", True)
    rule_top = "\\toprule" if booktabs else "\\hline"
    rule_mid = "\\midrule" if booktabs else "\\hline"
    rule_bot = "\\bottomrule" if booktabs else "\\hline"

    mean_cols = [c for c in stratified.columns if c.endswith("_mean")]
    display_cols = [c.replace("_mean", "") for c in mean_cols]

    header = " & ".join(["Estado", "Grupo", "Politica"] + display_cols)
    col_spec = "llr" + "r" * len(mean_cols)

    lines = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Resultados estratificados por estado e grupo de demanda}",
        "\\label{tab:kpis_stratified}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        rule_top,
        header + " \\\\",
        rule_mid,
    ]

    for _, row in stratified.iterrows():
        vals = [str(row.get("warehouse", "?")),
                str(row.get("group", "?")),
                str(row.get("policy", "?"))]
        for mc in mean_cols:
            vals.append(f"${_fmt_num(row.get(mc, np.nan), 2, sep)}$")
        lines.append(" & ".join(vals) + " \\\\")

    lines += [rule_bot, "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _latex_wilcoxon_table(wilcoxon: pd.DataFrame, cfg: dict) -> str:
    """Tabela de resultados do teste de Wilcoxon."""
    sep = cfg.get("decimal_sep", "{,}")
    booktabs = cfg.get("booktabs", True)
    rule_top = "\\toprule" if booktabs else "\\hline"
    rule_mid = "\\midrule" if booktabs else "\\hline"
    rule_bot = "\\bottomrule" if booktabs else "\\hline"

    lines = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Resultados do teste de Wilcoxon signed-rank (politica proposta vs baselines)}",
        "\\label{tab:wilcoxon}",
        "\\begin{tabular}{llrrrl}",
        rule_top,
        "Politica A & Politica B & Metrica & Estatistica & $p$-valor & Sig. \\\\",
        rule_mid,
    ]

    for _, row in wilcoxon.iterrows():
        sig = "$^{*}$" if row.get("significant", False) else ""
        p_str = f"${_fmt_num(row['p_value'], 4, sep)}$"
        lines.append(
            f"{row['policy_a']} & {row['policy_b']} & {row['metric']} & "
            f"${_fmt_num(row['statistic'], 1, sep)}$ & {p_str} & {sig} \\\\"
        )

    lines += [rule_bot, "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def generate_profile_policy_analysis(kpis: pd.DataFrame,
                                      demand_profiles: pd.DataFrame,
                                      params: dict) -> None:
    """
    Wrapper Kedro para reporting/profile_policy_analysis.py.
    O script le kpis.parquet e demand_profiles.parquet diretamente do disco
    (mesmos caminhos do catalog); os argumentos aqui apenas fixam a ordem de
    execucao no DAG (depois que esses datasets foram salvos).
    """
    from reporting.profile_policy_analysis import run
    run()


def generate_strategy_cost_comparison(kpis: pd.DataFrame,
                                       demand_profiles: pd.DataFrame,
                                       params: dict) -> None:
    """Wrapper Kedro para reporting/strategy_cost_comparison.py (mesmo padrao de I/O)."""
    from reporting.strategy_cost_comparison import run
    run()


def generate_cti_adjusted_analysis(kpis: pd.DataFrame,
                                    demand_profiles: pd.DataFrame,
                                    params: dict) -> None:
    """Wrapper Kedro para reporting/cti_adjusted_analysis.py (mesmo padrao de I/O)."""
    from reporting.cti_adjusted_analysis import run
    run()


def _latex_effect_sizes_table(effect_sizes: pd.DataFrame, cfg: dict) -> str:
    """Tabela de Cohen's d."""
    sep = cfg.get("decimal_sep", "{,}")
    booktabs = cfg.get("booktabs", True)
    rule_top = "\\toprule" if booktabs else "\\hline"
    rule_mid = "\\midrule" if booktabs else "\\hline"
    rule_bot = "\\bottomrule" if booktabs else "\\hline"

    lines = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Tamanhos de efeito (Cohen's $d$) -- politica proposta vs baselines}",
        "\\label{tab:effect_sizes}",
        "\\begin{tabular}{llrrr}",
        rule_top,
        "Politica A & Politica B & Metrica & Cohen's $d$ & Magnitude \\\\",
        rule_mid,
    ]

    for _, row in effect_sizes.iterrows():
        lines.append(
            f"{row['policy_a']} & {row['policy_b']} & {row['metric']} & "
            f"${_fmt_num(row['cohens_d'], 3, sep)}$ & {row['magnitude']} \\\\"
        )

    lines += [rule_bot, "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
