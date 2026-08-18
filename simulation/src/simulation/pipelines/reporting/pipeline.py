from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.reporting.nodes import (
    generate_comparison_plots,
    generate_demand_plots,
    generate_map_plots,
    generate_latex_tables,
    generate_dissertation_report,
    generate_profile_policy_analysis,
    generate_strategy_cost_comparison,
    generate_cti_adjusted_analysis,
    generate_presentation_visuals,
)


def create_profile_analysis_pipeline(**kwargs) -> Pipeline:
    """
    2026-08-18: sub-pipeline isolado, so com os nos de analise por perfil
    que dependem apenas de `kpis` + `demand_profiles` -- ao contrario de
    `create_pipeline()` (que tem nos dependentes de
    `demand_forecasting`/`statistical_validation`, nao incluidos em
    benchmark_bot/benchmark_m5). Pensado pra rodar DEPOIS que essas
    pipelines terminarem, sem reexecutar a simulacao:
    `kedro run --pipeline profile_analysis [--env bot|m5]`.

    `generate_strategy_cost_comparison`: responde diretamente "a selecao
    por perfil e melhor que a unica global?" (estrategias A1/A2/B/C +
    teste de Wilcoxon pareado, pedido explicito do usuario).
    """
    return pipeline([
        node(
            func=generate_profile_policy_analysis,
            inputs=["kpis", "demand_profiles", "params:reporting"],
            outputs=None,
            name="generate_profile_policy_analysis",
        ),
        node(
            func=generate_strategy_cost_comparison,
            inputs=["kpis", "demand_profiles", "params:reporting"],
            outputs=None,
            name="generate_strategy_cost_comparison",
        ),
    ])


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_comparison_plots,
            inputs=[
                "kpis", "wilcoxon_results", "friedman_results",
                "effect_sizes", "params:reporting",
            ],
            outputs="comparison_plots",
            name="generate_comparison_plots",
        ),
        node(
            func=generate_demand_plots,
            inputs=[
                "scenarios", "scenarios_meta",
                "forecast_predictions", "forecast_metrics",
                "params:reporting",
            ],
            outputs="demand_plots",
            name="generate_demand_plots",
        ),
        node(
            func=generate_map_plots,
            inputs=["scenarios_meta", "kpis", "params:reporting"],
            outputs="map_plots",
            name="generate_map_plots",
        ),
        node(
            func=generate_latex_tables,
            inputs=[
                "kpis", "wilcoxon_results", "effect_sizes",
                "stratified_summary", "params:reporting",
            ],
            outputs="latex_tables",
            name="generate_latex_tables",
        ),
        node(
            func=generate_dissertation_report,
            inputs=[
                "scenarios_meta", "forecast_metrics", "kpis",
                "params:reporting",
            ],
            outputs="dissertation_report",
            name="generate_dissertation_report",
        ),
        node(
            func=generate_profile_policy_analysis,
            inputs=["kpis", "demand_profiles", "params:reporting"],
            outputs=None,
            name="generate_profile_policy_analysis",
        ),
        node(
            func=generate_strategy_cost_comparison,
            inputs=["kpis", "demand_profiles", "params:reporting"],
            outputs=None,
            name="generate_strategy_cost_comparison",
        ),
        node(
            func=generate_cti_adjusted_analysis,
            inputs=["kpis", "demand_profiles", "params:reporting"],
            outputs=None,
            name="generate_cti_adjusted_analysis",
        ),
        node(
            func=generate_presentation_visuals,
            inputs=["kpis", "demand_profiles", "params:reporting"],
            outputs=None,
            name="generate_presentation_visuals",
        ),
    ])
