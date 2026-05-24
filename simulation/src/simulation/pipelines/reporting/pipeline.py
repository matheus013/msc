from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.reporting.nodes import (
    generate_comparison_plots,
    generate_demand_plots,
    generate_map_plots,
    generate_latex_tables,
    generate_dissertation_report,
)


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
    ])
