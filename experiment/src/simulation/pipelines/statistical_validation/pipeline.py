from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.statistical_validation.nodes import (
    run_wilcoxon_tests,
    run_friedman_nemenyi,
    compute_effect_sizes,
    stratified_analysis,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_wilcoxon_tests,
            inputs="kpis",
            outputs="wilcoxon_results",
            name="run_wilcoxon_tests",
        ),
        node(
            func=run_friedman_nemenyi,
            inputs="kpis",
            outputs="friedman_results",
            name="run_friedman_nemenyi",
        ),
        node(
            func=compute_effect_sizes,
            inputs="kpis",
            outputs="effect_sizes",
            name="compute_effect_sizes",
        ),
        node(
            func=stratified_analysis,
            inputs="kpis",
            outputs="stratified_summary",
            name="stratified_analysis",
        ),
    ])
