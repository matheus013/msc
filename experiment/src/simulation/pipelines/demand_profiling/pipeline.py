from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.demand_profiling.nodes import (
    compute_demand_features,
    classify_operational_profiles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_demand_features,
            inputs=["scenarios", "scenarios_meta", "params:demand_profiling"],
            outputs="demand_features",
            name="compute_demand_features",
        ),
        node(
            func=classify_operational_profiles,
            inputs=["demand_features", "params:demand_profiling"],
            outputs="demand_profiles",
            name="classify_operational_profiles",
        ),
    ])
