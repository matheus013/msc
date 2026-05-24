from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.policy_selection.nodes import (
    generate_policy_labels,
    train_policy_selector,
    apply_policy_selector,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_policy_labels,
            inputs=["kpis", "demand_features", "params:policy_selection"],
            outputs="policy_labels",
            name="generate_policy_labels",
        ),
        node(
            func=train_policy_selector,
            inputs=["demand_features", "policy_labels", "params:policy_selection"],
            outputs=["policy_selector_model", "policy_selector_metrics"],
            name="train_policy_selector",
        ),
        node(
            func=apply_policy_selector,
            inputs=["demand_features", "policy_selector_model"],
            outputs="policy_recommendations",
            name="apply_policy_selector",
        ),
    ])
