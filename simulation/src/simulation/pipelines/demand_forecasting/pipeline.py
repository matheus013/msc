from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.demand_forecasting.nodes import (
    build_walkforward_splits,
    run_walkforward_forecasting,
    evaluate_forecast_accuracy,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_walkforward_splits,
            inputs=["scenarios", "params:forecasting"],
            outputs="walkforward_splits",
            name="build_walkforward_splits",
        ),
        node(
            func=run_walkforward_forecasting,
            inputs=["walkforward_splits", "params:forecasting"],
            outputs=["forecast_predictions", "trained_forecasters"],
            name="run_walkforward_forecasting",
        ),
        node(
            func=evaluate_forecast_accuracy,
            inputs="forecast_predictions",
            outputs="forecast_metrics",
            name="evaluate_forecast_accuracy",
        ),
    ])
