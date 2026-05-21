# src/demand_forecast_pipeline/pipeline_registry.py

from kedro.pipeline import Pipeline
from demand_forecast_pipeline.pipelines import data_processing, model_training

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()

    return {
        "__default__": data_processing_pipeline + model_training_pipeline,
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,
    }
