# src/demand_forecast_pipeline/pipelines/model_training/pipeline.py

from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_model,
            inputs="processed_data",
            outputs="trained_model",
            name="train_model_node"
        )
    ])

def train_model(data):
    # Treinamento fictício
    return {"model": "dummy_model"}
