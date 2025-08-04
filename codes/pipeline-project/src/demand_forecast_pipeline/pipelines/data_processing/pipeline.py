# src/demand_forecast_pipeline/pipelines/data_processing/pipeline.py

from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=process_data,
            inputs="raw_data",
            outputs="processed_data",
            name="process_data_node"
        )
    ])

# Função de exemplo — troque pela sua lógica real
def process_data(data):
    # Processamento fictício
    return data
