from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.data_ingestion.nodes import (
    load_raw_sales,
    filter_by_parameters,
    clean_sales_data,
    build_demand_scenarios,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_raw_sales,
            inputs=["vendas_partitioned", "params:data_ingestion"],
            outputs="sales_raw",
            name="load_raw_sales",
        ),
        node(
            func=filter_by_parameters,
            inputs=["sales_raw", "params:data_ingestion"],
            outputs="sales_filtered",
            name="filter_by_parameters",
        ),
        node(
            func=clean_sales_data,
            inputs=["sales_filtered", "params:data_cleaning"],
            outputs="sales_cleaned",
            name="clean_sales_data",
        ),
        node(
            func=build_demand_scenarios,
            inputs=["sales_cleaned", "params:data_ingestion"],
            outputs=["scenarios", "scenarios_meta"],
            name="build_demand_scenarios",
        ),
    ])
