from kedro.pipeline import Pipeline, node
from .nodes import (
    forecast_naive_node, forecast_holt_winters_node, forecast_sarimax_node,
    forecast_xgb_node, forecast_prophet_node
)


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(forecast_naive_node,
             inputs=dict(vendas_enrich="vendas_enrich", params="params:naive_params"),
             outputs=["metrics_naive", "params_naive_out"],
             name="forecast_naive"),

        node(forecast_holt_winters_node,
             inputs=dict(vendas_enrich="vendas_enrich", params="params:holt_winters_params"),
             outputs=["metrics_holt_winters", "params_holt_winters_out"],
             name="forecast_holt_winters"),

        node(forecast_sarimax_node,
             inputs=dict(vendas_enrich="vendas_enrich", params="params:sarimax_params"),
             outputs=["metrics_sarimax", "params_sarimax_out"],
             name="forecast_sarimax"),

        node(forecast_xgb_node,
             inputs=dict(vendas_enrich="vendas_enrich", params="params:xgb_params"),
             outputs=["metrics_xgb", "params_xgb_out"],
             name="forecast_xgb"),

        node(forecast_prophet_node,
             inputs=dict(vendas_enrich="vendas_enrich", params="params:prophet_params"),
             outputs=["metrics_prophet", "params_prophet_out"],
             name="forecast_prophet"),
    ])
