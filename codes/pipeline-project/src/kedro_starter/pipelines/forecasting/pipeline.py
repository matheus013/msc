from __future__ import annotations
from kedro.pipeline import Pipeline, node

from .nodes import (
    forecast_naive_node,
    forecast_holt_winters_node,
    forecast_sarimax_node,
    forecast_xgb_node,
    forecast_prophet_node,
    choose_best_model_node,
    tune_best_model_node,
)

def create_pipeline() -> Pipeline:
    return Pipeline([
        # --- cada método: 3 saídas (mem, params_out, metrics_mlflow)
        node(
            func=forecast_naive_node,
            inputs=dict(vendas_enrich="vendas_enrich", params="params:forecast.naive"),
            outputs=["metrics_naive_mem", "params_naive_out", "metrics_naive"],
            name="forecast_naive",
            tags=["forecast"],
        ),
        node(
            func=forecast_holt_winters_node,
            inputs=dict(vendas_enrich="vendas_enrich", params="params:forecast.holt_winters"),
            outputs=["metrics_holt_winters_mem", "params_holt_winters_out", "metrics_holt_winters"],
            name="forecast_holt_winters",
            tags=["forecast"],
        ),
        node(
            func=forecast_sarimax_node,
            inputs=dict(vendas_enrich="vendas_enrich", params="params:forecast.sarimax"),
            outputs=["metrics_sarimax_mem", "params_sarimax_out", "metrics_sarimax"],
            name="forecast_sarimax",
            tags=["forecast"],
        ),
        node(
            func=forecast_xgb_node,
            inputs=dict(vendas_enrich="vendas_enrich", params="params:forecast.xgb"),
            outputs=["metrics_xgb_mem", "params_xgb_out", "metrics_xgb"],
            name="forecast_xgb",
            tags=["forecast"],
        ),
        node(
            func=forecast_prophet_node,
            inputs=dict(vendas_enrich="vendas_enrich", params="params:forecast.prophet"),
            outputs=["metrics_prophet_mem", "params_prophet_out", "metrics_prophet"],
            name="forecast_prophet",
            tags=["forecast"],
        ),

        # --- seleção do melhor modelo (lê TODOS os resultados em memória + params)
        node(
            func=choose_best_model_node,
            inputs=[
                "metrics_naive_mem", "params_naive_out",
                "metrics_holt_winters_mem", "params_holt_winters_out",
                "metrics_sarimax_mem", "params_sarimax_out",
                "metrics_xgb_mem", "params_xgb_out",
                "metrics_prophet_mem", "params_prophet_out",
                "params:forecast.selection",   # ex.: {"metric": "RMSE"}
            ],
            outputs="chosen_model",  # JSONDataset no seu catálogo
            name="choose_best_model",
            tags=["forecast", "selection"],
        ),

        # --- tuning do melhor (usa chosen_model + vendas_enrich + grid)
        node(
            func=tune_best_model_node,
            inputs=[
                "vendas_enrich",
                "chosen_model",
                "params:forecast.tuning",      # grids por método
            ],
            outputs=["best_forecast_params", "tuned_metrics"],  # ambos já no catálogo
            name="tune_best_model",
            tags=["forecast", "tuning"],
        ),
    ])
