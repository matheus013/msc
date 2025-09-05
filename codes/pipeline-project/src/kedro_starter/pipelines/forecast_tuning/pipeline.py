# src/kedro_starter/pipelines/forecast_tune/pipeline.py
from __future__ import annotations
from kedro.pipeline import Pipeline, node
from .nodes import select_best_and_tune

def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=select_best_and_tune,
            inputs=[
                "vendas_enrich",
                "params:seasonal_periods",
                "params:forecast_horizon",
                # métricas e params produzidos pelos nodes de cada método:
                "metrics_naive_last", "params_naive_last",
                "metrics_holt_winters", "params_holt_winters",
                "metrics_sarimax", "params_sarimax",
                "metrics_xgboost_lags", "params_xgboost_lags",
                # prophet é opcional — registre vazio se não usar
                "metrics_prophet", "params_prophet",
            ],
            outputs={
                "chosen_model": "chosen_model",
                "best_params": "best_forecast_params",
                "tuned_metrics": "tuned_metrics",
            },
            name="select_best_and_tune",
            tags=["tune", "model-selection"],
        ),
    ])
