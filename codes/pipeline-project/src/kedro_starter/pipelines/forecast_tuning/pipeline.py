# # src/kedro_starter/pipelines/forecast_tune/pipeline.py
# from __future__ import annotations
#
# from kedro.pipeline import Pipeline, node
#
# from .nodes import select_best_and_tune
#
#
# def create_pipeline() -> Pipeline:
#     return Pipeline([
#         node(
#             func=select_best_and_tune,
#             inputs=[
#                 "vendas_enrich",
#                 "params:seasonal_periods",
#                 "params:forecast_horizon",
#
#                 # IMPORTANTe: estes inputs devem vir das SAÍDAS dos nodes de previsão,
#                 # não do catálogo (MlflowMetricsHistoryDataset é write-only).
#                 # Ou seja, nos pipelines dos modelos, defina outputs extras
#                 # "metrics_naive_mem", ... e aponte aqui.
#                 # Se você já está passando as métricas em memória com estes nomes,
#                 # basta manter assim:
#
#                 "metrics_naive_mem", "params_naive_out",
#                 "metrics_holt_winters_mem", "params_holt_winters_out",
#                 "metrics_sarimax_mem", "params_sarimax_out",
#                 "metrics_xgb_mem", "params_xgb_out",
#                 "metrics_prophet_mem", "params_prophet_out",
#             ],
#             outputs={
#                 "chosen_model": "chosen_model",
#                 "best_params": "best_forecast_params",
#                 "tuned_metrics": "tuned_metrics",
#             },
#             name="select_best_and_tune",
#             tags=["tune", "model-selection"],
#         ),
#     ])
