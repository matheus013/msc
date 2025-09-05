from __future__ import annotations

from kedro.pipeline import Pipeline

from .pipelines.data_clean import create_pipeline as ds_clean_pipeline
from .pipelines.data_enrich import create_pipeline as ds_enrich_pipeline
# from .pipelines.forecast_tuning import create_pipeline as ds_forecast_tune
from .pipelines.forecasting import create_pipeline as ds_forecasting_pipeline
from .pipelines.reporting import create_pipeline as ds_reporting_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    clean_pipe = ds_clean_pipeline()
    enrich_pipe = ds_enrich_pipeline()
    reporting_pipe = ds_reporting_pipeline()
    forecasting_pipe = ds_forecasting_pipeline()
    # forecast_tune_pipe = ds_forecast_tune()

    return {
        "__default__": clean_pipe + enrich_pipe + reporting_pipe + forecasting_pipe ,
        "no-sample": enrich_pipe + reporting_pipe + forecasting_pipe ,
        # "forecast_tune": forecast_tune_pipe,
        "forecasting": forecasting_pipe,
        "reporting": reporting_pipe,
        "data_clean": clean_pipe,
        "data_enrich": enrich_pipe,
    }
