from __future__ import annotations
from typing import Dict, List, Tuple, Callable
import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape(y_true, y_pred)}


def prepare_series(vendas_enrich: pd.DataFrame, target_col: str = "qty") -> pd.Series:
    s = (vendas_enrich.groupby("cycle", as_index=True)[target_col]
         .sum()
         .sort_index())
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def rolling_origin_forecast(
        y: pd.Series,
        horizon: int,
        fit_and_forecast_fn: Callable[[pd.Series, int], np.ndarray],
        min_train: int | None = None,
        step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward: expande o treino de forma incremental e NUNCA usa informação do futuro.
    Retorna (y_true_concat, y_pred_concat) acumulando previsões fora-da-amostra.
    """
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    if min_train is None:
        min_train = max(horizon, 6)  # um mínimo razoável

    for end in range(min_train, len(y) - horizon + 1, step):
        y_train = y.iloc[:end]
        y_test = y.iloc[end: end + horizon]
        y_hat = fit_and_forecast_fn(y_train, horizon)  # só treina com passado
        y_true_all.extend(y_test.to_numpy(dtype=float))
        y_pred_all.extend(np.asarray(y_hat, dtype=float))

    return np.asarray(y_true_all), np.asarray(y_pred_all)


def to_mlflow_history_payload(prefix: str, m: Dict[str, float], step: int = 1) -> Dict[str, Dict[str, float]]:
    """
    MlflowMetricsHistoryDataset espera dict de { "<nome>.MAE": {"value": float,"step":int}, ... }.
    """
    return {f"{prefix}.{k}": {"value": float(v), "step": int(step)} for k, v in m.items()}
