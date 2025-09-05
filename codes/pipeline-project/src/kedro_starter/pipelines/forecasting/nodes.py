from __future__ import annotations
from typing import Dict, Any, Tuple
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from .utils import (
    prepare_series,
    rolling_origin_forecast,
    metrics,
    to_mlflow_history_payload,
)

# =========================
# NAIVE
# =========================
def _fitpredict_naive(y_train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(y_train.iloc[-1], horizon)

def forecast_naive_node(
    vendas_enrich: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Outputs:
      - metrics_naive_mem (dict em memória, p/ seleção/tuning)
      - params_naive_out (json snapshot)
      - metrics_naive (dict para MlflowMetricsHistoryDataset)
    """
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(
        y, horizon=h, fit_and_forecast_fn=_fitpredict_naive, min_train=params.get("min_train")
    )
    m = metrics(y_true, y_pred)
    metrics_mem = {k: float(v) for k, v in m.items()}
    metrics_mlflow = to_mlflow_history_payload("naive", m, step=1)
    return metrics_mem, params, metrics_mlflow


# =========================
# HOLT-WINTERS
# =========================
def _fitpredict_hw(params: Dict[str, Any]):
    def _inner(y_train: pd.Series, horizon: int) -> np.ndarray:
        model = ExponentialSmoothing(
            y_train.values,
            trend=params.get("trend", "add"),
            seasonal=params.get("seasonal", "add") if params.get("seasonal_periods", 0) > 1 else None,
            seasonal_periods=params.get("seasonal_periods", None),
        )
        fitted = model.fit(optimized=True)
        return fitted.forecast(horizon)
    return _inner

def forecast_holt_winters_node(
    vendas_enrich: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Dict[str, float]]]:
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(
        y, h, _fitpredict_hw(params), min_train=params.get("min_train")
    )
    m = metrics(y_true, y_pred)
    metrics_mem = {k: float(v) for k, v in m.items()}
    metrics_mlflow = to_mlflow_history_payload("holt_winters", m, step=1)
    return metrics_mem, params, metrics_mlflow


# =========================
# SARIMAX
# =========================
def _fitpredict_sarimax(params: Dict[str, Any]):
    order = tuple(params.get("order", (1, 1, 1)))
    seasP, seasD, seasQ, s = tuple(
        params.get("seasonal_order", (1, 0, 1, max(1, params.get("seasonal_periods", 1))))
    )
    seasonal_order = (seasP, seasD, seasQ, s)

    def _inner(y_train: pd.Series, horizon: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train.values,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
        return res.forecast(steps=horizon)
    return _inner

def forecast_sarimax_node(
    vendas_enrich: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Dict[str, float]]]:
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(
        y, h, _fitpredict_sarimax(params), min_train=params.get("min_train")
    )
    m = metrics(y_true, y_pred)
    metrics_mem = {k: float(v) for k, v in m.items()}
    metrics_mlflow = to_mlflow_history_payload("sarimax", m, step=1)
    return metrics_mem, params, metrics_mlflow


# =========================
# XGBOOST (defasagens)
# =========================
def _make_lags(y: pd.Series, lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"y": y.values})
    for L in range(1, lags + 1):
        df[f"lag_{L}"] = df["y"].shift(L)
    return df.dropna()

def _fitpredict_xgb(params: Dict[str, Any]):
    lags = int(params.get("lags", 6))
    seed = int(params.get("random_seed", 42))

    def _inner(y_train: pd.Series, horizon: int) -> np.ndarray:
        df = _make_lags(y_train, lags)
        X, y = df.drop(columns=["y"]).values, df["y"].values
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            random_state=seed,
            n_jobs=0,
            objective="reg:squarederror",
        )
        model.fit(X, y)
        # previsão recursiva preservando causalidade
        hist = list(y_train.values)
        preds: list[float] = []
        for _ in range(horizon):
            feats = [hist[-k] for k in range(1, lags + 1)]
            Xh = np.array(feats, dtype=float).reshape(1, -1)
            yhat = float(model.predict(Xh)[0])
            preds.append(yhat)
            hist.append(yhat)
        return np.array(preds, dtype=float)
    return _inner

def forecast_xgb_node(
    vendas_enrich: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Dict[str, float]]]:
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(
        y, h, _fitpredict_xgb(params), min_train=params.get("min_train")
    )
    m = metrics(y_true, y_pred)
    metrics_mem = {k: float(v) for k, v in m.items()}
    metrics_mlflow = to_mlflow_history_payload("xgb", m, step=1)
    return metrics_mem, params, metrics_mlflow


# =========================
# PROPHET (opcional)
# =========================
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

def _fitpredict_prophet(params: Dict[str, Any]):
    monthly_start = pd.Timestamp(params.get("monthly_start", "2000-01-31"))

    def _inner(y_train: pd.Series, horizon: int) -> np.ndarray:
        # mapeia índice 'cycle' (inteiro/ordinal) para datas mensais artificiais
        idx = y_train.index
        ds = monthly_start + pd.to_timedelta((idx - idx.min()).astype(int), unit="MS")
        df = pd.DataFrame({"ds": ds, "y": y_train.values})

        m = Prophet(weekly_seasonality=False, daily_seasonality=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)

        last = ds.max()
        future_tail = pd.date_range(last, periods=horizon + 1, freq="M")[1:]
        future = pd.DataFrame({"ds": np.r_[ds, future_tail]})
        fcst = m.predict(future)
        return fcst["yhat"].tail(horizon).to_numpy(dtype=float)
    return _inner

def forecast_prophet_node(
    vendas_enrich: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Dict[str, float]]]:
    if not _HAS_PROPHET:
        # ainda assim retornamos 3 saídas
        empty = {}
        return empty, {"warning": "prophet_not_installed", **params}, empty

    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(
        y, h, _fitpredict_prophet(params), min_train=params.get("min_train")
    )
    m = metrics(y_true, y_pred)
    metrics_mem = {k: float(v) for k, v in m.items()}
    metrics_mlflow = to_mlflow_history_payload("prophet", m, step=1)
    return metrics_mem, params, metrics_mlflow

from typing import Dict, Any, Tuple

def choose_best_model_node(
    metrics_naive: Dict[str, Dict[str, float]],
    params_naive: Dict[str, Any],
    metrics_hw: Dict[str, Dict[str, float]],
    params_hw: Dict[str, Any],
    metrics_sarimax: Dict[str, Dict[str, float]],
    params_sarimax: Dict[str, Any],
    metrics_xgb: Dict[str, Dict[str, float]],
    params_xgb: Dict[str, Any],
    metrics_prophet: Dict[str, Dict[str, float]],
    params_prophet: Dict[str, Any],
    selection_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Escolhe o melhor modelo com base na métrica definida (por exemplo, 'RMSE').
    Retorna um dicionário com nome do modelo, métrica escolhida, valor e parâmetros usados.
    """
    metric_key = selection_params.get("metric", "RMSE")

    all_models = {
        "naive_last": (metrics_naive, params_naive),
        "holt_winters": (metrics_hw, params_hw),
        "sarimax": (metrics_sarimax, params_sarimax),
        "xgboost_lags": (metrics_xgb, params_xgb),
        "prophet": (metrics_prophet, params_prophet),
    }

    scores = {}
    for model_name, (m, p) in all_models.items():
        if not m:  # se o modelo não rodou ou está vazio
            continue
        # métrica está no formato {"value": ..., "step": ...}
        key = f"{model_name}.{metric_key}"
        score = m.get(key, {}).get("value", float("inf"))
        scores[model_name] = (score, p)

    best_model = min(scores.items(), key=lambda x: x[1][0])
    model_name, (best_score, best_params) = best_model

    return {
        "chosen_model": model_name,
        "metric": metric_key,
        "score": best_score,
        "params": best_params,
    }


import pandas as pd
from typing import Dict, Any, Tuple

def tune_best_model_node(
    vendas_enrich: pd.DataFrame,
    chosen_model: Dict[str, Any],
    tuning_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Faz tuning do modelo escolhido usando um grid de hiperparâmetros simples.
    Retorna:
    - best_params: dicionário com parâmetros otimizados
    - metrics_df: DataFrame com resultados de todos os testes
    """
    model_name = chosen_model["chosen_model"]
    grid = tuning_params.get(model_name, [])

    if not grid:
        return chosen_model["params"], pd.DataFrame([chosen_model])

    results = []
    best_score = float("inf")
    best_params = None

    # simples: avalia RMSE com rolling_origin_forecast
    from .utils import prepare_series, rolling_origin_forecast, metrics
    y = prepare_series(vendas_enrich, target_col="qty")
    horizon = int(chosen_model.get("params", {}).get("horizon", 1))

    # função dinâmica
    from .nodes import (
        _fitpredict_naive, _fitpredict_hw, _fitpredict_sarimax,
        _fitpredict_xgb, _fitpredict_prophet
    )
    mapping = {
        "naive_last": lambda p: _fitpredict_naive,
        "holt_winters": _fitpredict_hw,
        "sarimax": _fitpredict_sarimax,
        "xgboost_lags": _fitpredict_xgb,
        "prophet": _fitpredict_prophet,
    }
    fit_fn = mapping[model_name]

    for p in grid:
        try:
            y_true, y_pred = rolling_origin_forecast(y, horizon, fit_fn(p), min_train=p.get("min_train", None))
            m = metrics(y_true, y_pred)
            row = {"params": p, "RMSE": m["RMSE"], "MAE": m["MAE"], "MAPE": m["MAPE"]}
            results.append(row)
            if m["RMSE"] < best_score:
                best_score = m["RMSE"]
                best_params = p
        except Exception as e:
            results.append({"params": p, "error": str(e)})

    df = pd.DataFrame(results)
    return best_params, df
