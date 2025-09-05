from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from .utils import prepare_series, rolling_origin_forecast, metrics, to_mlflow_history_payload


def _fitpredict_naive(y_train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(y_train.iloc[-1], horizon)


def forecast_naive_node(
        vendas_enrich: pd.DataFrame,
        params: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(y, horizon=h, fit_and_forecast_fn=_fitpredict_naive,
                                             min_train=params.get("min_train"))
    m = metrics(y_true, y_pred)
    return to_mlflow_history_payload("naive_last", m, step=1), params


from statsmodels.tsa.holtwinters import ExponentialSmoothing


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


def forecast_holt_winters_node(vendas_enrich: pd.DataFrame, params: Dict[str, Any]):
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(y, h, _fitpredict_hw(params), min_train=params.get("min_train"))
    m = metrics(y_true, y_pred)
    return to_mlflow_history_payload("holt_winters", m, step=1), params


import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _fitpredict_sarimax(params: Dict[str, Any]):
    order = tuple(params.get("order", (1, 1, 1)))
    seasP, seasD, seasQ, s = tuple(params.get("seasonal_order", (1, 0, 1, max(1, params.get("seasonal_periods", 1)))))
    seasonal_order = (seasP, seasD, seasQ, s)

    def _inner(y_train: pd.Series, horizon: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y_train.values, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
        return res.forecast(steps=horizon)

    return _inner


def forecast_sarimax_node(vendas_enrich: pd.DataFrame, params: Dict[str, Any]):
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(y, h, _fitpredict_sarimax(params), min_train=params.get("min_train"))
    m = metrics(y_true, y_pred)
    return to_mlflow_history_payload("sarimax", m, step=1), params


import numpy as np
from xgboost import XGBRegressor


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
        hist = list(y_train.values)
        preds = []
        for _ in range(horizon):
            feats = [hist[-k] for k in range(1, lags + 1)]
            yhat = float(model.predict(np.array(feats).reshape(1, -1))[0])
            preds.append(yhat)
            hist.append(yhat)
        return np.array(preds, dtype=float)

    return _inner


def forecast_xgb_node(vendas_enrich: pd.DataFrame, params: Dict[str, Any]):
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(y, h, _fitpredict_xgb(params), min_train=params.get("min_train"))
    m = metrics(y_true, y_pred)
    return to_mlflow_history_payload("xgboost_lags", m, step=1), params


import warnings

try:
    from prophet import Prophet

    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False


def _fitpredict_prophet(params: Dict[str, Any]):
    monthly_start = pd.Timestamp(params.get("monthly_start", "2000-01-31"))

    def _inner(y_train: pd.Series, horizon: int) -> np.ndarray:
        # mapeia cycle->datas mensais para o Prophet
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


def forecast_prophet_node(vendas_enrich: pd.DataFrame, params: Dict[str, Any]):
    if not _HAS_PROPHET:
        # retorna payload vazio e params (para rastrear tentativa)
        return {}, {"warning": "prophet_not_installed", **params}
    y = prepare_series(vendas_enrich, target_col=params.get("target_col", "qty"))
    h = int(params["horizon"])
    y_true, y_pred = rolling_origin_forecast(y, h, _fitpredict_prophet(params), min_train=params.get("min_train"))
    m = metrics(y_true, y_pred)
    return to_mlflow_history_payload("prophet", m, step=1), params
