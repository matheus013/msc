# src/kedro_starter/pipelines/forecast_tune/nodes.py
from __future__ import annotations

import itertools
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


# ---------- util ----------
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = _mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def _prepare_series(vendas_enrich: pd.DataFrame, target_col: str = "qty") -> pd.Series:
    s = (vendas_enrich.groupby("cycle", as_index=True)[target_col]
         .sum().sort_index())
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _walk_forward_eval(y: pd.Series, horizon: int, fit_func) -> Tuple[np.ndarray, Dict[str, float]]:
    preds, actuals = [], []
    start = max(horizon, 3)
    for t in range(start, len(y)):
        y_train = y.iloc[:t]
        y_test = y.iloc[t:t + 1]  # 1 passo
        yhat = fit_func(y_train, 1)
        yhat = float(np.asarray(yhat).ravel()[0])
        preds.append(yhat)
        actuals.append(float(y_test.values[0]))
    metrics = _metrics(np.array(actuals, float), np.array(preds, float))
    return np.array(preds, float), metrics


def _fit_hw(y_train: pd.Series, steps: int, trend="add", seasonal="add", seasonal_periods=6):
    model = ExponentialSmoothing(
        y_train.values,
        trend=trend,
        seasonal=seasonal if seasonal_periods > 1 else None,
        seasonal_periods=seasonal_periods if seasonal_periods > 1 else None,
    )
    return model.fit(optimized=True).forecast(steps)


def _fit_sarimax(y_train: pd.Series, steps: int, order=(1, 1, 1), seasonal_order=(1, 0, 1, 6)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y_train.values, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return res.forecast(steps=steps)

def _fit_xgb_lags(y_train: pd.Series, steps: int, lags: int = 6, **xgb_kwargs):
    # treino
    df = pd.DataFrame({"y": y_train.values})
    for L in range(1, lags + 1):
        df[f"lag_{L}"] = df["y"].shift(L)
    df.dropna(inplace=True)
    X, y = df.drop(columns=["y"]).values, df["y"].values
    model = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.9,
        objective="reg:squarederror", n_jobs=0, **xgb_kwargs
    )
    model.fit(X, y)
    # previsão recursiva
    hist = list(y_train.values)
    preds = []
    for _ in range(steps):
        feats = [hist[-k] for k in range(1, lags + 1)]
        Xh = np.array(feats, float).reshape(1, -1)
        yhat = float(model.predict(Xh)[0])
        preds.append(yhat)
        hist.append(yhat)
    return np.array(preds, float)


def _rmse_from_metrics_obj(m) -> float:
    """
    Aceita:
      - dict no padrão MlflowMetricsHistoryDataset: {"RMSE": {"value": float, "step": int}, ...}
        ou {"x.RMSE": {"value": ...}, ...}
      - DataFrame com colunas ["metric","value"] (ou ["name","value"])
    Retorna float(RMSE).
    """
    if isinstance(m, dict):
        # chaves possíveis: "RMSE" ou "algo.RMSE"
        # 1) direto
        if "RMSE" in m and isinstance(m["RMSE"], dict) and "value" in m["RMSE"]:
            return float(m["RMSE"]["value"])
        # 2) procurar *.RMSE
        for k, v in m.items():
            if k.upper().endswith(".RMSE") and isinstance(v, dict) and "value" in v:
                return float(v["value"])
        # 3) plano: {"RMSE": 123}
        if "RMSE" in m and isinstance(m["RMSE"], (int, float)):
            return float(m["RMSE"])
        raise ValueError("Não foi possível encontrar RMSE no dict de métricas.")
    # DataFrame
    df = pd.DataFrame(m).copy()
    name_col = "metric" if "metric" in df.columns else ("name" if "name" in df.columns else None)
    if name_col is None:
        raise ValueError("DataFrame de métricas sem coluna 'metric' ou 'name'.")
    key = df[name_col].str.upper() == "RMSE"
    return float(df.loc[key, "value"].iloc[0])


# ---------- node principal ----------
def select_best_and_tune(
    vendas_enrich: pd.DataFrame,
    seasonal_periods: int,
    forecast_horizon: int,

        # métricas/params vinda EM MEMÓRIA dos nodes dos modelos
        metrics_naive: dict | pd.DataFrame,
        params_naive_out: Dict[str, Any],
        metrics_holt_winters: dict | pd.DataFrame,
        params_holt_winters_out: Dict[str, Any],
        metrics_sarimax: dict | pd.DataFrame,
        params_sarimax_out: Dict[str, Any],
        metrics_xgb: dict | pd.DataFrame,
        params_xgb_out: Dict[str, Any],
        metrics_prophet: dict | pd.DataFrame | None = None,
        params_prophet_out: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Seleciona melhor método por RMSE e faz tuning (walk-forward) com grid pequena.
    Retorna dict com 3 saídas mapeadas no pipeline:
      - chosen_model (json)
      - best_params (json)
      - tuned_metrics (DataFrame)
    """
    pools = {
        "naive": (_rmse_from_metrics_obj(metrics_naive), params_naive_out or {}),
        "holt_winters": (_rmse_from_metrics_obj(metrics_holt_winters), params_holt_winters_out or {}),
        "sarimax": (_rmse_from_metrics_obj(metrics_sarimax), params_sarimax_out or {}),
        "xgb": (_rmse_from_metrics_obj(metrics_xgb), params_xgb_out or {}),
    }
    if metrics_prophet is not None:
        try:
            pools["prophet"] = (_rmse_from_metrics_obj(metrics_prophet), params_prophet_out or {})
        except Exception:
            pass  # se não houver métricas válidas de prophet, ignora

    best_model = min(pools.items(), key=lambda kv: kv[1][0])[0]
    y = _prepare_series(vendas_enrich, target_col="qty")

    best_params: Dict[str, Any] = {}
    best_score = np.inf
    tuned_metrics: Dict[str, float] = {}

    if best_model == "holt_winters":
        for trend, seasonal in itertools.product(["add", "mul"], ["add", "mul", None]):
            def _fitf(yt, st, tr=trend, se=seasonal):
                return _fit_hw(yt, st, trend=tr, seasonal=se, seasonal_periods=seasonal_periods)

            _, m = _walk_forward_eval(y, forecast_horizon, _fitf)
            if m["RMSE"] < best_score:
                best_score, tuned_metrics = m["RMSE"], m
                best_params = {"trend": trend, "seasonal": seasonal, "seasonal_periods": seasonal_periods}

    elif best_model == "sarimax":
        for o in [(1, 1, 1), (2, 1, 1)]:
            for so in [(1, 0, 1, seasonal_periods), (1, 1, 1, seasonal_periods)]:
                def _fitf(yt, st, order=o, seasonal_order=so):
                    return _fit_sarimax(yt, st, order=order, seasonal_order=seasonal_order)

                _, m = _walk_forward_eval(y, forecast_horizon, _fitf)
                if m["RMSE"] < best_score:
                    best_score, tuned_metrics = m["RMSE"], m
                    best_params = {"order": o, "seasonal_order": so}

    elif best_model == "xgb":
        for lags in [max(3, min(6, seasonal_periods)), max(6, min(12, seasonal_periods))]:
            for lr in [0.05, 0.1]:
                for md in [3, 4, 5]:
                    def _fitf(yt, st, l=lags, lr_=lr, md_=md):
                        return _fit_xgb_lags(yt, st, lags=l, learning_rate=lr_, max_depth=md_)

                    _, m = _walk_forward_eval(y, forecast_horizon, _fitf)
                    if m["RMSE"] < best_score:
                        best_score, tuned_metrics = m["RMSE"], m
                        best_params = {"lags": lags, "learning_rate": lr, "max_depth": md}

    else:
        # naive/prophet: sem grid aqui; reavalia via walk-forward do naive
        def _fit_naive(yt, st):
            return np.repeat(yt.iloc[-1], st)

        _, tuned_metrics = _walk_forward_eval(y, forecast_horizon, _fit_naive)
        best_params = pools[best_model][1]

    chosen = {"model": best_model}
    tuned = {"model": best_model, "params": best_params}
    tuned_metrics_df = pd.DataFrame([{"metric": k, "value": float(v)} for k, v in tuned_metrics.items()])

    return {
        "chosen_model": chosen,
        "best_params": tuned,
        "tuned_metrics": tuned_metrics_df,
    }
