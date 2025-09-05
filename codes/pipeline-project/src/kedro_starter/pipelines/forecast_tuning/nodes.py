# src/kedro_starter/pipelines/forecast_tune/nodes.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import warnings

# -------- utilidades ----------
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
    """
    Walk-forward (rolling-origin): para t em [start .. len(y)-horizon):
      - treina em y[:t], prevê y[t:t+horizon] (um passo à frente por padrão)
    Aqui usamos 1 passo por iteração e repetimos horizon vezes no final do histórico.
    """
    preds, actuals = [], []
    # janela inicial precisa ser maior que o horizonte
    start = max(horizon, 3)
    for t in range(start, len(y)):
        y_train = y.iloc[:t]
        y_test = y.iloc[t:t+1]  # 1 passo à frente
        yhat = fit_func(y_train, 1)  # prevê 1 passo
        if np.size(yhat) == 1:
            preds.append(float(yhat))
        else:
            preds.append(float(np.asarray(yhat).ravel()[0]))
        actuals.append(float(y_test.values[0]))
    metrics = _metrics(np.array(actuals, float), np.array(preds, float))
    return np.array(preds, float), metrics

# -------- modelos básicos ----------
def _fit_hw(y_train: pd.Series, steps: int, trend: str = "add",
            seasonal: str | None = "add", seasonal_periods: int = 6):
    model = ExponentialSmoothing(
        y_train.values,
        trend=trend,
        seasonal=seasonal if seasonal_periods > 1 else None,
        seasonal_periods=seasonal_periods if seasonal_periods > 1 else None,
    )
    return model.fit(optimized=True).forecast(steps)

def _fit_sarimax(y_train: pd.Series, steps: int,
                 order=(1, 1, 1), seasonal_order=(1, 0, 1, 6)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y_train.values, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return res.forecast(steps=steps)

def _fit_xgb_lags(y_train: pd.Series, steps: int, lags: int = 6, **xgb_kwargs):
    # constrói features defasadas a partir do histórico completo e faz previsão recursiva
    def _make_feats(hist: List[float]) -> np.ndarray:
        feats = [hist[-k] for k in range(1, lags + 1)]
        return np.array(feats, dtype=float).reshape(1, -1)

    # treina no histórico com muitas amostras (dropna nas janelas)
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

    hist = list(y_train.values)
    preds = []
    for _ in range(steps):
        Xh = _make_feats(hist)
        yhat = float(model.predict(Xh)[0])
        preds.append(yhat)
        hist.append(yhat)
    return np.array(preds, float)

# -------- seleção e tuning ----------
def select_best_and_tune(
    vendas_enrich: pd.DataFrame,
    seasonal_periods: int,
    forecast_horizon: int,
    # métricas e params salvos de cada método
    metrics_naive: pd.DataFrame,
    params_naive: Dict[str, Any],
    metrics_holt_winters: pd.DataFrame,
    params_holt_winters: Dict[str, Any],
    metrics_sarimax: pd.DataFrame,
    params_sarimax: Dict[str, Any],
    metrics_xgboost_lags: pd.DataFrame,
    params_xgboost_lags: Dict[str, Any],
    # prophet é opcional: se não usar, passe DataFrame vazio e {}
    metrics_prophet: pd.DataFrame | None = None,
    params_prophet: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    - decide o melhor método por RMSE
    - roda tuning (walk-forward) numa grid pequena
    - retorna artefatos para catalog: chosen_model, best_params, tuned_metrics
    """
    # 1) junta métricas (espera colunas: metric, value)
    def _rmse_of(df: pd.DataFrame) -> float:
        df = df.copy()
        key = df["metric"].str.upper() == "RMSE"
        return float(df.loc[key, "value"].iloc[0])

    pools = {
        "naive_last": (_rmse_of(metrics_naive), params_naive),
        "holt_winters": (_rmse_of(metrics_holt_winters), params_holt_winters),
        "sarimax": (_rmse_of(metrics_sarimax), params_sarimax),
        "xgboost_lags": (_rmse_of(metrics_xgboost_lags), params_xgboost_lags),
    }
    if metrics_prophet is not None and not metrics_prophet.empty:
        pools["prophet"] = (_rmse_of(metrics_prophet), params_prophet or {})

    best_model = min(pools.items(), key=lambda kv: kv[1][0])[0]

    # 2) série completa e função de treino para cada grade
    y = _prepare_series(vendas_enrich, target_col="qty")

    best_params: Dict[str, Any] = {}
    best_score: float = np.inf
    tuned_metrics: Dict[str, float] = {}

    if best_model == "holt_winters":
        trend_opts = ["add", "mul"]
        seasonal_opts = ["add", "mul", None]
        grid = itertools.product(trend_opts, seasonal_opts)
        for trend, seasonal in grid:
            def _fitfunc(y_train, steps):
                return _fit_hw(y_train, steps, trend=trend,
                               seasonal=seasonal, seasonal_periods=seasonal_periods)
            _, m = _walk_forward_eval(y, forecast_horizon, _fitfunc)
            if m["RMSE"] < best_score:
                best_score = m["RMSE"]
                best_params = {"trend": trend, "seasonal": seasonal, "seasonal_periods": seasonal_periods}
                tuned_metrics = m

    elif best_model == "sarimax":
        # grade pequena e robusta
        orders = [(1,1,1), (2,1,1)]
        seas = [(1,0,1, seasonal_periods), (1,1,1, seasonal_periods)]
        for o, so in itertools.product(orders, seas):
            def _fitfunc(y_train, steps, order=o, seasonal_order=so):
                return _fit_sarimax(y_train, steps, order=order, seasonal_order=seasonal_order)
            _, m = _walk_forward_eval(y, forecast_horizon, _fitfunc)
            if m["RMSE"] < best_score:
                best_score = m["RMSE"]
                best_params = {"order": o, "seasonal_order": so}
                tuned_metrics = m

    elif best_model == "xgboost_lags":
        lags_opts = [max(3, min(6, seasonal_periods)), max(6, min(12, seasonal_periods))]
        eta = [0.05, 0.1]
        depth = [3, 4, 5]
        for l, lr, md in itertools.product(lags_opts, eta, depth):
            def _fitfunc(y_train, steps, lags=l, lr=lr, md=md):
                return _fit_xgb_lags(y_train, steps, lags=lags, learning_rate=lr, max_depth=md)
            _, m = _walk_forward_eval(y, forecast_horizon, _fitfunc)
            if m["RMSE"] < best_score:
                best_score = m["RMSE"]
                best_params = {"lags": l, "learning_rate": lr, "max_depth": md}
                tuned_metrics = m

    else:
        # naïve/prophet: sem tuning aqui
        def _fit_naive(y_train, steps):
            return np.repeat(y_train.iloc[-1], steps)
        fitf = _fit_naive if best_model == "naive_last" else (lambda yt, st: np.repeat(yt.iloc[-1], st))
        _, tuned_metrics = _walk_forward_eval(y, forecast_horizon, fitf)
        best_params = pools[best_model][1] or {}

    # 3) payloads de saída (para salvar no catálogo)
    chosen = {"model": best_model}
    tuned = {"model": best_model, "params": best_params}
    tuned_metrics_df = pd.DataFrame(
        [{"metric": k, "value": float(v)} for k, v in tuned_metrics.items()]
    )
    return {
        "chosen_model": chosen,
        "best_params": tuned,
        "tuned_metrics": tuned_metrics_df,
    }
