"""
demand_forecasting/nodes.py — Previsão com validação walk-forward temporal.

Princípio: para prever o ciclo t, apenas dados dos ciclos 0..t-1 são usados.

Dois modos (parâmetro forecasting.walkforward):
  false (padrão/rápido)  — treina UMA vez nos primeiros min_train_cycles,
                           prevê cada ciclo de teste com a janela correta (sem re-treino).
                           Garante ausência de vazamento; não usa dados futuros no treino.
  true  (rigoroso/lento) — re-treina a cada novo ciclo (janela expansível real).
                           Mais correto para dissertação; ~10× mais lento.

Saídas:
  forecast_predictions  — DataFrame [warehouse, store_id, item_id, cycle,
                          model, predicted, actual, n_train]
  trained_forecasters   — modelos treinados em TODOS os dados disponíveis
                          (usados pelo inventory_simulation como referência)
  forecast_metrics      — MAE/RMSE/MAPE agregado e por ciclo
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Construção dos splits temporais
# ══════════════════════════════════════════════════════════════════════════════

def build_walkforward_splits(scenarios: pd.DataFrame, params: dict) -> dict:
    """
    Para cada série (warehouse, store_id, item_id), registra:
      - Série completa ordenada por ciclo
      - Rótulos de ciclo por posição
      - Índice do primeiro ciclo de teste (min_train_cycles)

    Nenhum dado é descartado — o split real é feito em run_walkforward_forecasting.
    """
    lookback = params.get("lookback", 6)
    min_train_cycles = params.get("min_train_cycles", 17)

    splits = {}
    for (w, s, i), grp in scenarios.groupby(["warehouse", "store_id", "item_id"]):
        grp = grp.sort_values("venda_ciclo")
        cycles = grp["venda_ciclo"].astype(str).tolist()
        series = grp["demand"].values.astype(float)

        n = len(series)
        if n < min_train_cycles + 1:
            log.debug("Série curta ignorada (%s,%s,%s): %d ciclos < min_train=%d",
                      w, s, i, n, min_train_cycles)
            continue

        splits[(w, s, i)] = {
            "cycles":           cycles,
            "series":           series,
            "lookback":         lookback,
            "min_train_cycles": min_train_cycles,
            # primeiro índice que pode ser PREVISTO (tudo antes é treino)
            "first_test_idx":   min_train_cycles,
        }

    log.info("Walk-forward splits: %d séries | lookback=%d | min_train=%d",
             len(splits), lookback, min_train_cycles)
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 2. Execução walk-forward + treinamento do modelo final
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward_forecasting(walkforward_splits: dict,
                                params: dict) -> tuple:
    """
    Executa o ciclo de previsão temporal e retorna:
      (forecast_predictions_df, trained_forecasters_dict)

    Para cada ciclo t no período de teste:
      - Apenas demand[0:t] é visto pelo modelo
      - Predição gerada para demand[t]
      - demand[t] real registrado junto com a predição

    Ao final, treina o modelo definitivo em TODA a série (para uso na simulação).
    """
    walkforward_mode = params.get("walkforward", False)

    if walkforward_mode:
        log.info("Modo: walk-forward expansível (re-treina por ciclo)")
        predictions_df, final_models = _expanding_window(walkforward_splits, params)
    else:
        log.info("Modo: split único cronológico + previsão sequencial (rápido)")
        predictions_df, final_models = _single_split_sequential(walkforward_splits, params)

    return predictions_df, final_models


def _single_split_sequential(splits: dict, params: dict) -> tuple:
    """
    Treina uma vez nos primeiros min_train_cycles.
    Prevê cada ciclo de teste usando apenas a janela anterior (sem re-treino).
    Sem vazamento temporal: a janela [t-lookback:t] nunca inclui t ou além.
    """
    from simulation.core.forecasting import LSTMNumpy, ANNForecaster, XGBoostForecaster

    lc  = params.get("lstm",    {})
    ac  = params.get("ann",     {})
    xc  = params.get("xgboost", {})
    cc  = params.get("croston", {})
    arc = params.get("arima",   {})

    all_rows = []
    final_models = {}
    total = len(splits)

    for idx, (key, split) in enumerate(splits.items()):
        w, s, i = key
        series     = split["series"]
        cycles     = split["cycles"]
        lookback   = split["lookback"]
        train_end  = split["first_test_idx"]   # cutoff estrito

        log.info("[%d/%d] Forecast walk-forward (%s,%s,%s) | treino=ciclos 0..%d | teste=%d..%d",
                 idx+1, total, w, s, i, train_end-1, train_end, len(series)-1)

        # ── Treino: APENAS ciclos [0, train_end) ─────────────────────────
        X_tr, y_tr = _sliding_windows(series[:train_end], lookback)
        if len(X_tr) < 3:
            log.warning("Treino insuficiente para (%s,%s,%s) — pulando", w, s, i)
            continue

        models = _fit_all(X_tr, y_tr, lc, ac, xc, cc, arc)

        # ── Previsão sequencial: ciclos [train_end, n) ───────────────────
        n = len(series)
        for t in range(train_end, n):
            if t < lookback:
                continue
            # Janela de entrada: apenas dados ANTERIORES a t
            window = series[t - lookback: t].reshape(1, -1)
            actual = float(series[t])
            cycle  = cycles[t]

            for model_name, model in models.items():
                try:
                    pred = float(np.clip(model.predict(window)[0], 0, None))
                    all_rows.append({
                        "warehouse": w, "store_id": s, "item_id": i,
                        "cycle":     cycle,
                        "model":     model_name,
                        "predicted": pred,
                        "actual":    actual,
                        "n_train":   train_end,
                        "mode":      "single_split",
                    })
                except Exception as e:
                    log.debug("Previsão falhou (%s,%s,%s) ciclo %s %s: %s",
                              w, s, i, cycle, model_name, e)

        # ── Modelo final: treina em TODA a série ─────────────────────────
        X_all, y_all = _sliding_windows(series, lookback)
        final_models[key] = _fit_all(X_all, y_all, lc, ac, xc, cc, arc)

    predictions_df = pd.DataFrame(all_rows)
    log.info("Previsões geradas: %d linhas | %d ciclos únicos",
             len(predictions_df),
             predictions_df["cycle"].nunique() if not predictions_df.empty else 0)
    return predictions_df, final_models


def _expanding_window(splits: dict, params: dict) -> tuple:
    """
    Janela expansível: para cada ciclo t, re-treina em demand[0:t], prevê t.
    Correto para validação rigorosa — mas ~10× mais lento que single-split.
    """
    from simulation.core.forecasting import LSTMNumpy, ANNForecaster, XGBoostForecaster

    lc  = params.get("lstm",    {})
    ac  = params.get("ann",     {})
    xc  = params.get("xgboost", {})
    cc  = params.get("croston", {})
    arc = params.get("arima",   {})

    all_rows = []
    final_models = {}
    total = len(splits)

    for idx, (key, split) in enumerate(splits.items()):
        w, s, i = key
        series     = split["series"]
        cycles     = split["cycles"]
        lookback   = split["lookback"]
        first_test = split["first_test_idx"]
        n = len(series)

        log.info("[%d/%d] Expanding window (%s,%s,%s) | %d previsões",
                 idx+1, total, w, s, i, n - first_test)

        for t in range(first_test, n):
            # Treino estrito: demand[0:t] — nunca inclui t
            X_tr, y_tr = _sliding_windows(series[:t], lookback)
            if len(X_tr) < 3:
                continue

            models = _fit_all(X_tr, y_tr, lc, ac, xc, cc, arc)
            window = series[t - lookback: t].reshape(1, -1)
            actual = float(series[t])
            cycle  = cycles[t]

            for model_name, model in models.items():
                try:
                    pred = float(np.clip(model.predict(window)[0], 0, None))
                    all_rows.append({
                        "warehouse": w, "store_id": s, "item_id": i,
                        "cycle":     cycle,
                        "model":     model_name,
                        "predicted": pred,
                        "actual":    actual,
                        "n_train":   t,
                        "mode":      "expanding_window",
                    })
                except Exception as e:
                    log.debug("Falha (%s,%s,%s) ciclo %s %s: %s",
                              w, s, i, cycle, model_name, e)

        # Modelo final em TODA a série
        X_all, y_all = _sliding_windows(series, lookback)
        final_models[key] = _fit_all(X_all, y_all, lc, ac, xc, cc, arc)

    predictions_df = pd.DataFrame(all_rows)
    log.info("Expanding window: %d previsões", len(predictions_df))
    return predictions_df, final_models


# ══════════════════════════════════════════════════════════════════════════════
# 3. Métricas por ciclo
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_forecast_accuracy(forecast_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula MAE, RMSE, MAPE e Accuracy por:
      - (warehouse, store_id, item_id, model) — agregado
      - (warehouse, store_id, item_id, cycle, model) — por ciclo

    Retorna o DataFrame com ambas as granularidades.
    """
    if forecast_predictions.empty:
        log.warning("forecast_predictions está vazio — sem métricas")
        return pd.DataFrame()

    rows = []
    group_cols = ["warehouse", "store_id", "item_id", "model"]

    # ── Métricas agregadas por série × modelo ────────────────────────────
    for key, grp in forecast_predictions.groupby(group_cols):
        w, s, i, model = key
        m = _metrics(grp["actual"].values, grp["predicted"].values)
        rows.append({
            "warehouse": w, "store_id": s, "item_id": i,
            "cycle": "ALL", "model": model,
            "n_points": len(grp),
            **m,
        })

    # ── Métricas por ciclo × série × modelo ──────────────────────────────
    for key, grp in forecast_predictions.groupby(group_cols + ["cycle"]):
        w, s, i, model, cycle = key
        m = _metrics(grp["actual"].values, grp["predicted"].values)
        rows.append({
            "warehouse": w, "store_id": s, "item_id": i,
            "cycle": cycle, "model": model,
            "n_points": len(grp),
            **m,
        })

    df = pd.DataFrame(rows)
    log.info("Metricas de previsao: %d linhas (%d series x modelos x ciclos)",
             len(df), forecast_predictions.groupby(group_cols).ngroups)
    return df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Literature-standard forecast accuracy metrics for intermittent demand:
      MAE   — Mean Absolute Error
      RMSE  — Root Mean Squared Error
      MAPE  — Mean Absolute Percentage Error (non-zero actuals only)
      sMAPE — Symmetric MAPE (Makridakis 1993); handles zeros, bounded [0, 200%]
      MASE  — Mean Absolute Scaled Error (Hyndman & Koehler 2006); scale-free,
               denominator = naive random-walk MAE on y_true sequence
      RMSSE — Root Mean Squared Scaled Error (M4 competition); same scaling as MASE
      MBE   — Mean Bias Error; positive = model under-forecasts, negative = over-forecasts
      TheilsU — Theil's U2: RMSE(model) / RMSE(naive); <1 beats random walk
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    errors     = y_true - y_pred
    abs_errors = np.abs(errors)
    sq_errors  = errors ** 2

    mae  = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    mbe  = float(np.mean(errors))

    # MAPE — only on non-zero actuals (standard for intermittent demand)
    mask = y_true > 0
    mape = float(np.mean(abs_errors[mask] / y_true[mask]) * 100) if mask.any() else float("nan")
    acc  = max(0.0, 100.0 - mape) if not np.isnan(mape) else 0.0

    # sMAPE: 2*|e| / (|y| + |ŷ|) — symmetric, handles zero actuals (Makridakis 1993)
    denom_s = np.abs(y_true) + np.abs(y_pred)
    smask   = denom_s > 0
    smape   = float(np.mean(2.0 * abs_errors[smask] / denom_s[smask]) * 100) \
              if smask.any() else float("nan")

    # Naive (random-walk) errors: |y_t - y_{t-1}|
    if n >= 2:
        naive_diffs  = np.diff(y_true)
        naive_mae    = float(np.mean(np.abs(naive_diffs)))
        naive_mse    = float(np.mean(naive_diffs ** 2))
        rmse_naive   = float(np.sqrt(naive_mse))
    else:
        naive_mae = naive_mse = rmse_naive = float("nan")

    # MASE (Hyndman & Koehler 2006)
    mase  = mae  / naive_mae   if (not np.isnan(naive_mae)  and naive_mae  > 0) else float("nan")
    # RMSSE (M4 competition)
    rmsse = rmse / rmse_naive  if (not np.isnan(rmse_naive) and rmse_naive > 0) else float("nan")
    # Theil's U2
    theils_u = rmse / rmse_naive if (not np.isnan(rmse_naive) and rmse_naive > 0) else float("nan")

    return {
        "MAE": mae, "RMSE": rmse, "MAPE": mape, "Accuracy": acc,
        "sMAPE": smape, "MASE": mase, "RMSSE": rmsse,
        "MBE": mbe, "TheilsU": theils_u,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sliding_windows(series: np.ndarray, lookback: int):
    """Cria X (N, lookback) e y (N,) via sliding window sobre série ordenada."""
    n = len(series)
    if n <= lookback:
        return np.empty((0, lookback)), np.empty(0)
    idx = np.arange(lookback)[None, :] + np.arange(n - lookback)[:, None]
    return series[idx], series[lookback:]


def _fit_all(X_train: np.ndarray, y_train: np.ndarray,
             lc: dict, ac: dict, xc: dict,
             cc: dict = None, arc: dict = None) -> dict:
    """Treina LSTM, ANN, XGBoost, Croston e ARIMA nos mesmos dados de treino."""
    from simulation.core.forecasting import (
        LSTMNumpy, ANNForecaster, XGBoostForecaster,
        CrostonForecaster, ARIMAForecaster,
    )
    cc  = cc  or {}
    arc = arc or {}
    models = {}

    try:
        lstm = LSTMNumpy(
            input_size=X_train.shape[1],
            hidden_size=lc.get("hidden_size", 64),
            lr=lc.get("learning_rate", 0.001))
        lstm.fit(X_train, y_train,
                 epochs=lc.get("epochs", 50),
                 batch_size=lc.get("batch_size", 16),
                 verbose=False)
        models["LSTM"] = lstm
    except Exception as e:
        log.debug("LSTM treino falhou: %s", e)

    try:
        ann = ANNForecaster(
            hidden_layer_sizes=tuple(ac.get("hidden_layers", [64, 32])),
            max_iter=ac.get("max_iter", 200))
        ann.fit(X_train, y_train)
        models["ANN"] = ann
    except Exception as e:
        log.debug("ANN treino falhou: %s", e)

    try:
        xgbm = XGBoostForecaster(
            n_estimators=xc.get("n_estimators", 200),
            max_depth=xc.get("max_depth", 6),
            learning_rate=xc.get("learning_rate", 0.05),
            subsample=xc.get("subsample", 0.8))
        xgbm.fit(X_train, y_train)
        models["XGBoost"] = xgbm
    except Exception as e:
        log.debug("XGBoost treino falhou: %s", e)

    try:
        croston = CrostonForecaster(alpha=cc.get("alpha", 0.1))
        croston.fit(X_train, y_train)
        models["Croston"] = croston
    except Exception as e:
        log.debug("Croston treino falhou: %s", e)

    try:
        order = tuple(arc.get("order", [1, 1, 1]))
        arima = ARIMAForecaster(order=order)
        arima.fit(X_train, y_train)
        models["ARIMA"] = arima
    except Exception as e:
        log.debug("ARIMA treino falhou: %s", e)

    return models
