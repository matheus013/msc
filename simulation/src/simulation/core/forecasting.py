"""
forecasting.py — Módulo 1: Previsão de Demanda
LSTM vetorizado (numpy batch), ANN/MLP (sklearn), XGBoost, Croston, ARIMA
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


class LSTMNumpy:
    """LSTM batch-vetorizado em numpy. Muito mais rápido que sample-by-sample."""

    def __init__(self, input_size: int, hidden_size: int, lr: float = 0.01):
        self.H  = hidden_size
        self.T  = input_size   # lookback = timesteps
        self.lr = lr
        sc = 0.05
        self.Wh = np.random.randn(4 * hidden_size, hidden_size) * sc
        self.Wx = np.random.randn(4 * hidden_size, 1) * sc
        self.b  = np.zeros(4 * hidden_size)
        self.Wy = np.random.randn(hidden_size) * sc
        self.by = 0.0
        self._scaler_x = MinMaxScaler()
        self._scaler_y = MinMaxScaler()

    @staticmethod
    def _sig(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x, -10, 10))

    def _forward_batch(self, X_sc):
        """X_sc: (N, T) → h_last: (N, H)"""
        N, T = X_sc.shape
        H = self.H
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        for t in range(T):
            xt = X_sc[:, t:t+1]
            z  = h @ self.Wh.T + xt @ self.Wx.T + self.b
            i_ = self._sig( z[:, :H])
            f_ = self._sig( z[:, H:2*H])
            g_ = self._tanh(z[:, 2*H:3*H])
            o_ = self._sig( z[:, 3*H:])
            c  = f_ * c + i_ * g_
            h  = o_ * self._tanh(c)
        return h

    def fit(self, X, y, epochs=50, batch_size=64, verbose=True):
        N = len(X)
        X_sc = self._scaler_x.fit_transform(X)
        y_sc = self._scaler_y.fit_transform(y.reshape(-1,1)).ravel()

        for ep in range(epochs):
            idx = np.random.permutation(N)
            total_loss = 0.0
            for start in range(0, N, batch_size):
                bidx = idx[start:start+batch_size]
                Xb = X_sc[bidx]; yb = y_sc[bidx]
                B = len(bidx)

                h_last = self._forward_batch(Xb)
                pred = h_last @ self.Wy + self.by
                err  = pred - yb
                total_loss += float(np.mean(err**2))

                dWy = np.clip((h_last.T @ err) / B, -5, 5)
                dby = float(np.mean(err))
                dh  = np.outer(err, self.Wy) / B

                xt_last = Xb[:, -1:]
                h_prev  = self._forward_batch(Xb[:, :-1]) if self.T > 1 else np.zeros((B, self.H))
                z = h_prev @ self.Wh.T + xt_last @ self.Wx.T + self.b
                H = self.H
                i_ = self._sig(z[:, :H]);     dsi = i_*(1-i_)
                f_ = self._sig(z[:, H:2*H]);  dsf = f_*(1-f_)
                g_ = self._tanh(z[:,2*H:3*H]); dsg = 1-g_**2
                o_ = self._sig(z[:, 3*H:]);   dso = o_*(1-o_)
                c_l = i_*g_
                do = dh * self._tanh(c_l)
                dc = dh * o_ * (1 - self._tanh(c_l)**2)
                di = dc*g_; df = dc*0; dg = dc*i_
                dz = np.hstack([di*dsi, df*dsf, dg*dsg, do*dso])

                clip = 5.0
                dWh = np.clip((dz.T @ h_prev)/B, -clip, clip)
                dWx = np.clip((dz.T @ xt_last)/B, -clip, clip)
                db  = np.clip(dz.mean(axis=0), -clip, clip)
                np.clip(dWy, -clip, clip, out=dWy)

                self.Wh -= self.lr * dWh
                self.Wx -= self.lr * dWx
                self.b  -= self.lr * db
                self.Wy -= self.lr * dWy
                self.by -= self.lr * dby

            if verbose and (ep+1) % 10 == 0:
                print(f"  [LSTM] Epoch {ep+1}/{epochs} | Loss: {total_loss:.4f}")

    def predict(self, X):
        X_sc = self._scaler_x.transform(X)
        h_last = self._forward_batch(X_sc)
        preds_sc = h_last @ self.Wy + self.by
        return np.clip(
            self._scaler_y.inverse_transform(preds_sc.reshape(-1,1)).ravel(), 0, None)


class ANNForecaster:
    def __init__(self, hidden_layer_sizes=(64,32), max_iter=200, learning_rate_init=0.001):
        self.sx = MinMaxScaler(); self.sy = MinMaxScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
            learning_rate_init=learning_rate_init, random_state=42,
            early_stopping=True, validation_fraction=0.1, verbose=False)

    def fit(self, X, y):
        self.model.fit(self.sx.fit_transform(X),
                       self.sy.fit_transform(y.reshape(-1,1)).ravel())
        print(f"  [ANN] Treinado | Iterações: {self.model.n_iter_}")

    def predict(self, X):
        p = self.model.predict(self.sx.transform(X))
        return np.clip(self.sy.inverse_transform(p.reshape(-1,1)).ravel(), 0, None)


class XGBoostForecaster:
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8):
        self.sx = MinMaxScaler(); self.sy = MinMaxScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            random_state=42, verbosity=0)

    def fit(self, X, y):
        self.model.fit(self.sx.fit_transform(X),
                       self.sy.fit_transform(y.reshape(-1,1)).ravel())
        print(f"  [XGBoost] Treinado | Estimators: {self.model.n_estimators}")

    def predict(self, X):
        p = self.model.predict(self.sx.transform(X))
        return np.clip(self.sy.inverse_transform(p.reshape(-1,1)).ravel(), 0, None)


def evaluate_forecaster(y_true, y_pred, name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.any() else 100.0
    acc  = max(0.0, 100.0 - mape)
    print(f"  [{name}] MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.1f}% | Accuracy={acc:.1f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Accuracy": acc}


# ══════════════════════════════════════════════════════════════════════════════
# Croston (proposta seção 3.1)
# ══════════════════════════════════════════════════════════════════════════════
class CrostonForecaster:
    """Método de Croston para demanda intermitente.

    Suaviza separadamente o tamanho das demandas positivas (q̂) e o intervalo
    entre elas (p̂). Previsão constante: q̂ / p̂.
    Interface fit(X, y) compatível com o pipeline walk-forward:
    a série completa é reconstruída a partir da janela deslizante X e y.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.forecast_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        series = (np.concatenate([X[0], y]) if len(X) > 0 else y).astype(float)
        self._fit_series(series)
        return self

    def _fit_series(self, series: np.ndarray):
        pos_idx = np.where(series > 0)[0]
        if len(pos_idx) == 0:
            self.forecast_ = 0.0
            return
        q = float(series[pos_idx[0]])
        p = float(pos_idx[0] + 1) if pos_idx[0] > 0 else 1.0
        interval = 1
        for t in range(pos_idx[0] + 1, len(series)):
            if series[t] > 0:
                q = self.alpha * series[t] + (1 - self.alpha) * q
                p = self.alpha * interval + (1 - self.alpha) * p
                interval = 1
            else:
                interval += 1
        self.forecast_ = max(0.0, q / max(p, 1e-6))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.forecast_)


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA (proposta seção 3.1)
# ══════════════════════════════════════════════════════════════════════════════
class ARIMAForecaster:
    """ARIMA com ordem configurável.

    Treina em statsmodels; armazena previsões step-ahead no momento do fit.
    predict(X) consome as previsões em sequência; após esgotar, repete a última.
    Interface fit(X, y) compatível com o pipeline walk-forward.
    """

    def __init__(self, order: tuple = (1, 1, 1)):
        self.order = order
        self._forecasts: list = []
        self._pred_idx: int = 0
        self._fallback: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        series = (np.concatenate([X[0], y]) if len(X) > 0 else y).astype(float)
        pos = series[series > 0]
        self._fallback = float(np.mean(pos)) if len(pos) > 0 else 0.0
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _ARIMA(series, order=self.order).fit()
            n_steps = max(30, len(y))
            self._forecasts = np.clip(result.forecast(steps=n_steps), 0, None).tolist()
        except Exception:
            self._forecasts = []
        self._pred_idx = 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        results = []
        for _ in range(len(X)):
            if self._pred_idx < len(self._forecasts):
                results.append(self._forecasts[self._pred_idx])
                self._pred_idx += 1
            else:
                last = self._forecasts[-1] if self._forecasts else self._fallback
                results.append(last)
        return np.array(results)


def train_all_forecasters(X_train, y_train, X_test, y_test, fcfg):
    results = {}

    print("\n[Forecasting] Treinando LSTM...")
    lc = fcfg.get("LSTM", {})
    lstm = LSTMNumpy(input_size=X_train.shape[1],
                     hidden_size=lc.get("hidden_size", 64),
                     lr=lc.get("learning_rate", 0.01))
    lstm.fit(X_train, y_train, epochs=lc.get("epochs", 50),
             batch_size=lc.get("batch_size", 64))
    preds = lstm.predict(X_test)
    results["LSTM"] = {"model": lstm, "predictions": preds,
                       "metrics": evaluate_forecaster(y_test, preds, "LSTM")}

    print("\n[Forecasting] Treinando ANN...")
    ac = fcfg.get("ANN", {})
    ann = ANNForecaster(hidden_layer_sizes=tuple(ac.get("hidden_layer_sizes", [64,32])),
                        max_iter=ac.get("max_iter", 200),
                        learning_rate_init=ac.get("learning_rate_init", 0.001))
    ann.fit(X_train, y_train)
    preds = ann.predict(X_test)
    results["ANN"] = {"model": ann, "predictions": preds,
                      "metrics": evaluate_forecaster(y_test, preds, "ANN")}

    print("\n[Forecasting] Treinando XGBoost...")
    xc = fcfg.get("XGBOOST", {})
    xgbm = XGBoostForecaster(n_estimators=xc.get("n_estimators", 200),
                              max_depth=xc.get("max_depth", 6),
                              learning_rate=xc.get("learning_rate", 0.05),
                              subsample=xc.get("subsample", 0.8))
    xgbm.fit(X_train, y_train)
    preds = xgbm.predict(X_test)
    results["XGBoost"] = {"model": xgbm, "predictions": preds,
                          "metrics": evaluate_forecaster(y_test, preds, "XGBoost")}

    return results
