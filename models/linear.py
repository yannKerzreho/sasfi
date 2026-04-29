"""
linear.py — Linear forecasters for univariate RV.

Models
------
HARForecaster    : HAR-RV (Corsi 2009) — daily / weekly / monthly averages.
                   ridge=False → OLS,  ridge=True → ridge with rolling CV.
NLinearForecaster: Subtract last value → ridge on full window residuals.
                   (NLinear from Zeng et al. 2023, "Are Transformers Effective …")
DLinearForecaster: Decompose into trend (MA) + seasonal → ridge on both.
                   (DLinear from the same paper.)

All models accept raw or log-RV input, z-score internally on the training
window, and return predictions in the original (input) scale.
Direct multi-step: one separate regression per horizon h.
"""

from collections import deque
import numpy as np
from .base import BaseForecaster
from .ridge import ALPHAS as _ALPHAS, ridge_fit, ridge_cv_select, har_ridge_cv_and_fit


# ══════════════════════════════════════════════════════════════════════════════
# HAR
# ══════════════════════════════════════════════════════════════════════════════

class HARForecaster(BaseForecaster):
    """
    HAR-RV direct multi-step forecaster (Corsi 2009).

    Features: [1, RV_d, RV_w, RV_m]
        RV_d = x_t
        RV_w = mean(x_{t-4} … x_t)    (5-day)
        RV_m = mean(x_{t-21} … x_t)   (22-day)

    Parameters
    ----------
    ridge : OLS if False, ridge with rolling CV if True.
    """

    def __init__(self, lags_d=1, lags_w=5, lags_m=22, ridge=False, n_cv_folds=5):
        self.lags_d     = lags_d
        self.lags_w     = lags_w
        self.lags_m     = lags_m
        self.ridge      = ridge
        self.n_cv_folds = n_cv_folds
        self._W: dict[int, np.ndarray] = {}
        self._buf: deque | None        = None

    def _features(self, buf: deque) -> np.ndarray:
        arr  = np.array(buf)
        rv_d = arr[-1]
        rv_w = arr[-min(self.lags_w, len(arr)):].mean()
        rv_m = arr[-min(self.lags_m, len(arr)):].mean()
        return np.array([1.0, rv_d, rv_w, rv_m], dtype=np.float64)

    def _build_Xy(self, history: np.ndarray, h: int):
        buf = deque(maxlen=self.lags_m)
        rows, ys = [], []
        for t in range(len(history) - h):
            buf.append(history[t])
            if len(buf) < self.lags_m:
                continue
            rows.append(self._features(buf))
            ys.append(history[t + h])
        if not rows:
            return np.empty((0, 4)), np.empty(0)
        return np.array(rows, dtype=np.float64), np.array(ys, dtype=np.float64)

    def fit(self, history: np.ndarray, horizons: list[int]) -> "HARForecaster":
        history          = np.asarray(history, dtype=np.float64)
        self._mu, self._sigma = self._fit_scaler(history)
        history_z        = self._zscore(history, self._mu, self._sigma)
        self.alpha_log_: dict[int, float] = {}
        for h in horizons:
            X, y = self._build_Xy(history_z, h)
            if len(X) < 5:
                self._W[h] = np.zeros(4)
            elif self.ridge:
                w, alpha           = har_ridge_cv_and_fit(X, y, self.n_cv_folds)
                self._W[h]         = w
                self.alpha_log_[h] = alpha
            else:
                self._W[h] = ridge_fit(X, y, alpha=1e-10)
        self._buf = deque(history_z[-self.lags_m:].tolist(), maxlen=self.lags_m)
        return self

    def update(self, x: float) -> "HARForecaster":
        self._buf.append(float(self._zscore(float(x), self._mu, self._sigma)))
        return self

    def predict(self, h: int) -> float:
        y_z = float(self._features(self._buf) @ self._W[h])
        return float(self._unzscore(y_z, self._mu, self._sigma))


# ══════════════════════════════════════════════════════════════════════════════
# NLinear
# ══════════════════════════════════════════════════════════════════════════════

class NLinearForecaster(BaseForecaster):
    """
    NLinear direct multi-step forecaster (Zeng et al. 2023).

    Subtracts the last observed value before regression:
        x̃ = x_{window} − x_{last}
        ŷ_{t+h} = x̃ @ w_h + x_{last}

    This simple normalisation makes the model predict *changes* relative
    to the current level, which is more stationary and helps generalise
    across different volatility regimes.  Ridge regularisation is always
    used because the window can be large relative to the training horizon.

    Parameters
    ----------
    lookback : input window length L.
    """

    def __init__(self, lookback: int = 20, n_cv_folds: int = 5):
        self.lookback   = lookback
        self.n_cv_folds = n_cv_folds
        self._W: dict[int, np.ndarray] = {}
        self._buf: deque | None        = None

    def _build_Xy(self, history: np.ndarray, h: int):
        L, T = self.lookback, len(history)
        rows, ys = [], []
        for t in range(L - 1, T - h):
            window = history[t - L + 1: t + 1]    # (L,)
            x_norm = window - window[-1]            # subtract last
            rows.append(x_norm)
            ys.append(history[t + h] - window[-1]) # predict change
        if not rows:
            return np.empty((0, L)), np.empty(0)
        return np.array(rows, dtype=np.float64), np.array(ys, dtype=np.float64)

    def fit(self, history: np.ndarray, horizons: list[int]) -> "NLinearForecaster":
        history          = np.asarray(history, dtype=np.float64)
        self._mu, self._sigma = self._fit_scaler(history)
        history_z        = self._zscore(history, self._mu, self._sigma)
        self.alpha_log_: dict[int, float] = {}
        for h in horizons:
            X, y = self._build_Xy(history_z, h)
            if len(X) >= 5:
                alpha        = ridge_cv_select(X, y, self.n_cv_folds)
                self._W[h]   = ridge_fit(X, y, alpha)
                self.alpha_log_[h] = alpha
            else:
                self._W[h] = np.zeros(self.lookback)
        self._buf = deque(history_z[-self.lookback:].tolist(), maxlen=self.lookback)
        return self

    def update(self, x: float) -> "NLinearForecaster":
        self._buf.append(float(self._zscore(float(x), self._mu, self._sigma)))
        return self

    def predict(self, h: int) -> float:
        arr    = np.array(self._buf)
        x_norm = arr - arr[-1]
        y_z    = float(x_norm @ self._W[h]) + arr[-1]
        return float(self._unzscore(y_z, self._mu, self._sigma))


# ══════════════════════════════════════════════════════════════════════════════
# DLinear
# ══════════════════════════════════════════════════════════════════════════════

class DLinearForecaster(BaseForecaster):
    """
    DLinear direct multi-step forecaster (Zeng et al. 2023).

    Decomposes the input window into trend (moving average) and seasonal
    (residual), then fits a separate ridge regression for each component:
        trend    = MA(x_window, kernel)           shape (L,)
        seasonal = x_window − trend               shape (L,)
        ŷ_{t+h} = trend @ w_trend_h + seasonal @ w_seasonal_h

    Parameters
    ----------
    lookback  : input window length L.
    ma_kernel : moving-average kernel size for trend extraction.
    """

    def __init__(self, lookback: int = 20, ma_kernel: int = 5, n_cv_folds: int = 5):
        self.lookback   = lookback
        self.ma_kernel  = ma_kernel
        self.n_cv_folds = n_cv_folds
        self._W: dict[int, np.ndarray] = {}   # 2L features
        self._buf: deque | None        = None

    def _decompose(self, window: np.ndarray) -> np.ndarray:
        """Return concatenated [trend(L), seasonal(L)] feature vector."""
        k     = min(self.ma_kernel, len(window))
        pad   = np.pad(window, (k // 2, k - 1 - k // 2), mode="edge")
        trend = np.convolve(pad, np.ones(k) / k, mode="valid")[: len(window)]
        return np.concatenate([trend, window - trend])   # (2L,)

    def _build_Xy(self, history: np.ndarray, h: int):
        L, T = self.lookback, len(history)
        rows, ys = [], []
        for t in range(L - 1, T - h):
            feat = self._decompose(history[t - L + 1: t + 1])
            rows.append(feat)
            ys.append(history[t + h])
        if not rows:
            return np.empty((0, 2 * L)), np.empty(0)
        return np.array(rows, dtype=np.float64), np.array(ys, dtype=np.float64)

    def fit(self, history: np.ndarray, horizons: list[int]) -> "DLinearForecaster":
        history          = np.asarray(history, dtype=np.float64)
        self._mu, self._sigma = self._fit_scaler(history)
        history_z        = self._zscore(history, self._mu, self._sigma)
        self.alpha_log_: dict[int, float] = {}
        for h in horizons:
            X, y = self._build_Xy(history_z, h)
            if len(X) >= 5:
                alpha        = ridge_cv_select(X, y, self.n_cv_folds)
                self._W[h]   = ridge_fit(X, y, alpha)
                self.alpha_log_[h] = alpha
            else:
                self._W[h] = np.zeros(2 * self.lookback)
        self._buf = deque(history_z[-self.lookback:].tolist(), maxlen=self.lookback)
        return self

    def update(self, x: float) -> "DLinearForecaster":
        self._buf.append(float(self._zscore(float(x), self._mu, self._sigma)))
        return self

    def predict(self, h: int) -> float:
        feat = self._decompose(np.array(self._buf))
        y_z  = float(feat @ self._W[h])
        return float(self._unzscore(y_z, self._mu, self._sigma))
