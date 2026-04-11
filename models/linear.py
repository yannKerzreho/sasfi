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

All models expect z-scored log-RV input and predict in the same space.
Direct multi-step: one separate regression per horizon h.
"""

from collections import deque
import numpy as np
from .base import BaseForecaster


# ── shared ridge solver ───────────────────────────────────────────────────

_ALPHAS = [10 ** x for x in np.arange(-2, 3.01, 0.25)]


def _ridge_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """Ridge with rolling-window CV to select alpha. Returns weight vector."""
    T, n     = X.shape
    val_size = max(12, T // (n_folds + 1))
    best, best_mse = _ALPHAS[0], np.inf
    for alpha in _ALPHAS:
        mses = []
        for fold in range(n_folds):
            cut = T - (n_folds - fold) * val_size
            if cut < n + 5:
                continue
            w   = np.linalg.solve(X[:cut].T @ X[:cut] + alpha * np.eye(n),
                                  X[:cut].T @ y[:cut])
            err = X[cut:cut + val_size] @ w - y[cut:cut + val_size]
            mses.append(float(np.mean(err ** 2)))
        if mses and np.mean(mses) < best_mse:
            best_mse = np.mean(mses)
            best     = alpha
    return np.linalg.solve(X.T @ X + best * np.eye(n), X.T @ y)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X + 1e-10 * np.eye(X.shape[1]), X.T @ y)


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
        history = np.asarray(history, dtype=np.float64)
        for h in horizons:
            X, y = self._build_Xy(history, h)
            if len(X) < 5:
                self._W[h] = np.zeros(4)
            else:
                self._W[h] = _ridge_cv(X, y, self.n_cv_folds) if self.ridge else _ols(X, y)
        self._buf = deque(history[-self.lags_m:].tolist(), maxlen=self.lags_m)
        return self

    def update(self, x: float) -> "HARForecaster":
        self._buf.append(float(x))
        return self

    def predict(self, h: int) -> float:
        return float(self._features(self._buf) @ self._W[h])


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
        history = np.asarray(history, dtype=np.float64)
        for h in horizons:
            X, y = self._build_Xy(history, h)
            self._W[h] = _ridge_cv(X, y, self.n_cv_folds) if len(X) >= 5 else np.zeros(self.lookback)
        self._buf = deque(history[-self.lookback:].tolist(), maxlen=self.lookback)
        return self

    def update(self, x: float) -> "NLinearForecaster":
        self._buf.append(float(x))
        return self

    def predict(self, h: int) -> float:
        arr    = np.array(self._buf)
        x_norm = arr - arr[-1]
        return float(x_norm @ self._W[h]) + arr[-1]


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
        history = np.asarray(history, dtype=np.float64)
        for h in horizons:
            X, y = self._build_Xy(history, h)
            self._W[h] = _ridge_cv(X, y, self.n_cv_folds) if len(X) >= 5 else np.zeros(2 * self.lookback)
        self._buf = deque(history[-self.lookback:].tolist(), maxlen=self.lookback)
        return self

    def update(self, x: float) -> "DLinearForecaster":
        self._buf.append(float(x))
        return self

    def predict(self, h: int) -> float:
        feat = self._decompose(np.array(self._buf))
        return float(feat @ self._W[h])
