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


# ── shared ridge helpers ──────────────────────────────────────────────────

# 29 log-spaced values from 10^-3 to 10^4.
#
# Why extended from [-2, 3]:
#  • HAR_Ridge h=1 kept hitting the old lower bound 0.01 → needs ~0.001
#    (4 features / 450 samples: OLS is already well-determined)
#  • h=22 models (DLinear, SAS) kept hitting the old upper bound 1000
#    → shrinkage benefits from going up to 10 000
#  Inspect chosen values via model.alpha_log_ after fit.
_ALPHAS = [10 ** x for x in np.arange(-4, 5.01, 0.25)]


def _ridge_cv_select(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5,
    alphas=None, penalty_mask: np.ndarray | None = None,
) -> float:
    """
    Rolling-window CV to select the best ridge alpha.

    Parameters
    ----------
    X            : (T, n) feature matrix.
    y            : (T,) targets.
    n_folds      : rolling CV folds.
    alphas       : candidate alphas (default: module-level _ALPHAS).
    penalty_mask : (n,) array of per-feature penalty multipliers.
                   Ridge penalty is  alpha * diag(penalty_mask).
                   Pass [0, 1, 1, 1] to skip the intercept (HAR ridge).
                   None → standard ridge  (penalty_mask = ones).

    Returns
    -------
    best alpha (float) — does NOT compute final weights.
    """
    if alphas is None:
        alphas = _ALPHAS
    T, n     = X.shape
    R        = (np.diag(penalty_mask.astype(float))
                if penalty_mask is not None else np.eye(n))
    val_size = max(12, T // (n_folds + 1))
    best, best_mse = alphas[0], np.inf
    for alpha in alphas:
        mses = []
        for fold in range(n_folds):
            cut = T - (n_folds - fold) * val_size
            if cut < n + 5:
                continue
            w   = np.linalg.solve(X[:cut].T @ X[:cut] + alpha * R,
                                  X[:cut].T @ y[:cut])
            err = X[cut:cut + val_size] @ w - y[cut:cut + val_size]
            mses.append(float(np.mean(err ** 2)))
        if mses and np.mean(mses) < best_mse:
            best_mse = np.mean(mses)
            best     = alpha
    return best


def _ridge_fit(
    X: np.ndarray, y: np.ndarray, alpha: float,
    penalty_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Ridge regression with a fixed alpha and optional per-feature penalty mask."""
    n = X.shape[1]
    R = (np.diag(penalty_mask.astype(float))
         if penalty_mask is not None else np.eye(n))
    return np.linalg.solve(X.T @ X + alpha * R, X.T @ y)


def _ridge_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """Select alpha via CV then return the fitted weight vector (standard ridge)."""
    alpha = _ridge_cv_select(X, y, n_folds)
    return _ridge_fit(X, y, alpha)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(X.T @ X + 1e-10 * np.eye(X.shape[1]), X.T @ y)


# ── HAR-specific column-normalized ridge ──────────────────────────────────
#
# Features = [1 (intercept), RV_d, RV_w, RV_m].  After z-scoring the input,
# RV_d has std ≈ 1 but RV_w and RV_m are rolling averages (std < 1 and
# decreasing with window length).  A single global alpha penalises all four
# coefficients equally in weight space, which implicitly over-regularises the
# smoother, lower-variance monthly component and under-regularises the noisier
# daily one.
#
# Fixes:
#  (1) Scale each feature column to unit std before the solve, so alpha acts
#      uniformly in "normalised-coefficient" space.
#  (2) Do NOT penalise the intercept — it carries the level correction and
#      should not be shrunk toward zero across regimes.

_HAR_FEATURE_NAMES = ["intercept", "RV_d", "RV_w", "RV_m"]


def _har_ridge_cv_select_and_fit(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5, alphas=None,
) -> tuple[np.ndarray, float]:
    """
    Column-normalised ridge for HAR features [1, RV_d, RV_w, RV_m].

    Steps
    -----
    1. Compute column stds on the full training set (col 0 = intercept kept at 1).
    2. Divide each non-intercept column by its std so all three features enter
       the penalty on an equal footing.
    3. Run rolling-window CV with a penalty mask [0, 1, 1, 1]
       (intercept is NOT regularised).
    4. Fit final weights on the full (T, 4) scaled matrix, then back-transform
       to the original feature space.

    Returns
    -------
    w      : (n,) weight vector in the original (unscaled) feature space.
    alpha  : the chosen regularisation parameter.
    """
    if alphas is None:
        alphas = _ALPHAS
    T, n = X.shape

    # Column scaling — skip col 0 (intercept = all-ones, std = 0)
    col_stds     = np.maximum(X[:, 1:].std(axis=0), 1e-8)  # (n-1,)
    col_scale    = np.concatenate([[1.0], col_stds])        # (n,)
    X_sc         = X / col_scale                            # (T, n) scaled

    # Penalty mask: intercept not penalised
    pmask = np.array([0.0] + [1.0] * (n - 1))

    # CV on the scaled features
    alpha = _ridge_cv_select(X_sc, y, n_folds, alphas=alphas, penalty_mask=pmask)

    # Final fit in scaled space, then back-transform
    w_sc = _ridge_fit(X_sc, y, alpha, penalty_mask=pmask)
    w    = w_sc / col_scale        # back to original-feature weights

    return w, alpha


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
                w, alpha         = _har_ridge_cv_select_and_fit(X, y, self.n_cv_folds)
                self._W[h]       = w
                self.alpha_log_[h] = alpha
            else:
                self._W[h] = _ols(X, y)
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
                alpha        = _ridge_cv_select(X, y, self.n_cv_folds)
                self._W[h]   = _ridge_fit(X, y, alpha)
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
                alpha        = _ridge_cv_select(X, y, self.n_cv_folds)
                self._W[h]   = _ridge_fit(X, y, alpha)
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
