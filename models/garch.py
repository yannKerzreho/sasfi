"""
garch.py — AR(p_ar)-GARCH(1,1) forecaster.

Fix applied here
----------------
* AR mean equation with p_ar=5 lags (default) instead of 1.
* Robust OLS fallback: if MLE fails, estimate AR(p_ar) via OLS so the
  model still produces meaningful predictions instead of predicting 0.
* For h-step mean: iterate the AR companion form.

Requires: pip install arch
"""

from collections import deque
import numpy as np
from .base import BaseForecaster

try:
    from arch import arch_model as _arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


def _ols_ar(history: np.ndarray, p: int):
    """OLS estimate of AR(p) coefficients. Returns (const, phi[1..p])."""
    T   = len(history)
    X   = np.column_stack([np.ones(T - p)] +
                          [history[p - k - 1: T - k - 1] for k in range(p)])
    y   = history[p:]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coef[0]), coef[1:]   # const, phi[1..p]


class GARCHForecaster(BaseForecaster):
    """
    AR(p_ar)-GARCH(1,1) forecaster.

    Point forecast uses the AR(p_ar) mean equation iterated h steps ahead.
    GARCH(1,1) models residual variance (useful for density forecasts, less
    so for point forecasts — see module docstring).

    Parameters
    ----------
    p_ar  : number of AR lags in the mean equation (default 5).
    p, q  : GARCH order (default 1, 1).
    """

    def __init__(self, p_ar: int = 5, p: int = 1, q: int = 1):
        if not _ARCH_AVAILABLE:
            raise ImportError("GARCH requires the 'arch' package: pip install arch")
        self.p_ar = p_ar
        self.p    = p
        self.q    = q
        # Mean-equation parameters
        self.const = 0.0
        self.phi   = np.zeros(p_ar)   # AR coefficients φ[1..p_ar]
        # GARCH parameters
        self.omega  = 1e-5
        self.alpha1 = 0.05
        self.beta1  = 0.90
        # Streaming state
        self.sigma2_last = 1.0
        self.eps_last    = 0.0
        self._buf: deque = deque(maxlen=5000)   # last p_ar+1 needed for predict

    # ── mean prediction ───────────────────────────────────────────────────

    def _predict_mean(self, h: int) -> float:
        """
        Iterate the AR(p_ar) companion form h steps ahead.

        State vector: [y_t, y_{t-1}, ..., y_{t-p+1}]  (p,)
        Companion:    [φ_1, φ_2, …, φ_p ; I_{p-1} | 0]
        """
        p    = self.p_ar
        phi  = self.phi          # (p,)
        buf  = list(self._buf)
        state = np.array(buf[-p:][::-1], dtype=np.float64)  # [y_t, y_{t-1}, …]

        # Companion matrix (p × p)
        A = np.zeros((p, p))
        A[0, :] = phi
        if p > 1:
            A[1:, :-1] = np.eye(p - 1)

        # Intercept vector (only first row matters)
        c = np.zeros(p)
        c[0] = self.const

        for _ in range(h):
            state = A @ state + c

        # y_{t+h} is the first component
        return float(state[0])

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, history: np.ndarray, horizons: list[int]) -> "GARCHForecaster":
        history = np.asarray(history, dtype=np.float64)
        self._buf.clear()
        self._buf.extend(history.tolist())

        try:
            am  = _arch_model(history, mean="AR", lags=self.p_ar,
                              vol="GARCH", p=self.p, q=self.q,
                              dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            par = res.params

            self.const = float(par.get("Const", 0.0))
            self.phi   = np.array(
                [float(par.get(f"y[{k}]", 0.0)) for k in range(1, self.p_ar + 1)]
            )
            self.omega  = float(par["omega"])
            self.alpha1 = float(par["alpha[1]"])
            self.beta1  = float(par["beta[1]"])

            cond_vol         = res.conditional_volatility
            self.sigma2_last = float(cond_vol.iloc[-1] ** 2)
            resid            = res.resid
            self.eps_last    = float(resid.iloc[-1])

        except Exception:
            # Robust OLS fallback: AR(p_ar) estimated by least squares
            self.const, self.phi = _ols_ar(history, self.p_ar)
            self.sigma2_last     = float(np.var(history))
            self.eps_last        = 0.0

        return self

    # ── streaming update ─────────────────────────────────────────────────

    def update(self, x: float) -> "GARCHForecaster":
        self._buf.append(float(x))
        # Recompute last residual and update GARCH variance
        buf = list(self._buf)
        if len(buf) > self.p_ar:
            state    = np.array(buf[-self.p_ar - 1:-1][::-1])
            mean_t   = self.const + float(self.phi @ state)
            eps      = float(x) - mean_t
        else:
            eps = 0.0
        self.sigma2_last = (self.omega
                            + self.alpha1 * eps ** 2
                            + self.beta1  * self.sigma2_last)
        self.eps_last    = eps
        return self

    # ── prediction ───────────────────────────────────────────────────────

    def predict(self, h: int) -> float:
        return self._predict_mean(h)
