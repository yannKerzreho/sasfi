"""
sas.py — SAS (Spectral Associative Scan) reservoir forecaster.

Architecture
------------
The reservoir evolves as a polynomial recurrence:

    s_t = P(z_t) ⊛ s_{t-1} + Q(z_t)

where ⊛ is matrix-vector product (LinearPoly, TrigoPoly) or element-wise
multiplication (DiagonalPoly).  The readout is linear:

    ŷ_{t+h} = s_t · W_h

Polynomial bases are registered JAX pytree nodes and are passed **directly**
into the JIT-compiled kernels — no caching, no NumPy conversion, arbitrary
degree.  JAX traces a separate compiled version for each concrete basis type.

Parallel training (two-level associative scan)
----------------------------------------------
The state sequence is computed via a divide-and-conquer algorithm:

  Phase 1 – intra-chunk cumulative transitions  (vmap over K chunks of B).
  Phase 2 – inter-chunk carries via a second associative scan over K.
  Phase 3 – apply carries to resolve all states in parallel (vmap over K).

This runs in O(log T) depth instead of O(T), fully exploiting XLA parallelism.

Streaming protocol (BaseForecaster interface)
----------------------------------------------
  fit(history, horizons)  → parallel scan + ridge regression per horizon
  update(x)               → single-step recurrence advance (JIT'd)
  predict(h)              → dot(s_last, W_h)

Basis options (string shortcuts in SASForecaster.__init__)
----------------------------------------------------------
  "diagonal"  → DiagonalPoly   O(n) per step — use n up to 1 000+
  "linear"    → LinearPoly     O(n²) per step — keep n ≤ 200
  "trigo"     → TrigoPoly      O(n²), trig features — keep n ≤ 200

Advanced: pass any BasePoly *instance* directly for full control over degree,
spectral_norm, and all hyperparameters before calling fit.

Requires: JAX  (pip install jax)
"""

from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp
from collections import deque

from .base import BaseForecaster
from .sas_utils import BasePoly, LinearPoly, DiagonalPoly, TrigoPoly


# ══════════════════════════════════════════════════════════════════════════════
# Pure JAX kernels  (module-level so JIT cache is shared across instances)
# ══════════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("n", "chunk_size"))
def _collect_states(basis, u, s0, n: int, chunk_size: int):
    """
    Two-level parallel associative scan over input sequence u.

    Parameters
    ----------
    basis      : any registered BasePoly pytree (LinearPoly / DiagonalPoly / …)
    u          : (T, d) input sequence — float32
    s0         : (n,)  initial reservoir state
    n          : reservoir dimension (static — determines zero-state shape)
    chunk_size : B (static — determines reshape)

    Returns
    -------
    all_states : (T, n) state matrix
    s_last     : (n,)  state at the last real timestep
    """
    T   = u.shape[0]
    B   = chunk_size
    pad = (B - T % B) % B
    u_p = jnp.pad(u, ((0, pad), (0, 0)))        # (T+pad, d)
    K   = u_p.shape[0] // B
    U   = u_p.reshape(K, B, -1)                  # (K, B, d)

    # ── Phase 1: intra-chunk cumulative scans (K independent, parallel) ───
    def chunk_scans(u_c):
        # u_c: (B, d) → cumulative (A_rep, b) pairs within one chunk
        return jax.lax.associative_scan(
            basis.combine,
            (basis.batch_eval_p(u_c), basis.batch_eval_q(u_c)),
        )

    Acum, bcum = jax.vmap(chunk_scans)(U)    # (K, B, *A_shape), (K, B, n)

    # ── Phase 2: inter-chunk scan over the K last-step transforms ─────────
    A_inter, b_inter = jax.lax.associative_scan(
        basis.combine, (Acum[:, -1], bcum[:, -1])
    )                                        # (K, *A_shape), (K, n)

    # ── Phase 3: carry = state at the START of each chunk ─────────────────
    # carries[0] = s0
    # carries[k] = apply(A_inter[k-1], s0) + b_inter[k-1]  for k ≥ 1
    rest = jax.vmap(lambda A, b: basis.apply(A, s0) + b)(
        A_inter[:-1], b_inter[:-1]
    )                                                      # (K-1, n)
    carries = jnp.concatenate([s0[None], rest], axis=0)   # (K, n)

    # ── Phase 4: resolve all states (K chunks in parallel) ────────────────
    all_s = jax.vmap(
        lambda Ac, bc, c: jax.vmap(lambda A, b: basis.apply(A, c) + b)(Ac, bc)
    )(Acum, bcum, carries).reshape(K * B, n)              # (K*B, n)

    return all_s[:T], all_s[T - 1]


@jax.jit
def _step_once(basis, s, z):
    """
    Advance reservoir state by a single step.

    Parameters
    ----------
    basis : BasePoly pytree (already initialised)
    s     : (n,)  current state
    z     : (d,)  new input  (d=1 for univariate)

    Returns
    -------
    s_new : (n,)
    """
    return basis.apply(basis.eval_p(z), s) + basis.eval_q(z)


# ══════════════════════════════════════════════════════════════════════════════
# Ridge helpers (pure NumPy — fast for typical training sizes ~500 × 100)
# ══════════════════════════════════════════════════════════════════════════════

_ALPHAS = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0]


def _ridge_cv(S: np.ndarray, Y: np.ndarray, n_folds: int, alphas) -> float:
    """Rolling-window cross-validation to pick the best ridge alpha."""
    T, n     = S.shape
    val_size = max(12, T // (n_folds + 1))
    best, best_mse = alphas[0], np.inf
    for alpha in alphas:
        mses = []
        for fold in range(n_folds):
            cut = T - (n_folds - fold) * val_size
            if cut < max(5, n // 10):
                continue
            A   = S[:cut].T @ S[:cut] + alpha * np.eye(n)
            w   = np.linalg.solve(A, S[:cut].T @ Y[:cut])
            err = S[cut: cut + val_size] @ w - Y[cut: cut + val_size]
            mses.append(float(np.mean(err ** 2)))
        if mses and np.mean(mses) < best_mse:
            best_mse = np.mean(mses)
            best     = alpha
    return best


def _ridge_fit(S: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression: (S.T S + αI)⁻¹ S.T Y → weight vector."""
    n = S.shape[1]
    return np.linalg.solve(S.T @ S + alpha * np.eye(n), S.T @ Y)


# ══════════════════════════════════════════════════════════════════════════════
# SASForecaster
# ══════════════════════════════════════════════════════════════════════════════

class SASForecaster(BaseForecaster):
    """
    SAS reservoir computing forecaster — direct multi-step regression.

    Parameters
    ----------
    n_reservoir  : reservoir dimension n.
    basis        : polynomial basis.  Either a string shorthand
                   ("diagonal" | "linear" | "trigo") or any BasePoly instance
                   (initialised or not) for full control.
    spectral_norm: when basis is a string, this sets the spectral radius bound.
                   Ignored when a BasePoly instance is passed (use its own sn).
    p_degree     : polynomial degree for P  (string-shorthand only).
    q_degree     : polynomial degree for Q  (string-shorthand only).
    washout      : steps discarded at the start of each training window before
                   the ridge regression.
    chunk_size   : B in the two-level scan.  Must be a static integer; larger B
                   → more intra-chunk parallelism but more padding waste.
    n_cv_folds   : rolling CV folds for ridge alpha selection.
    seed         : JAX random seed for basis initialisation.
    alphas       : ridge penalty candidates (default: log-spaced 1e-3…1e3).
    """

    def __init__(
        self,
        n_reservoir:   int   = 100,
        basis                = "diagonal",   # str | BasePoly instance
        spectral_norm: float = 0.9,
        p_degree:      int   = 1,
        q_degree:      int   = 1,
        washout:       int   = 50,
        chunk_size:    int   = 64,
        n_cv_folds:    int   = 3,
        seed:          int   = 42,
        alphas               = None,
    ):
        self.n_reservoir = n_reservoir
        self.washout     = washout
        self.chunk_size  = chunk_size
        self.n_cv_folds  = n_cv_folds
        self.seed        = seed
        self.alphas      = list(alphas) if alphas is not None else _ALPHAS

        # ── resolve basis spec → uninitialized BasePoly ───────────────────
        if isinstance(basis, str):
            _map = {
                "diagonal": DiagonalPoly,
                "linear":   LinearPoly,
                "trigo":    TrigoPoly,
            }
            if basis not in _map:
                raise ValueError(
                    f"Unknown basis string '{basis}'. "
                    f"Choose from {list(_map)} or pass a BasePoly instance."
                )
            self._basis: BasePoly = _map[basis](p_degree, q_degree, spectral_norm)
        elif isinstance(basis, BasePoly):
            self._basis = basis
        else:
            raise TypeError(
                f"basis must be a str or BasePoly instance, got {type(basis)}"
            )

        self._W: dict[int, np.ndarray] = {}
        self._s_last: np.ndarray | None = None

    # ── BaseForecaster interface ───────────────────────────────────────────

    def fit(self, history: np.ndarray, horizons: list[int]) -> "SASForecaster":
        """
        Fit the readout for each horizon.

        1. (Re-)initialise the basis with a fresh JAX random key.
        2. Run the two-level parallel scan → (T, n) state matrix.
        3. For each h ∈ horizons: rolling ridge CV then solve for W_h.
        """
        history = np.asarray(history, dtype=np.float32)
        T       = len(history)
        u       = jnp.array(history[:, None])   # (T, 1) — univariate

        # Always re-init: ensures each OOS refit gets a fresh random reservoir
        key         = jax.random.PRNGKey(self.seed)
        self._basis = self._basis.initialize(self.n_reservoir, key)

        s0             = jnp.zeros(self.n_reservoir)
        states, s_last = _collect_states(
            self._basis, u, s0, self.n_reservoir, self.chunk_size
        )
        states_np    = np.array(states, dtype=np.float32)   # (T, n)
        self._s_last = np.array(s_last, dtype=np.float32)   # (n,)

        n       = self.n_reservoir
        self._W = {}
        for h in horizons:
            S = states_np[self.washout: T - h]
            Y = history  [self.washout + h: T]
            if len(S) < 5:
                self._W[h] = np.zeros(n, dtype=np.float32)
                continue
            alpha      = _ridge_cv(S, Y, self.n_cv_folds, self.alphas)
            self._W[h] = _ridge_fit(S, Y, alpha).astype(np.float32)

        return self

    def update(self, x: float) -> "SASForecaster":
        """Advance the reservoir state by one step (streaming, JIT'd)."""
        z            = jnp.array([float(x)])           # (1,)
        s_new        = _step_once(self._basis, jnp.array(self._s_last), z)
        self._s_last = np.array(s_new, dtype=np.float32)
        return self

    def predict(self, h: int) -> float:
        """Linear readout: ŷ_{t+h} = s_last · W_h."""
        return float(self._s_last @ self._W[h])


# ══════════════════════════════════════════════════════════════════════════════
# SASEnsemble — average K independent reservoirs (variance reduction)
# ══════════════════════════════════════════════════════════════════════════════

class SASEnsemble(BaseForecaster):
    """
    Ensemble of K independent SASForecasters averaged at prediction time.

    Each member uses seed = base_seed + k, producing diverse random reservoirs.
    Averaging reduces variance ∝ 1/√K without changing the bias.

    Parameters
    ----------
    K        : number of ensemble members.
    **kwargs : forwarded to each SASForecaster; `seed` is offset by member k.
    """

    def __init__(self, K: int = 5, **kwargs):
        self.K       = K
        base_seed    = kwargs.pop("seed", 42)
        self.members = [
            SASForecaster(seed=base_seed + k, **kwargs) for k in range(K)
        ]

    def fit(self, history: np.ndarray, horizons: list[int]) -> "SASEnsemble":
        for m in self.members:
            m.fit(history, horizons)
        return self

    def update(self, x: float) -> "SASEnsemble":
        for m in self.members:
            m.update(x)
        return self

    def predict(self, h: int) -> float:
        return float(np.mean([m.predict(h) for m in self.members]))


# ══════════════════════════════════════════════════════════════════════════════
# AugSASForecaster — reservoir state augmented with HAR features
# ══════════════════════════════════════════════════════════════════════════════

class AugSASForecaster(BaseForecaster):
    """
    Augmented SAS: concatenates the reservoir state with HAR features before
    the ridge readout.

    Feature vector at time t:
        f_t = [ s_t (n,)  |  1, RV_d, RV_w, RV_m (4,) ]   shape (n+4,)

    The HAR block guarantees the readout is at least as expressive as pure HAR
    (set reservoir weights to zero to recover HAR exactly), while the reservoir
    adds nonlinear long-memory structure on top.

    Parameters
    ----------
    Same as SASForecaster, plus lags_w / lags_m for the HAR feature windows.
    """

    def __init__(
        self,
        n_reservoir:   int   = 200,
        basis                = "diagonal",
        spectral_norm: float = 0.9,
        p_degree:      int   = 1,
        q_degree:      int   = 1,
        washout:       int   = 50,
        chunk_size:    int   = 64,
        n_cv_folds:    int   = 3,
        seed:          int   = 42,
        alphas               = None,
        lags_w:        int   = 5,
        lags_m:        int   = 22,
    ):
        self._sas = SASForecaster(
            n_reservoir=n_reservoir, basis=basis,
            spectral_norm=spectral_norm, p_degree=p_degree, q_degree=q_degree,
            washout=washout, chunk_size=chunk_size, n_cv_folds=n_cv_folds,
            seed=seed, alphas=alphas,
        )
        self.lags_w     = lags_w
        self.lags_m     = lags_m
        self.n_cv_folds = n_cv_folds
        self.alphas     = list(alphas) if alphas is not None else _ALPHAS
        self._buf: deque | None        = None
        self._W: dict[int, np.ndarray] = {}

    def _har_feat(self) -> np.ndarray:
        arr  = np.array(self._buf)
        rv_d = arr[-1]
        rv_w = arr[-min(self.lags_w, len(arr)):].mean()
        rv_m = arr[-min(self.lags_m, len(arr)):].mean()
        return np.array([1.0, rv_d, rv_w, rv_m], dtype=np.float32)

    def fit(self, history: np.ndarray, horizons: list[int]) -> "AugSASForecaster":
        history = np.asarray(history, dtype=np.float32)
        T       = len(history)
        n       = self._sas.n_reservoir

        # ── run reservoir (re-init + parallel scan) ────────────────────────
        u              = jnp.array(history[:, None])
        key            = jax.random.PRNGKey(self._sas.seed)
        self._sas._basis = self._sas._basis.initialize(n, key)

        s0             = jnp.zeros(n)
        states, s_last = _collect_states(
            self._sas._basis, u, s0, n, self._sas.chunk_size
        )
        states_np         = np.array(states, dtype=np.float32)   # (T, n)
        self._sas._s_last = np.array(s_last, dtype=np.float32)

        # ── HAR feature matrix ─────────────────────────────────────────────
        har_buf   = deque(maxlen=self.lags_m)
        har_feats = []
        for t in range(T):
            har_buf.append(float(history[t]))
            arr  = np.array(har_buf)
            rv_d = arr[-1]
            rv_w = arr[-min(self.lags_w, len(arr)):].mean()
            rv_m = arr[-min(self.lags_m, len(arr)):].mean()
            har_feats.append([1.0, rv_d, rv_w, rv_m])
        har_feats = np.array(har_feats, dtype=np.float32)     # (T, 4)

        aug   = np.concatenate([states_np, har_feats], axis=1) # (T, n+4)
        n_aug = aug.shape[1]
        self._buf = deque(history[-self.lags_m:].tolist(), maxlen=self.lags_m)

        # ── ridge per horizon ──────────────────────────────────────────────
        wo    = self._sas.washout
        self._W = {}
        for h in horizons:
            S = aug    [wo: T - h]
            Y = history[wo + h: T]
            if len(S) < 5:
                self._W[h] = np.zeros(n_aug, dtype=np.float32)
                continue
            alpha    = _ridge_cv(S, Y, self.n_cv_folds, self.alphas)
            self._W[h] = _ridge_fit(S, Y, alpha).astype(np.float32)

        return self

    def update(self, x: float) -> "AugSASForecaster":
        self._sas.update(x)
        self._buf.append(float(x))
        return self

    def predict(self, h: int) -> float:
        s   = self._sas._s_last.astype(np.float32)
        har = self._har_feat()
        f   = np.concatenate([s, har])
        return float(f @ self._W[h])
