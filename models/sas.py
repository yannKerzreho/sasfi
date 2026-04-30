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
from .ridge import (
    ALPHAS        as _ALPHAS,
    ALPHAS_GROUPED as _ALPHAS_GROUPED,
    ridge_cv_select as _ridge_cv,
    ridge_fit       as _ridge_fit,
    ridge_cv_grouped     as _ridge_cv_grouped,
    ridge_fit_grouped    as _ridge_fit_grouped,
)


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
        n_reservoir:     int   = 100,
        basis                  = "diagonal",
        spectral_norm:   float = 0.9,
        p_degree:        int   = 1,
        q_degree:        int   = 1,
        washout:         int   = 50,
        chunk_size:      int   = 64,
        n_cv_folds:      int   = 3,
        seed:            int   = 42,
        alphas                 = None,
        apply_log:       bool  = False,
        input_scale:     float = 4.0,
        clip:            bool  = True,
        target_log:      bool  = False,
        residual_target: bool  = False,
    ):
        self.n_reservoir     = n_reservoir
        self.washout         = washout
        self.chunk_size      = chunk_size
        self.n_cv_folds      = n_cv_folds
        self.seed            = seed
        self.alphas          = list(alphas) if alphas is not None else _ALPHAS

        # Data Pipeline Controls
        self.apply_log       = apply_log
        self.target_log      = target_log
        self.input_scale     = input_scale
        self.clip            = clip

        # Residual-target mode: fit readout on (y_{t+h} - y_t) and add anchor
        # at prediction time.  Anchor stays positive → improves QLIKE stability.
        self.residual_target = residual_target
        self._anchor_last:  float = 0.0   # last known target value (raw/log scale)
        self._mu_resid:     dict  = {}    # per-horizon residual mean
        self._sigma_resid:  dict  = {}    # per-horizon residual std

        if isinstance(basis, str):
            _map = {"diagonal": DiagonalPoly, "linear": LinearPoly, "trigo": TrigoPoly}
            if basis not in _map:
                raise ValueError(f"Unknown basis string '{basis}'.")
            self._basis: BasePoly = _map[basis](p_degree, q_degree, spectral_norm)
        elif isinstance(basis, BasePoly):
            self._basis = basis
        else:
            raise TypeError("basis must be a str or BasePoly instance")

        self._W: dict[int, np.ndarray] = {}
        self._s_last: np.ndarray | None = None

    def _to_log(self, arr: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(arr, 1e-10))

    def _to_log_scalar(self, x: float) -> float:
        return float(np.log(max(x, 1e-10)))

    def fit(self, history: np.ndarray, horizons: list[int]) -> "SASForecaster":
        history = np.asarray(history, dtype=np.float64)
        
        # 1. Prepare Target Data
        if self.target_log:
            target_raw = self._to_log(history)
        else:
            target_raw = history

        self._mu_target, self._sigma_target = self._fit_scaler(target_raw)
        Y_z = self._zscore(target_raw, self._mu_target, self._sigma_target).astype(np.float32)

        # Store anchor = last training value in target space (raw or log).
        # Used by residual_target mode at predict time.
        self._anchor_last = float(target_raw[-1])

        # 2. Prepare Input Data
        if self.apply_log:
            input_raw = self._to_log(history)
        else:
            input_raw = history
            
        self._mu_input, self._sigma_input = self._fit_scaler(input_raw)
        history_z = self._zscore(input_raw, self._mu_input, self._sigma_input)
        
        # Scale and clip for reservoir
        if self.clip:
            history_u = np.clip(history_z / self.input_scale, -1.0, 1.0)
        else:
            history_u = history_z / self.input_scale
            
        history_u = history_u.astype(np.float32)

        T = len(history_u)
        u = jnp.array(history_u[:, None])

        key         = jax.random.PRNGKey(self.seed)
        self._basis = self._basis.initialize(self.n_reservoir, key)

        s0             = jnp.zeros(self.n_reservoir)
        states, s_last = _collect_states(self._basis, u, s0, self.n_reservoir, self.chunk_size)
        states_np      = np.array(states, dtype=np.float32)
        self._s_last   = np.array(s_last, dtype=np.float32)

        n  = self.n_reservoir
        wo = self.washout
        self._W = {}
        self.alpha_log_: dict[int, float] = {}
        self._mu_resid    = {}
        self._sigma_resid = {}

        for h in horizons:
            S = states_np[wo: T - h]
            if len(S) < 5:
                self._W[h] = np.zeros(n, dtype=np.float32)
                continue

            if self.residual_target:
                # Target = y_{t+h} - y_t  (in raw/log target space)
                R_raw = (target_raw[wo + h: T]
                         - target_raw[wo    : T - h])
                mu_r, sigma_r          = self._fit_scaler(R_raw)
                self._mu_resid[h]      = mu_r
                self._sigma_resid[h]   = sigma_r
                Y_h = self._zscore(R_raw, mu_r, sigma_r).astype(np.float32)
            else:
                Y_h = Y_z[wo + h: T]

            alpha              = _ridge_cv(S, Y_h, self.n_cv_folds, self.alphas)
            self._W[h]         = _ridge_fit(S, Y_h, alpha).astype(np.float32)
            self.alpha_log_[h] = alpha

        return self

    def update(self, x: float) -> "SASForecaster":
        x_raw = float(x)
        if self.apply_log:
            x_raw = self._to_log_scalar(x_raw)

        x_z = float(self._zscore(x_raw, self._mu_input, self._sigma_input))

        if self.clip:
            x_u = float(np.clip(x_z / self.input_scale, -1.0, 1.0))
        else:
            x_u = float(x_z / self.input_scale)

        z = jnp.array([x_u], dtype=jnp.float32)
        s_new = _step_once(self._basis, jnp.array(self._s_last), z)
        self._s_last = np.array(s_new, dtype=np.float32)

        # Track anchor for residual_target mode:
        # store last known value in target space (log or level).
        if self.residual_target:
            x_target = float(x)
            if self.target_log:
                x_target = self._to_log_scalar(x_target)
            self._anchor_last = x_target

        return self

    def predict(self, h: int) -> float:
        raw_out = float(self._s_last @ self._W[h])

        if self.residual_target:
            # Unscale the predicted increment, then add the last known anchor.
            r = float(self._unzscore(raw_out,
                                     self._mu_resid[h],
                                     self._sigma_resid[h]))
            y = self._anchor_last + r
            if self.target_log:
                return float(np.exp(y))
            return float(y)

        y_unscaled = float(self._unzscore(raw_out,
                                          self._mu_target,
                                          self._sigma_target))
        if self.target_log:
            return float(np.exp(y_unscaled))
        return y_unscaled


# ══════════════════════════════════════════════════════════════════════════════
# SASEnsemble
# ══════════════════════════════════════════════════════════════════════════════

class SASEnsemble(BaseForecaster):
    def __init__(self, K: int = 5, apply_log: bool = False, input_scale: float = 4.0, target_log: bool = False, **kwargs):
        self.K       = K
        base_seed    = kwargs.pop("seed", 42)
        self.members = [
            SASForecaster(seed=base_seed + k, apply_log=apply_log,
                          input_scale=input_scale, target_log=target_log, **kwargs)
            for k in range(K)
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
# AugSASForecaster
# ══════════════════════════════════════════════════════════════════════════════

class AugSASForecaster(BaseForecaster):
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
        apply_log:     bool  = False,
        input_scale:   float = 4.0,
        clip:          bool  = True,
        target_log:    bool  = False,
    ):
        self._sas = SASForecaster(
            n_reservoir=n_reservoir, basis=basis,
            spectral_norm=spectral_norm, p_degree=p_degree, q_degree=q_degree,
            washout=washout, chunk_size=chunk_size, n_cv_folds=n_cv_folds,
            seed=seed, alphas=alphas, apply_log=apply_log, input_scale=input_scale,
            clip=clip, target_log=target_log,
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
        history = np.asarray(history, dtype=np.float64)
        T       = len(history)
        n       = self._sas.n_reservoir

        # Input preprocessing 
        if self._sas.apply_log:
            history_input = self._sas._to_log(history)
        else:
            history_input = history
            
        self._sas._mu_input, self._sas._sigma_input = self._sas._fit_scaler(history_input)
        history_z_input = self._sas._zscore(history_input, self._sas._mu_input, self._sas._sigma_input)
        
        if self._sas.clip:
            history_u = np.clip(history_z_input / self._sas.input_scale, -1.0, 1.0).astype(np.float32)
        else:
            history_u = (history_z_input / self._sas.input_scale).astype(np.float32)

        # Target preprocessing
        if self._sas.target_log:
            history_target = self._sas._to_log(history)
        else:
            history_target = history
            
        self._sas._mu_target, self._sas._sigma_target = self._sas._fit_scaler(history_target)
        Y_z = self._sas._zscore(history_target, self._sas._mu_target, self._sas._sigma_target).astype(np.float32)

        # Run reservoir
        u              = jnp.array(history_u[:, None])
        key            = jax.random.PRNGKey(self._sas.seed)
        self._sas._basis = self._sas._basis.initialize(n, key)

        s0             = jnp.zeros(n)
        states, s_last = _collect_states(self._sas._basis, u, s0, n, self._sas.chunk_size)
        states_np         = np.array(states, dtype=np.float32)
        self._sas._s_last = np.array(s_last, dtype=np.float32)

        # HAR features (using the TARGET distribution for HAR logic)
        har_buf   = deque(maxlen=self.lags_m)
        har_feats = []
        for t in range(T):
            har_buf.append(float(Y_z[t]))
            arr  = np.array(har_buf)
            rv_d = arr[-1]
            rv_w = arr[-min(self.lags_w, len(arr)):].mean()
            rv_m = arr[-min(self.lags_m, len(arr)):].mean()
            har_feats.append([1.0, rv_d, rv_w, rv_m])
            
        har_feats = np.array(har_feats, dtype=np.float32)
        aug   = np.concatenate([states_np, har_feats], axis=1)
        n_aug = aug.shape[1]
        self._buf = deque(Y_z[-self.lags_m:].tolist(), maxlen=self.lags_m)

        # Ridge
        wo    = self._sas.washout
        self._W = {}
        for h in horizons:
            S = aug[wo: T - h]
            Y = Y_z[wo + h: T]
            if len(S) < 5:
                self._W[h] = np.zeros(n_aug, dtype=np.float32)
                continue
            alpha    = _ridge_cv(S, Y, self.n_cv_folds, self.alphas)
            self._W[h] = _ridge_fit(S, Y, alpha).astype(np.float32)

        return self

    def update(self, x: float) -> "AugSASForecaster":
        self._sas.update(x)
        
        # Update HAR buffer with the target distribution
        x_raw = float(x)
        if self._sas.target_log:
            x_raw = self._sas._to_log_scalar(x_raw)
            
        x_z_target = float(self._sas._zscore(x_raw, self._sas._mu_target, self._sas._sigma_target))
        self._buf.append(x_z_target)
        return self

    def predict(self, h: int) -> float:
        s   = self._sas._s_last.astype(np.float32)
        har = self._har_feat()
        f   = np.concatenate([s, har])
        y_z_target = float(f @ self._W[h])
        
        y_unscaled = float(self._sas._unzscore(y_z_target, self._sas._mu_target, self._sas._sigma_target))
        if self._sas.target_log:
            return float(np.exp(y_unscaled))
        return y_unscaled


# ══════════════════════════════════════════════════════════════════════════════
# MultiDegreeSASForecaster
# ══════════════════════════════════════════════════════════════════════════════

class MultiDegreeSASForecaster(BaseForecaster):
    def __init__(
        self,
        n_per_group:   int         = 50,
        max_degree:    int         = 1,
        q_degree:      int         = 1,
        spectral_norm: float       = 0.95,
        washout:       int         = 50,
        chunk_size:    int         = 64,
        n_cv_folds:    int         = 3,
        seed:          int         = 42,
        alphas_1d                  = None,
        grouped_ridge: bool        = True,
        p_degrees:     list | None = None,
        q_degrees:     list | None = None,
        apply_log:     bool        = False,
        input_scale:   float       = 4.0,
        clip:          bool        = True,
        target_log:    bool        = False,
    ):
        self.n_per_group   = n_per_group
        self.max_degree    = max_degree
        self.q_degree      = q_degree
        self.spectral_norm = spectral_norm
        self.washout       = washout
        self.chunk_size    = chunk_size
        self.n_cv_folds    = n_cv_folds
        self.seed          = seed
        self.alphas_1d     = list(alphas_1d) if alphas_1d is not None else _ALPHAS_GROUPED
        self.grouped_ridge = grouped_ridge
        
        self.apply_log     = apply_log
        self.input_scale   = input_scale
        self.clip          = clip
        self.target_log    = target_log

        _p_list = list(p_degrees) if p_degrees is not None else list(range(max_degree + 1))
        _q_list = list(q_degrees) if q_degrees is not None else [q_degree] * len(_p_list)
        if len(_p_list) != len(_q_list):
            raise ValueError("p_degrees and q_degrees must have same length")
        self._p_list = _p_list
        self._q_list = _q_list

        self._bases: list = [
            DiagonalPoly(p_degree=p_d, q_degree=q_d, spectral_norm=spectral_norm)
            for p_d, q_d in zip(_p_list, _q_list)
        ]
        self._s_lasts: list                  = []
        self._W:       dict[int, np.ndarray] = {}

    @property
    def n_groups(self) -> int: return len(self._bases)
    @property
    def n_reservoir(self) -> int: return self.n_per_group * self.n_groups
    @property
    def group_sizes(self) -> list: return [self.n_per_group] * self.n_groups

    def fit(self, history: np.ndarray, horizons: list[int]) -> "MultiDegreeSASForecaster":
        history = np.asarray(history, dtype=np.float64)
        T       = len(history)

        # Target Preprocessing
        if self.target_log:
            history_target = np.log(np.maximum(history, 1e-10))
        else:
            history_target = history
            
        self._mu_target, self._sigma_target = self._fit_scaler(history_target)
        Y_z = self._zscore(history_target, self._mu_target, self._sigma_target).astype(np.float32)

        # Input Preprocessing
        if self.apply_log:
            history_input = np.log(np.maximum(history, 1e-10))
        else:
            history_input = history
            
        self._mu_input, self._sigma_input = self._fit_scaler(history_input)
        history_z_input = self._zscore(history_input, self._mu_input, self._sigma_input)
        
        if self.clip:
            history_u = np.clip(history_z_input / self.input_scale, -1.0, 1.0).astype(np.float32)
        else:
            history_u = (history_z_input / self.input_scale).astype(np.float32)

        u        = jnp.array(history_u[:, None])
        base_key = jax.random.PRNGKey(self.seed)
        states_list: list[np.ndarray] = []
        self._s_lasts = []

        for d, basis in enumerate(self._bases):
            sub_key       = jax.random.fold_in(base_key, d)
            init_basis    = basis.initialize(self.n_per_group, sub_key)
            self._bases[d] = init_basis

            s0             = jnp.zeros(self.n_per_group)
            states_d, s_last_d = _collect_states(init_basis, u, s0, self.n_per_group, self.chunk_size)
            states_list.append(np.array(states_d,  dtype=np.float32))
            self._s_lasts.append(np.array(s_last_d, dtype=np.float32))

        all_states = np.concatenate(states_list, axis=1)
        gsizes     = self.group_sizes
        wo         = self.washout

        self._W = {}
        self.alpha_log_: dict[int, object] = {}

        for h in horizons:
            S = all_states[wo: T - h]
            Y = Y_z       [wo + h: T]
            if len(S) < 5:
                self._W[h] = np.zeros(self.n_reservoir, dtype=np.float32)
                continue
            if self.grouped_ridge:
                best_alphas      = _ridge_cv_grouped(S, Y, gsizes, self.n_cv_folds, self.alphas_1d)
                self._W[h]       = _ridge_fit_grouped(S, Y, gsizes, best_alphas).astype(np.float32)
                self.alpha_log_[h] = best_alphas
            else:
                alpha            = _ridge_cv(S, Y, self.n_cv_folds, self.alphas_1d)
                self._W[h]       = _ridge_fit(S, Y, alpha).astype(np.float32)
                self.alpha_log_[h] = alpha

        return self

    def update(self, x: float) -> "MultiDegreeSASForecaster":
        x_raw = float(x)
        if self.apply_log:
            x_raw = float(np.log(max(x_raw, 1e-10)))
            
        x_z = float(self._zscore(x_raw, self._mu_input, self._sigma_input))
        
        if self.clip:
            x_u = float(np.clip(x_z / self.input_scale, -1.0, 1.0))
        else:
            x_u = float(x_z / self.input_scale)
            
        z = jnp.array([x_u], dtype=jnp.float32)
        for d, basis in enumerate(self._bases):
            s_new = _step_once(basis, jnp.array(self._s_lasts[d]), z)
            self._s_lasts[d] = np.array(s_new, dtype=np.float32)
        return self

    def predict(self, h: int) -> float:
        s = np.concatenate(self._s_lasts)
        y_z_target = float(s @ self._W[h])
        
        y_unscaled = float(self._unzscore(y_z_target, self._mu_target, self._sigma_target))
        if self.target_log:
            return float(np.exp(y_unscaled))
        return y_unscaled