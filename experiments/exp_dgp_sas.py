"""
experiments/exp_dgp_sas.py — SAS-as-DGP specification experiment.

Design
------
For each real-data symbol with T ≥ MIN_REAL_T and each DGP seed in DGP_SEEDS:

  Step 1 — DGP fit
    Fit SAS(p,q) (seed=dgp_seed) on the full real-data series (all T obs).
    Gives: _basis (P,Q), _W[h] for h in {1,5,10}, _s_last (=s_T),
           residual noise scale σ, and the CV-selected lambda per horizon.

  Step 2 — Autoregressive generation from s_T  (N_GEN = 4 000 steps)
    z_{t+1} = W_1·s_t + σ·ε,  ε ~ t(ν)   [no clipping]
    s_{t+1} = P(z_{t+1}) ⊙ s_t + Q(z_{t+1})
    Stability: DiagonalPoly.eval_p clips P(z) to (−0.9999, 0.9999) for any z,
    so the state dynamics are unconditionally contractive.
    SVD of state matrix S ∈ R^{N_GEN × n} is computed for q1/q2 comparison.

  Step 3 — Rolling OOS on synthetic series (W=2000, R=20, H=[1,5,10])
    1. Noiseless       — oracle floor: DGP's fixed W_h + DGP's reservoir
    2. SAS_same_fixedλ — same reservoir as DGP (same seed), fixed λ = DGP's λ
    3. SAS_same_cvλ    — same reservoir as DGP (same seed), rolling CV for λ
    4. SAS_diff        — right degree, unique foreign seed, CV for λ
    5. SAS_q{alt}      — wrong degree, unique foreign seed, CV for λ
    6. HAR, DLinear, GARCH  — standard benchmarks

  Lambda analysis
    - DGP's λ (per horizon, from fitting on full real series)
    - SAS_same_fixedλ: uses DGP's λ verbatim
    - SAS_same_cvλ / SAS_diff: CV-selected λ tracked per refit (AlphaTracker)
    - Oracle λ: post-hoc best λ that minimises total OOS MSE (custom loop)

  SVD analysis
    SVD of the state matrix S ∈ R^{N_GEN × n} from the DGP generation.
    Effective rank and 95%-variance rank reported for DGP degree.

Usage
-----
    python experiments/exp_dgp_sas.py                        # all symbols, q=2 DGP
    python experiments/exp_dgp_sas.py --symbols .AEX .SPX   # quick test
    python experiments/exp_dgp_sas.py --q 1                  # SAS_q1 as DGP
    python experiments/exp_dgp_sas.py --nu 0                 # Gaussian noise
    python experiments/exp_dgp_sas.py --no-mcs               # skip MCS
    python experiments/exp_dgp_sas.py --no-svd               # skip SVD analysis
    python experiments/exp_dgp_sas.py --no-oracle-lambda     # skip oracle λ
"""

from __future__ import annotations

import sys
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    quick_oos, print_mse_table, print_mcs_frequency,
    _sq_errors_to_eval_df,
)


def _quick_oos_raw(
    synth:      np.ndarray,
    models:     dict,
    horizons:   list[int],
    window:     int,
    refit_freq: int,
) -> dict:
    """
    Rolling OOS without any inner z-scoring.

    The synthetic series is already in z-space (DGP output), so no further
    normalisation is applied.  All models receive and predict in the same
    z-space as the DGP's readout W_h — making Noiseless the true oracle floor.
    """
    T     = len(synth)
    H_max = max(horizons)
    steps_since_refit = {n: refit_freq for n in models}
    losses = {n: {h: [] for h in horizons} for n in models}

    for t in range(window, T - H_max):
        train = synth[t - window: t]

        for name, model in models.items():
            if steps_since_refit[name] >= refit_freq:
                try:
                    model.fit(train, horizons)
                    steps_since_refit[name] = 0
                except Exception as e:
                    print(f"  [fit {name}] {e}")

        z_t = float(synth[t])

        # Update FIRST so s_last = s_t before predicting.
        # After fit(), s_last = s_{t-1}.  The DGP convention is
        #   z_{t+1} = W · s_t + ε,
        # so we must advance to s_t before calling predict(h),
        # otherwise predict(1) computes W·s_{t-1} (= mean of z_t, not z_{t+1}).
        for name, model in models.items():
            try:
                model.update(z_t)
                steps_since_refit[name] += 1
            except Exception as e:
                print(f"  [upd {name}] {e}")

        for name, model in models.items():
            for h in horizons:
                if t + h >= T:
                    continue
                z_tgt = float(synth[t + h])
                try:
                    losses[name][h].append((float(model.predict(h)) - z_tgt) ** 2)
                except Exception as e:
                    print(f"  [pred {name} h={h}] {e}")

    return losses
from data.data_loader import load_rv, available_symbols, fit_scaler, apply_scaler
from models.linear import HARForecaster, DLinearForecaster
from models.garch  import GARCHForecaster
from models.sas    import (
    SASForecaster, _ALPHAS, _collect_states, _step_once,
    _ridge_cv, _ridge_fit,
)

CSV        = ROOT / "rv.csv"
HORIZONS   = [1]
WINDOW     = 2000
REFIT_FREQ = 20
N_GEN      = 4_000
MIN_REAL_T = 4_000.
DGP_SEEDS  = [42, 137, 256]
ALT_SEED   = 999
WASHOUT    = 50
SN         = 0.95
NU         = 5
DIVERGE_THRESH = 1.5
# Z_CLIP removed: output clipping was causing oracle bias (cond_means[t] ≠ E[synth[t]|F_{t-1}]).
# Safety: DiagonalPoly.eval_p already clips P(z) to (−0.9999, 0.9999) for any z,
# so the state recurrence is unconditionally contractive.  Q(z) spikes from heavy-tailed
# noise decay at rate 0.9999/step.  Diverged series are caught by the isfinite /
# synth.std check below.
CHUNK      = 64
N_RES      = 50


def _model_order(q_degree: int) -> list[str]:
    alt_q = 1 if q_degree != 1 else 2
    return [
        "Noiseless", "SAS_same_fixedλ", "SAS_same_cvλ",
        "SAS_diff", f"SAS_q{alt_q}",
        "HAR", "DLinear", "GARCH",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Custom forecaster classes
# ══════════════════════════════════════════════════════════════════════════════

class NoiselessForecaster(SASForecaster):
    """
    Oracle floor: uses the DGP's fixed W_h readouts with the DGP's reservoir.

    State evolves under P(z), Q(z) (same matrices as DGP, via same seed).
    The readout is NEVER refitted — it is the optimal linear readout fitted
    on the full real-data series.  This is the conditional mean of the DGP
    given the current state: the theoretical MSE floor.
    """

    def __init__(self, dgp_model: SASForecaster, **kwargs):
        super().__init__(**kwargs)
        # Store DGP's readouts — never overwritten
        self._fixed_W: dict[int, np.ndarray] = {
            h: w.copy() for h, w in dgp_model._W.items()
        }

    def fit(self, history: np.ndarray, horizons: list[int]) -> "NoiselessForecaster":
        """Run the reservoir scan (to update s_last) but keep DGP's readout."""
        history = np.asarray(history, dtype=np.float32)
        u       = jnp.array(history[:, None])
        key     = jax.random.PRNGKey(self.seed)
        self._basis = self._basis.initialize(self.n_reservoir, key)
        s0 = jnp.zeros(self.n_reservoir)
        _, s_last = _collect_states(self._basis, u, s0, self.n_reservoir, self.chunk_size)
        self._s_last = np.array(s_last, dtype=np.float32)
        self._W      = dict(self._fixed_W)    # fixed DGP readout, never refitted
        return self


class SASFixedLambda(SASForecaster):
    """
    SAS with a fixed ridge lambda per horizon (no cross-validation).

    Used to isolate the cost of CV lambda selection: same reservoir as DGP
    (same seed), but the ridge alpha is fixed to the DGP's value instead of
    being re-estimated from each OOS window.
    """

    def __init__(self, fixed_alphas: dict[int, float], **kwargs):
        super().__init__(**kwargs)
        self._fixed_alphas: dict[int, float] = dict(fixed_alphas)

    def fit(self, history: np.ndarray, horizons: list[int]) -> "SASFixedLambda":
        history = np.asarray(history, dtype=np.float32)
        T       = len(history)
        u       = jnp.array(history[:, None])
        key     = jax.random.PRNGKey(self.seed)
        self._basis = self._basis.initialize(self.n_reservoir, key)
        s0 = jnp.zeros(self.n_reservoir)
        states, s_last = _collect_states(
            self._basis, u, s0, self.n_reservoir, self.chunk_size
        )
        states_np    = np.array(states,  dtype=np.float32)
        self._s_last = np.array(s_last, dtype=np.float32)
        self._W          = {}
        self.alpha_log_  = {}
        n = self.n_reservoir
        for h in horizons:
            S = states_np[self.washout: T - h]
            Y = history  [self.washout + h: T]
            if len(S) < 5:
                self._W[h] = np.zeros(n, dtype=np.float32)
                continue
            alpha           = self._fixed_alphas.get(h, self.alphas[0])
            self._W[h]      = _ridge_fit(S, Y, alpha).astype(np.float32)
            self.alpha_log_[h] = alpha
        return self


class AlphaTracker:
    """
    Thin wrapper that records the CV-selected alpha after every refit.

    Delegates all BaseForecaster calls to the wrapped model.
    """

    def __init__(self, model: SASForecaster):
        self.model         = model
        self.alpha_history: list[dict[int, float]] = []

    def fit(self, history, horizons):
        self.model.fit(history, horizons)
        if hasattr(self.model, "alpha_log_"):
            self.alpha_history.append(dict(self.model.alpha_log_))
        return self

    def update(self, x):
        self.model.update(x)
        return self

    def predict(self, h):
        return self.model.predict(h)


# ══════════════════════════════════════════════════════════════════════════════
# DGP class
# ══════════════════════════════════════════════════════════════════════════════

class SASDGP:
    """
    Autoregressive DGP built from a fitted SASForecaster.

    z_{t+1} = W_1·s_t + σ·ε,   ε ~ t(ν)   (no clipping — see note above)
    s_{t+1} = P(z_{t+1}) ⊙ s_t + Q(z_{t+1})
    """

    def __init__(self, model: SASForecaster, sigma: float, nu: int | None):
        self._basis = model._basis
        self._W1    = model._W[1]
        self._s0    = model._s_last.copy()
        self.sigma  = float(sigma)
        self.nu     = nu

    def generate(
        self,
        n_steps:           int,
        seed:              int  = 0,
        return_states:     bool = False,
        return_cond_means: bool = False,
    ) -> np.ndarray | tuple:
        """
        Generate n_steps of synthetic z-scored log-RV.

        Parameters
        ----------
        return_states     : also return the (n_steps, n) state matrix S where
                            S[t] = s_{t+1}^{DGP} (state AFTER consuming out[t]).
        return_cond_means : also return cond_means[t] = s_t^{DGP} @ W1, i.e.
                            the DGP's conditional mean used to generate out[t].
                            The oracle h=1 prediction for out[t+1] at time t
                            is cond_means[t+1].
        """
        rng = np.random.default_rng(seed)
        s   = jnp.array(self._s0)
        W1  = jnp.array(self._W1)
        out        = np.empty(n_steps, dtype=np.float32)
        states     = np.empty((n_steps, len(self._s0)), dtype=np.float32) if return_states else None
        cond_means = np.empty(n_steps, dtype=np.float32)   # always computed (cheap)

        for t in range(n_steps):
            z_hat       = float(s @ W1)
            cond_means[t] = z_hat
            eps         = (float(rng.standard_t(self.nu))
                           if self.nu is not None
                           else float(rng.standard_normal()))
            z_next      = z_hat + self.sigma * eps   # no clipping — oracle bias removed
            out[t]      = z_next
            s           = _step_once(self._basis, s, jnp.array([z_next], dtype=jnp.float32))
            if states is not None:
                states[t] = np.array(s, dtype=np.float32)

        ret = (out,)
        if return_states:
            ret += (states,)
        if return_cond_means:
            ret += (cond_means,)
        return ret[0] if len(ret) == 1 else ret


# ══════════════════════════════════════════════════════════════════════════════
# Noiseless oracle validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate_noiseless(
    dgp:        SASDGP,
    synth:      np.ndarray,
    cond_means: np.ndarray,
    n_check:    int = 200,
    tol:        float = 1e-4,
) -> float:
    """
    Verify that stepping through synth from s_T^{real} recovers cond_means exactly.

    This confirms:
      (a) the generation is deterministic and self-consistent, and
      (b) NoiselessForecaster (continuous, no refit reset) would produce
          predictions identical to the stored conditional means.

    Returns the max absolute error over the first n_check steps.
    """
    s  = jnp.array(dgp._s0)
    W1 = jnp.array(dgp._W1)
    max_err = 0.0
    for t in range(min(n_check, len(synth))):
        z_hat   = float(s @ W1)
        max_err = max(max_err, abs(z_hat - float(cond_means[t])))
        s       = _step_once(dgp._basis, s, jnp.array([synth[t]], dtype=jnp.float32))
    if max_err > tol:
        raise RuntimeError(
            f"Noiseless validation FAILED: max |z_hat − cond_means| = {max_err:.2e} > {tol:.2e}. "
            "This indicates a mismatch between DGP generation and the stored conditional means."
        )
    return max_err


# ══════════════════════════════════════════════════════════════════════════════
# DGP fitting
# ══════════════════════════════════════════════════════════════════════════════

def _fit_dgp(
    log_values: np.ndarray,
    q_degree:   int,
    p_degree:   int,
    n_res:      int,
    dgp_seed:   int,
) -> tuple[SASForecaster, float]:
    """
    Fit SAS(p,q) on full z-scored real-data series.

    Returns (model, sigma) where:
      - model._W[h]       : DGP readout per horizon h in {1,5,10}
      - model.alpha_log_  : CV-selected ridge alpha per horizon
      - model._s_last     : state s_T (= starting point for DGP generation)
      - sigma             : std of h=1 in-sample residuals
    """
    mu_z, sd_z = fit_scaler(log_values)
    z = apply_scaler(log_values, mu_z, sd_z).astype(np.float32)
    T = len(z)

    model = SASForecaster(
        n_reservoir   = n_res,
        basis         = "diagonal",
        spectral_norm = SN,
        p_degree      = p_degree,
        q_degree      = q_degree,
        washout       = WASHOUT,
        chunk_size    = CHUNK,
        seed          = dgp_seed,
        alphas        = _ALPHAS,
    )
    model.fit(z, horizons=HORIZONS)   # fits W[1], W[5], W[10] with CV

    # Compute h=1 residual noise scale from in-sample predictions
    u        = jnp.array(z[:, None])
    s0       = jnp.zeros(n_res)
    all_s, _ = _collect_states(model._basis, u, s0, n_res, CHUNK)
    S        = np.array(all_s[WASHOUT: T - 1], dtype=np.float32)
    Y        = z[WASHOUT + 1: T]
    sigma    = float(np.std(Y - S @ model._W[1]))

    return model, sigma


# ══════════════════════════════════════════════════════════════════════════════
# SVD analysis
# ══════════════════════════════════════════════════════════════════════════════

def _svd_stats(state_matrix: np.ndarray) -> dict:
    """
    Compute SVD statistics of the DGP state matrix S ∈ R^{T × n}.

    The SVD S = UΣV^T reveals the effective dimensionality of the reservoir:
    how many directions of R^n the dynamics actually use.

    Returns
    -------
    dict with:
      singular_values : (n,) array — sorted descending
      effective_rank  : participation ratio (Σσ)² / Σσ²
      rank_80 / rank_95 / rank_99 : number of singular values to explain
                                    80 / 95 / 99% of total variance
    """
    S  = state_matrix.astype(np.float64)
    _, sv, _ = np.linalg.svd(S, full_matrices=False)
    sv2      = sv ** 2
    total    = sv2.sum()
    cumvar   = np.cumsum(sv2) / total
    eff_rank = float((sv.sum() ** 2) / total)

    def _rank_at(p):
        idx = np.searchsorted(cumvar, p)
        return int(min(idx + 1, len(sv)))

    return {
        "singular_values": sv,
        "effective_rank":  eff_rank,
        "rank_80":  _rank_at(0.80),
        "rank_95":  _rank_at(0.95),
        "rank_99":  _rank_at(0.99),
        "n_total":  len(sv),
    }


def _compute_model_svd(
    synth:    np.ndarray,
    q_degree: int,
    p_degree: int,
    n_res:    int,
    seed:     int,
) -> dict:
    """Compute SVD of the state matrix for SASForecaster(seed, q, p) on synth."""
    key   = jax.random.PRNGKey(seed)
    proto = SASForecaster(
        n_reservoir=n_res, basis="diagonal", spectral_norm=SN,
        p_degree=p_degree, q_degree=q_degree, washout=WASHOUT,
        chunk_size=CHUNK, seed=seed, alphas=_ALPHAS,
    )
    basis  = proto._basis.initialize(n_res, key)
    u      = jnp.array(synth[:, None].astype(np.float32))
    s0     = jnp.zeros(n_res)
    states, _ = _collect_states(basis, u, s0, n_res, CHUNK)
    return _svd_stats(np.array(states, dtype=np.float32))


def _print_svd(svd: dict, label: str) -> None:
    sv = svd["singular_values"]
    print(
        f"    SVD [{label}]  n={svd['n_total']}  "
        f"eff_rank={svd['effective_rank']:.1f}  "
        f"rank@80%={svd['rank_80']}  "
        f"rank@95%={svd['rank_95']}  "
        f"rank@99%={svd['rank_99']}  "
        f"σ_max={sv[0]:.2f}  σ_min={sv[-1]:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Oracle lambda analysis
# ══════════════════════════════════════════════════════════════════════════════

def _oracle_lambda(
    synth:      np.ndarray,
    basis,                      # already-initialised DiagonalPoly (DGP's basis)
    n_res:      int,
    horizons:   list[int],
    window:     int     = WINDOW,
    refit_freq: int     = REFIT_FREQ,
    washout:    int     = WASHOUT,
) -> dict[int, float]:
    """
    Post-hoc oracle lambda: the ridge alpha that would have minimised total
    OOS MSE for SAS_same if chosen with perfect foresight.

    Algorithm
    ---------
    At each refit t we run the associative scan on the training window to get
    states S_train.  We then advance the state step-by-step for the next
    refit_freq steps to build S_oos.  For each alpha in _ALPHAS we compute:

        W_α = ridge(S_train, Y_train, α)
        OOS_err_α += mean( (S_oos @ W_α − y_oos)² )

    The oracle alpha is argmin_α  Σ_refits OOS_err_α.

    Note: this assumes the same reservoir as DGP (basis = DGP's basis,
    re-initialised with the same key at every refit, matching SAS_same_cvλ).
    """
    T        = len(synth)
    n_a      = len(_ALPHAS)
    oos_err  = {h: np.zeros(n_a) for h in horizons}

    s_cur = np.zeros(n_res, dtype=np.float32)

    for t in range(window, T - max(horizons), refit_freq):
        train = synth[t - window: t].astype(np.float32)

        # The DGP's basis is already initialised with the same key as SAS_same
        # (same seed → same P, Q matrices at every refit), so we use it directly.
        u               = jnp.array(train[:, None])
        s0              = jnp.zeros(n_res)
        states_jnp, s_L = _collect_states(basis, u, s0, n_res, CHUNK)
        states_np       = np.array(states_jnp, dtype=np.float32)
        s_cur           = np.array(s_L, dtype=np.float32)

        # OOS states: advance from s_L for the next refit_freq steps
        h_end       = min(t + refit_freq, T - max(horizons))
        n_oos_steps = h_end - t
        oos_states  = np.empty((n_oos_steps, n_res), dtype=np.float32)
        s_tmp       = jnp.array(s_cur)
        for k in range(n_oos_steps):
            z_k   = jnp.array([synth[t + k]], dtype=jnp.float32)
            s_tmp = _step_once(basis, s_tmp, z_k)
            oos_states[k] = np.array(s_tmp, dtype=np.float32)

        for h in horizons:
            S_tr = states_np[washout: window - h]
            Y_tr = train[washout + h: window]
            if len(S_tr) < 5 or n_oos_steps == 0:
                continue

            Y_oos = synth[t + h: h_end + h].astype(np.float32)
            n_use = min(n_oos_steps, len(Y_oos))
            if n_use == 0:
                continue
            S_oos = oos_states[:n_use]
            Y_oos = Y_oos[:n_use]

            for ai, alpha in enumerate(_ALPHAS):
                W   = _ridge_fit(S_tr, Y_tr, alpha)
                err = S_oos @ W - Y_oos
                oos_err[h][ai] += float(np.mean(err ** 2))

    oracle = {h: _ALPHAS[int(np.argmin(oos_err[h]))] for h in horizons}
    # Normalise by number of refits so values are comparable across symbols
    n_refits = max(1, sum(1 for t in range(window, T - max(horizons), refit_freq)))
    oos_err_norm = {h: oos_err[h] / n_refits for h in horizons}
    return oracle, oos_err_norm


def _print_lambda_summary(
    dgp_alpha:        dict[int, float],
    same_fixed_alpha: dict[int, float],
    same_cv_tracker:  AlphaTracker,
    diff_tracker:     AlphaTracker,
    oracle_alpha:     dict[int, float] | None,
) -> None:
    """Print a compact lambda comparison table."""
    print("\n  ── Lambda analysis (log10 scale) ──")
    print(f"  {'horizon':>8}", end="")
    for h in HORIZONS:
        print(f"   h={h:2d}", end="")
    print()

    def _log10(a):
        return f"{np.log10(a):+.2f}" if a and a > 0 else "  n/a"

    def _tracker_summary(tracker: AlphaTracker, h: int) -> str:
        if not tracker.alpha_history:
            return "  n/a"
        vals = [np.log10(d[h]) for d in tracker.alpha_history if h in d]
        if not vals:
            return "  n/a"
        return f"{np.median(vals):+.2f}±{np.std(vals):.2f}"

    rows = [
        ("DGP λ",          lambda h: _log10(dgp_alpha.get(h))),
        ("same_fixedλ",    lambda h: _log10(same_fixed_alpha.get(h))),
        ("same_cvλ (med)", lambda h: _tracker_summary(same_cv_tracker, h)),
        ("SAS_diff (med)", lambda h: _tracker_summary(diff_tracker, h)),
        ("oracle λ",       lambda h: _log10(oracle_alpha.get(h)) if oracle_alpha else "  n/a"),
    ]
    for label, fn in rows:
        print(f"  {label:>16}", end="")
        for h in HORIZONS:
            print(f"  {fn(h):>8}", end="")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def _build_models(
    dgp_model: SASForecaster,
    q_degree:  int,
    p_degree:  int,
    n_res:     int,
    seed_idx:  int,
) -> tuple[dict, AlphaTracker, AlphaTracker]:
    """
    Build all estimator models for one (symbol, dgp_seed) run.

    Returns (models_dict, same_cv_tracker, diff_tracker).

    Seed assignment (unique per model per run):
      SAS_same_*  : dgp_seed  (same reservoir as DGP)
      SAS_diff    : ALT_SEED + seed_idx            (999, 1000, 1001, …)
      SAS_q{alt}  : ALT_SEED + len(DGP_SEEDS) + seed_idx
    """
    n_seeds   = len(DGP_SEEDS)
    seed_diff = ALT_SEED + seed_idx
    alt_q     = 1 if q_degree != 1 else 2
    seed_altq = ALT_SEED + n_seeds + seed_idx
    dgp_seed  = dgp_model.seed

    shared_kw = dict(
        n_reservoir=n_res, basis="diagonal", spectral_norm=SN,
        p_degree=p_degree, washout=WASHOUT, chunk_size=CHUNK, alphas=_ALPHAS,
    )

    same_fixed = SASFixedLambda(
        fixed_alphas=dict(dgp_model.alpha_log_),
        seed=dgp_seed, q_degree=q_degree, **shared_kw,
    )

    same_cv   = SASForecaster(seed=dgp_seed, q_degree=q_degree, **shared_kw)
    diff      = SASForecaster(seed=seed_diff, q_degree=q_degree, **shared_kw)
    altq      = SASForecaster(seed=seed_altq, q_degree=alt_q,   **shared_kw)

    same_cv_tracker = AlphaTracker(same_cv)
    diff_tracker    = AlphaTracker(diff)
    altq_tracker    = AlphaTracker(altq)

    models = {
        "SAS_same_fixedλ": same_fixed,
        "SAS_same_cvλ":    same_cv_tracker,
        "SAS_diff":        diff_tracker,
        f"SAS_q{alt_q}":  altq_tracker,
        "HAR":             HARForecaster(ridge=False),
        "DLinear":         DLinearForecaster(lookback=20, ma_kernel=5),
        "GARCH":           GARCHForecaster(p_ar=5),
    }
    return models, same_cv_tracker, diff_tracker


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_mse_csv(mse_by_sym: dict, stable_syms: list, out_dir: Path, slug: str):
    rows = []
    for sym, sym_mse in mse_by_sym.items():
        for m, h_dict in sym_mse.items():
            row = {"key": sym, "model": m, "stable": sym in stable_syms}
            for h in HORIZONS:
                row[f"mse_h{h}"] = h_dict.get(h, np.nan)
            row["mse_avg"] = float(np.nanmean([h_dict.get(h, np.nan) for h in HORIZONS]))
            rows.append(row)
    path = out_dir / f"{slug}_mse.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  → {path.name}  ({len(rows)} rows)")


def _save_svd_csv(svd_records: list[dict], out_dir: Path, slug: str):
    """Save SVD summary + full singular value spectrum (one row per model/run)."""
    if not svd_records:
        return
    path = out_dir / f"{slug}_svd.csv"
    rows = []
    for rec in svd_records:
        sv  = rec.pop("singular_values", np.array([]))
        row = dict(rec)
        for i, s in enumerate(sv):
            row[f"sv_{i}"] = float(s)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  → {path.name}  ({len(rows)} rows)")


def _save_lambda_csv(lambda_records: list[dict], out_dir: Path, slug: str):
    """
    Save per-refit lambda data in tidy format.

    Columns: sym, dgp_seed, model, horizon, refit_idx, alpha, log10_alpha
      refit_idx = -1  for single-value entries (DGP, oracle, same_fixedλ)
      refit_idx = 0,1,… for per-refit entries (same_cvλ, SAS_diff, SAS_q{alt})
    """
    if not lambda_records:
        return
    path = out_dir / f"{slug}_lambda.csv"
    pd.DataFrame(lambda_records).to_csv(path, index=False)
    print(f"  → {path.name}  ({len(lambda_records)} rows)")


def _save_mse_vs_lambda_csv(records: list[dict], out_dir: Path, slug: str):
    """
    Save OOS MSE vs lambda curve (from oracle computation).

    Columns: sym, dgp_seed, horizon, alpha_idx, alpha, log10_alpha, oos_mse
    One row per (sym, dgp_seed, horizon, alpha) — enables plotting the full
    MSE(λ) curve and checking convexity across seeds/symbols.
    """
    if not records:
        return
    path = out_dir / f"{slug}_mse_vs_lambda.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"  → {path.name}  ({len(records)} rows)")


def _save_mcs_csv(mcs_df: pd.DataFrame, out_dir: Path, slug: str):
    if mcs_df is None or mcs_df.empty:
        return
    path = out_dir / f"{slug}_mcs.csv"
    mcs_df.to_csv(path, index=False)
    print(f"  → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-seed inner run
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_seed(
    sym:            str,
    log_vals:       np.ndarray,
    q_degree:       int,
    p_degree:       int,
    n_res:          int,
    nu:             int | None,
    n_gen:          int,
    dgp_seed:       int,
    seed_idx:       int,
    do_svd:         bool = True,
    do_oracle_lam:  bool = True,
    no_zscale:      bool = False,
) -> tuple[dict | None, dict | None, dict | None]:
    """
    One (symbol, dgp_seed) run.

    Returns (sym_mse, losses, extras) where extras = {svd, oracle_lambda, dgp_alpha}.
    """
    t0 = time.perf_counter()
    dgp_model, sigma = _fit_dgp(log_vals, q_degree, p_degree, n_res, dgp_seed)
    dt_fit = time.perf_counter() - t0

    dgp  = SASDGP(dgp_model, sigma=sigma, nu=nu)

    # Generate synthetic series (+ states for SVD, + conditional means for Noiseless oracle)
    if do_svd:
        synth, state_matrix, cond_means = dgp.generate(
            n_gen, seed=dgp_seed, return_states=True, return_cond_means=True,
        )
    else:
        synth, cond_means = dgp.generate(
            n_gen, seed=dgp_seed, return_states=False, return_cond_means=True,
        )
        state_matrix = None

    # Validate: stepping from s_T^{real} through synth must recover cond_means exactly.
    # This confirms the Noiseless oracle is self-consistent with the DGP.
    val_err = _validate_noiseless(dgp, synth, cond_means)
    print(f"    [noiseless validation] max_err={val_err:.2e} ✓", end="")

    # Divergence check: isfinite catches NaN/Inf; std bounds catch degenerate or
    # explosive series (no clipping → closed-loop DGP can drift for unlucky seeds).
    synth_std = synth.std()
    if not np.isfinite(synth).all() or synth_std < 1e-6 or synth_std > 20.0:
        print(f"    [skip s={dgp_seed}] degenerate/diverged series "
              f"(std={synth_std:.2f}, finite={np.isfinite(synth).all()})")
        return None, None, None

    alt_q = 1 if q_degree != 1 else 2
    models, same_cv_tracker, diff_tracker = _build_models(
        dgp_model, q_degree, p_degree, n_res, seed_idx
    )
    seed_diff = ALT_SEED + seed_idx
    seed_altq = ALT_SEED + len(DGP_SEEDS) + seed_idx
    print(
        f"    dgp={dgp_seed}  diff={seed_diff}  q{alt_q}={seed_altq}  "
        f"σ={sigma:.4f}  synth μ={synth.mean():.3f} σ={synth.std():.3f}  "
        f"fit={dt_fit:.1f}s  "
        f"DGP_λ={{{', '.join(f'h{h}:{np.log10(dgp_model.alpha_log_[h]):+.1f}' for h in HORIZONS)}}}",
        end="",
    )

    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if no_zscale:
            losses = _quick_oos_raw(synth, models, HORIZONS, WINDOW, REFIT_FREQ)
        else:
            dummy_dates = pd.date_range("2000-01-01", periods=n_gen, freq="B")
            losses = quick_oos(
                synth, dummy_dates, models,
                horizons=HORIZONS, window=WINDOW, refit_freq=REFIT_FREQ,
            )
    print(f"  OOS={time.perf_counter()-t1:.1f}s")

    # ── Noiseless oracle: direct from DGP conditional means ──────────────────
    # cond_means[t] = s_t^{DGP} @ W1 = conditional mean used to generate synth[t].
    # Oracle h-step prediction for synth[t+h] at step t uses cond_means[t+h]
    # (the conditional mean computed right before generating synth[t+h], i.e.
    # after the state has been stepped through synth[t+h-1]).
    # This is the true noise floor: error = synth[t+h] − cond_means[t+h] = σ·ε_{t+h}.
    T_synth = len(synth)
    noiseless_losses: dict[int, list[float]] = {h: [] for h in HORIZONS}
    for t in range(WINDOW, T_synth - max(HORIZONS)):
        for h in HORIZONS:
            if t + h >= T_synth:
                continue
            err = float(cond_means[t + h]) - float(synth[t + h])
            noiseless_losses[h].append(err * err)
    losses["Noiseless"] = noiseless_losses

    sym_mse = {
        name: {h: float(np.mean(errs)) for h, errs in h_map.items()}
        for name, h_map in losses.items()
    }

    # Report oracle noise floor
    nl_mse_h1 = float(np.mean(noiseless_losses.get(1, [np.nan])))
    sigma_implied = float(np.sqrt(nl_mse_h1 * (dgp.nu - 2) / dgp.nu)) if dgp.nu else float(np.sqrt(nl_mse_h1))
    print(f"    Noiseless oracle MSE={nl_mse_h1:.4f}  (σ²·ν/(ν-2)={dgp.sigma**2 * dgp.nu/(dgp.nu-2):.4f})"
          if dgp.nu else f"    Noiseless oracle MSE={nl_mse_h1:.4f}  (σ²={dgp.sigma**2:.4f})")

    # ── SVD — DGP, SAS_diff, SAS_q{alt} ─────────────────────────────────────
    svd_records: list[dict] = []
    if do_svd and state_matrix is not None:
        def _svd_row(sv_dict, model_name, q_deg, seed_val):
            return {
                "sym": sym, "dgp_seed": dgp_seed,
                "model_name": model_name, "q_degree": q_deg, "seed": seed_val,
                "eff_rank": sv_dict["effective_rank"],
                "rank_80":  sv_dict["rank_80"],
                "rank_95":  sv_dict["rank_95"],
                "rank_99":  sv_dict["rank_99"],
                "singular_values": sv_dict["singular_values"],
            }

        svd_dgp  = _svd_stats(state_matrix)
        _print_svd(svd_dgp, f"DGP(q={q_degree}) {sym} s={dgp_seed}")
        svd_records.append(_svd_row(svd_dgp,  "DGP",           q_degree, dgp_seed))

        svd_diff = _compute_model_svd(synth, q_degree, p_degree, n_res, seed_diff)
        _print_svd(svd_diff, f"SAS_diff(q={q_degree}) s={seed_diff}")
        svd_records.append(_svd_row(svd_diff, "SAS_diff",       q_degree, seed_diff))

        svd_altq = _compute_model_svd(synth, alt_q, p_degree, n_res, seed_altq)
        _print_svd(svd_altq, f"SAS_q{alt_q} s={seed_altq}")
        svd_records.append(_svd_row(svd_altq, f"SAS_q{alt_q}", alt_q,    seed_altq))

    # ── Oracle lambda + MSE-vs-lambda curve ───────────────────────────────────
    oracle_lam    = None
    oos_err_dict  = None
    if do_oracle_lam:
        t2 = time.perf_counter()
        oracle_lam, oos_err_dict = _oracle_lambda(
            synth, dgp_model._basis, n_res, HORIZONS, WINDOW, REFIT_FREQ, WASHOUT,
        )
        dt_ol = time.perf_counter() - t2
        _print_lambda_summary(
            dgp_alpha        = dgp_model.alpha_log_,
            same_fixed_alpha = dgp_model.alpha_log_,
            same_cv_tracker  = same_cv_tracker,
            diff_tracker     = diff_tracker,
            oracle_alpha     = oracle_lam,
        )
        print(f"    oracle λ computed in {dt_ol:.1f}s")

    # ── Build tidy lambda records ─────────────────────────────────────────────
    lambda_records: list[dict] = []

    def _lrec(model, h, refit_idx, alpha):
        return {
            "sym": sym, "dgp_seed": dgp_seed,
            "model": model, "horizon": h,
            "refit_idx": refit_idx,
            "alpha": float(alpha),
            "log10_alpha": float(np.log10(alpha)) if alpha > 0 else np.nan,
        }

    for h in HORIZONS:
        if h in dgp_model.alpha_log_:
            lambda_records.append(_lrec("DGP",          h, -1, dgp_model.alpha_log_[h]))
        if oracle_lam and h in oracle_lam:
            lambda_records.append(_lrec("oracle",        h, -1, oracle_lam[h]))

    altq_tracker = models.get(f"SAS_q{alt_q}")
    for model_name, tracker in [
        ("SAS_same_cvλ", same_cv_tracker),
        ("SAS_diff",     diff_tracker),
        (f"SAS_q{alt_q}", altq_tracker),
    ]:
        if not isinstance(tracker, AlphaTracker):
            continue
        for refit_idx, alpha_dict in enumerate(tracker.alpha_history):
            for h, alpha in alpha_dict.items():
                lambda_records.append(_lrec(model_name, h, refit_idx, alpha))

    # ── Build tidy MSE-vs-lambda records ──────────────────────────────────────
    mse_vs_lambda: list[dict] = []
    if oos_err_dict:
        for h in HORIZONS:
            if h not in oos_err_dict:
                continue
            for ai, alpha in enumerate(_ALPHAS):
                mse_vs_lambda.append({
                    "sym": sym, "dgp_seed": dgp_seed,
                    "horizon": h,
                    "alpha_idx": ai,
                    "alpha": float(alpha),
                    "log10_alpha": float(np.log10(alpha)),
                    "oos_mse": float(oos_err_dict[h][ai]),
                })

    extras = {
        "svd_records":    svd_records,
        "lambda_records": lambda_records,
        "mse_vs_lambda":  mse_vs_lambda,
        "oracle_lam":     oracle_lam,
        "dgp_alpha":      dict(dgp_model.alpha_log_),
    }
    return sym_mse, losses, extras


# ══════════════════════════════════════════════════════════════════════════════
# Per-symbol outer run
# ══════════════════════════════════════════════════════════════════════════════

def run_symbol(
    sym:           str,
    log_vals:      np.ndarray,
    q_degree:      int,
    p_degree:      int,
    n_res:         int,
    nu:            int | None,
    n_gen:         int,
    do_svd:        bool = True,
    do_oracle_lam: bool = True,
    no_zscale:     bool = False,
) -> tuple[dict, dict, list]:
    mse_out:    dict = {}
    loss_out:   dict = {}
    extras_out: list = []

    for seed_idx, dgp_seed in enumerate(DGP_SEEDS):
        key = f"{sym}_s{dgp_seed}"
        sym_mse, sym_losses, extras = _run_one_seed(
            sym, log_vals, q_degree, p_degree, n_res, nu, n_gen,
            dgp_seed, seed_idx, do_svd=do_svd, do_oracle_lam=do_oracle_lam,
            no_zscale=no_zscale,
        )
        if sym_mse is not None:
            mse_out[key]  = sym_mse
            loss_out[key] = sym_losses
            extras_out.append(extras)

    return mse_out, loss_out, extras_out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAS-as-DGP specification experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols",          nargs="+", default=["all"])
    parser.add_argument("--q",                type=int,   default=2)
    parser.add_argument("--p",                type=int,   default=1)
    parser.add_argument("--n",                type=int,   default=N_RES)
    parser.add_argument("--nu",               type=int,   default=NU)
    parser.add_argument("--n-gen",            type=int,   default=N_GEN)
    parser.add_argument("--no-mcs",           action="store_true")
    parser.add_argument("--no-svd",           action="store_true")
    parser.add_argument("--no-oracle-lambda", action="store_true")
    parser.add_argument("--no-zscale",        action="store_true",
                        help="Skip inner z-scoring in OOS (synth is already in z-space)")
    parser.add_argument("--mcs-alpha",        type=float, default=0.10)
    parser.add_argument("--mcs-nboot",        type=int,   default=1000)
    parser.add_argument("--out-dir",          default=None)
    args = parser.parse_args()

    nu        = None if args.nu == 0 else args.nu
    n_gen     = args.n_gen
    alt_q     = 1 if args.q != 1 else 2
    model_ord = _model_order(args.q)
    slug      = f"dgp_p{args.p}q{args.q}_n{args.n}_nu{args.nu}{'_raw' if args.no_zscale else ''}"

    syms = available_symbols(CSV) if args.symbols == ["all"] else args.symbols

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "experiments" / "results_dgp_sas" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults   → {out_dir}")
    print(f"DGP       SAS(p={args.p}, q={args.q}, n={args.n}, sn={SN})")
    print(f"Seeds     {DGP_SEEDS}  (ALT_SEED={ALT_SEED})")
    print(f"Noise     σ·t({nu})  |  N_GEN={n_gen}  |  W={WINDOW}  R={REFIT_FREQ}")
    print(f"Z-score   {'disabled (raw synth space)' if args.no_zscale else 'enabled (window re-normalised)'}")
    print(f"Models    {model_ord}")
    print(f"Symbols   {len(syms)}\n")

    mse_by_key:        dict = {}
    loss_by_key:       dict = {}
    all_extras:        list = []
    all_svd_records:   list = []
    all_lambda_records: list = []
    all_mse_vs_lambda: list = []

    for sym in syms:
        print(f"\n── {sym} {'─'*48}")
        try:
            log_vals, _ = load_rv(CSV, sym)
        except Exception as e:
            print(f"  [skip] {e}")
            continue

        if len(log_vals) < MIN_REAL_T:
            print(f"  [skip] T={len(log_vals)} < {MIN_REAL_T}")
            continue

        sym_mse, sym_losses, extras_list = run_symbol(
            sym, log_vals, args.q, args.p, args.n, nu, n_gen,
            do_svd=not args.no_svd,
            do_oracle_lam=not args.no_oracle_lambda,
            no_zscale=args.no_zscale,
        )
        mse_by_key.update(sym_mse)
        loss_by_key.update(sym_losses)
        all_extras.extend(extras_list)
        for ex in extras_list:
            if not ex:
                continue
            all_svd_records   .extend(ex.get("svd_records",    []))
            all_lambda_records.extend(ex.get("lambda_records",  []))
            all_mse_vs_lambda .extend(ex.get("mse_vs_lambda",   []))

    if not mse_by_key:
        print("No results — all symbols skipped.")
        return

    # ── Stability filter ──────────────────────────────────────────────────────
    CORE_MODELS = {"HAR", "Noiseless", "SAS_same_cvλ"}
    stable_keys = [
        k for k, sym_mse in mse_by_key.items()
        if all(
            float(np.nanmean(list(v.values()))) < DIVERGE_THRESH
            for m, v in sym_mse.items() if m in CORE_MODELS
        )
    ]
    n_stable = len(stable_keys)
    sym_stable = sum(
        1 for sym in syms
        if all(f"{sym}_s{s}" in stable_keys for s in DGP_SEEDS)
    )
    print(f"\nStable {n_stable}/{len(mse_by_key)} runs  "
          f"({sym_stable} symbols fully stable × {len(DGP_SEEDS)} seeds)")

    _save_mse_csv(mse_by_key, stable_keys, out_dir, slug)
    _save_svd_csv(all_svd_records, out_dir, slug)
    _save_lambda_csv(all_lambda_records, out_dir, slug)
    _save_mse_vs_lambda_csv(all_mse_vs_lambda, out_dir, slug)

    # ── MSE table ─────────────────────────────────────────────────────────────
    stable_mse = {k: mse_by_key[k] for k in stable_keys}
    print_mse_table(
        stable_mse, HORIZONS,
        title=(f"DGP=SAS(p={args.p},q={args.q},n={args.n})  noise=σ·t({nu})  "
               f"[{n_stable} runs, {sym_stable} sym × {len(DGP_SEEDS)} seeds]"),
        model_order=model_ord,
    )

    # ── Oracle lambda summary across all runs ─────────────────────────────────
    if not args.no_oracle_lambda and all_lambda_records:
        import collections
        oracle_by_h: dict = collections.defaultdict(list)
        dgp_by_h:    dict = collections.defaultdict(list)
        for rec in all_lambda_records:
            h = rec["horizon"]
            if rec["model"] == "oracle":
                oracle_by_h[h].append(rec["log10_alpha"])
            elif rec["model"] == "DGP":
                dgp_by_h[h].append(rec["log10_alpha"])
        print("\n  ── Oracle λ summary across all runs (log10 scale) ──")
        print(f"  {'':>12}", end="")
        for h in HORIZONS:
            print(f"   h={h:2d}", end="")
        print()
        for label, d in [("DGP λ (med)", dgp_by_h), ("oracle λ (med)", oracle_by_h)]:
            print(f"  {label:>16}", end="")
            for h in HORIZONS:
                vals = d[h]
                s = f"{np.median(vals):+.2f}" if vals else "  n/a"
                print(f"  {s:>8}", end="")
            print()

    # ── MCS ───────────────────────────────────────────────────────────────────
    if not args.no_mcs and stable_keys:
        stable_eval = _sq_errors_to_eval_df(
            {k: loss_by_key[k] for k in stable_keys}
        )
        mcs_df = print_mcs_frequency(
            stable_eval, HORIZONS,
            alpha         = args.mcs_alpha,
            n_boot        = args.mcs_nboot,
            seed          = DGP_SEEDS[0],
            title         = (f"MCS — DGP=SAS(p={args.p},q={args.q}) "
                             f"[{len(stable_keys)} runs]"),
            mse_by_symbol = stable_mse,
            model_order   = model_ord,
        )
        _save_mcs_csv(mcs_df, out_dir, slug)

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
