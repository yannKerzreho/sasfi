"""
ridge.py — Shared ridge regression utilities for all forecasters.

Functions
---------
ridge_cv_select      : rolling-window CV → best alpha.
ridge_fit            : fit with a fixed alpha.
ridge_cv             : select + fit in one call (standard ridge).
har_ridge_cv_and_fit : HAR column-normalised ridge → (weights, alpha).
ridge_cv_grouped     : block-diagonal ridge CV (one alpha per feature group).
ridge_fit_grouped    : block-diagonal ridge fit.

Optimisations
-------------
• Gram matrix (X.T @ X) is precomputed *once per fold*, outside the alpha
  loop.  Previously it was rebuilt O(n_folds × n_alphas) times.

• For uniform penalty (no penalty_mask): eigendecomposition of the fold
  Gram is computed once per fold.  Each alpha then costs O(val_size·n)
  for the residual — vs O(n³) for a fresh solve.

  Example (n=200, 37 alphas, 3 folds, val_size≈300):
    Old: 37 × 3 × O(n³)  ≈ 888 M ops
    New: 3 × O(n³) + 37 × 3 × O(val_size·n) ≈ 24 M + 6.6 M ≈ 30 M ops

• For non-uniform penalty (HAR intercept exemption): Gram precomputed per
  fold; diagonal modified O(n) per alpha before each solve.
"""

from __future__ import annotations

import numpy as np
from itertools import product as _cart_product


# ── alpha grids ───────────────────────────────────────────────────────────────

# 37 log-spaced values from 10⁻⁴ to 10⁵ (spacing 0.25 in log10).
ALPHAS: list[float] = [10 ** x for x in np.arange(-4, 5.01, 0.25)]

# Coarser grid for grouped CV: every 6th value → 7 candidates.
# 7² = 49 combos (2-group), 7³ = 343 (3-group) — fast, ~1.5 log10 spacing.
ALPHAS_GROUPED: list[float] = ALPHAS[::6]


# ══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════════════════════

def ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    penalty_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Ridge regression: (X.T X + α R)⁻¹ X.T y.

    penalty_mask : (n,) per-feature multipliers for α.
                   None → standard ridge (R = I).
                   [0,1,1,1] → skip intercept column (HAR).
    """
    n = X.shape[1]
    R = (np.diag(penalty_mask.astype(float))
         if penalty_mask is not None else np.eye(n))
    return np.linalg.solve(X.T @ X + alpha * R, X.T @ y)


def ridge_cv_select(
    X:            np.ndarray,
    y:            np.ndarray,
    n_folds:      int = 5,
    alphas:       list | None = None,
    penalty_mask: np.ndarray | None = None,
) -> float:
    """
    Rolling-window CV to select the best ridge alpha.

    For uniform penalty (penalty_mask=None) uses eigendecomposition of the
    fold Gram so each alpha costs O(val_size·n) instead of O(n³).
    For non-uniform penalty precomputes the Gram per fold and modifies only
    the diagonal per alpha (O(n) overhead instead of a full build).

    Returns
    -------
    best_alpha : float
    """
    if alphas is None:
        alphas = ALPHAS
    T, n     = X.shape
    val_size = max(12, T // (n_folds + 1))
    diag_ix  = np.diag_indices(n)

    uniform = penalty_mask is None
    if not uniform:
        pen_diag = np.asarray(penalty_mask, dtype=float)

    # ── precompute per-fold data ──────────────────────────────────────────
    fold_cache: list = []
    for fold in range(n_folds):
        cut = T - (n_folds - fold) * val_size
        if cut < n + 5:
            continue
        XTX  = X[:cut].T @ X[:cut]   # (n, n) — shared across all alphas
        XTy  = X[:cut].T @ y[:cut]   # (n,)
        X_val = X[cut: cut + val_size]
        y_val = y[cut: cut + val_size]

        if uniform:
            # Eigendecompose once per fold; O(n³) amortised over all alphas.
            lam, V = np.linalg.eigh(XTX)
            lam    = np.maximum(lam, 0.0)   # numerical safety (X.T X is PSD)
            VTXTy  = V.T @ XTy              # (n,)
            XV     = X_val @ V              # (val_size, n)
            fold_cache.append((lam, VTXTy, XV, y_val))
        else:
            xtx_diag = XTX[diag_ix].copy()
            fold_cache.append((XTX, XTy, xtx_diag, X_val, y_val))

    if not fold_cache:
        return alphas[0]

    # ── sweep alphas ──────────────────────────────────────────────────────
    best, best_mse = alphas[0], np.inf

    if uniform:
        for alpha in alphas:
            total = 0.0
            for lam, VTXTy, XV, y_val in fold_cache:
                # w = V @ (VTXTy / (lam + α))  →  X_val @ w = XV @ coef
                coef  = VTXTy / (lam + alpha)   # O(n)
                err   = XV @ coef - y_val        # O(val_size · n)
                total += float(np.mean(err ** 2))
            mse = total / len(fold_cache)
            if mse < best_mse:
                best_mse, best = mse, alpha
    else:
        for alpha in alphas:
            total = 0.0
            for XTX, XTy, xtx_diag, X_val, y_val in fold_cache:
                A          = XTX.copy()
                A[diag_ix] = xtx_diag + alpha * pen_diag   # O(n) diagonal update
                w          = np.linalg.solve(A, XTy)
                err        = X_val @ w - y_val
                total     += float(np.mean(err ** 2))
            mse = total / len(fold_cache)
            if mse < best_mse:
                best_mse, best = mse, alpha

    return best


def ridge_cv(
    X:       np.ndarray,
    y:       np.ndarray,
    n_folds: int = 5,
    alphas:  list | None = None,
) -> np.ndarray:
    """Select alpha via CV then return fitted weights (standard ridge, no mask)."""
    alpha = ridge_cv_select(X, y, n_folds, alphas)
    return ridge_fit(X, y, alpha)


# ══════════════════════════════════════════════════════════════════════════════
# HAR-specific: column-normalised ridge
# ══════════════════════════════════════════════════════════════════════════════

def har_ridge_cv_and_fit(
    X:       np.ndarray,
    y:       np.ndarray,
    n_folds: int = 5,
    alphas:  list | None = None,
) -> tuple[np.ndarray, float]:
    """
    Column-normalised ridge for HAR features [1, RV_d, RV_w, RV_m].

    Steps
    -----
    1. Scale each non-intercept column to unit std so all three features
       enter the penalty on equal footing.
    2. CV with penalty_mask=[0,1,1,1] (intercept not regularised).
    3. Final fit on full training set.
    4. Back-transform weights to original (unscaled) feature space.

    Returns
    -------
    w     : (n,) weight vector in the original feature space.
    alpha : chosen regularisation parameter.
    """
    if alphas is None:
        alphas = ALPHAS
    T, n = X.shape

    col_stds  = np.maximum(X[:, 1:].std(axis=0), 1e-8)   # (n-1,)
    col_scale = np.concatenate([[1.0], col_stds])          # (n,)
    X_sc      = X / col_scale                              # (T, n) scaled

    pmask = np.array([0.0] + [1.0] * (n - 1))

    alpha  = ridge_cv_select(X_sc, y, n_folds, alphas, penalty_mask=pmask)
    w_sc   = ridge_fit(X_sc, y, alpha, penalty_mask=pmask)
    w      = w_sc / col_scale        # back to original-feature weights

    return w, alpha


# ══════════════════════════════════════════════════════════════════════════════
# Grouped ridge (block-diagonal penalty, one alpha per feature group)
# ══════════════════════════════════════════════════════════════════════════════

def ridge_cv_grouped(
    S:            np.ndarray,
    Y:            np.ndarray,
    group_sizes:  list[int],
    n_folds:      int,
    alpha_grid_1d: list[float],
) -> tuple:
    """
    Block-diagonal ridge CV: a separate alpha for each feature group.

    The penalty matrix is Λ = diag(α₀·I_{g₀}, α₁·I_{g₁}, …).
    The Gram S[:cut].T @ S[:cut] is precomputed once per fold; each
    alpha-combo only modifies the diagonal (O(n)) before solving.

    Parameters
    ----------
    S             : (T, n) feature matrix, columns ordered by group.
    Y             : (T,) targets.
    group_sizes   : ints summing to n, one per group.
    n_folds       : rolling CV folds.
    alpha_grid_1d : 1-D alpha candidates searched independently per group.

    Returns
    -------
    best_alphas : tuple[float, …], one per group.
    """
    T, n     = S.shape
    val_size = max(12, T // (n_folds + 1))
    combos   = list(_cart_product(alpha_grid_1d, repeat=len(group_sizes)))
    diag_ix  = np.diag_indices(n)

    # Pre-build λ vectors once (avoid re-allocating inside the inner loop)
    lam_vecs = [
        np.concatenate([a * np.ones(g) for a, g in zip(combo, group_sizes)])
        for combo in combos
    ]

    sum_mse = np.zeros(len(combos))
    cnt_mse = np.zeros(len(combos), dtype=int)

    for fold in range(n_folds):
        cut = T - (n_folds - fold) * val_size
        if cut < max(5, n // 10):
            continue
        STS      = S[:cut].T @ S[:cut]     # (n, n) — shared across combos
        STY      = S[:cut].T @ Y[:cut]     # (n,)
        val_S    = S[cut: cut + val_size]
        val_Y    = Y[cut: cut + val_size]
        sts_diag = STS[diag_ix].copy()

        for ci, lam in enumerate(lam_vecs):
            A            = STS.copy()
            A[diag_ix]   = sts_diag + lam        # O(n) diagonal update
            w            = np.linalg.solve(A, STY)
            err          = val_S @ w - val_Y
            sum_mse[ci] += float(np.mean(err ** 2))
            cnt_mse[ci] += 1

    valid = cnt_mse > 0
    if not valid.any():
        return combos[0]
    avg = np.where(valid, sum_mse / np.maximum(cnt_mse, 1), np.inf)
    return combos[int(np.argmin(avg))]


def ridge_fit_grouped(
    S:           np.ndarray,
    Y:           np.ndarray,
    group_sizes: list[int],
    alphas:      tuple,
) -> np.ndarray:
    """Block-diagonal ridge: (S.T S + Λ)⁻¹ S.T Y."""
    n   = S.shape[1]
    lam = np.concatenate([a * np.ones(g) for a, g in zip(alphas, group_sizes)])
    A   = S.T @ S
    A[np.diag_indices(n)] += lam
    return np.linalg.solve(A, S.T @ Y)
