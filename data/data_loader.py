"""
data_loader.py — Load and preprocess daily realised volatility data.

Expected CSV format (Oxford-Man Realised Library style):
  index column : date (datetime)
  columns      : Symbol, open_time, close_price, rv5_ss, bv, rk_parzen, …

Preprocessing pipeline
----------------------
1. Filter rows to a single symbol.
2. Select the target RV column.
3. Drop NaN / non-positive values.
4. Log-transform: x_t = log(RV_t)   (RV is log-normally distributed).
5. Z-scoring is done in the evaluation loop (statistics frozen at each refit).

The data_loader only returns (log_values, dates) — no z-scoring here so that
the rolling OOS loop can recompute normalisation statistics at every refit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ── public helpers ────────────────────────────────────────────────────────────

def load_rv(
    csv_path: str | Path,
    symbol: str,
    target: str = "rv5",
    log_transform: bool = True,
    clip_lower: float = 1e-10,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load a univariate RV series for one symbol.

    Parameters
    ----------
    csv_path      : path to the CSV file (root-level by convention).
    symbol        : e.g. ".AEX", ".FTSE", ".SPX".
    target        : RV column to forecast, e.g. "rv5", "rk_parzen", "bv".
    log_transform : apply log transform (recommended — RV is right-skewed).
    clip_lower    : minimum value before log (avoids log(0)).

    Returns
    -------
    values : (T,) float64 array, log-transformed if log_transform=True.
    dates  : DatetimeIndex of length T.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[df["Symbol"] == symbol].sort_index()

    if df.empty:
        available = sorted(df["Symbol"].unique()) if "Symbol" in df.columns else []
        raise ValueError(
            f"Symbol '{symbol}' not found. Available: {available}"
        )
    if target not in df.columns:
        raise ValueError(
            f"Column '{target}' not found. Available RV columns: "
            f"{[c for c in df.columns if c not in ('Symbol', 'open_time', 'close_time', 'open_price', 'close_price', 'nobs')]}"
        )

    series = df[target].dropna()
    series = series[series > 0]            # remove non-positive entries

    values = series.values.astype(np.float64)
    if log_transform:
        values = np.log(np.maximum(values, clip_lower))

    # Always return a tz-naive DatetimeIndex regardless of CSV index format.
    # utc=True handles mixed-timezone rows; then tz_convert strips the tz info.
    idx = pd.to_datetime(series.index, utc=True).tz_localize(None)
    return values, pd.DatetimeIndex(idx)


def available_symbols(csv_path: str | Path) -> list[str]:
    """Return sorted list of symbols present in the CSV."""
    df = pd.read_csv(csv_path, usecols=["Symbol"])
    return sorted(df["Symbol"].unique().tolist())


def available_targets(csv_path: str | Path) -> list[str]:
    """Return RV-like column names (heuristic: exclude metadata columns)."""
    df = pd.read_csv(csv_path, nrows=1)
    _meta = {"Symbol", "open_time", "close_time", "open_price",
              "close_price", "nobs", "open_to_close"}
    return [c for c in df.columns if c not in _meta and not c.startswith("Unnamed")]


# ── z-score utilities (used by the evaluation loop, not by models) ─────────

def fit_scaler(train: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) computed on the training window."""
    mu    = float(np.mean(train))
    sigma = float(np.std(train))
    return mu, max(sigma, 1e-8)


def apply_scaler(values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Standardise with pre-computed (mu, sigma)."""
    return (values - mu) / sigma


def inverse_scaler(
    z: np.ndarray | float,
    mu: float,
    sigma: float,
    log_space: bool = True,
) -> np.ndarray | float:
    """
    Inverse-transform from standardised log-space back to RV scale.

    If log_space=True : output = exp(z * sigma + mu)
    If log_space=False: output = z * sigma + mu
    """
    x = np.asarray(z) * sigma + mu
    return np.exp(x) if log_space else x
