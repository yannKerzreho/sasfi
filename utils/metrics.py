"""
metrics.py — Evaluation metrics for volatility forecasting.

MSE      : Mean Squared Error     — symmetric, penalises large errors.
MAE      : Mean Absolute Error    — symmetric, robust.
QLIKE    : Quasi-Likelihood loss  — asymmetric, standard benchmark for RV.
               QLIKE(ŷ, y) = y/ŷ − log(y/ŷ) − 1
MDA      : Mean Directional Accuracy (Hit Rate) — fraction of steps where
           the model correctly predicts the sign of the change.
           > 0.5 means the model has directional skill.
"""

from __future__ import annotations
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """
    QLIKE loss in original (positive) RV scale.
    Pass inverse_scaler output — both arrays must be positive.
    """
    yt = np.asarray(y_true,  dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = (yt > eps) & (yp > eps)
    if not mask.any():
        return np.nan
    r = yt[mask] / yp[mask]
    return float(np.mean(r - np.log(r) - 1.0))


def mda(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray | None = None,
) -> float:
    """
    Mean Directional Accuracy (Hit Rate).

    Measures the fraction of steps where the predicted direction of change
    matches the actual direction:
        MDA = mean( sign(y_true − y_prev) == sign(y_pred − y_prev) )

    If y_prev is None, uses y_true[:-1] as the previous value (shifts by 1).
    Returns a value in [0, 1]; 0.5 = no directional skill.

    Parameters
    ----------
    y_true : actual values at time t+h.
    y_pred : predicted values at time t+h.
    y_prev : values at time t (the "current" observation before prediction).
             If None, inferred as y_true[:-1] (only works for h=1).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if y_prev is None:
        yt, yp = yt[1:], yp[1:]
        prev   = yt[:-1]
        yt, yp = yt[1:], yp[1:]
        # Simpler: just shift
        prev = np.asarray(y_true, dtype=np.float64)[:-1]
        yt   = np.asarray(y_true, dtype=np.float64)[1:]
        yp   = np.asarray(y_pred, dtype=np.float64)[1:]
    else:
        prev = np.asarray(y_prev, dtype=np.float64)
    hits = np.sign(yt - prev) == np.sign(yp - prev)
    return float(hits.mean())


def compute_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    metric_names: tuple[str, ...] = ("mse", "mae"),
    y_prev:       np.ndarray | None = None,
) -> dict[str, float]:
    """Compute multiple metrics. Returns {metric_name: value}."""
    fns = {"mse": mse, "mae": mae, "qlike": qlike}
    out = {}
    for name in metric_names:
        if name == "mda":
            out["mda"] = mda(y_true, y_pred, y_prev)
        elif name in fns:
            out[name] = fns[name](y_true, y_pred)
    return out


def summary_table(
    df_eval: "pd.DataFrame",
    metrics: list[str] = ("mse", "mae", "mda"),
    model_col:   str = "config",
    horizon_col: str = "horizon",
) -> "pd.DataFrame":
    """
    Build a multi-level (metric × horizon) summary table for quick inspection.

    Returns a DataFrame indexed by model, with a MultiIndex column
    (metric, horizon).
    """
    import pandas as pd

    rows = {}
    for (name, h), grp in df_eval.groupby([model_col, horizon_col]):
        yt = grp["y_true"].values
        yp = grp["y_pred"].values
        for m in metrics:
            if m == "mda":
                val = mda(yt, yp)
            elif m == "mse":
                val = mse(yt, yp)
            elif m == "mae":
                val = mae(yt, yp)
            else:
                val = np.nan
            rows.setdefault(name, {})[(m, h)] = val

    df = pd.DataFrame(rows).T
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["metric", "horizon"])
    return df.sort_index()
