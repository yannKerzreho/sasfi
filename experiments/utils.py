"""
experiments/utils.py — shared helpers for all experiment scripts.

Rolling OOS protocol
--------------------
  fit_scaler / apply_scaler freeze z-score stats at each refit.
  Refit happens every `refit_freq` steps.
  Predictions are recorded in z-space; losses are squared errors in z-space.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import load_rv, fit_scaler, apply_scaler

HORIZONS   = [1, 5, 22]
WINDOW     = 500
REFIT_FREQ = 252


def quick_oos(
    log_values: np.ndarray,
    dates:      "pd.DatetimeIndex",
    models:     dict,
    horizons:   list[int] = HORIZONS,
    window:     int        = WINDOW,
    refit_freq: int        = REFIT_FREQ,
    max_steps:  int | None = None,
) -> dict[str, dict[int, list[float]]]:
    """
    Rolling OOS evaluation.

    Returns
    -------
    losses : {model_name: {horizon: [sq_errors]}}
    """
    T     = len(log_values)
    H_max = max(horizons)
    mu, sigma = 0.0, 1.0
    steps_since_refit = {n: refit_freq for n in models}
    losses = {n: {h: [] for h in horizons} for n in models}
    limit  = (window + max_steps) if max_steps else (T - H_max)

    for t in range(window, min(limit, T - H_max)):
        train_raw = log_values[t - window: t]

        for name, model in models.items():
            if steps_since_refit[name] >= refit_freq:
                mu, sigma = fit_scaler(train_raw)
                train_z   = apply_scaler(train_raw, mu, sigma)
                try:
                    model.fit(train_z, horizons)
                    steps_since_refit[name] = 0
                except Exception as e:
                    print(f"  [fit {name}] {e}")

        z_t = apply_scaler(float(log_values[t]), mu, sigma)

        for name, model in models.items():
            for h in horizons:
                if t + h >= T:
                    continue
                z_tgt = apply_scaler(float(log_values[t + h]), mu, sigma)
                try:
                    y_hat = float(model.predict(h))
                    losses[name][h].append((y_hat - z_tgt) ** 2)
                except Exception as e:
                    print(f"  [pred {name} h={h}] {e}")

        for name, model in models.items():
            try:
                model.update(z_t)
                steps_since_refit[name] += 1
            except Exception as e:
                print(f"  [upd {name}] {e}")

    return losses


def mean_losses(losses: dict) -> dict[str, dict[int, float]]:
    """Average sq errors: {model: {h: mse}}."""
    return {
        n: {h: (float(np.mean(v)) if v else np.nan) for h, v in hd.items()}
        for n, hd in losses.items()
    }


def print_table(
    results: dict[str, dict[int, float]],
    horizons: list[int] = HORIZONS,
    title: str = "",
    sort_by: str = "avg",
):
    """Print a formatted MSE table sorted by average MSE."""
    rows = []
    for name, mses in results.items():
        row = {"config": name}
        for h in horizons:
            row[f"h{h}"] = mses.get(h, np.nan)
        row["avg"] = float(np.nanmean([mses.get(h, np.nan) for h in horizons]))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(sort_by)
    w  = max(len(r["config"]) for r in rows) + 2

    if title:
        print(f"\n{'─' * (w + 30)}")
        print(f"  {title}")
    print(f"{'─' * (w + 30)}")
    header = f"{'Config':<{w}}" + "".join(f"{'h='+str(h):>9}" for h in horizons) + f"{'avg':>9}"
    print(header)
    print("─" * (w + 30))
    for _, r in df.iterrows():
        line = f"{r['config']:<{w}}"
        for h in horizons:
            v = r.get(f"h{h}", np.nan)
            line += f"  {v:7.4f}" if np.isfinite(v) else f"  {'nan':>7}"
        line += f"  {r['avg']:7.4f}"
        print(line)
    print(f"{'─' * (w + 30)}")
    return df
