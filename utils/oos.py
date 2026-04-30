"""
utils/oos.py — Canonical rolling OOS evaluation loop.

run_oos : rolling-window OOS, returns DataFrame with sq_err and QLIKE.
          Models own all preprocessing; raw values are passed unchanged.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _fmt(d) -> str:
    """Return YYYY-MM-DD string from any date-like object."""
    try:
        return d.date().isoformat()
    except AttributeError:
        return str(d)[:10]


def run_oos(
    values:     np.ndarray,
    dates:      "pd.DatetimeIndex",
    models:     dict,
    horizons:   list[int],
    window:     int            = 2000,
    refit_freq: int            = 20,
    alphas_out: "dict | None" = None,
    verbose:    bool           = True,
    log_mode:   bool           = False,
) -> pd.DataFrame:
    """
    Rolling OOS with periodic refit.

    Models own all preprocessing (z-scoring, log-transform, etc.).
    This loop passes raw ``values`` unchanged and records predictions in the
    original scale together with sq_err and QLIKE.

    Steps per t
    -----------
    1. Refit all models on window [t-window, t) if refit_freq elapsed.
    2. model.update(x_t)  — advance state so s_last = s_t.
    3. model.predict(h)   — forecast ŷ_{t+h} in original scale.

    Parameters
    ----------
    values, dates  : time-series data (raw or log, depending on caller).
    models         : {name: forecaster} — modified in-place.
    horizons       : forecast horizons.
    window         : rolling training-window size.
    refit_freq     : refit every N OOS steps.
    alphas_out     : optional dict populated with chosen ridge alpha per refit.
                     Scale factor input_scale² applied for SAS models.
    verbose        : print date range and model list.
    log_mode       : if True, values are in log-space; QLIKE uses the formula
                       QLIKE_log = exp(ŷ−y) + (ŷ−y) − 1
                     (equals level QLIKE applied to exp(·), valid for any
                     finite y and ŷ). If False, level QLIKE y/ŷ−log(y/ŷ)−1
                     is used (requires y_true > 0).

    Returns
    -------
    DataFrame[config, horizon, test_date, y_pred, y_true, sq_err, abs_err, qlike]
    """
    T     = len(values)
    H_max = max(horizons)
    steps_since_refit: dict[str, int] = {n: refit_freq for n in models}
    records: list[dict] = []

    if verbose:
        print(f"  OOS: {_fmt(dates[window])} → {_fmt(dates[T - H_max - 1])}")
        print(f"       window={window}, refit_freq={refit_freq}, "
              f"models: {list(models)}")

    if alphas_out is not None:
        for name in models:
            alphas_out.setdefault(name, {h: [] for h in horizons})

    for t in range(window, T - H_max):
        train = values[t - window: t]

        for name, model in models.items():
            if steps_since_refit[name] >= refit_freq:
                try:
                    model.fit(train, horizons)
                    steps_since_refit[name] = 0
                    if alphas_out is not None and hasattr(model, "alpha_log_"):
                        sf = float(getattr(model, "input_scale", 1.0)) ** 2
                        for h in horizons:
                            if h in model.alpha_log_:
                                alphas_out[name][h].append(
                                    float(model.alpha_log_[h]) * sf
                                )
                except Exception as e:
                    print(f"    [fit {name} @ {_fmt(dates[t])}] {e}")

        x_t = float(values[t])

        # Update FIRST so s_last = s_t.  After fit(), s_last = s_{t-1}.
        for name, model in models.items():
            try:
                model.update(x_t)
                steps_since_refit[name] += 1
            except Exception as e:
                print(f"    [upd {name} @ {_fmt(dates[t])}] {e}")

        for name, model in models.items():
            for h in horizons:
                if t + h >= T:
                    continue
                try:
                    y_hat  = float(model.predict(h))
                    y_true = float(values[t + h])

                    if log_mode:
                        # y and ŷ in log-space: exp(ŷ)/exp(y) − (y−ŷ) − 1
                        #                     = exp(ŷ−y) + (ŷ−y) − 1
                        if np.isfinite(y_hat) and np.isfinite(y_true):
                            delta = float(y_hat - y_true)
                            q = float(np.exp(delta) + delta - 1)
                        else:
                            q = np.nan
                    else:
                        # Level-space QLIKE: requires y_true > 0
                        if y_true > 0 and np.isfinite(y_hat):
                            yp    = max(y_hat, 1e-8)
                            ratio = min(y_true / yp, 1e6)
                            q     = float(ratio - np.log(ratio) - 1)
                        else:
                            q = np.nan

                    records.append(dict(
                        config    = name,
                        horizon   = h,
                        test_date = dates[t],
                        y_pred    = y_hat,
                        y_true    = y_true,
                        sq_err    = (y_hat - y_true) ** 2,
                        abs_err   = abs(y_hat - y_true),
                        qlike     = q,
                    ))
                except Exception as e:
                    print(f"    [pred {name} h={h} @ {_fmt(dates[t])}] {e}")

    return pd.DataFrame(records)
