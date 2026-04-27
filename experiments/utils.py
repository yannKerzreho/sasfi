"""
experiments/utils.py — shared helpers for all experiment scripts.

Rolling OOS protocol
--------------------
  fit_scaler / apply_scaler freeze z-score stats at each refit.
  Refit happens every `refit_freq` steps.
  Predictions are recorded in z-space; losses are squared errors in z-space.

Display helpers
---------------
  print_mse_table        : mean ± std MSE table across multiple symbols.
  print_mcs_frequency    : MCS run per symbol, frequency-of-survival table.
"""

from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import load_rv, fit_scaler, apply_scaler

HORIZONS   = [1, 5, 22]
WINDOW     = 2000
REFIT_FREQ = 20


# ── rolling OOS ──────────────────────────────────────────────────────────────

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

        # Update FIRST so s_last = s_t.  After fit(), s_last = s_{t-1}.
        # The convention is z_{t+1} = W·s_t + ε, so we must ingest z_t
        # before predicting z_{t+h}.
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
                z_tgt = apply_scaler(float(log_values[t + h]), mu, sigma)
                try:
                    y_hat = float(model.predict(h))
                    losses[name][h].append((y_hat - z_tgt) ** 2)
                except Exception as e:
                    print(f"  [pred {name} h={h}] {e}")

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


# ── multi-symbol MSE table ────────────────────────────────────────────────────

def print_mse_table(
    mse_by_symbol: "dict[str, dict[str, dict[int, float]]]",
    horizons:      list[int],
    title:         str = "",
    model_order:   list[str] | None = None,
) -> pd.DataFrame:
    """
    Print a mean ± std MSE table across symbols.

    Parameters
    ----------
    mse_by_symbol : {symbol: {model: {h: mse}}}
    horizons      : list of horizons to display.
    title         : optional header string.
    model_order   : if given, fix the display order of models (others appended).

    Returns
    -------
    DataFrame with columns [config, h{k}_mean, h{k}_std, avg_mean].
    """
    # Collect models preserving insertion order
    all_models: list[str] = []
    seen: set[str] = set()
    if model_order:
        for m in model_order:
            all_models.append(m)
            seen.add(m)
    for sym_mse in mse_by_symbol.values():
        for cfg in sym_mse:
            if cfg not in seen:
                all_models.append(cfg)
                seen.add(cfg)

    # {model: {h: [mse across symbols]}}
    agg: dict[str, dict[int, list[float]]] = {
        m: {h: [] for h in horizons} for m in all_models
    }
    for sym_mse in mse_by_symbol.values():
        for m, h_dict in sym_mse.items():
            if m in agg:
                for h in horizons:
                    if h in h_dict and np.isfinite(h_dict[h]):
                        agg[m][h].append(h_dict[h])

    rows = []
    for m in all_models:
        row: dict = {"config": m}
        avgs = []
        for h in horizons:
            vals = agg[m][h]
            mu   = float(np.mean(vals)) if vals else np.nan
            sd   = float(np.std(vals, ddof=min(1, len(vals) - 1))) if len(vals) > 1 else np.nan
            row[f"h{h}_mean"] = mu
            row[f"h{h}_std"]  = sd
            if np.isfinite(mu):
                avgs.append(mu)
        row["avg_mean"] = float(np.mean(avgs)) if avgs else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("avg_mean")
    n_sym = len(mse_by_symbol)
    N_str = f"N={n_sym} symbol{'s' if n_sym > 1 else ''}"

    w   = max(len(r["config"]) for r in rows) + 2
    col = len(horizons) * 18 + 10
    sep = "─" * (w + col)

    if title:
        print(f"\n{sep}")
        print(f"  {title}  ({N_str})")
    print(sep)
    header = f"{'Model':<{w}}" + "".join(f"{'h='+str(h):>18}" for h in horizons) + f"{'avg':>12}"
    print(header)
    print(sep)
    for _, r in df.iterrows():
        line = f"{r['config']:<{w}}"
        for h in horizons:
            mu, sd = r[f"h{h}_mean"], r[f"h{h}_std"]
            if not np.isfinite(mu):
                cell = "—"
            elif not np.isfinite(sd):
                cell = f"{mu:7.4f}          "
            else:
                cell = f"{mu:7.4f}±{sd:6.4f}"
            line += f"  {cell:>16}"
        avg = r["avg_mean"]
        line += f"  {avg:8.4f}" if np.isfinite(avg) else f"  {'—':>8}"
        print(line)
    print(sep)
    return df


# ── MCS frequency table ───────────────────────────────────────────────────────

def _run_mcs_one_symbol(
    df_eval:  pd.DataFrame,
    horizons: list[int],
    alpha:    float,
    n_boot:   int,
    seed:     int,
) -> dict[int, list[str]]:
    """
    Run per-horizon MCS for one symbol's eval DataFrame.

    df_eval must have columns: config, horizon, test_date, sq_err.

    Returns {horizon: [models_in_mcs]}.
    """
    from utils.mcs_utils import ModelConfidenceSet

    result: dict[int, list[str]] = {}
    for h in horizons:
        sub = df_eval[df_eval["horizon"] == h]
        if sub.empty:
            result[h] = []
            continue

        pivot = (
            sub.pivot_table(index="test_date", columns="config",
                            values="sq_err", aggfunc="mean")
               .dropna(axis=1, how="all")
               .dropna(axis=0, how="any")
        )
        if pivot.shape[1] < 2 or pivot.shape[0] < 10:
            result[h] = list(pivot.columns)   # too few obs → all survive
            continue

        # Row-normalise so high-vol regimes don't dominate bootstrap variance
        row_mean = pivot.mean(axis=1).clip(lower=1e-12)
        pivot_n  = pivot.div(row_mean, axis=0)

        T = pivot_n.shape[0]
        w = max(1, int(T ** (1 / 3)))
        # Pass ndarray + explicit names array so mcs_utils avoids Arrow-backed
        # pandas indexing (which chokes on the 2-D index returned by iterate()).
        mcs = ModelConfidenceSet(
            data      = pivot_n.values,
            alpha     = alpha,
            B         = n_boot,
            w         = w,
            algorithm = "R",
            seed      = seed,
            names     = np.array(list(pivot_n.columns)),
        ).run()
        result[h] = mcs.included
    return result


def _sq_errors_to_eval_df(
    losses_by_sym: "dict[str, dict[str, dict[int, list[float]]]]",
) -> "dict[str, pd.DataFrame]":
    """
    Convert {symbol: {model: {h: [sq_errors]}}} to
            {symbol: DataFrame[config, horizon, test_date, sq_err]}.

    test_date is a synthetic integer index (0, 1, 2, …) — sufficient for MCS.
    """
    result = {}
    for sym, model_dict in losses_by_sym.items():
        records = []
        for model, h_dict in model_dict.items():
            for h, errs in h_dict.items():
                for i, e in enumerate(errs):
                    records.append({"config": model, "horizon": h,
                                    "test_date": i, "sq_err": e})
        result[sym] = pd.DataFrame(records)
    return result


def print_mcs_frequency(
    evals_by_symbol: "dict[str, pd.DataFrame]",
    horizons:        list[int],
    alpha:           float = 0.10,
    n_boot:          int   = 1000,
    seed:            int   = 42,
    title:           str   = "",
    mse_by_symbol:   "dict[str, dict[str, dict[int, float]]] | None" = None,
    model_order:     "list[str] | None" = None,
) -> pd.DataFrame:
    """
    Run the MCS independently for each symbol × horizon, then print a
    frequency table combining MCS survival rate and mean MSE.

    Parameters
    ----------
    evals_by_symbol : {symbol: DataFrame[config, horizon, test_date, sq_err]}
    mse_by_symbol   : optional {symbol: {model: {h: mse}}} — if provided,
                      adds a mean MSE column and flags high-variance models
                      (CV > 0.5 on any horizon) with ⚠.

    Note on apparent inconsistencies
    ---------------------------------
    MCS frequency and mean MSE can diverge when a model has high cross-symbol
    variance: it may survive the MCS on most symbols (where it is competitive)
    while its mean MSE is inflated by a few symbols where it degrades.
    The ⚠ flag marks models where std/mean > 0.5 on at least one horizon —
    read their MCS frequency with caution.

    Returns
    -------
    DataFrame with columns [config, h{k}_count, h{k}_frac, total_count,
                             avg_mse (if mse_by_symbol provided)].
    """
    n_sym   = len(evals_by_symbol)
    symbols = list(evals_by_symbol.keys())
    print(f"\n  Running MCS (α={alpha}, B={n_boot}) for {n_sym} symbol(s) …")

    # {model: {h: count_in_mcs}}
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    valid:  dict[int, int]            = defaultdict(int)

    for sym in symbols:
        df = evals_by_symbol[sym]
        if df.empty or "horizon" not in df.columns:
            continue
        mcs_by_h = _run_mcs_one_symbol(df, horizons, alpha, n_boot, seed)
        for h, mcs_set in mcs_by_h.items():
            if mcs_set:
                valid[h] += 1
                for m in mcs_set:
                    counts[m][h] += 1

    if not counts:
        print("  No MCS results (insufficient data).")
        return pd.DataFrame()

    # ── build per-model MSE stats if provided ────────────────────────────
    mse_stats: dict[str, tuple[float, bool]] = {}  # model → (avg_mse, high_var)
    if mse_by_symbol:
        agg: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for sym_mse in mse_by_symbol.values():
            for m, h_dict in sym_mse.items():
                for h in horizons:
                    if h in h_dict and np.isfinite(h_dict[h]):
                        agg[m][h].append(h_dict[h])
        for m, h_dict in agg.items():
            avgs, high_var = [], False
            for h in horizons:
                vals = h_dict[h]
                if vals:
                    mu = float(np.mean(vals))
                    sd = float(np.std(vals, ddof=min(1, len(vals)-1))) if len(vals) > 1 else 0.0
                    avgs.append(mu)
                    if mu > 0 and sd / mu > 0.5:   # CV > 50% flags instability
                        high_var = True
            mse_stats[m] = (float(np.mean(avgs)) if avgs else np.nan, high_var)

    if model_order is not None:
        # Preserve requested order; append any extra models alphabetically
        known = [m for m in model_order if m in counts]
        extra = sorted(m for m in counts if m not in model_order)
        all_models = known + extra
    else:
        all_models = sorted(counts.keys())
    rows = []
    for m in all_models:
        row: dict = {"config": m}
        total = 0
        for h in horizons:
            c  = counts[m][h]
            nv = valid[h]
            row[f"h{h}_count"] = c
            row[f"h{h}_frac"]  = c / nv if nv > 0 else np.nan
            total += c
        row["total_count"] = total
        if m in mse_stats:
            row["avg_mse"],  row["high_var"] = mse_stats[m]
        else:
            row["avg_mse"],  row["high_var"] = np.nan, False
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if model_order is None:
        df_out = df_out.sort_values("total_count", ascending=False)

    # ── pretty print ─────────────────────────────────────────────────────
    show_mse = mse_by_symbol is not None
    w        = max(len(r["config"]) for r in rows) + 2
    extra    = 12 if show_mse else 0
    sep      = "─" * (w + len(horizons) * 14 + 8 + extra)
    hdr      = title or "MCS frequency table"

    print(f"\n{sep}")
    print(f"  {hdr}  (α={alpha}, N={n_sym} symbols, row-normalised losses)")
    print(f"  freq = #symbols model survived MCS best set  |  ⚠ = CV>0.5 on ≥1 horizon")
    print(sep)
    header = f"{'Model':<{w}}" + "".join(
        f"{'h='+str(h)+'  ('+str(valid[h])+')':>14}" for h in horizons
    ) + f"{'total':>8}"
    if show_mse:
        header += f"  {'avg_mse':>8}"
    print(header)
    print(sep)
    for _, r in df_out.iterrows():
        flag = " ⚠" if r.get("high_var", False) else "  "
        line = f"{r['config']:<{w}}{flag}"[: w + 2]
        # pad to w if flag makes it too long
        line = f"{r['config'] + (' ⚠' if r.get('high_var', False) else ''):<{w}}"
        for h in horizons:
            c, nv = r[f"h{h}_count"], valid[h]
            cell  = f"{c}/{nv} ({100*c/nv:.0f}%)" if nv > 0 else "—"
            line += f"  {cell:>12}"
        line += f"  {int(r['total_count']):>6}"
        if show_mse:
            mse_v = r.get("avg_mse", np.nan)
            line += f"  {mse_v:8.4f}" if np.isfinite(mse_v) else f"  {'—':>8}"
        print(line)
    print(sep)
    if show_mse:
        print("  ⚠  std/mean > 0.5 on ≥1 horizon: MCS freq may overstate reliability")
    return df_out
