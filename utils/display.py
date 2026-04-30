"""
utils/display.py — Display helpers for multi-symbol OOS evaluation.

print_precision_table    : MSE mean±std + QLIKE mean±std (two stacked blocks).
print_mcs_frequency      : MCS survival frequency table (raw sq_err, no row-norm).
print_per_horizon_scoring: per-symbol RMSE + MCS survivors.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd


# ── precision table ───────────────────────────────────────────────────────────

def _agg_metric(
    metric_by_symbol: "dict[str, dict[str, dict[int, float]]]",
    all_models:       "list[str]",
    horizons:         "list[int]",
) -> "dict[str, dict[int, tuple[float, float]]]":
    """Return {model: {h: (mean, std)}} aggregated across symbols."""
    raw: dict[str, dict[int, list[float]]] = {
        m: {h: [] for h in horizons} for m in all_models
    }
    for sym_dict in metric_by_symbol.values():
        for m, h_dict in sym_dict.items():
            if m in raw:
                for h in horizons:
                    v = h_dict.get(h, np.nan)
                    if np.isfinite(v):
                        raw[m][h].append(v)
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for m in all_models:
        out[m] = {}
        for h in horizons:
            vals = raw[m][h]
            mu   = float(np.mean(vals)) if vals else np.nan
            sd   = float(np.std(vals, ddof=min(1, len(vals)-1))) \
                   if len(vals) > 1 else np.nan
            out[m][h] = (mu, sd)
    return out


def _print_metric_block(
    agg:        "dict[str, dict[int, tuple[float, float]]]",
    all_models: "list[str]",
    horizons:   "list[int]",
    block_title: str,
    n_sym:      int,
    fmt:        str = ".4f",
) -> None:
    """Print one metric block (MSE or QLIKE) with mean±std columns."""
    avgs = {}
    for m in all_models:
        vals = [agg[m][h][0] for h in horizons if np.isfinite(agg[m][h][0])]
        avgs[m] = float(np.mean(vals)) if vals else np.nan

    order = sorted(all_models, key=lambda m: avgs.get(m, np.inf))
    w   = max(len(m) for m in all_models) + 2
    col = len(horizons) * 18 + 10
    sep = "─" * (w + col)

    print(f"\n{sep}")
    print(f"  {block_title}  (N={n_sym} symbols)")
    print(sep)
    header = (f"{'Model':<{w}}"
              + "".join(f"{'h='+str(h):>18}" for h in horizons)
              + f"{'avg':>12}")
    print(header)
    print(sep)
    for m in order:
        line = f"{m:<{w}}"
        for h in horizons:
            mu, sd = agg[m][h]
            if not np.isfinite(mu):
                cell = "—"
            elif not np.isfinite(sd):
                cell = format(mu, fmt) + "          "
            else:
                cell = f"{mu:{fmt}}±{sd:{fmt}}"
            line += f"  {cell:>16}"
        avg = avgs[m]
        line += f"  {avg:>10{fmt}}" if np.isfinite(avg) else f"  {'—':>10}"
        print(line)
    print(sep)


def print_precision_table(
    mse_by_symbol:   "dict[str, dict[str, dict[int, float]]]",
    qlike_by_symbol: "dict[str, dict[str, dict[int, float]]]",
    horizons:        list[int],
    title:           str = "",
    model_order:     "list[str] | None" = None,
) -> None:
    """
    Print two stacked metric blocks — MSE then QLIKE — with the same model
    ordering so rows are directly comparable.

    Parameters
    ----------
    mse_by_symbol   : {symbol: {model: {h: MSE}}}
    qlike_by_symbol : {symbol: {model: {h: mean_QLIKE}}}
    """
    all_models: list[str] = []
    seen: set[str] = set()
    if model_order:
        for m in model_order:
            all_models.append(m); seen.add(m)
    for d in list(mse_by_symbol.values()) + list(qlike_by_symbol.values()):
        for m in d:
            if m not in seen:
                all_models.append(m); seen.add(m)

    n_sym = max(len(mse_by_symbol), len(qlike_by_symbol))
    if title:
        w   = max(len(m) for m in all_models) + 2
        col = len(horizons) * 18 + 10
        sep = "═" * (w + col)
        print(f"\n{sep}\n  {title}\n{sep}")

    agg_mse   = _agg_metric(mse_by_symbol,   all_models, horizons)
    agg_qlike = _agg_metric(qlike_by_symbol, all_models, horizons)
    _print_metric_block(agg_mse,   all_models, horizons,
                        "MSE", n_sym, fmt=".4e")
    _print_metric_block(agg_qlike, all_models, horizons,
                        "QLIKE (lower is better)", n_sym, fmt=".4f")


# ── MCS frequency table ───────────────────────────────────────────────────────

def _run_mcs_one_symbol(
    df_eval:  pd.DataFrame,
    horizons: list[int],
    alpha:    float,
    n_boot:   int,
    seed:     int,
) -> dict[int, list[str]]:
    """
    Run per-horizon MCS for one symbol's eval DataFrame (raw sq_err, no
    row-normalisation).

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

        T = pivot.shape[0]
        w = max(1, int(T ** (1 / 3)))
        mcs = ModelConfidenceSet(
            data      = pivot.values,
            alpha     = alpha,
            B         = n_boot,
            w         = w,
            algorithm = "R",
            seed      = seed,
            names     = np.array(list(pivot.columns)),
        ).run()
        result[h] = mcs.included
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
    Run MCS independently per symbol × horizon; print survival frequency.

    Returns DataFrame[config, h{k}_count, h{k}_frac, total_count, avg_mse].
    """
    n_sym   = len(evals_by_symbol)
    symbols = list(evals_by_symbol.keys())
    print(f"\n  Running MCS (α={alpha}, B={n_boot}) for {n_sym} symbol(s) …")

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

    # optional MSE stats for avg_mse column
    mse_stats: dict[str, tuple[float, bool]] = {}
    if mse_by_symbol:
        agg: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list))
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
                    sd = float(np.std(vals, ddof=min(1, len(vals)-1))) \
                         if len(vals) > 1 else 0.0
                    avgs.append(mu)
                    if mu > 0 and sd / mu > 0.5:
                        high_var = True
            mse_stats[m] = (float(np.mean(avgs)) if avgs else np.nan, high_var)

    if model_order is not None:
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
            row["avg_mse"], row["high_var"] = mse_stats[m]
        else:
            row["avg_mse"], row["high_var"] = np.nan, False
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if model_order is None:
        df_out = df_out.sort_values("total_count", ascending=False)

    show_mse = mse_by_symbol is not None
    w        = max(len(r["config"]) for r in rows) + 2
    extra_w  = 12 if show_mse else 0
    sep      = "─" * (w + len(horizons) * 14 + 8 + extra_w)
    hdr      = title or "MCS frequency table"

    print(f"\n{sep}")
    print(f"  {hdr}  (α={alpha}, N={n_sym} symbols)")
    print(f"  freq = #symbols model survived MCS  |  ⚠ = CV>0.5 on ≥1 horizon")
    print(sep)
    header = (f"{'Model':<{w}}"
              + "".join(f"{'h='+str(h)+'  ('+str(valid[h])+')':>14}"
                        for h in horizons)
              + f"{'total':>8}")
    if show_mse:
        header += f"  {'avg_mse':>8}"
    print(header)
    print(sep)
    for _, r in df_out.iterrows():
        line = f"{r['config'] + (' ⚠' if r.get('high_var', False) else ''):<{w}}"
        for h in horizons:
            c, nv = r[f"h{h}_count"], valid[h]
            cell  = f"{c}/{nv} ({100*c/nv:.0f}%)" if nv > 0 else "—"
            line += f"  {cell:>12}"
        line += f"  {int(r['total_count']):>6}"
        if show_mse:
            mse_v = r.get("avg_mse", np.nan)
            line += f"  {mse_v:8.4e}" if np.isfinite(mse_v) else f"  {'—':>8}"
        print(line)
    print(sep)
    if show_mse:
        print("  ⚠  std/mean > 0.5 on ≥1 horizon: MCS freq may overstate "
              "reliability")
    return df_out


# ── per-horizon per-symbol scoring ────────────────────────────────────────────

def print_per_horizon_scoring(
    evals_by_symbol: "dict[str, pd.DataFrame]",
    horizons:        list[int],
    alpha:           float = 0.10,
    model_order:     "list[str] | None" = None,
) -> None:
    """
    For each horizon, print a per-symbol RMSE table (raw, no row-normalisation).
    A star (*) marks models in the MCS best set for that symbol × horizon.
    """
    for h in horizons:
        table_rows: list[dict] = []
        m_list_ref: list[str]  = []

        for sym, df in evals_by_symbol.items():
            df_h = df[df["horizon"] == h]
            if df_h.empty:
                continue

            m_list = (
                [m for m in model_order if m in df_h["config"].unique()]
                if model_order
                else sorted(df_h["config"].unique())
            )
            if not m_list_ref:
                m_list_ref = m_list

            pivot = (
                df_h.pivot_table(index="test_date", columns="config",
                                 values="sq_err", aggfunc="mean")
                    .reindex(columns=m_list)
                    .dropna(how="all")
            )

            try:
                mcs_res   = _run_mcs_one_symbol(df_h, [h], alpha, 1000, 42)
                survivors = set(mcs_res.get(h, []))
            except Exception:
                survivors = set()

            row: dict = {"Symbol": sym}
            for m in m_list:
                col = (pivot[m].dropna()
                       if m in pivot.columns
                       else pd.Series([], dtype=float))
                if not col.empty:
                    rmse   = float(np.sqrt(col.mean()))
                    star   = "*" if m in survivors else " "
                    row[m] = f"{rmse:.4e}{star}"
                else:
                    row[m] = "    —   "
            table_rows.append(row)

        if not table_rows:
            continue

        sym_w = max(max(len(r["Symbol"]) for r in table_rows), 6)
        col_w = {m: max(len(m), 9) for m in m_list_ref}
        header = f"{'Symbol':<{sym_w}}" + "".join(
            f"  {m:>{col_w[m]}}" for m in m_list_ref
        )
        sep = "─" * len(header)

        print(f"\n{sep}")
        print(f"  RMSE — H={h}  (α={alpha}, * = in MCS best set)")
        print(sep)
        print(header)
        print(sep)
        for row in table_rows:
            line = f"{row['Symbol']:<{sym_w}}"
            for m in m_list_ref:
                cell = row.get(m, "    —   ")
                line += f"  {cell:>{col_w[m]}}"
            print(line)
        print(sep)
        print("  * model survives the MCS at this horizon")
