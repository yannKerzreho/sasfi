"""
utils/dm.py — Diebold-Mariano test helpers.

_dm_pvalue          : raw-diff DM p-values with NW-HAC + HLN correction.
_dm_wins            : run DM tests across all symbols for one loss column.
print_beats_benchmark : formatted win-rate table vs a benchmark model.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd


def _dm_pvalue(
    loss_model: np.ndarray,
    loss_bench: np.ndarray,
    h:          int,
) -> "tuple[float, float]":
    """
    Two-sided Diebold-Mariano test with raw loss differences:
        d_t = loss_model_t − loss_bench_t

    HAC variance: Newey-West with (h−1) Bartlett lags (MA(h−1) structure
    of h-step forecast errors).
    Finite-sample correction: Harvey-Leybourne-Newbold (1997).

    Returns
    -------
    p_model_better : small → model significantly better (d_bar < 0).
    p_bench_better : small → benchmark significantly better (d_bar > 0).
    """
    from scipy import stats

    d     = loss_model - loss_bench
    T     = len(d)
    d_bar = float(np.mean(d))
    d_c   = d - d_bar

    # Newey-West HAC with h-1 lags
    n_lags  = max(0, h - 1)
    hac_var = float(np.mean(d_c ** 2))
    for lag in range(1, n_lags + 1):
        w_k      = 1.0 - lag / (n_lags + 1)          # Bartlett weight
        gamma_k  = float(np.mean(d_c[lag:] * d_c[:-lag]))
        hac_var += 2.0 * w_k * gamma_k

    if hac_var <= 0.0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(hac_var / T)

    # Harvey-Leybourne-Newbold finite-sample correction
    hln     = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    dm_stat *= hln

    p_model_better = float(stats.t.cdf(dm_stat, df=T - 1))
    p_bench_better = float(stats.t.sf(dm_stat,  df=T - 1))
    return p_model_better, p_bench_better


def _dm_wins(
    evals_by_symbol: "dict[str, pd.DataFrame]",
    horizons:        list[int],
    benchmark:       str,
    alpha:           float,
    loss_col:        str,
) -> "tuple[dict, dict, dict]":
    """
    Run raw-diff DM tests for one loss column across all symbols.

    Returns
    -------
    wins_model : {model: {h: count}}  — symbols where model beats benchmark.
    wins_bench : {model: {h: count}}  — symbols where benchmark beats model.
    total      : {model: {h: count}}  — valid test count.
    """
    wins_model: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    wins_bench: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    total:      dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for sym, df in evals_by_symbol.items():
        if benchmark not in df["config"].unique():
            continue
        if loss_col not in df.columns:
            continue
        for h in horizons:
            sub = df[df["horizon"] == h].dropna(subset=[loss_col])
            bm  = sub[sub["config"] == benchmark].set_index("test_date")[loss_col]
            if bm.empty:
                continue
            for model in sub["config"].unique():
                if model == benchmark:
                    continue
                mod_loss = sub[sub["config"] == model].set_index("test_date")[loss_col]
                common   = bm.index.intersection(mod_loss.index)
                if len(common) < max(10, 2 * h):
                    continue
                p_mod, p_bm = _dm_pvalue(
                    mod_loss[common].values, bm[common].values, h=h
                )
                total[model][h] += 1
                if np.isfinite(p_mod) and p_mod < alpha:
                    wins_model[model][h] += 1
                if np.isfinite(p_bm) and p_bm < alpha:
                    wins_bench[model][h] += 1
    return wins_model, wins_bench, total


def print_beats_benchmark(
    evals_by_symbol: "dict[str, pd.DataFrame]",
    horizons:        list[int],
    benchmark:       str = "HAR",
    alpha:           float = 0.10,
    model_order:     "list[str] | None" = None,
    title:           str = "",
) -> pd.DataFrame:
    """
    Diebold-Mariano frequency table (raw-diff, NW-HAC + HLN correction).

    For each (model, horizon, symbol):
        d_t = loss_model_t − loss_bench_t   (MSE and QLIKE separately)
    Prints one table per horizon showing win rates for both loss functions.

    Returns
    -------
    DataFrame with MSE and QLIKE win counts per model × horizon.
    """
    wins_mse,  bm_mse,  tot_mse  = _dm_wins(
        evals_by_symbol, horizons, benchmark, alpha, "sq_err"
    )
    wins_qlik, bm_qlik, tot_qlik = _dm_wins(
        evals_by_symbol, horizons, benchmark, alpha, "qlike"
    )
    has_qlike = bool(tot_qlik)

    all_seen = set(tot_mse) | set(tot_qlik)
    if not all_seen:
        print(f"  No DM results (benchmark '{benchmark}' not found or "
              f"insufficient data).")
        return pd.DataFrame()

    if model_order:
        all_models = [m for m in model_order if m in all_seen and m != benchmark]
        all_models += sorted(m for m in all_seen
                             if m not in model_order and m != benchmark)
    else:
        all_models = sorted(all_seen - {benchmark})

    rows = []
    for m in all_models:
        row: dict = {"config": m}
        for h in horizons:
            row[f"h{h}_mse_wins"]    = wins_mse[m][h]
            row[f"h{h}_mse_bm_wins"] = bm_mse[m][h]
            row[f"h{h}_mse_n"]       = tot_mse[m][h]
            row[f"h{h}_ql_wins"]     = wins_qlik[m][h]
            row[f"h{h}_ql_bm_wins"]  = bm_qlik[m][h]
            row[f"h{h}_ql_n"]        = tot_qlik[m][h]
        row["total_mse_wins"] = sum(wins_mse[m][h] for h in horizons)
        row["total_ql_wins"]  = sum(wins_qlik[m][h] for h in horizons)
        rows.append(row)

    df_out = pd.DataFrame(rows)

    n_sym = len(evals_by_symbol)
    w     = max(len(r["config"]) for r in rows) + 2
    hdr   = title or f"DM test — significant wins vs {benchmark}"
    cw    = 11

    def _frac(ww, nn):
        return "—" if nn == 0 else f"{int(ww)}/{int(nn)} ({100*ww/nn:.0f}%)"

    print(f"\n  {hdr}  (α={alpha}, N={n_sym} symbols)")
    print(f"  raw-diff DM tests (NW-HAC + HLN correction)")
    suffix = "" if has_qlike else "  (MSE only)"
    print(f"  mod = model beats {benchmark}  |  bm = {benchmark} beats model"
          + suffix)

    for h in horizons:
        ns    = [tot_mse[m][h] for m in all_models if tot_mse[m][h] > 0]
        N     = max(ns) if ns else 0
        order = sorted(all_models, key=lambda m: (-wins_mse[m][h], m))

        if has_qlike:
            sep = "─" * (w + 4 * (cw + 2) + 4)
            print(f"\n{sep}\n  h={h}  (N={N} symbols)\n{sep}")
            print(f"{'Model':<{w}}"
                  f"  {'MSE mod':>{cw}}  {'MSE bm':>{cw}}"
                  f"  {'QL mod':>{cw}}  {'QL bm':>{cw}}")
            print(sep)
            for m in order:
                print(
                    f"{m:<{w}}"
                    f"  {_frac(wins_mse[m][h],  tot_mse[m][h]):>{cw}}"
                    f"  {_frac(bm_mse[m][h],    tot_mse[m][h]):>{cw}}"
                    f"  {_frac(wins_qlik[m][h], tot_qlik[m][h]):>{cw}}"
                    f"  {_frac(bm_qlik[m][h],   tot_qlik[m][h]):>{cw}}"
                )
            print(sep)
        else:
            sep = "─" * (w + 2 * (cw + 2) + 4)
            print(f"\n{sep}\n  h={h}  (N={N} symbols)\n{sep}")
            print(f"{'Model':<{w}}  {'MSE mod':>{cw}}  {'MSE bm':>{cw}}")
            print(sep)
            for m in order:
                print(
                    f"{m:<{w}}"
                    f"  {_frac(wins_mse[m][h], tot_mse[m][h]):>{cw}}"
                    f"  {_frac(bm_mse[m][h],   tot_mse[m][h]):>{cw}}"
                )
            print(sep)

    return df_out
