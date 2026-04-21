"""
experiments/exp_sas.py — Focused SAS ablation suite.

Context / what we know
----------------------
After the full 30-symbol benchmark (main.py, W=2000, R=20, h∈{1,5,10}):

  • SAS_q2  wins on avg MSE (0.4955) and h=1 (0.3958) but has low MCS
            frequency (41/90) — gains are real but small relative to noise.
  • SAS_q1  is more *consistent*: 78/90 MCS survivals vs 41/90 for q=2,
            despite being 0.003 worse on avg MSE.
  • n=200, sn=0.95, p=1 are confirmed best for q=1; unclear for q=2.
  • AugSAS  removed (no longer competitive once n is large enough).
  • h=22 dropped (source of divergences; not used in main benchmark).

Open questions — what this experiment answers
---------------------------------------------
  Q1.  q-degree:   is q=2 robustly better, or does it win only on some symbols?
       → q=1 vs q=2 vs q=3 at fixed (n=200, sn=0.95, p=1).
       Specifically track *per-symbol* wins to understand MCS vs MSE tension.

  Q2.  n for q=2:  q=2 injects richer features — does it need more units?
       → q=2 at n=100 / 200 / 400 (same sn, p).

  Q3.  sn for q=2: q=1 was sensitive to sn (0.95>0.90>>0.99).
       Is q=2 similarly sensitive, or does richer Q make it more robust?
       → q=2 at sn=0.85 / 0.90 / 0.95.

  Q4.  p-degree with q=2: does a quadratic P (input-gated timescales) add
       anything on top of quadratic Q, or does the redundancy hurt?
       → (p=1,q=2) vs (p=2,q=2) at n=200, sn=0.95.

Baselines
---------
  HAR (OLS)      — primary reference throughout the paper.
  SAS_q1_n200    — confirmed best q=1 config from main benchmark.

Protocol
--------
  Rolling OOS: W=2000, R=20, h∈{1,5,10}   (matches main.py exactly).
  All 30 symbols by default; pass --symbols .AEX .SPX for a quick check.

  Stability flag: any config with avg MSE > 1.5 on a symbol is counted
  as a divergence and excluded from the stable-subset MSE table.

Usage
-----
    python experiments/exp_sas.py                       # all symbols
    python experiments/exp_sas.py --symbols .AEX .SPX   # quick check
    python experiments/exp_sas.py --no-mcs              # skip MCS
    python experiments/exp_sas.py --q-only              # Q1 only (fast)
"""

from __future__ import annotations
import sys
import time
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    quick_oos, print_mse_table, print_mcs_frequency,
    _sq_errors_to_eval_df,
)
from data.data_loader import load_rv, available_symbols
from models.linear    import HARForecaster
from models.sas       import SASForecaster

CSV        = ROOT / "rv.csv"
HORIZONS   = [1, 5, 10]          # matches main.py
WINDOW     = 2000
REFIT_FREQ = 20
SEED       = 42
WASHOUT    = 50
DIVERGE_THRESH = 1.5             # avg MSE above this → diverged

# ── model order for display ───────────────────────────────────────────────────
_ORDER_Q    = ["HAR", "q1_n200_sn95", "q2_n200_sn95", "q3_n200_sn95"]
_ORDER_N    = ["HAR", "q1_n200_sn95", "q2_n100_sn95", "q2_n200_sn95", "q2_n400_sn95"]
_ORDER_SN   = ["HAR", "q1_n200_sn95", "q2_n200_sn85", "q2_n200_sn90", "q2_n200_sn95"]
_ORDER_P    = ["HAR", "q1_n200_sn95", "q2_n200_sn95", "p2q2_n200_sn95"]


def _sas(n, sn, p=1, q=1):
    """Factory returning a fresh SASForecaster with given hyperparams."""
    return SASForecaster(
        n_reservoir=n, basis="diagonal",
        spectral_norm=sn, p_degree=p, q_degree=q,
        washout=WASHOUT, seed=SEED,
    )


def build_q_grid() -> dict[str, callable]:
    """Q1 — q-degree sweep at fixed n=200, sn=0.95, p=1."""
    return {
        "HAR":          lambda: HARForecaster(ridge=False),
        "q1_n200_sn95": lambda: _sas(200, 0.95, p=1, q=1),
        "q2_n200_sn95": lambda: _sas(200, 0.95, p=1, q=2),
        "q3_n200_sn95": lambda: _sas(200, 0.95, p=1, q=3),
    }


def build_n_grid() -> dict[str, callable]:
    """Q2 — n sweep for q=2 at sn=0.95, p=1."""
    return {
        "HAR":           lambda: HARForecaster(ridge=False),
        "q1_n200_sn95":  lambda: _sas(200, 0.95, p=1, q=1),
        "q2_n100_sn95":  lambda: _sas(100, 0.95, p=1, q=2),
        "q2_n200_sn95":  lambda: _sas(200, 0.95, p=1, q=2),
        "q2_n400_sn95":  lambda: _sas(400, 0.95, p=1, q=2),
    }


def build_sn_grid() -> dict[str, callable]:
    """Q3 — sn sweep for q=2 at n=200, p=1."""
    return {
        "HAR":           lambda: HARForecaster(ridge=False),
        "q1_n200_sn95":  lambda: _sas(200, 0.95, p=1, q=1),
        "q2_n200_sn85":  lambda: _sas(200, 0.85, p=1, q=2),
        "q2_n200_sn90":  lambda: _sas(200, 0.90, p=1, q=2),
        "q2_n200_sn95":  lambda: _sas(200, 0.95, p=1, q=2),
    }


def build_p_grid() -> dict[str, callable]:
    """Q4 — p-degree with q=2 at n=200, sn=0.95."""
    return {
        "HAR":            lambda: HARForecaster(ridge=False),
        "q1_n200_sn95":   lambda: _sas(200, 0.95, p=1, q=1),
        "q2_n200_sn95":   lambda: _sas(200, 0.95, p=1, q=2),
        "p2q2_n200_sn95": lambda: _sas(200, 0.95, p=2, q=2),
    }


# ── per-symbol win tracker ─────────────────────────────────────────────────────

def print_per_symbol_h1(
    mse_by_sym: dict[str, dict[str, dict[int, float]]],
    models: list[str],
) -> None:
    """
    For h=1: show which model wins per symbol, and flag ties (< 0.001 gap).
    Helps diagnose why MCS and mean-MSE rankings diverge.
    """
    print("\n── Per-symbol h=1 winner ──────────────────────────────────────")
    wins = {m: 0 for m in models}
    for sym, sym_mse in sorted(mse_by_sym.items()):
        row = {m: sym_mse.get(m, {}).get(1, np.nan) for m in models}
        valid = {m: v for m, v in row.items() if np.isfinite(v)}
        if not valid:
            continue
        best_val = min(valid.values())
        best_m   = min(valid, key=valid.get)
        wins[best_m] += 1
        # flag if margin over 2nd-best is tiny
        sorted_vals = sorted(valid.values())
        margin = (sorted_vals[1] - sorted_vals[0]) if len(sorted_vals) > 1 else 0
        marker = "~" if margin < 0.001 else " "
        vals_str = "  ".join(f"{m}={v:.4f}" for m, v in sorted(valid.items()))
        print(f"  {marker}{sym:<12} best={best_m}  {vals_str}")
    print(f"\n  Win counts (h=1): " +
          "  ".join(f"{m}={wins[m]}" for m in models))


def print_stability(
    mse_by_sym: dict[str, dict[str, dict[int, float]]],
    models: list[str],
    thresh: float = DIVERGE_THRESH,
) -> list[str]:
    """
    Print divergence counts and return list of stable symbols
    (all models below thresh on all horizons).
    """
    print(f"\n── Stability (avg MSE > {thresh}) ────────────────────────────")
    w = max(len(m) for m in models) + 2
    diverged_syms: dict[str, list[str]] = {sym: [] for sym in mse_by_sym}
    for sym, sym_mse in mse_by_sym.items():
        for m in models:
            vals = list(sym_mse.get(m, {}).values())
            if vals and float(np.nanmean(vals)) > thresh:
                diverged_syms[sym].append(m)

    for m in models:
        n_div = sum(1 for sym, bad in diverged_syms.items() if m in bad)
        n_tot = len(mse_by_sym)
        bar   = "█" * n_div + "░" * (n_tot - n_div)
        print(f"  {m:<{w}} {n_div:>2}/{n_tot}  {bar}")

    stable = [
        sym for sym, bad in diverged_syms.items() if not bad
    ]
    print(f"\n  Stable symbols ({len(stable)}/{len(mse_by_sym)}): "
          f"{' '.join(sorted(stable)[:8])}"
          f"{'…' if len(stable) > 8 else ''}")
    return stable


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_mse_csv(
    mse_by_sym: dict,
    stable_syms: list[str],
    horizons: list[int],
    out_dir: Path,
    slug: str,
) -> None:
    """Save per-symbol MSE to <out_dir>/<slug>_mse.csv."""
    rows = []
    for sym, sym_mse in mse_by_sym.items():
        for model, h_dict in sym_mse.items():
            row = {"symbol": sym, "model": model, "stable": sym in stable_syms}
            for h in horizons:
                row[f"mse_h{h}"] = h_dict.get(h, np.nan)
            row["mse_avg"] = float(np.nanmean([h_dict.get(h, np.nan) for h in horizons]))
            rows.append(row)
    path = out_dir / f"{slug}_mse.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  → saved {path.name}  ({len(rows)} rows)")


def _save_mcs_csv(
    mcs_df: pd.DataFrame,
    out_dir: Path,
    slug: str,
) -> None:
    """Save MCS frequency DataFrame to <out_dir>/<slug>_mcs.csv."""
    if mcs_df is None or mcs_df.empty:
        return
    path = out_dir / f"{slug}_mcs.csv"
    mcs_df.to_csv(path, index=False)
    print(f"  → saved {path.name}")


# ── run one grid ──────────────────────────────────────────────────────────────

def run_grid(
    syms: list[str],
    grid: dict[str, callable],
    model_order: list[str],
    title: str,
    slug: str,
    out_dir: Path,
    run_mcs: bool,
    mcs_alpha: float,
    mcs_nboot: int,
) -> None:
    mse_by_sym:  dict = {}
    loss_by_sym: dict = {}

    for sym in syms:
        try:
            log_vals, dates = load_rv(CSV, sym)
        except Exception as e:
            print(f"  [skip] {sym}: {e}")
            continue

        T = len(log_vals)
        if T < WINDOW + max(HORIZONS) + 2:
            print(f"  [skip] {sym}: T={T} too short")
            continue

        models = {name: factory() for name, factory in grid.items()}
        t0 = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            losses = quick_oos(
                log_vals, dates, models,
                horizons=HORIZONS, window=WINDOW, refit_freq=REFIT_FREQ,
            )

        dt = time.perf_counter() - t0
        sym_mse = {
            name: {h: float(np.mean(errs)) for h, errs in h_map.items()}
            for name, h_map in losses.items()
        }
        mse_by_sym[sym]  = sym_mse
        loss_by_sym[sym] = losses

        avg_str = "  ".join(
            f"{n}={float(np.nanmean(list(v.values()))):.4f}"
            for n, v in sym_mse.items()
        )
        print(f"  {sym:<12} {dt:5.1f}s  {avg_str}")

    if not mse_by_sym:
        print("  No results.")
        return

    print()

    # stability
    stable_syms = print_stability(mse_by_sym, model_order)
    stable_mse  = {sym: mse_by_sym[sym] for sym in stable_syms}

    # save per-symbol MSE (all symbols, stable flag included)
    _save_mse_csv(mse_by_sym, stable_syms, HORIZONS, out_dir, slug)

    # per-symbol h=1 breakdown
    print_per_symbol_h1(mse_by_sym, model_order)

    # MSE tables
    print_mse_table(
        stable_mse, HORIZONS,
        title=f"{title}  [stable {len(stable_syms)}/{len(mse_by_sym)} symbols]",
        model_order=model_order,
    )
    if len(stable_syms) < len(mse_by_sym):
        print_mse_table(
            mse_by_sym, HORIZONS,
            title=f"{title}  [all {len(mse_by_sym)} symbols — means inflated by divergences]",
            model_order=model_order,
        )

    # MCS
    mcs_df = None
    if run_mcs:
        stable_eval = _sq_errors_to_eval_df(
            {sym: loss_by_sym[sym] for sym in stable_syms if sym in loss_by_sym}
        )
        if stable_eval:
            mcs_df = print_mcs_frequency(
                stable_eval, HORIZONS,
                alpha=mcs_alpha, n_boot=mcs_nboot, seed=SEED,
                title=f"MCS₀.₁₀ — {title} [stable subset]",
                mse_by_symbol=stable_mse,
                model_order=model_order,
            )
            _save_mcs_csv(mcs_df, out_dir, slug)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused SAS ablation: q-degree, n, sn, p-degree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", nargs="+", default=["all"])
    parser.add_argument("--no-mcs",    action="store_true")
    parser.add_argument("--q-only",    action="store_true",
                        help="Run Q1 (q-degree sweep) only — fastest option.")
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-nboot", type=int,   default=1000)
    parser.add_argument("--out-dir",   default=None,
                        help="Output directory for CSV results. "
                             "Defaults to experiments/results_sas/<timestamp>/")
    args = parser.parse_args()

    if args.symbols == ["all"]:
        syms = available_symbols(CSV)
    else:
        syms = args.symbols

    # ── output directory ──────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from datetime import datetime
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "experiments" / "results_sas" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {out_dir}")

    print(f"\nRunning on {len(syms)} symbol(s): {' '.join(syms[:6])}"
          f"{'…' if len(syms) > 6 else ''}")
    print(f"Protocol: W={WINDOW}, R={REFIT_FREQ}, H={HORIZONS}\n")

    grids = [
        ("Q1 — q-degree sweep (n=200, sn=0.95, p=1)", "q1_qdegree",
         build_q_grid(), _ORDER_Q),
    ]
    if not args.q_only:
        grids += [
            ("Q2 — n sweep for q=2 (sn=0.95, p=1)", "q2_nsize",
             build_n_grid(), _ORDER_N),
            ("Q3 — sn sweep for q=2 (n=200, p=1)", "q3_sn",
             build_sn_grid(), _ORDER_SN),
            ("Q4 — p-degree with q=2 (n=200, sn=0.95)", "q4_pdegree",
             build_p_grid(), _ORDER_P),
        ]

    for title, slug, grid, order in grids:
        print(f"\n{'═'*70}")
        print(f"  {title}")
        print(f"{'═'*70}\n")
        run_grid(
            syms, grid, order, title,
            slug      = slug,
            out_dir   = out_dir,
            run_mcs   = not args.no_mcs,
            mcs_alpha = args.mcs_alpha,
            mcs_nboot = args.mcs_nboot,
        )

    print(f"\nAll results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
