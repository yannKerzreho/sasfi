"""
experiments/exp_sas.py — SAS hyperparameter sweep (multi-symbol).

Goal: identify the best SAS variant to compete with HAR/NLinear across assets.

Grid (10 SAS configs)
---------------------
  Diagonal basis — O(n) per step, stable for any n:
    diag_n100_sn{0.80,0.90,0.95,0.99}      : degree-1, varying spectral norm
    diag_n200_sn{0.90,0.95}                 : larger reservoir, degree-1
    diag_p2_n100_sn{0.90,0.95}              : degree-2 input polynomial

  Augmented SAS (reservoir + HAR features, guaranteed ≥ HAR quality):
    aug_n100_sn{0.90,0.95}                  : fallback for h=22 stability

  No degree > 2: degree-1 consistently outperforms degree-2 empirically.
  Linear/Trigo bases excluded: O(n²) cost + catastrophic overfit in sweep.

Baselines
---------
  HAR, HAR_Ridge  (always included — the main reference for this paper)

Protocol
--------
Rolling OOS: window=500, refit_freq=252.
Runs on all symbols by default (--symbols all).
Results: mean ± std MSE table + MCS frequency table.

Usage
-----
    python experiments/exp_sas.py                         # all symbols
    python experiments/exp_sas.py --symbols .AEX .SPX    # subset
    python experiments/exp_sas.py --symbols .AEX          # quick check

Results saved to experiments/results_sas.csv.
Run from repo root.
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    quick_oos, mean_losses, print_mse_table, print_mcs_frequency,
    _sq_errors_to_eval_df, HORIZONS, WINDOW, REFIT_FREQ,
)
from data.data_loader import load_rv, available_symbols
from models.linear    import HARForecaster
from models.sas       import SASForecaster, AugSASForecaster
from models.sas_utils import DiagonalPoly

CSV         = ROOT / "rv.csv"
DEFAULT_SYM = ".AEX"

# Ordered list for display (baselines first, then SAS variants sorted by family)
_MODEL_ORDER = [
    "HAR", "HAR_Ridge",
    "diag_n100_sn0.80", "diag_n100_sn0.90", "diag_n100_sn0.95", "diag_n100_sn0.99",
    "diag_n200_sn0.90", "diag_n200_sn0.95",
    "diag_p2_n100_sn0.90", "diag_p2_n100_sn0.95",
    "aug_n100_sn0.90", "aug_n100_sn0.95",
]


def build_grid() -> list[tuple[str, callable]]:
    """
    Return list of (name, factory_fn) pairs.
    Factory functions are called fresh for each symbol to avoid state leakage.
    """
    configs: list[tuple[str, callable]] = []

    # ── Baselines ─────────────────────────────────────────────────────────
    configs.append(("HAR",       lambda: HARForecaster(ridge=False)))
    configs.append(("HAR_Ridge", lambda: HARForecaster(ridge=True)))

    # ── Diagonal, degree 1, n=100 — vary spectral norm ────────────────────
    for sn in [0.80, 0.90, 0.95, 0.99]:
        tag = f"diag_n100_sn{sn:.2f}"
        configs.append((tag, lambda sn=sn: SASForecaster(
            n_reservoir=100, basis="diagonal",
            spectral_norm=sn, washout=50, seed=42,
        )))

    # ── Diagonal, degree 1, n=200 — larger reservoir ──────────────────────
    for sn in [0.90, 0.95]:
        tag = f"diag_n200_sn{sn:.2f}"
        configs.append((tag, lambda sn=sn: SASForecaster(
            n_reservoir=200, basis="diagonal",
            spectral_norm=sn, washout=50, seed=42,
        )))

    # ── Diagonal, degree 2 — richer input modulation ──────────────────────
    for sn in [0.90, 0.95]:
        tag = f"diag_p2_n100_sn{sn:.2f}"
        configs.append((tag, lambda sn=sn: SASForecaster(
            n_reservoir=100,
            basis=DiagonalPoly(p_degree=2, q_degree=2, spectral_norm=sn),
            washout=50, seed=42,
        )))

    # ── Augmented SAS (reservoir + HAR features) ──────────────────────────
    # Guarantees readout ≥ HAR quality by construction: set reservoir weights
    # to zero → pure HAR.  Much more stable at h=22 than pure SAS.
    for sn in [0.90, 0.95]:
        tag = f"aug_n100_sn{sn:.2f}"
        configs.append((tag, lambda sn=sn: AugSASForecaster(
            n_reservoir=100, basis="diagonal",
            spectral_norm=sn, washout=50, seed=42,
        )))

    return configs


def run_one_symbol(
    symbol: str,
    grid:   list[tuple[str, callable]],
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, list[float]]]]:
    """
    Run all configs on one symbol.

    Returns
    -------
    mse_dict  : {model: {h: mean_mse}}
    loss_dict : {model: {h: [sq_errors]}}  (for MCS pivot)
    """
    log_values, dates = load_rv(CSV, symbol=symbol, target="rv5")
    T = len(log_values)
    if T < WINDOW + max(HORIZONS) + 2:
        print(f"  Skipping {symbol}: T={T}")
        return {}, {}

    print(f"  {symbol}  T={T}  — {len(grid)} configs")
    t0 = time.time()

    loss_dict: dict[str, dict[int, list[float]]] = {}
    for i, (tag, factory) in enumerate(grid):
        model = factory()
        losses = quick_oos(log_values, dates, {tag: model})
        loss_dict[tag] = losses[tag]
        mses = mean_losses(losses)
        avg  = float(np.nanmean([mses[tag].get(h, np.nan) for h in HORIZONS]))
        print(f"    [{i+1:2d}/{len(grid)}] {tag:<25s} "
              f"h1={mses[tag].get(1, np.nan):.4f}  "
              f"h5={mses[tag].get(5, np.nan):.4f}  "
              f"h22={mses[tag].get(22, np.nan):.4f}  "
              f"avg={avg:.4f}")

    mse_dict = {tag: {h: (float(np.mean(v)) if v else np.nan)
                      for h, v in h_dict.items()}
                for tag, h_dict in loss_dict.items()}
    print(f"    done in {time.time()-t0:.1f}s")
    return mse_dict, loss_dict


def main():
    parser = argparse.ArgumentParser(
        description="SAS hyperparameter sweep (multi-symbol)."
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help=(
            "Symbols to evaluate. Pass 'all' to use every symbol in rv.csv. "
            f"Defaults to {DEFAULT_SYM}."
        ),
    )
    parser.add_argument(
        "--mcs-alpha", type=float, default=0.10,
        help="MCS significance level.",
    )
    parser.add_argument(
        "--mcs-nboot", type=int, default=1000,
        help="MCS bootstrap replications.",
    )
    args = parser.parse_args()

    if args.symbols is None:
        symbols = [DEFAULT_SYM]
    elif args.symbols == ["all"]:
        symbols = available_symbols(CSV)
        print(f"Using all {len(symbols)} symbols: {symbols}")
    else:
        symbols = args.symbols

    grid = build_grid()
    print(f"\nGrid: {len(grid)} configs  ({len(grid)-2} SAS + 2 baselines)")
    print(f"Symbols: {symbols}\n")

    mse_by_symbol:  dict[str, dict[str, dict[int, float]]]       = {}
    loss_by_symbol: dict[str, dict[str, dict[int, list[float]]]] = {}
    all_records:    list[dict] = []

    for symbol in symbols:
        print(f"\n{'═'*60}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'═'*60}")
        try:
            mse_dict, loss_dict = run_one_symbol(symbol, grid)
        except ValueError as e:
            print(f"  Error: {e}")
            continue
        if not mse_dict:
            continue

        mse_by_symbol[symbol]  = mse_dict
        loss_by_symbol[symbol] = loss_dict

        for tag, h_dict in mse_dict.items():
            row = {"symbol": symbol, "config": tag}
            for h in HORIZONS:
                row[f"mse_h{h}"] = h_dict.get(h, np.nan)
            row["mse_avg"] = float(np.nanmean([h_dict.get(h, np.nan) for h in HORIZONS]))
            all_records.append(row)

    if not all_records:
        print("\nNo results.")
        return

    # ── MSE table ─────────────────────────────────────────────────────────
    print_mse_table(
        mse_by_symbol,
        horizons     = HORIZONS,
        title        = "SAS sweep  MSE (mean ± std, window=2000)",
        model_order  = _MODEL_ORDER,
    )

    # ── MCS frequency table ───────────────────────────────────────────────
    # Convert loss lists → DataFrame[config, horizon, test_date, sq_err]
    evals_by_symbol = _sq_errors_to_eval_df(loss_by_symbol)
    print_mcs_frequency(
        evals_by_symbol,
        horizons      = HORIZONS,
        alpha         = args.mcs_alpha,
        n_boot        = args.mcs_nboot,
        seed          = 42,
        title         = "SAS sweep MCS frequency",
        mse_by_symbol = mse_by_symbol,
    )

    # ── save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    out = ROOT / "experiments" / "results_sas.csv"
    df.to_csv(out, index=False)
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
