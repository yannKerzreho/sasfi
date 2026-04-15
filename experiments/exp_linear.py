"""
experiments/exp_linear.py — Lookback window sweep for NLinear and DLinear.

Goal: find the optimal lookback for the linear models.

Grid
----
  model   : NLinear, DLinear
  lookback: 5, 10, 15, 20, 30, 40, 60, 90

HAR (OLS) and HAR_Ridge are always included as reference baselines.

Protocol
--------
Rolling OOS with window=500, refit every 252 steps.
When multiple symbols are requested the mean MSE across symbols is reported.

Usage
-----
  # Single symbol (default .AEX):
      python experiments/exp_linear.py

  # All available symbols:
      python experiments/exp_linear.py --symbols all

  # Specific symbols:
      python experiments/exp_linear.py --symbols .AEX .SPX .FTSE

Results saved to experiments/results_linear.csv.
Run from repo root.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    quick_oos, mean_losses, print_table, HORIZONS, WINDOW, REFIT_FREQ
)
from data.data_loader import load_rv, available_symbols
from models.linear    import HARForecaster, NLinearForecaster, DLinearForecaster

LOOKBACKS    = [5, 10, 15, 20, 30, 40, 60, 90]
DL_KERNELS   = {5: 3, 10: 3, 15: 5, 20: 5, 30: 7, 40: 7, 60: 9, 90: 11}
CSV          = ROOT / "rv.csv"
DEFAULT_SYM  = ".AEX"


def build_grid() -> list[tuple[str, object]]:
    configs = []
    configs.append(("HAR",       HARForecaster(ridge=False)))
    configs.append(("HAR_Ridge", HARForecaster(ridge=True)))
    for L in LOOKBACKS:
        configs.append((f"NLinear_L{L:02d}",
                        NLinearForecaster(lookback=L)))
        configs.append((f"DLinear_L{L:02d}",
                        DLinearForecaster(lookback=L, ma_kernel=DL_KERNELS[L])))
    return configs


def run_one_symbol(symbol: str) -> dict[str, dict[int, float]]:
    """Returns {config: {h: mse}} for one symbol."""
    log_values, dates = load_rv(CSV, symbol=symbol, target="rv5")
    T = len(log_values)
    if T < WINDOW + max(HORIZONS) + 2:
        print(f"  Skipping {symbol}: only {T} observations")
        return {}
    configs = build_grid()
    models  = dict(configs)
    print(f"  {symbol}  T={T} — running {len(models)} configs …")
    losses  = quick_oos(log_values, dates, models)
    return mean_losses(losses)


def main():
    parser = argparse.ArgumentParser(
        description="NLinear / DLinear lookback sweep (multi-symbol)."
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help=(
            "Symbols to evaluate. Pass 'all' to use every symbol in rv.csv. "
            f"Defaults to {DEFAULT_SYM}."
        ),
    )
    args = parser.parse_args()

    if args.symbols is None:
        symbols = [DEFAULT_SYM]
    elif args.symbols == ["all"]:
        symbols = available_symbols(CSV)
        print(f"Using all {len(symbols)} symbols: {symbols}")
    else:
        symbols = args.symbols

    all_records: list[dict] = []

    for symbol in symbols:
        print(f"\n{'═'*60}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'═'*60}")
        try:
            results = run_one_symbol(symbol)
        except ValueError as e:
            print(f"  Error: {e}")
            continue
        if not results:
            continue

        print_table(results, HORIZONS,
                    title=f"NLinear / DLinear lookback sweep — {symbol} rv5")

        for config, mses in results.items():
            row = {"symbol": symbol, "config": config}
            for h in HORIZONS:
                row[f"mse_h{h}"] = mses.get(h, np.nan)
            row["mse_avg"] = float(np.nanmean([mses.get(h, np.nan) for h in HORIZONS]))
            all_records.append(row)

        # ── per-family analysis ───────────────────────────────────────────
        for family, prefix in [("NLinear", "NLinear"), ("DLinear", "DLinear")]:
            sub = {k: v for k, v in results.items() if k.startswith(prefix)}
            if not sub:
                continue
            best_key = min(sub, key=lambda k: np.nanmean(list(sub[k].values())))
            print(f"\n  Best {family} [{symbol}]: {best_key}")
            for h in HORIZONS:
                best_h = min(sub, key=lambda k: sub[k].get(h, np.inf))
                print(f"    h={h}: best={best_h}  MSE={sub[best_h].get(h, np.nan):.4f}")

    if not all_records:
        print("\nNo results to save.")
        return

    df = pd.DataFrame(all_records)

    # ── aggregate over symbols (if multiple) ──────────────────────────────
    if len(symbols) > 1:
        agg = (
            df.groupby("config")[["mse_h1", "mse_h5", "mse_h22", "mse_avg"]]
            .mean()
            .sort_values("mse_avg")
        )
        print(f"\n{'═'*60}")
        print(f"  AGGREGATE (mean across {len(symbols)} symbols)")
        print(f"{'═'*60}")
        print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

    out = ROOT / "experiments" / "results_linear.csv"
    df.to_csv(out, index=False)
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
