"""
experiments/exp_rnn.py — RNN architecture sweep (multi-symbol).

Goal: find the best RNN configuration for RV forecasting.

Grid
----
  cell       : lstm, gru, lru
  hidden_size: 16, 32, 64, 128
  n_layers   : 1, 2
  lookback   : 10, 20, 40

Early stopping (patience=20, val_frac=0.2) is always active.

Fixed:
  n_epochs   = 300  (early stopping caps this in practice)
  lr         = 1e-3
  batch_size = 64

Protocol
--------
Rolling OOS with window=500, refit every 252 steps.
HAR included as a reference baseline (not affected by RNN params).

Usage
-----
  # Single symbol (default .AEX):
      python experiments/exp_rnn.py

  # All available symbols:
      python experiments/exp_rnn.py --symbols all

  # Specific symbols:
      python experiments/exp_rnn.py --symbols .AEX .SPX

Note: each config is run once (seed=42). 72 RNN configs + HAR,
      multiple minutes per symbol on CPU.

Results saved to experiments/results_rnn.csv.
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
    quick_oos, mean_losses, print_table, HORIZONS, WINDOW, REFIT_FREQ
)
from data.data_loader import load_rv, available_symbols
from models.linear    import HARForecaster
from models.rnn       import RNNForecaster

# ── grid ─────────────────────────────────────────────────────────────────────

CELLS     = ["lstm", "gru", "lru"]
HIDDENS   = [16, 32, 64, 128]
N_LAYERS  = [1, 2]
LOOKBACKS = [10, 20, 40]

CSV         = ROOT / "rv.csv"
DEFAULT_SYM = ".AEX"


def build_grid() -> list[tuple[str, object]]:
    configs: list[tuple[str, object]] = []
    configs.append(("HAR", HARForecaster(ridge=False)))
    for cell in CELLS:
        for h in HIDDENS:
            for nl in N_LAYERS:
                for lb in LOOKBACKS:
                    tag = f"{cell}_h{h}_l{nl}_lb{lb:02d}"
                    configs.append((
                        tag,
                        RNNForecaster(
                            cell        = cell,
                            hidden_size = h,
                            n_layers    = nl,
                            lookback    = lb,
                            n_epochs    = 300,
                            patience    = 20,
                            val_frac    = 0.20,
                            lr          = 1e-3,
                            batch_size  = 64,
                            seed        = 42,
                        )
                    ))
    return configs


def run_one_symbol(symbol: str) -> list[dict]:
    """Returns a list of row-dicts {symbol, config, mse_h1, …, mse_avg}."""
    log_values, dates = load_rv(CSV, symbol=symbol, target="rv5")
    T = len(log_values)
    if T < WINDOW + max(HORIZONS) + 2:
        print(f"  Skipping {symbol}: only {T} observations")
        return []

    configs = build_grid()
    n_rnn   = len(configs) - 1
    print(f"  {symbol}  T={T}  configs={len(configs)} "
          f"(RNN={n_rnn}: cells={CELLS}, hidden={HIDDENS}, "
          f"layers={N_LAYERS}, lookbacks={LOOKBACKS})\n")

    records = []
    t0 = time.time()

    for i, (tag, model) in enumerate(configs):
        t1 = time.time()
        losses  = quick_oos(log_values, dates, {tag: model})
        results = mean_losses(losses)
        row = {"symbol": symbol, "config": tag}
        for h in HORIZONS:
            row[f"mse_h{h}"] = results[tag].get(h, np.nan)
        row["mse_avg"] = float(np.nanmean([results[tag].get(h, np.nan)
                                           for h in HORIZONS]))
        records.append(row)
        elapsed = time.time() - t1
        print(f"  [{symbol}] [{i+1:3d}/{len(configs)}] {tag:<30s} "
              f"h1={row['mse_h1']:.4f}  h5={row['mse_h5']:.4f}  "
              f"h22={row['mse_h22']:.4f}  avg={row['mse_avg']:.4f}  "
              f"({elapsed:.1f}s)")

    print(f"\n  [{symbol}] Total time: {(time.time()-t0)/60:.1f} min")
    return records


def print_ablation(df_rnn: pd.DataFrame, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"\n── Best per cell{tag} ───────────────────────────────────")
    for cell in CELLS:
        sub = df_rnn[df_rnn["config"].str.startswith(cell)]
        if sub.empty:
            continue
        best = sub.sort_values("mse_avg").iloc[0]
        print(f"  {cell}: {best['config']}  avg={best['mse_avg']:.4f}  "
              f"h1={best['mse_h1']:.4f}  h5={best['mse_h5']:.4f}  "
              f"h22={best['mse_h22']:.4f}")

    print(f"\n── Avg MSE by hidden size{tag} ──")
    for hid in HIDDENS:
        sub = df_rnn[df_rnn["config"].str.contains(f"_h{hid}_")]
        if not sub.empty:
            print(f"  hidden={hid:3d}: {sub['mse_avg'].mean():.4f}")

    print(f"\n── Avg MSE by n_layers{tag} ──")
    for nl in N_LAYERS:
        sub = df_rnn[df_rnn["config"].str.contains(f"_l{nl}_")]
        if not sub.empty:
            print(f"  layers={nl}: {sub['mse_avg'].mean():.4f}")

    print(f"\n── Avg MSE by lookback{tag} ──")
    for lb in LOOKBACKS:
        sub = df_rnn[df_rnn["config"].str.contains(f"_lb{lb:02d}")]
        if not sub.empty:
            print(f"  lookback={lb:2d}: {sub['mse_avg'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="RNN architecture sweep (multi-symbol)."
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
            records = run_one_symbol(symbol)
        except ValueError as e:
            print(f"  Error: {e}")
            continue
        if not records:
            continue
        all_records.extend(records)

        df_sym = pd.DataFrame(records).sort_values("mse_avg")
        results_sym = {r["config"]: {h: r[f"mse_h{h}"] for h in HORIZONS}
                       for _, r in df_sym.iterrows()}
        print_table(results_sym, HORIZONS,
                    title=f"RNN sweep — {symbol} rv5")

        rnn_df = df_sym[df_sym["config"] != "HAR"].copy()
        print_ablation(rnn_df, label=symbol)

    if not all_records:
        print("\nNo results to save.")
        return

    df_all = pd.DataFrame(all_records)

    # ── aggregate over symbols (if multiple) ─────────────────────────────
    if len(symbols) > 1:
        agg = (
            df_all.groupby("config")[["mse_h1", "mse_h5", "mse_h22", "mse_avg"]]
            .mean()
            .sort_values("mse_avg")
        )
        print(f"\n{'═'*60}")
        print(f"  AGGREGATE (mean across {len(symbols)} symbols)")
        print(f"{'═'*60}")
        print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

        rnn_agg = agg[agg.index != "HAR"].reset_index()
        if "index" in rnn_agg.columns:
            rnn_agg = rnn_agg.rename(columns={"index": "config"})
        print_ablation(rnn_agg, label="all symbols")

    out = ROOT / "experiments" / "results_rnn.csv"
    df_all.to_csv(out, index=False)
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
