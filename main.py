"""
main.py — Univariate financial time series forecasting benchmark.

Compares Stat (HAR, GARCH), ML (HAR-Ridge, NLinear, DLinear),
RC (SAS-diagonal) and DL (LSTM / GRU / LRU) models on daily realised
volatility data.

Evaluation tables (in order)
-----------------------------
1. MSE / OOS-Var  : variance-normalised mean ± std across symbols.
2. MCS frequency  : how often each model survives the Model Confidence Set
                    (Hansen et al. 2011) across symbols × horizons.
3. Beats HAR      : step-level fraction of time steps each model beats HAR
                    in squared error, averaged across symbols.
4. Per-horizon    : per-symbol RMSE and MCS survivors.

All preprocessing is done inside each model class.  The OOS loop passes raw
RV (default) or log RV (--log) unchanged and compares predictions in the
same scale.

Usage
-----
    python main.py                          # all symbols, raw RV
    python main.py --log                    # work on log RV
    python main.py --symbol .FTSE
    python main.py --symbol .AEX .SPX .FTSE
    python main.py --horizons 1 5 22 --window 750
    python main.py --no-rnn --no-garch
    python main.py --sas-n 200
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

sys.path.insert(0, str(ROOT))
from utils import (
    run_oos,
    print_precision_table,
    print_mcs_frequency,
    print_beats_benchmark,
    print_per_horizon_scoring,
)


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Univariate RV forecasting benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",     default="rv.csv")
    p.add_argument("--symbol",  nargs="+", default=["all"])
    p.add_argument("--target",  default="rv5",
                   help="RV column: rv5, rk_parzen, bv, rv10, …")
    p.add_argument("--log",     action="store_true",
                   help="Pass log-RV to models (raw RV by default).")
    p.add_argument("--horizons",    nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--window",      type=int, default=2000)
    p.add_argument("--refit-freq",  type=int, default=20)
    # model toggles
    p.add_argument("--no-har",      action="store_true")
    p.add_argument("--no-nlinear",  action="store_true")
    p.add_argument("--no-dlinear",  action="store_true")
    p.add_argument("--no-garch",    action="store_true")
    p.add_argument("--no-sas",      action="store_true")
    p.add_argument("--no-rnn",      action="store_true")
    # model hyperparams
    p.add_argument("--lookback",      type=int,   default=20)
    p.add_argument("--dlinear-kernel",type=int,   default=5)
    p.add_argument("--garch-ar-lags", type=int,   default=5)
    p.add_argument("--sas-n",         type=int,   default=200)
    p.add_argument("--sas-washout",   type=int,   default=50)
    p.add_argument("--sas-p-degree",  type=int,   default=1)
    p.add_argument("--sas-chunk",     type=int,   default=64)
    p.add_argument("--rnn-cell",    default="gru", choices=["lstm", "gru", "lru"])
    p.add_argument("--rnn-hidden",  type=int,   default=32)
    p.add_argument("--rnn-layers",  type=int,   default=1)
    p.add_argument("--rnn-epochs",  type=int,   default=500)
    p.add_argument("--rnn-lr",      type=float, default=1e-3)
    # evaluation
    p.add_argument("--mcs-alpha",   type=float, default=0.10)
    p.add_argument("--mcs-nboot",   type=int,   default=1000)
    p.add_argument("--no-beats",    action="store_true",
                   help="Skip the beats-HAR table.")
    p.add_argument("--no-detail",   action="store_true",
                   help="Skip the per-horizon detail table.")
    # output
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args(argv)


# ── model registry ────────────────────────────────────────────────────────────

def build_models(args: argparse.Namespace) -> dict:
    from models.linear import HARForecaster, NLinearForecaster, DLinearForecaster
    models: dict = {}

    if not args.no_har:
        models["HAR"]     = HARForecaster(ridge=False)
    if not args.no_nlinear:
        models["NLinear"] = NLinearForecaster(lookback=args.lookback)
    if not args.no_dlinear:
        models["DLinear"] = DLinearForecaster(
            lookback=args.lookback, ma_kernel=args.dlinear_kernel)

    if not args.no_garch:
        try:
            from models.garch import GARCHForecaster
            models["GARCH"] = GARCHForecaster(p_ar=args.garch_ar_lags)
        except Exception as e:
            print(f"  [skip GARCH] {e}")

    if not args.no_sas:
        try:
            from models.sas import SASForecaster
            for q in [1, 2]:
                models[f"SAS_q{q}"] = SASForecaster(
                    n_reservoir   = args.sas_n,
                    basis         = "diagonal",
                    spectral_norm = 0.95,
                    p_degree      = args.sas_p_degree,
                    q_degree      = q,
                    washout       = args.sas_washout,
                    chunk_size    = args.sas_chunk,
                    seed          = args.seed,
                    apply_log     = False,
                    target_log    = False,
                    clip          = False,
                    input_scale   = 1.0,
                )
        except Exception as e:
            print(f"  [skip SAS] {e}")

    if not args.no_rnn:
        try:
            from models.rnn import RNNForecaster
            models[args.rnn_cell.upper()] = RNNForecaster(
                cell        = args.rnn_cell,
                hidden_size = args.rnn_hidden,
                n_layers    = args.rnn_layers,
                lookback    = args.lookback,
                n_epochs    = args.rnn_epochs,
                lr          = args.rnn_lr,
                seed        = args.seed,
            )
        except ImportError as e:
            print(f"  [skip RNN] {e}")

    return models


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    args    = parse_args(argv)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from data.data_loader import load_rv, available_symbols
    csv_path = ROOT / args.csv
    symbols  = available_symbols(csv_path) if args.symbol == ["all"] else args.symbol

    print(f"\nBenchmark: {len(symbols)} symbol(s), "
          f"target={args.target}, window={args.window}, "
          f"horizons={args.horizons}")

    evals_by_symbol:  dict[str, pd.DataFrame]               = {}
    mse_by_symbol:    dict[str, dict[str, dict[int, float]]] = {}
    qlike_by_symbol:  dict[str, dict[str, dict[int, float]]] = {}

    pbar = tqdm(symbols, unit="symbol", desc="OOS eval")
    for symbol in pbar:
        pbar.set_postfix(sym=symbol)
        t0 = time.time()
        try:
            values, dates = load_rv(
                csv_path, symbol=symbol,
                target=args.target, log_transform=args.log,
            )
        except ValueError as e:
            tqdm.write(f"  [skip {symbol}] {e}")
            continue

        if len(values) < 2 * args.window:
            tqdm.write(f"  [skip {symbol}] T={len(values)} < 2×window")
            continue

        models  = build_models(args)
        if not models:
            sys.exit("No models enabled.")

        df_eval = run_oos(
            values, dates, models,
            horizons   = args.horizons,
            window     = args.window,
            refit_freq = args.refit_freq,
            log_mode   = args.log,
            verbose    = False,
        )
        df_eval["symbol"] = symbol

        # ── per-symbol losses (for precision + DM tables) ────────────────
        sym_mse:   dict[str, dict[int, float]] = {}
        sym_qlike: dict[str, dict[int, float]] = {}
        for (cfg, h), grp in df_eval.groupby(["config", "horizon"]):
            sym_mse.setdefault(cfg, {})[h]   = float(grp["sq_err"].mean())
            sym_qlike.setdefault(cfg, {})[h] = float(grp["qlike"].mean())
        mse_by_symbol[symbol]   = sym_mse
        qlike_by_symbol[symbol] = sym_qlike
        evals_by_symbol[symbol]  = df_eval

        df_eval.to_csv(out_dir / f"errors_{symbol.lstrip('.')}.csv", index=False)
        tqdm.write(f"  ✓ {symbol:<10} T={len(values):<5}  {time.time()-t0:.1f}s")

    if not evals_by_symbol:
        sys.exit("No usable symbols.")

    model_order = list(next(iter(evals_by_symbol.values()))["config"].unique())

    # ── TABLE 1: MSE + QLIKE ─────────────────────────────────────────────
    print_precision_table(
        mse_by_symbol,
        qlike_by_symbol,
        horizons    = args.horizons,
        title       = f"Precision  (target={args.target}, window={args.window})",
        model_order = model_order,
    )

    # ── TABLE 2: MCS frequency ────────────────────────────────────────────
    mcs_df = print_mcs_frequency(
        evals_by_symbol,
        horizons     = args.horizons,
        alpha        = args.mcs_alpha,
        n_boot       = args.mcs_nboot,
        seed         = args.seed,
        mse_by_symbol= mse_by_symbol,
        model_order  = model_order,
    )
    if not mcs_df.empty:
        mcs_df.to_csv(out_dir / "mcs_frequency.csv", index=False)

    # ── TABLE 3: DM test vs HAR ───────────────────────────────────────────
    if not args.no_beats and "HAR" in model_order:
        beats_df = print_beats_benchmark(
            evals_by_symbol,
            horizons    = args.horizons,
            benchmark   = "HAR",
            alpha       = args.mcs_alpha,
            model_order = model_order,
        )
        if not beats_df.empty:
            beats_df.to_csv(out_dir / "beats_har.csv", index=False)

    # ── TABLE 4: Per-horizon detail ───────────────────────────────────────
    if not args.no_detail:
        print_per_horizon_scoring(
            evals_by_symbol,
            horizons    = args.horizons,
            alpha       = args.mcs_alpha,
            model_order = model_order,
        )

    pd.concat(evals_by_symbol.values(), ignore_index=True).to_csv(
        out_dir / "errors_all.csv", index=False
    )
    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
