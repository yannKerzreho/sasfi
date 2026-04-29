"""
main.py — Univariate financial time series forecasting benchmark.

Compares Stat (HAR, GARCH), ML (HAR-Ridge, NLinear, DLinear),
RC (SAS-diagonal) and DL (LSTM / GRU / LRU) models on daily realised
volatility data.

Evaluation: rolling OOS → MSE table (mean ± std across symbols) +
            Model Confidence Set frequency table (Hansen et al. 2011).

The MCS is run independently on each symbol, then the frequency table
counts how many symbols each model survived into the best set — a robust
cross-asset ranking that is not sensitive to the variance of any single TS.

All preprocessing (z-scoring, log-transform, scaling) is done inside each
model.  The OOS loop passes raw RV (default) or log RV (--log) unchanged
and compares predictions directly against the same-scale targets.

Usage
-----
    python main.py                                          # all symbols, raw RV
    python main.py --log                                    # work on log RV
    python main.py --symbol .FTSE                          # one symbol
    python main.py --symbol .AEX .SPX .FTSE                # subset
    python main.py --symbol .AEX --target rk_parzen
    python main.py --horizons 1 5 22 --window 750
    python main.py --rnn-cell gru
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
from tqdm.auto import trange

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

sys.path.insert(0, str(ROOT))
from experiments.utils import print_mse_table, print_mcs_frequency


# ── argument parsing ────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Univariate RV forecasting benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--csv",    default="rv.csv")
    p.add_argument("--symbol", nargs="+", default=["all"])
    p.add_argument("--target", default="rv5")
    p.add_argument("--log", action="store_true")
    p.add_argument("--horizons",   nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--window",     type=int, default=2000)
    p.add_argument("--refit-freq", type=int, default=20)
    p.add_argument("--no-har",     action="store_true")
    p.add_argument("--no-nlinear", action="store_true")
    p.add_argument("--no-dlinear", action="store_true")
    p.add_argument("--no-garch",   action="store_true")
    p.add_argument("--no-sas",     action="store_true")
    p.add_argument("--no-rnn",     action="store_true")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--dlinear-kernel", type=int, default=5)
    p.add_argument("--garch-ar-lags", type=int, default=5)
    p.add_argument("--sas-n",        type=int,   default=200)
    p.add_argument("--sas-washout",  type=int,   default=50)
    p.add_argument("--sas-p-degree", type=int,   default=1)
    p.add_argument("--sas-chunk",    type=int,   default=64)
    p.add_argument("--rnn-cell",   default="gru", choices=["lstm", "gru", "lru"])
    p.add_argument("--rnn-hidden", type=int,   default=32)
    p.add_argument("--rnn-layers", type=int,   default=1)
    p.add_argument("--rnn-epochs", type=int,   default=500)
    p.add_argument("--rnn-lr",     type=float, default=1e-3)
    p.add_argument("--mcs-alpha", type=float, default=0.10)
    p.add_argument("--mcs-nboot", type=int,   default=1000)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed",    type=int, default=42)

    return p.parse_args(argv)


# ── model registry ──────────────────────────────────────────────────────────

def build_models(args: argparse.Namespace) -> dict:
    from models.linear import HARForecaster, NLinearForecaster, DLinearForecaster
    models: dict = {}

    if not args.no_har:
        models["HAR"]       = HARForecaster(ridge=False)
    if not args.no_nlinear:
        models["NLinear"]   = NLinearForecaster(lookback=args.lookback)
    if not args.no_dlinear:
        models["DLinear"]   = DLinearForecaster(lookback=args.lookback, ma_kernel=args.dlinear_kernel)

    if not args.no_garch:
        try:
            from models.garch import GARCHForecaster
            models["GARCH"] = GARCHForecaster(p_ar=args.garch_ar_lags)
        except: pass

    if not args.no_sas:
        try:
            from models.sas import SASForecaster
            for q in [1, 2]:
                models[f"SAS_q{q}"] = SASForecaster(
                    n_reservoir=args.sas_n, basis="diagonal", spectral_norm=0.95,
                    p_degree=args.sas_p_degree, q_degree=q, washout=args.sas_washout,
                    chunk_size=args.sas_chunk, seed=args.seed, apply_log=False,
                    target_log=False, clip=True, input_scale=4.0
                )
        except: pass

    if not args.no_rnn:
        try:
            from models.rnn import RNNForecaster
            models[args.rnn_cell.upper()] = RNNForecaster(
                cell=args.rnn_cell, hidden_size=args.rnn_hidden, n_layers=args.rnn_layers,
                lookback=args.lookback, n_epochs=args.rnn_epochs, lr=args.rnn_lr, seed=args.seed,
            )
        except: pass

    return models

# ── rolling OOS loop ─────────────────────────────────────────────────────────

def run_oos(values, dates, models, horizons, window, refit_freq):
    T, H_max = len(values), max(horizons)
    steps_since_refit = {n: refit_freq for n in models}
    records = []

    for t in trange(window, T - H_max, desc="  OOS Prog.", leave=False):
        train = values[t - window: t]
        for name, model in models.items():
            if steps_since_refit[name] >= refit_freq:
                try:
                    model.fit(train, horizons)
                    steps_since_refit[name] = 0
                except: pass

        x_t = float(values[t])
        for name, model in models.items():
            try:
                model.update(x_t)
                steps_since_refit[name] += 1
                for h in horizons:
                    y_hat = float(model.predict(h))
                    y_true = float(values[t + h])
                    records.append(dict(
                        config=name, horizon=h, test_date=dates[t],
                        sq_err=(y_hat - y_true)**2
                    ))
            except: pass
    return pd.DataFrame(records)

# ── helpers ──────────────────────────────────────────────────────────────────

def print_per_horizon_scoring(evals_by_symbol, horizons, alpha, out_dir):
    """Prints per-symbol Score (Relative RMSE) and MCS survivors."""
    from arch.bootstrap import MCS
    
    for h in horizons:
        print(f"\n{'═'*80}\n  SCORE (REL. RMSE) & MCS SURVIVORS — HORIZON: {h}\n{'═'*80}")
        rows = []
        for sym, df in evals_by_symbol.items():
            df_h = df[df["horizon"] == h]
            if df_h.empty: continue
            
            models = df_h["config"].unique()
            row = {"Symbol": sym}
            losses_dict = {m: df_h[df_h["config"] == m]["norm_sq_err"].values for m in models}
            
            # Score = Sqrt of Mean Normalized Squared Error
            for m in models:
                row[m] = np.sqrt(np.mean(losses_dict[m]))
                
            # MCS on normalized losses
            try:
                loss_df = pd.DataFrame(losses_dict).dropna()
                mcs = MCS(loss_df, size=alpha, block_size=int(len(loss_df)**(1/3)), method='R')
                mcs.compute()
                survivors = list(mcs.pvalues[mcs.pvalues >= alpha].sort_values(ascending=False).index)
            except: survivors = ["error"]
            
            row["MCS_Survivors"] = ", ".join(survivors)
            rows.append(row)
            
        res_df = pd.DataFrame(rows)
        print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        res_df.to_csv(out_dir / f"score_table_h{h}.csv", index=False)

# ── entry point ──────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from data.data_loader import load_rv, available_symbols
    csv_path = ROOT / args.csv
    symbols = available_symbols(csv_path) if args.symbol == ["all"] else args.symbol
        
    print(f"\nBenchmark: {len(symbols)} symbols, Target={args.target}, Window={args.window}")

    evals_by_symbol = {}
    mse_abs_scaled = {} # MSE / OOS_Var

    for symbol in symbols:
        t0 = time.time()
        try:
            values, dates = load_rv(csv_path, symbol=symbol, target=args.target, log_transform=args.log)
        except: continue

        if len(values) < 2*args.window: continue

        models = build_models(args)
        df_eval = run_oos(values, dates, models, args.horizons, args.window, args.refit_freq)
        df_eval["symbol"] = symbol

        # 1. Row-Normalization for MCS and relative Score
        # We compute this PER SYMBOL so models compete within the asset
        mean_errs = df_eval.groupby(["horizon", "test_date"])["sq_err"].transform("mean")
        df_eval["norm_sq_err"] = df_eval["sq_err"] / mean_errs.replace(0, np.nan)
        df_eval["norm_sq_err"] = df_eval["norm_sq_err"].fillna(1.0)
        
        evals_by_symbol[symbol] = df_eval
        
        # 2. Absolute Scaling (MSE / Var) for precision table
        oos_var = np.var(values[args.window:])
        sym_mse = {}
        for (cfg, h), grp in df_eval.groupby(["config", "horizon"]):
            sym_mse.setdefault(cfg, {})[h] = grp["sq_err"].mean() / oos_var
        mse_abs_scaled[symbol] = sym_mse

        print(f"  ✓ {symbol:<8} | T={len(values):<5} | {time.time()-t0:>4.1f}s")

    if not evals_by_symbol: sys.exit("No results.")

    # TABLE 1: Precision (MSE / OOS Var)
    print_mse_table(mse_abs_scaled, args.horizons, title="Precision: MSE / OOS_Variance (Absolute Scale)")

    # TABLE 2: MCS Frequency (Asset-wise Consistency)
    # We pass mse_abs_scaled just for the avg_mse column in the print helper
    print_mcs_frequency(evals_by_symbol, args.horizons, alpha=args.mcs_alpha, 
                        n_boot=args.mcs_nboot, seed=args.seed, mse_by_symbol=mse_abs_scaled)

    # TABLE 3: Per-Horizon Detail (Score & Survivors)
    print_per_horizon_scoring(evals_by_symbol, args.horizons, args.mcs_alpha, out_dir)

if __name__ == "__main__":
    main()