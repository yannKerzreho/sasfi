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

Usage
-----
    python main.py                                          # all symbols
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

    # ── data ─────────────────────────────────────────────────────────────
    p.add_argument("--csv",    default="rv.csv")
    p.add_argument(
        "--symbol", nargs="+", default=["all"],
        help=(
            "Symbol(s) to analyse, e.g. .AEX .SPX, or 'all' for every "
            "symbol in the CSV. Default: all."
        ),
    )
    p.add_argument("--target", default="rv5",
                   help="RV column: rv5, rk_parzen, bv, rv10, …")
    p.add_argument("--no-log", action="store_true",
                   help="Skip log-transform (not recommended)")

    # ── evaluation ───────────────────────────────────────────────────────
    p.add_argument("--horizons",   nargs="+", type=int, default=[1, 5, 22])
    p.add_argument("--window",     type=int, default=2000)
    p.add_argument("--refit-freq", type=int, default=20,
                   help="Refit every N steps (20 ≈ 1 months)")

    # ── model toggles ────────────────────────────────────────────────────
    p.add_argument("--no-har",     action="store_true")
    p.add_argument("--no-ridge",   action="store_true")
    p.add_argument("--no-nlinear", action="store_true")
    p.add_argument("--no-dlinear", action="store_true")
    p.add_argument("--no-garch",   action="store_true")
    p.add_argument("--no-sas",     action="store_true")
    p.add_argument("--no-rnn",     action="store_true")

    # ── HAR / NLinear / DLinear ───────────────────────────────────────────
    p.add_argument("--lookback", type=int, default=20,
                   help="Window length for NLinear / DLinear / RNN")
    p.add_argument("--dlinear-kernel", type=int, default=5)

    # ── GARCH ────────────────────────────────────────────────────────────
    p.add_argument("--garch-ar-lags", type=int, default=5,
                   help="AR lags in GARCH mean equation")

    # ── SAS ──────────────────────────────────────────────────────────────
    p.add_argument("--sas-n",        type=int,   default=100,
                   help="Base reservoir size n (SAS_aug=n, SAS_diag=2n)")
    p.add_argument("--sas-washout",  type=int,   default=50)
    p.add_argument("--sas-p-degree", type=int,   default=1)
    p.add_argument("--sas-q-degree", type=int,   default=1)
    p.add_argument("--sas-chunk",    type=int,   default=64)

    # ── RNN ──────────────────────────────────────────────────────────────
    p.add_argument("--rnn-cell",   default="gru",
                   choices=["lstm", "gru", "lru"])
    p.add_argument("--rnn-hidden", type=int,   default=32)
    p.add_argument("--rnn-layers", type=int,   default=1)
    p.add_argument("--rnn-epochs", type=int,   default=500)
    p.add_argument("--rnn-lr",     type=float, default=1e-3)

    # ── MCS ──────────────────────────────────────────────────────────────
    p.add_argument("--mcs-alpha", type=float, default=0.10)
    p.add_argument("--mcs-nboot", type=int,   default=1000)

    # ── output ───────────────────────────────────────────────────────────
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed",    type=int, default=42)

    return p.parse_args(argv)


# ── model registry ──────────────────────────────────────────────────────────

def build_models(args: argparse.Namespace) -> dict:
    from models.linear import HARForecaster, NLinearForecaster, DLinearForecaster
    models: dict = {}

    if not args.no_har:
        models["HAR"]       = HARForecaster(ridge=False)
    if not args.no_ridge:
        models["HAR_Ridge"] = HARForecaster(ridge=True)
    if not args.no_nlinear:
        models["NLinear"]   = NLinearForecaster(lookback=args.lookback)
    if not args.no_dlinear:
        models["DLinear"]   = DLinearForecaster(lookback=args.lookback,
                                                ma_kernel=args.dlinear_kernel)

    if not args.no_garch:
        try:
            from models.garch import GARCHForecaster
            models["GARCH"] = GARCHForecaster(p_ar=args.garch_ar_lags)
        except (ImportError, Exception) as e:
            print(f"  [skip GARCH] {e}")

    if not args.no_sas:
        try:
            from models.sas import SASForecaster, AugSASForecaster
            # Best configs from exp_sas sweep (window=2000, refit=20, 30 symbols):
            #   diag_n200_sn0.95  — 89/90 MCS, stable, best mean MSE among SAS
            #   aug_n100_sn0.90   — 87/90 MCS, reservoir + HAR features,
            #                       guaranteed ≥ HAR quality by construction,
            #                       guards against h=22 ridge instability
            # Excluded: diag_n100_sn0.90/0.95 — high CV (std/mean > 0.5 at h=1/h=22)
            models["SAS_diag"] = SASForecaster(
                n_reservoir   = args.sas_n * 2,   # 200 by default
                basis         = "diagonal",
                spectral_norm = 0.95,
                p_degree      = args.sas_p_degree,
                q_degree      = args.sas_q_degree,
                washout       = args.sas_washout,
                chunk_size    = args.sas_chunk,
                seed          = args.seed,
            )
            models["SAS_aug"] = AugSASForecaster(
                n_reservoir   = args.sas_n,        # 100 by default
                basis         = "diagonal",
                spectral_norm = 0.90,
                p_degree      = args.sas_p_degree,
                q_degree      = args.sas_q_degree,
                washout       = args.sas_washout,
                chunk_size    = args.sas_chunk,
                seed          = args.seed,
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


# ── date formatting helper ────────────────────────────────────────────────────

def _fmt(d) -> str:
    """Return YYYY-MM-DD string from any date-like object (Timestamp or str)."""
    try:
        return d.date().isoformat()
    except AttributeError:
        return str(d)[:10]


# ── rolling OOS loop ─────────────────────────────────────────────────────────

def run_oos(
    log_values: np.ndarray,
    dates:      pd.DatetimeIndex,
    models:     dict,
    horizons:   list[int],
    window:     int,
    refit_freq: int,
) -> pd.DataFrame:
    """
    Rolling OOS with periodic refit.

    Per step t:
      1. Refit all models (if refit_freq steps elapsed) on the z-scored window.
      2. Predict every horizon h; record (y_pred, y_true, sq_err, abs_err).
      3. model.update(z_t) — advance state without refit.

    Returns a DataFrame with columns:
        config, horizon, test_date, y_pred, y_true, sq_err, abs_err
    """
    from data.data_loader import fit_scaler, apply_scaler

    T     = len(log_values)
    H_max = max(horizons)
    mu, sigma = 0.0, 1.0
    steps_since_refit: dict[str, int] = {n: refit_freq for n in models}
    records: list[dict] = []

    print(f"  OOS: {_fmt(dates[window])} → {_fmt(dates[T - H_max - 1])}")
    print(f"       window={window}, refit_freq={refit_freq}, "
          f"models: {list(models)}")

    for t in range(window, T - H_max):
        train_raw = log_values[t - window: t]

        for name, model in models.items():
            if steps_since_refit[name] >= refit_freq:
                mu, sigma = fit_scaler(train_raw)
                train_z   = apply_scaler(train_raw, mu, sigma)
                try:
                    model.fit(train_z, horizons)
                    steps_since_refit[name] = 0
                except Exception as e:
                    print(f"    [fit {name} @ {_fmt(dates[t])}] {e}")

        z_t = apply_scaler(float(log_values[t]), mu, sigma)

        for name, model in models.items():
            for h in horizons:
                if t + h >= T:
                    continue
                z_target = apply_scaler(float(log_values[t + h]), mu, sigma)
                try:
                    y_hat = float(model.predict(h))
                    records.append(dict(
                        config    = name,
                        horizon   = h,
                        test_date = dates[t],
                        y_pred    = y_hat,
                        y_true    = z_target,
                        sq_err    = (y_hat - z_target) ** 2,
                        abs_err   = abs(y_hat - z_target),
                    ))
                except Exception as e:
                    print(f"    [predict {name} h={h} @ {_fmt(dates[t])}] {e}")

        for name, model in models.items():
            try:
                model.update(z_t)
                steps_since_refit[name] += 1
            except Exception as e:
                print(f"    [update {name} @ {_fmt(dates[t])}] {e}")

    return pd.DataFrame(records)


# ── results helpers ──────────────────────────────────────────────────────────

def _mse_per_model(df_eval: pd.DataFrame, horizons: list[int]) -> dict[str, dict[int, float]]:
    """Compute mean MSE per (model, horizon) from one symbol's eval DataFrame."""
    out: dict[str, dict[int, float]] = {}
    for (cfg, h), grp in df_eval.groupby(["config", "horizon"]):
        if h in horizons:
            out.setdefault(cfg, {})[h] = float(grp["sq_err"].mean())
    return out


# ── entry point ──────────────────────────────────────────────────────────────

def main(argv=None):
    args    = parse_args(argv)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from data.data_loader import load_rv, available_symbols

    csv_path = ROOT / args.csv

    # ── resolve symbols ───────────────────────────────────────────────────
    if args.symbol == ["all"]:
        symbols = available_symbols(csv_path)
        print(f"Running on all {len(symbols)} available symbols: {symbols}\n")
    else:
        symbols = args.symbol
        print(f"Running on {len(symbols)} symbol(s): {symbols}\n")

    # ── per-symbol OOS ────────────────────────────────────────────────────
    evals_by_symbol: dict[str, pd.DataFrame]               = {}
    mse_by_symbol:   dict[str, dict[str, dict[int, float]]] = {}

    for symbol in symbols:
        print(f"\n{'═'*62}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'═'*62}")

        try:
            log_values, dates = load_rv(
                csv_path,
                symbol        = symbol,
                target        = args.target,
                log_transform = not args.no_log,
            )
        except ValueError as e:
            print(f"  Error loading data: {e}")
            continue

        T = len(log_values)
        if T < args.window + max(args.horizons) + 2:
            print(f"  Skipping: T={T} < window+horizon ({args.window + max(args.horizons) + 2})")
            continue
        print(f"  {_fmt(dates[0])} → {_fmt(dates[-1])},  T={T}")

        models = build_models(args)
        if not models:
            sys.exit("No models enabled.")

        df_eval = run_oos(
            log_values = log_values,
            dates      = dates,
            models     = models,
            horizons   = args.horizons,
            window     = args.window,
            refit_freq = args.refit_freq,
        )
        df_eval["symbol"] = symbol

        sym_tag = symbol.lstrip(".")
        df_eval.to_csv(out_dir / f"errors_{sym_tag}.csv", index=False)
        print(f"  Saved {len(df_eval):,} records → errors_{sym_tag}.csv")

        evals_by_symbol[symbol] = df_eval
        mse_by_symbol[symbol]   = _mse_per_model(df_eval, args.horizons)

    if not evals_by_symbol:
        sys.exit("No usable symbols — check data and --window setting.")

    df_all = pd.concat(evals_by_symbol.values(), ignore_index=True)
    df_all.to_csv(out_dir / "errors_all.csv", index=False)

    # ── MSE table: mean ± std across symbols ─────────────────────────────
    n_sym = len(evals_by_symbol)
    print_mse_table(
        mse_by_symbol,
        horizons = args.horizons,
        title    = f"MSE  (mean ± std, target={args.target}, window={args.window})",
    )

    # Single-symbol: also show MAE / MDA
    if n_sym == 1:
        from utils.metrics import summary_table
        df_one = next(iter(evals_by_symbol.values()))
        tbl = summary_table(df_one, metrics=["mse", "mae", "mda"])
        for metric in ["mse", "mae", "mda"]:
            if metric not in tbl.columns.get_level_values("metric"):
                continue
            sub = tbl[metric].sort_index()
            fmt = "{:.4f}".format if metric in ("mse", "mae") else "{:.3f}".format
            print(f"\n── {metric.upper()} ─────────────────────────────────")
            print(sub.to_string(float_format=fmt))

    # ── MCS frequency table ───────────────────────────────────────────────
    mcs_df = print_mcs_frequency(
        evals_by_symbol,
        horizons      = args.horizons,
        alpha         = args.mcs_alpha,
        n_boot        = args.mcs_nboot,
        seed          = args.seed,
        mse_by_symbol = mse_by_symbol,
    )
    if not mcs_df.empty:
        mcs_df.to_csv(out_dir / "mcs_frequency.csv", index=False)
        print(f"  MCS frequency table → {out_dir / 'mcs_frequency.csv'}")

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
