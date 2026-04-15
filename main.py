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
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent


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
    p.add_argument("--refit-freq", type=int, default=252,
                   help="Refit every N steps (252 ≈ 1 year)")

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
    p.add_argument("--sas-n",        type=int,   default=100)
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
        except ImportError as e:
            print(f"  [skip GARCH] {e}")

    if not args.no_sas:
        try:
            from models.sas import SASForecaster
            for sn, label in [(0.90, "SAS_90"), (0.95, "SAS_95")]:
                models[label] = SASForecaster(
                    n_reservoir   = args.sas_n,
                    basis         = "diagonal",
                    spectral_norm = sn,
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

    print(f"  OOS: {dates[window].date()} → {dates[T - H_max - 1].date()}")
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
                    print(f"    [fit {name} @ {dates[t].date()}] {e}")

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
                    print(f"    [predict {name} h={h} @ {dates[t].date()}] {e}")

        for name, model in models.items():
            try:
                model.update(z_t)
                steps_since_refit[name] += 1
            except Exception as e:
                print(f"    [update {name} @ {dates[t].date()}] {e}")

    return pd.DataFrame(records)


# ── results display ──────────────────────────────────────────────────────────

def _mse_per_model(df_eval: pd.DataFrame, horizons: list[int]) -> dict[str, dict[int, float]]:
    """Compute mean MSE per (model, horizon) from one symbol's eval DataFrame."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for (cfg, h), grp in df_eval.groupby(["config", "horizon"]):
        if h in horizons:
            out[cfg][h] = float(grp["sq_err"].mean())
    return dict(out)


def print_mse_table(
    mse_by_symbol: dict[str, dict[str, dict[int, float]]],
    horizons:      list[int],
    title:         str = "",
) -> pd.DataFrame:
    """
    Print a mean ± std MSE table across symbols.

    mse_by_symbol : {symbol: {model: {h: mse}}}

    Returns the underlying DataFrame with columns
        [config, h{k}_mean, h{k}_std, avg_mean, avg_std]
    """
    # Collect all models that appear in at least one symbol
    all_models: list[str] = []
    seen: set[str] = set()
    for sym_mse in mse_by_symbol.values():
        for cfg in sym_mse:
            if cfg not in seen:
                all_models.append(cfg)
                seen.add(cfg)

    # {model: {h: [mse_sym1, mse_sym2, ...]}}
    agg: dict[str, dict[int, list[float]]] = {
        m: {h: [] for h in horizons} for m in all_models
    }
    for sym_mse in mse_by_symbol.values():
        for m, h_dict in sym_mse.items():
            for h in horizons:
                if h in h_dict:
                    agg[m][h].append(h_dict[h])

    rows = []
    for m in all_models:
        row: dict = {"config": m}
        avgs = []
        for h in horizons:
            vals = agg[m][h]
            mu   = float(np.mean(vals)) if vals else np.nan
            sd   = float(np.std(vals,  ddof=min(1, len(vals) - 1))) if len(vals) > 1 else np.nan
            row[f"h{h}_mean"] = mu
            row[f"h{h}_std"]  = sd
            if not np.isnan(mu):
                avgs.append(mu)
        row["avg_mean"] = float(np.mean(avgs)) if avgs else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("avg_mean")
    n_sym = len(mse_by_symbol)
    N_str = f"N={n_sym} symbol{'s' if n_sym > 1 else ''}"

    w = max(len(r["config"]) for r in rows) + 2
    sep = "─" * (w + len(horizons) * 18 + 10)

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
            if np.isnan(mu):
                cell = "     —"
            elif np.isnan(sd):
                cell = f"{mu:7.4f}          "
            else:
                cell = f"{mu:7.4f}±{sd:6.4f}"
            line += f"  {cell:>16}"
        avg = r["avg_mean"]
        line += f"  {avg:8.4f}" if np.isfinite(avg) else f"  {'—':>8}"
        print(line)
    print(sep)
    return df


# ── MCS frequency table ──────────────────────────────────────────────────────

def _run_mcs_one_symbol(
    df_eval:    pd.DataFrame,
    horizons:   list[int],
    alpha:      float,
    n_boot:     int,
    seed:       int,
) -> dict[int, list[str]]:
    """
    Run per-horizon MCS for one symbol's eval DataFrame.

    Returns {horizon: [models_in_mcs]}.
    """
    from utils.mcs import model_confidence_set

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

        # Row-normalise so high-vol periods don't dominate bootstrap variance
        row_mean = pivot.mean(axis=1).clip(lower=1e-12)
        pivot_n  = pivot.div(row_mean, axis=0)

        mcs_set, _ = model_confidence_set(
            pivot_n, alpha=alpha, n_boot=n_boot, seed=seed,
        )
        result[h] = mcs_set
    return result


def print_mcs_frequency(
    evals_by_symbol: dict[str, pd.DataFrame],
    horizons:        list[int],
    alpha:           float,
    n_boot:          int,
    seed:            int,
) -> pd.DataFrame:
    """
    Run the MCS independently for each symbol × horizon, then print a
    frequency table: for each (model, horizon), how many symbols did that
    model survive into the MCS best set?

    This cross-asset frequency is robust to TS heterogeneity — a model
    consistently in the MCS across many assets is genuinely best-in-class.

    Returns a DataFrame with columns [config, h{k}_count, h{k}_pct, total_count].
    """
    n_sym = len(evals_by_symbol)
    symbols = list(evals_by_symbol.keys())

    print(f"\n  Running MCS (α={alpha}, B={n_boot}) for each of {n_sym} symbols …")

    # {model: {h: count_in_mcs}}
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    valid:  dict[int, int]            = defaultdict(int)   # how many symbols had enough data

    for sym in symbols:
        mcs_by_h = _run_mcs_one_symbol(
            evals_by_symbol[sym], horizons, alpha, n_boot, seed
        )
        for h, mcs_set in mcs_by_h.items():
            if mcs_set:
                valid[h] += 1
                for m in mcs_set:
                    counts[m][h] += 1

    if not counts:
        print("  No MCS results.")
        return pd.DataFrame()

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
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("total_count", ascending=False)

    # ── pretty print ─────────────────────────────────────────────────────
    w   = max(len(r["config"]) for r in rows) + 2
    sep = "─" * (w + len(horizons) * 14 + 10)

    print(f"\n{sep}")
    print(f"  MCS frequency table  (α={alpha}, N={n_sym} symbols, row-normalised losses)")
    print(f"  Entry = count/N symbols where model survived into best set")
    print(sep)
    header = f"{'Model':<{w}}" + "".join(
        f"{'h='+str(h)+'  ('+str(valid[h])+')':>14}" for h in horizons
    ) + f"{'total':>8}"
    print(header)
    print(sep)
    for _, r in df.iterrows():
        line = f"{r['config']:<{w}}"
        for h in horizons:
            c, nv = r[f"h{h}_count"], valid[h]
            if nv > 0:
                cell = f"{c}/{nv} ({100*c/nv:.0f}%)"
            else:
                cell = "—"
            line += f"  {cell:>12}"
        line += f"  {int(r['total_count']):>6}"
        print(line)
    print(sep)
    return df


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
    evals_by_symbol: dict[str, pd.DataFrame] = {}
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
        print(f"  {dates[0].date()} → {dates[-1].date()},  T={T}")

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

        # Save per-symbol errors
        sym_tag = symbol.lstrip(".")
        df_eval.to_csv(out_dir / f"errors_{sym_tag}.csv", index=False)
        print(f"  Saved {len(df_eval):,} records → errors_{sym_tag}.csv")

        evals_by_symbol[symbol] = df_eval
        mse_by_symbol[symbol]   = _mse_per_model(df_eval, args.horizons)

    if not evals_by_symbol:
        sys.exit("No usable symbols — check data and --window setting.")

    # ── save combined errors ──────────────────────────────────────────────
    df_all = pd.concat(evals_by_symbol.values(), ignore_index=True)
    df_all.to_csv(out_dir / "errors_all.csv", index=False)

    # ── MSE table: mean ± std across symbols ─────────────────────────────
    n_sym = len(evals_by_symbol)
    print_mse_table(
        mse_by_symbol,
        horizons = args.horizons,
        title    = f"MSE  (mean ± std, target={args.target}, window={args.window})",
    )

    # If only one symbol, also print a single-symbol summary with MAE/MDA
    if n_sym == 1:
        from utils.metrics import summary_table
        df_one = next(iter(evals_by_symbol.values()))
        tbl = summary_table(df_one, metrics=["mse", "mae", "mda"])
        for metric in ["mse", "mae", "mda"]:
            lvl = tbl.columns.get_level_values("metric")
            if metric not in lvl:
                continue
            sub = tbl[metric].sort_index()
            fmt = "{:.4f}".format if metric in ("mse", "mae") else "{:.3f}".format
            print(f"\n── {metric.upper()} ─────────────────────────────────")
            print(sub.to_string(float_format=fmt))

    # ── MCS frequency table ───────────────────────────────────────────────
    mcs_df = print_mcs_frequency(
        evals_by_symbol,
        horizons = args.horizons,
        alpha    = args.mcs_alpha,
        n_boot   = args.mcs_nboot,
        seed     = args.seed,
    )
    if not mcs_df.empty:
        mcs_df.to_csv(out_dir / "mcs_frequency.csv", index=False)
        print(f"  MCS frequency table → {out_dir / 'mcs_frequency.csv'}")

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
