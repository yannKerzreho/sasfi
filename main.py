"""
main.py — Univariate financial time series forecasting benchmark.

Compares Stat (HAR, GARCH), ML (HAR-Ridge, NLinear, DLinear),
RC (SAS-linear, SAS-diagonal) and DL (LSTM / GRU / LRU) models
on daily realised volatility data.

Evaluation: rolling OOS + Model Confidence Set (Hansen et al. 2011).

Usage
-----
    python main.py                                     # defaults
    python main.py --symbol .FTSE --target rk_parzen
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


# ── argument parsing ────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Univariate RV forecasting benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ─────────────────────────────────────────────────────────────
    p.add_argument("--csv",    default="rv.csv")
    p.add_argument("--symbol", default=".AEX")
    p.add_argument("--target", default="rv5",
                   help="RV column: rv5, rk_parzen, bv, rv10, …")
    p.add_argument("--no-log", action="store_true",
                   help="Skip log-transform (not recommended)")

    # ── evaluation ───────────────────────────────────────────────────────
    p.add_argument("--horizons",   nargs="+", type=int, default=[1, 5, 22])
    p.add_argument("--window",     type=int, default=500)
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
    # Experiment exp_sas.py showed: diagonal basis is the only stable choice.
    # Linear basis overfits catastrophically (h=5 MSE up to 2342×) because
    # n² = 10 000 random params create ill-conditioned state matrices.
    # Best diagonal configs: sn=0.90 for h=1/5, sn=0.95 for h=22.
    # Two variants are always run: (n, sn=0.90) and (n, sn=0.95).
    p.add_argument("--sas-n",        type=int,   default=100,
                   help="Reservoir size for diagonal SAS (100 is sufficient)")
    p.add_argument("--sas-washout",  type=int,   default=50)
    p.add_argument("--sas-p-degree", type=int,   default=1,
                   help="Polynomial degree for P(z) in the SAS recurrence")
    p.add_argument("--sas-q-degree", type=int,   default=1,
                   help="Polynomial degree for Q(z) in the SAS recurrence")
    p.add_argument("--sas-chunk",    type=int,   default=64,
                   help="Chunk size B for the two-level parallel scan")

    # ── RNN ──────────────────────────────────────────────────────────────
    p.add_argument("--rnn-cell",   default="lstm",
                   choices=["lstm", "gru", "lru"])
    p.add_argument("--rnn-hidden", type=int,   default=32)
    p.add_argument("--rnn-layers", type=int,   default=1)
    p.add_argument("--rnn-epochs", type=int,   default=100)
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
            # Two diagonal variants from the exp_sas.py sweep:
            #   sn=0.90 → best for h=1,5  (diag_n100_sn0.90: h5=0.748)
            #   sn=0.95 → best for h=22   (diag_n100_sn0.95: h22=1.565)
            # Linear basis is excluded — it overfits catastrophically (see exp_sas.py).
            # Polys are JAX pytrees — passed directly to the JIT parallel scan kernel.
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
    Rolling OOS evaluation with periodic refit and streaming state updates.

    Per step t:
      1. Refit all models (if refit_freq steps have elapsed) on z-scored
         training window.  Z-score stats are frozen until next refit.
      2. Record model.predict(h) vs z-scored target for each horizon h.
      3. model.update(z_t) — advance state without refit.
    """
    from data.data_loader import fit_scaler, apply_scaler

    T     = len(log_values)
    H_max = max(horizons)
    mu, sigma = 0.0, 1.0
    steps_since_refit: dict[str, int] = {n: refit_freq for n in models}
    records: list[dict] = []

    print(f"\nOOS: {dates[window]} → {dates[T - H_max - 1]}")
    print(f"     window={window}, refit_freq={refit_freq}, "
          f"models: {list(models)}\n")

    for t in range(window, T - H_max):
        train_raw = log_values[t - window: t]

        # ── (re)fit ──────────────────────────────────────────────────────
        for name, model in models.items():
            if steps_since_refit[name] >= refit_freq:
                mu, sigma = fit_scaler(train_raw)
                train_z   = apply_scaler(train_raw, mu, sigma)
                try:
                    model.fit(train_z, horizons)
                    steps_since_refit[name] = 0
                    if t == window:
                        print(f"  Initial fit: {name}")
                except Exception as e:
                    print(f"  [fit {name} @ {dates[t]}] {e}")

        z_t = apply_scaler(float(log_values[t]), mu, sigma)

        # ── predict ──────────────────────────────────────────────────────
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
                    print(f"  [predict {name} h={h} @ {dates[t]}] {e}")

        # ── update state ──────────────────────────────────────────────────
        for name, model in models.items():
            try:
                model.update(z_t)
                steps_since_refit[name] += 1
            except Exception as e:
                print(f"  [update {name} @ {dates[t]}] {e}")

    return pd.DataFrame(records)


# ── summary display ──────────────────────────────────────────────────────────

def print_summary(df_eval: pd.DataFrame) -> None:
    from utils.metrics import summary_table
    tbl = summary_table(df_eval, metrics=["mse", "mae", "mda"])
    # Print one block per metric
    for metric in ["mse", "mae", "mda"]:
        if metric not in tbl.columns.get_level_values("metric"):
            continue
        sub = tbl[metric].sort_index()
        fmt = "{:.4f}".format if metric in ("mse", "mae") else "{:.3f}".format
        print(f"\n── {metric.upper()} ─────────────────────────────────")
        print(sub.to_string(float_format=fmt))


# ── entry point ──────────────────────────────────────────────────────────────

def main(argv=None):
    args    = parse_args(argv)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from data.data_loader import load_rv
    from utils.mcs        import run_mcs_analysis, run_mcs_grouped, mcs_summary_text

    # ── data ─────────────────────────────────────────────────────────────
    csv_path = ROOT / args.csv
    print(f"Loading {args.symbol} / {args.target} from {args.csv} …")
    log_values, dates = load_rv(
        csv_path,
        symbol        = args.symbol,
        target        = args.target,
        log_transform = not args.no_log,
    )
    print(f"  {dates[0]} → {dates[-1]},  T={len(log_values)}\n")

    # ── models ───────────────────────────────────────────────────────────
    models = build_models(args)
    if not models:
        sys.exit("No models enabled.")

    # ── rolling OOS ──────────────────────────────────────────────────────
    df_eval = run_oos(
        log_values = log_values,
        dates      = dates,
        models     = models,
        horizons   = args.horizons,
        window     = args.window,
        refit_freq = args.refit_freq,
    )
    df_eval.to_csv(out_dir / "errors.csv", index=False)
    print(f"\nSaved {len(df_eval)} records → {out_dir / 'errors.csv'}")

    # ── metrics ───────────────────────────────────────────────────────────
    print_summary(df_eval)

    # ── MCS ──────────────────────────────────────────────────────────────
    print(f"\nRunning per-horizon MCS  (α={args.mcs_alpha}, B={args.mcs_nboot}) …")
    mcs_ph = run_mcs_analysis(
        df_eval, horizons=args.horizons,
        periods=["full"],
        alpha=args.mcs_alpha, n_boot=args.mcs_nboot, seed=args.seed,
    )
    mcs_ph.to_csv(out_dir / "mcs_per_horizon.csv", index=False)
    print(mcs_summary_text(mcs_ph, title="per-horizon"))

    horizons = sorted(args.horizons)
    if len(horizons) >= 2:
        mid    = len(horizons) // 2
        groups = [horizons[:mid], horizons[mid:]]
        print(f"\nRunning grouped MCS  (groups={groups}) …")
        mcs_grp = run_mcs_grouped(
            df_eval, horizon_groups=groups,
            periods=["full"],
            alpha=args.mcs_alpha, n_boot=args.mcs_nboot, seed=args.seed,
        )
        mcs_grp.to_csv(out_dir / "mcs_grouped.csv", index=False)
        print(mcs_summary_text(mcs_grp, title="grouped"))

    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
