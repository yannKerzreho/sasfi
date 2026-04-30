"""
expv2/exp_sas_residual.py — Residual-target ablation for SAS.

Idea
----
Instead of fitting the readout directly on y_{t+h}, fit it on the increment
    r_{t,h} = y_{t+h} - y_t
and at prediction time reconstruct:
    ŷ_{t+h} = y_t  +  r̂_{t,h}

The anchor y_t is always positive (level RV), so the prediction is bounded
away from zero even when the SAS readout predicts r̂ ≈ 0 — directly
addressing the QLIKE blowup episodes identified in the failure analysis.

Grid compared
-------------
  HAR                   — benchmark
  SAS_q1 / SAS_q2       — standard (direct level forecast)
  SAS_q1_res / SAS_q2_res — residual-target variant

Tables produced
---------------
1. Precision   — MSE and mean QLIKE
2. MCS freq    — Model Confidence Set survival rate
3. Beats HAR   — DM test (MSE normalised, QLIKE raw)
4. Alpha track — ridge penalty statistics
5. Per-horizon — RMSE + MCS survivors
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    run_oos,
    print_precision_table,
    print_mcs_frequency,
    print_beats_benchmark,
    print_per_horizon_scoring,
)
from data.data_loader import load_rv, available_symbols
from models.linear import HARForecaster
from models.sas    import SASForecaster

CSV        = ROOT / "rv.csv"
HORIZONS   = [1, 5, 10]
WINDOW     = 2000
REFIT_FREQ = 20
SEED       = 42
WASHOUT    = 50

# ── SAS factory ───────────────────────────────────────────────────────────────

def _sas(q: int = 1, residual: bool = False) -> SASForecaster:
    return SASForecaster(
        n_reservoir    = 200,
        basis          = "diagonal",
        spectral_norm  = 0.95,
        p_degree       = 1,
        q_degree       = q,
        washout        = WASHOUT,
        seed           = SEED,
        apply_log      = False,
        clip           = True,
        input_scale    = 6.0,
        target_log     = False,
        residual_target= residual,
    )


def build_grid() -> dict[str, callable]:
    return {
        "HAR":        lambda: HARForecaster(ridge=False),
        "SAS_q1":     lambda: _sas(q=1, residual=False),
        "SAS_q1_res": lambda: _sas(q=1, residual=True),
        "SAS_q2":     lambda: _sas(q=2, residual=False),
        "SAS_q2_res": lambda: _sas(q=2, residual=True),
    }


# ── alpha-tracking table ──────────────────────────────────────────────────────

def print_alpha_tables(alpha_by_sym, horizons, model_order):
    for label, func in [
        ("MEAN SCALED RIDGE PENALTY",   np.mean),
        ("STD DEV OF SCALED RIDGE PENALTY", np.std),
    ]:
        print(f"\n{'═'*70}\n  {label}\n{'═'*70}")
        header = f"{'Model':<20} " + " ".join(f"{h:>10}" for h in horizons)
        print(header + "\n" + "-" * len(header))
        for m in model_order:
            if m == "HAR":
                continue
            row = f"{m:<20} "
            for h in horizons:
                vals = [
                    func(alpha_by_sym[s][m][h])
                    for s in alpha_by_sym
                    if m in alpha_by_sym[s] and alpha_by_sym[s][m][h]
                ]
                row += f"{np.mean(vals):>10.2e} " if vals else f"{'N/A':>10} "
            print(row)


# ── main loop ─────────────────────────────────────────────────────────────────

def run(syms, grid, model_order, title, out_dir, mcs_alpha, mcs_nboot):
    mse_by_symbol   = {}
    qlike_by_symbol = {}
    alpha_by_sym    = {}
    evals_by_symbol = {}

    pbar = tqdm(syms, unit="symbol", desc=title[:35])
    for sym in pbar:
        pbar.set_postfix(sym=sym)
        try:
            values, dates = load_rv(CSV, sym)
        except Exception:
            continue
        if len(values) < WINDOW + max(HORIZONS) + 2:
            continue

        models = {name: factory() for name, factory in grid.items()}
        alphas = {}
        df_eval = run_oos(
            values, dates, models,
            horizons   = HORIZONS,
            window     = WINDOW,
            refit_freq = REFIT_FREQ,
            alphas_out = alphas,
            verbose    = False,
        )

        sym_mse:   dict[str, dict[int, float]] = {}
        sym_qlike: dict[str, dict[int, float]] = {}
        for (name, h), grp in df_eval.groupby(["config", "horizon"]):
            sym_mse.setdefault(name, {})[h]   = float(grp["sq_err"].mean())
            sym_qlike.setdefault(name, {})[h] = float(grp["qlike"].mean())

        evals_by_symbol[sym]  = df_eval
        alpha_by_sym[sym]     = alphas
        mse_by_symbol[sym]    = sym_mse
        qlike_by_symbol[sym]  = sym_qlike
        tqdm.write(f"  ✓ {sym:<12} | Done")

    if not mse_by_symbol:
        print("No usable symbols.")
        return

    # 1. Precision
    print_precision_table(
        mse_by_symbol, qlike_by_symbol, HORIZONS,
        title=f"{title} — Precision",
        model_order=model_order,
    )

    # 2. MCS frequency
    print_mcs_frequency(
        evals_by_symbol, HORIZONS,
        alpha=mcs_alpha, n_boot=mcs_nboot, seed=SEED,
        mse_by_symbol=mse_by_symbol, model_order=model_order,
    )

    # 3. DM test vs HAR
    print_beats_benchmark(
        evals_by_symbol, HORIZONS,
        benchmark="HAR", alpha=mcs_alpha, model_order=model_order,
    )

    # 4. Alpha tracking
    print_alpha_tables(alpha_by_sym, HORIZONS, model_order)

    # 5. Per-horizon detail
    print_per_horizon_scoring(
        evals_by_symbol, HORIZONS,
        alpha=mcs_alpha, model_order=model_order,
    )

    # Save
    pd.concat(evals_by_symbol.values(), ignore_index=True).to_csv(
        out_dir / "errors_residual.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="SAS residual-target ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols",   nargs="+", default=["all"])
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-nboot", type=int,   default=1000)
    parser.add_argument("--out-dir",   default=None)
    args = parser.parse_args()

    syms    = available_symbols(CSV) if args.symbols == ["all"] else args.symbols
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "experiments/results_sas_residual"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    run(
        syms, grid, list(grid.keys()),
        "SAS Residual-Target Ablation",
        out_dir, args.mcs_alpha, args.mcs_nboot,
    )


if __name__ == "__main__":
    main()
