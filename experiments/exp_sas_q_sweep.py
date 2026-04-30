"""
expv2/exp_sas_q_sweep.py — SAS ablation: q-degree sweep.

Grid: q ∈ {1,2,3}  ×  {noclip/1.0, clip/4.0}
      + HAR as benchmark.

Tables produced (per grid)
--------------------------
1. MSE / OOS-Var          — precision table.
2. MCS frequency          — survival rate across symbols.
3. Beats HAR              — step-level beat-rate vs HAR.
4. Alpha tracking         — mean ± std of chosen ridge penalty.
5. Per-horizon detail     — per-symbol RMSE + MCS survivors.
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
from models.linear    import HARForecaster
from models.sas       import SASForecaster

CSV        = ROOT / "rv.csv"
HORIZONS   = [1, 5, 10]
WINDOW     = 2000
REFIT_FREQ = 20
SEED       = 42
WASHOUT    = 50


def _sas(q=1, clip=True, input_scale=4.0):
    return SASForecaster(
        n_reservoir=200, basis="diagonal", spectral_norm=0.95,
        p_degree=1, q_degree=q, washout=WASHOUT, seed=SEED,
        apply_log=False, clip=clip, input_scale=input_scale, target_log=False,
    )


def build_q_grid() -> dict[str, callable]:
    grid = {"HAR": lambda: HARForecaster(ridge=False)}
    configs = [("noclip", False, 1.0), ("clip6", True, 6.0)]
    for q in [1, 2, 3]:
        for name_suffix, use_clip, scale in configs:
            name = f"SAS_q{q}_{name_suffix}"
            grid[name] = lambda q_val=q, cv=use_clip, sv=scale: _sas(
                q=q_val, clip=cv, input_scale=sv)
    return grid


# ── alpha-tracking table ──────────────────────────────────────────────────────

def print_alpha_tables(alpha_by_sym, horizons, model_order):
    for label, func in [
        ("MEAN SCALED RIDGE PENALTY", np.mean),
        ("STD DEV OF SCALED RIDGE PENALTY", np.std),
    ]:
        print(f"\n{'═'*75}\n  {label}\n{'═'*75}")
        header = f"{'Model':<25} " + " ".join(f"{h:>12}" for h in horizons)
        print(header + "\n" + "-" * len(header))
        for m in model_order:
            if m == "HAR":
                continue
            row = f"{m:<25} "
            for h in horizons:
                vals = [
                    func(alpha_by_sym[s][m][h])
                    for s in alpha_by_sym
                    if m in alpha_by_sym[s] and alpha_by_sym[s][m][h]
                ]
                row += f"{np.mean(vals):>12.2e} " if vals else f"{'N/A':>12} "
            print(row)


# ── run one grid ──────────────────────────────────────────────────────────────

def run_grid(syms, grid, model_order, title, out_dir, mcs_alpha, mcs_nboot):
    mse_by_symbol   = {}
    qlike_by_symbol = {}
    alpha_by_sym    = {}
    evals_by_symbol = {}

    pbar = tqdm(syms, unit="symbol", desc=title[:30])
    for sym in pbar:
        pbar.set_postfix(sym=sym)
        try:
            values, dates = load_rv(CSV, sym)
        except Exception:
            continue
        if len(values) < WINDOW + max(HORIZONS) + 2:
            continue

        models  = {name: factory() for name, factory in grid.items()}
        alphas  = {}   # populated by run_oos via alphas_out
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
        return

    # 1. Precision table (RMSE + QLIKE)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",   nargs="+", default=["all"])
    parser.add_argument("--no-mcs",    action="store_true")
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-nboot", type=int,   default=1000)
    parser.add_argument("--out-dir",   default=None)
    args = parser.parse_args()

    syms    = available_symbols(CSV) if args.symbols == ["all"] else args.symbols
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "experiments/results_sas_q_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = build_q_grid()
    run_grid(
        syms, grid, list(grid.keys()),
        "Q-Degree & Clipping Sweep",
        out_dir, args.mcs_alpha, args.mcs_nboot,
    )


if __name__ == "__main__":
    main()
