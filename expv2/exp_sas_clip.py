"""
expv2/exp_sas_clip.py — SAS ablation for clipping and input scaling.
"""

from __future__ import annotations
import sys
import time
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import trange

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    print_mse_table, print_mcs_frequency, _sq_errors_to_eval_df
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
DIVERGE_THRESH = 1.5

def _sas(q=1, clip=True, input_scale=4.0):
    return SASForecaster(
        n_reservoir=200, basis="diagonal", spectral_norm=0.95, p_degree=1, q_degree=q,
        washout=WASHOUT, seed=SEED, apply_log=False, clip=clip, input_scale=input_scale, target_log=False 
    )

def build_clip_grid() -> dict[str, callable]:
    grid = {"HAR": lambda: HARForecaster(ridge=False)}
    configs = [("noclip", False, 1.0), ("clip2", True, 2.0), ("clip4", True, 4.0), ("clip6", True, 6.0)]
    for q in [1, 2]:
        for name_suffix, use_clip, scale in configs:
            name = f"SAS_q{q}_{name_suffix}"
            grid[name] = lambda q_val=q, cv=use_clip, sv=scale: _sas(q=q_val, clip=cv, input_scale=sv)
    return grid

# ── Clean OOS Loop with Alpha Tracking ────────────────────────────────────────

def run_experiment_oos(values, dates, models, horizons, window, refit_freq):
    T, H_max = len(values), max(horizons)
    steps_since_refit = {n: refit_freq for n in models}
    losses = {name: {h: [] for h in horizons} for name in models}
    alphas = {name: {h: [] for h in horizons} for name in models}
    
    test_dates = dates[window : T - H_max]
    
    for i, t in enumerate(range(window, T - H_max)):
        train = values[t - window: t]
        for name, model in models.items():
            if steps_since_refit[name] >= refit_freq:
                model.fit(train, horizons)
                steps_since_refit[name] = 0
                if hasattr(model, 'alpha_log_'):
                    sf = (model.input_scale ** 2) if hasattr(model, 'input_scale') else 1.0
                    for h in horizons:
                        if h in model.alpha_log_:
                            val = model.alpha_log_[h]
                            alphas[name][h].append(float(np.mean(val)) * sf)
        
        x_t = float(values[t])
        for name, model in models.items():
            model.update(x_t)
            steps_since_refit[name] += 1
            for h in horizons:
                y_hat, y_true = float(model.predict(h)), float(values[t + h])
                losses[name][h].append((y_hat - y_true) ** 2)
                
    return losses, alphas, test_dates

# ── Table Helpers ─────────────────────────────────────────────────────────────

def print_alpha_tables(alpha_by_sym, horizons, model_order):
    for label, func in [("MEAN SCALED RIDGE PENALTY", np.mean), ("STD DEV OF SCALED RIDGE PENALTY", np.std)]:
        print(f"\n{'═'*75}\n  {label}\n{'═'*75}")
        header = f"{'Model':<25} " + " ".join(f"{h:>12}" for h in horizons)
        print(header + "\n" + "-"*len(header))
        for m in model_order:
            if m == "HAR": continue
            row = f"{m:<25} "
            for h in horizons:
                vals = [func(alpha_by_sym[s][m][h]) for s in alpha_by_sym if alpha_by_sym[s][m][h]]
                row += f"{np.mean(vals):>12.2e} " if vals else f"{'N/A':>12} "
            print(row)

def print_per_horizon_scoring(evals_by_sym, horizons, alpha, model_order):
    from arch.bootstrap import MCS
    for h in horizons:
        print(f"\n{'═'*75}\n  SCORE (ROW-NORM RMSE) & MCS SURVIVORS — H={h}\n{'═'*75}")
        rows = []
        for sym, df in evals_by_sym.items():
            df_h = df[df["horizon"] == h]
            m_list = [m for m in model_order if m in df_h["config"].unique()]
            row = {"Symbol": sym}
            losses_dict = {m: df_h[df_h["config"] == m]["norm_sq_err"].values for m in m_list}
            for m in m_list:
                row[m] = np.sqrt(np.mean(losses_dict[m]))
            try:
                loss_df = pd.DataFrame(losses_dict).dropna()
                mcs = MCS(loss_df, size=alpha, block_size=int(len(loss_df)**(1/3)), method='R')
                mcs.compute()
                survivors = list(mcs.pvalues[mcs.pvalues >= alpha].sort_values(ascending=False).index)
            except: survivors = ["error"]
            row["MCS_Survivors"] = ", ".join(survivors)
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── run one grid ──────────────────────────────────────────────────────────────

def run_grid(syms, grid, model_order, title, out_dir):
    mse_abs_scaled = {}
    alpha_by_sym = {}
    evals_by_sym = {}

    for sym in syms:
        try:
            values, dates = load_rv(CSV, sym)
        except: continue
        if len(values) < WINDOW + max(HORIZONS) + 2: continue

        models = {name: factory() for name, factory in grid.items()}
        losses, alphas, test_dates = run_experiment_oos(values, dates, models, HORIZONS, WINDOW, REFIT_FREQ)
        
        # Build local DataFrame for this symbol
        chunks = []
        for name, h_map in losses.items():
            for h, errs in h_map.items():
                chunks.append(pd.DataFrame({"config": name, "horizon": h, "test_date": test_dates[:len(errs)], "sq_err": errs}))
        df_eval = pd.concat(chunks, ignore_index=True)
        
        # Row-Normalization (Daily Relative Competition)
        mean_errs = df_eval.groupby(["horizon", "test_date"])["sq_err"].transform("mean")
        df_eval["norm_sq_err"] = df_eval["sq_err"] / mean_errs.replace(0, np.nan)
        df_eval["norm_sq_err"] = df_eval["norm_sq_err"].fillna(1.0)
        
        evals_by_sym[sym] = df_eval
        alpha_by_sym[sym] = alphas
        
        # Abs Precision (MSE / Var)
        oos_var = np.var(values[WINDOW:])
        mse_abs_scaled[sym] = {n: {h: np.mean(losses[n][h])/oos_var for h in HORIZONS} for n in losses}
        print(f"  ✓ {sym:<12} | Done")

    if not mse_abs_scaled: return

    # 1. Precision Table
    print_mse_table(mse_abs_scaled, HORIZONS, title=f"{title} (MSE / OOS_Var)", model_order=model_order)
    
    # 2. MCS Frequency Table (Aggregated)
    print_mcs_frequency(evals_by_sym, HORIZONS, alpha=0.10, n_boot=1000, seed=SEED, mse_by_symbol=mse_abs_scaled)

    # 3. Alpha Tracking
    print_alpha_tables(alpha_by_sym, HORIZONS, model_order)

    # 4. Detailed Symbol Scores + MCS Survivors
    print_per_horizon_scoring(evals_by_sym, HORIZONS, 0.10, model_order)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["all"])
    args = parser.parse_args()
    syms = available_symbols(CSV) if args.symbols == ["all"] else args.symbols

    grid = build_clip_grid()
    run_grid(syms, grid, list(grid.keys()), "Clipping & Scaling Ablation", ROOT)

if __name__ == "__main__":
    main()