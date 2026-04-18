"""
experiments/exp_sn_sweep.py — Spectral-norm × reservoir-size sweep for MdSAS.

Research question
-----------------
Can we recover sn=0.95 stability in the multi-degree architecture by
increasing n_per_group?  And does larger n genuinely select larger ridge α,
explaining why n=200 sn=0.95 (SAS_diag) is stable while n=100 sn=0.95 diverges?

Models
------
HAR                : reference baseline.
SAS_diag           : current main-benchmark best — n=200, sn=0.95, single alpha.
                     (standard SASForecaster with full _ALPHAS grid)

MdSAS_n50_sn90     : current stable MdSAS config   (2×50=100,  sn=0.90)
MdSAS_n50_sn95     : raise sn, keep n   (2×50=100,  sn=0.95)  — baseline instability
MdSAS_n100_sn90    : raise n, keep sn   (2×100=200, sn=0.90)  — n effect
MdSAS_n100_sn95    : raise both         (2×100=200, sn=0.95)  — target config

Design note
-----------
MdSAS_n100 has total n=200 (same as SAS_diag), so both have the same
parameter count in the readout.  The difference is:
  SAS_diag       : all-p1 single reservoir,  single alpha (37-value grid)
  MdSAS_n100_sn95 : p0+p1 mixed,             per-group alpha (7-value, 49 combos)

Alpha-log diagnostic
--------------------
At the end of the run, for each model, we do one final .fit() on the last
training window of each symbol and record the chosen alpha(s).  This directly
tests the "larger n ⇒ larger CV-selected α" hypothesis.

Usage
-----
    python experiments/exp_sn_sweep.py                         # all symbols
    python experiments/exp_sn_sweep.py --symbols .AEX .SPX    # quick check
    python experiments/exp_sn_sweep.py --no-mcs                # skip MCS
"""

from __future__ import annotations

import sys
import time
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    quick_oos, print_mse_table, print_mcs_frequency,
    _sq_errors_to_eval_df, HORIZONS, WINDOW, REFIT_FREQ,
)
from data.data_loader    import load_rv, available_symbols
from models.linear       import HARForecaster
from models.sas          import (
    SASForecaster, MultiDegreeSASForecaster,
    _ALPHAS, _ALPHAS_GROUPED,
)

CSV      = ROOT / "rv.csv"
WASHOUT  = 50
SEED     = 42

# ── divergence threshold ───────────────────────────────────────────────────────
DIVERGE_THRESH = 5.0

# ── model order for display ────────────────────────────────────────────────────
_MODEL_ORDER = [
    "HAR",
    "SAS_diag",
    "MdSAS_n50_sn90",
    "MdSAS_n50_sn95",
    "MdSAS_n100_sn90",
    "MdSAS_n100_sn95",
]

# Same 7-value grid for all grouped models — coarser 5-value grid causes
# too-large gaps (0.018 → 3.16) that underregularise the degree-0 sub-reservoir
# on some symbols, leading to MORE divergences than the smaller n=50 models.

def make_models() -> dict:
    return {
        # ── reference ─────────────────────────────────────────────────────────
        "HAR": HARForecaster(ridge=False),

        # ── current main-benchmark best (SAS_diag from main.py) ───────────────
        "SAS_diag": SASForecaster(
            n_reservoir=200,
            basis="diagonal",
            spectral_norm=0.95,
            p_degree=1, q_degree=1,
            washout=WASHOUT, seed=SEED,
            alphas=_ALPHAS,           # full 37-value grid
        ),

        # ── MdSAS sweep: (n_per_group, sn) ────────────────────────────────────
        # n=50 per group (total=100): use 7-value grid (49 combos, fast)

        # Current stable baseline
        "MdSAS_n50_sn90": MultiDegreeSASForecaster(
            n_per_group=50, max_degree=1, q_degree=1,
            spectral_norm=0.90, washout=WASHOUT, seed=SEED,
            grouped_ridge=True, alphas_1d=_ALPHAS_GROUPED,   # 7 values
        ),
        # Higher sn, same n — does instability return?
        "MdSAS_n50_sn95": MultiDegreeSASForecaster(
            n_per_group=50, max_degree=1, q_degree=1,
            spectral_norm=0.95, washout=WASHOUT, seed=SEED,
            grouped_ridge=True, alphas_1d=_ALPHAS_GROUPED,   # 7 values
        ),

        # n=100 per group (total=200, same as SAS_diag): 7-value grid (49 combos)

        # Larger n, same sn — does larger n stabilise?
        "MdSAS_n100_sn90": MultiDegreeSASForecaster(
            n_per_group=100, max_degree=1, q_degree=1,
            spectral_norm=0.90, washout=WASHOUT, seed=SEED,
            grouped_ridge=True, alphas_1d=_ALPHAS_GROUPED,   # 7 values
        ),
        # Larger n + higher sn — can we match SAS_diag with grouped ridge?
        "MdSAS_n100_sn95": MultiDegreeSASForecaster(
            n_per_group=100, max_degree=1, q_degree=1,
            spectral_norm=0.95, washout=WASHOUT, seed=SEED,
            grouped_ridge=True, alphas_1d=_ALPHAS_GROUPED,   # 7 values
        ),
    }


# ── alpha diagnostic ───────────────────────────────────────────────────────────

def collect_final_alphas(
    log_values: np.ndarray,
    models_dict: dict,
    horizons: list[int],
    window: int = WINDOW,
) -> dict[str, dict[int, object]]:
    """
    Fit each model once on the last training window, return chosen alpha(s).
    For SASForecaster  : alpha_log_[h]  → float
    For MultiDegree    : alpha_log_[h]  → tuple of floats
    """
    def fit_scaler(arr):
        mu, sigma = float(arr.mean()), float(arr.std())
        return mu, sigma if sigma > 1e-8 else 1.0

    train_raw = log_values[-window:]
    mu, sigma = fit_scaler(train_raw)
    train_z   = (train_raw - mu) / sigma

    result = {}
    for name, model in models_dict.items():
        model.fit(train_z, horizons)
        result[name] = dict(getattr(model, "alpha_log_", {}))
    return result


def print_alpha_diagnostic(
    alpha_by_sym: dict[str, dict[str, dict[int, object]]],
    horizons: list[int],
    model_order: list[str],
) -> None:
    """
    For each model × horizon: print mean of log10(alpha) across symbols.
    For per-group models: print mean log10(alpha) for each group separately.
    This tests whether larger n or higher sn → larger selected alpha.
    """
    print("\n" + "═" * 72)
    print("  Alpha diagnostic — mean log₁₀(α) selected by CV across symbols")
    print("  (larger α = more regularisation; hypothesis: larger n → larger α)")
    print("═" * 72)

    w = max(len(m) for m in model_order) + 2

    for model_name in model_order:
        if model_name == "HAR":
            continue

        # Gather alpha tuples/floats across symbols
        by_h: dict[int, list] = {h: [] for h in horizons}
        for sym, model_alphas in alpha_by_sym.items():
            if model_name not in model_alphas:
                continue
            for h, aval in model_alphas[model_name].items():
                if h in by_h:
                    by_h[h].append(aval)

        if not any(by_h.values()):
            continue

        # Determine if grouped (tuple) or scalar
        sample = next(v for v in by_h.values() if v)
        is_grouped = isinstance(sample[0], tuple)

        print(f"\n  {model_name}")
        for h in horizons:
            vals = by_h[h]
            if not vals:
                print(f"    h={h}: —")
                continue
            if is_grouped:
                n_groups = len(vals[0])
                parts = []
                for g in range(n_groups):
                    log_alphas = [np.log10(v[g]) for v in vals if v[g] > 0]
                    parts.append(f"α{g}={np.mean(log_alphas):+.2f}±{np.std(log_alphas):.2f}")
                print(f"    h={h}:  " + "   ".join(parts)
                      + f"  [log₁₀ scale, {len(vals)} symbols]")
            else:
                log_alphas = [np.log10(v) for v in vals if v > 0]
                print(f"    h={h}:  α={np.mean(log_alphas):+.2f}±{np.std(log_alphas):.2f}"
                      f"  [log₁₀ scale, {len(vals)} symbols]")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="sn × n sweep for MdSAS.")
    parser.add_argument("--symbols", nargs="+", default=["all"])
    parser.add_argument("--no-mcs",   action="store_true")
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-nboot", type=int,   default=1000)
    args = parser.parse_args()

    if args.symbols == ["all"]:
        syms = available_symbols(CSV)
    else:
        syms = args.symbols

    print(f"\nRunning on {len(syms)} symbol(s): {' '.join(syms[:6])}"
          f"{'…' if len(syms) > 6 else ''}")
    print(f"Protocol: W={WINDOW}, R={REFIT_FREQ}, H={HORIZONS}\n")

    evals_by_sym: dict[str, pd.DataFrame] = {}
    mse_by_sym:   dict[str, dict]         = {}
    # {sym: {model_name: {h: alpha}}}
    alpha_by_sym: dict[str, dict]         = {}

    for sym in syms:
        try:
            log_vals, dates = load_rv(CSV, sym)
        except Exception as exc:
            print(f"  [skip] {sym}: {exc}")
            continue

        t0     = time.perf_counter()
        models = make_models()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            losses = quick_oos(
                log_vals, dates, models,
                horizons=HORIZONS, window=WINDOW, refit_freq=REFIT_FREQ,
            )
        dt = time.perf_counter() - t0

        mse_by_sym[sym] = {
            name: {h: float(np.mean(errs)) for h, errs in h_map.items()}
            for name, h_map in losses.items()
        }
        evals_by_sym[sym] = _sq_errors_to_eval_df({sym: losses})[sym]

        # Alpha diagnostic on last training window
        try:
            alpha_by_sym[sym] = collect_final_alphas(log_vals, make_models(), HORIZONS)
        except Exception:
            pass

        avg_mse = {
            n: np.mean([v for v in hm.values() if np.isfinite(v)])
            for n, hm in mse_by_sym[sym].items()
        }
        mse_str = "  ".join(f"{n}={v:.4f}" for n, v in avg_mse.items())
        print(f"  {sym:<12}  {dt:5.1f}s  {mse_str}")

    if not mse_by_sym:
        print("No symbols processed.")
        return

    print()

    # ── 1. Stability bar chart ─────────────────────────────────────────────────
    print("═" * 72)
    print("  Stability: #symbols diverged (avg MSE > %.0f)" % DIVERGE_THRESH)
    print("═" * 72)
    w_stab = max(len(m) for m in _MODEL_ORDER) + 2
    for m in _MODEL_ORDER:
        if m == "HAR":
            continue
        n_div = sum(
            1 for sym_mse in mse_by_sym.values()
            if m in sym_mse and
               float(np.nanmean(list(sym_mse[m].values()))) > DIVERGE_THRESH
        )
        n_tot = sum(1 for sym_mse in mse_by_sym.values() if m in sym_mse)
        bar   = "█" * n_div + "░" * (n_tot - n_div)
        print(f"  {m:<{w_stab}} {n_div:>2}/{n_tot}  {bar}")
    print()

    # ── 2. Stable-subset MSE ───────────────────────────────────────────────────
    stable_syms = [
        sym for sym, sym_mse in mse_by_sym.items()
        if all(
            float(np.nanmean(list(sym_mse.get(m, {1: 0}).values()))) <= DIVERGE_THRESH
            for m in _MODEL_ORDER
        )
    ]
    stable_mse  = {sym: mse_by_sym[sym] for sym in stable_syms}
    stable_eval = {sym: evals_by_sym[sym] for sym in stable_syms if sym in evals_by_sym}

    print_mse_table(
        stable_mse, HORIZONS,
        title=(f"OOS MSE — stable subset "
               f"({len(stable_syms)}/{len(mse_by_sym)} symbols, all models ≤ {DIVERGE_THRESH:.0f})"),
        model_order=_MODEL_ORDER,
    )
    print_mse_table(
        mse_by_sym, HORIZONS,
        title=f"OOS MSE — all {len(mse_by_sym)} symbols (means inflated by diverged runs)",
        model_order=_MODEL_ORDER,
    )

    # ── 3. MCS on stable subset ────────────────────────────────────────────────
    if not args.no_mcs and stable_eval:
        print_mcs_frequency(
            stable_eval, HORIZONS,
            alpha=args.mcs_alpha, n_boot=args.mcs_nboot,
            title=f"MCS₀.₁₀ — stable subset ({len(stable_eval)} symbols)",
            mse_by_symbol=stable_mse,
            model_order=_MODEL_ORDER,
        )

    # ── 4. Alpha diagnostic ────────────────────────────────────────────────────
    if alpha_by_sym:
        print_alpha_diagnostic(alpha_by_sym, HORIZONS, _MODEL_ORDER)


if __name__ == "__main__":
    main()
