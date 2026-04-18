"""
experiments/exp_degree_ridge.py — Per-degree grouped ridge ablation.

Research question
-----------------
Does assigning *different* ridge penalties to sub-reservoirs of different
polynomial degrees improve OOS performance compared to a single shared alpha?

Architecture variants (clean 4-way ablation)
--------------------------------------------
SAS_uniform  : standard SAS, n=100, p_degree=1, single alpha for all 100 dims.
               Pure uniform gating — baseline for the grouped approach.

MdSAS_arch   : MultiDegreeSAS, 2 groups × 50 dims (total=100), SINGLE shared alpha.
               Group 0: DiagonalPoly(p_degree=0) — linear filter (no input gating).
               Group 1: DiagonalPoly(p_degree=1) — multiplicative gating.
               SAME architecture as MdSAS_d1 but single alpha → isolates
               the ARCHITECTURAL benefit (mixing degrees) from REGULARIZATION benefit.

MdSAS_d1     : Same 2×50 architecture, TWO independent alphas (block-diagonal ridge).
               Adds the per-group regularization on top of the architecture.

MdSAS_d2     : 3 groups × 50 dims (total=150), THREE independent alphas.
               Groups 0,1,2 with p_degree 0,1,2.

Comparison logic
----------------
SAS_uniform vs MdSAS_arch : pure architectural effect (mixing degree-0 + degree-1)
MdSAS_arch  vs MdSAS_d1   : pure regularization effect (per-group alpha vs shared)
SAS_uniform vs MdSAS_d1   : combined effect

All use spectral_norm=0.90 (stable for n≈100), washout=50, q_degree=1.
HAR is the overall reference.

Usage
-----
    python experiments/exp_degree_ridge.py                    # all symbols
    python experiments/exp_degree_ridge.py --symbols .AEX .SPX  # quick test
    python experiments/exp_degree_ridge.py --no-mcs           # skip MCS (faster)
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
    quick_oos, mean_losses, print_mse_table, print_mcs_frequency,
    _sq_errors_to_eval_df, HORIZONS, WINDOW, REFIT_FREQ,
)
from data.data_loader    import load_rv, available_symbols
from models.linear       import HARForecaster
from models.sas          import SASForecaster, MultiDegreeSASForecaster, _ALPHAS

CSV         = ROOT / "rv.csv"
DEFAULT_SYM = ".AEX"

# ── spectral norm shared across all variants ──────────────────────────────────
# sn=0.90: the stable choice for n≈100 reservoirs (diag_n100_sn0.90 had no ‡
# flag in exp_sas; diag_n100_sn0.95 was flagged unstable on several symbols).
SN      = 0.90
WASHOUT = 50
SEED    = 42

# ── model registry ────────────────────────────────────────────────────────────
_MODEL_ORDER = [
    "HAR",
    "SAS_uniform",
    "MdSAS_arch",
    "MdSAS_d1",
    "MdSAS_d2",
]


def make_models() -> dict:
    return {
        "HAR": HARForecaster(ridge=False),

        # Control: uniform p_degree=1 reservoir, single alpha
        "SAS_uniform": SASForecaster(
            n_reservoir=100,
            basis="diagonal",
            spectral_norm=SN,
            p_degree=1,
            q_degree=1,
            washout=WASHOUT,
            seed=SEED,
        ),

        # Architectural ablation: mixed degree-0/1 sub-reservoirs, SINGLE alpha.
        # Same total n=100 as SAS_uniform — isolates architecture from regularisation.
        # Uses the full _ALPHAS grid (37 values) for the single-alpha CV so it
        # gets the best possible single-alpha performance, making the architecture
        # comparison as informative as possible.
        "MdSAS_arch": MultiDegreeSASForecaster(
            n_per_group=50,
            max_degree=1,
            q_degree=1,
            spectral_norm=SN,
            washout=WASHOUT,
            seed=SEED,
            grouped_ridge=False,
            alphas_1d=_ALPHAS,     # full 37-value grid for best single-alpha CV
        ),

        # Full model: mixed degree-0/1, PER-GROUP alpha (grouped_ridge=True)
        "MdSAS_d1": MultiDegreeSASForecaster(
            n_per_group=50,
            max_degree=1,
            q_degree=1,
            spectral_norm=SN,
            washout=WASHOUT,
            seed=SEED,
            grouped_ridge=True,
        ),

        # Extension: 3 groups (p_degree 0,1,2), per-group alpha, total n=150
        "MdSAS_d2": MultiDegreeSASForecaster(
            n_per_group=50,
            max_degree=2,
            q_degree=1,
            spectral_norm=SN,
            washout=WASHOUT,
            seed=SEED,
            grouped_ridge=True,
        ),
    }


# ── alpha-ratio collection ────────────────────────────────────────────────────

def collect_alpha_ratios(
    log_values: np.ndarray,
    model: MultiDegreeSASForecaster,
    horizons: list[int],
    window: int = WINDOW,
    refit_freq: int = REFIT_FREQ,
) -> dict[int, list[tuple]]:
    """
    Run a rolling OOS collecting the chosen alpha tuples at every refit.

    Returns
    -------
    alpha_records : {h: [(α₀, α₁, ...), ...]}  — one tuple per refit per horizon
    """
    T      = len(log_values)
    H_max  = max(horizons)
    limit  = T - H_max

    def fit_scaler(arr):
        mu, sigma = float(arr.mean()), float(arr.std())
        sigma     = sigma if sigma > 1e-8 else 1.0
        return mu, sigma

    def apply_scaler(x, mu, sigma):
        return (x - mu) / sigma

    mu, sigma  = 0.0, 1.0
    steps_since = refit_freq
    alpha_records: dict[int, list] = {h: [] for h in horizons}

    for t in range(window, limit):
        if steps_since >= refit_freq:
            train_raw = log_values[t - window: t]
            mu, sigma = fit_scaler(train_raw)
            train_z   = apply_scaler(train_raw, mu, sigma)
            model.fit(train_z, horizons)
            for h in horizons:
                if h in model.alpha_log_:
                    alpha_records[h].append(model.alpha_log_[h])
            steps_since = 0
        z_t = apply_scaler(float(log_values[t]), mu, sigma)
        model.update(z_t)
        steps_since += 1

    return alpha_records


def print_alpha_ratio_table(
    all_ratios: dict[str, dict[str, dict[int, list[tuple]]]],
    horizons: list[int],
) -> None:
    """
    Print mean log₁₀(αᵢ / α₀) per group-pair, model, horizon.
    Ratio > 0 → group i is penalised *more* than group 0.
    """
    print("\n" + "═" * 64)
    print(" Alpha-ratio analysis  (mean log₁₀(αᵢ/α₀) across symbols & refits)")
    print("═" * 64)

    for model_name, sym_records in all_ratios.items():
        print(f"\n  {model_name}")
        # Collect all (α₀, α₁, ...) tuples
        by_h: dict[int, list[tuple]] = {h: [] for h in horizons}
        for sym, h_map in sym_records.items():
            for h, tuples in h_map.items():
                by_h[h].extend(tuples)

        n_groups = len(next(iter(next(iter(sym_records.values())).values()), [(1, 1)])[0]) if sym_records else 2

        header = "  h   " + "  ".join(f"log(α{d}/α0)" for d in range(1, n_groups))
        print("  " + header)
        for h in horizons:
            tuples = by_h[h]
            if not tuples:
                print(f"  {h:>2}   —")
                continue
            ratios_per_group = []
            for d in range(1, n_groups):
                log_ratios = [np.log10(t[d] / t[0]) for t in tuples if t[0] > 0]
                ratios_per_group.append(f"{np.mean(log_ratios):+.2f} ± {np.std(log_ratios):.2f}")
            print(f"  h={h:<2}  " + "  ".join(ratios_per_group))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-degree grouped ridge ablation.")
    parser.add_argument("--symbols", nargs="+", default=["all"],
                        help="Symbols to run (default: all).")
    parser.add_argument("--no-mcs",  action="store_true",
                        help="Skip MCS (faster, show MSE table only).")
    parser.add_argument("--ratios", action="store_true",
                        help="Collect alpha-ratio diagnostics (runs a 2nd OOS pass per MultiDegree model — slow).")
    parser.add_argument("--mcs-alpha",  type=float, default=0.10)
    parser.add_argument("--mcs-nboot",  type=int,   default=1000)
    args = parser.parse_args()

    # ── resolve symbols ───────────────────────────────────────────────────────
    if args.symbols == ["all"]:
        syms = available_symbols(CSV)
    else:
        syms = args.symbols
    print(f"\nRunning on {len(syms)} symbol(s): {' '.join(syms[:5])}"
          f"{'…' if len(syms) > 5 else ''}")
    print(f"Protocol: W={WINDOW}, R={REFIT_FREQ}, H={HORIZONS}\n")

    # ── per-symbol OOS ────────────────────────────────────────────────────────
    evals_by_sym: dict[str, pd.DataFrame]    = {}
    mse_by_sym:   dict[str, dict]            = {}
    # {model_name: {sym: {h: [(α₀,α₁,...), ...]}}}
    ratio_data:   dict[str, dict]            = defaultdict(dict)

    for sym in syms:
        try:
            log_vals, dates = load_rv(CSV, sym)
        except Exception as exc:
            print(f"  [skip] {sym}: {exc}")
            continue

        t0     = time.perf_counter()
        models = make_models()

        losses = quick_oos(
            log_vals, dates, models,
            horizons=HORIZONS, window=WINDOW, refit_freq=REFIT_FREQ,
        )
        dt = time.perf_counter() - t0

        # MSE per model × horizon
        mse_by_sym[sym] = {
            name: {h: float(np.mean(errs)) for h, errs in h_map.items()}
            for name, h_map in losses.items()
        }
        # DataFrame for MCS
        evals_by_sym[sym] = _sq_errors_to_eval_df({sym: losses})[sym]

        avg_mse = {
            name: np.mean(list(h_map.values()))
            for name, h_map in mse_by_sym[sym].items()
        }
        mse_str = "  ".join(f"{n}={v:.4f}" for n, v in avg_mse.items())
        print(f"  {sym:<12}  {dt:5.1f}s  {mse_str}")

        # ── alpha-ratio collection (optional, runs a 2nd OOS pass) ──────────
        if args.ratios:
            for mname in ["MdSAS_d1", "MdSAS_d2"]:
                m_fresh = make_models()[mname]   # grouped_ridge=True
                ar = collect_alpha_ratios(log_vals, m_fresh, HORIZONS)
                ratio_data[mname][sym] = ar

    if not mse_by_sym:
        print("No symbols processed.")
        return

    print()

    # ── 1. Stability table (divergence count per model) ────────────────────────
    DIVERGE_THRESH = 5.0   # avg-MSE above this → symbol considered diverged
    models_seen = _MODEL_ORDER if _MODEL_ORDER else list(
        next(iter(mse_by_sym.values())).keys()
    )
    print("═" * 64)
    print(" Stability: #symbols diverged (avg MSE > %.0f)" % DIVERGE_THRESH)
    print("═" * 64)
    w_stab = max(len(m) for m in models_seen) + 2
    for m in models_seen:
        n_div = sum(
            1 for sym_mse in mse_by_sym.values()
            if m in sym_mse and
               float(np.mean([v for v in sym_mse[m].values() if np.isfinite(v)] or [0])) > DIVERGE_THRESH
        )
        n_total = sum(1 for sym_mse in mse_by_sym.values() if m in sym_mse)
        bar = "█" * n_div + "░" * (n_total - n_div)
        print(f"  {m:<{w_stab}} {n_div:>2}/{n_total}  {bar}")
    print()

    # ── 2. Stable-subset MSE — only symbols where ALL models are stable ────────
    stable_syms = [
        sym for sym, sym_mse in mse_by_sym.items()
        if all(
            float(np.mean([v for v in sym_mse.get(m, {}).values() if np.isfinite(v)] or [9e9]))
            <= DIVERGE_THRESH
            for m in models_seen
        )
    ]
    stable_mse = {sym: mse_by_sym[sym] for sym in stable_syms}
    print_mse_table(
        stable_mse, HORIZONS,
        title=(f"OOS MSE — stable subset ({len(stable_syms)}/{len(mse_by_sym)} symbols, "
               f"all models MSE ≤ {DIVERGE_THRESH:.0f})"),
        model_order=_MODEL_ORDER,
    )

    # ── 3. Full MSE table (all symbols, NaN-safe mean) ────────────────────────
    print_mse_table(
        mse_by_sym, HORIZONS,
        title=f"OOS MSE — all symbols ({len(mse_by_sym)} total, means inflated by diverged runs)",
        model_order=_MODEL_ORDER,
    )

    # ── 4. MCS frequency table (stable-subset only for clean comparison) ──────
    if not args.no_mcs:
        stable_evals = {sym: evals_by_sym[sym] for sym in stable_syms if sym in evals_by_sym}
        print_mcs_frequency(
            stable_evals, HORIZONS,
            alpha=args.mcs_alpha,
            n_boot=args.mcs_nboot,
            title=(f"MCS₀.₁₀ frequency — stable subset "
                   f"({len(stable_evals)} symbols)"),
            mse_by_symbol=stable_mse,
            model_order=_MODEL_ORDER,
        )

    # ── 5. Alpha-ratio table ───────────────────────────────────────────────────
    if args.ratios and ratio_data:
        print_alpha_ratio_table(ratio_data, HORIZONS)


if __name__ == "__main__":
    main()
