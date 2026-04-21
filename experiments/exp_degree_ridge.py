"""
experiments/exp_degree_ridge.py — Mixed-degree reservoir ablation.

What we know (from exp_sas.py results)
---------------------------------------
  • q=2 wins at h=1 on 20/30 symbols; q=1 wins on 3; q=3 on 6.
  • (p=2, q=2) improves MCS vs (p=1, q=2) with nearly identical MSE.
  • Mixing p-degrees (MdSAS) with a single shared q did not beat SAS_diag
    at n=100.  At n=200 the comparison was not run.

Open questions
--------------
  A — P-mixing:  does MdSAS (p=0+p=1) benefit from the larger n=200 budget?
                 Does adding per-group ridge on top of the architecture help
                 when n is large enough to be stable?

  B — Q-mixing:  group reservoirs by q-degree instead of p-degree.
                 Since different q values win on different assets, a mixed-q
                 reservoir lets the readout pick the best drive per asset.
                 → [q=1|n=100] + [q=2|n=100]  vs  SAS_q2_n200 alone.
                 → [q=1|q=2|q=3]×67  — three groups.

  C — PQ-mixing: combine both axes:
                 [p=0,q=1|n=100] + [p=1,q=2|n=100]
                 [p=1,q=1|n=67]  + [p=1,q=2|n=67]  + [p=2,q=2|n=67]

Alpha boundary check
--------------------
  If CV selects the smallest or largest candidate in the grid, the true
  optimum likely lies outside — a diagnostic warning is printed.

Saving
------
  Each grid writes <slug>_mse.csv and <slug>_mcs.csv to:
      experiments/results_degree_ridge/<YYYYMMDD_HHMMSS>/
  (or --out-dir if provided).

Usage
-----
    python experiments/exp_degree_ridge.py                        # all 3 grids
    python experiments/exp_degree_ridge.py --grid A               # P-mixing only
    python experiments/exp_degree_ridge.py --grid B               # Q-mixing only
    python experiments/exp_degree_ridge.py --grid C               # PQ-mixing only
    python experiments/exp_degree_ridge.py --symbols .AEX .SPX   # quick test
    python experiments/exp_degree_ridge.py --no-mcs              # skip MCS
"""

from __future__ import annotations

import sys
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.utils import (
    quick_oos, print_mse_table, print_mcs_frequency,
    _sq_errors_to_eval_df,
)
from data.data_loader import load_rv, available_symbols
from models.linear    import HARForecaster
from models.sas       import (
    SASForecaster, MultiDegreeSASForecaster,
    _ALPHAS, _ALPHAS_GROUPED,
)

CSV        = ROOT / "rv.csv"
HORIZONS   = [1, 5, 10]        # matches main.py benchmark
WINDOW     = 2000
REFIT_FREQ = 20
SEED       = 42
WASHOUT    = 50
SN         = 0.95              # confirmed best spectral norm for n≥200
DIVERGE_THRESH = 1.5


# ── alpha grid for grouped CV — wider than _ALPHAS_GROUPED, not too coarse ──
# 7 values, log₁₀ spacing 1.5 → covers [1e-4, 1e5] — same as current default.
# Extended upper end to 1e5 after noting boundary hits at h=10 in some configs.
_ALPHAS_EXP = [10 ** x for x in np.arange(-4, 5.5, 1.5)]   # 7 values


# ── alpha boundary check ──────────────────────────────────────────────────────

def check_alpha_boundary(
    alpha,          # float or tuple of floats
    grid: list,
    model: str,
    sym:  str,
    h:    int,
) -> None:
    """Warn if any selected alpha is at the edge of the search grid."""
    lo, hi = min(grid), max(grid)
    vals   = (alpha,) if isinstance(alpha, (int, float)) else tuple(alpha)
    for i, a in enumerate(vals):
        if a <= lo:
            g = f"[group {i}] " if len(vals) > 1 else ""
            print(f"  ⚠  {model} {sym} h={h}: {g}α={a:.2e} hit LOWER bound "
                  f"({lo:.2e}) — grid may be too narrow")
        if a >= hi:
            g = f"[group {i}] " if len(vals) > 1 else ""
            print(f"  ⚠  {model} {sym} h={h}: {g}α={a:.2e} hit UPPER bound "
                  f"({hi:.2e}) — grid may be too narrow")


# ── model factories ───────────────────────────────────────────────────────────

def _sas(n, p=1, q=1, sn=SN):
    return SASForecaster(
        n_reservoir=n, basis="diagonal", spectral_norm=sn,
        p_degree=p, q_degree=q, washout=WASHOUT, seed=SEED,
        alphas=_ALPHAS,
    )


def _md(n_per_group, p_list, q_list, sn=SN, grouped=True):
    return MultiDegreeSASForecaster(
        n_per_group=n_per_group,
        p_degrees=p_list,
        q_degrees=q_list,
        spectral_norm=sn,
        washout=WASHOUT, seed=SEED,
        grouped_ridge=grouped,
        alphas_1d=_ALPHAS_EXP,
    )


def build_grid_A() -> dict[str, callable]:
    """
    A — P-mixing at n≈200.
    Q1 (A): does MdSAS (p=0+p=1) help when n is large?
    Comparison: SAS_p1 vs MdSAS_p01 (both n=200 total, per-group α).
    Also tests single-α architecture-only (MdSAS_p01_shared) to isolate
    the regularisation benefit.
    """
    return {
        "HAR":
            lambda: HARForecaster(ridge=False),
        # single p=1 reservoir, n=200 — main benchmark reference
        "SAS_p1_n200":
            lambda: _sas(200, p=1, q=1),
        # p=0 + p=1 groups, 2×100=200 total, SHARED alpha (architecture only)
        "MdSAS_p01_shared":
            lambda: _md(100, [0, 1], [1, 1], grouped=False),
        # p=0 + p=1 groups, 2×100=200 total, PER-GROUP alpha
        "MdSAS_p01_grouped":
            lambda: _md(100, [0, 1], [1, 1], grouped=True),
        # p=0 + p=1 + p=2, 3×67=201 total, per-group alpha
        "MdSAS_p012_grouped":
            lambda: _md(67, [0, 1, 2], [1, 1, 1], grouped=True),
    }


def build_grid_B() -> dict[str, callable]:
    """
    B — Q-mixing at n≈200, all p=1.
    Q2 (B): does grouping states by q-degree help?
    Since q=2 wins on 20/30 symbols but q=1 wins on others, a mixed-q
    reservoir lets the readout learn asset-specific weights per q.
    """
    return {
        "HAR":
            lambda: HARForecaster(ridge=False),
        # single-q baselines (from main benchmark)
        "SAS_q1_n200":
            lambda: _sas(200, p=1, q=1),
        "SAS_q2_n200":
            lambda: _sas(200, p=1, q=2),
        # q=1 + q=2 groups, 2×100=200 total, per-group alpha
        "MixedQ_12":
            lambda: _md(100, [1, 1], [1, 2], grouped=True),
        # q=1 + q=2 + q=3, 3×67=201 total, per-group alpha
        "MixedQ_123":
            lambda: _md(67, [1, 1, 1], [1, 2, 3], grouped=True),
        # q=2 + q=3 groups — skip q=1 entirely
        "MixedQ_23":
            lambda: _md(100, [1, 1], [2, 3], grouped=True),
    }


def build_grid_C() -> dict[str, callable]:
    """
    C — PQ-mixing: both p and q vary across groups.
    Q3 (C): does combining p-diversity with q-diversity further help?
    """
    return {
        "HAR":
            lambda: HARForecaster(ridge=False),
        # best single-group from main benchmark
        "SAS_q2_n200":
            lambda: _sas(200, p=1, q=2),
        # mix p-axis only (as in original MdSAS)
        "MdSAS_p01_q1":
            lambda: _md(100, [0, 1], [1, 1], grouped=True),
        # mix q-axis only (best from Grid B)
        "MixedQ_12":
            lambda: _md(100, [1, 1], [1, 2], grouped=True),
        # mix both: linear smoother (p=0,q=1) + gated quadratic (p=1,q=2)
        "MixedPQ_p01_q12":
            lambda: _md(100, [0, 1], [1, 2], grouped=True),
        # 3-way: (p=1,q=1) + (p=1,q=2) + (p=2,q=2) — richer combined
        "MixedPQ_p112_q122":
            lambda: _md(67, [1, 1, 2], [1, 2, 2], grouped=True),
    }


_GRIDS = {
    "A": ("A — P-mixing at n=200",    "A_pmix",  build_grid_A,
          ["HAR", "SAS_p1_n200", "MdSAS_p01_shared", "MdSAS_p01_grouped", "MdSAS_p012_grouped"]),
    "B": ("B — Q-mixing at n≈200",    "B_qmix",  build_grid_B,
          ["HAR", "SAS_q1_n200", "SAS_q2_n200", "MixedQ_12", "MixedQ_123", "MixedQ_23"]),
    "C": ("C — PQ-mixing at n≈200",   "C_pqmix", build_grid_C,
          ["HAR", "SAS_q2_n200", "MdSAS_p01_q1", "MixedQ_12", "MixedPQ_p01_q12", "MixedPQ_p112_q122"]),
}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_mse_csv(mse_by_sym, stable_syms, out_dir, slug):
    rows = []
    for sym, sym_mse in mse_by_sym.items():
        for model, h_dict in sym_mse.items():
            row = {"symbol": sym, "model": model, "stable": sym in stable_syms}
            for h in HORIZONS:
                row[f"mse_h{h}"] = h_dict.get(h, np.nan)
            row["mse_avg"] = float(np.nanmean([h_dict.get(h, np.nan) for h in HORIZONS]))
            rows.append(row)
    path = out_dir / f"{slug}_mse.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  → {path.name}  ({len(rows)} rows)")


def _save_mcs_csv(mcs_df, out_dir, slug):
    if mcs_df is None or (hasattr(mcs_df, "empty") and mcs_df.empty):
        return
    path = out_dir / f"{slug}_mcs.csv"
    mcs_df.to_csv(path, index=False)
    print(f"  → {path.name}")


# ── run one grid ──────────────────────────────────────────────────────────────

def run_grid(
    syms:        list[str],
    grid:        dict[str, callable],
    model_order: list[str],
    title:       str,
    slug:        str,
    out_dir:     Path,
    run_mcs:     bool,
    mcs_alpha:   float,
    mcs_nboot:   int,
    check_bounds: bool,
) -> None:
    mse_by_sym:  dict = {}
    loss_by_sym: dict = {}

    for sym in syms:
        try:
            log_vals, dates = load_rv(CSV, sym)
        except Exception as e:
            print(f"  [skip] {sym}: {e}")
            continue

        T = len(log_vals)
        if T < WINDOW + max(HORIZONS) + 2:
            print(f"  [skip] {sym}: T={T} too short")
            continue

        models = {name: factory() for name, factory in grid.items()}
        t0 = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            losses = quick_oos(
                log_vals, dates, models,
                horizons=HORIZONS, window=WINDOW, refit_freq=REFIT_FREQ,
            )

        dt = time.perf_counter() - t0
        sym_mse = {
            name: {h: float(np.mean(errs)) for h, errs in h_map.items()}
            for name, h_map in losses.items()
        }
        mse_by_sym[sym]  = sym_mse
        loss_by_sym[sym] = losses

        avg_str = "  ".join(
            f"{n}={float(np.nanmean(list(v.values()))):.4f}"
            for n, v in sym_mse.items()
        )
        print(f"  {sym:<12} {dt:5.1f}s  {avg_str}")

        # Alpha boundary check: re-fit once on last window to read alpha_log_
        if check_bounds:
            from data.data_loader import fit_scaler, apply_scaler
            train_raw = log_vals[-WINDOW:]
            mu, sigma = fit_scaler(train_raw)
            train_z   = apply_scaler(train_raw, mu, sigma)
            for name, model in models.items():
                if not hasattr(model, "alpha_log_"):
                    continue
                model.fit(train_z, HORIZONS)
                for h, alpha in model.alpha_log_.items():
                    grid_used = (model.alphas if hasattr(model, "alphas")
                                 else model.alphas_1d)
                    check_alpha_boundary(alpha, grid_used, name, sym, h)

    if not mse_by_sym:
        print("  No results.")
        return

    print()

    # ── stability ─────────────────────────────────────────────────────────────
    diverged: dict[str, list] = {sym: [] for sym in mse_by_sym}
    for sym, sym_mse in mse_by_sym.items():
        for m in model_order:
            vals = list(sym_mse.get(m, {}).values())
            if vals and float(np.nanmean(vals)) > DIVERGE_THRESH:
                diverged[sym].append(m)

    w = max(len(m) for m in model_order) + 2
    print(f"\n── Stability (avg MSE > {DIVERGE_THRESH}) {'─'*30}")
    for m in model_order:
        n_div = sum(1 for bad in diverged.values() if m in bad)
        n_tot = len(mse_by_sym)
        print(f"  {m:<{w}} {n_div:>2}/{n_tot}  {'█'*n_div}{'░'*(n_tot-n_div)}")

    stable_syms = [sym for sym, bad in diverged.items() if not bad]
    stable_mse  = {sym: mse_by_sym[sym] for sym in stable_syms}
    print(f"\n  Stable: {len(stable_syms)}/{len(mse_by_sym)}")

    # ── save MSE ──────────────────────────────────────────────────────────────
    _save_mse_csv(mse_by_sym, stable_syms, out_dir, slug)

    # ── MSE tables ────────────────────────────────────────────────────────────
    print_mse_table(
        stable_mse, HORIZONS,
        title=f"{title}  [stable {len(stable_syms)}/{len(mse_by_sym)}]",
        model_order=model_order,
    )
    if len(stable_syms) < len(mse_by_sym):
        print_mse_table(
            mse_by_sym, HORIZONS,
            title=f"{title}  [all — inflated by divergences]",
            model_order=model_order,
        )

    # ── MCS ───────────────────────────────────────────────────────────────────
    mcs_df = None
    if run_mcs and stable_syms:
        stable_eval = _sq_errors_to_eval_df(
            {sym: loss_by_sym[sym] for sym in stable_syms}
        )
        if stable_eval:
            mcs_df = print_mcs_frequency(
                stable_eval, HORIZONS,
                alpha=mcs_alpha, n_boot=mcs_nboot, seed=SEED,
                title=f"MCS₀.₁₀ — {title} [stable]",
                mse_by_symbol=stable_mse,
                model_order=model_order,
            )
            _save_mcs_csv(mcs_df, out_dir, slug)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-degree SAS ablation (P-mix, Q-mix, PQ-mix).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols",  nargs="+", default=["all"])
    parser.add_argument("--grid",     nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"],
                        help="Which grid(s) to run.")
    parser.add_argument("--no-mcs",       action="store_true")
    parser.add_argument("--no-bounds",    action="store_true",
                        help="Skip alpha boundary diagnostic (faster).")
    parser.add_argument("--mcs-alpha",    type=float, default=0.10)
    parser.add_argument("--mcs-nboot",    type=int,   default=1000)
    parser.add_argument("--out-dir",      default=None,
                        help="Output dir for CSVs. "
                             "Defaults to experiments/results_degree_ridge/<timestamp>/")
    args = parser.parse_args()

    if args.symbols == ["all"]:
        syms = available_symbols(CSV)
    else:
        syms = args.symbols

    # output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "experiments" / "results_degree_ridge" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults → {out_dir}")
    print(f"Running on {len(syms)} symbol(s): {' '.join(syms[:6])}"
          f"{'…' if len(syms) > 6 else ''}")
    print(f"Protocol: W={WINDOW}, R={REFIT_FREQ}, H={HORIZONS}, sn={SN}\n")

    for key in args.grid:
        title, slug, builder, order = _GRIDS[key]
        print(f"\n{'═'*70}")
        print(f"  Grid {key}: {title}")
        print(f"{'═'*70}\n")
        run_grid(
            syms        = syms,
            grid        = builder(),
            model_order = order,
            title       = title,
            slug        = slug,
            out_dir     = out_dir,
            run_mcs     = not args.no_mcs,
            mcs_alpha   = args.mcs_alpha,
            mcs_nboot   = args.mcs_nboot,
            check_bounds= not args.no_bounds,
        )

    print(f"\nAll results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
