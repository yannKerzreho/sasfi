"""
experiments/exp_basis.py — SAS Basis comparison (Diagonal vs BlockLinear vs BlockTrigo).

Context / Protocol
------------------
  • Baseline: HAR (ridge=False).
  • Diagonal: n=200 (cheap O(n) scaling, high dimensionality).
  • BlockLinear & BlockTrigo: 10 blocks of 10x10 (n=100). 
    Tests if local dense mixing (via batched block matmuls) provides better 
    expressivity than pure diagonal, without the O(n^3) parallel scan penalty.
  • Rolling OOS: W=2000, R=20, h∈{1,5,10}.

Usage
-----
    python experiments/exp_basis.py                       # all symbols
    python experiments/exp_basis.py --symbols .AEX .SPX   # quick check
    python experiments/exp_basis.py --no-mcs              # skip MCS
"""

from __future__ import annotations
import sys
import time
import argparse
import warnings
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
from models.sas       import SASForecaster, MultiDegreeSASForecaster

# Ajuste cet import selon le fichier où tu as placé BlockLinearPoly et BlockTrigoPoly
from models.sas_utils import BlockLinearPoly, BlockTrigoPoly

CSV        = ROOT / "rv.csv"
HORIZONS   = [1, 5, 10]
WINDOW     = 2000
REFIT_FREQ = 20
SEED       = 42
WASHOUT    = 50
DIVERGE_THRESH = 1.5


def _sas_string(n=200, basis="diagonal", sn=0.95, p=1, q=1):
    """Factory for standard string-based bases (like diagonal)."""
    return SASForecaster(
        n_reservoir=n, basis=basis,
        spectral_norm=sn, p_degree=p, q_degree=q,
        washout=WASHOUT, seed=SEED,
    )

def _sas_block(poly_class, n_blocks=10, block_size=10, sn=0.95, p=1, q=1):
    """Factory for custom block-diagonal bases passing the instance directly."""
    basis_instance = poly_class(
        n_blocks=n_blocks, 
        block_size=block_size, 
        p_degree=p, 
        q_degree=q, 
        spectral_norm=sn
    )
    return SASForecaster(
        n_reservoir=n_blocks * block_size, 
        basis=basis_instance, # On passe l'instance au lieu d'un string
        washout=WASHOUT, 
        seed=SEED,
    )


def build_basis_grid() -> dict[str, callable]:
    grid = {
        "HAR": lambda: HARForecaster(ridge=False),
        
        # Baseline pure diagonale (200 neurones, gating linéaire)
        "diag_200_p1q1": lambda: SASForecaster(
            n_reservoir=200, basis="diagonal", p_degree=1, q_degree=1
        ),
        
        # Multi-Degree: 100 filtres linéaires classiques (p=0) + 100 filtres à gating (p=1)
        "multi_diag_p0_p1": lambda: MultiDegreeSASForecaster(
            n_per_group=100, p_degrees=[0, 1], q_degrees=[1, 1], grouped_ridge=True
        ),
        
        # Multi-Degree avec injection non-linéaire (q=2) sur le gating
        "multi_diag_p0_p1q2": lambda: MultiDegreeSASForecaster(
            n_per_group=100, p_degrees=[0, 1], q_degrees=[1, 2], grouped_ridge=True
        ),
        
        # L'artillerie lourde : 3 sous-groupes (3 x 66 = 198 neurones)
        # Groupe 1: Linéaire (p=0)
        # Groupe 2: Gating standard (p=1, q=1)
        # Groupe 3: Gating aux extrêmes (p=1, q=2)
        "multi_diag_triplet": lambda: MultiDegreeSASForecaster(
            n_per_group=66, p_degrees=[0, 1, 1], q_degrees=[1, 1, 2], grouped_ridge=True
        ),
    }
    return grid


# ── per-symbol win tracker ─────────────────────────────────────────────────────

def print_per_symbol_h1(
    mse_by_sym: dict[str, dict[str, dict[int, float]]],
    models: list[str],
) -> None:
    print("\n── Per-symbol h=1 winner ──────────────────────────────────────")
    wins = {m: 0 for m in models}
    for sym, sym_mse in sorted(mse_by_sym.items()):
        row = {m: sym_mse.get(m, {}).get(1, np.nan) for m in models}
        valid = {m: v for m, v in row.items() if np.isfinite(v)}
        if not valid:
            continue
        best_val = min(valid.values())
        best_m   = min(valid, key=valid.get)
        wins[best_m] += 1
        sorted_vals = sorted(valid.values())
        margin = (sorted_vals[1] - sorted_vals[0]) if len(sorted_vals) > 1 else 0
        marker = "~" if margin < 0.001 else " "
        vals_str = "  ".join(f"{m}={v:.4f}" for m, v in sorted(valid.items()))
        print(f"  {marker}{sym:<12} best={best_m}  {vals_str}")
    print(f"\n  Win counts (h=1): " +
          "  ".join(f"{m}={wins[m]}" for m in models if wins[m] > 0))


def print_stability(
    mse_by_sym: dict[str, dict[str, dict[int, float]]],
    models: list[str],
    thresh: float = DIVERGE_THRESH,
) -> list[str]:
    print(f"\n── Stability (avg MSE > {thresh}) ────────────────────────────")
    w = max(len(m) for m in models) + 2
    diverged_syms: dict[str, list[str]] = {sym: [] for sym in mse_by_sym}
    for sym, sym_mse in mse_by_sym.items():
        for m in models:
            vals = list(sym_mse.get(m, {}).values())
            if vals and float(np.nanmean(vals)) > thresh:
                diverged_syms[sym].append(m)

    for m in models:
        n_div = sum(1 for sym, bad in diverged_syms.items() if m in bad)
        n_tot = len(mse_by_sym)
        bar   = "█" * n_div + "░" * (n_tot - n_div)
        print(f"  {m:<{w}} {n_div:>2}/{n_tot}  {bar}")

    stable = [sym for sym, bad in diverged_syms.items() if not bad]
    print(f"\n  Stable symbols ({len(stable)}/{len(mse_by_sym)}): "
          f"{' '.join(sorted(stable)[:8])}"
          f"{'…' if len(stable) > 8 else ''}")
    return stable


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _save_mse_csv(
    mse_by_sym: dict, stable_syms: list[str],
    horizons: list[int], out_dir: Path, slug: str,
) -> None:
    rows = []
    for sym, sym_mse in mse_by_sym.items():
        for model, h_dict in sym_mse.items():
            row = {"symbol": sym, "model": model, "stable": sym in stable_syms}
            for h in horizons:
                row[f"mse_h{h}"] = h_dict.get(h, np.nan)
            row["mse_avg"] = float(np.nanmean([h_dict.get(h, np.nan) for h in horizons]))
            rows.append(row)
    path = out_dir / f"{slug}_mse.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  → saved {path.name}  ({len(rows)} rows)")


def _save_mcs_csv(mcs_df: pd.DataFrame, out_dir: Path, slug: str) -> None:
    if mcs_df is None or mcs_df.empty:
        return
    path = out_dir / f"{slug}_mcs.csv"
    mcs_df.to_csv(path, index=False)
    print(f"  → saved {path.name}")


# ── run one grid ──────────────────────────────────────────────────────────────

def run_grid(
    syms: list[str],
    grid: dict[str, callable],
    model_order: list[str],
    title: str,
    slug: str,
    out_dir: Path,
    run_mcs: bool,
    mcs_alpha: float,
    mcs_nboot: int,
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

    if not mse_by_sym:
        print("  No results.")
        return

    print()

    stable_syms = print_stability(mse_by_sym, model_order)
    stable_mse  = {sym: mse_by_sym[sym] for sym in stable_syms}

    _save_mse_csv(mse_by_sym, stable_syms, HORIZONS, out_dir, slug)
    print_per_symbol_h1(mse_by_sym, model_order)

    print_mse_table(
        stable_mse, HORIZONS,
        title=f"{title}  [stable {len(stable_syms)}/{len(mse_by_sym)} symbols]",
        model_order=model_order,
    )
    if len(stable_syms) < len(mse_by_sym):
        print_mse_table(
            mse_by_sym, HORIZONS,
            title=f"{title}  [all {len(mse_by_sym)} symbols — means inflated by divergences]",
            model_order=model_order,
        )

    mcs_df = None
    if run_mcs:
        stable_eval = _sq_errors_to_eval_df(
            {sym: loss_by_sym[sym] for sym in stable_syms if sym in loss_by_sym}
        )
        if stable_eval:
            mcs_df = print_mcs_frequency(
                stable_eval, HORIZONS,
                alpha=mcs_alpha, n_boot=mcs_nboot, seed=SEED,
                title=f"MCS₀.₁₀ — {title} [stable subset]",
                mse_by_symbol=stable_mse,
                model_order=model_order,
            )
            _save_mcs_csv(mcs_df, out_dir, slug)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basis Comparison: Diagonal vs Block matrices (10x10).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", nargs="+", default=["all"])
    parser.add_argument("--no-mcs",    action="store_true")
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-nboot", type=int,   default=1000)
    parser.add_argument("--out-dir",   default=None,
                        help="Output directory for CSV results.")
    args = parser.parse_args()

    if args.symbols == ["all"]:
        syms = available_symbols(CSV)
    else:
        syms = args.symbols

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from datetime import datetime
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "experiments" / "results_basis" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nResults will be saved to: {out_dir}")
    print(f"Running on {len(syms)} symbol(s): {' '.join(syms[:6])}{'…' if len(syms) > 6 else ''}")
    print(f"Protocol: W={WINDOW}, R={REFIT_FREQ}, H={HORIZONS}\n")

    grid_dict = build_basis_grid()
    model_order = list(grid_dict.keys())

    print(f"\n{'═'*70}")
    print(f"  Basis Architecture Sweep (Diagonal vs BlockLinear vs BlockTrigo)")
    print(f"{'═'*70}\n")
    
    run_grid(
        syms, grid_dict, model_order,
        title     = "Basis Sweep",
        slug      = "basis_sweep",
        out_dir   = out_dir,
        run_mcs   = not args.no_mcs,
        mcs_alpha = args.mcs_alpha,
        mcs_nboot = args.mcs_nboot,
    )

    print(f"\nAll results saved to: {out_dir}/")


if __name__ == "__main__":
    main()