"""
experiments/exp_alpha.py — Ridge alpha diagnostic (multi-symbol).

Runs one full rolling OOS on HAR_Ridge, NLinear, DLinear, SAS_90
and logs the ridge alpha chosen at every refit.

For each model × horizon this prints:
  • distribution of chosen alphas (ASCII histogram)
  • % of refits that hit the boundary (first or last grid value)
  • recommendation: whether the grid range should be extended

The alpha grid is  _ALPHAS = [10^x  for x in arange(-3, 4.01, 0.25)]
i.e. 29 log-spaced values from 0.001 to 10000.

If ≥ 20% of choices hit the boundary the range likely needs extending.

Usage
-----
  # All available symbols (default):
      python experiments/exp_alpha.py

  # Specific symbols:
      python experiments/exp_alpha.py --symbols .AEX .SPX .FTSE

  # Single symbol, quick check:
      python experiments/exp_alpha.py --symbols .AEX

Run from repo root.
"""

from __future__ import annotations
import sys
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import load_rv, fit_scaler, apply_scaler, available_symbols
from models.linear    import HARForecaster, NLinearForecaster, DLinearForecaster, _ALPHAS
from models.sas       import SASForecaster

HORIZONS   = [1, 5, 22]
WINDOW     = 2000
REFIT_FREQ = 20
CSV        = ROOT / "rv.csv"


# ── alpha logging OOS ─────────────────────────────────────────────────────────

def run_with_alpha_log(log_values, models):
    """
    Same rolling OOS as quick_oos, but after every refit we harvest
    model.alpha_log_ and accumulate the per-horizon alpha choices.
    """
    T     = len(log_values)
    H_max = max(HORIZONS)
    mu, sigma = 0.0, 1.0
    steps_since_refit = {n: REFIT_FREQ for n in models}
    # alpha_records[model][h] = list of chosen alphas
    alpha_records: dict[str, dict[int, list]] = {
        n: {h: [] for h in HORIZONS} for n in models
    }

    for t in range(WINDOW, T - H_max):
        train_raw = log_values[t - WINDOW: t]

        for name, model in models.items():
            if steps_since_refit[name] >= REFIT_FREQ:
                mu, sigma = fit_scaler(train_raw)
                train_z   = apply_scaler(train_raw, mu, sigma)
                try:
                    model.fit(train_z, HORIZONS)
                    steps_since_refit[name] = 0
                    # Harvest alpha log
                    alog = getattr(model, "alpha_log_", {})
                    for h, alpha in alog.items():
                        if h in alpha_records[name]:
                            alpha_records[name][h].append(alpha)
                except Exception as e:
                    print(f"  [fit {name}] {e}")

        z_t = apply_scaler(float(log_values[t]), mu, sigma)
        for name, model in models.items():
            try:
                model.update(z_t)
                steps_since_refit[name] += 1
            except Exception:
                pass

    return alpha_records


# ── ASCII histogram ────────────────────────────────────────────────────────────

def ascii_hist(values: list[float], alphas: list[float], width: int = 30) -> str:
    """Bar chart of how often each alpha in the grid was chosen."""
    counts = Counter(values)
    max_c  = max(counts.values()) if counts else 1
    lines  = []
    for a in alphas:
        c   = counts.get(a, 0)
        bar = "█" * int(width * c / max_c)
        lines.append(f"  {a:>9.4g} │{bar:<{width}}│ {c}")
    return "\n".join(lines)


# ── boundary check ────────────────────────────────────────────────────────────

def boundary_pct(values: list[float], alphas: list[float]) -> float:
    """Fraction of choices that hit the first or last alpha value."""
    if not values:
        return 0.0
    lo, hi = min(alphas), max(alphas)
    hits   = sum(1 for v in values if v == lo or v == hi)
    return hits / len(values)


# ── per-symbol analysis ───────────────────────────────────────────────────────

def analyze_symbol(symbol: str, all_warnings: list[str]) -> dict:
    """
    Run the alpha-logging OOS for one symbol.
    Returns a summary dict {model: {h: mode_alpha}}.
    """
    log_values, _ = load_rv(CSV, symbol=symbol, target="rv5")
    T = len(log_values)
    if T < WINDOW + max(HORIZONS) + 2:
        print(f"  Skipping {symbol}: only {T} observations (need ≥{WINDOW + max(HORIZONS) + 2})")
        return {}

    # Fresh model instances for each symbol
    models = {
        "HAR_Ridge": HARForecaster(ridge=True),
        "NLinear":   NLinearForecaster(lookback=20),
        "DLinear":   DLinearForecaster(lookback=20),
        "SAS_90":    SASForecaster(n_reservoir=100, basis="diagonal",
                                   spectral_norm=0.90, washout=50, seed=42),
    }

    print(f"\n  Running OOS for {symbol}  (T={T}) …")
    alpha_records = run_with_alpha_log(log_values, models)

    print(f"\n{'─'*60}")
    print(f"  ALPHA DISTRIBUTION — {symbol}")
    print(f"{'─'*60}")

    summary: dict[str, dict[int, str]] = {}
    for name, h_dict in alpha_records.items():
        print(f"\n▶ {name}")
        summary[name] = {}
        for h in HORIZONS:
            vals = h_dict[h]
            if not vals:
                print(f"  h={h}: no data")
                summary[name][h] = "—"
                continue
            bp  = boundary_pct(vals, _ALPHAS)
            arr = np.array(vals)
            print(f"\n  h={h}  n_refits={len(vals)}"
                  f"  median={np.median(arr):.4g}"
                  f"  mean={np.mean(arr):.4g}"
                  f"  boundary_hits={bp:.0%}")
            print(ascii_hist(vals, _ALPHAS))
            if bp >= 0.20:
                msg = (f"  ⚠  [{symbol}] {name} h={h}: "
                       f"{bp:.0%} boundary hits — consider extending alpha grid")
                all_warnings.append(msg)
            mc = Counter(vals).most_common(1)[0]
            summary[name][h] = f"{mc[0]:.4g} ({mc[1]}×)"

    return summary


# ── aggregate summary across symbols ─────────────────────────────────────────

def print_aggregate(
    symbol_summaries: dict[str, dict[str, dict[int, str]]],
    model_names: list[str],
) -> None:
    """
    Print a compact cross-symbol comparison: for each model × horizon,
    show the most common alpha chosen across all symbols.
    """
    print(f"\n{'═'*70}")
    print("  CROSS-SYMBOL SUMMARY: most-frequent alpha per model × horizon")
    print(f"{'═'*70}")

    rows = []
    for name in model_names:
        row = {"model": name}
        for h in HORIZONS:
            all_vals_str = []
            for sym, summ in symbol_summaries.items():
                v = summ.get(name, {}).get(h, "—")
                if v != "—":
                    all_vals_str.append(f"{sym}:{v}")
            row[f"h{h}"] = "  ".join(all_vals_str) if all_vals_str else "—"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string())


# ── boundary heat-map across symbols ─────────────────────────────────────────

def print_boundary_heatmap(
    records_by_symbol: dict[str, dict[str, dict[int, list]]],
    model_names: list[str],
) -> None:
    """
    Show boundary-hit % for every (symbol, model, h) as a compact table.
    """
    print(f"\n{'═'*70}")
    print("  BOUNDARY HIT % — (rows=symbol × model, cols=horizon)")
    print(f"{'═'*70}")
    rows = []
    for sym, alpha_records in records_by_symbol.items():
        for name in model_names:
            h_dict = alpha_records.get(name, {})
            row = {"symbol": sym, "model": name}
            for h in HORIZONS:
                vals = h_dict.get(h, [])
                row[f"h{h}"] = f"{boundary_pct(vals, _ALPHAS):.0%}" if vals else "—"
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["symbol", "model"])
    print(df.to_string())


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ridge alpha diagnostic (multi-symbol).")
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Symbols to analyse. Defaults to all available symbols in rv.csv.",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = available_symbols(CSV)
        print(f"No --symbols provided; using all {len(symbols)} available: {symbols}")

    print(f"\nAlpha grid ({len(_ALPHAS)} values): {_ALPHAS[0]:.4g} … {_ALPHAS[-1]:.4g}\n")

    model_names = ["HAR_Ridge", "NLinear", "DLinear", "SAS_90"]
    all_warnings: list[str] = []
    symbol_summaries:  dict[str, dict[str, dict[int, str]]]  = {}
    records_by_symbol: dict[str, dict[str, dict[int, list]]] = {}

    for symbol in symbols:
        print(f"\n{'═'*60}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'═'*60}")
        try:
            log_values, _ = load_rv(CSV, symbol=symbol, target="rv5")
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        T = len(log_values)
        if T < WINDOW + max(HORIZONS) + 2:
            print(f"  Skipping: only {T} observations")
            continue

        # Fresh model instances
        models = {
            "HAR_Ridge": HARForecaster(ridge=True),
            "NLinear":   NLinearForecaster(lookback=20),
            "DLinear":   DLinearForecaster(lookback=20),
            "SAS_90":    SASForecaster(n_reservoir=100, basis="diagonal",
                                       spectral_norm=0.90, washout=50, seed=42),
        }

        print(f"  T={T} — running rolling OOS with alpha logging …")
        alpha_records = run_with_alpha_log(log_values, models)
        records_by_symbol[symbol] = alpha_records

        print(f"\n{'─'*60}")
        print(f"  ALPHA DISTRIBUTION — {symbol}")
        print(f"{'─'*60}")

        summ: dict[str, dict[int, str]] = {}
        for name, h_dict in alpha_records.items():
            print(f"\n▶ {name}")
            summ[name] = {}
            for h in HORIZONS:
                vals = h_dict[h]
                if not vals:
                    print(f"  h={h}: no data")
                    summ[name][h] = "—"
                    continue
                bp  = boundary_pct(vals, _ALPHAS)
                arr = np.array(vals)
                print(f"\n  h={h}  n_refits={len(vals)}"
                      f"  median={np.median(arr):.4g}"
                      f"  mean={np.mean(arr):.4g}"
                      f"  boundary_hits={bp:.0%}")
                print(ascii_hist(vals, _ALPHAS))
                if bp >= 0.20:
                    msg = (f"  ⚠  [{symbol}] {name} h={h}: {bp:.0%} boundary hits")
                    all_warnings.append(msg)
                mc = Counter(vals).most_common(1)[0]
                summ[name][h] = f"{mc[0]:.4g} ({mc[1]}×)"
        symbol_summaries[symbol] = summ

    # ── cross-symbol tables ───────────────────────────────────────────────
    if len(records_by_symbol) > 1:
        print_boundary_heatmap(records_by_symbol, model_names)

    # ── warnings ─────────────────────────────────────────────────────────
    if all_warnings:
        print(f"\n{'═'*70}")
        print("  WARNINGS")
        print(f"{'═'*70}")
        for w in all_warnings:
            print(w)
    else:
        print(f"\n✓ No boundary issues — alpha grid [{_ALPHAS[0]:.4g} … {_ALPHAS[-1]:.4g}] looks adequate.")

    # ── per-symbol summary table ──────────────────────────────────────────
    for symbol, summ in symbol_summaries.items():
        print(f"\n{'═'*60}")
        print(f"  SUMMARY [{symbol}]: most-frequent alpha per model × horizon")
        print(f"{'═'*60}")
        rows = []
        for name in model_names:
            row = {"model": name}
            for h in HORIZONS:
                row[f"h{h}_mode"] = summ.get(name, {}).get(h, "—")
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        print(df.to_string())


if __name__ == "__main__":
    main()
