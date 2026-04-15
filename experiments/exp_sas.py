"""
experiments/exp_sas.py — SAS hyperparameter sweep.

Goal: find the best SAS configuration to compete with NLinear (MSE≈0.52)
on .AEX rv5 data.

Protocol
--------
Rolling OOS with window=500, refit every 252 steps.
HAR is always included as the reference baseline.

Grid
----
  basis          : diagonal (stable), linear (n≤200, may overfit),
                   trigo (trig features, n≤100)
  n_reservoir    : 50–1000 (diagonal), 50–100 (linear/trigo)
  spectral_norm  : 0.80, 0.90, 0.95, 0.99
  p_degree       : 1, 2  (polynomial degree for P(z))
  washout        : 25, 50, 100
  K (ensemble)   : 1, 5
  augmented (HAR): True, False

All polynomials are JAX pytrees — passed directly to the JIT parallel scan.
Results saved to experiments/results_sas.csv.
Run from repo root: python experiments/exp_sas.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader      import load_rv, fit_scaler, apply_scaler
from models.linear         import HARForecaster
from models.sas            import SASForecaster, SASEnsemble, AugSASForecaster
from models.sas_utils      import DiagonalPoly, LinearPoly, TrigoPoly

HORIZONS    = [1, 5, 22]
WINDOW      = 500
REFIT_FREQ  = 252


# ── lightweight rolling OOS ───────────────────────────────────────────────────

def quick_oos(log_values, dates, models_dict, max_steps=None):
    """Returns dict {name: {h: [sq_errors]}}."""
    T     = len(log_values)
    H_max = max(HORIZONS)
    mu, sigma = 0.0, 1.0
    steps_since_refit = {n: REFIT_FREQ for n in models_dict}
    losses = {n: {h: [] for h in HORIZONS} for n in models_dict}

    limit = (WINDOW + max_steps) if max_steps else (T - H_max)

    for t in range(WINDOW, min(limit, T - H_max)):
        train_raw = log_values[t - WINDOW: t]

        for name, model in models_dict.items():
            if steps_since_refit[name] >= REFIT_FREQ:
                mu, sigma = fit_scaler(train_raw)
                train_z   = apply_scaler(train_raw, mu, sigma)
                try:
                    model.fit(train_z, HORIZONS)
                    steps_since_refit[name] = 0
                except Exception as e:
                    print(f"  [fit {name}] {e}")

        z_t = apply_scaler(float(log_values[t]), mu, sigma)

        for name, model in models_dict.items():
            for h in HORIZONS:
                if t + h >= T:
                    continue
                z_target = apply_scaler(float(log_values[t + h]), mu, sigma)
                try:
                    y_hat = float(model.predict(h))
                    losses[name][h].append((y_hat - z_target) ** 2)
                except Exception as e:
                    print(f"  [pred {name}] {e}")

        for name, model in models_dict.items():
            try:
                model.update(z_t)
                steps_since_refit[name] += 1
            except Exception as e:
                print(f"  [upd {name}] {e}")

    return {n: {h: np.mean(v) if v else np.nan
                for h, v in hd.items()}
            for n, hd in losses.items()}


# ── grid ─────────────────────────────────────────────────────────────────────

def build_grid():
    configs = []

    # ── Baselines ────────────────────────────────────────────────────────────
    configs.append(("HAR",       lambda: HARForecaster(ridge=False)))
    configs.append(("HAR_Ridge", lambda: HARForecaster(ridge=True)))

    # ── Diagonal basis — O(n) per step, any n ───────────────────────────────
    # p_degree=1 (linear modulation of eigenvalues)
    for n in [100, 200, 500]:
        for sn in [0.90, 0.95, 0.99]:
            tag = f"diag_p1_n{n}_sn{sn:.2f}"
            configs.append((tag, lambda n=n, sn=sn:
                SASForecaster(n_reservoir=n, basis="diagonal",
                              spectral_norm=sn, washout=50, seed=42)))

    # p_degree=2 (quadratic input-modulation — richer timescale dynamics)
    for n in [100, 200]:
        for sn in [0.90, 0.95, 0.99]:
            tag = f"diag_p2_n{n}_sn{sn:.2f}"
            configs.append((tag, lambda n=n, sn=sn:
                SASForecaster(
                    n_reservoir=n,
                    basis=DiagonalPoly(p_degree=2, q_degree=2, spectral_norm=sn),
                    washout=50, seed=42,
                )))
    
    for n in [100, 200]:
        for sn in [0.90, 0.95, 0.99]:
            tag = f"diag_p3_n{n}_sn{sn:.2f}"
            configs.append((tag, lambda n=n, sn=sn:
                SASForecaster(
                    n_reservoir=n,
                    basis=DiagonalPoly(p_degree=3, q_degree=3, spectral_norm=sn),
                    washout=50, seed=42,
                )))

    # ── Linear basis — O(n²) per step, n ≤ 200 ─────────────────────────────
    for n in [100, 200]:
        for sn in [0.80, 0.90, 0.95]:
            tag = f"lin_n{n}_sn{sn:.2f}"
            configs.append((tag, lambda n=n, sn=sn, wo=50:
                SASForecaster(n_reservoir=n, basis="linear",
                                spectral_norm=sn, washout=wo, seed=42)))

    # ── Trigo basis — full matrix, trig features ─────────────────────────────
    for n in [50, 100]:
        for sn in [0.9, 0.95]:
            for p in [1, 2]:
                tag = f"trigo_p{p}_n{n}_sn{sn:.2f}"
                configs.append((tag, lambda n=n, sn=sn, p=p:
                    SASForecaster(
                        n_reservoir=n,
                        basis=TrigoPoly(p_degree=p, q_degree=1, spectral_norm=sn),
                        washout=50, seed=42,
                    )))

    # ── Ensemble (diagonal, best expected configs) ───────────────────────────
    for n in [100, 200, 500]:
        for sn in [0.8, 0.90, 0.95]:
            tag = f"ens5_diag_n{n}_sn{sn:.2f}"
            configs.append((tag, lambda n=n, sn=sn:
                SASEnsemble(K=5, n_reservoir=n, basis="diagonal",
                            spectral_norm=sn, washout=50, seed=0)))

    # ── Augmented SAS (reservoir + HAR features) ─────────────────────────────
    # for n in [100, 200, 500]:
    #     for sn in [0.90, 0.95]:
    #         tag = f"aug_diag_n{n}_sn{sn:.2f}"
    #         configs.append((tag, lambda n=n, sn=sn:
    #             AugSASForecaster(n_reservoir=n, basis="diagonal",
    #                              spectral_norm=sn, washout=50, seed=42)))

    return configs


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    log_values, dates = load_rv(ROOT / "rv.csv", symbol=".AEX", target="rv5")
    print(f"  T={len(log_values)}\n")

    grid    = build_grid()
    records = []

    for i, (tag, factory) in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] {tag}")
        model  = factory()
        result = quick_oos(log_values, dates, {tag: model})
        row    = {"config": tag}
        for h in HORIZONS:
            row[f"mse_h{h}"] = result[tag][h]
        row["mse_avg"] = float(np.nanmean([result[tag][h] for h in HORIZONS]))
        records.append(row)
        print(f"       h1={row['mse_h1']:.4f}  h5={row['mse_h5']:.4f}"
              f"  h22={row['mse_h22']:.4f}  avg={row['mse_avg']:.4f}")

    df = pd.DataFrame(records).sort_values("mse_avg")
    out = ROOT / "experiments" / "results_sas.csv"
    df.to_csv(out, index=False)

    print(f"\n{'─'*70}")
    print(f"{'Config':<35} {'h=1':>7} {'h=5':>7} {'h=22':>8} {'avg':>7}")
    print(f"{'─'*70}")
    for _, r in df.head(20).iterrows():
        print(f"{r['config']:<35} {r['mse_h1']:>7.4f} {r['mse_h5']:>7.4f}"
              f" {r['mse_h22']:>8.4f} {r['mse_avg']:>7.4f}")
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
