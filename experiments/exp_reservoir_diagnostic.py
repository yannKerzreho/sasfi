"""
experiments/exp_reservoir_diagnostic.py — Reservoir state diagnostic.

Sweeps over (p_degree, q_degree) combinations and tracks:
  - Effective rank of the state matrix S  (key bottleneck metric)
  - Clip activation rate on P(z_t)
  - Degree-1 contribution: std(P(z_t) - P₀) as % of mean |P₀|
  - State autocorrelation by |P₀| quartile (slow vs fast dims)
  - State std by |P₀| quartile
  - Q injection scale: how much z_t^k grows with degree
  - State evolution: rolling std over time to detect regime sensitivity

Hypothesis being tested
-----------------------
  Adding p_degree doesn't increase effective rank (P₁ is too small by design).
  Adding q_degree DOES increase rank because it injects independent signals
  [1, z_t, z_t², ...] into the reservoir.
  BUT: higher q_degree also causes state explosion for large |z_t|.

Usage
-----
    python experiments/exp_reservoir_diagnostic.py
    python experiments/exp_reservoir_diagnostic.py --symbol .AEX
    python experiments/exp_reservoir_diagnostic.py --sn 0.95 --max-p 2 --max-q 4
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import load_rv
from models.sas_utils import DiagonalPoly

CSV          = ROOT / "rv.csv"
N            = 200
WASHOUT      = 50
SEED         = 42
CLIP_EPS     = 1e-4          # |P(z_t)| ≥ 0.9999 - ε counts as clipped
ROLL_WIN     = 100           # rolling window for state-std evolution


# ── data ──────────────────────────────────────────────────────────────────────

def load_z(symbol: str, window: int = 2000) -> np.ndarray:
    log_vals, _ = load_rv(CSV, symbol)
    train = log_vals[-window:]
    mu, sigma = float(train.mean()), float(train.std())
    return (train - mu) / sigma


# ── reservoir runner ───────────────────────────────────────────────────────────

def run_reservoir(sn: float, p_degree: int, q_degree: int,
                  zs: np.ndarray, n: int = N):
    """
    Roll DiagonalPoly forward; return states, P-history, and initialised arrays.

    Returns
    -------
    S       : (T-WASHOUT, n)  state matrix
    P_hist  : (T, n)          P(z_t) at each step  (clipped)
    Q_hist  : (T, n)          Q(z_t) injection at each step
    P0      : (n,)            base eigenvalues
    P1      : (n,) | None
    """
    basis = DiagonalPoly(p_degree=p_degree, q_degree=q_degree,
                         spectral_norm=sn)
    basis = basis.initialize(n, jax.random.PRNGKey(SEED))

    P0 = np.array(basis.P[0])
    P1 = np.array(basis.P[1]) if p_degree >= 1 else None

    s = jnp.zeros(n)
    states, p_hist, q_hist = [], [], []

    for t, z in enumerate(zs):
        z_jnp = jnp.array([float(z)])
        p_z   = basis.eval_p(z_jnp)          # (n,)  clipped
        q_z   = basis.eval_q(z_jnp)          # (n,)  un-clipped
        p_hist.append(np.array(p_z))
        q_hist.append(np.array(q_z))
        s = basis.apply(p_z, s) + q_z
        if t >= WASHOUT:
            states.append(np.array(s))

    return (np.stack(states), np.stack(p_hist),
            np.stack(q_hist), P0, P1)


# ── metrics ────────────────────────────────────────────────────────────────────

def effective_rank(S: np.ndarray) -> float:
    """Participation ratio: (Σσᵢ)² / Σσᵢ²."""
    sv = np.linalg.svd(S, compute_uv=False)
    sv = sv[sv > 0]
    return float((sv.sum() ** 2) / (sv ** 2).sum())


def top_k_variance(S: np.ndarray, k: int) -> float:
    sv = np.linalg.svd(S, compute_uv=False)
    return float((sv[:k] ** 2).sum() / (sv ** 2).sum())


def clip_rate(P_hist: np.ndarray) -> float:
    return float((np.abs(P_hist) >= 1.0 - CLIP_EPS).mean())


def p1_variation(P_hist: np.ndarray, P0: np.ndarray) -> tuple[float, float]:
    """Mean and max std of (P(z_t) - P₀) across dims."""
    dev = (P_hist - P0[None, :]).std(axis=0)
    return float(dev.mean()), float(dev.max())


def autocorr_lag1(S: np.ndarray) -> np.ndarray:
    ac = np.array([np.corrcoef(S[:-1, i], S[1:, i])[0, 1]
                   for i in range(S.shape[1])])
    return np.where(np.isnan(ac), 0.0, ac)


def q_scale_stats(Q_hist: np.ndarray) -> tuple[float, float, float]:
    """Std, 95th pct, and max of |Q(z_t)| across all dims and time."""
    q_abs = np.abs(Q_hist)
    return (float(q_abs.std()), float(np.percentile(q_abs, 95)),
            float(q_abs.max()))


def rolling_state_std(S: np.ndarray, win: int = ROLL_WIN) -> np.ndarray:
    """
    Rolling std of the mean-absolute state across dims.
    Shape: (T-WASHOUT-win+1,).  High values = turbulent periods.
    """
    mean_abs = np.abs(S).mean(axis=1)   # (T-WASHOUT,)
    return np.array([mean_abs[i:i+win].std()
                     for i in range(len(mean_abs) - win + 1)])


def state_explosion_rate(S: np.ndarray, threshold: float = 10.0) -> float:
    """Fraction of (t, i) pairs where |s_t_i| > threshold."""
    return float((np.abs(S) > threshold).mean())


def p0_quartile_table(arr: np.ndarray, P0: np.ndarray,
                      n_bins: int = 4, fmt: str = ".3f") -> list[str]:
    """Return list of formatted strings, one per |P₀| quartile."""
    abs_p0 = np.abs(P0)
    edges  = np.quantile(abs_p0, np.linspace(0, 1, n_bins + 1))
    lines  = []
    for q in range(n_bins):
        lo, hi = edges[q], edges[q + 1]
        hi_adj = hi + 1e-8 if q == n_bins - 1 else hi
        mask   = (abs_p0 >= lo) & (abs_p0 < hi_adj)
        vals   = arr[mask]
        tag    = "slow" if q == n_bins - 1 else ("fast" if q == 0 else "    ")
        lines.append(f"    [{lo:.2f},{hi:.2f}] {tag}  "
                     f"μ={vals.mean():{fmt}}  σ={vals.std():{fmt}}  "
                     f"max={vals.max():{fmt}}")
    return lines


# ── single-config analysis ────────────────────────────────────────────────────

def analyse(sn: float, p_degree: int, q_degree: int,
            zs: np.ndarray, verbose: bool = True) -> dict:
    """Run one config, print diagnostics, return summary dict."""
    label = f"p={p_degree} q={q_degree} sn={sn:.2f}"

    S, P_hist, Q_hist, P0, P1 = run_reservoir(sn, p_degree, q_degree, zs)

    # ── key metrics ───────────────────────────────────────────────────────────
    er        = effective_rank(S)
    top1_var  = top_k_variance(S, 1)
    top3_var  = top_k_variance(S, 3)
    top10_var = top_k_variance(S, 10)
    cr        = clip_rate(P_hist)
    p1_mean, p1_max = (p1_variation(P_hist, P0) if p_degree >= 1
                       else (0.0, 0.0))
    ac        = autocorr_lag1(S)
    s_std     = S.std(axis=0)
    q_std, q_p95, q_max = q_scale_stats(Q_hist)
    roll_std  = rolling_state_std(S)
    explosion = state_explosion_rate(S)

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  {label}   N={N}, washout={WASHOUT}")
        print(f"{'═'*70}")

        print(f"\n  [rank]  eff_rank={er:.1f}  "
              f"| top-1={top1_var*100:.1f}%  top-3={top3_var*100:.1f}%  "
              f"top-10={top10_var*100:.1f}% of variance")

        print(f"\n  [clip]  |P(z_t)| ≥ 0.9999 :  {cr*100:.3f}%  "
              + ("← essentially zero" if cr < 1e-4 else
                 "← moderate" if cr < 0.05 else "← HIGH"))

        if p_degree >= 1:
            p0_mean = np.abs(P0).mean()
            print(f"\n  [P₁ contribution]  std(P(z_t)−P₀):  "
                  f"mean={p1_mean:.5f}  max={p1_max:.5f}  "
                  f"= {p1_mean/p0_mean*100:.2f}% of mean|P₀|")
        else:
            print(f"\n  [P₁ contribution]  P(z_t) = P₀  (constant, p_degree=0)")

        print(f"\n  [Q injection]  std={q_std:.4f}  p95={q_p95:.4f}  "
              f"max={q_max:.4f}")
        print(f"  [state std]   mean={s_std.mean():.4f}  max={s_std.max():.4f}  "
              f"| explosion(|s|>10): {explosion*100:.3f}%")

        print(f"\n  [autocorr]  overall: mean={ac.mean():.3f}  "
              f"min={ac.min():.3f}  max={ac.max():.3f}")
        for line in p0_quartile_table(ac, P0):
            print(f"  {line}")

        # State volatility during market stress
        pct_stress = np.percentile(roll_std, 95)
        pct_calm   = np.percentile(roll_std, 5)
        print(f"\n  [state volatility]  rolling-{ROLL_WIN} std of mean|s_t|:  "
              f"calm(p5)={pct_calm:.4f}  stress(p95)={pct_stress:.4f}  "
              f"ratio={pct_stress/max(pct_calm, 1e-8):.1f}×")

    return dict(label=label, p=p_degree, q=q_degree, sn=sn,
                eff_rank=er, top1=top1_var, top3=top3_var, top10=top10_var,
                clip_rate=cr, p1_mean=p1_mean,
                q_std=q_std, q_p95=q_p95, q_max=q_max,
                s_std_mean=s_std.mean(), s_std_max=s_std.max(),
                explosion=explosion,
                ac_mean=ac.mean(), ac_max=ac.max(),
                roll_calm=float(np.percentile(roll_std, 5)),
                roll_stress=float(np.percentile(roll_std, 95)),
                S=S, P0=P0)


# ── summary table ──────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    sep = "─" * 100
    print(f"\n\n{'═'*100}")
    print("  SUMMARY TABLE — all (p_degree, q_degree) combinations")
    print(f"{'═'*100}")
    hdr = (f"{'config':<20}  {'eff_rank':>8}  {'top1%':>6}  {'top3%':>6}  "
           f"{'top10%':>6}  {'clip%':>6}  {'P₁%P₀':>7}  "
           f"{'q_std':>6}  {'q_max':>7}  {'s_max':>7}  "
           f"{'explo%':>7}  {'stress/calm':>11}")
    print(hdr)
    print(sep)
    for r in results:
        p1pct = f"{r['p1_mean']/0.001*0.001*100:.2f}" if r['p'] >= 1 else "  —   "
        # re-compute p1 as % of mean |P0| — store it properly
        p1pct = (f"{r['p1_mean']:.4f}" if r['p'] >= 1 else "  —   ")
        ratio = (r['roll_stress'] / max(r['roll_calm'], 1e-8))
        print(
            f"  {r['label']:<18}  {r['eff_rank']:>8.1f}  "
            f"{r['top1']*100:>5.1f}%  {r['top3']*100:>5.1f}%  "
            f"{r['top10']*100:>5.1f}%  {r['clip_rate']*100:>5.3f}%  "
            f"{p1pct:>7}  {r['q_std']:>6.4f}  {r['q_max']:>7.3f}  "
            f"{r['s_std_max']:>7.3f}  {r['explosion']*100:>6.3f}%  "
            f"{ratio:>8.1f}×"
        )
    print(sep)


# ── cross-config rank comparison ───────────────────────────────────────────────

def print_rank_grid(results: list[dict]) -> None:
    """Print effective rank in a (p_degree × q_degree) grid."""
    p_vals = sorted(set(r['p'] for r in results))
    q_vals = sorted(set(r['q'] for r in results))
    lookup = {(r['p'], r['q']): r['eff_rank'] for r in results}

    print(f"\n{'═'*60}")
    print("  Effective rank grid  (rows=p_degree, cols=q_degree)")
    print(f"{'═'*60}")
    header = "  p\\q   " + "  ".join(f"q={q:>2}" for q in q_vals)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for p in p_vals:
        row = f"  p={p}    "
        for q in q_vals:
            val = lookup.get((p, q), float('nan'))
            row += f"{val:>5.1f}  "
        print(row)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default=".SPX")
    parser.add_argument("--sn",      type=float, default=0.95)
    parser.add_argument("--max-p",   type=int,   default=2,
                        help="max p_degree to sweep (inclusive)")
    parser.add_argument("--max-q",   type=int,   default=4,
                        help="max q_degree to sweep (inclusive)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet",   action="store_true",
                        help="only print summary table (no per-config details)")
    args = parser.parse_args()
    verbose = not args.quiet

    print(f"\nReservoir diagnostic — {args.symbol}  sn={args.sn}  N={N}")
    zs = load_z(args.symbol)
    print(f"Input z stats:  mean={zs.mean():.3f}  std={zs.std():.3f}  "
          f"min={zs.min():.2f}  max={zs.max():.2f}  "
          f"|z|>2: {(np.abs(zs)>2).mean()*100:.1f}%  "
          f"|z|>3: {(np.abs(zs)>3).mean()*100:.1f}%")

    # Q injection scale: theoretical z_t^k std for standard normal
    print("\n  Theoretical std of z^k for z~N(0,1):")
    from math import factorial
    for k in range(1, args.max_q + 2):
        # E[z^{2k}] = (2k-1)!! for standard normal
        e2k = 1
        for i in range(1, 2*k, 2):
            e2k *= i
        std_zk = e2k ** 0.5
        print(f"    k={k}: std(z^{k}) = {std_zk:.2f}  "
              f"→ Q_k·z^k has std ≈ {std_zk/N**0.5:.4f} (Q_k ~ N(0,1/n))")

    # ── sweep ────────────────────────────────────────────────────────────────
    configs = [(p, q)
               for p in range(0, args.max_p + 1)
               for q in range(1, args.max_q + 1)]

    results = []
    for p_deg, q_deg in configs:
        r = analyse(args.sn, p_deg, q_deg, zs, verbose=verbose)
        results.append(r)

    # ── summary ───────────────────────────────────────────────────────────────
    print_rank_grid(results)
    print_summary_table(results)

    # ── cross-q correlation: do q=2 states look different from q=1? ──────────
    print(f"\n{'═'*70}")
    print("  Cross-q correlation: are higher-q states truly different?")
    print(f"  (same sn={args.sn}, p_degree=0; vary q_degree)")
    print(f"{'═'*70}")

    ref_p, ref_q = 0, 1
    ref = next((r for r in results if r['p'] == ref_p and r['q'] == ref_q), None)
    if ref is not None:
        for r in results:
            if r['p'] != ref_p or r['q'] == ref_q:
                continue
            S0, S1 = ref['S'], r['S']
            T = min(S0.shape[0], S1.shape[0])
            per_col = np.array([np.corrcoef(S0[:T, i], S1[:T, i])[0, 1]
                                for i in range(N)])
            per_col = np.where(np.isnan(per_col), 0, per_col)
            print(f"\n  q=1 vs q={r['q']}  (p=0, sn={args.sn}):")
            print(f"    per-column corr:  "
                  f"mean={per_col.mean():.4f}  "
                  f"median={np.median(per_col):.4f}  "
                  f"min={per_col.min():.4f}  max={per_col.max():.4f}")
            print(f"    |corr|>0.99: {(np.abs(per_col)>0.99).sum()}/{N} dims  "
                  f"  |corr|>0.999: {(np.abs(per_col)>0.999).sum()}/{N} dims")


if __name__ == "__main__":
    main()
