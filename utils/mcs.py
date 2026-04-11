"""
mcs.py — Model Confidence Set (Hansen, Lunde & Nason 2011)
==========================================================

Implements the MCS elimination procedure with the range statistic T_R and
a circular block bootstrap for time-series loss sequences.

Two analysis modes
------------------
1. Per-horizon    : one MCS per (period, horizon).
2. Grouped-horizon: stack h1-h2 into a single loss sequence before running MCS.
   Triples/doubles the effective sample size and increases statistical power.
   For the pooled "full" period the rows are also row-normalised (each cell ÷
   cross-model arithmetic mean) so that high-variance periods (e.g. COVID)
   do not dominate the bootstrap variance.  Block size is rounded up to the
   next multiple of the group width so every bootstrap block covers a complete
   set of horizon observations.

Reference
---------
Hansen, P. R., Lunde, A., & Nason, J. M. (2011).
  The Model Confidence Set. Econometrica, 79(2), 453–497.
"""

import numpy as np
import pandas as pd


# ── circular block bootstrap ───────────────────────────────────────────────

def _cbb_indices(T: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw T indices via circular block bootstrap."""
    n_blocks = int(np.ceil(T / block_size))
    starts   = rng.integers(0, T, size=n_blocks)
    idx      = np.concatenate([np.arange(s, s + block_size) % T for s in starts])
    return idx[:T]


# ── core MCS ──────────────────────────────────────────────────────────────

def model_confidence_set(
    losses:     pd.DataFrame,
    alpha:      float = 0.10,
    block_size: int   = None,
    n_boot:     int   = 1000,
    seed:       int   = 42,
) -> tuple[list, dict]:
    """
    Model Confidence Set elimination procedure.

    Parameters
    ----------
    losses     : DataFrame [T, M] — per-observation losses, columns = model names.
    alpha      : significance level (default 0.10).
    block_size : block length for circular block bootstrap.
                 Defaults to max(1, floor(T^(1/3))).
    n_boot     : number of bootstrap replications.
    seed       : random seed.

    Returns
    -------
    mcs_set  : list of model names surviving in the MCS.
    p_elim   : dict {model_name: p-value at elimination} for eliminated models.
               Models in the MCS have p-value ≥ alpha.
    """
    L     = losses.values.astype(float)
    names = list(losses.columns)
    T, M  = L.shape

    if block_size is None:
        block_size = max(1, int(T ** (1 / 3)))

    rng   = np.random.default_rng(seed)
    alive = list(range(M))
    p_elim: dict[str, float] = {}

    while len(alive) > 1:
        m   = len(alive)
        sub = L[:, alive]

        # Relative loss: deviation from cross-sectional mean
        d     = sub - sub.mean(axis=1, keepdims=True)   # (T, m)
        d_bar = d.mean(axis=0)                           # (m,)

        # Bootstrap distribution of d_bar under H0
        boot_d_bar = np.empty((n_boot, m))
        for b in range(n_boot):
            idx           = _cbb_indices(T, block_size, rng)
            boot_d_bar[b] = d[idx].mean(axis=0)

        # HAC-consistent SE from bootstrap variance
        se = boot_d_bar.std(axis=0, ddof=0)
        se = np.maximum(se, 1e-12)

        # T_R statistic: largest standardised relative loss
        t_stats = d_bar / se
        T_R     = t_stats.max()

        # Bootstrap null distribution of T_R (centred under H0)
        T_R_boot = ((boot_d_bar - d_bar) / se).max(axis=1)
        p_val    = float((T_R_boot >= T_R).mean())

        if p_val >= alpha:
            break   # cannot reject H0 → all remaining models in MCS

        worst_local  = int(t_stats.argmax())
        worst_global = alive[worst_local]
        p_elim[names[worst_global]] = p_val
        alive.pop(worst_local)

    mcs_set = [names[i] for i in alive]
    return mcs_set, p_elim


# ── pivot helpers ──────────────────────────────────────────────────────────

def _make_pivot(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot (test_date, config) → sq_err.  Drop any row where any model has NaN
    so all models are compared on the same observations.
    """
    return (
        sub.pivot_table(index="test_date", columns="config",
                        values="sq_err", aggfunc="mean")
           .dropna(axis=1, how="all")
           .dropna(axis=0, how="any")
    )


def _normalise_rows(pivot: pd.DataFrame) -> pd.DataFrame:
    """Divide each row by its cross-model arithmetic mean (clips at 1e-12)."""
    row_mean = pivot.mean(axis=1).clip(lower=1e-12)
    return pivot.div(row_mean, axis=0)


def _make_grouped_pivot(
    sub_p: pd.DataFrame,
    group: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stack losses for all horizons in `group` into a single tall DataFrame.

    Rows are sorted by (test_date, horizon) so that the stacked sequence is:
        [date1_h1, date1_h2, ..., date1_hH, date2_h1, ...]
    This ordering ensures a bootstrap block of width H covers exactly one
    period's worth of horizon observations.

    Returns (raw_pivot, stacked_pivot) where raw_pivot uses the first horizon's
    losses (for rank computation) and stacked_pivot is the T*H × M matrix
    ready for MCS.
    """
    pivots = []
    for h in group:
        sub_h = sub_p[sub_p["horizon"] == h]
        pv    = _make_pivot(sub_h)
        pv.index = pd.MultiIndex.from_arrays(
            [pv.index, [h] * len(pv)], names=["test_date", "horizon"]
        )
        pivots.append(pv)

    common_configs = pivots[0].columns
    for pv in pivots[1:]:
        common_configs = common_configs.intersection(pv.columns)
    pivots = [pv[common_configs] for pv in pivots]

    stacked = pd.concat(pivots).sort_index()
    raw     = _make_pivot(sub_p[sub_p["horizon"] == group[0]])[common_configs]
    return raw, stacked


# ── per-horizon MCS ────────────────────────────────────────────────────────

def run_mcs_analysis(
    df_eval:    pd.DataFrame,
    horizons:   list,
    periods:    list[str] | None = None,
    alpha:      float = 0.10,
    block_size: int   = None,
    n_boot:     int   = 1000,
    seed:       int   = 42,
) -> pd.DataFrame:
    """
    Run MCS for every (period, horizon) combination.

    Parameters
    ----------
    df_eval   : raw evaluation DataFrame with columns
                [config, horizon, period, test_date, sq_err].
    horizons  : list of horizon values to analyse.
    periods   : list of period labels to analyse (default: auto-detect + "full").
    alpha, block_size, n_boot, seed : passed to model_confidence_set.

    Returns
    -------
    DataFrame with columns [period, horizon, config, in_mcs, p_elim,
    rank_loss, mean_loss].
    """
    if periods is None:
        detected = sorted(df_eval["period"].unique().tolist())
        periods  = ["full"] + detected

    records = []
    for period in periods:
        sub_p = df_eval if period == "full" else df_eval[df_eval["period"] == period]
        if sub_p.empty:
            continue

        for h in horizons:
            sub = sub_p[sub_p["horizon"] == h]
            if sub.empty:
                continue

            pivot = _make_pivot(sub)
            if pivot.shape[1] < 2 or pivot.shape[0] < 4:
                for cfg in pivot.columns:
                    records.append(dict(period=period, horizon=str(h), config=cfg,
                                        in_mcs=True, p_elim=np.nan, rank_loss=1,
                                        mean_loss=np.nan))
                continue

            mean_loss = pivot.mean(axis=0).sort_values()
            rank_map  = {c: r + 1 for r, c in enumerate(mean_loss.index)}

            losses_for_mcs = _normalise_rows(pivot) if period == "full" else pivot
            mcs_set, p_elim = model_confidence_set(
                losses_for_mcs, alpha=alpha,
                block_size=block_size, n_boot=n_boot, seed=seed,
            )

            for cfg in pivot.columns:
                records.append(dict(
                    period    = period,
                    horizon   = str(h),
                    config    = cfg,
                    in_mcs    = cfg in mcs_set,
                    p_elim    = p_elim.get(cfg, np.nan),
                    rank_loss = rank_map[cfg],
                    mean_loss = float(mean_loss[cfg]),
                ))

    return pd.DataFrame(records)


# ── grouped-horizon MCS ────────────────────────────────────────────────────

def run_mcs_grouped(
    df_eval:        pd.DataFrame,
    horizon_groups: list[list[int]],
    periods:        list[str] | None = None,
    alpha:          float = 0.10,
    n_boot:         int   = 1000,
    seed:           int   = 42,
) -> pd.DataFrame:
    """
    Run MCS for every (period, horizon_group) combination.

    Parameters
    ----------
    df_eval        : raw evaluation DataFrame.
    horizon_groups : e.g. [[1, 5], [22]] or [[1, 5, 22]].
    periods        : period labels (default: auto-detect + "full").
    alpha, n_boot, seed : passed to model_confidence_set.

    Returns
    -------
    DataFrame with columns [period, horizon, config, in_mcs, p_elim,
    rank_loss, mean_loss] where horizon is a label like "h1-5".
    """
    if periods is None:
        detected = sorted(df_eval["period"].unique().tolist())
        periods  = ["full"] + detected

    records = []
    for period in periods:
        sub_p = df_eval if period == "full" else df_eval[df_eval["period"] == period]
        if sub_p.empty:
            continue

        for group in horizon_groups:
            label = f"h{group[0]}-{group[-1]}"
            H     = len(group)

            raw_pivot, stacked = _make_grouped_pivot(sub_p, group)

            if stacked.shape[1] < 2 or stacked.shape[0] < 4:
                for cfg in stacked.columns:
                    records.append(dict(period=period, horizon=label, config=cfg,
                                        in_mcs=True, p_elim=np.nan, rank_loss=1,
                                        mean_loss=np.nan))
                continue

            mean_loss = stacked.mean(axis=0).sort_values()
            rank_map  = {c: r + 1 for r, c in enumerate(mean_loss.index)}

            losses_for_mcs = _normalise_rows(stacked) if period == "full" else stacked

            T          = len(stacked)
            base_block = max(1, int(T ** (1 / 3)))
            block_size = int(np.ceil(base_block / H) * H)

            mcs_set, p_elim = model_confidence_set(
                losses_for_mcs, alpha=alpha,
                block_size=block_size, n_boot=n_boot, seed=seed,
            )

            for cfg in stacked.columns:
                records.append(dict(
                    period    = period,
                    horizon   = label,
                    config    = cfg,
                    in_mcs    = cfg in mcs_set,
                    p_elim    = p_elim.get(cfg, np.nan),
                    rank_loss = rank_map[cfg],
                    mean_loss = float(mean_loss[cfg]),
                ))

    return pd.DataFrame(records)


# ── summary text ──────────────────────────────────────────────────────────

def mcs_summary_text(mcs_df: pd.DataFrame, title: str = "") -> str:
    """Format MCS results as a readable text table."""
    lines = []
    period_order = ["full"] + sorted(
        p for p in mcs_df["period"].unique() if p != "full"
    )
    for period in period_order:
        sub = mcs_df[mcs_df["period"] == period]
        if sub.empty:
            continue
        lines.append(f"\n{'─' * 62}")
        label = f"FULL" if period == "full" else period.upper().replace("_", "-")
        lines.append(f"  MCS — {label}  (α=10%)  {title}")
        lines.append(f"{'─' * 62}")
        for h in sorted(sub["horizon"].unique(), key=str):
            row       = sub[sub["horizon"] == h].sort_values("rank_loss")
            survivors = row[row["in_mcs"]]["config"].tolist()
            elim_rows = row[~row["in_mcs"]][["config", "p_elim"]]
            eliminated = [f"{r.config}(p={r.p_elim:.3f})"
                          for _, r in elim_rows.iterrows()]
            lines.append(
                f"  {h}: IN={survivors}  OUT=[{', '.join(eliminated)}]"
            )
    return "\n".join(lines)
