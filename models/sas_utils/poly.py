"""
poly.py — Polynomial basis classes for the SAS recurrence.

Recurrence:  s_t = P(z_t) ⊛ s_{t-1} + Q(z_t)

LinearPoly   — full (n×n) matrix transition.  O(n²) per step.
               Expressive but memory-intensive for large n.

DiagonalPoly — scalar-per-unit (diagonal) transition.  O(n) per step.
               Allows much larger n (e.g. n=1000) at the same cost.
               s_t[i] = (p0[i] + p1[i]·z) · s_{t-1}[i] + (q0[i] + q1[i]·z)
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _JAX = True
except ImportError:
    _JAX = False


# ── LinearPoly ────────────────────────────────────────────────────────────────

class LinearPoly:
    """
    Linear polynomial basis: full (n×n) matrix transition.

    P(z) = P[0] + P[1]·z + … + P[p]·z^p     shape (p+1, n, n)
    Q(z) = Q[0] + Q[1]·z + … + Q[q]·z^q     shape (q+1, n)

    Stability: sum of spectral norms of P[k] ≤ spectral_norm < 1.
    """

    kind = "linear"

    def __init__(self, p_degree: int = 1, q_degree: int = 1):
        self.p_degree = p_degree
        self.q_degree = q_degree
        self.P = None  # (p_degree+1, n, n)
        self.Q = None  # (q_degree+1, n)

    def initialize(self, n: int, key, spectral_norm: float = 0.9) -> "LinearPoly":
        if _JAX:
            k1, k2 = jax.random.split(key)
            P_raw  = jax.random.normal(k1, (self.p_degree + 1, n, n)) / (n ** 0.5)
            norms  = jax.vmap(lambda M: jnp.linalg.norm(M, ord=2))(P_raw)
            self.P = P_raw * (spectral_norm / jnp.maximum(jnp.sum(norms), 1e-8))
            self.Q = jax.random.normal(k2, (self.q_degree + 1, n)) / (n ** 0.5)
        else:
            seed  = int(key) if isinstance(key, (int, np.integer)) else 42
            rng   = np.random.default_rng(seed)
            P_raw = rng.standard_normal((self.p_degree + 1, n, n)) / (n ** 0.5)
            norms = np.array([np.linalg.norm(P_raw[k], ord=2)
                              for k in range(self.p_degree + 1)])
            self.P = P_raw * (spectral_norm / max(float(norms.sum()), 1e-8))
            self.Q = rng.standard_normal((self.q_degree + 1, n)) / (n ** 0.5)
        return self

    def eval_p(self, z):
        """(n, n) matrix P(z)."""
        out, z_k = self.P[0], z
        for k in range(1, self.p_degree + 1):
            out  = out + self.P[k] * z_k
            z_k  = z_k * z
        return out

    def eval_q(self, z):
        """(n,) vector Q(z)."""
        out, z_k = self.Q[0], z
        for k in range(1, self.q_degree + 1):
            out  = out + self.Q[k] * z_k
            z_k  = z_k * z
        return out


# ── DiagonalPoly ──────────────────────────────────────────────────────────────

class DiagonalPoly:
    """
    Diagonal polynomial basis: scalar-per-unit transition.

    p(z)[i] = p0[i] + p1[i]·z      shape (n,)  — diagonal of the transition
    q(z)[i] = q0[i] + q1[i]·z      shape (n,)

    Recurrence (elementwise):
        s_t = p(z_t) * s_{t-1} + q(z_t)

    Stability: |p0[i]| + |p1[i]| · |z_max| < 1 (approximately), enforced by
    clipping |p0[i]| ≤ spectral_norm and scaling p1 so the modulation is small.
    """

    kind = "diagonal"

    def __init__(self, p_degree: int = 1, q_degree: int = 1):
        # Only linear degree is implemented for the diagonal basis
        self.p_degree = p_degree
        self.q_degree = q_degree
        self.p0 = self.p1 = None   # (n,) each
        self.q0 = self.q1 = None   # (n,) each

    def initialize(self, n: int, key, spectral_norm: float = 0.9) -> "DiagonalPoly":
        """
        Initialise with p0 uniform in [-sn, sn] (diverse timescales) and
        small p1 so input modulation keeps |p(z)| < 1 for |z| ≤ 5.
        """
        if _JAX:
            k1, k2, k3, k4 = jax.random.split(key, 4)
            self.p0 = (jax.random.uniform(k1, (n,)) * 2 - 1) * spectral_norm
            margin  = max(0.005, (1.0 - spectral_norm) * 0.3)
            self.p1 = jax.random.normal(k2, (n,)) * (margin / 5.0)
            self.q0 = jax.random.normal(k3, (n,)) / (n ** 0.5)
            self.q1 = jax.random.normal(k4, (n,)) / (n ** 0.5)
        else:
            seed  = int(key) if isinstance(key, (int, np.integer)) else 42
            rng   = np.random.default_rng(seed)
            self.p0 = (rng.uniform(size=n) * 2 - 1) * spectral_norm
            margin  = max(0.005, (1.0 - spectral_norm) * 0.3)
            self.p1 = rng.standard_normal(n) * (margin / 5.0)
            self.q0 = rng.standard_normal(n) / (n ** 0.5)
            self.q1 = rng.standard_normal(n) / (n ** 0.5)
        return self
