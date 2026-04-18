"""
diagonal.py — Scalar-per-unit (diagonal) polynomial basis.

The transition "matrix" is purely diagonal — represented as a vector of n
scalars, one per reservoir unit.  This drops the O(n²) memory to O(n) and
allows n up to thousands.

Recurrence (element-wise):
    s_t[i] = p(z_t)[i] · s_{t-1}[i] + q(z_t)[i]

where p and q are degree-p_degree / q_degree polynomials evaluated via
Horner's method:
    p(z)[i] = P[0][i] + P[1][i]·z + P[2][i]·z² + …   clipped to (−1, 1)
    q(z)[i] = Q[0][i] + Q[1][i]·z + Q[2][i]·z² + …

P: (p_degree+1, n)   Q: (q_degree+1, n)

Stability: the clip ensures |p(z)| ≤ 0.9999 element-wise for any z.

Higher degrees (p_degree > 1) make the input-modulation of the timescales
richer at negligible extra cost.
"""

import jax
import jax.numpy as jnp
from .base import BasePoly


@jax.tree_util.register_pytree_node_class
class DiagonalPoly(BasePoly):
    """
    Diagonal polynomial basis.

    Parameters
    ----------
    p_degree     : degree of z in the diagonal transition polynomial.
    q_degree     : degree of z in the input-drive polynomial.
    spectral_norm: base eigenvalue range — p0[i] ∈ (−sn, sn), encoding
                   timescales from fast-decaying to near-unit-root.
    """

    def __init__(self, p_degree: int = 1, q_degree: int = 1,
                 spectral_norm: float = 0.9):
        super().__init__(p_degree, q_degree)
        self.spectral_norm = spectral_norm

    # ── pytree protocol ────────────────────────────────────────────────────

    def tree_flatten(self):
        # leaves are the learned arrays; aux carries construction hyperparams
        return (self.P, self.Q), (self.p_degree, self.q_degree, self.spectral_norm)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P, obj.Q = children
        return obj

    # ── factory ────────────────────────────────────────────────────────────

    def initialize(self, n: int, key) -> "DiagonalPoly":
        """
        Initialise P and Q.

        P[0] (base eigenvalues): uniform in (−sn, sn) — diverse timescales.
        P[k≥1] (input modulation): small normal so clip rarely activates at
                 typical z values (|z| ≤ 5 after z-scoring).

        Q[k]  : scaled by 1 / (√n · sqrt((2k−1)!!)) where (2k−1)!! is the
                 double factorial.  This approximates 1/(√n · std(z^k)) and
                 shrinks higher-degree terms, keeping Q(z_t) injection variance
                 roughly uniform across degrees for typical inputs.
                 (Exact for odd k; slight overestimate for even k → conservative.)

                 k=0: scale = 1/√n    (std≈1,    (−1)!!=1)
                 k=1: scale = 1/√n    (std=1,    ( 1)!!=1)
                 k=2: scale ≈ 0.58/√n (std≈√2,  ( 3)!!=3)
                 k=3: scale ≈ 0.26/√n (std=√15, ( 5)!!=15)
                 k=4: scale ≈ 0.10/√n (std≈√96, ( 7)!!=105)
        """
        sn  = self.spectral_norm
        # margin: how much headroom above sn before clip activates
        margin = max(0.005, (1.0 - sn) * 0.3)

        # Keys: 1 (P₀) + p_degree (P₁…Pp) + q_degree + 1 (one slot per Q row,
        # but we use a SINGLE call for the full Q matrix — the extra slots are
        # intentionally unused, preserving the same split as the original code
        # so keys[ki_Q] is identical across all q_degree values for backward
        # compatibility.
        n_keys = self.p_degree + self.q_degree + 2
        keys   = jax.random.split(key, n_keys)
        ki     = 0

        # P: (p_degree+1, n)
        # Degree-0 term: spread eigenvalues across the full allowed range
        p0 = (jax.random.uniform(keys[ki], (n,)) * 2 - 1) * sn
        ki += 1
        p_rows = [p0]
        for k in range(1, self.p_degree + 1):
            # Higher-degree modulation gets exponentially smaller
            scale = margin / (5.0 ** k)
            p_rows.append(jax.random.normal(keys[ki], (n,)) * scale)
            ki += 1
        P = jnp.stack(p_rows, axis=0)   # (p_degree+1, n)

        # Q: (q_degree+1, n)
        # Single key → same random draws as before for any q_degree.
        # Base scale 1/√n matches the original q=1 code exactly (bit-for-bit).
        # For k≥2, an additional shrinkage factor 1/sqrt((2k-1)!!) is applied
        # to counteract the growing variance of z^k for z~N(0,1).
        # (2k-1)!! = 1·3·5·…·(2k-1) is the double factorial; returns 1 for k≤1.
        # This keeps Q(z_t) injection variance roughly equal across degrees,
        # preventing state explosion for large inputs.
        #   k=0: no extra shrinkage  — (−1)!!=1,  same as original
        #   k=1: no extra shrinkage  — ( 1)!!=1,  same as original
        #   k=2: ×1/√3 ≈ 0.577      — ( 3)!!=3   ← shrunk
        #   k=3: ×1/√15 ≈ 0.258     — ( 5)!!=15  ← shrunk more
        def _double_factorial(m: int) -> float:
            """(2k−1)!! = 1·3·5·…·(2k−1); returns 1 for m ≤ 0."""
            result = 1.0
            for i in range(1, m + 1, 2):
                result *= i
            return result

        # Use the original scalar division so that k=0,1 entries are bit-for-bit
        # identical to the pre-shrinkage code.  Higher-degree rows are scaled down.
        Q_raw = jax.random.normal(keys[ki], (self.q_degree + 1, n))
        Q     = Q_raw / n ** 0.5          # (q_degree+1, n) — original scaling
        if self.q_degree >= 2:
            # correction[k] = 1/sqrt((2k-1)!!)  — equals 1 for k=0,1
            correction = jnp.array([
                1.0 / _double_factorial(2 * k - 1) ** 0.5
                for k in range(self.q_degree + 1)
            ])                            # (q_degree+1,) — first two entries = 1.0
            Q = Q * correction[:, None]   # rows 0,1 unchanged; rows ≥2 shrunk

        obj   = DiagonalPoly(self.p_degree, self.q_degree, self.spectral_norm)
        obj.P, obj.Q = P, Q
        return obj

    # ── Horner evaluators ─────────────────────────────────────────────────

    def eval_p(self, z):
        """
        z: (d=1,) → a: (n,)  diagonal elements, clipped to (−0.9999, 0.9999).

        Uses Horner's method:  val = P[p]; for k=p-1..0: val = val*z + P[k]
        """
        z_sc = z[0]                       # scalar
        val  = self.P[-1]                 # (n,)
        for k in range(self.p_degree - 1, -1, -1):
            val = val * z_sc + self.P[k]
        return jnp.clip(val, -0.9999, 0.9999)

    def eval_q(self, z):
        """z: (d=1,) → b: (n,)  via Horner."""
        z_sc = z[0]
        val  = self.Q[-1]
        for k in range(self.q_degree - 1, -1, -1):
            val = val * z_sc + self.Q[k]
        return val

    def batch_eval_p(self, z_batch):
        """
        z_batch: (B, 1) → (B, n)  — batched Horner over the scalar dimension.

        The Python for-loop is unrolled at JAX trace time → one fused kernel.
        """
        z_sc = z_batch[:, 0:1]                             # (B, 1)
        B    = z_sc.shape[0]
        val  = jnp.broadcast_to(self.P[-1], (B, self.P.shape[1]))  # (B, n)
        for k in range(self.p_degree - 1, -1, -1):
            val = val * z_sc + self.P[k]                   # (B,n) * (B,1) + (n,)
        return jnp.clip(val, -0.9999, 0.9999)

    def batch_eval_q(self, z_batch):
        """z_batch: (B, 1) → (B, n)."""
        z_sc = z_batch[:, 0:1]
        B    = z_sc.shape[0]
        val  = jnp.broadcast_to(self.Q[-1], (B, self.Q.shape[1]))
        for k in range(self.q_degree - 1, -1, -1):
            val = val * z_sc + self.Q[k]
        return val

    # ── algebraic primitives ────────────────────────────────────────────────

    def apply(self, a, s):
        """(n,) a, (n,) s → (n,)  element-wise multiplication."""
        return a * s

    def combine(self, i, j):
        """
        Monoid for the associative scan (element-wise version).
        i = (a_i: (n,), b_i: (n,))
        j = (a_j: (n,), b_j: (n,))
        → (a_j * a_i,  a_j * b_i + b_j)
        """
        a_i, b_i = i
        a_j, b_j = j
        return a_j * a_i, a_j * b_i + b_j
