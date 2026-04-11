"""
linear.py — Full (n×n) matrix polynomial basis.

P(z) = P[0] + P[1]·z[0] + P[2]·z[1] + ...   shape (d+1, n, n)
Q(z) = Q[0] + Q[1]·z[0] + Q[2]·z[1] + ...   shape (d+1, n)

For univariate SAS (d=1):  P: (2, n, n),  Q: (2, n)

Stability: sum of spectral norms of the P terms ≤ spectral_norm < 1.

Memory: O(n²) per step — keep n ≤ 200 for d=1.
"""

import jax
import jax.numpy as jnp
from .base import BasePoly


@jax.tree_util.register_pytree_node_class
class LinearPoly(BasePoly):
    """
    Full-matrix polynomial basis (d+1 terms, each n×n).

    Parameters
    ----------
    p_degree     : polynomial degree for P (currently only degree=1 used,
                   i.e. constant + linear; higher degrees use more P slices).
    q_degree     : polynomial degree for Q.
    spectral_norm: upper bound on sum of spectral norms across P terms.
    """

    def __init__(self, p_degree: int = 1, q_degree: int = 1,
                 spectral_norm: float = 0.9):
        super().__init__(p_degree, q_degree)
        self.spectral_norm = spectral_norm

    # ── pytree protocol ────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.P, self.Q), (self.p_degree, self.q_degree, self.spectral_norm)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P, obj.Q = children
        return obj

    # ── factory ────────────────────────────────────────────────────────────

    def initialize(self, n: int, key) -> "LinearPoly":
        """
        Random initialisation for a univariate input (d=1).

        P[k] ~ N(0, 1/n) scaled so sum(spectral_norms) = spectral_norm.
        Q[k] ~ N(0, 1/n).
        """
        d = 1   # univariate: z is a scalar embedded as (1,)
        ka, kb = jax.random.split(key)
        P_raw = jax.random.normal(ka, (d + 1, n, n)) / n ** 0.5
        P     = self._scale_p(P_raw, self.spectral_norm)
        Q     = jax.random.normal(kb, (d + 1, n)) / n ** 0.5
        obj   = LinearPoly(self.p_degree, self.q_degree, self.spectral_norm)
        obj.P, obj.Q = P, Q
        return obj

    # ── forward evaluators ─────────────────────────────────────────────────

    def eval_p(self, z):
        """z: (d,) → A: (n, n)   linear combination of P slices."""
        feats = jnp.concatenate([jnp.ones(1), z])          # (d+1,)
        return jnp.einsum("f,fij->ij", feats, self.P)      # (n, n)

    def eval_q(self, z):
        """z: (d,) → b: (n,)."""
        feats = jnp.concatenate([jnp.ones(1), z])
        return jnp.einsum("f,fi->i", feats, self.Q)        # (n,)

    def batch_eval_p(self, z_batch):
        """z_batch: (B, d) → (B, n, n) — batched efficient einsum."""
        # P[0] broadcast + z-weighted sum of P[1:]
        return self.P[0] + jnp.einsum("bd,dij->bij", z_batch, self.P[1:])

    def batch_eval_q(self, z_batch):
        """z_batch: (B, d) → (B, n)."""
        return self.Q[0] + jnp.einsum("bd,di->bi", z_batch, self.Q[1:])

    # ── algebraic primitives ────────────────────────────────────────────────

    def apply(self, A, s):
        """(n,n) A, (n,) s → (n,)  via matrix-vector product."""
        return A @ s

    def combine(self, i, j):
        """
        Monoid for the associative scan.
        i = (A_i: (n,n), b_i: (n,))
        j = (A_j: (n,n), b_j: (n,))
        → (A_j @ A_i, A_j @ b_i + b_j)

        Note: jax.lax.associative_scan internally vmaps combine, so A_i/A_j
        may arrive as (batch, n, n) and b_i/b_j as (batch, n).  The
        `b_i[..., None]` trick makes the matvec unambiguous for any batch dim.
        """
        A_i, b_i = i
        A_j, b_j = j
        return A_j @ A_i, (A_j @ b_i[..., None])[..., 0] + b_j
