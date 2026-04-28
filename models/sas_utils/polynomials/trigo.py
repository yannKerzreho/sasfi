"""
trigo.py — Trigonometric polynomial basis (full matrix).

Feature map for degree p, input z ∈ ℝ^d:
    φ_p(z) = [1, sin(z), cos(z), sin(2z), cos(2z), …, sin(pz), cos(pz)]
    n_feats = 1 + 2 · p_degree · d

P: (n_feats_p, n, n)   Q: (n_feats_q, n)

The trig features wrap the input signal in a richer nonlinear space before
forming the recurrence matrix, giving the reservoir more spectral diversity
than a plain polynomial basis at the same matrix size.

Memory: O(n²) per step — same as LinearPoly.  Keep n ≤ 200.
"""

import jax
import jax.numpy as jnp
from .base import BasePoly


@jax.tree_util.register_pytree_node_class
class TrigoPoly(BasePoly):
    """
    Trigonometric polynomial basis (full n×n matrices).

    Parameters
    ----------
    p_degree     : number of sine/cosine frequency pairs in P features.
    q_degree     : number of sine/cosine frequency pairs in Q features.
    spectral_norm: upper bound on sum of spectral norms of P terms.
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

    # ── feature helpers ────────────────────────────────────────────────────

    @staticmethod
    def _feats(z, degree: int):
        """z: (d,) → φ: (1 + 2·degree·d,)."""
        parts = [jnp.ones(1)]
        for f in range(1, degree + 1):
            parts += [jnp.sin(f * z), jnp.cos(f * z)]
        return jnp.concatenate(parts)

    @staticmethod
    def _batch_feats(z_batch, degree: int):
        """z_batch: (B, d) → (B, 1 + 2·degree·d)."""
        parts = [jnp.ones((z_batch.shape[0], 1))]
        for f in range(1, degree + 1):
            parts += [jnp.sin(f * z_batch), jnp.cos(f * z_batch)]
        return jnp.concatenate(parts, axis=-1)

    # ── factory ────────────────────────────────────────────────────────────

    def initialize(self, n: int, key) -> "TrigoPoly":
        """
        Random initialisation for univariate input (d=1).
        Uses a smart spectral budget to prevent vanishing memory.
        """
        d   = 1
        n_p = 1 + 2 * self.p_degree * d
        n_q = 1 + 2 * self.q_degree * d
        ka, kb = jax.random.split(key)
        
        P_raw = jax.random.normal(ka, (n_p, n, n)) / n ** 0.5
        
        # --- Smart Spectral Budgeting ---
        sn = self.spectral_norm
        base_sn = sn * 0.8  # Give 80% of stability budget to base dynamics
        mod_sn  = sn * 0.2  # 20% budget for trig modulations
        
        # 1. Scale the base matrix P[0]
        p0_norm = jnp.linalg.norm(P_raw[0], ord=2)
        P0 = P_raw[0] * (base_sn / jnp.maximum(p0_norm, 1e-8))
        
        # 2. Scale the modulation matrices P[1:]
        if n_p > 1:
            P_mod_raw = P_raw[1:]
            # Sum the norms of the modulation matrices and bound them to mod_sn
            mod_norms = jax.vmap(lambda M: jnp.linalg.norm(M, ord=2))(P_mod_raw)
            P_mod = P_mod_raw * (mod_sn / jnp.maximum(jnp.sum(mod_norms), 1e-8))
            P = jnp.concatenate([P0[None], P_mod], axis=0)
        else:
            P = P0[None]
            
        # Optional: Shrink Q terms for higher frequencies (similar to DiagonalPoly)
        Q = jax.random.normal(kb, (n_q, n)) / n ** 0.5
        # Since sin/cos are bounded to [-1, 1], their variance doesn't explode 
        # like z^k, so Q scaling is actually fine as-is without double-factorials.

        obj = TrigoPoly(self.p_degree, self.q_degree, self.spectral_norm)
        obj.P, obj.Q = P, Q
        return obj

    # ── forward evaluators ─────────────────────────────────────────────────

    def eval_p(self, z):
        """z: (d,) → A: (n, n)."""
        φ = self._feats(z, self.p_degree)
        return jnp.einsum("f,fij->ij", φ, self.P)

    def eval_q(self, z):
        """z: (d,) → b: (n,)."""
        φ = self._feats(z, self.q_degree)
        return jnp.einsum("f,fi->i", φ, self.Q)

    def batch_eval_p(self, z_batch):
        """z_batch: (B, d) → (B, n, n)."""
        Φ = self._batch_feats(z_batch, self.p_degree)   # (B, n_p)
        return jnp.einsum("bf,fij->bij", Φ, self.P)

    def batch_eval_q(self, z_batch):
        """z_batch: (B, d) → (B, n)."""
        Φ = self._batch_feats(z_batch, self.q_degree)
        return jnp.einsum("bf,fi->bi", Φ, self.Q)

    # ── algebraic primitives ────────────────────────────────────────────────

    def apply(self, A, s):
        """(n,n) A, (n,) s → (n,)  matrix-vector product."""
        return A @ s

    def combine(self, i, j):
        """
        i = (A_i: (n,n), b_i: (n,))
        j = (A_j: (n,n), b_j: (n,))
        → (A_j @ A_i, A_j @ b_i + b_j)

        The `b_i[..., None]` trick handles the internally-batched calls that
        jax.lax.associative_scan makes during its parallel tree reduction.
        """
        A_i, b_i = i
        A_j, b_j = j
        return A_j @ A_i, (A_j @ b_i[..., None])[..., 0] + b_j


@jax.tree_util.register_pytree_node_class
class BlockTrigoPoly(BasePoly):
    """
    Block-Diagonal Trigonometric basis.
    Preserves bounded feature space and block-wise isolation.
    """

    def __init__(self, n_blocks: int, block_size: int, 
                 p_degree: int = 1, q_degree: int = 1,
                 spectral_norm: float = 0.9):
        super().__init__(p_degree, q_degree)
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.spectral_norm = spectral_norm

    @property
    def n(self):
        return self.n_blocks * self.block_size

    def tree_flatten(self):
        return (self.P, self.Q), (self.n_blocks, self.block_size, self.p_degree, self.q_degree, self.spectral_norm)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P, obj.Q = children
        return obj

    @staticmethod
    def _feats(z, degree: int):
        parts = [jnp.ones(1)]
        for f in range(1, degree + 1):
            parts += [jnp.sin(f * z), jnp.cos(f * z)]
        return jnp.concatenate(parts)

    @staticmethod
    def _batch_feats(z_batch, degree: int):
        parts = [jnp.ones((z_batch.shape[0], 1))]
        for f in range(1, degree + 1):
            parts += [jnp.sin(f * z_batch), jnp.cos(f * z_batch)]
        return jnp.concatenate(parts, axis=-1)

    def initialize(self, n: int, key) -> "BlockTrigoPoly":
        K, B = self.n_blocks, self.block_size
        d = 1
        n_p = 1 + 2 * self.p_degree * d
        n_q = 1 + 2 * self.q_degree * d
        
        ka, kb = jax.random.split(key)
        P_raw = jax.random.normal(ka, (n_p, K, B, B)) / B ** 0.5
        
        sn = self.spectral_norm
        base_sn = sn * 0.8
        mod_sn  = sn * 0.2
        
        P_k = jnp.swapaxes(P_raw, 0, 1) # (K, n_p, B, B)
        
        def scale_block_trigo(seq):
            # seq: (n_p, B, B)
            p0_norm = jnp.linalg.norm(seq[0], ord=2)
            P0 = seq[0] * (base_sn / jnp.maximum(p0_norm, 1e-8))
            if n_p > 1:
                mod_norms = jax.vmap(lambda M: jnp.linalg.norm(M, ord=2))(seq[1:])
                P_mod = seq[1:] * (mod_sn / jnp.maximum(jnp.sum(mod_norms), 1e-8))
                return jnp.concatenate([P0[None], P_mod], axis=0)
            return P0[None]
            
        P_scaled = jax.vmap(scale_block_trigo)(P_k)
        self.P = jnp.swapaxes(P_scaled, 0, 1) # Back to (n_p, K, B, B)
        self.Q = jax.random.normal(kb, (n_q, n)) / B ** 0.5
        return self

    def eval_p(self, z):
        φ = self._feats(z, self.p_degree)
        return jnp.einsum("f,fkij->kij", φ, self.P)

    def eval_q(self, z):
        φ = self._feats(z, self.q_degree)
        return jnp.einsum("f,fi->i", φ, self.Q)

    def batch_eval_p(self, z_batch):
        Φ = self._batch_feats(z_batch, self.p_degree)
        return jnp.einsum("bf,fkij->bkij", Φ, self.P)

    def batch_eval_q(self, z_batch):
        Φ = self._batch_feats(z_batch, self.q_degree)
        return jnp.einsum("bf,fi->bi", Φ, self.Q)

    def apply(self, A, s):
        s_blocked = s.reshape(self.n_blocks, self.block_size, 1)
        return jnp.matmul(A, s_blocked).reshape(self.n)

    def combine(self, i, j):
        A_i, b_i = i
        A_j, b_j = j
        K, B = self.n_blocks, self.block_size
        
        A_new = jnp.matmul(A_j, A_i)
        b_i_resh = b_i.reshape(b_i.shape[:-1] + (K, B, 1))
        b_term = jnp.matmul(A_j, b_i_resh).reshape(b_j.shape)
        
        return A_new, b_term + b_j