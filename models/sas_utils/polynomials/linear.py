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
    Full-matrix polynomial basis with exact powers of z.
    P(z) = P[0] + P[1]·z + P[2]·z² + ...
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
        ka, kb = jax.random.split(key)
        
        # P: (p_degree + 1, n, n)
        P_raw = jax.random.normal(ka, (self.p_degree + 1, n, n)) / n ** 0.5
        
        # ENFORCE NILPOTENCY (Strictly upper triangular)
        P_nilpotent = jnp.triu(P_raw, k=1)
        P = self._scale_p(P_nilpotent, self.spectral_norm)
        
        # Q: (q_degree + 1, n)
        Q = jax.random.normal(kb, (self.q_degree + 1, n)) / n ** 0.5
        
        obj = LinearPoly(self.p_degree, self.q_degree, self.spectral_norm)
        obj.P, obj.Q = P, Q
        return obj

    # ── feature helpers ────────────────────────────────────────────────────

    @staticmethod
    def _feats(z, degree: int):
        """z: (1,) → [1, z, z², ..., z^degree]"""
        # Calcule les puissances de 0 à degree
        return jnp.power(z[0], jnp.arange(degree + 1))

    @staticmethod
    def _batch_feats(z_batch, degree: int):
        """z_batch: (B, 1) → (B, degree + 1)"""
        return jnp.power(z_batch[:, 0:1], jnp.arange(degree + 1))

    # ── forward evaluators ─────────────────────────────────────────────────

    def eval_p(self, z):
        feats = self._feats(z, self.p_degree)
        return jnp.einsum("f,fij->ij", feats, self.P)

    def eval_q(self, z):
        feats = self._feats(z, self.q_degree)
        return jnp.einsum("f,fi->i", feats, self.Q)

    def batch_eval_p(self, z_batch):
        feats = self._batch_feats(z_batch, self.p_degree)
        return jnp.einsum("bf,fij->bij", feats, self.P)

    def batch_eval_q(self, z_batch):
        feats = self._batch_feats(z_batch, self.q_degree)
        return jnp.einsum("bf,fi->bi", feats, self.Q)

    # ── algebraic primitives ────────────────────────────────────────────────

    def apply(self, A, s):
        return A @ s

    def combine(self, i, j):
        A_i, b_i = i
        A_j, b_j = j
        return A_j @ A_i, (A_j @ b_i[..., None])[..., 0] + b_j
    

@jax.tree_util.register_pytree_node_class
class BlockLinearPoly(BasePoly):
    """
    Block-Diagonal Polynomial basis.
    Transitions are dense within blocks of size (B x B), but isolated between blocks.
    Guarantees O(K * B^3) scan complexity instead of O(n^3).
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

    # ── pytree protocol ────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.P, self.Q), (self.n_blocks, self.block_size, self.p_degree, self.q_degree, self.spectral_norm)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P, obj.Q = children
        return obj

    # ── factory ────────────────────────────────────────────────────────────

    def initialize(self, n: int, key) -> "BlockLinearPoly":
        if n != self.n:
            raise ValueError(f"Requested n={n} but BlockPoly is configured for {self.n_blocks}x{self.block_size}={self.n}")

        K, B = self.n_blocks, self.block_size
        ka, kb = jax.random.split(key)
        
        # P_raw: (p_deg+1, K, B, B)
        P_raw = jax.random.normal(ka, (self.p_degree + 1, K, B, B)) / B ** 0.5
        
        # 1. ENFORCE NILPOTENCY PER BLOCK
        P_nilpotent = jnp.triu(P_raw, k=1)
        
        # 2. SCALE PER BLOCK (Sum of spectral norms over p_degrees for each block <= sn)
        # Swap axes to iterate over blocks: (K, p_deg+1, B, B)
        P_k = jnp.swapaxes(P_nilpotent, 0, 1)
        
        def scale_block_seq(seq):
            norms = jax.vmap(lambda M: jnp.linalg.norm(M, ord=2))(seq)
            return seq * (self.spectral_norm / jnp.maximum(jnp.sum(norms), 1e-8))
            
        P_scaled = jax.vmap(scale_block_seq)(P_k) 
        self.P = jnp.swapaxes(P_scaled, 0, 1)  # Back to (p_deg+1, K, B, B)
        
        # Q: (q_deg+1, n)
        self.Q = jax.random.normal(kb, (self.q_degree + 1, n)) / B ** 0.5
        return self

    # ── feature helpers ────────────────────────────────────────────────────

    @staticmethod
    def _feats(z, degree: int):
        return jnp.power(z[0], jnp.arange(degree + 1))

    @staticmethod
    def _batch_feats(z_batch, degree: int):
        return jnp.power(z_batch[:, 0:1], jnp.arange(degree + 1))

    # ── forward evaluators ─────────────────────────────────────────────────

    def eval_p(self, z):
        feats = self._feats(z, self.p_degree)
        return jnp.einsum("f,fkij->kij", feats, self.P) # Returns (K, B, B)

    def eval_q(self, z):
        feats = self._feats(z, self.q_degree)
        return jnp.einsum("f,fi->i", feats, self.Q)

    def batch_eval_p(self, z_batch):
        feats = self._batch_feats(z_batch, self.p_degree)
        return jnp.einsum("bf,fkij->bkij", feats, self.P) # Returns (Batch, K, B, B)

    def batch_eval_q(self, z_batch):
        feats = self._batch_feats(z_batch, self.q_degree)
        return jnp.einsum("bf,fi->bi", feats, self.Q)

    # ── algebraic primitives (The Magic Happens Here) ───────────────────────

    def apply(self, A, s):
        """A: (K, B, B), s: (n,)"""
        s_blocked = s.reshape(self.n_blocks, self.block_size, 1)
        return jnp.matmul(A, s_blocked).reshape(self.n)

    def combine(self, i, j):
        """
        Matrix multiplication of block-diagonal matrices.
        JAX 'matmul' automatically broadcasts over batch and block dimensions!
        """
        A_i, b_i = i  # A_i: (..., K, B, B), b_i: (..., n)
        A_j, b_j = j
        
        K, B = self.n_blocks, self.block_size
        
        # 1. Multiply the blocks together
        A_new = jnp.matmul(A_j, A_i)
        
        # 2. Reshape b_i to treat it as a vector per block: (..., K, B, 1)
        b_i_reshaped = b_i.reshape(b_i.shape[:-1] + (K, B, 1))
        
        # 3. Apply A_j to b_i block-wise, then reshape back to (..., n)
        b_term = jnp.matmul(A_j, b_i_reshaped).reshape(b_j.shape)
        
        return A_new, b_term + b_j