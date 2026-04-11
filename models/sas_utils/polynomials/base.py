"""
base.py — Abstract base class for SAS polynomial bases.

All concrete subclasses are registered JAX pytree nodes.  This means an
*instance* (including its learned arrays P and Q) can be passed directly to
any @jax.jit compiled function — JAX traces through the leaves (arrays) while
the Python class is resolved at trace time, giving one compiled kernel per
concrete polynomial type.

Algebraic contract
------------------
The SAS recurrence is   s_t = P(z_t) ⊛ s_{t-1} + Q(z_t)

where ⊛ depends on the polynomial type:
  LinearPoly   : full matrix-vector product    (P: (n,n), ⊛ = @)
  DiagonalPoly : element-wise multiplication   (P: (n,),  ⊛ = *)
  TrigoPoly    : full matrix, trig features    (P: (n,n), ⊛ = @)

Two methods drive the parallel associative scan:

    combine(i, j) — monoid composition of two (A_rep, b) pairs
                    representing "apply i then j"
    apply(A_rep, s) — apply A_rep to state vector s  → (n,)

These are called inside JAX-traced code and must use only JAX operations.
"""

import abc
import jax
import jax.numpy as jnp


class BasePoly(abc.ABC):
    """Abstract base for SAS polynomial bases (registered JAX pytree nodes)."""

    def __init__(self, p_degree: int = 1, q_degree: int = 1):
        self.p_degree = p_degree
        self.q_degree = q_degree
        self.P = None   # initialised by .initialize()
        self.Q = None

    def is_initialized(self) -> bool:
        return self.P is not None

    # ── factory ────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def initialize(self, n: int, key) -> "BasePoly":
        """Return a NEW initialized instance with reservoir dimension n."""

    # ── forward evaluators (JAX-traceable) ────────────────────────────────

    @abc.abstractmethod
    def eval_p(self, z):
        """Single input z: (d,) → A_rep   — (n,n) for full, (n,) for diagonal."""

    @abc.abstractmethod
    def eval_q(self, z):
        """Single input z: (d,) → b: (n,)."""

    def batch_eval_p(self, z_batch):
        """z_batch: (B, d) → (B, *A_rep_shape).  Override for efficiency."""
        return jax.vmap(self.eval_p)(z_batch)

    def batch_eval_q(self, z_batch):
        """z_batch: (B, d) → (B, n).  Override for efficiency."""
        return jax.vmap(self.eval_q)(z_batch)

    # ── algebraic primitives for the two-level associative scan ───────────

    @abc.abstractmethod
    def apply(self, A_rep, s):
        """
        Apply A_rep to state vector s → (n,).

        LinearPoly  : A_rep @ s       (matrix-vector)
        DiagonalPoly: A_rep * s       (element-wise)
        """

    @abc.abstractmethod
    def combine(self, i, j):
        """
        Monoid for jax.lax.associative_scan.

        Each element is (A_rep, b: (n,)).  Returns the composition
        representing "apply i then j":
            A_new = A_j ∘ A_i
            b_new = apply(A_j, b_i) + b_j
        """

    # ── JAX pytree protocol ────────────────────────────────────────────────

    @abc.abstractmethod
    def tree_flatten(self): ...

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux, children): ...

    # ── shared utilities ───────────────────────────────────────────────────

    @staticmethod
    def _scale_p(P, max_norm: float):
        """
        Scale P so the sum of per-term spectral norms equals max_norm.
        Guarantees the Echo-State Property (ESP) when max_norm < 1.
        """
        norms = jax.vmap(lambda M: jnp.linalg.norm(M, ord=2))(P)
        return P * (max_norm / jnp.maximum(jnp.sum(norms), 1e-8))
