"""
sas_utils — utilities for the SAS (Spectral Associative Scan) reservoir.

The polynomial bases live in the `polynomials` sub-package.  Each class is a
registered JAX pytree so it can be passed directly to @jax.jit functions.
"""

from .polynomials import BasePoly, LinearPoly, DiagonalPoly, TrigoPoly, BlockLinearPoly, BlockTrigoPoly

__all__ = ["BasePoly", "LinearPoly", "BlockLinearPoly", "DiagonalPoly", "TrigoPoly", "BlockTrigoPoly"]
