"""
experiments/utils.py — thin re-export shim.

All logic lives in utils/ (ROOT-level):
  utils.oos      → run_oos
  utils.dm       → print_beats_benchmark
  utils.display  → print_precision_table, print_mcs_frequency,
                   print_per_horizon_scoring
"""
from utils import (  # noqa: F401
    run_oos,
    print_precision_table,
    print_mcs_frequency,
    print_beats_benchmark,
    print_per_horizon_scoring,
)
