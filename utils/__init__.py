"""
utils/ — shared evaluation utilities for the RV forecasting benchmark.

Files
-----
  oos.py     : run_oos — canonical rolling OOS loop.
  dm.py      : DM test helpers — print_beats_benchmark.
  display.py : display tables — print_precision_table, print_mcs_frequency,
               print_per_horizon_scoring.
  metrics.py : scalar loss functions — mse, mae, qlike (point estimates).
  mcs.py     : full MCS analysis with grouped horizons (advanced use).
  mcs_utils.py : ModelConfidenceSet class (backend for display.py).
"""

from utils.oos import run_oos
from utils.dm  import print_beats_benchmark
from utils.display import (
    print_precision_table,
    print_mcs_frequency,
    print_per_horizon_scoring,
)

__all__ = [
    "run_oos",
    "print_precision_table",
    "print_mcs_frequency",
    "print_beats_benchmark",
    "print_per_horizon_scoring",
]
