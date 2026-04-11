"""Abstract base for all univariate forecasters."""
from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """
    Streaming-friendly univariate forecaster protocol.

    Contract
    --------
    fit(history, horizons)  : calibrate on 1-D numpy array (T,).
    update(x)               : ingest one new scalar; update state without refit.
    predict(h)              : scalar point forecast h steps ahead.

    The evaluation loop only ever calls these three methods, so the model
    can internally use any memory structure (deque, scan state, matrix).
    This means the loop is decoupled from how a model stores the past —
    whether it is a deque, an associative scan state, or a full history array.
    """

    @abstractmethod
    def fit(self, history: np.ndarray, horizons: list[int]) -> "BaseForecaster":
        """Fit on history (T,) and prepare to forecast all requested horizons."""
        ...

    @abstractmethod
    def update(self, x: float) -> "BaseForecaster":
        """Incorporate one new observation without refitting."""
        ...

    @abstractmethod
    def predict(self, h: int) -> float:
        """Point forecast h steps ahead (in the same space as the input)."""
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__
