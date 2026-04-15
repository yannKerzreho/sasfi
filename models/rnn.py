"""
rnn.py — Recurrent neural network direct multi-step forecasters (PyTorch).

Three cell types, selectable via the `cell` parameter:

'lstm'  : Long Short-Term Memory — gated, handles long-range dependencies well.

'gru'   : Gated Recurrent Unit — lighter than LSTM, similar performance.

'lru'   : Linear Recurrent Unit (Orvieto et al. 2023, "Resurrecting Recurrent
          Neural Networks for Long Sequences").
          Real-valued diagonal linear recurrence with guaranteed stability:
              h_t = diag(σ(ν)) · h_{t-1} + B · x_t
          where σ(ν) ∈ (0, 1) ensures |eigenvalues| < 1 unconditionally.
          No gates — purely linear memory, parallel-scan-friendly.
          Output projected through a dense layer to get the LSTM-equivalent
          hidden representation.

All three expose the same (output, hidden) interface so the shared _RNNNet
head code works without modification.

Requires: pip install torch
"""

from collections import deque
import numpy as np
from .base import BaseForecaster

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False


# ── LRU cell ─────────────────────────────────────────────────────────────────

class _LRUCell(nn.Module):
    """
    Single-layer real-valued Linear Recurrent Unit.

    Recurrence : h_t = a ⊙ h_{t-1} + B · x_t
                 a   = sigmoid(ν)  ∈ (0, 1)  ← always stable
    Output     : y_t = C · h_t    ∈ ℝ^{hidden_size}

    The output y_t is a linear projection of the state and acts as the
    drop-in replacement for the LSTM/GRU output at each timestep.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Decay parameter: a = sigmoid(nu) — initialised near 0.5
        self.nu = nn.Parameter(torch.zeros(hidden_size))
        self.B  = nn.Linear(input_size, hidden_size, bias=True)
        self.C  = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, x: "torch.Tensor", h0: "torch.Tensor | None" = None
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        # x  : (B, L, input_size)
        # h0 : (1, B, hidden_size) to match LSTM/GRU convention
        B, L, _  = x.shape
        a        = torch.sigmoid(self.nu)   # (hidden,) — stable decay
        h        = (h0[0] if h0 is not None
                    else torch.zeros(B, self.hidden_size, device=x.device))
        outputs  = []
        for t in range(L):
            h = a * h + self.B(x[:, t, :])     # (B, hidden)
            outputs.append(self.C(h).unsqueeze(1))
        out      = torch.cat(outputs, dim=1)    # (B, L, hidden)
        h_last   = h.unsqueeze(0)               # (1, B, hidden)
        return out, h_last


class _LRUStack(nn.Module):
    """Stack of LRUCells to match num_layers behaviour of LSTM/GRU."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers
        self.cells = nn.ModuleList(
            [_LRUCell(sizes[i], hidden_size) for i in range(num_layers)]
        )

    def forward(self, x, h0=None):
        out = x
        for cell in self.cells:
            out, h0 = cell(out, h0)
        return out, h0


# ── shared multi-head network ─────────────────────────────────────────────────

class _RNNNet(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, n_horizons: int, cell: str):
        super().__init__()
        if cell == "lstm":
            self.rnn = nn.LSTM(1, hidden_size, n_layers, batch_first=True)
        elif cell == "gru":
            self.rnn = nn.GRU(1, hidden_size, n_layers, batch_first=True)
        elif cell == "lru":
            self.rnn = _LRUStack(1, hidden_size, n_layers)
        else:
            raise ValueError(f"cell must be lstm / gru / lru, got '{cell}'")
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_horizons)])

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _  = self.rnn(x)               # (B, L, hidden)
        last    = out[:, -1, :]             # (B, hidden) — last timestep
        return torch.cat([h(last) for h in self.heads], dim=1)   # (B, n_h)


# ── forecaster ────────────────────────────────────────────────────────────────

class RNNForecaster(BaseForecaster):
    """
    Direct multi-step forecaster using LSTM, GRU, or LRU.

    Parameters
    ----------
    cell        : 'lstm' | 'gru' | 'lru'.
    hidden_size : recurrent hidden dimension.
    n_layers    : number of stacked recurrent layers.
    lookback    : input window length.
    n_epochs    : maximum training epochs.
    lr          : Adam learning rate.
    batch_size  : mini-batch size.
    patience    : early-stopping patience (epochs without val improvement).
                  Set to None to disable early stopping.
    val_frac    : fraction of windows reserved for validation (chronological
                  split — the most recent val_frac of the training windows).
    seed        : torch manual seed.
    """

    def __init__(
        self,
        cell:        str        = "lstm",
        hidden_size: int        = 32,
        n_layers:    int        = 1,
        lookback:    int        = 20,
        n_epochs:    int        = 500,
        lr:          float      = 1e-3,
        batch_size:  int        = 64,
        patience:    int | None = 20,
        val_frac:    float      = 0.2,
        seed:        int        = 42,
    ):
        if not _TORCH:
            raise ImportError("RNN models require PyTorch: pip install torch")
        self.cell        = cell
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.lookback    = lookback
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.batch_size  = batch_size
        self.patience    = patience
        self.val_frac    = val_frac
        self.seed        = seed
        self._net: _RNNNet | None     = None
        self._horizons: list[int]     = []
        self._buf: deque | None       = None
        # Fitted diagnostics (set after fit)
        self.n_epochs_   = 0
        self.best_val_loss_ = np.inf

    def _build_windows(self, history: np.ndarray, horizons: list[int]):
        T, L, H = len(history), self.lookback, max(horizons)
        X_list, Y_list = [], []
        for t in range(L - 1, T - H):
            X_list.append(history[t - L + 1: t + 1])
            Y_list.append([history[t + h] for h in horizons])
        if not X_list:
            return np.empty((0, L), np.float32), np.empty((0, len(horizons)), np.float32)
        return np.array(X_list, np.float32), np.array(Y_list, np.float32)

    def fit(self, history: np.ndarray, horizons: list[int]) -> "RNNForecaster":
        torch.manual_seed(self.seed)
        history        = np.asarray(history, dtype=np.float32)
        self._horizons = horizons
        n_h            = len(horizons)

        self._net = _RNNNet(self.hidden_size, self.n_layers, n_h, self.cell)

        X, Y = self._build_windows(history, horizons)
        N    = len(X)

        if N < 8:
            self._net.eval()
            self._buf = deque(history[-self.lookback:].tolist(), maxlen=self.lookback)
            return self

        # ── chronological train / val split ──────────────────────────────
        use_es   = (self.patience is not None) and (N >= 10)
        val_size = max(1, min(int(N * self.val_frac), N - 4)) if use_es else 0
        n_train  = N - val_size

        X_tr = torch.from_numpy(X[:n_train]).unsqueeze(-1)   # (N_tr, L, 1)
        Y_tr = torch.from_numpy(Y[:n_train])
        dl   = DataLoader(TensorDataset(X_tr, Y_tr),
                          batch_size=self.batch_size, shuffle=True)

        if use_es:
            X_val = torch.from_numpy(X[n_train:]).unsqueeze(-1)
            Y_val = torch.from_numpy(Y[n_train:])

        opt     = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val   = np.inf
        best_state = None
        no_improve = 0

        self._net.train()
        for epoch in range(1, self.n_epochs + 1):
            # ── one epoch on training set ─────────────────────────────────
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                opt.step()

            # ── early stopping check on val set ───────────────────────────
            if use_es:
                self._net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(self._net(X_val), Y_val))
                self._net.train()

                if val_loss < best_val - 1e-6:
                    best_val   = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        self.n_epochs_      = epoch
        self.best_val_loss_ = best_val if use_es else np.nan

        # Restore best-validation weights
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._net.eval()
        self._buf = deque(history[-self.lookback:].tolist(), maxlen=self.lookback)
        return self

    def update(self, x: float) -> "RNNForecaster":
        self._buf.append(float(x))
        return self

    def predict(self, h: int) -> float:
        h_idx = self._horizons.index(h)
        x     = np.array(self._buf, dtype=np.float32)
        x_t   = torch.from_numpy(x).unsqueeze(0).unsqueeze(-1)   # (1, L, 1)
        with torch.no_grad():
            out = self._net(x_t)
        return float(out[0, h_idx])


# Backwards-compatible alias
LSTMForecaster = RNNForecaster
