"""PyTorch forecasters: an LSTM and a Transformer-encoder model.

Both take ``(batch, lookback, n_features)`` and emit ``(batch, horizon)``
direct multi-step forecasts of the target feature.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Stacked LSTM; the last hidden state feeds a linear forecast head."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1 or horizon < 1:
            raise ValueError("hidden_size, num_layers and horizon must be >= 1")
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, L, H)
        return self.head(out[:, -1, :])  # (B, horizon)


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerForecaster(nn.Module):
    """Transformer encoder over the lookback window, mean-pooled to a forecast."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_heads: int = 4,
        horizon: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1 or horizon < 1 or n_heads < 1:
            raise ValueError("hidden_size, num_layers, n_heads and horizon must be >= 1")
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.horizon = horizon
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.pos = PositionalEncoding(hidden_size, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)  # (B, L, H)
        h = self.pos(h)
        h = self.encoder(h)  # (B, L, H)
        return self.head(h.mean(dim=1))  # (B, horizon)


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    """Factory: ``name`` is ``"lstm"`` or ``"transformer"``."""
    if name == "lstm":
        return LSTMForecaster(n_features=n_features, **kwargs)
    if name == "transformer":
        return TransformerForecaster(n_features=n_features, **kwargs)
    raise ValueError(f"unknown model {name!r}; expected 'lstm' or 'transformer'")
