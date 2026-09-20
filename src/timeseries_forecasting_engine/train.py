"""Training loop for both forecasters, with checkpoint saving."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import metrics as M
from .models import build_model
from .windowing import make_windows

MODEL_NAMES = ("lstm", "transformer")


@dataclass
class TrainConfig:
    lookback: int = 48
    horizon: int = 12
    hidden_size: int = 64
    num_layers: int = 2
    n_heads: int = 4
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 42
    model_dir: str = "models"


@dataclass
class Scaler:
    """Per-feature mean/std fitted on the training windows."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


def fit_scaler(X_train: np.ndarray, y_train: np.ndarray) -> Scaler:
    x_mean = X_train.mean(axis=(0, 1))
    x_std = X_train.std(axis=(0, 1))
    x_std[x_std == 0.0] = 1.0
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    return Scaler(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_model(
    name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[nn.Module, list[float]]:
    """Train a single model; returns (model, per-epoch train loss)."""
    _seed_all(cfg.seed)
    n_features = X_train.shape[2]
    model_kwargs = {
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "horizon": cfg.horizon,
    }
    if name == "transformer":
        model_kwargs["n_heads"] = cfg.n_heads
    model = build_model(name, n_features=n_features, **model_kwargs)

    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=gen,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    history: list[float] = []
    for _ in range(cfg.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        history.append(epoch_loss / len(X_train))
    return model, history


def evaluate(
    model: nn.Module, scaler: Scaler, X_val: np.ndarray, y_val: np.ndarray
) -> dict[str, float]:
    """Validation metrics in the original (unscaled) target units."""
    model.eval()
    with torch.no_grad():
        Xs = torch.from_numpy(scaler.transform_x(X_val).astype(np.float32))
        pred = model(Xs).numpy()
    pred = scaler.inverse_y(pred)
    return M.all_metrics(y_val, pred)


def save_checkpoint(model: nn.Module, scaler: Scaler, cfg: TrainConfig, name: str) -> Path:
    """Persist weights + everything needed to forecast later."""
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "n_features": int(scaler.x_mean.shape[0]),
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "n_heads": cfg.n_heads,
        "horizon": cfg.horizon,
        "lookback": cfg.lookback,
        "x_mean": scaler.x_mean.tolist(),
        "x_std": scaler.x_std.tolist(),
        "y_mean": scaler.y_mean,
        "y_std": scaler.y_std,
        "state_dict": model.state_dict(),
    }
    path = model_dir / f"{name}.pt"
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path) -> tuple[nn.Module, Scaler, dict]:
    """Restore a model, its scaler, and its config dict from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kwargs = {
        "hidden_size": payload["hidden_size"],
        "num_layers": payload["num_layers"],
        "horizon": payload["horizon"],
    }
    if payload["name"] == "transformer":
        kwargs["n_heads"] = payload["n_heads"]
    model = build_model(payload["name"], n_features=payload["n_features"], **kwargs)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    scaler = Scaler(
        x_mean=np.array(payload["x_mean"]),
        x_std=np.array(payload["x_std"]),
        y_mean=payload["y_mean"],
        y_std=payload["y_std"],
    )
    return model, scaler, payload


def forecast_from_history(
    model: nn.Module, scaler: Scaler, history: np.ndarray, lookback: int, horizon: int
) -> np.ndarray:
    """Direct multi-step forecast from the trailing ``lookback`` observations."""
    window = np.asarray(history, dtype=np.float32)[-lookback:]
    Xs = torch.from_numpy(scaler.transform_x(window[None, :, :]).astype(np.float32))
    model.eval()
    with torch.no_grad():
        pred = model(Xs).numpy()[0]
    return scaler.inverse_y(pred)[:horizon]


def train_all(values: np.ndarray, cfg: TrainConfig) -> dict[str, dict[str, float]]:
    """Train both models on ``values``; save checkpoints; return val metrics."""
    X, y = make_windows(values, cfg.lookback, cfg.horizon)
    n_val = max(1, int(len(X) * cfg.val_fraction))
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    scaler = fit_scaler(X_train, y_train)
    Xs_train = torch.from_numpy(scaler.transform_x(X_train).astype(np.float32))
    ys_train = torch.from_numpy(((y_train - scaler.y_mean) / scaler.y_std).astype(np.float32))

    results: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        model, _history = train_one_model(name, Xs_train, ys_train, cfg)
        results[name] = evaluate(model, scaler, X_val, y_val)
        save_checkpoint(model, scaler, cfg, name)

    meta_path = Path(cfg.model_dir) / "metrics.json"
    meta_path.write_text(json.dumps({"config": asdict(cfg), "metrics": results}, indent=2))
    return results
