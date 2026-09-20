"""Walk-forward backtesting: expanding-window evaluation vs a naive baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import metrics as M
from .train import MODEL_NAMES, TrainConfig, fit_scaler, forecast_from_history, train_one_model
from .windowing import make_windows, walk_forward_splits


@dataclass
class BacktestConfig:
    lookback: int = 48
    horizon: int = 12
    train_size: int = 600
    step: int = 120
    season_period: int = 24
    hidden_size: int = 32
    num_layers: int = 1
    n_heads: int = 4
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42


def seasonal_naive_forecast(
    train_target: np.ndarray, horizon: int, season_period: int
) -> np.ndarray:
    """Repeat the most recent seasonal cycle (or last value) as the forecast."""
    train_target = np.asarray(train_target, dtype=np.float64).ravel()
    if train_target.size == 0:
        raise ValueError("train_target must not be empty")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    season = (
        train_target[-season_period:] if train_target.size >= season_period else train_target[-1:]
    )
    reps = int(np.ceil(horizon / season.size))
    return np.tile(season, reps)[:horizon]


def _train_cfg(cfg: BacktestConfig) -> TrainConfig:
    return TrainConfig(
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        n_heads=cfg.n_heads,
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        val_fraction=0.15,
        seed=cfg.seed,
        model_dir=".backtest-tmp",  # never persisted; checkpoints not needed here
    )


def walk_forward(values: np.ndarray, cfg: BacktestConfig) -> dict:
    """Expanding-window backtest of both deep models + seasonal-naive baseline.

    For each split, both models are trained from scratch on the data seen so
    far, then scored on the next ``horizon`` steps. Metrics are pooled over
    all splits. Returns per-model ``mae``/``rmse``/``mape`` plus ``n_splits``.
    """
    values = np.asarray(values, dtype=np.float64)
    target = values[:, 0]
    tcfg = _train_cfg(cfg)

    preds: dict[str, list[np.ndarray]] = {name: [] for name in (*MODEL_NAMES, "seasonal_naive")}
    trues: list[np.ndarray] = []

    splits = walk_forward_splits(len(values), cfg.train_size, cfg.horizon, cfg.step)
    for train_slice, test_slice in splits:
        train_vals = values[train_slice]
        y_true = target[test_slice]
        trues.append(y_true)

        preds["seasonal_naive"].append(
            seasonal_naive_forecast(target[train_slice], cfg.horizon, cfg.season_period)
        )

        # train each deep model on everything seen so far, forecast the test window
        for name in MODEL_NAMES:
            X, y = make_windows(train_vals, cfg.lookback, cfg.horizon)
            n_val = max(1, int(len(X) * 0.15))
            scaler = fit_scaler(X[:-n_val], y[:-n_val])
            Xs = torch.from_numpy(scaler.transform_x(X[:-n_val]).astype(np.float32))
            ys = torch.from_numpy(((y[:-n_val] - scaler.y_mean) / scaler.y_std).astype(np.float32))
            model, _ = train_one_model(name, Xs, ys, tcfg)
            hist = train_vals[-cfg.lookback :]
            preds[name].append(
                forecast_from_history(model, scaler, hist, cfg.lookback, cfg.horizon)
            )

    y_all = np.concatenate(trues)
    report: dict = {"n_splits": len(splits)}
    for name, plist in preds.items():
        report[name] = M.all_metrics(y_all, np.concatenate(plist))
    return report
