"""FastAPI service: train, forecast, and backtest time-series models."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .backtest import BacktestConfig, walk_forward
from .data import SeriesConfig, generate_series
from .train import (
    MODEL_NAMES,
    TrainConfig,
    fit_scaler,
    forecast_from_history,
    train_all,
    train_one_model,
)
from .windowing import make_windows

app = FastAPI(
    title="Time-Series Forecasting Engine",
    description=(
        "Train LSTM / Transformer forecasters on synthetic multivariate series, "
        "serve forecasts, and run walk-forward backtests. "
        "All metrics are computed on seeded synthetic data."
    ),
    version=__version__,
)

_SYNTHETIC_NOTE = "synthetic-data performance: seeded generator, no real-world data"


class TrainRequest(BaseModel):
    n_steps: int = Field(default=600, ge=100, le=20000)
    n_features: int = Field(default=3, ge=1, le=10)
    seed: int = 7
    lookback: int = Field(default=48, ge=4, le=512)
    horizon: int = Field(default=12, ge=1, le=128)
    hidden_size: int = Field(default=32, ge=4, le=512)
    num_layers: int = Field(default=1, ge=1, le=6)
    epochs: int = Field(default=8, ge=1, le=200)
    model_dir: str = "models"


class ForecastRequest(BaseModel):
    series: list[list[float]] = Field(description="Multivariate history, rows are time steps")
    horizon: int = Field(default=12, ge=1, le=128)
    model: Literal["lstm", "transformer"] = "lstm"
    lookback: int = Field(default=48, ge=4, le=512)
    hidden_size: int = Field(default=16, ge=4, le=512)
    epochs: int = Field(default=5, ge=1, le=200)
    seed: int = 42


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "models": list(MODEL_NAMES)}


@app.post("/train")
def train(req: TrainRequest) -> dict:
    """Train both models on fresh synthetic data; persist checkpoints."""
    series = generate_series(
        SeriesConfig(n_steps=req.n_steps, n_features=req.n_features, seed=req.seed)
    )
    cfg = TrainConfig(
        lookback=req.lookback,
        horizon=req.horizon,
        hidden_size=req.hidden_size,
        num_layers=req.num_layers,
        epochs=req.epochs,
        seed=req.seed,
        model_dir=req.model_dir,
    )
    metrics = train_all(series.values, cfg)
    return {
        "metrics": metrics,
        "n_steps": req.n_steps,
        "n_features": req.n_features,
        "seed": req.seed,
        "note": _SYNTHETIC_NOTE,
    }


@app.post("/forecast")
def forecast(req: ForecastRequest) -> dict:
    """Fit the chosen model on the provided series, then forecast ahead."""
    values = np.asarray(req.series, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < req.lookback + req.horizon + 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"series must be 2-D with at least lookback + horizon + 1 "
                f"({req.lookback + req.horizon + 1}) rows"
            ),
        )
    cfg = TrainConfig(
        lookback=req.lookback,
        horizon=req.horizon,
        hidden_size=req.hidden_size,
        num_layers=1,
        epochs=req.epochs,
        seed=req.seed,
        model_dir=".forecast-tmp",
    )
    X, y = make_windows(values, req.lookback, req.horizon)
    scaler = fit_scaler(X, y)
    Xs = torch.from_numpy(scaler.transform_x(X).astype(np.float32))
    ys = torch.from_numpy(((y - scaler.y_mean) / scaler.y_std).astype(np.float32))
    model, _ = train_one_model(req.model, Xs, ys, cfg)
    preds = forecast_from_history(model, scaler, values, req.lookback, req.horizon)
    return {
        "model": req.model,
        "horizon": req.horizon,
        "forecast": [float(v) for v in preds],
        "note": _SYNTHETIC_NOTE + " (model fit on the submitted series)",
    }


@app.get("/backtest")
def backtest(
    n_steps: int = 800,
    seed: int = 7,
    lookback: int = 48,
    horizon: int = 12,
    train_size: int = 400,
    step: int = 100,
    season_period: int = 24,
    hidden_size: int = 16,
    epochs: int = 4,
) -> dict:
    """Walk-forward backtest of both models vs a seasonal-naive baseline."""
    series = generate_series(SeriesConfig(n_steps=n_steps, n_features=3, seed=seed))
    cfg = BacktestConfig(
        lookback=lookback,
        horizon=horizon,
        train_size=train_size,
        step=step,
        season_period=season_period,
        hidden_size=hidden_size,
        epochs=epochs,
        seed=seed,
    )
    report = walk_forward(series.values, cfg)
    report["note"] = _SYNTHETIC_NOTE
    return report
