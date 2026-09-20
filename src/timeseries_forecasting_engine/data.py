"""Deterministic synthetic multivariate time-series generator.

Produces a target series composed of a linear trend, multiple sinusoidal
seasonalities, Gaussian noise, and occasional injected anomalies (spikes).
Extra features are lagged / correlated copies of the target plus their own
noise, mimicking real telemetry where channels move together.

Everything is seeded: the same config always yields the same series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeriesConfig:
    """Knobs for the synthetic generator."""

    n_steps: int = 1000
    n_features: int = 3
    trend_slope: float = 0.01
    periods: tuple[float, ...] = (24.0, 168.0)
    amplitudes: tuple[float, ...] = (2.0, 5.0)
    noise_std: float = 0.5
    anomaly_prob: float = 0.01
    anomaly_scale: float = 8.0
    seed: int = 42


@dataclass
class SyntheticSeries:
    """Container for one generated series."""

    values: np.ndarray  # shape (n_steps, n_features), float64
    anomaly_mask: np.ndarray  # shape (n_steps,), bool
    config: SeriesConfig


def generate_series(config: SeriesConfig | None = None) -> SyntheticSeries:
    """Generate a deterministic synthetic multivariate series.

    Reproducibility: every source of randomness draws from a single
    ``np.random.default_rng(config.seed)`` stream, so identical configs
    produce bit-identical output.
    """
    cfg = config or SeriesConfig()
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.n_features < 1:
        raise ValueError("n_features must be >= 1")
    if len(cfg.periods) != len(cfg.amplitudes):
        raise ValueError("periods and amplitudes must have the same length")

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps
    t = np.arange(n, dtype=np.float64)

    # --- target: trend + seasonalities + noise ---------------------------
    target = cfg.trend_slope * t
    for period, amp in zip(cfg.periods, cfg.amplitudes):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        target = target + amp * np.sin(2.0 * np.pi * t / period + phase)
    target = target + rng.normal(0.0, cfg.noise_std, size=n)

    # --- injected anomalies: sparse large-magnitude spikes ---------------
    anomaly_mask = rng.random(n) < cfg.anomaly_prob
    signs = rng.choice([-1.0, 1.0], size=n)
    magnitudes = rng.uniform(cfg.anomaly_scale * 0.75, cfg.anomaly_scale * 1.5, size=n)
    target = target + np.where(anomaly_mask, signs * magnitudes, 0.0)

    # --- companion features: lagged, correlated copies -------------------
    values = np.zeros((n, cfg.n_features), dtype=np.float64)
    values[:, 0] = target
    for j in range(1, cfg.n_features):
        lag = j * 3
        shifted = np.roll(target, lag)
        shifted[:lag] = target[0]  # no wrap-around artifact
        companion = (
            0.6 * shifted
            + 0.4 * np.sin(2.0 * np.pi * t / (48.0 + 12.0 * j))
            + rng.normal(0.0, cfg.noise_std * 0.5, size=n)
        )
        values[:, j] = companion

    return SyntheticSeries(values=values, anomaly_mask=anomaly_mask, config=cfg)
