"""Regression metrics for forecast evaluation (NumPy)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _as_1d(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error as a fraction (0.05 == 5%).

    Guards against division by zero with a small epsilon.
    """
    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    return float(np.mean(np.abs(yt - yp) / (np.abs(yt) + _EPS)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Convenience: all three metrics in one dict."""
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}
