"""Sliding-window dataset construction and walk-forward split helpers."""

from __future__ import annotations

import numpy as np


def make_windows(
    values: np.ndarray,
    lookback: int,
    horizon: int,
    target_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) supervised windows from a multivariate series.

    ``X[i]`` is the ``lookback``-long input window ending at position
    ``i + lookback`` and ``y[i]`` the next ``horizon`` values of the target
    feature. Returns ``X`` of shape ``(n_windows, lookback, n_features)``
    and ``y`` of shape ``(n_windows, horizon)`` as float32.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D array (n_steps, n_features)")
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be >= 1")
    if not 0 <= target_idx < values.shape[1]:
        raise ValueError("target_idx out of range")

    n = values.shape[0]
    n_windows = n - lookback - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"series too short: need at least lookback + horizon "
            f"({lookback + horizon}) steps, got {n}"
        )

    x_idx = np.arange(lookback)[None, :] + np.arange(n_windows)[:, None]
    X = values[x_idx]  # (n_windows, lookback, n_features)
    y_idx = np.arange(horizon)[None, :] + lookback + np.arange(n_windows)[:, None]
    y = values[y_idx, target_idx]  # (n_windows, horizon)
    return X.astype(np.float32), y.astype(np.float32)


def walk_forward_splits(
    n: int,
    train_size: int,
    horizon: int,
    step: int,
) -> list[tuple[slice, slice]]:
    """Expanding-window walk-forward splits.

    Split ``k`` trains on ``[0, end)`` and evaluates on ``[end, end + horizon)``,
    where ``end`` starts at ``train_size`` and advances by ``step``.
    Returns a list of ``(train_slice, test_slice)`` pairs.
    """
    if train_size < 1 or horizon < 1 or step < 1:
        raise ValueError("train_size, horizon and step must be >= 1")
    splits: list[tuple[slice, slice]] = []
    end = train_size
    while end + horizon <= n:
        splits.append((slice(0, end), slice(end, end + horizon)))
        end += step
    if not splits:
        raise ValueError("no walk-forward splits fit in the series")
    return splits
