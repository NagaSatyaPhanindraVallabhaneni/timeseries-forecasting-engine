import numpy as np
import pytest

from timeseries_forecasting_engine.windowing import make_windows, walk_forward_splits


def test_window_shapes():
    values = np.arange(40, dtype=np.float64).reshape(20, 2)
    X, y = make_windows(values, lookback=5, horizon=3)
    assert X.shape == (13, 5, 2)
    assert y.shape == (13, 3)
    assert X.dtype == np.float32 and y.dtype == np.float32


def test_window_values_hand_checked():
    values = np.arange(12, dtype=np.float64).reshape(6, 2)  # rows 0..5, cols f0/f1
    X, y = make_windows(values, lookback=2, horizon=2, target_idx=1)
    # first window: rows 0,1 ; target rows 2,3 of column 1 -> [5, 7]
    assert np.array_equal(X[0], values[0:2])
    assert np.array_equal(y[0], np.array([5.0, 7.0], dtype=np.float32))
    # last window starts at row 6-2-2=2: rows 2,3 ; target rows 4,5 col1 -> [9, 11]
    assert np.array_equal(X[-1], values[2:4])
    assert np.array_equal(y[-1], np.array([9.0, 11.0], dtype=np.float32))


def test_too_short_series_raises():
    values = np.zeros((5, 2))
    with pytest.raises(ValueError):
        make_windows(values, lookback=4, horizon=3)
    with pytest.raises(ValueError):
        make_windows(values, lookback=0, horizon=1)
    with pytest.raises(ValueError):
        make_windows(np.zeros(5), lookback=2, horizon=1)  # not 2-D


def test_walk_forward_splits():
    splits = walk_forward_splits(n=100, train_size=60, horizon=10, step=20)
    assert len(splits) == 2
    (tr0, te0), (tr1, te1) = splits
    assert (tr0.start, tr0.stop) == (0, 60)
    assert (te0.start, te0.stop) == (60, 70)
    assert (tr1.start, tr1.stop) == (0, 80)
    assert (te1.start, te1.stop) == (80, 90)


def test_walk_forward_splits_none_fit_raises():
    with pytest.raises(ValueError):
        walk_forward_splits(n=50, train_size=45, horizon=10, step=5)
