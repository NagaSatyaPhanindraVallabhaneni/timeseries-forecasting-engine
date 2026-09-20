import numpy as np
import pytest

from timeseries_forecasting_engine.metrics import all_metrics, mae, mape, rmse


def test_mae_hand_computed():
    # |1-2| + |2-2| + |3-4| = 2 -> 2/3
    assert mae([1, 2, 3], [2, 2, 4]) == pytest.approx(2 / 3)


def test_rmse_hand_computed():
    # ((3^2 + 4^2) / 2) ** 0.5 = sqrt(12.5)
    assert rmse([0, 0], [3, 4]) == pytest.approx(12.5**0.5)


def test_mape_hand_computed():
    # (0.1 + 0.1 + 0.1) / 3 = 0.1
    assert mape([10, 20, 40], [11, 18, 44]) == pytest.approx(0.1)


def test_mape_zero_actual_is_finite():
    val = mape([0.0, 10.0], [1.0, 11.0])
    assert np.isfinite(val)


def test_empty_raises():
    with pytest.raises(ValueError):
        mae([], [])
    with pytest.raises(ValueError):
        rmse(np.array([]), np.array([]))


def test_all_metrics_keys():
    m = all_metrics([1, 2], [1, 3])
    assert set(m) == {"mae", "rmse", "mape"}
    assert m["mae"] == pytest.approx(0.5)
