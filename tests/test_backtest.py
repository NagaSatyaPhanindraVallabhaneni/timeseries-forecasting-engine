import numpy as np
import pytest

from timeseries_forecasting_engine.backtest import (
    BacktestConfig,
    seasonal_naive_forecast,
    walk_forward,
)
from timeseries_forecasting_engine.data import SeriesConfig, generate_series


def test_seasonal_naive_hand_checked():
    out = seasonal_naive_forecast(np.array([1.0, 2.0, 3.0, 4.0]), horizon=3, season_period=2)
    assert np.array_equal(out, np.array([3.0, 4.0, 3.0]))


def test_seasonal_naive_short_history_falls_back_to_last_value():
    out = seasonal_naive_forecast(np.array([5.0]), horizon=2, season_period=24)
    assert np.array_equal(out, np.array([5.0, 5.0]))
    with pytest.raises(ValueError):
        seasonal_naive_forecast(np.array([]), horizon=2, season_period=24)


def _tiny_cfg() -> BacktestConfig:
    return BacktestConfig(
        lookback=12,
        horizon=6,
        train_size=150,
        step=75,
        season_period=12,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        epochs=2,
        seed=11,
    )


def test_walk_forward_report_structure():
    series = generate_series(SeriesConfig(n_steps=300, n_features=2, seed=11))
    report = walk_forward(series.values, _tiny_cfg())
    assert report["n_splits"] == 2
    for name in ("lstm", "transformer", "seasonal_naive"):
        assert set(report[name]) == {"mae", "rmse", "mape"}
        for v in report[name].values():
            assert isinstance(v, float) and v >= 0.0 and np.isfinite(v)


def test_walk_forward_beats_nothing_but_runs():
    # smoke: the pipeline runs end-to-end and pooled truths match split counts
    series = generate_series(SeriesConfig(n_steps=300, n_features=2, seed=11))
    cfg = _tiny_cfg()
    report = walk_forward(series.values, cfg)
    assert report["n_splits"] * cfg.horizon == 12  # 2 splits x horizon 6
