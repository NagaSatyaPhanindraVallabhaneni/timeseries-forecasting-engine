import numpy as np
import pytest

from timeseries_forecasting_engine.data import SeriesConfig, generate_series


def test_deterministic_given_seed():
    a = generate_series(SeriesConfig(n_steps=200, seed=123))
    b = generate_series(SeriesConfig(n_steps=200, seed=123))
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.anomaly_mask, b.anomaly_mask)


def test_different_seeds_differ():
    a = generate_series(SeriesConfig(n_steps=200, seed=1))
    b = generate_series(SeriesConfig(n_steps=200, seed=2))
    assert not np.array_equal(a.values, b.values)


def test_shapes_and_dtypes():
    s = generate_series(SeriesConfig(n_steps=150, n_features=4, seed=7))
    assert s.values.shape == (150, 4)
    assert s.values.dtype == np.float64
    assert s.anomaly_mask.shape == (150,)
    assert s.anomaly_mask.dtype == bool


def test_anomaly_injection_extremes():
    none_cfg = SeriesConfig(n_steps=100, anomaly_prob=0.0, seed=3)
    all_cfg = SeriesConfig(n_steps=100, anomaly_prob=1.0, seed=3)
    assert not generate_series(none_cfg).anomaly_mask.any()
    assert generate_series(all_cfg).anomaly_mask.all()
    # with anomalies enabled by default, a 2000-step run should contain some
    s = generate_series(SeriesConfig(n_steps=2000, seed=7))
    assert 0 < s.anomaly_mask.sum() < 2000


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        generate_series(SeriesConfig(n_steps=0))
    with pytest.raises(ValueError):
        generate_series(SeriesConfig(n_features=0))
    with pytest.raises(ValueError):
        generate_series(SeriesConfig(periods=(24.0,), amplitudes=(1.0, 2.0)))
