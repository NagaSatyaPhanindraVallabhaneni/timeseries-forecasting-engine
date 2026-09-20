import numpy as np
from fastapi.testclient import TestClient

from timeseries_forecasting_engine.app import app
from timeseries_forecasting_engine.data import SeriesConfig, generate_series

client = TestClient(app)


def _series_payload(rows: int = 60, features: int = 2, seed: int = 5):
    s = generate_series(SeriesConfig(n_steps=rows, n_features=features, seed=seed))
    return s.values.tolist()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"lstm", "transformer"}


def test_forecast_smoke():
    r = client.post(
        "/forecast",
        json={
            "series": _series_payload(),
            "horizon": 2,
            "model": "lstm",
            "lookback": 8,
            "hidden_size": 8,
            "epochs": 1,
            "seed": 5,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "lstm"
    assert len(body["forecast"]) == 2
    assert all(isinstance(v, float) and np.isfinite(v) for v in body["forecast"])


def test_forecast_transformer_smoke():
    r = client.post(
        "/forecast",
        json={
            "series": _series_payload(),
            "horizon": 3,
            "model": "transformer",
            "lookback": 8,
            "hidden_size": 8,
            "epochs": 1,
            "seed": 5,
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["forecast"]) == 3


def test_forecast_rejects_short_series():
    r = client.post(
        "/forecast",
        json={"series": [[1.0], [2.0]], "horizon": 2, "lookback": 8},
    )
    assert r.status_code == 422


def test_train_smoke(tmp_path):
    r = client.post(
        "/train",
        json={
            "n_steps": 120,
            "n_features": 2,
            "seed": 9,
            "lookback": 12,
            "horizon": 3,
            "hidden_size": 8,
            "num_layers": 1,
            "epochs": 1,
            "model_dir": str(tmp_path / "models"),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["metrics"]) == {"lstm", "transformer"}
    for m in body["metrics"].values():
        assert set(m) == {"mae", "rmse", "mape"}
    assert (tmp_path / "models" / "lstm.pt").exists()
    assert (tmp_path / "models" / "transformer.pt").exists()


def test_backtest_smoke():
    r = client.get(
        "/backtest",
        params={
            "n_steps": 220,
            "seed": 9,
            "lookback": 10,
            "horizon": 4,
            "train_size": 120,
            "step": 50,
            "season_period": 12,
            "hidden_size": 8,
            "epochs": 1,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_splits"] >= 1
    assert "note" in body
    assert set(body["lstm"]) == {"mae", "rmse", "mape"}
