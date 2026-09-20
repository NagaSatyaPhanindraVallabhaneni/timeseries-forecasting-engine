import pytest
import torch

from timeseries_forecasting_engine.models import (
    LSTMForecaster,
    TransformerForecaster,
    build_model,
)


def _batch():
    torch.manual_seed(0)
    return torch.randn(4, 10, 3)  # (batch, lookback, features)


def test_lstm_forward_shape():
    model = LSTMForecaster(n_features=3, hidden_size=8, num_layers=1, horizon=5)
    model.eval()
    out = model(_batch())
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_transformer_forward_shape():
    model = TransformerForecaster(n_features=3, hidden_size=8, num_layers=1, n_heads=2, horizon=5)
    model.eval()
    out = model(_batch())
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_bigger_config_has_more_params():
    small = LSTMForecaster(n_features=3, hidden_size=8, num_layers=1, horizon=2)
    big = LSTMForecaster(n_features=3, hidden_size=16, num_layers=2, horizon=2)
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    assert n_big > n_small > 0


def test_eval_mode_is_deterministic():
    model = TransformerForecaster(n_features=3, hidden_size=8, num_layers=1, n_heads=2)
    model.eval()
    x = _batch()
    assert torch.equal(model(x), model(x))


def test_build_model_factory():
    assert isinstance(build_model("lstm", n_features=2), LSTMForecaster)
    assert isinstance(build_model("transformer", n_features=2), TransformerForecaster)
    with pytest.raises(ValueError):
        build_model("xgboost", n_features=2)


def test_transformer_rejects_bad_head_split():
    with pytest.raises(ValueError):
        TransformerForecaster(n_features=2, hidden_size=7, n_heads=4)
