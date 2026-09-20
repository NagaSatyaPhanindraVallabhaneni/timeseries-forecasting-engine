"""Time-series forecasting engine: synthetic data, LSTM + Transformer models,
walk-forward backtesting, and a FastAPI serving layer."""

from .backtest import BacktestConfig, seasonal_naive_forecast, walk_forward
from .data import SeriesConfig, SyntheticSeries, generate_series
from .metrics import all_metrics, mae, mape, rmse
from .models import LSTMForecaster, TransformerForecaster, build_model
from .train import TrainConfig, forecast_from_history, load_checkpoint, train_all
from .windowing import make_windows, walk_forward_splits

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "LSTMForecaster",
    "SeriesConfig",
    "SyntheticSeries",
    "TrainConfig",
    "TransformerForecaster",
    "all_metrics",
    "build_model",
    "forecast_from_history",
    "generate_series",
    "load_checkpoint",
    "mae",
    "make_windows",
    "mape",
    "rmse",
    "seasonal_naive_forecast",
    "train_all",
    "walk_forward",
    "walk_forward_splits",
]
