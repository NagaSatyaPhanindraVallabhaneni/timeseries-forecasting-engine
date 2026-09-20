"""CLI: train LSTM + Transformer forecasters on synthetic data, save checkpoints.

Usage:
    PYTHONPATH=src python train.py [--n-steps 2000] [--epochs 20] [--model-dir models]
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from timeseries_forecasting_engine.data import SeriesConfig, generate_series
from timeseries_forecasting_engine.train import TrainConfig, train_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    series = generate_series(
        SeriesConfig(n_steps=args.n_steps, n_features=args.n_features, seed=args.seed)
    )
    cfg = TrainConfig(
        lookback=args.lookback,
        horizon=args.horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        seed=args.seed,
        model_dir=args.model_dir,
    )
    metrics = train_all(series.values, cfg)
    for name, m in metrics.items():
        print(f"{name}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} MAPE={m['mape'] * 100:.2f}%")
    print(f"checkpoints saved to {args.model_dir}/ (synthetic-data performance)")


if __name__ == "__main__":
    main()
