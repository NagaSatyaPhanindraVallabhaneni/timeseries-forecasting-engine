#!/usr/bin/env python3
"""End-to-end demo: generate data, train both models, walk-forward backtest.

Prints a metrics table comparing the LSTM and Transformer forecasters
against a seasonal-naive baseline. All numbers are synthetic-data
performance from the seeded generator — never real-world results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from timeseries_forecasting_engine.backtest import BacktestConfig, walk_forward
from timeseries_forecasting_engine.data import SeriesConfig, generate_series

SEED = 7


def main() -> None:
    print("== Time-Series Forecasting Engine — demo ==")
    print(f"Generating synthetic multivariate series (seed={SEED}) ...")
    series = generate_series(SeriesConfig(n_steps=2500, n_features=3, seed=SEED))
    print(f"  shape={series.values.shape}, injected anomalies={int(series.anomaly_mask.sum())}")

    print("Running walk-forward backtest (this trains both models per split) ...")
    cfg = BacktestConfig(
        lookback=48,
        horizon=12,
        train_size=1200,
        step=400,
        season_period=24,
        hidden_size=32,
        num_layers=2,
        n_heads=4,
        epochs=15,
        seed=SEED,
    )
    report = walk_forward(series.values, cfg)

    print()
    print(f"Walk-forward backtest — {report['n_splits']} splits, horizon={cfg.horizon}")
    print(f"{'model':<16}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}")
    print("-" * 46)
    for name in ("lstm", "transformer", "seasonal_naive"):
        m = report[name]
        print(f"{name:<16}{m['mae']:>10.4f}{m['rmse']:>10.4f}{m['mape']:>9.2%}")
    print("-" * 46)
    print(
        "* synthetic-data performance: seeded generator "
        f"(seed={SEED}), trend + seasonality + noise + injected anomalies"
    )


if __name__ == "__main__":
    main()
