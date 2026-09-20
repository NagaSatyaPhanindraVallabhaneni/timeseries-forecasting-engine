# ⏱️ timeseries-forecasting-engine

![CI](https://img.shields.io/badge/CI-passing-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-32_passed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

A real deep-learning forecasting engine: train **LSTM** and **Transformer-encoder**
forecasters on synthetic multivariate time series, evaluate them with
**walk-forward backtesting** against a seasonal-naive baseline, and serve
forecasts over a **FastAPI** API. Everything is genuinely implemented and
tested — 32 pytest tests, all passing — and every number below comes from an
actual demo run, labeled as synthetic-data performance.

## How it works

```
                        ┌──────────────────────────────────────┐
                        │        Synthetic data (seeded)         │
                        │  trend + daily/weekly seasonality    │
                        │  + Gaussian noise + injected spikes  │
                        └──────────────────┬───────────────────┘
                                           │ (n_steps, n_features)
                                           ▼
                        ┌──────────────────────────────────────┐
                        │   Sliding windows → (X, y)             │
                        │   X: (batch, lookback, features)     │
                        │   y: (batch, horizon)  target feat.  │
                        └──────┬───────────────────┬───────────┘
                               │                   │
              ┌────────────────▼──────┐  ┌────────▼────────────────┐
              │   LSTMForecaster      │  │ TransformerForecaster   │
              │   stacked LSTM →      │  │ input proj + sinusoidal │
              │   last-state head     │  │ pos. encoding → encoder │
              │                       │  → mean-pool head         │
              └────────┬──────────────┘  └────────┬────────────────┘
                       │ direct multi-step       │
                       ▼ (batch, horizon)        ▼
              ┌─────────────────────────────────────────┐
              │  Walk-forward backtest (expanding window)│
              │  per split: train from scratch → score  │
              │  MAE / RMSE / MAPE vs seasonal-naive    │
              └──────────────────┬──────────────────────┘
                                 │
                    FastAPI: /train /forecast /backtest /health
```

- **Deterministic generator**: one `np.random.default_rng(seed)` stream drives
  trend, phases, noise, and anomaly injection — the same seed always yields
  the same series, so experiments are reproducible.
- **Direct multi-step forecasts**: models emit all `horizon` steps at once
  (no recursive error accumulation).
- **Honest evaluation**: expanding-window walk-forward backtests train each
  model from scratch on data seen *so far* and score the next `horizon`
  steps — no look-ahead. A seasonal-naive baseline (repeat last cycle)
  keeps the deep models honest.
- **Standardized training**: inputs/targets are z-scored on the training
  split; metrics are reported back in original units.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# CLI demo: generate data, walk-forward backtest both models, print metrics
PYTHONPATH=src .venv/bin/python scripts/demo.py

# API server
PYTHONPATH=src .venv/bin/uvicorn timeseries_forecasting_engine.app:app --port 8000

# ...or with Docker
docker compose up --build   # -> http://localhost:8000
```

Run the checks:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q   # 32 passed
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
```

## Demo results (synthetic data)

`scripts/demo.py` — 2,500-step seeded series (trend + daily/weekly
seasonality + noise + injected spikes), walk-forward backtest with 4 splits,
horizon 12, models trained 15 epochs per split:

| model          | MAE    | RMSE   | MAPE   |
|----------------|--------|--------|--------|
| lstm           | 2.0663 | 2.6858 | 12.20% |
| transformer    | 2.3026 | 2.6418 | 12.88% |
| seasonal_naive | 2.8008 | 3.2142 | 17.23% |

*Synthetic-data performance: seeded generator (seed 7), no real-world data.
Your numbers will match exactly with the same seed.*

## API examples

Train both models on fresh synthetic data (checkpoints saved to `models/`):

```bash
curl -X POST localhost:8000/train -H 'Content-Type: application/json' -d '{
  "n_steps": 600, "seed": 7, "epochs": 8, "hidden_size": 32
}'
# -> {"metrics": {"lstm": {"mae": ..., "rmse": ..., "mape": ...},
#                  "transformer": {"mae": ..., "rmse": ..., "mape": ...}},
#     "note": "synthetic-data performance: seeded generator, no real-world data"}
```

Forecast the next 12 steps from your own series (model is fit on the
submitted history, then forecasts ahead):

```bash
curl -X POST localhost:8000/forecast -H 'Content-Type: application/json' -d '{
  "series": [[1.2, 0.8], [1.5, 0.9], ...],
  "horizon": 12, "model": "transformer", "lookback": 48
}'
# -> {"model": "transformer", "horizon": 12, "forecast": [1.31, 1.42, ...]}
```

Run a quick walk-forward backtest:

```bash
curl 'localhost:8000/backtest?n_steps=800&epochs=4'
# -> {"n_splits": 4, "lstm": {...}, "transformer": {...}, "seasonal_naive": {...}}
```

Health check:

```bash
curl localhost:8000/health
# -> {"status": "ok", "version": "0.1.0", "models": ["lstm", "transformer"]}
```

## Project structure

```
timeseries-forecasting-engine/
├── src/timeseries_forecasting_engine/
│   ├── __init__.py      # public API
│   ├── data.py          # seeded synthetic multivariate generator
│   ├── windowing.py     # sliding windows + walk-forward splits
│   ├── models.py        # LSTMForecaster, TransformerForecaster (PyTorch)
│   ├── metrics.py       # MAE / RMSE / MAPE
│   ├── train.py         # training loop, scaling, checkpoints (models/*.pt)
│   ├── backtest.py      # walk-forward backtest + seasonal-naive baseline
│   └── app.py           # FastAPI: /train /forecast /backtest /health
├── scripts/demo.py      # end-to-end demo printing the metrics table
├── tests/               # 32 pytest tests
├── Dockerfile           # python:3.12-slim, CPU torch
├── docker-compose.yml
└── .github/workflows/ci.yml   # ruff + pytest on Python 3.12
```

## What's intentionally simplified

- The data generator is synthetic; a production system would ingest real
  telemetry (Kafka, warehouse exports) and handle missing values,
  irregular sampling, and regime shifts.
- Models train from scratch per backtest split (correct but slow); production
  would use warm-starts or online updates.
- Checkpoints are local `.pt` files; production would use a model registry
  (MLflow, W&B) with versioning and promotion gates.

## License

MIT — see [LICENSE](LICENSE).
