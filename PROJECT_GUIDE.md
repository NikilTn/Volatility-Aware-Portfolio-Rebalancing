# Volatility-Aware Portfolio Rebalancing — Project Guide

This project implements a complete pipeline for volatility forecasting and portfolio optimization, from data loading through backtesting.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run all stages
python run_pipeline.py --stage all

# Or run individual stages
python run_pipeline.py --stage data       # Data preparation
python run_pipeline.py --stage features   # Feature engineering
python run_pipeline.py --stage models     # Train all models
python run_pipeline.py --stage evaluate   # Forecast evaluation
python run_pipeline.py --stage portfolio  # Portfolio backtest
```

## Project Structure

```
project/
├── data/
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Processed and aligned data
│   └── external/               # External data sources
├── src/
│   ├── data/                   # Data loading and splitting
│   │   ├── load_and_align.py
│   │   └── splits.py
│   ├── features/               # Feature engineering
│   │   ├── build_features.py
│   │   └── scaling.py
│   ├── models/                 # Forecasting models
│   │   ├── har_rv.py          # HAR-RV baseline
│   │   ├── garch.py           # GARCH baseline
│   │   └── tft.py             # Temporal Fusion Transformer
│   ├── training/               # TFT training scripts
│   │   ├── tft_dataset.py
│   │   ├── train_tft.py
│   │   └── predict_tft.py
│   ├── evaluation/             # Forecast evaluation
│   │   ├── forecast_metrics.py
│   │   └── regime_dm.py
│   └── portfolio/              # Portfolio management
│       ├── weights.py
│       └── backtest.py
├── configs/
│   └── config.yaml             # Main configuration
├── checkpoints/                # Model checkpoints
├── results/
│   ├── preds/                  # Model predictions
│   ├── tables/                 # Evaluation metrics
│   ├── figs/                   # Visualizations
│   └── backtests/              # Portfolio results
├── notebooks/                  # Jupyter notebooks (optional)
├── run_pipeline.py             # Master pipeline script
└── requirements.txt            # Python dependencies
```

## Pipeline Stages

### Stage 1: Data Preparation

**Script:** `src/data/load_and_align.py`, `src/data/splits.py`

- Downloads price data from Yahoo Finance
- Computes realized volatility
- Downloads VIX data
- Aligns all data on common trading calendar
- Creates train/val/test splits with embargo

**Outputs:**
- `data/processed/aligned_panel.parquet`
- `data/processed/split_indices.json`

### Stage 2: Feature Engineering

**Script:** `src/features/build_features.py`, `src/features/scaling.py`

- Computes rolling statistics (returns, volatility)
- Creates VIX-based features
- Adds calendar features
- Computes targets for horizons 1, 5, 22 days
- Scales features using StandardScaler (fit on train only)

**Outputs:**
- `data/processed/features_{train,val,test}.parquet`
- `data/processed/features_{train,val,test}_scaled.parquet`
- `data/processed/scaler.pkl`

### Stage 3: Baseline Models

**Scripts:** `src/models/har_rv.py`, `src/models/garch.py`

#### HAR-RV (Heterogeneous Autoregressive)
- Forecasts using daily, weekly, monthly log RV
- Linear regression model
- Fast and interpretable

#### GARCH(1,1)
- Forecasts using conditional variance
- Rolling window with periodic refitting
- Captures volatility clustering

**Outputs:**
- `results/preds/har_H{1,5,22}.csv`
- `results/preds/garch_H{1,5,22}.csv`

### Stage 4: Deep Learning Model

**Scripts:** `src/models/tft.py`, `src/training/train_tft.py`, `src/training/predict_tft.py`

#### Temporal Fusion Transformer (TFT)
- LSTM encoder with attention mechanism
- Multi-horizon forecasting
- Trained separately for each asset
- Early stopping on validation set

**Hyperparameters:**
- Window length: 90 days
- Hidden size: 128
- LSTM layers: 2
- Dropout: 0.1
- Learning rate: 1e-3

**Outputs:**
- `checkpoints/tft/{asset}/best.ckpt`
- `results/preds/tft_H{1,5,22}.csv`

### Stage 5: Forecast Evaluation

**Scripts:** `src/evaluation/forecast_metrics.py`, `src/evaluation/regime_dm.py`

#### Metrics
- RMSE (Root Mean Squared Error)
- RMSPE (Root Mean Squared Percentage Error)
- QLIKE (Quasi-Likelihood)

#### Statistical Tests
- Diebold-Mariano tests (TFT vs HAR, TFT vs GARCH)
- Newey-West HAC standard errors
- Regime analysis by volatility quartiles

**Outputs:**
- `results/tables/forecast_metrics.csv`
- `results/tables/dm_tests_*.csv`
- `results/figs/error_by_regime_*.png`

### Stage 6: Portfolio Backtesting

**Scripts:** `src/portfolio/weights.py`, `src/portfolio/backtest.py`

#### Weight Computation
- Inverse-volatility weighting
- Volatility targeting (10% annualized)
- Position size limits

#### Backtest Features
- Transaction costs: 10 bps
- No-trade band: 5 percentage points
- Partial rebalancing: 70%
- Max leverage: 2.0
- Max position: 50%

#### Performance Metrics
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Volatility tracking error
- Average turnover

**Outputs:**
- `results/backtests/weights_{model}_H{horizon}.csv`
- `results/backtests/equity_curve_{model}_H{horizon}.csv`
- `results/tables/portfolio_metrics.csv`

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
data:
  tickers: [SPY, QQQ, IWM, EFA, EEM, TLT, GLD]
  start_date: "2010-01-01"
  end_date: "2024-12-31"

tft:
  hidden_size: 128
  num_layers: 2
  batch_size: 64
  learning_rate: 0.001

portfolio:
  target_vol: 0.10
  transaction_cost: 0.0010
  max_leverage: 2.0
```

## Running Individual Scripts

Each module can be run independently:

```bash
# Data preparation
python src/data/load_and_align.py
python src/data/splits.py

# Feature engineering
python src/features/build_features.py
python src/features/scaling.py

# Models
python src/models/har_rv.py
python src/models/garch.py
python src/training/train_tft.py
python src/training/predict_tft.py

# Evaluation
python src/evaluation/forecast_metrics.py
python src/evaluation/regime_dm.py

# Portfolio
python src/portfolio/weights.py
python src/portfolio/backtest.py
```

## Key Features

### Leakage Prevention
- Strict time-based splits with embargo periods
- Features computed only on past data
- Scaler fit only on training set
- Rolling window validation for GARCH

### Realistic Backtesting
- Transaction costs
- Partial rebalancing
- Position limits
- No-trade bands
- Leverage constraints

### Comprehensive Evaluation
- Multiple forecast horizons (1, 5, 22 days)
- Statistical significance tests
- Regime-conditional analysis
- Portfolio performance metrics

## Expected Runtime

On a modern laptop (CPU only):
- Data preparation: ~5 minutes
- Feature engineering: ~2 minutes
- Baseline models: ~10 minutes
- TFT training (7 assets): ~30-60 minutes
- Evaluation: ~2 minutes
- Portfolio backtest: ~1 minute

**Total: ~50-80 minutes**

With GPU acceleration, TFT training can be 5-10x faster.

## Troubleshooting

### Missing Data
If download fails, data files are cached in `data/raw/`. Delete and re-run.

### GARCH Convergence
Some GARCH models may fail to converge. This is normal and handled gracefully.

### TFT Memory
If OOM errors occur, reduce `batch_size` in `configs/config.yaml`.

### Checkpoint Loading
Ensure TFT models are trained before running predictions.

## Citation

This implementation is based on:

1. **HAR-RV:** Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility"
2. **TFT:** Lim et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
3. **Volatility Forecasting:** Andersen et al. (2006) "Volatility and Correlation Forecasting"


