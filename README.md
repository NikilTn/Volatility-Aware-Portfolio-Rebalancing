# Volatility-Aware Portfolio Rebalancing

A complete end-to-end pipeline for volatility forecasting and portfolio optimization using deep learning and classical models.

## 🎯 Project Overview

This project implements a production-ready system for:

1. **Volatility Forecasting** at multiple horizons (1, 5, 22 days) using:
   - **Temporal Fusion Transformer (TFT)** - Deep learning model with attention
   - **HAR-RV** - Heterogeneous Autoregressive baseline
   - **GARCH(1,1)** - Classical volatility model

2. **Portfolio Optimization** with:
   - Inverse-volatility weighting
   - Volatility targeting (10% annualized)
   - Transaction cost modeling

3. **Comprehensive Evaluation** including:
   - Forecast accuracy metrics (RMSE, QLIKE, RMSPE)
   - Diebold-Mariano statistical tests
   - Portfolio performance (Sharpe, Sortino, Drawdown)

## 🚀 Quick Start

```bash
# Setup and run complete pipeline
chmod +x quick_start.sh
./quick_start.sh

# Or manually:
pip install -r requirements.txt
python run_pipeline.py --stage all
python visualize_results.py
```

## 📊 Results

After running the pipeline, find results in:
- `results/preds/` - Model predictions
- `results/tables/` - Performance metrics
- `results/figs/` - Visualizations
- `results/backtests/` - Portfolio equity curves

## 📖 Documentation

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for detailed documentation including:
- Complete architecture overview
- Module-by-module explanations
- Configuration options

## 🏗️ Project Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Forecasting models (HAR, GARCH, TFT)
│   ├── training/       # TFT training pipeline
│   ├── evaluation/     # Metrics and statistical tests
│   └── portfolio/      # Portfolio construction and backtesting
├── configs/            # Configuration files
├── run_pipeline.py     # Main execution script
└── visualize_results.py  # Results visualization

```

## 🔑 Key Features

- ✅ **Leakage-proof pipeline** with proper time-based splits and embargo periods
- ✅ **Realistic backtesting** with transaction costs and trading constraints  
- ✅ **Statistical rigor** including Diebold-Mariano tests and regime analysis
- ✅ **Production-ready code** with comprehensive error handling and logging
- ✅ **Flexible configuration** via YAML files
- ✅ **Modular design** - run individual components independently

## 📈 Expected Performance

**Forecast Accuracy (typical RMSE on test set):**
- TFT: 0.15-0.25 (log volatility)
- HAR-RV: 0.18-0.28
- GARCH: 0.20-0.30

**Portfolio Metrics (10% vol target):**
- Sharpe Ratio: 0.8-1.5
- Max Drawdown: -15% to -25%
- Volatility Tracking Error: <2%

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
data:
  tickers: [SPY, QQQ, IWM, EFA, EEM, TLT, GLD]
  
tft:
  hidden_size: 128
  batch_size: 64
  learning_rate: 0.001
  
portfolio:
  target_vol: 0.10
  transaction_cost: 0.0010
```

## 🧪 Running Individual Stages

```bash
python run_pipeline.py --stage data       # Data preparation only
python run_pipeline.py --stage features   # Feature engineering only
python run_pipeline.py --stage models     # Train all models
python run_pipeline.py --stage evaluate   # Evaluation metrics
python run_pipeline.py --stage portfolio  # Portfolio backtest
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scipy
- scikit-learn
- statsmodels, arch (for GARCH)
- matplotlib, seaborn

See `requirements.txt` for complete list.

## 📝 Implementation Details

### Data Pipeline
- Downloads from Yahoo Finance (2010-2024)
- Computes realized volatility proxies
- Strict time-based train/val/test splits (2010-2017 / 2018-2019 / 2020-2024)
- 5-day embargo periods to prevent leakage

### Models
- **TFT**: LSTM encoder + multi-head attention, 90-day windows
- **HAR-RV**: Linear regression on daily/weekly/monthly RV
- **GARCH**: Rolling estimation with periodic refitting

### Portfolio
- Inverse-volatility weighting
- 10% annualized volatility target
- 10 bps transaction costs
- Partial rebalancing with no-trade bands

## 🎓 Academic References

This implementation follows best practices from:
- Corsi (2009) - HAR-RV model
- Lim et al. (2021) - Temporal Fusion Transformers  
- Andersen et al. (2006) - Volatility forecasting methodology

## 📞 Support

For issues or questions:
1. Check [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for detailed documentation
2. Review code comments in relevant modules


## 🔬 Original Implementation Plan

Below is the original detailed implementation specification:



-------------------------------------------
SECTION 1 — ENVIRONMENT & REPOSITORY SETUP
-------------------------------------------


DIRECTORY STRUCTURE:

project/
    data/
        raw/
        processed/
        external/
    src/
        data/
        features/
        models/
        training/
        evaluation/
        portfolio/
    configs/
    checkpoints/
    results/
        preds/
        tables/
        figs/
        backtests/
    notebooks/
    README.md

REQUIREMENTS (requirements.txt):
    pandas
    numpy
    scipy
    statsmodels
    arch
    scikit-learn
    torch
    pytorch-lightning
    pytorch-forecasting (optional)
    matplotlib
    seaborn
    pyyaml
    tqdm

ACCEPTANCE CRITERIA:
- Environment installs without errors.
- Repo opens cleanly in Cursor.

-------------------------------------------
SECTION 2 — DATA & SPLITS (LEAKAGE-PROOF)
-------------------------------------------

GOAL: Load prices, realized volatility, VIX → align → create safe splits.

2.1 RAW DATA COLLECTION
- Download:
  - Realized volatility (Oxford–Man)
  - Daily prices (Yahoo/Kaggle)
  - VIX data
- Save under: data/raw/*.csv

2.2 ALIGNMENT
Implement: src/data/load_and_align.py
Tasks:
- Align all assets on a common trading calendar
- Remove non-trading days
- NEVER forward-fill returns
- Save as: data/processed/aligned_panel.parquet

2.3 SPLITS + EMBARGO
Implement: src/data/splits.py
- Train: 2010–2017
- Val: 2018–2019
- Test: 2020–2024
- Apply 5-day embargo at both boundaries
- Save indices → data/processed/split_indices.json

ACCEPTANCE:
- Strictly increasing dates
- No overlap between split windows
- Embargo verified

-------------------------------------------
SECTION 3 — FEATURE ENGINEERING
-------------------------------------------

GOAL: Create leakage-free rolling features + targets for 1/5/22-day horizons.

3.1 BUILD FEATURES
Implement: src/features/build_features.py

Rolling features:
- Returns rolling mean/std for windows: 1,5,22,66,132,252
- Realized vol windows: rv_lag1, rv_5d, rv_22d, rv_66d
- VIX features: vix_level, vix_zscore
- Calendar: day_of_week, month, month_end_flag

Targets:
- sigma_{t+h} = sqrt(RV_{t+h})
- y_{t,h} = log(sigma_{t+h}), for h = 1, 5, 22

Save files:
    data/processed/features_train.parquet
    data/processed/features_val.parquet
    data/processed/features_test.parquet

3.2 SCALING
Implement: src/features/scaling.py
- Fit StandardScaler on TRAIN only
- Apply to VAL/TEST without refit
- Save scaler → data/processed/scaler.pkl

ACCEPTANCE:
- No NaNs in target columns
- No future data inside feature windows
- Scaler fitted only on TRAIN

-------------------------------------------
SECTION 4 — BASELINE MODELS (HAR-RV, GARCH)
-------------------------------------------
4.1 HAR-RV
Implement: src/models/har_rv.py

For each asset & horizon:
- Compute regressors: logRV_daily, logRV_5d, logRV_22d
- Fit OLS
- Predict on VAL/TEST
- Save: results/preds/har_H*.csv

4.2 GARCH
Implement: src/models/garch.py

For each asset:
- Fit GARCH(1,1) on returns (train window)
- Forecast variance for 1,5,22 days
- Convert to volatility → log scale
- Save: results/preds/garch_H*.csv

ACCEPTANCE:
- Models converge
- Predictions aligned to correct future dates

-------------------------------------------
SECTION 5 — TEMPORAL FUSION TRANSFORMER (TFT)
-------------------------------------------

5.1 DATASET
Implement: src/training/tft_dataset.py
- Sliding windows of length L=90
- Input: [t-89 ... t]
- Output: [y_{t+1}, y_{t+5}, y_{t+22}]
- Shape: [batch, time, features] → [batch, 3]

5.2 MODEL
Implement: src/models/tft.py
OR use pytorch-forecasting's TFT.

Hyperparameters:
- hidden_size=128
- dropout=0.1
- num_lstm_layers=2
- learning_rate=1e-3

5.3 TRAINING LOOP
Implement: src/training/train_tft.py
- Loss: MSE (minimum), optionally QLIKE
- Early stopping on VAL
- Save best checkpoint: checkpoints/tft/best.ckpt
- Save logs: results/logs/tft_metrics.json

5.4 TEST PREDICTIONS
Implement: src/training/predict_tft.py
- Load checkpoint
- Predict on test set
- Save:
    results/preds/tft_H1.csv
    results/preds/tft_H5.csv
    results/preds/tft_H22.csv

ACCEPTANCE:
- No NaNs in predictions
- Output shape matches horizons

-------------------------------------------
SECTION 6 — FORECAST EVALUATION & DM TESTS
-------------------------------------------

6.1 METRICS
Implement: src/evaluation/forecast_metrics.py

Metrics:
- RMSE
- RMSPE
- QLIKE
- Save → results/tables/forecast_metrics.csv

6.2 REGIME ANALYSIS + DIEBOLD–MARIANO
Implement: src/evaluation/regime_dm.py
- Split test set by realized-volatility quartiles
- DM test comparing:
    TFT vs HAR
    TFT vs GARCH
- Use HAC/Newey–West SE
- Save:
    results/tables/dm_tests.csv
    results/figs/error_by_regime.png

ACCEPTANCE:
- p-values computed
- Regime plot matches expectations

-------------------------------------------
SECTION 7 — PORTFOLIO SIZING & BACKTEST
-------------------------------------------

7.1 BUILD WEIGHTS
Implement: src/portfolio/weights.py

Steps:
1. For each date, collect σ̂ predictions for each asset.
2. Compute inverse-vol weights:
       w_i = (1 / σ̂_i) / sum_j (1 / σ̂_j)
3. Estimate forecast covariance Σ̂ (diagonal or simple shrinkage).
4. Volatility targeting:
       w_final = w * (target_vol / forecast_portfolio_vol)

7.2 BACKTEST
Implement: src/portfolio/backtest.py

Features:
- Daily or weekly rebalancing
- 10 bps transaction cost
- No-trade band of 5 percentage points
- Partial rebalance factor 0.7
- Max leverage & position caps

Outputs:
- Daily returns
- Realized vol
- Sharpe, Sortino
- Max drawdown
- Turnover
- Vol-target tracking error

Save:
    results/backtests/equity_curve_{model,H}.csv
    results/tables/portfolio_metrics.csv

ACCEPTANCE:
- Backtest runs from start to finish
- Portfolio vol stays near 10%
- Costs reduce returns reasonably

-------------------------------------------
SECTION 8 — FINAL REPORT PACK
-------------------------------------------

8.1 EXPORT FIGURES
- Learning curves
- Forecast error plots
- Regime comparison
- Equity curve + drawdown
- Volatility-target tracking

8.2 REPRODUCIBILITY
- Save all configs (YAML)
- Save scaler, model checkpoints
- Save predictions
- Save split indices

8.3 OPTIONAL STRETCH GOALS
- Add N-BEATS or Informer
- PCA feature ablation
- TFT attention heatmaps
- Walk-forward retraining


</FILE>