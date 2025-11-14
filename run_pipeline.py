"""
Master pipeline script to run the entire volatility forecasting project.

Usage:
    python run_pipeline.py --stage all              # Run all stages
    python run_pipeline.py --stage data             # Run only data preparation
    python run_pipeline.py --stage features         # Run only feature engineering
    python run_pipeline.py --stage models           # Run all models
    python run_pipeline.py --stage evaluate         # Run evaluation
    python run_pipeline.py --stage portfolio        # Run portfolio backtest
"""
import argparse
import sys
from pathlib import Path
import yaml
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent))


def load_config(config_path: Path = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_data_preparation():
    """Stage 1: Data loading and splitting."""
    print("\n" + "="*70)
    print("STAGE 1: DATA PREPARATION")
    print("="*70)
    
    from src.data.load_and_align import load_and_align_all
    from src.data.splits import create_and_save_splits
    
    config = load_config()
    
    # Load and align data
    print("\n1.1 Loading and aligning data...")
    df = load_and_align_all(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    # Create splits
    print("\n1.2 Creating train/val/test splits...")
    splits = create_and_save_splits(
        df=df,
        train_end=config['splits']['train_end'],
        val_end=config['splits']['val_end'],
        test_end=config['splits']['test_end'],
        embargo_days=config['splits']['embargo_days']
    )
    
    print("\n✓ Data preparation complete!")


def run_feature_engineering():
    """Stage 2: Feature engineering."""
    print("\n" + "="*70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("="*70)
    
    from src.features.build_features import build_and_save_features
    from src.features.scaling import scale_features
    import json
    import numpy as np
    
    config = load_config()
    project_root = Path(__file__).parent
    
    # Load splits
    splits_path = project_root / 'data' / 'processed' / 'split_indices.json'
    with open(splits_path, 'r') as f:
        split_indices_dict = json.load(f)
    split_indices = {k: np.array(v) for k, v in split_indices_dict.items()}
    
    # Build features
    print("\n2.1 Building features...")
    build_and_save_features(
        data_path=project_root / 'data' / 'processed' / 'aligned_panel.parquet',
        split_indices=split_indices,
        output_dir=project_root / 'data' / 'processed',
        assets=config['data']['tickers']
    )
    
    # Scale features
    print("\n2.2 Scaling features...")
    data_dir = project_root / 'data' / 'processed'
    scale_features(
        train_path=data_dir / 'features_train.parquet',
        val_path=data_dir / 'features_val.parquet',
        test_path=data_dir / 'features_test.parquet',
        scaler_path=data_dir / 'scaler.pkl'
    )
    
    print("\n✓ Feature engineering complete!")


def run_baseline_models():
    """Stage 3: Train baseline models."""
    print("\n" + "="*70)
    print("STAGE 3: BASELINE MODELS")
    print("="*70)
    
    from src.models.har_rv import train_har_models, generate_predictions as har_predict
    from src.models.garch import generate_predictions as garch_predict
    
    config = load_config()
    project_root = Path(__file__).parent
    data_dir = project_root / 'data' / 'processed'
    results_dir = project_root / 'results' / 'preds'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # HAR-RV
    print("\n3.1 Training HAR-RV models...")
    har_models = train_har_models(
        train_path=data_dir / 'features_train.parquet',
        val_path=data_dir / 'features_val.parquet',
        test_path=data_dir / 'features_test.parquet',
        assets=config['data']['tickers'],
        horizons=config['features']['horizons']
    )
    
    har_predict(
        models=har_models,
        test_path=data_dir / 'features_test.parquet',
        output_dir=results_dir,
        assets=config['data']['tickers'],
        horizons=config['features']['horizons']
    )
    
    # GARCH
    print("\n3.2 Training GARCH models...")
    import pandas as pd
    train_df = pd.read_parquet(data_dir / 'features_train.parquet')
    test_df = pd.read_parquet(data_dir / 'features_test.parquet')
    
    garch_predict(
        train_df=train_df,
        test_df=test_df,
        output_dir=results_dir,
        assets=config['data']['tickers'],
        horizons=config['features']['horizons'],
        use_rolling=True
    )
    
    print("\n✓ Baseline models complete!")


def run_tft_model():
    """Stage 4: Train TFT model."""
    print("\n" + "="*70)
    print("STAGE 4: TEMPORAL FUSION TRANSFORMER")
    print("="*70)
    
    from src.training.train_tft import main as train_tft
    from src.training.predict_tft import main as predict_tft
    
    print("\n4.1 Training TFT models...")
    train_tft()
    
    print("\n4.2 Generating TFT predictions...")
    predict_tft()
    
    print("\n✓ TFT model complete!")


def run_evaluation():
    """Stage 5: Evaluate forecasts."""
    print("\n" + "="*70)
    print("STAGE 5: FORECAST EVALUATION")
    print("="*70)
    
    from src.evaluation.forecast_metrics import evaluate_all_models
    from src.evaluation.regime_dm import dm_test_all_assets, plot_error_by_regime
    
    config = load_config()
    project_root = Path(__file__).parent
    
    # Forecast metrics
    print("\n5.1 Computing forecast metrics...")
    evaluate_all_models(
        test_path=project_root / 'data' / 'processed' / 'features_test.parquet',
        preds_dir=project_root / 'results' / 'preds',
        models=config['models'],
        assets=config['data']['tickers'],
        horizons=config['features']['horizons'],
        output_path=project_root / 'results' / 'tables' / 'forecast_metrics.csv'
    )
    
    # Diebold-Mariano tests
    print("\n5.2 Performing Diebold-Mariano tests...")
    dm_test_all_assets(
        test_path=project_root / 'data' / 'processed' / 'features_test.parquet',
        preds_dir=project_root / 'results' / 'preds',
        model1='har',
        model2='tft',
        assets=config['data']['tickers'],
        horizons=config['features']['horizons'],
        output_path=project_root / 'results' / 'tables' / 'dm_tests_tft_vs_har.csv'
    )
    
    dm_test_all_assets(
        test_path=project_root / 'data' / 'processed' / 'features_test.parquet',
        preds_dir=project_root / 'results' / 'preds',
        model1='garch',
        model2='tft',
        assets=config['data']['tickers'],
        horizons=config['features']['horizons'],
        output_path=project_root / 'results' / 'tables' / 'dm_tests_tft_vs_garch.csv'
    )
    
    # Regime plots
    print("\n5.3 Generating regime plots...")
    for h in config['features']['horizons']:
        plot_error_by_regime(
            test_path=project_root / 'data' / 'processed' / 'features_test.parquet',
            preds_dir=project_root / 'results' / 'preds',
            models=config['models'],
            asset='SPY',
            horizon=h,
            output_path=project_root / 'results' / 'figs' / f'error_by_regime_SPY_H{h}.png'
        )
    
    print("\n✓ Evaluation complete!")


def run_portfolio_backtest():
    """Stage 6: Portfolio construction and backtesting."""
    print("\n" + "="*70)
    print("STAGE 6: PORTFOLIO BACKTESTING")
    print("="*70)
    
    from src.portfolio.weights import load_forecast_and_compute_weights, save_weights
    from src.portfolio.backtest import backtest_all_models
    
    config = load_config()
    project_root = Path(__file__).parent
    preds_dir = project_root / 'results' / 'preds'
    weights_dir = project_root / 'results' / 'backtests'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute weights
    print("\n6.1 Computing portfolio weights...")
    for model in config['models']:
        for h in config['features']['horizons']:
            pred_file = preds_dir / f'{model}_H{h}.csv'
            
            if not pred_file.exists():
                print(f"  Warning: {pred_file} not found")
                continue
            
            weights_df = load_forecast_and_compute_weights(
                pred_path=pred_file,
                assets=config['data']['tickers'],
                horizon=h,
                target_vol=config['portfolio']['target_vol'],
                use_vol_target=True
            )
            
            save_weights(weights_df, weights_dir, model, h)
    
    # Run backtests
    print("\n6.2 Running backtests...")
    backtest_all_models(
        data_path=project_root / 'data' / 'processed' / 'features_test.parquet',
        weights_dir=weights_dir,
        output_dir=weights_dir,
        models=config['models'],
        horizons=config['features']['horizons'],
        assets=config['data']['tickers']
    )
    
    print("\n✓ Portfolio backtesting complete!")


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description='Run volatility forecasting pipeline')
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['all', 'data', 'features', 'baselines', 'tft', 'models', 'evaluate', 'portfolio'],
        help='Pipeline stage to run'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VOLATILITY FORECASTING PIPELINE")
    print("="*70)
    
    if args.stage in ['all', 'data']:
        run_data_preparation()
    
    if args.stage in ['all', 'features']:
        run_feature_engineering()
    
    if args.stage in ['all', 'baselines', 'models']:
        run_baseline_models()
    
    if args.stage in ['all', 'tft', 'models']:
        run_tft_model()
    
    if args.stage in ['all', 'evaluate']:
        run_evaluation()
    
    if args.stage in ['all', 'portfolio']:
        run_portfolio_backtest()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Predictions: results/preds/")
    print("  - Metrics: results/tables/")
    print("  - Figures: results/figs/")
    print("  - Backtests: results/backtests/")


if __name__ == '__main__':
    main()

