"""
Forecast evaluation metrics: RMSE, RMSPE, QLIKE.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Args:
        y_true: True values (log volatility)
        y_pred: Predicted values (log volatility)
        
    Returns:
        RMSE
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    mse = np.mean((y_true[mask] - y_pred[mask])**2)
    return np.sqrt(mse)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Percentage Error.
    
    Args:
        y_true: True values (log volatility)
        y_pred: Predicted values (log volatility)
        
    Returns:
        RMSPE (in percentage)
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    # Convert back to volatility for percentage error
    vol_true = np.exp(y_true[mask])
    vol_pred = np.exp(y_pred[mask])
    
    # Avoid division by zero
    valid_mask = vol_true > 1e-8
    if valid_mask.sum() == 0:
        return np.nan
    
    percentage_errors = ((vol_true[valid_mask] - vol_pred[valid_mask]) / vol_true[valid_mask])**2
    return np.sqrt(np.mean(percentage_errors)) * 100


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    QLIKE (Quasi-Likelihood) loss for volatility forecasting.
    
    QLIKE = log(σ²_pred) + σ²_true / σ²_pred
    
    where σ² is the variance (not log volatility).
    
    Args:
        y_true: True values (log volatility)
        y_pred: Predicted values (log volatility)
        
    Returns:
        QLIKE score (lower is better)
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    # Convert log volatility to variance
    # log(σ) -> σ = exp(log(σ)) -> σ² = exp(2*log(σ))
    var_true = np.exp(2 * y_true[mask])
    var_pred = np.exp(2 * y_pred[mask])
    
    # Clip to avoid numerical issues
    var_pred = np.clip(var_pred, 1e-8, None)
    
    # QLIKE formula
    qlike_values = np.log(var_pred) + var_true / var_pred
    return np.mean(qlike_values)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    asset: str,
    horizon: int,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate predictions for a single asset, horizon, and model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        asset: Asset ticker
        horizon: Forecast horizon
        model_name: Name of the model
        
    Returns:
        Dictionary with metric values
    """
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx].values
    y_pred_aligned = y_pred.loc[common_idx].values
    
    # Compute metrics
    metrics = {
        'asset': asset,
        'horizon': horizon,
        'model': model_name,
        'n_samples': len(common_idx),
        'rmse': rmse(y_true_aligned, y_pred_aligned),
        'rmspe': rmspe(y_true_aligned, y_pred_aligned),
        'qlike': qlike(y_true_aligned, y_pred_aligned)
    }
    
    return metrics


def load_predictions(
    pred_path: Path,
    test_df: pd.DataFrame,
    asset: str,
    horizon: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Load predictions and extract true values.
    
    Args:
        pred_path: Path to prediction CSV
        test_df: Test DataFrame with targets
        asset: Asset ticker
        horizon: Forecast horizon
        
    Returns:
        Tuple of (y_true, y_pred)
    """
    # Load predictions
    pred_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    
    if asset not in pred_df.columns:
        return None, None
    
    y_pred = pred_df[asset]
    
    # Extract true values
    target_col = f'{asset}_target_h{horizon}'
    if target_col not in test_df.columns:
        return None, None
    
    y_true = test_df[target_col]
    
    return y_true, y_pred


def evaluate_all_models(
    test_path: Path,
    preds_dir: Path,
    models: List[str],
    assets: List[str],
    horizons: List[int],
    output_path: Path
):
    """
    Evaluate all models on all assets and horizons.
    
    Args:
        test_path: Path to test data with true values
        preds_dir: Directory with prediction files
        models: List of model names
        assets: List of asset tickers
        horizons: List of forecast horizons
        output_path: Path to save results
    """
    print("Loading test data...")
    test_df = pd.read_parquet(test_path)
    
    print("\nEvaluating models...")
    results = []
    
    for model in models:
        print(f"\n{model}:")
        
        for h in horizons:
            pred_file = preds_dir / f'{model}_H{h}.csv'
            
            if not pred_file.exists():
                print(f"  Warning: {pred_file} not found")
                continue
            
            print(f"  Horizon {h}...")
            
            for asset in assets:
                # Load predictions
                y_true, y_pred = load_predictions(pred_file, test_df, asset, h)
                
                if y_true is None or y_pred is None:
                    print(f"    {asset}: Data not found")
                    continue
                
                # Evaluate
                metrics = evaluate_predictions(y_true, y_pred, asset, h, model)
                results.append(metrics)
                
                print(f"    {asset}: RMSE={metrics['rmse']:.4f}, "
                      f"RMSPE={metrics['rmspe']:.2f}%, QLIKE={metrics['qlike']:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved evaluation results to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Average RMSE by Model and Horizon")
    print("="*70)
    summary = results_df.groupby(['model', 'horizon'])['rmse'].mean().unstack()
    print(summary)
    
    print("\n" + "="*70)
    print("SUMMARY: Average QLIKE by Model and Horizon")
    print("="*70)
    summary = results_df.groupby(['model', 'horizon'])['qlike'].mean().unstack()
    print(summary)
    
    return results_df


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parents[2]
    test_path = project_root / 'data' / 'processed' / 'features_test.parquet'
    preds_dir = project_root / 'results' / 'preds'
    tables_dir = project_root / 'results' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = tables_dir / 'forecast_metrics.csv'
    
    # Evaluate all models
    results_df = evaluate_all_models(
        test_path=test_path,
        preds_dir=preds_dir,
        models=['har', 'garch', 'tft'],
        assets=['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD'],
        horizons=[1, 5, 22],
        output_path=output_path
    )

