"""
Regime analysis and Diebold-Mariano tests.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def compute_forecast_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute forecast errors (squared errors).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Array of squared errors
    """
    return (y_true - y_pred) ** 2


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    use_hac: bool = True
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    H0: Both models have equal forecast accuracy
    H1: Model 1 is more accurate than Model 2
    
    Args:
        errors1: Squared errors from model 1
        errors2: Squared errors from model 2
        h: Forecast horizon (for HAC correction)
        use_hac: Whether to use Newey-West HAC standard errors
        
    Returns:
        Tuple of (DM statistic, p-value)
    """
    # Loss differential
    d = errors1 - errors2
    
    # Remove NaN
    d = d[~np.isnan(d)]
    
    if len(d) == 0:
        return np.nan, np.nan
    
    # Mean difference
    d_mean = np.mean(d)
    
    # Standard error
    if use_hac:
        # Newey-West HAC standard error
        # Number of lags based on horizon
        max_lag = h + 1
        
        # Compute autocovariances
        gamma_0 = np.var(d, ddof=1)
        
        gamma_sum = 0
        for lag in range(1, min(max_lag, len(d))):
            gamma_lag = np.cov(d[:-lag], d[lag:])[0, 1]
            weight = 1 - lag / (max_lag + 1)
            gamma_sum += 2 * weight * gamma_lag
        
        variance = gamma_0 + gamma_sum
        se = np.sqrt(variance / len(d))
    else:
        # Simple standard error
        se = np.std(d, ddof=1) / np.sqrt(len(d))
    
    # DM statistic
    if se > 1e-10:
        dm_stat = d_mean / se
    else:
        dm_stat = 0.0
    
    # p-value (one-sided test: model 1 better than model 2)
    p_value = stats.norm.cdf(dm_stat)
    
    return dm_stat, p_value


def assign_regimes(test_df: pd.DataFrame, asset: str, n_regimes: int = 4) -> pd.Series:
    """
    Assign volatility regimes based on realized volatility quartiles.
    
    Args:
        test_df: Test DataFrame
        asset: Asset ticker
        n_regimes: Number of regimes (default 4 for quartiles)
        
    Returns:
        Series with regime assignments
    """
    rv_col = f'{asset}_rv'
    
    if rv_col not in test_df.columns:
        return None
    
    # Get realized volatility
    rv = test_df[rv_col]
    
    # Assign to quartiles
    regimes = pd.qcut(rv, q=n_regimes, labels=[f'Q{i+1}' for i in range(n_regimes)])
    
    return regimes


def evaluate_by_regime(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    asset: str,
    horizon: int,
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Evaluate forecast errors by volatility regime.
    
    Args:
        test_df: Test DataFrame with targets
        pred_df: DataFrame with predictions
        asset: Asset ticker
        horizon: Forecast horizon
        regimes: Series with regime assignments
        
    Returns:
        DataFrame with errors by regime
    """
    # Get true values and predictions
    target_col = f'{asset}_target_h{horizon}'
    y_true = test_df[target_col]
    y_pred = pred_df[asset]
    
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index).intersection(regimes.index)
    
    if len(common_idx) == 0:
        return None
    
    y_true = y_true.loc[common_idx]
    y_pred = y_pred.loc[common_idx]
    regimes = regimes.loc[common_idx]
    
    # Compute errors
    errors = (y_true - y_pred) ** 2
    
    # Create DataFrame
    df = pd.DataFrame({
        'error': errors,
        'regime': regimes
    })
    
    return df


def dm_test_all_assets(
    test_path: Path,
    preds_dir: Path,
    model1: str,
    model2: str,
    assets: List[str],
    horizons: List[int],
    output_path: Path
):
    """
    Perform Diebold-Mariano tests comparing two models across all assets and horizons.
    
    Args:
        test_path: Path to test data
        preds_dir: Directory with predictions
        model1: Name of first model (baseline)
        model2: Name of second model (to compare against)
        assets: List of asset tickers
        horizons: List of forecast horizons
        output_path: Path to save results
    """
    print(f"\nDiebold-Mariano Test: {model2} vs {model1}")
    print("="*70)
    
    # Load test data
    test_df = pd.read_parquet(test_path)
    
    results = []
    
    for h in horizons:
        print(f"\nHorizon {h}:")
        
        # Load predictions
        pred_file1 = preds_dir / f'{model1}_H{h}.csv'
        pred_file2 = preds_dir / f'{model2}_H{h}.csv'
        
        if not pred_file1.exists() or not pred_file2.exists():
            print(f"  Warning: Prediction files not found")
            continue
        
        pred_df1 = pd.read_csv(pred_file1, index_col=0, parse_dates=True)
        pred_df2 = pd.read_csv(pred_file2, index_col=0, parse_dates=True)
        
        for asset in assets:
            # Get true values
            target_col = f'{asset}_target_h{h}'
            if target_col not in test_df.columns:
                continue
            
            y_true = test_df[target_col]
            
            if asset not in pred_df1.columns or asset not in pred_df2.columns:
                continue
            
            y_pred1 = pred_df1[asset]
            y_pred2 = pred_df2[asset]
            
            # Align indices
            common_idx = y_true.index.intersection(y_pred1.index).intersection(y_pred2.index)
            
            if len(common_idx) == 0:
                continue
            
            y_true_aligned = y_true.loc[common_idx].values
            y_pred1_aligned = y_pred1.loc[common_idx].values
            y_pred2_aligned = y_pred2.loc[common_idx].values
            
            # Compute errors
            errors1 = compute_forecast_errors(y_true_aligned, y_pred1_aligned)
            errors2 = compute_forecast_errors(y_true_aligned, y_pred2_aligned)
            
            # DM test
            dm_stat, p_value = diebold_mariano_test(errors1, errors2, h=h, use_hac=True)
            
            # Compute mean squared errors
            mse1 = np.mean(errors1)
            mse2 = np.mean(errors2)
            
            results.append({
                'asset': asset,
                'horizon': h,
                'model1': model1,
                'model2': model2,
                'mse1': mse1,
                'mse2': mse2,
                'dm_stat': dm_stat,
                'p_value': p_value,
                'significant_5pct': p_value < 0.05 if not np.isnan(p_value) else False
            })
            
            # Print result
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
            print(f"  {asset}: DM={dm_stat:6.3f}, p={p_value:.4f} {significance}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved DM test results to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print(f"Summary: {model2} vs {model1}")
    print("="*70)
    print(f"Total tests: {len(results_df)}")
    print(f"Significant at 5%: {results_df['significant_5pct'].sum()}")
    print(f"Average DM statistic: {results_df['dm_stat'].mean():.4f}")
    
    return results_df


def plot_error_by_regime(
    test_path: Path,
    preds_dir: Path,
    models: List[str],
    asset: str,
    horizon: int,
    output_path: Path
):
    """
    Plot forecast errors by volatility regime.
    
    Args:
        test_path: Path to test data
        preds_dir: Directory with predictions
        models: List of model names
        asset: Asset ticker to analyze
        horizon: Forecast horizon
        output_path: Path to save plot
    """
    # Load test data
    test_df = pd.read_parquet(test_path)
    
    # Assign regimes
    regimes = assign_regimes(test_df, asset, n_regimes=4)
    
    if regimes is None:
        print(f"Could not assign regimes for {asset}")
        return
    
    # Collect errors by model
    all_errors = []
    
    for model in models:
        pred_file = preds_dir / f'{model}_H{horizon}.csv'
        
        if not pred_file.exists():
            continue
        
        pred_df = pd.read_csv(pred_file, index_col=0, parse_dates=True)
        
        error_df = evaluate_by_regime(test_df, pred_df, asset, horizon, regimes)
        
        if error_df is not None:
            error_df['model'] = model
            all_errors.append(error_df)
    
    if len(all_errors) == 0:
        print("No data to plot")
        return
    
    # Combine all errors
    combined_df = pd.concat(all_errors, ignore_index=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x='regime', y='error', hue='model')
    plt.title(f'Forecast Errors by Volatility Regime\n{asset}, Horizon={horizon}')
    plt.xlabel('Volatility Regime')
    plt.ylabel('Squared Error')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parents[2]
    test_path = project_root / 'data' / 'processed' / 'features_test.parquet'
    preds_dir = project_root / 'results' / 'preds'
    tables_dir = project_root / 'results' / 'tables'
    figs_dir = project_root / 'results' / 'figs'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    horizons = [1, 5, 22]
    
    # DM tests: TFT vs HAR
    dm_test_all_assets(
        test_path=test_path,
        preds_dir=preds_dir,
        model1='har',
        model2='tft',
        assets=assets,
        horizons=horizons,
        output_path=tables_dir / 'dm_tests_tft_vs_har.csv'
    )
    
    # DM tests: TFT vs GARCH
    dm_test_all_assets(
        test_path=test_path,
        preds_dir=preds_dir,
        model1='garch',
        model2='tft',
        assets=assets,
        horizons=horizons,
        output_path=tables_dir / 'dm_tests_tft_vs_garch.csv'
    )
    
    # Plot errors by regime for SPY
    print("\nGenerating regime plots...")
    for h in horizons:
        plot_error_by_regime(
            test_path=test_path,
            preds_dir=preds_dir,
            models=['har', 'garch', 'tft'],
            asset='SPY',
            horizon=h,
            output_path=figs_dir / f'error_by_regime_SPY_H{h}.png'
        )

