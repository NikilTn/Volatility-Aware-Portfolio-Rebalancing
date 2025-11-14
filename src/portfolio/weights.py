"""
Portfolio weight computation using volatility forecasts.
Implements inverse-volatility weighting and volatility targeting.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_inverse_vol_weights(forecasts: pd.Series) -> pd.Series:
    """
    Compute inverse-volatility weights.
    
    w_i = (1 / σ_i) / Σ(1 / σ_j)
    
    Args:
        forecasts: Series of volatility forecasts (log scale)
        
    Returns:
        Series of portfolio weights
    """
    # Convert from log volatility to volatility
    vols = np.exp(forecasts)
    
    # Compute inverse volatility
    inv_vols = 1.0 / (vols + 1e-8)
    
    # Normalize to sum to 1
    weights = inv_vols / inv_vols.sum()
    
    return weights


def apply_volatility_target(
    weights: pd.Series,
    forecasts: pd.Series,
    target_vol: float = 0.10,
    correlation: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Apply volatility targeting to scale portfolio.
    
    Args:
        weights: Portfolio weights (sum to 1)
        forecasts: Volatility forecasts (log scale)
        target_vol: Target annualized volatility (default 10%)
        correlation: Correlation matrix (if None, assume diagonal)
        
    Returns:
        Scaled weights
    """
    # Convert from log volatility to volatility
    vols = np.exp(forecasts)
    
    if correlation is None:
        # Assume uncorrelated (diagonal covariance)
        portfolio_var = np.sum((weights * vols) ** 2)
    else:
        # Use correlation matrix
        cov_matrix = np.outer(vols, vols) * correlation
        portfolio_var = weights @ cov_matrix @ weights
    
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Scale weights
    if portfolio_vol > 1e-8:
        scale_factor = target_vol / portfolio_vol
    else:
        scale_factor = 1.0
    
    scaled_weights = weights * scale_factor
    
    return scaled_weights


def compute_portfolio_weights(
    forecast_df: pd.DataFrame,
    assets: List[str],
    target_vol: float = 0.10,
    use_vol_target: bool = True,
    max_weight: float = 0.50,
    min_weight: float = 0.0
) -> pd.DataFrame:
    """
    Compute portfolio weights from volatility forecasts.
    
    Args:
        forecast_df: DataFrame with volatility forecasts (log scale)
        assets: List of asset tickers
        target_vol: Target annualized volatility
        use_vol_target: Whether to apply volatility targeting
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset
        
    Returns:
        DataFrame of portfolio weights over time
    """
    weights_list = []
    
    for date in forecast_df.index:
        # Get forecasts for this date
        forecasts = forecast_df.loc[date, assets]
        
        # Skip if any forecasts are NaN
        if forecasts.isna().any():
            weights_list.append(pd.Series(np.nan, index=assets))
            continue
        
        # Compute inverse-vol weights
        weights = compute_inverse_vol_weights(forecasts)
        
        # Apply volatility targeting
        if use_vol_target:
            weights = apply_volatility_target(weights, forecasts, target_vol)
        
        # Apply weight constraints
        weights = weights.clip(min_weight, max_weight)
        
        # Renormalize (after clipping)
        if not use_vol_target:
            weights = weights / weights.sum()
        
        weights_list.append(weights)
    
    # Combine into DataFrame
    weights_df = pd.DataFrame(weights_list, index=forecast_df.index)
    
    return weights_df


def load_forecast_and_compute_weights(
    pred_path: Path,
    assets: List[str],
    horizon: int = 1,
    target_vol: float = 0.10,
    use_vol_target: bool = True
) -> pd.DataFrame:
    """
    Load forecasts and compute portfolio weights.
    
    Args:
        pred_path: Path to prediction CSV
        assets: List of asset tickers
        horizon: Forecast horizon (for labeling)
        target_vol: Target annualized volatility
        use_vol_target: Whether to apply volatility targeting
        
    Returns:
        DataFrame of portfolio weights
    """
    # Load forecasts
    forecast_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    
    # Filter to desired assets
    forecast_df = forecast_df[assets]
    
    # Compute weights
    weights_df = compute_portfolio_weights(
        forecast_df=forecast_df,
        assets=assets,
        target_vol=target_vol,
        use_vol_target=use_vol_target
    )
    
    return weights_df


def save_weights(
    weights_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    horizon: int
):
    """
    Save portfolio weights to file.
    
    Args:
        weights_df: DataFrame of weights
        output_path: Directory to save weights
        model_name: Name of the model
        horizon: Forecast horizon
    """
    output_file = output_path / f'weights_{model_name}_H{horizon}.csv'
    weights_df.to_csv(output_file)
    print(f"Saved weights to {output_file}")


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parents[2]
    preds_dir = project_root / 'results' / 'preds'
    weights_dir = project_root / 'results' / 'backtests'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    models = ['har', 'garch', 'tft']
    horizons = [1, 5, 22]
    
    print("Computing portfolio weights from volatility forecasts...")
    
    for model in models:
        print(f"\n{model}:")
        
        for h in horizons:
            pred_file = preds_dir / f'{model}_H{h}.csv'
            
            if not pred_file.exists():
                print(f"  H{h}: Prediction file not found")
                continue
            
            # Compute weights
            weights_df = load_forecast_and_compute_weights(
                pred_path=pred_file,
                assets=assets,
                horizon=h,
                target_vol=0.10,
                use_vol_target=True
            )
            
            print(f"  H{h}: Computed weights for {len(weights_df)} dates")
            
            # Save
            save_weights(weights_df, weights_dir, model, h)
            
            # Show summary stats
            print(f"    Mean total leverage: {weights_df.sum(axis=1).mean():.2f}")
            print(f"    Max total leverage: {weights_df.sum(axis=1).max():.2f}")
    
    print("\n✓ Portfolio weights computed!")

