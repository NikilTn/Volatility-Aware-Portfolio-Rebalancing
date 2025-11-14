"""
Build features for volatility forecasting.
Creates rolling features and targets for horizons 1, 5, and 22 days.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def compute_rolling_features(df: pd.DataFrame, asset: str, windows: List[int]) -> pd.DataFrame:
    """
    Compute rolling statistics for a single asset.
    
    Args:
        df: DataFrame with price, return, and RV data
        asset: Asset ticker symbol
        windows: List of window sizes for rolling computations
        
    Returns:
        DataFrame with rolling features
    """
    features = pd.DataFrame(index=df.index)
    
    return_col = f'{asset}_return'
    rv_col = f'{asset}_rv'
    
    if return_col not in df.columns or rv_col not in df.columns:
        return features
    
    # Rolling return statistics
    for window in windows:
        features[f'{asset}_return_mean_{window}d'] = df[return_col].rolling(window).mean()
        features[f'{asset}_return_std_{window}d'] = df[return_col].rolling(window).std()
        features[f'{asset}_return_min_{window}d'] = df[return_col].rolling(window).min()
        features[f'{asset}_return_max_{window}d'] = df[return_col].rolling(window).max()
    
    # Rolling realized volatility
    for window in windows:
        features[f'{asset}_rv_mean_{window}d'] = df[rv_col].rolling(window).mean()
        features[f'{asset}_rv_std_{window}d'] = df[rv_col].rolling(window).std()
        features[f'{asset}_rv_min_{window}d'] = df[rv_col].rolling(window).min()
        features[f'{asset}_rv_max_{window}d'] = df[rv_col].rolling(window).max()
    
    # Lag features for RV
    for lag in [1, 5, 22]:
        features[f'{asset}_rv_lag{lag}'] = df[rv_col].shift(lag)
    
    # Log RV (used in HAR model)
    features[f'{asset}_log_rv'] = np.log(df[rv_col] + 1e-8)
    for lag in [1, 5, 22]:
        features[f'{asset}_log_rv_lag{lag}'] = features[f'{asset}_log_rv'].shift(lag)
    
    return features


def compute_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIX-based features.
    
    Args:
        df: DataFrame containing VIX data
        
    Returns:
        DataFrame with VIX features
    """
    features = pd.DataFrame(index=df.index)
    
    # VIX level
    features['vix_level'] = df['VIX']
    
    # VIX changes
    features['vix_change_1d'] = df['VIX'].diff(1)
    features['vix_change_5d'] = df['VIX'].diff(5)
    
    # VIX Z-score (rolling)
    for window in [22, 66, 252]:
        vix_mean = df['VIX'].rolling(window).mean()
        vix_std = df['VIX'].rolling(window).std()
        features[f'vix_zscore_{window}d'] = (df['VIX'] - vix_mean) / (vix_std + 1e-8)
    
    # VIX percentile
    features['vix_percentile_252d'] = df['VIX'].rolling(252).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else np.nan
    )
    
    return features


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calendar-based features.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with calendar features
    """
    features = pd.DataFrame(index=df.index)
    
    # Day of week (Monday=0, Friday=4)
    features['day_of_week'] = df.index.dayofweek
    
    # Month
    features['month'] = df.index.month
    
    # Quarter
    features['quarter'] = df.index.quarter
    
    # Month-end flag (last 3 trading days)
    features['is_month_end'] = (df.index.day >= 25).astype(int)
    
    # Year-end flag (December)
    features['is_year_end'] = (df.index.month == 12).astype(int)
    
    return features


def compute_targets(df: pd.DataFrame, asset: str, horizons: List[int] = [1, 5, 22]) -> pd.DataFrame:
    """
    Compute target variables for volatility forecasting.
    
    Target: log(sigma_{t+h}) where sigma = sqrt(RV_{t+h})
    
    Args:
        df: DataFrame with RV data
        asset: Asset ticker symbol
        horizons: List of forecast horizons
        
    Returns:
        DataFrame with target columns
    """
    targets = pd.DataFrame(index=df.index)
    
    rv_col = f'{asset}_rv'
    
    if rv_col not in df.columns:
        return targets
    
    for h in horizons:
        # Future RV at horizon h
        future_rv = df[rv_col].shift(-h)
        
        # Convert to volatility (sqrt of RV)
        future_vol = np.sqrt(future_rv)
        
        # Log transform
        targets[f'{asset}_target_h{h}'] = np.log(future_vol + 1e-8)
    
    return targets


def build_features_for_split(
    df: pd.DataFrame,
    assets: List[str],
    split_indices: np.ndarray,
    windows: List[int] = [1, 5, 22, 66, 132, 252]
) -> pd.DataFrame:
    """
    Build all features for a given split (train/val/test).
    
    Args:
        df: Full aligned DataFrame
        assets: List of asset tickers
        split_indices: Indices for this split
        windows: Window sizes for rolling features
        
    Returns:
        DataFrame with all features and targets (NaN-free)
    """
    # Extract split data
    split_df = df.iloc[split_indices].copy()
    
    all_features = [split_df[[col for col in df.columns if 'return' in col or 'rv' in col or 'price' in col]]]
    
    # Compute features for each asset
    for asset in assets:
        asset_features = compute_rolling_features(df, asset, windows)
        all_features.append(asset_features.iloc[split_indices])
    
    # VIX features
    vix_features = compute_vix_features(df)
    all_features.append(vix_features.iloc[split_indices])
    
    # Calendar features
    calendar_features = compute_calendar_features(df)
    all_features.append(calendar_features.iloc[split_indices])
    
    # Targets
    for asset in assets:
        asset_targets = compute_targets(df, asset, horizons=[1, 5, 22])
        all_features.append(asset_targets.iloc[split_indices])
    
    # Combine all features
    feature_df = pd.concat(all_features, axis=1)
    
    # Remove duplicate columns
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
    
    # Debug: Show sample columns
    print(f"  Total columns after concat: {len(feature_df.columns)}")
    print(f"  Sample columns: {feature_df.columns[:10].tolist()}")
    
    # Identify column types
    target_cols = [col for col in feature_df.columns if 'target' in col]
    price_cols = [col for col in feature_df.columns if 'price' in col]
    
    # Critical columns: BASE returns and RV only (not rolling features!)
    # These are the raw data columns that should never be NaN
    base_return_cols = [col for col in feature_df.columns 
                        if col.endswith('_return') and '_mean' not in col and '_std' not in col 
                        and '_min' not in col and '_max' not in col]
    base_rv_cols = [col for col in feature_df.columns 
                    if col.endswith('_rv') and '_mean' not in col and '_std' not in col 
                    and '_min' not in col and '_max' not in col and '_lag' not in col]
    
    print(f"  Found: {len(target_cols)} targets, {len(base_return_cols)} base returns, {len(base_rv_cols)} base RV")
    
    # Critical columns that MUST be non-NaN for a row to be valid
    # Only the targets and base data columns - NOT rolling features
    critical_cols = target_cols + base_return_cols + base_rv_cols
    
    # Debug: Show what we're checking
    print(f"  Checking {len(target_cols)} targets, {len(base_return_cols)} base returns, {len(base_rv_cols)} base RV columns")
    
    # Debug: Check NaN counts in critical columns
    if len(critical_cols) > 0:
        nan_counts = feature_df[critical_cols].isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if len(cols_with_nans) > 0:
            print(f"  Critical columns with NaNs: {len(cols_with_nans)}")
            print(f"    Examples: {cols_with_nans.head(10).to_dict()}")
    
    # Drop rows with NaNs in CRITICAL columns only (not all features)
    # This avoids dropping all rows due to NaNs in long-window rolling features
    initial_rows = len(feature_df)
    
    if len(critical_cols) > 0:
        feature_df = feature_df.dropna(subset=critical_cols)
    
    final_rows = len(feature_df)
    
    dropped_rows = initial_rows - final_rows
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows with NaNs in critical columns")
    else:
        print(f"  No rows dropped (all critical columns valid)")
    
    # Fill remaining NaNs in non-critical feature columns
    # (e.g., long-window rolling stats that have early NaNs)
    non_critical_cols = [col for col in feature_df.columns if col not in critical_cols and col not in price_cols]
    
    if non_critical_cols:
        # Forward fill then backward fill for any remaining NaNs
        feature_df[non_critical_cols] = feature_df[non_critical_cols].ffill().bfill()
        
        # If still NaNs (shouldn't happen), fill with 0
        remaining_nans = feature_df[non_critical_cols].isna().sum().sum()
        if remaining_nans > 0:
            print(f"  Filled {remaining_nans} remaining NaNs in rolling features with 0")
            feature_df[non_critical_cols] = feature_df[non_critical_cols].fillna(0)
    
    return feature_df


def build_and_save_features(
    data_path: Path,
    split_indices: Dict[str, np.ndarray],
    output_dir: Path,
    assets: List[str]
):
    """
    Main function to build and save features for all splits.
    
    Args:
        data_path: Path to aligned data
        split_indices: Dictionary with train/val/test indices
        output_dir: Directory to save feature files
        assets: List of asset tickers
    """
    print("Loading aligned data...")
    df = pd.read_parquet(data_path)
    
    print(f"\nBuilding features for {len(assets)} assets...")
    
    # Build features for each split
    for split_name, indices in split_indices.items():
        print(f"\nProcessing {split_name} split ({len(indices)} initial samples)...")
        
        feature_df = build_features_for_split(
            df=df,
            assets=assets,
            split_indices=indices,
            windows=[1, 5, 22, 66, 132, 252]
        )
        
        # Note: NaN dropping now happens inside build_features_for_split
        
        # Save to parquet
        output_path = output_dir / f'features_{split_name}.parquet'
        feature_df.to_parquet(output_path)
        print(f"  Saved to {output_path}")
        print(f"  Final shape: {feature_df.shape}")
        print(f"  Total columns: {len(feature_df.columns)}")
        
        # Verify no NaNs remain
        nan_count = feature_df.isna().sum().sum()
        if nan_count > 0:
            print(f"  ⚠ WARNING: {nan_count} NaNs remain in feature file!")
        else:
            print(f"  ✓ No NaNs in features")


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parents[2]
    data_path = project_root / 'data' / 'processed' / 'aligned_panel.parquet'
    splits_path = project_root / 'data' / 'processed' / 'split_indices.json'
    output_dir = project_root / 'data' / 'processed'
    
    # Check if data exists
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run load_and_align.py first.")
        exit(1)
    
    if not splits_path.exists():
        print(f"Error: {splits_path} not found. Run splits.py first.")
        exit(1)
    
    # Load splits
    import json
    with open(splits_path, 'r') as f:
        split_indices_dict = json.load(f)
    
    split_indices = {k: np.array(v) for k, v in split_indices_dict.items()}
    
    # Assets (should match those used in load_and_align.py)
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    
    # Build features
    build_and_save_features(
        data_path=data_path,
        split_indices=split_indices,
        output_dir=output_dir,
        assets=assets
    )
    
    print("\n✓ Feature engineering complete!")

