"""
Scaling and normalization for features.
Fit on train set only, apply to val/test without refitting.

Enhanced with:
- Feature diagnostics
- Automatic removal of all-NaN and constant columns
- Robust NaN handling
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def compute_feature_diagnostics(df: pd.DataFrame, feature_cols: List[str], output_path: Path = None) -> pd.DataFrame:
    """
    Compute diagnostics for each feature column.
    
    Args:
        df: DataFrame with features
        feature_cols: List of columns to diagnose
        output_path: Optional path to save diagnostics CSV
        
    Returns:
        DataFrame with diagnostic statistics
    """
    diagnostics = []
    
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        data = df[col]
        
        diag = {
            'column': col,
            'nan_count': data.isna().sum(),
            'nan_pct': data.isna().mean() * 100,
            'n_unique': data.nunique(),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        }
        diagnostics.append(diag)
    
    diag_df = pd.DataFrame(diagnostics)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diag_df.to_csv(output_path, index=False)
        print(f"Saved feature diagnostics to {output_path}")
    
    return diag_df


def identify_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify which columns to scale and which to leave as-is.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        Tuple of (columns_to_scale, columns_to_keep_unchanged)
    """
    # Don't scale target columns
    target_cols = [col for col in df.columns if 'target' in col]
    
    # Don't scale binary/categorical features
    categorical_cols = ['day_of_week', 'month', 'quarter', 'is_month_end', 'is_year_end']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Don't scale price columns (we use returns instead)
    price_cols = [col for col in df.columns if 'price' in col]
    
    # Columns to exclude from scaling
    exclude_cols = set(target_cols + categorical_cols + price_cols)
    
    # Columns to scale (all numerical features)
    scale_cols = [col for col in df.columns if col not in exclude_cols]
    
    keep_cols = list(exclude_cols)
    
    return scale_cols, keep_cols


def clean_features_before_scaling(train_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Remove problematic columns before scaling.
    
    Removes:
    - All-NaN columns
    - Constant columns (std == 0 or near-zero)
    - Columns with >50% NaN values
    - Zero-variance columns
    
    Args:
        train_df: Training DataFrame
        feature_cols: List of feature columns
        
    Returns:
        List of clean feature columns
    """
    print("\n" + "="*70)
    print("FEATURE CLEANING")
    print("="*70)
    
    clean_cols = []
    all_nan_cols = []
    constant_cols = []
    high_nan_cols = []
    zero_var_cols = []
    
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        
        data = train_df[col]
        
        # Check for all NaN
        if data.isna().all():
            all_nan_cols.append(col)
            continue
        
        # Check for >50% NaN
        nan_pct = data.isna().mean()
        if nan_pct > 0.5:
            high_nan_cols.append(col)
            continue
        
        # Check for constant or near-zero variance
        std_val = data.std()
        if pd.isna(std_val) or std_val == 0 or np.abs(std_val) < 1e-8:
            if std_val == 0:
                constant_cols.append(col)
            else:
                zero_var_cols.append(col)
            continue
        
        clean_cols.append(col)
    
    # Report
    print(f"Original feature columns: {len(feature_cols)}")
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN columns")
        if len(all_nan_cols) <= 10:
            print(f"    Examples: {all_nan_cols[:10]}")
    if high_nan_cols:
        print(f"  Dropping {len(high_nan_cols)} high-NaN (>50%) columns")
        if len(high_nan_cols) <= 10:
            print(f"    Examples: {high_nan_cols[:10]}")
    if constant_cols:
        print(f"  Dropping {len(constant_cols)} constant (std=0) columns")
        if len(constant_cols) <= 10:
            print(f"    Examples: {constant_cols[:10]}")
    if zero_var_cols:
        print(f"  Dropping {len(zero_var_cols)} near-zero variance columns")
        if len(zero_var_cols) <= 10:
            print(f"    Examples: {zero_var_cols[:10]}")
    print(f"Clean feature columns: {len(clean_cols)}")
    print("="*70 + "\n")
    
    return clean_cols


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """
    Fit StandardScaler on training data.
    
    Args:
        train_df: Training DataFrame
        feature_cols: List of columns to scale
        
    Returns:
        Fitted StandardScaler
    """
    print(f"Fitting scaler on {len(feature_cols)} features...")
    
    # Extract features to scale
    X_train = train_df[feature_cols].values
    
    # Replace inf with nan, then fill with median
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    print(f"  Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str],
    keep_cols: List[str]
) -> pd.DataFrame:
    """
    Apply fitted scaler to data.
    
    Strategy: Start with a copy of df, then replace scaled columns in place.
    This avoids KeyError from trying to reindex with dropped columns.
    
    Args:
        df: Input DataFrame
        scaler: Fitted StandardScaler
        feature_cols: Columns to scale (cleaned list from training)
        keep_cols: Columns to keep unchanged
        
    Returns:
        Scaled DataFrame with same columns as df (minus any dropped during cleaning)
    """
    # Only use feature columns that exist in this dataframe
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_feature_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_feature_cols)
        print(f"  Note: {len(missing)} feature columns from training not in this split (expected for test)")
    
    # Start with a copy of the dataframe
    scaled_df = df.copy()
    
    # Extract and scale features
    X = scaled_df[available_feature_cols].values
    X = np.where(np.isinf(X), np.nan, X)  # Replace inf with nan (shouldn't happen after feature cleaning)
    X_scaled = scaler.transform(X)
    
    # Replace scaled columns in place
    scaled_df[available_feature_cols] = X_scaled
    
    # Keep columns are already in scaled_df (unchanged)
    # No need to add them back explicitly
    
    return scaled_df


def scale_features(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    scaler_path: Path,
    save_diagnostics: bool = True
):
    """
    Main function to scale features across all splits.
    
    Enhanced with:
    - Feature diagnostics
    - Automatic column cleaning
    - NaN verification
    
    Args:
        train_path: Path to training features
        val_path: Path to validation features
        test_path: Path to test features
        scaler_path: Path to save scaler
        save_diagnostics: Whether to save feature diagnostics
    """
    print("Loading feature files...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Identify columns to scale
    feature_cols, keep_cols = identify_feature_columns(train_df)
    
    print(f"\nColumns to scale (before cleaning): {len(feature_cols)}")
    print(f"Columns to keep unchanged: {len(keep_cols)}")
    
    # Compute and save diagnostics
    if save_diagnostics:
        diag_path = train_path.parent.parent / 'tables' / 'feature_diagnostics.csv'
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        compute_feature_diagnostics(train_df, feature_cols, diag_path)
    
    # Clean features before scaling
    clean_feature_cols = clean_features_before_scaling(train_df, feature_cols)
    
    # Fit scaler on clean features
    scaler = fit_scaler(train_df, clean_feature_cols)
    
    # Save scaler and column list
    scaler_data = {
        'scaler': scaler,
        'feature_cols': clean_feature_cols,
        'keep_cols': keep_cols
    }
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_data, f)
    print(f"\nSaved scaler to {scaler_path}")
    
    # Apply to all splits
    print("\nScaling train set...")
    train_scaled = apply_scaler(train_df, scaler, clean_feature_cols, keep_cols)
    
    print("Scaling val set...")
    val_scaled = apply_scaler(val_df, scaler, clean_feature_cols, keep_cols)
    
    print("Scaling test set...")
    test_scaled = apply_scaler(test_df, scaler, clean_feature_cols, keep_cols)
    
    # Save scaled data (overwrite original files)
    print("\nSaving scaled features...")
    train_scaled.to_parquet(train_path.parent / 'features_train_scaled.parquet')
    val_scaled.to_parquet(val_path.parent / 'features_val_scaled.parquet')
    test_scaled.to_parquet(test_path.parent / 'features_test_scaled.parquet')
    
    print("✓ Scaling complete!")
    
    # Verification - Check ALL columns, not just scaled features
    print("\n" + "="*70)
    print("FINAL VERIFICATION")
    print("="*70)
    
    # Available feature cols (may be subset if some don't exist in val/test)
    train_feature_cols = [col for col in clean_feature_cols if col in train_scaled.columns]
    val_feature_cols = [col for col in clean_feature_cols if col in val_scaled.columns]
    test_feature_cols = [col for col in clean_feature_cols if col in test_scaled.columns]
    
    train_nan = train_scaled[train_feature_cols].isna().sum().sum()
    val_nan = val_scaled[val_feature_cols].isna().sum().sum()
    test_nan = test_scaled[test_feature_cols].isna().sum().sum()
    
    print(f"  Train scaled features - NaN count: {train_nan}")
    print(f"  Val scaled features - NaN count: {val_nan}")
    print(f"  Test scaled features - NaN count: {test_nan}")
    
    # Check ALL columns (including keep_cols like targets)
    train_total_nan = train_scaled.isna().sum().sum()
    val_total_nan = val_scaled.isna().sum().sum()
    test_total_nan = test_scaled.isna().sum().sum()
    
    print(f"  Train ALL columns - NaN count: {train_total_nan}")
    print(f"  Val ALL columns - NaN count: {val_total_nan}")
    print(f"  Test ALL columns - NaN count: {test_total_nan}")
    
    # Fail fast if NaNs detected
    errors = []
    
    if train_nan > 0:
        nan_cols = train_scaled[train_feature_cols].columns[train_scaled[train_feature_cols].isna().any()].tolist()
        errors.append(f"Train has {train_nan} NaNs in scaled features: {nan_cols[:10]}")
    
    if val_nan > 0:
        nan_cols = val_scaled[val_feature_cols].columns[val_scaled[val_feature_cols].isna().any()].tolist()
        errors.append(f"Val has {val_nan} NaNs in scaled features: {nan_cols[:10]}")
    
    if test_nan > 0:
        nan_cols = test_scaled[test_feature_cols].columns[test_scaled[test_feature_cols].isna().any()].tolist()
        errors.append(f"Test has {test_nan} NaNs in scaled features: {nan_cols[:10]}")
    
    if errors:
        print("\n❌ ERROR: NaNs detected in scaled features!")
        for error in errors:
            print(f"  {error}")
        print("\nThis should not happen after feature cleaning.")
        print("Please check build_features.py for rows with NaNs not being dropped.")
        print("="*70)
        raise ValueError("NaNs found in scaled features - cannot proceed to model training")
    else:
        print("  ✓ No NaNs in scaled features!")
        print("  ✓ Pipeline is clean and ready for model training")
    
    print("="*70)


def load_scaler(scaler_path: Path):
    """
    Load a saved scaler (with backward compatibility).
    
    Args:
        scaler_path: Path to scaler pickle file
        
    Returns:
        Scaler object or dict with scaler and column info
    """
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    
    # Backward compatibility: if old format (just scaler), return as-is
    if isinstance(scaler_data, StandardScaler):
        return scaler_data
    
    # New format: dict with scaler and columns
    return scaler_data


def get_tft_feature_subset(all_features: List[str], assets: List[str]) -> List[str]:
    """
    Get curated feature subset for TFT to reduce noise.
    
    Selects:
    - Recent RV lags (1, 5, 22)
    - Recent return statistics (1, 5, 22 day windows)
    - VIX features
    - Calendar features
    
    Args:
        all_features: List of all available features
        assets: List of asset tickers
        
    Returns:
        List of selected feature columns
    """
    selected = []
    
    # For each asset, select key features
    for asset in assets:
        # RV lags
        selected.extend([
            f'{asset}_rv_lag1',
            f'{asset}_rv_lag5',
            f'{asset}_rv_lag22',
        ])
        
        # RV rolling means (short-term)
        selected.extend([
            f'{asset}_rv_mean_5d',
            f'{asset}_rv_mean_22d',
            f'{asset}_rv_mean_66d',
        ])
        
        # Return statistics (recent)
        selected.extend([
            f'{asset}_return_mean_1d',
            f'{asset}_return_mean_5d',
            f'{asset}_return_mean_22d',
            f'{asset}_return_std_5d',
            f'{asset}_return_std_22d',
        ])
        
        # Log RV for HAR-style features
        selected.extend([
            f'{asset}_log_rv',
            f'{asset}_log_rv_lag1',
            f'{asset}_log_rv_lag5',
            f'{asset}_log_rv_lag22',
        ])
    
    # VIX features (market-wide)
    vix_features = [col for col in all_features if 'vix' in col.lower()]
    selected.extend(vix_features)
    
    # Calendar features
    calendar_features = ['day_of_week', 'month', 'quarter', 'is_month_end', 'is_year_end']
    selected.extend([f for f in calendar_features if f in all_features])
    
    # Filter to only features that actually exist
    selected = [f for f in selected if f in all_features]
    
    return selected


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parents[2]
    data_dir = project_root / 'data' / 'processed'
    
    train_path = data_dir / 'features_train.parquet'
    val_path = data_dir / 'features_val.parquet'
    test_path = data_dir / 'features_test.parquet'
    scaler_path = data_dir / 'scaler.pkl'
    
    # Check files exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            print(f"Error: {path} not found. Run build_features.py first.")
            exit(1)
    
    # Scale features
    scale_features(train_path, val_path, test_path, scaler_path)

