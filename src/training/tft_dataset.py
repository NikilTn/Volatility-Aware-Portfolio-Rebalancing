"""
Dataset class for Temporal Fusion Transformer.
Creates sliding windows for time series forecasting.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


class VolatilityForecastDataset(Dataset):
    """
    Dataset for volatility forecasting with sliding windows.
    
    Input: [t-L+1, ..., t] (window of length L)
    Output: [y_{t+1}, y_{t+5}, y_{t+22}] (multi-horizon targets)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        asset: str,
        feature_cols: List[str],
        window_length: int = 90,
        horizons: List[int] = [1, 5, 22],
        target_prefix: str = 'target'
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with features and targets
            asset: Asset ticker to forecast
            feature_cols: List of feature column names
            window_length: Length of input window
            horizons: List of forecast horizons
            target_prefix: Prefix for target columns
        """
        self.df = df
        self.asset = asset
        self.feature_cols = feature_cols
        self.window_length = window_length
        self.horizons = horizons
        
        # Target columns for this asset
        self.target_cols = [f'{asset}_{target_prefix}_h{h}' for h in horizons]
        
        # Extract features and targets
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[self.target_cols].values.astype(np.float32)
        
        # Valid indices (where we have full window + targets)
        self.valid_indices = self._compute_valid_indices()
        
    def _compute_valid_indices(self) -> np.ndarray:
        """
        Compute valid starting indices for windows.
        Must have: window_length history + valid targets.
        
        After feature cleaning, we should have no NaNs, but this is a safety check.
        """
        n = len(self.df)
        valid = []
        
        for i in range(self.window_length, n):
            # Check if targets are valid (most important)
            window_targets = self.targets[i]
            
            if np.isnan(window_targets).any():
                continue
            
            # Check if window has NaNs (should be zero after feature cleaning)
            window_features = self.features[i-self.window_length:i]
            nan_ratio = np.isnan(window_features).sum() / window_features.size
            
            # Strict check: no NaNs allowed (feature cleaning should have handled this)
            if nan_ratio == 0:
                valid.append(i)
            elif nan_ratio < 0.1:
                # Fallback: allow small amount of NaNs (will be filled in __getitem__)
                # This shouldn't happen after proper feature cleaning
                valid.append(i)
        
        return np.array(valid)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (features, targets)
            - features: [window_length, num_features]
            - targets: [num_horizons]
        """
        # Get the actual index in the dataframe
        actual_idx = self.valid_indices[idx]
        
        # Extract window
        window_start = actual_idx - self.window_length
        window_end = actual_idx
        
        features = self.features[window_start:window_end].copy()
        targets = self.targets[actual_idx]
        
        # Safety check: After proper feature cleaning, there should be NO NaNs
        # But we keep this as a fallback guardrail
        if np.isnan(features).any():
            # Fill any remaining NaNs in features (forward fill then backward fill)
            for col_idx in range(features.shape[1]):
                col = features[:, col_idx]
                if np.isnan(col).any():
                    # Forward fill
                    mask = np.isnan(col)
                    idx_filled = np.where(~mask, np.arange(len(mask)), 0)
                    np.maximum.accumulate(idx_filled, out=idx_filled)
                    col[mask] = col[idx_filled[mask]]
                    
                    # Backward fill any remaining (if all were NaN at start)
                    if np.isnan(col).any():
                        col = np.nan_to_num(col, nan=0.0)
                    
                    features[:, col_idx] = col
        
        # Final assertion: No NaNs should reach the model
        assert not np.isnan(features).any(), "NaNs in features after filling - this should not happen"
        assert not np.isnan(targets).any(), "NaNs in targets - this should not happen"
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        targets_tensor = torch.from_numpy(targets)
        
        return features_tensor, targets_tensor


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    asset: str,
    feature_cols: List[str],
    window_length: int = 90,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        asset: Asset ticker
        feature_cols: List of feature columns
        window_length: Length of input window
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = VolatilityForecastDataset(
        df=train_df,
        asset=asset,
        feature_cols=feature_cols,
        window_length=window_length
    )
    
    val_dataset = VolatilityForecastDataset(
        df=val_df,
        asset=asset,
        feature_cols=feature_cols,
        window_length=window_length
    )
    
    test_dataset = VolatilityForecastDataset(
        df=test_df,
        asset=asset,
        feature_cols=feature_cols,
        window_length=window_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset sizes for {asset}:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def get_feature_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
    """
    Get feature columns (exclude targets and specified patterns).
    
    Args:
        df: DataFrame
        exclude_patterns: List of patterns to exclude from features
        
    Returns:
        List of feature column names
    """
    if exclude_patterns is None:
        exclude_patterns = ['target', 'price']
    
    feature_cols = []
    
    for col in df.columns:
        # Skip if any exclude pattern is in column name
        if any(pattern in col for pattern in exclude_patterns):
            continue
        feature_cols.append(col)
    
    return feature_cols


if __name__ == '__main__':
    from pathlib import Path
    
    # Paths
    project_root = Path(__file__).parents[2]
    data_dir = project_root / 'data' / 'processed'
    
    # Load data
    train_df = pd.read_parquet(data_dir / 'features_train.parquet')
    val_df = pd.read_parquet(data_dir / 'features_val.parquet')
    test_df = pd.read_parquet(data_dir / 'features_test.parquet')
    
    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"Number of features: {len(feature_cols)}")
    
    # Create dataloaders for one asset
    asset = 'SPY'
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        asset=asset,
        feature_cols=feature_cols,
        window_length=90,
        batch_size=64
    )
    
    # Test loading a batch
    features, targets = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Features: {features.shape}")  # [batch, window, features]
    print(f"  Targets: {targets.shape}")    # [batch, horizons]

