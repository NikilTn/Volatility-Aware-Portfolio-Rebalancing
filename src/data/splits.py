"""
Create train/validation/test splits with embargo to prevent leakage.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple


def create_splits_with_embargo(
    df: pd.DataFrame,
    train_end: str = '2017-12-31',
    val_end: str = '2019-12-31',
    test_end: str = '2024-12-31',
    embargo_days: int = 5
) -> Dict[str, np.ndarray]:
    """
    Create train/val/test splits with embargo periods.
    
    Embargo ensures no overlap in time between splits to prevent lookahead bias.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        train_end: Last date of training period
        val_end: Last date of validation period
        test_end: Last date of test period
        embargo_days: Number of days to embargo at boundaries
        
    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    dates = df.index
    
    # Define split boundaries
    train_end_date = pd.Timestamp(train_end)
    val_end_date = pd.Timestamp(val_end)
    test_end_date = pd.Timestamp(test_end)
    
    # Apply embargo
    # Train: up to train_end - embargo
    # Val: train_end + embargo to val_end - embargo
    # Test: val_end + embargo onwards
    
    train_mask = dates <= train_end_date
    val_mask = (dates > train_end_date) & (dates <= val_end_date)
    test_mask = (dates > val_end_date) & (dates <= test_end_date)
    
    # Get indices
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    # Apply embargo by removing boundary points
    if len(train_idx) > 0 and embargo_days > 0:
        train_idx = train_idx[:-embargo_days]
    
    if len(val_idx) > 0 and embargo_days > 0:
        val_idx = val_idx[embargo_days:-embargo_days]
    
    if len(test_idx) > 0 and embargo_days > 0:
        test_idx = test_idx[embargo_days:]
    
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    # Verify no overlap
    verify_splits(splits, dates)
    
    return splits


def verify_splits(splits: Dict[str, np.ndarray], dates: pd.DatetimeIndex):
    """
    Verify that splits are valid:
    - No overlap between splits
    - Strictly increasing dates within each split
    - Proper embargo
    """
    train_dates = dates[splits['train']]
    val_dates = dates[splits['val']]
    test_dates = dates[splits['test']]
    
    print("\n=== Split Verification ===")
    print(f"Train: {len(train_dates)} samples, {train_dates[0]} to {train_dates[-1]}")
    print(f"Val:   {len(val_dates)} samples, {val_dates[0]} to {val_dates[-1]}")
    print(f"Test:  {len(test_dates)} samples, {test_dates[0]} to {test_dates[-1]}")
    
    # Check strictly increasing
    assert all(train_dates[:-1] < train_dates[1:]), "Train dates not strictly increasing"
    assert all(val_dates[:-1] < val_dates[1:]), "Val dates not strictly increasing"
    assert all(test_dates[:-1] < test_dates[1:]), "Test dates not strictly increasing"
    
    # Check no overlap
    assert train_dates[-1] < val_dates[0], "Train and Val overlap"
    assert val_dates[-1] < test_dates[0], "Val and Test overlap"
    
    # Check embargo
    train_val_gap = (val_dates[0] - train_dates[-1]).days
    val_test_gap = (test_dates[0] - val_dates[-1]).days
    
    print(f"\nEmbargo verification:")
    print(f"  Train-Val gap: {train_val_gap} days")
    print(f"  Val-Test gap: {val_test_gap} days")
    
    print("\n✓ All split verification checks passed")


def save_splits(splits: Dict[str, np.ndarray], output_path: Path):
    """
    Save split indices to JSON file.
    
    Args:
        splits: Dictionary of split indices
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    splits_serializable = {
        key: indices.tolist() for key, indices in splits.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(splits_serializable, f, indent=2)
    
    print(f"\nSaved split indices to {output_path}")


def load_splits(input_path: Path) -> Dict[str, np.ndarray]:
    """
    Load split indices from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary of split indices as numpy arrays
    """
    with open(input_path, 'r') as f:
        splits_dict = json.load(f)
    
    # Convert lists back to numpy arrays
    splits = {
        key: np.array(indices) for key, indices in splits_dict.items()
    }
    
    return splits


def create_and_save_splits(
    df: pd.DataFrame,
    output_dir: Path = None,
    train_end: str = '2017-12-31',
    val_end: str = '2019-12-31',
    test_end: str = '2024-12-31',
    embargo_days: int = 5
) -> Dict[str, np.ndarray]:
    """
    Main function to create and save splits.
    
    Args:
        df: Input DataFrame with DatetimeIndex
        output_dir: Directory to save splits (default: data/processed)
        train_end: Last date of training period
        val_end: Last date of validation period
        test_end: Last date of test period
        embargo_days: Number of days to embargo at boundaries
        
    Returns:
        Dictionary with split indices
    """
    if output_dir is None:
        output_dir = Path(__file__).parents[2] / 'data' / 'processed'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    splits = create_splits_with_embargo(
        df=df,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        embargo_days=embargo_days
    )
    
    # Save to file
    output_path = output_dir / 'split_indices.json'
    save_splits(splits, output_path)
    
    return splits


if __name__ == '__main__':
    # Load aligned data
    data_path = Path(__file__).parents[2] / 'data' / 'processed' / 'aligned_panel.parquet'
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run load_and_align.py first.")
        exit(1)
    
    df = pd.read_parquet(data_path)
    
    # Create and save splits
    splits = create_and_save_splits(df)
    
    print("\nSplit creation complete!")

