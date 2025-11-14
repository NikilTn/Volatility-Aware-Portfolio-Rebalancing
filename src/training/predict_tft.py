"""
Generate predictions using trained TFT models.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parents[2]))

from src.models.tft import create_model
from src.training.tft_dataset import VolatilityForecastDataset, get_feature_columns


def load_model(checkpoint_path: Path, config: Dict, num_features: int) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        num_features: Number of input features
        
    Returns:
        Loaded model
    """
    # Create model
    model = create_model(
        num_features=num_features,
        model_type=config['model_type'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_horizons=len(config['horizons'])
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_for_asset(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    asset: str,
    feature_cols: List[str],
    window_length: int,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Generate predictions for a single asset.
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        asset: Asset ticker
        feature_cols: List of feature columns
        window_length: Input window length
        device: Device to use
        
    Returns:
        DataFrame with predictions for all horizons
    """
    # Create dataset
    dataset = VolatilityForecastDataset(
        df=test_df,
        asset=asset,
        feature_cols=feature_cols,
        window_length=window_length
    )
    
    # Get predictions
    predictions = []
    indices = []
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i in range(len(dataset)):
            features, _ = dataset[i]
            features = features.unsqueeze(0).to(device)  # Add batch dimension
            
            pred = model(features)
            predictions.append(pred.cpu().numpy()[0])
            indices.append(dataset.valid_indices[i])
    
    # Create DataFrame
    predictions = np.array(predictions)
    pred_df = pd.DataFrame(
        predictions,
        index=test_df.index[indices],
        columns=[f'h{h}' for h in [1, 5, 22]]
    )
    
    return pred_df


def generate_all_predictions(
    checkpoint_dir: Path,
    test_path: Path,
    output_dir: Path,
    assets: List[str]
):
    """
    Generate predictions for all assets and save to files.
    
    Args:
        checkpoint_dir: Directory with trained models
        test_path: Path to test data
        output_dir: Directory to save predictions
        assets: List of asset tickers
    """
    print("Loading configuration...")
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    print("Loading test data...")
    test_df = pd.read_parquet(test_path)
    
    # Get feature columns
    feature_cols = get_feature_columns(test_df)
    num_features = len(feature_cols)
    
    print(f"Number of features: {num_features}")
    print(f"Test samples: {len(test_df)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Generate predictions for each asset
    all_predictions = {h: [] for h in [1, 5, 22]}
    
    for asset in assets:
        print(f"Generating predictions for {asset}...")
        
        # Load model
        checkpoint_path = checkpoint_dir / asset / 'best.ckpt'
        if not checkpoint_path.exists():
            print(f"  Warning: Checkpoint not found at {checkpoint_path}")
            continue
        
        model = load_model(checkpoint_path, config, num_features)
        
        # Generate predictions
        pred_df = predict_for_asset(
            model=model,
            test_df=test_df,
            asset=asset,
            feature_cols=feature_cols,
            window_length=config['window_length'],
            device=device
        )
        
        print(f"  Generated {len(pred_df)} predictions")
        
        # Store predictions by horizon
        for h_idx, h in enumerate([1, 5, 22]):
            pred_series = pred_df[f'h{h}']
            pred_series.name = asset
            all_predictions[h].append(pred_series)
    
    # Save predictions by horizon
    print("\nSaving predictions...")
    for h in [1, 5, 22]:
        # Combine predictions for all assets
        pred_df = pd.concat(all_predictions[h], axis=1)
        
        # Save
        output_path = output_dir / f'tft_H{h}.csv'
        pred_df.to_csv(output_path)
        print(f"  Saved H{h} predictions to {output_path}")
        print(f"    Shape: {pred_df.shape}")


def main():
    """Main prediction function."""
    # Paths
    project_root = Path(__file__).parents[2]
    checkpoint_dir = project_root / 'checkpoints' / 'tft'
    test_path = project_root / 'data' / 'processed' / 'features_test.parquet'
    output_dir = project_root / 'results' / 'preds'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Assets
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    
    # Generate predictions
    generate_all_predictions(
        checkpoint_dir=checkpoint_dir,
        test_path=test_path,
        output_dir=output_dir,
        assets=assets
    )
    
    print("\n✓ TFT predictions generated successfully!")


if __name__ == '__main__':
    main()

