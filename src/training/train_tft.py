"""
Training script for Temporal Fusion Transformer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parents[2]))

from src.models.tft import create_model
from src.training.tft_dataset import create_dataloaders, get_feature_columns


class Trainer:
    """Trainer for TFT model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        device: str = 'cpu',
        checkpoint_dir: Path = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Device to use ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.ckpt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.ckpt'
            torch.save(checkpoint, best_path)
            print(f"    Saved best checkpoint (val_loss: {self.best_val_loss:.6f})")
    
    def train(self, num_epochs: int, patience: int = 10):
        """
        Train model with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        print(f"\nTraining on {self.device}...")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")


def train_tft_for_asset(
    asset: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    checkpoint_dir: Path,
    config: Dict
) -> Trainer:
    """
    Train TFT model for a single asset.
    
    Args:
        asset: Asset ticker
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature columns
        checkpoint_dir: Directory to save checkpoints
        config: Training configuration
        
    Returns:
        Trained Trainer object
    """
    print(f"\n{'='*60}")
    print(f"Training TFT for {asset}")
    print(f"{'='*60}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        asset=asset,
        feature_cols=feature_cols,
        window_length=config['window_length'],
        batch_size=config['batch_size']
    )
    
    # Create model
    model = create_model(
        num_features=len(feature_cols),
        model_type=config['model_type'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_horizons=len(config['horizons'])
    )
    
    print(f"\nModel architecture: {config['model_type']}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    asset_checkpoint_dir = checkpoint_dir / asset
    asset_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=device,
        checkpoint_dir=asset_checkpoint_dir
    )
    
    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        patience=config['patience']
    )
    
    # Save training curves
    metrics = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss
    }
    
    metrics_path = asset_checkpoint_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return trainer


def main():
    """Main training function."""
    # Paths
    project_root = Path(__file__).parents[2]
    data_dir = project_root / 'data' / 'processed'
    checkpoint_dir = project_root / 'checkpoints' / 'tft'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(data_dir / 'features_train.parquet')
    val_df = pd.read_parquet(data_dir / 'features_val.parquet')
    test_df = pd.read_parquet(data_dir / 'features_test.parquet')
    
    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"Number of features: {len(feature_cols)}")
    
    # Configuration
    config = {
        'model_type': 'tft',  # or 'lstm'
        'window_length': 90,
        'batch_size': 64,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'patience': 10,
        'horizons': [1, 5, 22]
    }
    
    # Save config
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Assets to train
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    
    # Train models for each asset
    trainers = {}
    for asset in assets:
        trainer = train_tft_for_asset(
            asset=asset,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            checkpoint_dir=checkpoint_dir,
            config=config
        )
        trainers[asset] = trainer
    
    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60)
    
    # Summary
    print("\nValidation Loss Summary:")
    for asset, trainer in trainers.items():
        print(f"  {asset}: {trainer.best_val_loss:.6f}")


if __name__ == '__main__':
    main()

