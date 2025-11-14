"""
Heterogeneous Autoregressive model for Realized Volatility (HAR-RV).

The HAR-RV model forecasts volatility using lagged values at different frequencies:
    log(RV_{t+h}) = β0 + β1*log(RV_t) + β2*log(RV_{t-5:t}) + β3*log(RV_{t-22:t}) + ε

Reference: Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility"
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class HARModel:
    """Heterogeneous Autoregressive model for volatility forecasting."""
    
    def __init__(self, horizon: int = 1):
        """
        Initialize HAR model.
        
        Args:
            horizon: Forecast horizon in days
        """
        self.horizon = horizon
        self.model = LinearRegression()
        self.feature_cols = None
        
    def prepare_features(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Prepare HAR features: daily, weekly, monthly log RV.
        
        Args:
            df: DataFrame with features
            asset: Asset ticker
            
        Returns:
            DataFrame with HAR features
        """
        har_features = pd.DataFrame(index=df.index)
        
        # Log RV lags
        har_features[f'{asset}_log_rv_daily'] = df[f'{asset}_log_rv_lag1']
        har_features[f'{asset}_log_rv_weekly'] = df[f'{asset}_log_rv_lag5']
        har_features[f'{asset}_log_rv_monthly'] = df[f'{asset}_log_rv_lag22']
        
        return har_features
    
    def fit(self, train_df: pd.DataFrame, asset: str):
        """
        Fit HAR model on training data.
        
        Args:
            train_df: Training DataFrame
            asset: Asset ticker
        """
        # Prepare features
        X = self.prepare_features(train_df, asset)
        y = train_df[f'{asset}_target_h{self.horizon}']
        
        # Remove NaN rows
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        self.feature_cols = X.columns.tolist()
        
        # Fit model
        self.model.fit(X_clean, y_clean)
        
        return self
    
    def predict(self, test_df: pd.DataFrame, asset: str) -> pd.Series:
        """
        Generate predictions on test data.
        
        Args:
            test_df: Test DataFrame
            asset: Asset ticker
            
        Returns:
            Series of predictions
        """
        X = self.prepare_features(test_df, asset)
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=test_df.index, name=f'pred_h{self.horizon}')


def train_har_models(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    assets: List[str],
    horizons: List[int] = [1, 5, 22]
) -> Dict[str, Dict[int, HARModel]]:
    """
    Train HAR models for all assets and horizons.
    
    Args:
        train_path: Path to training features
        val_path: Path to validation features
        test_path: Path to test features
        assets: List of asset tickers
        horizons: List of forecast horizons
        
    Returns:
        Dictionary of trained models {asset: {horizon: model}}
    """
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    
    models = {}
    
    for asset in assets:
        print(f"\nTraining HAR models for {asset}...")
        models[asset] = {}
        
        for h in horizons:
            print(f"  Horizon {h}...")
            
            # Train model
            model = HARModel(horizon=h)
            model.fit(train_df, asset)
            
            models[asset][h] = model
            
            # Print coefficients
            coefs = model.model.coef_
            intercept = model.model.intercept_
            print(f"    Intercept: {intercept:.4f}")
            print(f"    Coefficients: {coefs}")
    
    return models


def generate_predictions(
    models: Dict[str, Dict[int, HARModel]],
    test_path: Path,
    output_dir: Path,
    assets: List[str],
    horizons: List[int] = [1, 5, 22]
):
    """
    Generate and save predictions for all assets and horizons.
    
    Args:
        models: Dictionary of trained models
        test_path: Path to test features
        output_dir: Directory to save predictions
        assets: List of asset tickers
        horizons: List of forecast horizons
    """
    print("\nGenerating predictions...")
    test_df = pd.read_parquet(test_path)
    
    for h in horizons:
        print(f"\nHorizon {h}...")
        predictions = []
        
        for asset in assets:
            model = models[asset][h]
            pred = model.predict(test_df, asset)
            pred.name = asset
            predictions.append(pred)
        
        # Combine predictions for all assets
        pred_df = pd.concat(predictions, axis=1)
        
        # Save
        output_path = output_dir / f'har_H{h}.csv'
        pred_df.to_csv(output_path)
        print(f"  Saved to {output_path}")
        print(f"  Shape: {pred_df.shape}")


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parents[2]
    data_dir = project_root / 'data' / 'processed'
    results_dir = project_root / 'results' / 'preds'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / 'features_train.parquet'
    val_path = data_dir / 'features_val.parquet'
    test_path = data_dir / 'features_test.parquet'
    
    # Assets
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    
    # Train models
    models = train_har_models(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        assets=assets,
        horizons=[1, 5, 22]
    )
    
    # Generate predictions
    generate_predictions(
        models=models,
        test_path=test_path,
        output_dir=results_dir,
        assets=assets,
        horizons=[1, 5, 22]
    )
    
    print("\n✓ HAR-RV model training and prediction complete!")

