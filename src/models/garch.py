"""
GARCH(1,1) model for volatility forecasting.

Uses the arch library to fit GARCH models on returns and forecast variance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class GARCHForecaster:
    """GARCH(1,1) model for volatility forecasting."""
    
    def __init__(self, horizon: int = 1):
        """
        Initialize GARCH forecaster.
        
        Args:
            horizon: Forecast horizon in days
        """
        self.horizon = horizon
        self.model = None
        self.model_fit = None
        
    def fit(self, returns: pd.Series):
        """
        Fit GARCH(1,1) model on returns.
        
        Args:
            returns: Series of log returns (percentage form, e.g., 0.01 for 1%)
        """
        # Scale returns to percentage
        returns_pct = returns * 100
        
        # Remove NaN
        returns_clean = returns_pct.dropna()
        
        try:
            # Fit GARCH(1,1) with constant mean
            self.model = arch_model(
                returns_clean,
                vol='Garch',
                p=1,
                q=1,
                mean='Constant',
                dist='normal'
            )
            
            self.model_fit = self.model.fit(disp='off', show_warning=False)
            
        except Exception as e:
            print(f"    Warning: GARCH fit failed - {str(e)}")
            self.model_fit = None
        
        return self
    
    def forecast_variance(self, horizon: int) -> float:
        """
        Forecast variance at given horizon.
        
        Args:
            horizon: Number of days ahead
            
        Returns:
            Forecasted variance (annualized)
        """
        if self.model_fit is None:
            return np.nan
        
        try:
            # Forecast variance
            forecast = self.model_fit.forecast(horizon=horizon)
            
            # Extract variance forecast (last value in the series)
            # variance is in percentage^2, convert to return^2 then annualize
            var_pct = forecast.variance.values[-1, horizon-1]  # percentage^2
            var_return = var_pct / (100**2)  # return^2
            var_annual = var_return * 252  # annualized
            
            return var_annual
            
        except Exception as e:
            print(f"    Warning: Forecast failed - {str(e)}")
            return np.nan
    
    def predict_log_volatility(self, horizon: int) -> float:
        """
        Predict log volatility at horizon.
        
        Args:
            horizon: Number of days ahead
            
        Returns:
            log(sigma) where sigma = sqrt(variance)
        """
        variance = self.forecast_variance(horizon)
        
        if np.isnan(variance) or variance <= 0:
            return np.nan
        
        volatility = np.sqrt(variance)
        log_vol = np.log(volatility)
        
        return log_vol


def train_garch_models(
    train_df: pd.DataFrame,
    assets: List[str],
    horizons: List[int] = [1, 5, 22]
) -> Dict[str, Dict[int, GARCHForecaster]]:
    """
    Train GARCH models for all assets.
    
    Args:
        train_df: Training DataFrame with returns
        assets: List of asset tickers
        horizons: List of forecast horizons (for fitting, models are fit once)
        
    Returns:
        Dictionary of trained models {asset: {horizon: model}}
    """
    print("Training GARCH models...")
    
    models = {}
    
    for asset in assets:
        print(f"\n{asset}...")
        models[asset] = {}
        
        return_col = f'{asset}_return'
        
        if return_col not in train_df.columns:
            print(f"  Warning: {return_col} not found")
            continue
        
        returns = train_df[return_col]
        
        # Fit one model per asset (can forecast multiple horizons)
        forecaster = GARCHForecaster()
        forecaster.fit(returns)
        
        if forecaster.model_fit is not None:
            # Print parameters
            params = forecaster.model_fit.params
            print(f"  omega: {params.get('omega', np.nan):.6f}")
            print(f"  alpha[1]: {params.get('alpha[1]', np.nan):.4f}")
            print(f"  beta[1]: {params.get('beta[1]', np.nan):.4f}")
            
            # Store model for each horizon
            for h in horizons:
                models[asset][h] = forecaster
        else:
            print(f"  Model fit failed")
    
    return models


def generate_rolling_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    asset: str,
    horizon: int,
    refit_frequency: int = 22
) -> pd.Series:
    """
    Generate predictions using rolling window (refit periodically).
    
    Args:
        train_df: Training data
        test_df: Test data
        asset: Asset ticker
        horizon: Forecast horizon
        refit_frequency: How often to refit model (in days)
        
    Returns:
        Series of predictions
    """
    return_col = f'{asset}_return'
    
    # Combine train and test
    all_returns = pd.concat([train_df[return_col], test_df[return_col]])
    
    predictions = []
    test_dates = test_df.index
    
    print(f"  Generating rolling predictions for {asset} H{horizon}...")
    
    for i, date in enumerate(test_dates):
        # Use all data up to (but not including) current date
        historical_returns = all_returns.loc[:date].iloc[:-1]
        
        # Refit model periodically or on first iteration
        if i % refit_frequency == 0 or i == 0:
            forecaster = GARCHForecaster()
            forecaster.fit(historical_returns)
        
        # Forecast
        pred = forecaster.predict_log_volatility(horizon)
        predictions.append(pred)
    
    return pd.Series(predictions, index=test_dates, name=asset)


def generate_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    assets: List[str],
    horizons: List[int] = [1, 5, 22],
    use_rolling: bool = True
):
    """
    Generate and save GARCH predictions.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save predictions
        assets: List of asset tickers
        horizons: List of forecast horizons
        use_rolling: If True, use rolling window with periodic refitting
    """
    print("\nGenerating GARCH predictions...")
    
    for h in horizons:
        print(f"\nHorizon {h}...")
        predictions = []
        
        for asset in assets:
            if use_rolling:
                pred = generate_rolling_predictions(
                    train_df=train_df,
                    test_df=test_df,
                    asset=asset,
                    horizon=h,
                    refit_frequency=22
                )
            else:
                # Simple approach: fit once on train, forecast once per test point
                # (less realistic but faster)
                forecaster = GARCHForecaster()
                return_col = f'{asset}_return'
                forecaster.fit(train_df[return_col])
                
                pred_value = forecaster.predict_log_volatility(h)
                pred = pd.Series([pred_value] * len(test_df), index=test_df.index, name=asset)
            
            predictions.append(pred)
        
        # Combine predictions
        pred_df = pd.concat(predictions, axis=1)
        
        # Save
        output_path = output_dir / f'garch_H{h}.csv'
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
    test_path = data_dir / 'features_test.parquet'
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Assets
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    
    # Generate predictions with rolling window
    generate_predictions(
        train_df=train_df,
        test_df=test_df,
        output_dir=results_dir,
        assets=assets,
        horizons=[1, 5, 22],
        use_rolling=True
    )
    
    print("\n✓ GARCH model prediction complete!")

