"""
Portfolio backtesting with transaction costs and constraints.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PortfolioBacktest:
    """
    Portfolio backtester with realistic constraints.
    """
    
    def __init__(
        self,
        returns_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        transaction_cost: float = 0.0010,  # 10 bps
        rebalance_band: float = 0.05,      # 5 percentage points
        rebalance_factor: float = 0.7,     # Partial rebalance
        max_leverage: float = 2.0,
        max_position: float = 0.50
    ):
        """
        Initialize backtester.
        
        Args:
            returns_df: DataFrame of asset returns
            weights_df: DataFrame of target weights
            transaction_cost: Transaction cost (as fraction, e.g., 0.001 = 10 bps)
            rebalance_band: No-trade band width
            rebalance_factor: Partial rebalance factor (1.0 = full rebalance)
            max_leverage: Maximum total leverage
            max_position: Maximum position size
        """
        self.returns_df = returns_df
        self.weights_df = weights_df
        self.transaction_cost = transaction_cost
        self.rebalance_band = rebalance_band
        self.rebalance_factor = rebalance_factor
        self.max_leverage = max_leverage
        self.max_position = max_position
        
        # Align dates
        common_dates = self.returns_df.index.intersection(self.weights_df.index)
        self.returns_df = self.returns_df.loc[common_dates]
        self.weights_df = self.weights_df.loc[common_dates]
        
        # Results
        self.portfolio_returns = []
        self.realized_vols = []
        self.turnovers = []
        self.actual_weights_history = []
        
    def run(self, initial_value: float = 1.0) -> pd.DataFrame:
        """
        Run backtest.
        
        Args:
            initial_value: Initial portfolio value
            
        Returns:
            DataFrame with backtest results
        """
        dates = self.weights_df.index
        assets = self.weights_df.columns
        
        # Initialize
        portfolio_value = initial_value
        current_weights = pd.Series(0.0, index=assets)  # Start with no positions
        
        print(f"Running backtest for {len(dates)} days...")
        
        for i, date in enumerate(dates):
            # Get target weights for today
            target_weights = self.weights_df.loc[date]
            
            # Skip if target weights are NaN
            if target_weights.isna().any():
                # No rebalance, just drift with returns
                if i > 0:
                    daily_returns = self.returns_df.loc[date]
                    portfolio_return = (current_weights * daily_returns).sum()
                else:
                    portfolio_return = 0.0
                
                self.portfolio_returns.append(portfolio_return)
                self.turnovers.append(0.0)
                self.actual_weights_history.append(current_weights.copy())
                continue
            
            # Apply constraints to target weights
            target_weights = self.apply_constraints(target_weights)
            
            # Decide whether to rebalance (based on drift from target)
            weight_drift = np.abs(current_weights - target_weights)
            needs_rebalance = (weight_drift > self.rebalance_band).any()
            
            if needs_rebalance or i == 0:
                # Compute new weights (partial rebalance)
                new_weights = (
                    current_weights + 
                    self.rebalance_factor * (target_weights - current_weights)
                )
                
                # Compute turnover
                turnover = np.abs(new_weights - current_weights).sum()
                
                # Apply transaction costs
                tc_cost = turnover * self.transaction_cost
                
                # Update current weights
                current_weights = new_weights
            else:
                # No rebalance
                turnover = 0.0
                tc_cost = 0.0
            
            # Compute returns for today
            daily_returns = self.returns_df.loc[date]
            portfolio_return = (current_weights * daily_returns).sum() - tc_cost
            
            # Update weights for drift (for next iteration)
            if i < len(dates) - 1:
                # Weights drift with returns
                position_values = current_weights * (1 + daily_returns)
                total_value = position_values.sum()
                if total_value > 0:
                    current_weights = position_values / total_value
                else:
                    current_weights = pd.Series(0.0, index=assets)
            
            # Store results
            self.portfolio_returns.append(portfolio_return)
            self.turnovers.append(turnover)
            self.actual_weights_history.append(current_weights.copy())
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'return': self.portfolio_returns,
            'turnover': self.turnovers
        }, index=dates)
        
        # Compute realized volatility (rolling)
        results_df['realized_vol'] = results_df['return'].rolling(22).std() * np.sqrt(252)
        
        # Compute cumulative returns
        results_df['cum_return'] = (1 + results_df['return']).cumprod()
        
        # Compute drawdown
        results_df['drawdown'] = (
            results_df['cum_return'] / results_df['cum_return'].cummax() - 1
        )
        
        return results_df
    
    def apply_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Apply portfolio constraints.
        
        Args:
            weights: Target weights
            
        Returns:
            Constrained weights
        """
        # Clip to max position size
        weights = weights.clip(-self.max_position, self.max_position)
        
        # Clip total leverage
        total_leverage = np.abs(weights).sum()
        if total_leverage > self.max_leverage:
            weights = weights * (self.max_leverage / total_leverage)
        
        return weights


def compute_performance_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute portfolio performance metrics.
    
    Args:
        results_df: Backtest results DataFrame
        
    Returns:
        Dictionary of performance metrics
    """
    returns = results_df['return']
    
    # Annualized metrics
    total_return = results_df['cum_return'].iloc[-1] - 1
    n_years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0 risk-free rate)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
    sortino = annualized_return / downside_vol
    
    # Max drawdown
    max_drawdown = results_df['drawdown'].min()
    
    # Average turnover
    avg_turnover = results_df['turnover'].mean()
    
    # Volatility tracking error (vs 10% target)
    target_vol = 0.10
    realized_vol = results_df['realized_vol'].dropna()
    vol_tracking_error = (realized_vol - target_vol).abs().mean() if len(realized_vol) > 0 else np.nan
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'avg_turnover': avg_turnover,
        'vol_tracking_error': vol_tracking_error,
        'n_days': len(returns)
    }
    
    return metrics


def run_backtest_for_model(
    returns_df: pd.DataFrame,
    weights_path: Path,
    model_name: str,
    horizon: int,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run backtest for a single model.
    
    Args:
        returns_df: DataFrame of asset returns
        weights_path: Path to weights file
        model_name: Name of the model
        horizon: Forecast horizon
        output_dir: Directory to save results
        
    Returns:
        Tuple of (results_df, metrics)
    """
    print(f"\nBacktesting {model_name} H{horizon}...")
    
    # Load weights
    weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    
    # Run backtest
    backtester = PortfolioBacktest(
        returns_df=returns_df,
        weights_df=weights_df,
        transaction_cost=0.0010,
        rebalance_band=0.05,
        rebalance_factor=0.7,
        max_leverage=2.0,
        max_position=0.50
    )
    
    results_df = backtester.run()
    
    # Compute metrics
    metrics = compute_performance_metrics(results_df)
    metrics['model'] = model_name
    metrics['horizon'] = horizon
    
    # Print metrics
    print(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"  Annualized Vol: {metrics['annualized_vol']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Avg Turnover: {metrics['avg_turnover']*100:.2f}%")
    
    # Save equity curve
    equity_file = output_dir / f'equity_curve_{model_name}_H{horizon}.csv'
    results_df.to_csv(equity_file)
    print(f"  Saved equity curve to {equity_file}")
    
    return results_df, metrics


def backtest_all_models(
    data_path: Path,
    weights_dir: Path,
    output_dir: Path,
    models: List[str],
    horizons: List[int],
    assets: List[str]
):
    """
    Run backtests for all models and save results.
    
    Args:
        data_path: Path to test data with returns
        weights_dir: Directory with weight files
        output_dir: Directory to save results
        models: List of model names
        horizons: List of forecast horizons
        assets: List of asset tickers
    """
    print("Loading returns data...")
    test_df = pd.read_parquet(data_path)
    
    # Extract returns
    return_cols = [f'{asset}_return' for asset in assets]
    returns_df = test_df[return_cols].copy()
    returns_df.columns = assets
    
    print(f"Returns shape: {returns_df.shape}")
    
    # Run backtests
    all_metrics = []
    
    for model in models:
        for h in horizons:
            weights_file = weights_dir / f'weights_{model}_H{h}.csv'
            
            if not weights_file.exists():
                print(f"\nWarning: {weights_file} not found")
                continue
            
            results_df, metrics = run_backtest_for_model(
                returns_df=returns_df,
                weights_path=weights_file,
                model_name=model,
                horizon=h,
                output_dir=output_dir
            )
            
            all_metrics.append(metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = output_dir.parent / 'tables' / 'portfolio_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n✓ Saved portfolio metrics to {metrics_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(metrics_df[['model', 'horizon', 'sharpe_ratio', 'annualized_return', 'annualized_vol']])


if __name__ == '__main__':
    # Paths
    project_root = Path(__file__).parents[2]
    data_path = project_root / 'data' / 'processed' / 'features_test.parquet'
    weights_dir = project_root / 'results' / 'backtests'
    output_dir = project_root / 'results' / 'backtests'
    
    assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    models = ['har', 'garch', 'tft']
    horizons = [1, 5, 22]
    
    # Run backtests
    backtest_all_models(
        data_path=data_path,
        weights_dir=weights_dir,
        output_dir=output_dir,
        models=models,
        horizons=horizons,
        assets=assets
    )

