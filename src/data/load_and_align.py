"""
Load and align price data, realized volatility, and VIX data on a common trading calendar.
Ensures no forward-filling of returns and proper data alignment.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from typing import Dict, List


def download_price_data(tickers: List[str], start_date: str, end_date: str, data_dir: Path) -> pd.DataFrame:
    """
    Download daily price data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_dir: Directory to save raw data
        
    Returns:
        DataFrame with adjusted close prices
    """
    print(f"Downloading price data for {len(tickers)} tickers...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # Extract adjusted close prices
    if len(tickers) == 1:
        # Single ticker returns simple DataFrame
        if 'Close' in data.columns:
            prices = data[['Close']].copy()
            prices.columns = [tickers[0]]
        else:
            prices = data.copy()
            if len(prices.columns) > 0:
                prices.columns = [tickers[0]]
    else:
        # Multiple tickers returns MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            # Check if 'Adj Close' or 'Close' exists in first level
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close'].copy()
            elif 'Close' in data.columns.get_level_values(0):
                prices = data['Close'].copy()
            else:
                # Fallback: try to extract from MultiIndex
                prices = data.xs('Close', axis=1, level=0, drop_level=True)
        else:
            # Simple columns (shouldn't happen with multiple tickers but handle it)
            prices = data.copy()
    
    # Save raw data
    prices.to_csv(data_dir / 'prices_raw.csv')
    print(f"Saved raw prices to {data_dir / 'prices_raw.csv'}")
    
    return prices


def compute_realized_volatility(prices: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Compute realized volatility from prices.
    
    For daily RV, we use squared returns as a proxy.
    In practice, you would use high-frequency data from Oxford-Man institute.
    
    Args:
        prices: DataFrame of prices
        window: Window for computing RV (default 1 for daily)
        
    Returns:
        DataFrame of realized volatility
    """
    # Compute log returns
    returns = np.log(prices / prices.shift(1))
    
    # Realized volatility (annualized)
    # RV_t = sum of squared returns over the window * 252
    rv = returns.rolling(window=window).apply(lambda x: np.sum(x**2) * 252, raw=True)
    
    return rv


def download_vix_data(start_date: str, end_date: str, data_dir: Path) -> pd.DataFrame:
    """
    Download VIX data from Yahoo Finance.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_dir: Directory to save raw data
        
    Returns:
        DataFrame with VIX close values
    """
    print("Downloading VIX data...")
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)
    vix = vix_data[['Close']].copy()
    vix.columns = ['VIX']
    
    # Save raw data
    vix.to_csv(data_dir / 'vix_raw.csv')
    print(f"Saved raw VIX to {data_dir / 'vix_raw.csv'}")
    
    return vix


def align_data(prices: pd.DataFrame, rv: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Align all data sources on a common trading calendar.
    
    Rules:
    - Use intersection of dates (only keep dates present in all datasets)
    - Remove non-trading days
    - NEVER forward-fill returns
    - Drop any rows with NaN values
    
    Args:
        prices: Price data
        rv: Realized volatility data
        vix: VIX data
        
    Returns:
        Aligned DataFrame with prices, returns, RV, and VIX
    """
    print("Aligning data on common trading calendar...")
    
    # Compute returns (never forward-fill)
    returns = np.log(prices / prices.shift(1))
    
    # Create multi-asset panel
    aligned_data = {}
    
    for col in prices.columns:
        aligned_data[f'{col}_price'] = prices[col]
        aligned_data[f'{col}_return'] = returns[col]
        aligned_data[f'{col}_rv'] = rv[col]
    
    # Add VIX
    aligned_data['VIX'] = vix['VIX']
    
    # Combine into single DataFrame
    df = pd.DataFrame(aligned_data)
    
    # Keep only dates where ALL data is available (intersection)
    df = df.dropna()
    
    # Ensure dates are strictly increasing
    df = df.sort_index()
    
    print(f"Aligned data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total trading days: {len(df)}")
    
    return df


def load_and_align_all(
    tickers: List[str],
    start_date: str = '2010-01-01',
    end_date: str = '2024-12-31',
    data_dir: Path = None
) -> pd.DataFrame:
    """
    Main function to load and align all data.
    
    Args:
        tickers: List of ticker symbols to download
        start_date: Start date for data
        end_date: End date for data
        data_dir: Directory for raw data (default: data/raw)
        
    Returns:
        Aligned DataFrame ready for feature engineering
    """
    if data_dir is None:
        data_dir = Path(__file__).parents[2] / 'data' / 'raw'
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download or load data
    prices = download_price_data(tickers, start_date, end_date, data_dir)
    
    # Compute realized volatility (in practice, use Oxford-Man data)
    rv = compute_realized_volatility(prices, window=1)
    
    # Download VIX
    vix = download_vix_data(start_date, end_date, data_dir)
    
    # Align all data
    aligned = align_data(prices, rv, vix)
    
    # Save aligned data
    processed_dir = data_dir.parent / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / 'aligned_panel.parquet'
    aligned.to_parquet(output_path)
    print(f"\nSaved aligned data to {output_path}")
    
    return aligned


if __name__ == '__main__':
    # Example usage with common ETFs/indices
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD']
    
    df = load_and_align_all(
        tickers=tickers,
        start_date='2010-01-01',
        end_date='2024-12-31'
    )
    
    print("\nData alignment complete!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")

