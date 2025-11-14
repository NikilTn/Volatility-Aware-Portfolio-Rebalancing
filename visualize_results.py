"""
Visualization script for project results.
Creates comprehensive plots for analysis and reporting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_forecast_metrics(metrics_path: Path, output_dir: Path):
    """Plot forecast evaluation metrics."""
    df = pd.read_csv(metrics_path)
    
    # RMSE by model and horizon
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(['rmse', 'rmspe', 'qlike']):
        pivot = df.pivot_table(values=metric, index='asset', columns='model', aggfunc='mean')
        
        pivot.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by Asset and Model')
        axes[i].set_xlabel('Asset')
        axes[i].set_ylabel(metric.upper())
        axes[i].legend(title='Model')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'forecast_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved forecast metrics comparison")
    plt.close()
    
    # Metrics by horizon
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, horizon in enumerate([1, 5, 22]):
        df_h = df[df['horizon'] == horizon]
        pivot = df_h.pivot_table(values='rmse', index='asset', columns='model')
        
        pivot.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'RMSE at Horizon {horizon}')
        axes[i].set_xlabel('Asset')
        axes[i].set_ylabel('RMSE')
        axes[i].legend(title='Model')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_by_horizon.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved RMSE by horizon")
    plt.close()


def plot_dm_tests(dm_path: Path, output_dir: Path, comparison: str):
    """Plot Diebold-Mariano test results."""
    df = pd.read_csv(dm_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # DM statistics heatmap
    pivot = df.pivot_table(values='dm_stat', index='asset', columns='horizon')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[0])
    axes[0].set_title(f'DM Statistics: {comparison}')
    axes[0].set_xlabel('Horizon')
    axes[0].set_ylabel('Asset')
    
    # P-values heatmap
    pivot = df.pivot_table(values='p_value', index='asset', columns='horizon')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=0.1, ax=axes[1])
    axes[1].set_title(f'P-values: {comparison}')
    axes[1].set_xlabel('Horizon')
    axes[1].set_ylabel('Asset')
    
    plt.tight_layout()
    output_file = output_dir / f'dm_tests_{comparison.replace(" ", "_").lower()}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved DM tests: {comparison}")
    plt.close()


def plot_equity_curves(backtests_dir: Path, output_dir: Path, models: list, horizon: int = 1):
    """Plot equity curves for all models."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for model in models:
        equity_file = backtests_dir / f'equity_curve_{model}_H{horizon}.csv'
        
        if not equity_file.exists():
            continue
        
        df = pd.read_csv(equity_file, index_col=0, parse_dates=True)
        
        # Equity curve
        axes[0].plot(df.index, df['cum_return'], label=model.upper(), linewidth=2)
        
        # Drawdown
        axes[1].plot(df.index, df['drawdown'] * 100, label=model.upper(), linewidth=2)
    
    axes[0].set_title(f'Equity Curves (Horizon {horizon})')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Drawdowns')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'equity_curves_H{horizon}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved equity curves for H{horizon}")
    plt.close()


def plot_portfolio_metrics(metrics_path: Path, output_dir: Path):
    """Plot portfolio performance metrics."""
    df = pd.read_csv(metrics_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = [
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('annualized_return', 'Annualized Return (%)'),
        ('annualized_vol', 'Annualized Volatility (%)'),
        ('max_drawdown', 'Max Drawdown (%)'),
        ('avg_turnover', 'Average Turnover (%)'),
        ('vol_tracking_error', 'Vol Tracking Error')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        
        # Prepare data
        pivot = df.pivot_table(values=metric, index='model', columns='horizon')
        
        # Convert to percentage if needed
        if metric in ['annualized_return', 'annualized_vol', 'max_drawdown', 'avg_turnover']:
            pivot = pivot * 100
        
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel(title)
        ax.legend(title='Horizon', labels=['H1', 'H5', 'H22'])
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0 for drawdown
        if metric == 'max_drawdown':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'portfolio_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved portfolio metrics")
    plt.close()


def plot_volatility_tracking(backtests_dir: Path, output_dir: Path, model: str = 'tft', horizon: int = 1):
    """Plot realized volatility vs target."""
    equity_file = backtests_dir / f'equity_curve_{model}_H{horizon}.csv'
    
    if not equity_file.exists():
        print(f"Warning: {equity_file} not found")
        return
    
    df = pd.read_csv(equity_file, index_col=0, parse_dates=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Realized volatility
    axes[0].plot(df.index, df['realized_vol'] * 100, label='Realized Vol', linewidth=1.5)
    axes[0].axhline(y=10, color='red', linestyle='--', label='Target (10%)', linewidth=2)
    axes[0].fill_between(df.index, 8, 12, alpha=0.2, color='red')
    axes[0].set_title(f'Volatility Tracking: {model.upper()} (H{horizon})')
    axes[0].set_ylabel('Annualized Volatility (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling returns
    rolling_returns = df['return'].rolling(22).sum()
    axes[1].plot(df.index, rolling_returns * 100, label='22-Day Rolling Return', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_title('Rolling Returns (22 days)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'volatility_tracking_{model}_H{horizon}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved volatility tracking for {model} H{horizon}")
    plt.close()


def generate_summary_table(metrics_path: Path, output_dir: Path):
    """Generate summary table of results."""
    df = pd.read_csv(metrics_path)
    
    # Pivot by model and horizon
    summary = df.pivot_table(
        values=['sharpe_ratio', 'annualized_return', 'annualized_vol', 'max_drawdown'],
        index='model',
        columns='horizon',
        aggfunc='mean'
    )
    
    # Format as percentages
    summary['annualized_return'] *= 100
    summary['annualized_vol'] *= 100
    summary['max_drawdown'] *= 100
    
    # Round
    summary = summary.round(2)
    
    # Save to CSV
    output_file = output_dir / 'portfolio_summary.csv'
    summary.to_csv(output_file)
    print(f"✓ Saved portfolio summary table")
    
    # Print to console
    print("\n" + "="*70)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("="*70)
    print(summary)


def main():
    """Generate all visualizations."""
    project_root = Path(__file__).parent
    results_dir = project_root / 'results'
    figs_dir = results_dir / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Forecast metrics
    print("\n1. Forecast metrics...")
    metrics_path = results_dir / 'tables' / 'forecast_metrics.csv'
    if metrics_path.exists():
        plot_forecast_metrics(metrics_path, figs_dir)
    else:
        print("  Warning: forecast_metrics.csv not found")
    
    # DM tests
    print("\n2. Diebold-Mariano tests...")
    for comparison, filename in [
        ('TFT vs HAR', 'dm_tests_tft_vs_har.csv'),
        ('TFT vs GARCH', 'dm_tests_tft_vs_garch.csv')
    ]:
        dm_path = results_dir / 'tables' / filename
        if dm_path.exists():
            plot_dm_tests(dm_path, figs_dir, comparison)
        else:
            print(f"  Warning: {filename} not found")
    
    # Equity curves
    print("\n3. Equity curves...")
    backtests_dir = results_dir / 'backtests'
    models = ['har', 'garch', 'tft']
    for h in [1, 5, 22]:
        plot_equity_curves(backtests_dir, figs_dir, models, horizon=h)
    
    # Portfolio metrics
    print("\n4. Portfolio metrics...")
    portfolio_metrics_path = results_dir / 'tables' / 'portfolio_metrics.csv'
    if portfolio_metrics_path.exists():
        plot_portfolio_metrics(portfolio_metrics_path, figs_dir)
        generate_summary_table(portfolio_metrics_path, results_dir / 'tables')
    else:
        print("  Warning: portfolio_metrics.csv not found")
    
    # Volatility tracking
    print("\n5. Volatility tracking...")
    for model in ['har', 'garch', 'tft']:
        plot_volatility_tracking(backtests_dir, figs_dir, model=model, horizon=1)
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {figs_dir}")


if __name__ == '__main__':
    main()

