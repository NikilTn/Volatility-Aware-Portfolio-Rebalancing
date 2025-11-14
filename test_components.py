"""
Quick test script to verify all components are working.
Runs minimal tests on each module without full pipeline execution.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))


def test_data_loading():
    """Test data loading module."""
    print("\n[1/7] Testing data loading...")
    try:
        from src.data.load_and_align import compute_realized_volatility
        
        # Create dummy price data
        dates = pd.date_range('2020-01-01', periods=100)
        prices = pd.DataFrame(
            {'SPY': np.random.randn(100).cumsum() + 100},
            index=dates
        )
        
        # Test RV computation
        rv = compute_realized_volatility(prices, window=1)
        
        assert len(rv) == len(prices), "RV length mismatch"
        assert not rv.isna().all(), "All RV values are NaN"
        
        print("  ✓ Data loading module works")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering module."""
    print("\n[2/7] Testing feature engineering...")
    try:
        from src.features.build_features import compute_rolling_features, compute_vix_features
        
        # Create dummy data
        dates = pd.date_range('2020-01-01', periods=300)
        df = pd.DataFrame({
            'SPY_return': np.random.randn(300) * 0.01,
            'SPY_rv': np.random.rand(300) * 0.1,
            'VIX': np.random.rand(300) * 20 + 15
        }, index=dates)
        
        # Test rolling features
        features = compute_rolling_features(df, 'SPY', windows=[5, 22])
        assert len(features) == len(df), "Feature length mismatch"
        
        # Test VIX features
        vix_features = compute_vix_features(df)
        assert 'vix_level' in vix_features.columns, "VIX level not computed"
        
        print("  ✓ Feature engineering module works")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_har_model():
    """Test HAR-RV model."""
    print("\n[3/7] Testing HAR-RV model...")
    try:
        from src.models.har_rv import HARModel
        
        # Create dummy data
        dates = pd.date_range('2020-01-01', periods=200)
        df = pd.DataFrame({
            'SPY_log_rv_lag1': np.random.randn(200) * 0.1,
            'SPY_log_rv_lag5': np.random.randn(200) * 0.1,
            'SPY_log_rv_lag22': np.random.randn(200) * 0.1,
            'SPY_target_h1': np.random.randn(200) * 0.1
        }, index=dates)
        
        # Train model
        model = HARModel(horizon=1)
        model.fit(df, 'SPY')
        
        # Predict
        pred = model.predict(df, 'SPY')
        
        assert len(pred) == len(df), "Prediction length mismatch"
        assert not pred.isna().all(), "All predictions are NaN"
        
        print("  ✓ HAR-RV model works")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_tft_model():
    """Test TFT model architecture."""
    print("\n[4/7] Testing TFT model...")
    try:
        import torch
        from src.models.tft import create_model
        
        # Create model
        model = create_model(
            num_features=10,
            model_type='tft',
            hidden_size=32,
            num_layers=1,
            num_horizons=3
        )
        
        # Test forward pass
        x = torch.randn(16, 30, 10)  # batch, seq, features
        output = model(x)
        
        assert output.shape == (16, 3), f"Output shape mismatch: {output.shape}"
        assert not torch.isnan(output).any(), "Model output contains NaN"
        
        print("  ✓ TFT model works")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n[5/7] Testing evaluation metrics...")
    try:
        from src.evaluation.forecast_metrics import rmse, qlike, rmspe
        
        # Create dummy predictions
        y_true = np.random.randn(100) * 0.1
        y_pred = y_true + np.random.randn(100) * 0.05
        
        # Compute metrics
        rmse_val = rmse(y_true, y_pred)
        qlike_val = qlike(y_true, y_pred)
        rmspe_val = rmspe(y_true, y_pred)
        
        assert not np.isnan(rmse_val), "RMSE is NaN"
        assert not np.isnan(qlike_val), "QLIKE is NaN"
        assert not np.isnan(rmspe_val), "RMSPE is NaN"
        assert rmse_val > 0, "RMSE should be positive"
        
        print("  ✓ Evaluation metrics work")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_portfolio_weights():
    """Test portfolio weight computation."""
    print("\n[6/7] Testing portfolio weights...")
    try:
        from src.portfolio.weights import compute_inverse_vol_weights
        
        # Create dummy forecasts (log volatility)
        forecasts = pd.Series({
            'SPY': -2.5,
            'QQQ': -2.3,
            'TLT': -2.7
        })
        
        # Compute weights
        weights = compute_inverse_vol_weights(forecasts)
        
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights don't sum to 1"
        assert (weights >= 0).all(), "Negative weights"
        assert len(weights) == len(forecasts), "Weight length mismatch"
        
        print("  ✓ Portfolio weights module works")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_backtesting():
    """Test backtesting module."""
    print("\n[7/7] Testing backtesting...")
    try:
        from src.portfolio.backtest import PortfolioBacktest
        
        # Create dummy data
        dates = pd.date_range('2020-01-01', periods=100)
        returns_df = pd.DataFrame({
            'SPY': np.random.randn(100) * 0.01,
            'QQQ': np.random.randn(100) * 0.01,
            'TLT': np.random.randn(100) * 0.005
        }, index=dates)
        
        weights_df = pd.DataFrame({
            'SPY': [0.4] * 100,
            'QQQ': [0.4] * 100,
            'TLT': [0.2] * 100
        }, index=dates)
        
        # Run backtest
        backtester = PortfolioBacktest(
            returns_df=returns_df,
            weights_df=weights_df,
            transaction_cost=0.001,
            max_leverage=2.0
        )
        
        results = backtester.run(initial_value=1.0)
        
        assert len(results) == len(dates), "Results length mismatch"
        assert 'cum_return' in results.columns, "Missing cumulative return"
        assert results['cum_return'].iloc[0] > 0, "Invalid starting value"
        
        print("  ✓ Backtesting module works")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all component tests."""
    print("="*70)
    print("TESTING PROJECT COMPONENTS")
    print("="*70)
    
    results = []
    
    results.append(("Data Loading", test_data_loading()))
    results.append(("Feature Engineering", test_feature_engineering()))
    results.append(("HAR-RV Model", test_har_model()))
    results.append(("TFT Model", test_tft_model()))
    results.append(("Evaluation Metrics", test_evaluation_metrics()))
    results.append(("Portfolio Weights", test_portfolio_weights()))
    results.append(("Backtesting", test_backtesting()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<50} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All components are working correctly!")
        print("Ready to run the full pipeline with: python run_pipeline.py --stage all")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())

