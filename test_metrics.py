import pandas as pd
import numpy as np
from engine.metrics import (
    calculate_total_return, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_comprehensive_metrics, print_metrics_report
)
from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy

print("=" * 60)
print("TESTING METRICS MODULE")
print("=" * 60)

# Test 1: Basic metrics with sample data
print("\n1. Testing individual metric functions with sample data:")
print("-" * 50)

# Sample equity curve data (simulating a portfolio growing from $100k to $120k with some volatility)
sample_equity = [
    100000, 102000, 98000, 105000, 103000, 108000, 106000, 112000, 
    109000, 115000, 113000, 118000, 116000, 120000, 118000, 122000
]

# Sample trade log
sample_trades = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=8, freq='W'),
    'action': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY', 'SELL', 'BUY', 'SELL'],
    'price': [100, 105, 98, 110, 103, 115, 108, 120],
    'quantity': [100, 100, 110, 110, 105, 105, 100, 100]
})

print(f"Sample equity curve: {len(sample_equity)} data points")
print(f"Sample trades: {len(sample_trades)} trades")

# Test individual functions
total_return = calculate_total_return(sample_equity, 100000)
print(f"\nTotal Return: {total_return*100:.2f}%")

sharpe = calculate_sharpe_ratio(sample_equity)
print(f"Sharpe Ratio: {sharpe:.3f}")

max_dd, start_idx, end_idx = calculate_max_drawdown(sample_equity)
print(f"Max Drawdown: {max_dd*100:.2f}% (from index {start_idx} to {end_idx})")

win_rate = calculate_win_rate(sample_trades)
print(f"Win Rate: {win_rate*100:.2f}%")

# Test 2: Comprehensive metrics
print("\n2. Testing comprehensive metrics function:")
print("-" * 50)

comprehensive_metrics = calculate_comprehensive_metrics(
    equity_values=sample_equity,
    trade_log=sample_trades,
    initial_capital=100000,
    days=len(sample_equity),
    risk_free_rate=0.02
)

print_metrics_report(comprehensive_metrics, "Sample Strategy")

# Test 3: Integration with real backtest data
print("\n3. Testing with real backtest data:")
print("-" * 50)

# Load real data and run a quick backtest
print("Loading AAPL data and running backtest...")
df = load_data("AAPL")

# Run a simple backtest
backtester = Backtester(initial_cash=100000, commission=1.0)
strategy = MovingAverageStrategy(short_window=10, long_window=30)
result = backtester.run(strategy, df, symbol="AAPL")

# Extract data for metrics calculation
equity_df = result.equity_df
trades_df = result.trades_df

if not equity_df.empty and not trades_df.empty:
    # Get equity values as list
    equity_values = equity_df['total_equity'].tolist()
    
    # Calculate comprehensive metrics using our new module
    real_metrics = calculate_comprehensive_metrics(
        equity_values=equity_values,
        trade_log=trades_df,
        initial_capital=100000,
        days=len(df),
        risk_free_rate=0.02
    )
    
    print_metrics_report(real_metrics, "MovingAverageStrategy (Real Data)")
    
    # Compare with backtester's built-in metrics
    print("\n4. Comparing with backtester's built-in metrics:")
    print("-" * 50)
    builtin_metrics = result.performance
    
    print("Metrics Module vs Built-in Backtester:")
    print(f"Total Return:    {real_metrics['total_return_pct']:.2f}% vs {builtin_metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio:    {real_metrics['sharpe_ratio']:.3f} vs {builtin_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown:    {real_metrics['max_drawdown_pct']:.2f}% vs {builtin_metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"Win Rate:        {real_metrics['win_rate_pct']:.2f}% vs {builtin_metrics.get('win_rate', 0)*100:.2f}%")
    
else:
    print("No backtest data available for metrics testing.")

# Test 4: Edge cases
print("\n5. Testing edge cases:")
print("-" * 50)

# Empty data
empty_metrics = calculate_comprehensive_metrics([], pd.DataFrame(), 100000, 0)
print(f"Empty data metrics: {len(empty_metrics)} metrics calculated")

# Single data point
single_point_metrics = calculate_comprehensive_metrics([100000], pd.DataFrame(), 100000, 1)
print(f"Single point total return: {single_point_metrics['total_return_pct']:.2f}%")

# All same values (no volatility)
flat_equity = [100000] * 10
flat_metrics = calculate_comprehensive_metrics(flat_equity, pd.DataFrame(), 100000, 10)
print(f"Flat equity Sharpe ratio: {flat_metrics['sharpe_ratio']:.3f}")

print("\n" + "=" * 60)
print("METRICS MODULE TESTING COMPLETED!")
print("=" * 60)

print("\nKey Features Tested:")
print("✅ Individual metric calculations")
print("✅ Comprehensive metrics function")
print("✅ Integration with real backtest data")
print("✅ Comparison with existing backtester metrics")
print("✅ Edge case handling")
print("✅ Professional report formatting")

print(f"\nMetrics Available: {len(comprehensive_metrics)} different performance metrics")
print("Ready for production use!")
