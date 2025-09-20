from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy, ExponentialMovingAverageStrategy

# Load Apple stock data
print("Loading AAPL data...")
df = load_data("AAPL")
print(f"Loaded {len(df)} rows of data")
print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

# Initialize backtester
backtester = Backtester(
    initial_cash=100000.0,  # $100,000 starting capital
    commission=1.0,         # $1 per trade
    max_position_size=0.25  # Max 25% of portfolio per position
)

print("\n" + "="*60)
print("BACKTESTING MOVING AVERAGE STRATEGY")
print("="*60)

# Test MovingAverageStrategy
ma_strategy = MovingAverageStrategy(short_window=10, long_window=30)
ma_result = backtester.run(ma_strategy, df, symbol="AAPL")

# Print detailed results
ma_result.print_summary()

print("\n" + "="*60)
print("BACKTESTING EXPONENTIAL MOVING AVERAGE STRATEGY")
print("="*60)

# Test ExponentialMovingAverageStrategy
ema_strategy = ExponentialMovingAverageStrategy(short_window=10, long_window=30)
ema_result = backtester.run(ema_strategy, df, symbol="AAPL")

# Print detailed results
ema_result.print_summary()

# Compare strategies
print("\n" + "="*60)
print("STRATEGY COMPARISON")
print("="*60)

results = {
    "MA_Strategy": ma_result,
    "EMA_Strategy": ema_result
}

backtester.compare_strategies(results)

# Show trade details for MA strategy
print("\n" + "="*60)
print("TRADE DETAILS - MOVING AVERAGE STRATEGY")
print("="*60)

trades_df = ma_result.trades_df
if len(trades_df) > 0:
    print("First 10 trades:")
    print(trades_df.head(10)[['timestamp', 'action', 'quantity', 'price', 'value']].to_string(index=False))
    
    print(f"\nTotal trades: {len(trades_df)}")
    print(f"Buy trades: {len(trades_df[trades_df['action'] == 'BUY'])}")
    print(f"Sell trades: {len(trades_df[trades_df['action'] == 'SELL'])}")
else:
    print("No trades executed.")

# Show equity curve data
print("\n" + "="*60)
print("EQUITY CURVE SAMPLE")
print("="*60)

equity_df = ma_result.equity_df
if len(equity_df) > 0:
    print("Last 10 equity snapshots:")
    equity_sample = equity_df.tail(10)[['timestamp', 'price', 'cash', 'position_qty', 'total_equity', 'total_return']]
    equity_sample['total_return'] = equity_sample['total_return'] * 100  # Convert to percentage
    print(equity_sample.to_string(index=False))

# Test different parameter combinations
print("\n" + "="*60)
print("PARAMETER OPTIMIZATION TEST")
print("="*60)

print("Testing different MA combinations...")

# Test different MA combinations
ma_combinations = [
    (5, 20),   # Fast
    (10, 30),  # Medium
    (20, 50),  # Slow
]

optimization_results = {}

for short, long in ma_combinations:
    strategy_name = f"MA_{short}_{long}"
    strategy = MovingAverageStrategy(short_window=short, long_window=long)
    result = backtester.run(strategy, df, symbol="AAPL")
    optimization_results[strategy_name] = result
    
    # Print brief summary
    perf = result.performance
    print(f"{strategy_name}: Return = {perf['total_return_pct']:.2f}%, "
          f"Trades = {perf['num_trades']}, "
          f"Sharpe = {perf.get('sharpe_ratio', 0):.3f}")

# Compare all parameter combinations
print("\nParameter Optimization Results:")
backtester.compare_strategies(optimization_results)

# Find best strategy
best_strategy = None
best_return = -float('inf')

for name, result in optimization_results.items():
    total_return = result.performance['total_return_pct']
    if total_return > best_return:
        best_return = total_return
        best_strategy = name

print(f"\nBest performing strategy: {best_strategy} with {best_return:.2f}% return")

print("\n" + "="*60)
print("BACKTEST COMPLETED!")
print("="*60)

# Optional: Plot results (if matplotlib is available)
try:
    print("\nGenerating plots...")
    ma_result.plot_results()
except Exception as e:
    print(f"Could not generate plots: {e}")
    print("Install matplotlib to see visual results: pip install matplotlib")
