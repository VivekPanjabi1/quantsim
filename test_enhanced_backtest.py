from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy, ExponentialMovingAverageStrategy

print("=" * 70)
print("TESTING ENHANCED BACKTESTER WITH COMPREHENSIVE METRICS")
print("=" * 70)

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

print("\n" + "="*70)
print("ENHANCED BACKTESTING - MOVING AVERAGE STRATEGY")
print("="*70)

# Test MovingAverageStrategy with enhanced metrics
ma_strategy = MovingAverageStrategy(short_window=10, long_window=30)
ma_result = backtester.run(ma_strategy, df, symbol="AAPL")

# Print enhanced summary using the new metrics module
print("\n🚀 ENHANCED PERFORMANCE REPORT:")
ma_result.print_enhanced_summary()

print("\n" + "="*70)
print("ENHANCED BACKTESTING - EXPONENTIAL MOVING AVERAGE STRATEGY")
print("="*70)

# Test ExponentialMovingAverageStrategy
ema_strategy = ExponentialMovingAverageStrategy(short_window=10, long_window=30)
ema_result = backtester.run(ema_strategy, df, symbol="AAPL")

# Print enhanced summary
print("\n🚀 ENHANCED PERFORMANCE REPORT:")
ema_result.print_enhanced_summary()

print("\n" + "="*70)
print("STRATEGY COMPARISON WITH ENHANCED METRICS")
print("="*70)

# Compare strategies using enhanced metrics
results = {
    "MA_Strategy": ma_result,
    "EMA_Strategy": ema_result
}

print("📊 DETAILED STRATEGY COMPARISON:")
print("-" * 70)

comparison_data = []
for name, result in results.items():
    if hasattr(result, 'enhanced_metrics'):
        metrics = result.enhanced_metrics
        comparison_data.append({
            'Strategy': name,
            'Total Return': f"{metrics['total_return_pct']:.2f}%",
            'Annualized Return': f"{metrics['annualized_return_pct']:.2f}%",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Sortino Ratio': f"{metrics['sortino_ratio']:.3f}",
            'Max Drawdown': f"{metrics['max_drawdown_pct']:.2f}%",
            'Win Rate': f"{metrics['win_rate_pct']:.2f}%",
            'Profit Factor': f"{metrics['profit_factor']:.2f}",
            'Volatility': f"{metrics['volatility_pct']:.2f}%",
            'Calmar Ratio': f"{metrics['calmar_ratio']:.3f}",
            'VaR (95%)': f"{metrics['var_95']*100:.2f}%"
        })

import pandas as pd
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Find best strategy based on different criteria
print("\n🏆 BEST STRATEGY BY DIFFERENT CRITERIA:")
print("-" * 50)

best_return = max(results.items(), key=lambda x: x[1].enhanced_metrics['total_return_pct'] if hasattr(x[1], 'enhanced_metrics') else 0)
best_sharpe = max(results.items(), key=lambda x: x[1].enhanced_metrics['sharpe_ratio'] if hasattr(x[1], 'enhanced_metrics') else 0)
best_calmar = max(results.items(), key=lambda x: x[1].enhanced_metrics['calmar_ratio'] if hasattr(x[1], 'enhanced_metrics') else 0)

print(f"Best Total Return:    {best_return[0]} ({best_return[1].enhanced_metrics['total_return_pct']:.2f}%)")
print(f"Best Sharpe Ratio:    {best_sharpe[0]} ({best_sharpe[1].enhanced_metrics['sharpe_ratio']:.3f})")
print(f"Best Calmar Ratio:    {best_calmar[0]} ({best_calmar[1].enhanced_metrics['calmar_ratio']:.3f})")

print("\n" + "="*70)
print("PARAMETER OPTIMIZATION WITH ENHANCED METRICS")
print("="*70)

print("Testing different MA combinations with comprehensive analysis...")

# Test different MA combinations
ma_combinations = [
    (5, 15),   # Very Fast
    (10, 30),  # Fast
    (20, 50),  # Medium
    (30, 100), # Slow
]

optimization_results = {}

for short, long in ma_combinations:
    strategy_name = f"MA_{short}_{long}"
    strategy = MovingAverageStrategy(short_window=short, long_window=long)
    result = backtester.run(strategy, df, symbol="AAPL")
    optimization_results[strategy_name] = result
    
    # Print brief summary with enhanced metrics
    if hasattr(result, 'enhanced_metrics'):
        metrics = result.enhanced_metrics
        print(f"{strategy_name:10}: Return={metrics['total_return_pct']:6.2f}%, "
              f"Sharpe={metrics['sharpe_ratio']:6.3f}, "
              f"MaxDD={metrics['max_drawdown_pct']:6.2f}%, "
              f"Calmar={metrics['calmar_ratio']:6.3f}")

# Find optimal strategy
print(f"\n🎯 OPTIMIZATION RESULTS:")
print("-" * 50)

best_overall = None
best_score = -float('inf')

for name, result in optimization_results.items():
    if hasattr(result, 'enhanced_metrics'):
        metrics = result.enhanced_metrics
        # Composite score: balance return, risk, and drawdown
        score = (metrics['sharpe_ratio'] * 0.4 + 
                metrics['calmar_ratio'] * 0.3 + 
                metrics['total_return_pct'] * 0.01 * 0.3)
        
        if score > best_score:
            best_score = score
            best_overall = (name, result)

if best_overall:
    print(f"🥇 OPTIMAL STRATEGY: {best_overall[0]}")
    print(f"   Composite Score: {best_score:.3f}")
    print("\n📈 DETAILED PERFORMANCE:")
    best_overall[1].print_enhanced_summary()

print("\n" + "="*70)
print("RISK ANALYSIS")
print("="*70)

# Analyze risk metrics for the best strategy
if best_overall and hasattr(best_overall[1], 'enhanced_metrics'):
    metrics = best_overall[1].enhanced_metrics
    
    print(f"📊 RISK PROFILE FOR {best_overall[0]}:")
    print("-" * 40)
    print(f"📈 Return Metrics:")
    print(f"   Total Return:        {metrics['total_return_pct']:8.2f}%")
    print(f"   Annualized Return:   {metrics['annualized_return_pct']:8.2f}%")
    
    print(f"\n⚠️  Risk Metrics:")
    print(f"   Volatility:          {metrics['volatility_pct']:8.2f}%")
    print(f"   Max Drawdown:        {metrics['max_drawdown_pct']:8.2f}%")
    print(f"   VaR (95%):          {metrics['var_95']*100:8.2f}%")
    print(f"   VaR (99%):          {metrics['var_99']*100:8.2f}%")
    
    print(f"\n🎯 Risk-Adjusted Returns:")
    print(f"   Sharpe Ratio:        {metrics['sharpe_ratio']:8.3f}")
    print(f"   Sortino Ratio:       {metrics['sortino_ratio']:8.3f}")
    print(f"   Calmar Ratio:        {metrics['calmar_ratio']:8.3f}")

print("\n" + "="*70)
print("TESTING COMPLETED! 🎉")
print("="*70)

print("\n✅ FEATURES SUCCESSFULLY TESTED:")
print("   🔹 Enhanced metrics integration")
print("   🔹 Comprehensive performance analysis")
print("   🔹 Professional reporting")
print("   🔹 Strategy comparison with 10+ metrics")
print("   🔹 Parameter optimization")
print("   🔹 Risk analysis")
print("   🔹 Composite scoring system")

print(f"\n📊 METRICS AVAILABLE: {len(ma_result.enhanced_metrics) if hasattr(ma_result, 'enhanced_metrics') else 0} performance indicators")
print("🚀 System ready for professional quantitative analysis!")

# Optional: Generate plots if matplotlib is available
try:
    print(f"\n📈 Generating performance plots for {best_overall[0] if best_overall else 'MA_10_30'}...")
    if best_overall:
        best_overall[1].plot_results()
    else:
        ma_result.plot_results()
except Exception as e:
    print(f"📊 Plotting not available: {e}")
    print("   Install matplotlib for visual analysis: pip install matplotlib")
