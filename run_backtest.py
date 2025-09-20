import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List, Tuple, Dict, Any

from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy, ExponentialMovingAverageStrategy


def create_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory for saving plots.")


def run_single_backtest_parallel(params: Tuple[int, int, str, str]) -> Dict[str, Any]:
    """
    Run a single backtest in parallel process.
    
    Args:
        params: Tuple of (short_window, long_window, strategy_name, symbol)
        
    Returns:
        Dictionary with backtest results and metrics
    """
    short_window, long_window, strategy_name, symbol = params
    
    try:
        # Load data (each process loads its own copy)
        df = load_data(symbol)
        
        # Initialize backtester
        backtester = Backtester(
            initial_cash=100000.0,
            commission=1.0,
            max_position_size=0.25
        )
        
        # Create strategy
        strategy = MovingAverageStrategy(short_window=short_window, long_window=long_window)
        
        # Run backtest
        result = backtester.run(strategy, df, symbol=symbol)
        
        # Extract key metrics
        metrics = {}
        if hasattr(result, 'enhanced_metrics'):
            metrics = result.enhanced_metrics.copy()
        else:
            metrics = result.performance.copy()
        
        # Add strategy info
        metrics['strategy_name'] = strategy_name
        metrics['short_window'] = short_window
        metrics['long_window'] = long_window
        metrics['symbol'] = symbol
        metrics['num_trades'] = len(result.trades_df)
        
        # Store result object for plotting (serialize key data)
        metrics['equity_data'] = result.equity_df.to_dict('records') if not result.equity_df.empty else []
        metrics['trades_data'] = result.trades_df.to_dict('records') if not result.trades_df.empty else []
        metrics['signals_data'] = result.signals.to_dict() if not result.signals.empty else {}
        metrics['price_data'] = result.data.to_dict('index')
        
        return metrics
        
    except Exception as e:
        return {
            'strategy_name': strategy_name,
            'short_window': short_window,
            'long_window': long_window,
            'symbol': symbol,
            'error': str(e),
            'total_return_pct': -999,  # Mark as failed
            'sharpe_ratio': -999,
            'max_drawdown_pct': -999
        }


def run_parallel_backtests(symbol: str, ma_combinations: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
    """
    Run multiple backtests in parallel.
    
    Args:
        symbol: Stock symbol to test
        ma_combinations: List of (short_window, long_window, name) tuples
        
    Returns:
        List of backtest results
    """
    print(f"\n🚀 RUNNING {len(ma_combinations)} BACKTESTS IN PARALLEL...")
    print("=" * 60)
    
    # Prepare parameters for parallel execution
    params_list = [(short, long, name, symbol) for short, long, name in ma_combinations]
    
    results = []
    start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=min(4, len(ma_combinations))) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(run_single_backtest_parallel, params): params 
                           for params in params_list}
        
        # Collect results as they complete
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                strategy_name = result['strategy_name']
                if 'error' not in result:
                    total_return = result.get('total_return_pct', 0)
                    sharpe = result.get('sharpe_ratio', 0)
                    print(f"✅ {strategy_name:20} | Return: {total_return:6.2f}% | Sharpe: {sharpe:6.3f}")
                else:
                    print(f"❌ {strategy_name:20} | Error: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {params[2]:20} | Exception: {str(e)}")
    
    end_time = time.time()
    print(f"\n⏱️  Parallel backtesting completed in {end_time - start_time:.2f} seconds")
    print(f"📊 Successfully completed {len([r for r in results if 'error' not in r])}/{len(results)} backtests")
    
    return results


def create_ranked_performance_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a ranked performance table sorted by Sharpe ratio.
    
    Args:
        results: List of backtest results
        
    Returns:
        DataFrame with ranked performance metrics
    """
    # Filter out failed results
    valid_results = [r for r in results if 'error' not in r and r.get('sharpe_ratio', -999) != -999]
    
    if not valid_results:
        print("❌ No valid results to rank!")
        return pd.DataFrame()
    
    # Create performance table
    table_data = []
    for result in valid_results:
        table_data.append({
            'Rank': 0,  # Will be filled after sorting
            'Strategy': result['strategy_name'],
            'Short MA': result['short_window'],
            'Long MA': result['long_window'],
            'Total Return (%)': result.get('total_return_pct', 0),
            'Annualized Return (%)': result.get('annualized_return_pct', 0),
            'Sharpe Ratio': result.get('sharpe_ratio', 0),
            'Sortino Ratio': result.get('sortino_ratio', 0),
            'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
            'Volatility (%)': result.get('volatility_pct', 0),
            'Win Rate (%)': result.get('win_rate_pct', 0),
            'Profit Factor': result.get('profit_factor', 0),
            'Calmar Ratio': result.get('calmar_ratio', 0),
            'Total Trades': result.get('num_trades', 0),
            'Avg Win ($)': result.get('avg_win', 0),
            'Avg Loss ($)': result.get('avg_loss', 0)
        })
    
    # Create DataFrame and sort by Sharpe ratio (descending)
    df = pd.DataFrame(table_data)
    df = df.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
    
    # Add rank column
    df['Rank'] = range(1, len(df) + 1)
    
    # Reorder columns to put Rank first
    cols = ['Rank'] + [col for col in df.columns if col != 'Rank']
    df = df[cols]
    
    return df


def print_ranked_table(df: pd.DataFrame, top_n: int = 10):
    """
    Print a formatted ranked performance table.
    
    Args:
        df: Performance DataFrame
        top_n: Number of top strategies to display
    """
    if df.empty:
        print("No results to display.")
        return
    
    print(f"\n🏆 TOP {min(top_n, len(df))} STRATEGIES RANKED BY SHARPE RATIO")
    print("=" * 120)
    
    # Select key columns for display
    display_cols = [
        'Rank', 'Strategy', 'Short MA', 'Long MA', 
        'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
        'Win Rate (%)', 'Total Trades'
    ]
    
    display_df = df[display_cols].head(top_n)
    
    # Format the display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(display_df.to_string(index=False, float_format='%.3f'))
    
    print("=" * 120)
    
    # Print summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Best Sharpe Ratio:     {df['Sharpe Ratio'].max():.3f}")
    print(f"   Best Total Return:     {df['Total Return (%)'].max():.2f}%")
    print(f"   Lowest Max Drawdown:   {df['Max Drawdown (%)'].max():.2f}%")  # Max because it's negative
    print(f"   Highest Win Rate:      {df['Win Rate (%)'].max():.2f}%")
    print(f"   Average Sharpe Ratio:  {df['Sharpe Ratio'].mean():.3f}")


def save_results_to_csv(df: pd.DataFrame, filename: str = "results/backtest_results.csv"):
    """Save results to CSV file."""
    create_results_directory()
    df.to_csv(filename, index=False)
    print(f"💾 Results saved to: {filename}")


def create_parallel_comparison_plot(results: List[Dict[str, Any]], symbol: str, top_n: int = 5):
    """
    Create comparison plot for top N strategies from parallel results.
    
    Args:
        results: List of backtest results
        symbol: Stock symbol
        top_n: Number of top strategies to plot
    """
    # Filter and sort results
    valid_results = [r for r in results if 'error' not in r and r.get('sharpe_ratio', -999) != -999]
    valid_results.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
    top_results = valid_results[:top_n]
    
    if not top_results:
        print("No valid results for plotting.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'Top {len(top_results)} Strategies Comparison - {symbol}', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot equity curves
    for i, result in enumerate(top_results):
        if result.get('equity_data'):
            equity_df = pd.DataFrame(result['equity_data'])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], utc=True).dt.tz_convert(None)
            
            color = colors[i % len(colors)]
            strategy_name = result['strategy_name']
            sharpe = result.get('sharpe_ratio', 0)
            
            ax1.plot(equity_df['timestamp'], equity_df['total_equity'], 
                    label=f"{strategy_name} (Sharpe: {sharpe:.3f})", 
                    color=color, linewidth=2)
    
    # Add initial capital line
    initial_cash = 100000
    ax1.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.7, 
               label=f'Initial Capital (${initial_cash:,.0f})')
    
    ax1.set_title('Equity Curves - Top Performers', fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot performance metrics comparison
    metrics_data = []
    strategy_names = []
    
    for result in top_results:
        metrics_data.append([
            result.get('total_return_pct', 0),
            result.get('sharpe_ratio', 0) * 10,  # Scale for visibility
            abs(result.get('max_drawdown_pct', 0))
        ])
        strategy_names.append(result['strategy_name'].replace('_', '\n'))
    
    if metrics_data:
        metrics_array = np.array(metrics_data)
        x = np.arange(len(strategy_names))
        width = 0.25
        
        ax2.bar(x - width, metrics_array[:, 0], width, label='Total Return (%)', color='green', alpha=0.7)
        ax2.bar(x, metrics_array[:, 1], width, label='Sharpe Ratio (×10)', color='blue', alpha=0.7)
        ax2.bar(x + width, metrics_array[:, 2], width, label='Max Drawdown (%)', color='red', alpha=0.7)
        
        ax2.set_title('Performance Metrics - Top Performers', fontweight='bold')
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_xlabel('Strategy', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_names, fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    comparison_filename = f"results/Top_{len(top_results)}_Strategies_Comparison_{symbol}.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved top strategies comparison plot: {comparison_filename}")


def plot_equity_curve(result, strategy_name, symbol, save_path):
    """
    Create and save a professional equity curve plot.
    
    Args:
        result: BacktestResult object
        strategy_name: Name of the strategy
        symbol: Stock symbol
        save_path: Path to save the plot
    """
    if result.equity_df.empty:
        print(f"No equity data available for {strategy_name}")
        return
    
    # Set up the plot style
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'{strategy_name} - {symbol} Backtest Results', fontsize=16, fontweight='bold')
    
    # Prepare data
    equity_df = result.equity_df.copy()
    # Handle timezone-aware timestamps
    if not equity_df.empty:
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], utc=True).dt.tz_convert(None)
    price_data = result.data
    signals = result.signals
    
    # Plot 1: Price and Trading Signals
    ax1.plot(price_data.index, price_data['Close'], 
             label=f'{symbol} Price', color='black', linewidth=1.5, alpha=0.8)
    
    # Plot buy signals
    buy_signals = signals[signals == 1]
    if len(buy_signals) > 0:
        buy_prices = price_data.loc[buy_signals.index, 'Close']
        ax1.scatter(buy_signals.index, buy_prices, 
                   color='green', marker='^', s=80, label='Buy Signal', 
                   zorder=5, alpha=0.8)
    
    # Plot sell signals
    sell_signals = signals[signals == -1]
    if len(sell_signals) > 0:
        sell_prices = price_data.loc[sell_signals.index, 'Close']
        ax1.scatter(sell_signals.index, sell_prices, 
                   color='red', marker='v', s=80, label='Sell Signal', 
                   zorder=5, alpha=0.8)
    
    ax1.set_title(f'{symbol} Price Chart with Trading Signals', fontweight='bold')
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Plot 2: Equity Curve
    initial_cash = result.performance['initial_cash']
    ax2.plot(equity_df['timestamp'], equity_df['total_equity'], 
             label='Portfolio Value', color='blue', linewidth=2.5)
    ax2.axhline(y=initial_cash, color='gray', linestyle='--', 
               alpha=0.7, label=f'Initial Capital (${initial_cash:,.0f})')
    
    # Add performance metrics as text
    final_equity = equity_df['total_equity'].iloc[-1]
    total_return = result.performance.get('total_return_pct', 0)
    sharpe_ratio = result.performance.get('sharpe_ratio', 0)
    max_drawdown = result.performance.get('max_drawdown_pct', 0)
    
    # Add performance text box
    performance_text = (
        f'Final Value: ${final_equity:,.0f}\n'
        f'Total Return: {total_return:.2f}%\n'
        f'Sharpe Ratio: {sharpe_ratio:.3f}\n'
        f'Max Drawdown: {max_drawdown:.2f}%'
    )
    
    ax2.text(0.02, 0.98, performance_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.8), fontsize=10)
    
    ax2.set_title('Portfolio Equity Curve', fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Format y-axis for currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Drawdown Chart
    equity_series = equity_df.set_index('timestamp')['total_equity']
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    
    ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    ax3.plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax3.set_title('Drawdown Analysis', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close to free memory
    
    print(f"✅ Saved equity curve plot: {save_path}")


def run_single_backtest(symbol, strategy, strategy_name):
    """
    Run a single backtest and save the results.
    
    Args:
        symbol: Stock symbol to test
        strategy: Strategy instance
        strategy_name: Name for saving files
    """
    print(f"\n{'='*60}")
    print(f"RUNNING BACKTEST: {strategy_name}")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading {symbol} data...")
    df = load_data(symbol)
    print(f"Loaded {len(df)} rows of data from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    # Initialize backtester
    backtester = Backtester(
        initial_cash=100000.0,
        commission=1.0,
        max_position_size=0.25
    )
    
    # Run backtest
    result = backtester.run(strategy, df, symbol=symbol)
    
    # Print enhanced summary
    if hasattr(result, 'print_enhanced_summary'):
        result.print_enhanced_summary()
    else:
        result.print_summary()
    
    # Create and save plot
    create_results_directory()
    plot_filename = f"results/{strategy_name}_{symbol}.png"
    plot_equity_curve(result, strategy_name, symbol, plot_filename)
    
    return result


def run_parameter_optimization(symbol):
    """
    Run parameter optimization and save results for each combination.
    
    Args:
        symbol: Stock symbol to test
    """
    print(f"\n{'='*60}")
    print(f"PARAMETER OPTIMIZATION: {symbol}")
    print(f"{'='*60}")
    
    # Load data once
    df = load_data(symbol)
    backtester = Backtester(initial_cash=100000.0, commission=1.0, max_position_size=0.25)
    
    # Test different MA combinations
    ma_combinations = [
        (5, 15, "Fast"),
        (10, 30, "Medium"),
        (20, 50, "Slow"),
        (30, 100, "Very_Slow")
    ]
    
    results = {}
    
    for short, long, speed in ma_combinations:
        strategy_name = f"MA_{short}_{long}_{speed}"
        strategy = MovingAverageStrategy(short_window=short, long_window=long)
        
        print(f"\nTesting {strategy_name}...")
        result = backtester.run(strategy, df, symbol=symbol)
        results[strategy_name] = result
        
        # Save individual plot
        create_results_directory()
        plot_filename = f"results/{strategy_name}_{symbol}.png"
        plot_equity_curve(result, strategy_name, symbol, plot_filename)
        
        # Print brief summary
        if hasattr(result, 'enhanced_metrics'):
            metrics = result.enhanced_metrics
            print(f"   Return: {metrics['total_return_pct']:6.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:6.3f} | "
                  f"MaxDD: {metrics['max_drawdown_pct']:6.2f}%")
    
    # Create comparison plot
    create_comparison_plot(results, symbol)
    
    return results


def create_comparison_plot(results, symbol):
    """
    Create a comparison plot of all strategies.
    
    Args:
        results: Dictionary of strategy results
        symbol: Stock symbol
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'Strategy Comparison - {symbol}', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Plot equity curves
    for i, (strategy_name, result) in enumerate(results.items()):
        if not result.equity_df.empty:
            equity_df = result.equity_df.copy()
            # Handle timezone-aware timestamps
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], utc=True).dt.tz_convert(None)
            
            color = colors[i % len(colors)]
            ax1.plot(equity_df['timestamp'], equity_df['total_equity'], 
                    label=strategy_name, color=color, linewidth=2)
    
    # Add initial capital line
    initial_cash = 100000
    ax1.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.7, 
               label=f'Initial Capital (${initial_cash:,.0f})')
    
    ax1.set_title('Equity Curves Comparison', fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot performance metrics comparison
    metrics_data = []
    strategy_names = []
    
    for strategy_name, result in results.items():
        if hasattr(result, 'enhanced_metrics'):
            metrics = result.enhanced_metrics
            metrics_data.append([
                metrics['total_return_pct'],
                metrics['sharpe_ratio'],
                abs(metrics['max_drawdown_pct'])  # Use absolute value for better visualization
            ])
            strategy_names.append(strategy_name.replace('_', '\n'))
    
    if metrics_data:
        metrics_array = np.array(metrics_data)
        x = np.arange(len(strategy_names))
        width = 0.25
        
        ax2.bar(x - width, metrics_array[:, 0], width, label='Total Return (%)', color='green', alpha=0.7)
        ax2.bar(x, metrics_array[:, 1] * 10, width, label='Sharpe Ratio (×10)', color='blue', alpha=0.7)  # Scale for visibility
        ax2.bar(x + width, metrics_array[:, 2], width, label='Max Drawdown (%)', color='red', alpha=0.7)
        
        ax2.set_title('Performance Metrics Comparison', fontweight='bold')
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_xlabel('Strategy', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_names, fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    comparison_filename = f"results/Strategy_Comparison_{symbol}.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved comparison plot: {comparison_filename}")


def main():
    """Main function to run parallel backtests and generate ranked results."""
    print("🚀 QUANTSIM PARALLEL BACKTESTING WITH RANKING")
    print("=" * 70)
    
    # Configuration
    SYMBOL = "AAPL"  # Change this to test different stocks
    
    # Extended MA combinations for comprehensive testing
    ma_combinations = [
        # Fast strategies
        (3, 7, "MA_3_7_Ultra_Fast"),
        (5, 10, "MA_5_10_Very_Fast"),
        (5, 15, "MA_5_15_Fast"),
        (8, 21, "MA_8_21_Fast"),
        
        # Medium strategies
        (10, 20, "MA_10_20_Medium_Fast"),
        (10, 30, "MA_10_30_Medium"),
        (12, 26, "MA_12_26_Medium"),
        (15, 35, "MA_15_35_Medium"),
        
        # Slow strategies
        (20, 40, "MA_20_40_Slow"),
        (20, 50, "MA_20_50_Slow"),
        (25, 50, "MA_25_50_Slow"),
        (30, 60, "MA_30_60_Slow"),
        
        # Very slow strategies
        (30, 100, "MA_30_100_Very_Slow"),
        (50, 100, "MA_50_100_Very_Slow"),
        (50, 200, "MA_50_200_Ultra_Slow"),
        
        # Golden cross variations
        (50, 200, "MA_50_200_Golden_Cross"),
        (21, 50, "MA_21_50_Modified_Golden"),
        (13, 48, "MA_13_48_Fibonacci"),
    ]
    
    try:
        print(f"📈 Testing {len(ma_combinations)} different MA combinations on {SYMBOL}")
        print(f"⚡ Using parallel processing for maximum speed")
        
        # Run parallel backtests
        results = run_parallel_backtests(SYMBOL, ma_combinations)
        
        # Create ranked performance table
        print(f"\n📊 CREATING PERFORMANCE RANKINGS...")
        performance_df = create_ranked_performance_table(results)
        
        # Print ranked table
        print_ranked_table(performance_df, top_n=15)
        
        # Save results to CSV
        save_results_to_csv(performance_df)
        
        # Create comparison plot for top strategies
        print(f"\n📈 GENERATING COMPARISON PLOTS...")
        create_parallel_comparison_plot(results, SYMBOL, top_n=8)
        
        # Additional analysis
        if not performance_df.empty:
            print(f"\n🎯 DETAILED ANALYSIS:")
            print("=" * 50)
            
            # Best performers by different criteria
            best_sharpe = performance_df.iloc[0]  # Already sorted by Sharpe
            best_return = performance_df.loc[performance_df['Total Return (%)'].idxmax()]
            best_calmar = performance_df.loc[performance_df['Calmar Ratio'].idxmax()]
            lowest_dd = performance_df.loc[performance_df['Max Drawdown (%)'].idxmax()]  # Max because negative
            
            print(f"🏆 CHAMPION BY SHARPE RATIO:")
            print(f"   {best_sharpe['Strategy']} | Sharpe: {best_sharpe['Sharpe Ratio']:.3f} | Return: {best_sharpe['Total Return (%)']:.2f}%")
            
            print(f"\n💰 HIGHEST TOTAL RETURN:")
            print(f"   {best_return['Strategy']} | Return: {best_return['Total Return (%)']:.2f}% | Sharpe: {best_return['Sharpe Ratio']:.3f}")
            
            print(f"\n🛡️  BEST RISK-ADJUSTED (CALMAR):")
            print(f"   {best_calmar['Strategy']} | Calmar: {best_calmar['Calmar Ratio']:.3f} | DD: {best_calmar['Max Drawdown (%)']:.2f}%")
            
            print(f"\n🔒 LOWEST DRAWDOWN:")
            print(f"   {lowest_dd['Strategy']} | Max DD: {lowest_dd['Max Drawdown (%)']:.2f}% | Return: {lowest_dd['Total Return (%)']:.2f}%")
            
            # Strategy insights
            print(f"\n💡 STRATEGY INSIGHTS:")
            print("=" * 30)
            fast_strategies = performance_df[performance_df['Long MA'] <= 20]
            slow_strategies = performance_df[performance_df['Long MA'] >= 50]
            
            if not fast_strategies.empty and not slow_strategies.empty:
                avg_fast_sharpe = fast_strategies['Sharpe Ratio'].mean()
                avg_slow_sharpe = slow_strategies['Sharpe Ratio'].mean()
                avg_fast_return = fast_strategies['Total Return (%)'].mean()
                avg_slow_return = slow_strategies['Total Return (%)'].mean()
                
                print(f"   Fast Strategies (MA ≤ 20): Avg Sharpe = {avg_fast_sharpe:.3f}, Avg Return = {avg_fast_return:.2f}%")
                print(f"   Slow Strategies (MA ≥ 50): Avg Sharpe = {avg_slow_sharpe:.3f}, Avg Return = {avg_slow_return:.2f}%")
                
                if avg_slow_sharpe > avg_fast_sharpe:
                    print(f"   🐢 Slower strategies show better risk-adjusted returns!")
                else:
                    print(f"   🐰 Faster strategies show better risk-adjusted returns!")
        
        # Summary
        successful_backtests = len([r for r in results if 'error' not in r])
        print(f"\n✅ PARALLEL BACKTESTING COMPLETE!")
        print("=" * 50)
        print(f"📊 Successfully tested: {successful_backtests}/{len(ma_combinations)} strategies")
        print(f"📁 Results saved in: 'results/' directory")
        print(f"📈 Generated files:")
        print(f"   - backtest_results.csv (detailed metrics)")
        print(f"   - Top_8_Strategies_Comparison_{SYMBOL}.png (visual comparison)")
        print(f"🏆 Best overall strategy: {performance_df.iloc[0]['Strategy'] if not performance_df.empty else 'None'}")
        
    except Exception as e:
        print(f"❌ Error during parallel backtesting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()