import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from .portfolio import Portfolio
from .metrics import calculate_comprehensive_metrics, print_metrics_report
from strategies.base_strategy import BaseStrategy


class BacktestResult:
    """Container for backtest results and analysis."""
    
    def __init__(self, portfolio: Portfolio, strategy: BaseStrategy, 
                 data: pd.DataFrame, signals: pd.Series):
        """
        Initialize backtest result.
        
        Args:
            portfolio: Final portfolio state
            strategy: Strategy that was tested
            data: Price data used
            signals: Generated signals
        """
        self.portfolio = portfolio
        self.strategy = strategy
        self.data = data
        self.signals = signals
        
        # Get DataFrames
        self.trades_df = portfolio.get_trades_df()
        self.equity_df = portfolio.get_equity_df()
        self.performance = portfolio.get_performance_summary()
        
        # Calculate additional metrics using the new metrics module
        self._calculate_enhanced_metrics()
    
    def _calculate_enhanced_metrics(self):
        """Calculate comprehensive performance metrics using the metrics module."""
        if len(self.equity_df) == 0:
            return
        
        # Get equity values as list
        equity_values = self.equity_df['total_equity'].tolist()
        
        # Calculate comprehensive metrics
        enhanced_metrics = calculate_comprehensive_metrics(
            equity_values=equity_values,
            trade_log=self.trades_df,
            initial_capital=self.performance['initial_cash'],
            days=len(self.data),
            risk_free_rate=0.02
        )
        
        # Update performance dictionary with enhanced metrics
        self.performance.update(enhanced_metrics)
        
        # Store the enhanced metrics separately for easy access
        self.enhanced_metrics = enhanced_metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns / returns.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_series) == 0:
            return 0.0
        
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        return drawdown.min()
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if len(self.trades_df) == 0:
            return 0.0
        
        # Group trades by buy/sell pairs to calculate P&L per trade
        # Simplified: assume each sell follows a buy
        buy_trades = self.trades_df[self.trades_df['action'] == 'BUY']
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0.0
        
        wins = 0
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_pairs):
            buy_price = buy_trades.iloc[i]['price']
            sell_price = sell_trades.iloc[i]['price']
            if sell_price > buy_price:
                wins += 1
        
        return wins / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade."""
        if len(self.trades_df) == 0:
            return 0.0
        
        buy_trades = self.trades_df[self.trades_df['action'] == 'BUY']
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0.0
        
        wins = []
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_pairs):
            buy_price = buy_trades.iloc[i]['price']
            sell_price = sell_trades.iloc[i]['price']
            pnl = sell_price - buy_price
            if pnl > 0:
                wins.append(pnl)
        
        return np.mean(wins) if wins else 0.0
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade."""
        if len(self.trades_df) == 0:
            return 0.0
        
        buy_trades = self.trades_df[self.trades_df['action'] == 'BUY']
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0.0
        
        losses = []
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_pairs):
            buy_price = buy_trades.iloc[i]['price']
            sell_price = sell_trades.iloc[i]['price']
            pnl = sell_price - buy_price
            if pnl < 0:
                losses.append(abs(pnl))
        
        return np.mean(losses) if losses else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
        
        return avg_win / avg_loss
    
    def print_summary(self):
        """Print backtest summary."""
        print("=" * 60)
        print(f"BACKTEST RESULTS - {self.strategy.name}")
        print("=" * 60)
        
        print(f"Strategy Parameters: {self.strategy.get_parameters()}")
        print(f"Data Period: {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}")
        print(f"Total Days: {len(self.data)}")
        
        print("\n" + "-" * 30 + " PERFORMANCE " + "-" * 30)
        print(f"Initial Capital: ${self.performance['initial_cash']:,.2f}")
        print(f"Final Equity: ${self.performance['final_equity']:,.2f}")
        print(f"Total Return: {self.performance['total_return_pct']:.2f}%")
        print(f"Realized P&L: ${self.performance['realized_pnl']:,.2f}")
        
        print(f"\nSharpe Ratio: {self.performance.get('sharpe_ratio', 0):.3f}")
        print(f"Volatility: {self.performance.get('volatility', 0)*100:.2f}%")
        print(f"Max Drawdown: {self.performance.get('max_drawdown', 0)*100:.2f}%")
        
        print("\n" + "-" * 30 + " TRADING " + "-" * 30)
        print(f"Total Trades: {self.performance['num_trades']}")
        print(f"Win Rate: {self.performance.get('win_rate', 0)*100:.2f}%")
        print(f"Average Win: ${self.performance.get('avg_win', 0):.2f}")
        print(f"Average Loss: ${self.performance.get('avg_loss', 0):.2f}")
        print(f"Profit Factor: {self.performance.get('profit_factor', 0):.2f}")
        print(f"Total Commission: ${self.performance['total_commission']:,.2f}")
        
        # Signal summary
        buy_signals = (self.signals == 1).sum()
        sell_signals = (self.signals == -1).sum()
        print(f"\nBuy Signals: {buy_signals}")
        print(f"Sell Signals: {sell_signals}")
        
        print("\n" + "-" * 30 + " POSITIONS " + "-" * 30)
        for symbol, qty in self.performance['current_positions'].items():
            if qty != 0:
                print(f"{symbol}: {qty:.2f} shares")
        
        print("=" * 60)
    
    def print_enhanced_summary(self):
        """Print enhanced summary using the metrics module."""
        if hasattr(self, 'enhanced_metrics'):
            print_metrics_report(self.enhanced_metrics, self.strategy.name)
        else:
            print("Enhanced metrics not available. Using standard summary.")
            self.print_summary()
    
    def plot_results(self, figsize=(15, 10)):
        """Plot backtest results."""
        if len(self.equity_df) == 0:
            print("No equity data to plot.")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Cannot plot results.")
            print("Install matplotlib to see visual results: pip install matplotlib")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot 1: Price and Signals
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Price', linewidth=1)
        
        # Plot buy signals
        buy_signals = self.signals[self.signals == 1]
        if len(buy_signals) > 0:
            ax1.scatter(buy_signals.index, self.data.loc[buy_signals.index, 'Close'], 
                       color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        # Plot sell signals
        sell_signals = self.signals[self.signals == -1]
        if len(sell_signals) > 0:
            ax1.scatter(sell_signals.index, self.data.loc[sell_signals.index, 'Close'], 
                       color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{self.strategy.name} - Price and Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Equity
        ax2 = axes[1]
        equity_df = self.equity_df.copy()
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        ax2.plot(equity_df['timestamp'], equity_df['total_equity'], 
                label='Portfolio Value', linewidth=2, color='blue')
        ax2.axhline(y=self.performance['initial_cash'], color='gray', 
                   linestyle='--', alpha=0.7, label='Initial Capital')
        
        ax2.set_title('Portfolio Equity Over Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = axes[2]
        equity_series = equity_df.set_index('timestamp')['total_equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown (%)')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class Backtester:
    """
    Backtester class that loops over data, calls strategy.generate_signal(),
    and uses Portfolio to execute trades and log equity over time.
    """
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 1.0,
                 max_position_size: float = 0.25):
        """
        Initialize backtester.
        
        Args:
            initial_cash: Starting capital
            commission: Commission per trade
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.max_position_size = max_position_size
    
    def run(self, strategy: BaseStrategy, data: pd.DataFrame, 
            symbol: str = "STOCK") -> BacktestResult:
        """
        Run backtest on given strategy and data.
        
        Args:
            strategy: Trading strategy to test
            data: OHLCV data with Date as index
            symbol: Stock symbol for tracking
            
        Returns:
            BacktestResult object with results and analysis
        """
        print(f"Running backtest for {strategy.name}...")
        print(f"Data period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        print(f"Total data points: {len(data)}")
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission=self.commission,
            max_position_size=self.max_position_size
        )
        
        # Generate all signals first
        signals = strategy.generate_signals(data)
        
        # Track state
        current_position = 0  # 0 = no position, 1 = long, -1 = short
        
        # Loop through data
        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data.iloc[i]['Close']
            current_signal = signals.iloc[i]
            
            # Update portfolio with current price
            portfolio.update(symbol, current_price, current_date)
            
            # Execute trades based on signals
            if current_signal == 1 and current_position <= 0:  # Buy signal
                if current_position < 0:
                    # Close short position first
                    portfolio.buy(symbol, current_price, 
                                abs(portfolio.get_position(symbol)), current_date)
                
                # Open long position
                portfolio.buy(symbol, current_price, timestamp=current_date)
                current_position = 1
                
            elif current_signal == -1 and current_position >= 0:  # Sell signal
                if current_position > 0:
                    # Close long position first
                    portfolio.sell(symbol, current_price, 
                                 portfolio.get_position(symbol), current_date)
                
                # Open short position
                portfolio.sell(symbol, current_price, timestamp=current_date)
                current_position = -1
        
        # Final portfolio update
        final_price = data.iloc[-1]['Close']
        portfolio.update(symbol, final_price, data.index[-1])
        
        print(f"Backtest completed. Generated {len(portfolio.trades)} trades.")
        
        return BacktestResult(portfolio, strategy, data, signals)
    
    def run_multiple(self, strategies: List[BaseStrategy], data: pd.DataFrame,
                    symbol: str = "STOCK") -> Dict[str, BacktestResult]:
        """
        Run backtest on multiple strategies.
        
        Args:
            strategies: List of strategies to test
            data: OHLCV data
            symbol: Stock symbol
            
        Returns:
            Dictionary of {strategy_name: BacktestResult}
        """
        results = {}
        
        for strategy in strategies:
            result = self.run(strategy, data, symbol)
            results[strategy.name] = result
        
        return results
    
    def compare_strategies(self, results: Dict[str, BacktestResult]):
        """
        Compare multiple strategy results.
        
        Args:
            results: Dictionary of strategy results
        """
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        
        comparison_data = []
        
        for name, result in results.items():
            perf = result.performance
            comparison_data.append({
                'Strategy': name,
                'Total Return (%)': f"{perf['total_return_pct']:.2f}%",
                'Sharpe Ratio': f"{perf.get('sharpe_ratio', 0):.3f}",
                'Max Drawdown (%)': f"{perf.get('max_drawdown', 0)*100:.2f}%",
                'Win Rate (%)': f"{perf.get('win_rate', 0)*100:.2f}%",
                'Total Trades': perf['num_trades'],
                'Final Equity': f"${perf['final_equity']:,.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print("=" * 80)
