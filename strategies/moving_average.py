import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    Generates buy signals when short-term MA crosses above long-term MA,
    and sell signals when short-term MA crosses below long-term MA.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30, **kwargs):
        """
        Initialize Moving Average Strategy.
        
        Args:
            short_window (int): Period for short-term moving average (default: 10)
            long_window (int): Period for long-term moving average (default: 30)
            **kwargs: Additional parameters passed to BaseStrategy
        """
        super().__init__(
            name="MovingAverageStrategy",
            short_window=short_window,
            long_window=long_window,
            **kwargs
        )
        
        self.short_window = short_window
        self.long_window = long_window
        
        # Cache for moving averages to avoid recalculation
        self._short_ma = None
        self._long_ma = None
        self._last_data_hash = None
    
    def _calculate_moving_averages(self, data: pd.DataFrame):
        """
        Calculate moving averages for the dataset.
        
        Args:
            data (pd.DataFrame): OHLCV data
        """
        # Use a simple hash to check if data has changed
        data_hash = hash(str(data.index[-1]) + str(len(data)))
        
        if self._last_data_hash != data_hash:
            self._short_ma = data['Close'].rolling(window=self.short_window).mean()
            self._long_ma = data['Close'].rolling(window=self.long_window).mean()
            self._last_data_hash = data_hash
    
    def generate_signal(self, data: pd.DataFrame, i: int) -> int:
        """
        Generate trading signal based on moving average crossover.
        
        Args:
            data (pd.DataFrame): OHLCV data with Date as index
            i (int): Current index position in the data
            
        Returns:
            int: Trading signal
                 1 = Buy signal (short MA crosses above long MA)
                 0 = Hold/No signal
                -1 = Sell signal (short MA crosses below long MA)
        """
        # Need at least long_window + 1 data points for crossover detection
        if i < self.long_window:
            return 0
        
        # Calculate moving averages if not cached or data changed
        self._calculate_moving_averages(data)
        
        # Get current and previous MA values
        current_short = self._short_ma.iloc[i]
        current_long = self._long_ma.iloc[i]
        prev_short = self._short_ma.iloc[i-1]
        prev_long = self._long_ma.iloc[i-1]
        
        # Check for NaN values
        if pd.isna(current_short) or pd.isna(current_long) or pd.isna(prev_short) or pd.isna(prev_long):
            return 0
        
        # Detect crossovers
        # Buy signal: short MA crosses above long MA
        if prev_short <= prev_long and current_short > current_long:
            return 1
        
        # Sell signal: short MA crosses below long MA
        elif prev_short >= prev_long and current_short < current_long:
            return -1
        
        # No signal
        return 0
    
    def get_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get the calculated moving averages for analysis.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with Close price and moving averages
        """
        self._calculate_moving_averages(data)
        
        result = pd.DataFrame(index=data.index)
        result['Close'] = data['Close']
        result[f'MA_{self.short_window}'] = self._short_ma
        result[f'MA_{self.long_window}'] = self._long_ma
        
        return result
    
    def plot_strategy(self, data: pd.DataFrame, signals: pd.Series = None):
        """
        Plot the strategy with price, moving averages, and signals.
        
        Args:
            data (pd.DataFrame): OHLCV data
            signals (pd.Series): Trading signals (optional, will generate if not provided)
        """
        try:
            import matplotlib.pyplot as plt
            
            if signals is None:
                signals = self.generate_signals(data)
            
            ma_data = self.get_moving_averages(data)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot price and moving averages
            ax.plot(ma_data.index, ma_data['Close'], label='Close Price', linewidth=1)
            ax.plot(ma_data.index, ma_data[f'MA_{self.short_window}'], 
                   label=f'MA {self.short_window}', alpha=0.7)
            ax.plot(ma_data.index, ma_data[f'MA_{self.long_window}'], 
                   label=f'MA {self.long_window}', alpha=0.7)
            
            # Plot buy signals
            buy_signals = signals[signals == 1]
            if len(buy_signals) > 0:
                ax.scatter(buy_signals.index, ma_data.loc[buy_signals.index, 'Close'], 
                          color='green', marker='^', s=100, label='Buy Signal', zorder=5)
            
            # Plot sell signals
            sell_signals = signals[signals == -1]
            if len(sell_signals) > 0:
                ax.scatter(sell_signals.index, ma_data.loc[sell_signals.index, 'Close'], 
                          color='red', marker='v', s=100, label='Sell Signal', zorder=5)
            
            ax.set_title(f'{self.name} - MA({self.short_window}) vs MA({self.long_window})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot strategy.")
        except Exception as e:
            print(f"Error plotting strategy: {e}")


class SimpleMovingAverageStrategy(MovingAverageStrategy):
    """Alias for MovingAverageStrategy for backward compatibility."""
    pass


class ExponentialMovingAverageStrategy(BaseStrategy):
    """
    Exponential Moving Average Crossover Strategy.
    
    Similar to MovingAverageStrategy but uses exponential moving averages
    which give more weight to recent prices.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30, **kwargs):
        """
        Initialize Exponential Moving Average Strategy.
        
        Args:
            short_window (int): Period for short-term EMA (default: 10)
            long_window (int): Period for long-term EMA (default: 30)
            **kwargs: Additional parameters passed to BaseStrategy
        """
        super().__init__(
            name="ExponentialMovingAverageStrategy",
            short_window=short_window,
            long_window=long_window,
            **kwargs
        )
        
        self.short_window = short_window
        self.long_window = long_window
        
        # Cache for EMAs
        self._short_ema = None
        self._long_ema = None
        self._last_data_hash = None
    
    def _calculate_emas(self, data: pd.DataFrame):
        """Calculate exponential moving averages."""
        data_hash = hash(str(data.index[-1]) + str(len(data)))
        
        if self._last_data_hash != data_hash:
            self._short_ema = data['Close'].ewm(span=self.short_window).mean()
            self._long_ema = data['Close'].ewm(span=self.long_window).mean()
            self._last_data_hash = data_hash
    
    def generate_signal(self, data: pd.DataFrame, i: int) -> int:
        """Generate signal based on EMA crossover."""
        if i < self.long_window:
            return 0
        
        self._calculate_emas(data)
        
        current_short = self._short_ema.iloc[i]
        current_long = self._long_ema.iloc[i]
        prev_short = self._short_ema.iloc[i-1]
        prev_long = self._long_ema.iloc[i-1]
        
        if pd.isna(current_short) or pd.isna(current_long) or pd.isna(prev_short) or pd.isna(prev_long):
            return 0
        
        # Buy signal: short EMA crosses above long EMA
        if prev_short <= prev_long and current_short > current_long:
            return 1
        
        # Sell signal: short EMA crosses below long EMA
        elif prev_short >= prev_long and current_short < current_long:
            return -1
        
        return 0
