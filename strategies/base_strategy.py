from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Dict, Any


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategy implementations should inherit from this class and implement
    the generate_signal method.
    """
    
    def __init__(self, name: str = None, **kwargs):
        """
        Initialize the base strategy.
        
        Args:
            name (str): Name of the strategy
            **kwargs: Additional parameters for the strategy
        """
        self.name = name or self.__class__.__name__
        self.parameters = kwargs
        self.signals = []
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, i: int) -> int:
        """
        Generate trading signal for a given data point.
        
        Args:
            data (pd.DataFrame): OHLCV data with Date as index
            i (int): Current index position in the data
            
        Returns:
            int: Trading signal
                 1 = Buy signal
                 0 = Hold/No signal
                -1 = Sell signal
        """
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals for the entire dataset.
        
        Args:
            data (pd.DataFrame): OHLCV data with Date as index
            
        Returns:
            pd.Series: Series of trading signals aligned with data index
        """
        signals = []
        
        for i in range(len(data)):
            try:
                signal = self.generate_signal(data, i)
                signals.append(signal)
            except Exception as e:
                # Handle cases where we don't have enough data (e.g., early periods)
                signals.append(0)
                
        return pd.Series(signals, index=data.index, name=f'{self.name}_signal')
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of strategy parameters
        """
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs):
        """
        Update strategy parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        self.parameters.update(kwargs)
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        params_str = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({params_str})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return self.__str__()
