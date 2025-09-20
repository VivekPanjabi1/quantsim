import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np


class Trade:
    """Represents a single trade transaction."""
    
    def __init__(self, timestamp: datetime, action: str, symbol: str, 
                 quantity: float, price: float, commission: float = 0.0):
        """
        Initialize a trade.
        
        Args:
            timestamp: When the trade occurred
            action: 'BUY' or 'SELL'
            symbol: Stock symbol
            quantity: Number of shares (positive for both buy/sell)
            price: Price per share
            commission: Trading commission/fees
        """
        self.timestamp = timestamp
        self.action = action.upper()
        self.symbol = symbol
        self.quantity = abs(quantity)  # Always positive
        self.price = price
        self.commission = commission
        self.value = self.quantity * self.price
        self.net_value = self.value + self.commission
    
    def __str__(self):
        return f"{self.timestamp.strftime('%Y-%m-%d')} {self.action} {self.quantity:.2f} {self.symbol} @ ${self.price:.2f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'action': self.action,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'value': self.value,
            'net_value': self.net_value
        }


class Portfolio:
    """
    Portfolio class to track cash, positions, equity value, and trade history.
    
    Supports both long and short positions.
    """
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.0, 
                 max_position_size: float = 0.25):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            commission: Commission per trade (fixed amount)
            max_position_size: Maximum position size as fraction of portfolio (0.25 = 25%)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission
        self.max_position_size = max_position_size
        
        # Position tracking
        self.positions = {}  # {symbol: quantity} - positive=long, negative=short
        self.avg_prices = {}  # {symbol: average_price}
        
        # Trade and equity history
        self.trades: List[Trade] = []
        self.equity_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.realized_pnl = 0.0
        
    def get_position(self, symbol: str) -> float:
        """Get current position size for a symbol."""
        return self.positions.get(symbol, 0.0)
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current market value of position."""
        quantity = self.get_position(symbol)
        return quantity * current_price
    
    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio equity.
        
        Args:
            prices: Dictionary of {symbol: current_price}
            
        Returns:
            Total portfolio value (cash + positions)
        """
        total_position_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if symbol in prices and quantity != 0:
                total_position_value += quantity * prices[symbol]
        
        return self.cash + total_position_value
    
    def can_buy(self, symbol: str, price: float, quantity: float) -> bool:
        """Check if we can afford to buy the specified quantity."""
        required_cash = quantity * price + self.commission
        return self.cash >= required_cash
    
    def can_sell(self, symbol: str, quantity: float) -> bool:
        """Check if we have enough shares to sell."""
        current_position = self.get_position(symbol)
        return current_position >= quantity
    
    def can_short(self, symbol: str, quantity: float) -> bool:
        """Check if we can short sell (simplified - assume always possible)."""
        return True  # In real trading, this would check margin requirements
    
    def calculate_position_size(self, price: float, portfolio_value: float) -> int:
        """
        Calculate position size based on max position size limit.
        
        Args:
            price: Price per share
            portfolio_value: Current portfolio value
            
        Returns:
            Number of shares to trade
        """
        max_value = portfolio_value * self.max_position_size
        max_shares = int(max_value / price)
        
        # Ensure we can afford it
        affordable_shares = int((self.cash - self.commission) / price)
        
        return min(max_shares, affordable_shares)
    
    def buy(self, symbol: str, price: float, quantity: Optional[float] = None, 
            timestamp: Optional[datetime] = None) -> bool:
        """
        Buy shares of a symbol.
        
        Args:
            symbol: Stock symbol
            price: Price per share
            quantity: Number of shares (if None, calculate based on max position size)
            timestamp: Trade timestamp
            
        Returns:
            True if trade executed successfully
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate quantity if not provided
        if quantity is None:
            portfolio_value = self.get_total_equity({symbol: price})
            quantity = self.calculate_position_size(price, portfolio_value)
        
        if quantity <= 0:
            return False
        
        # Check if we can afford the trade
        if not self.can_buy(symbol, price, quantity):
            return False
        
        # Execute trade
        cost = quantity * price + self.commission
        self.cash -= cost
        self.total_commission_paid += self.commission
        
        # Update position
        current_position = self.get_position(symbol)
        current_avg_price = self.avg_prices.get(symbol, 0.0)
        
        # Calculate new average price
        if current_position >= 0:  # Adding to long position or starting new long
            total_shares = current_position + quantity
            total_cost = (current_position * current_avg_price) + (quantity * price)
            new_avg_price = total_cost / total_shares if total_shares > 0 else price
        else:  # Covering short position
            if quantity > abs(current_position):
                # Covering short and going long
                remaining_quantity = quantity - abs(current_position)
                self.realized_pnl += abs(current_position) * (current_avg_price - price)
                new_avg_price = price
                total_shares = remaining_quantity
            else:
                # Partially covering short
                self.realized_pnl += quantity * (current_avg_price - price)
                new_avg_price = current_avg_price
                total_shares = current_position + quantity
        
        self.positions[symbol] = total_shares
        self.avg_prices[symbol] = new_avg_price
        
        # Log trade
        trade = Trade(timestamp, 'BUY', symbol, quantity, price, self.commission)
        self.trades.append(trade)
        
        return True
    
    def sell(self, symbol: str, price: float, quantity: Optional[float] = None,
             timestamp: Optional[datetime] = None) -> bool:
        """
        Sell shares of a symbol.
        
        Args:
            symbol: Stock symbol
            price: Price per share
            quantity: Number of shares (if None, sell entire position)
            timestamp: Trade timestamp
            
        Returns:
            True if trade executed successfully
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        current_position = self.get_position(symbol)
        
        # Calculate quantity if not provided
        if quantity is None:
            if current_position > 0:
                quantity = current_position  # Sell entire long position
            else:
                # For shorting, calculate based on max position size
                portfolio_value = self.get_total_equity({symbol: price})
                quantity = self.calculate_position_size(price, portfolio_value)
        
        if quantity <= 0:
            return False
        
        # Execute trade
        proceeds = quantity * price - self.commission
        self.cash += proceeds
        self.total_commission_paid += self.commission
        
        # Update position
        current_avg_price = self.avg_prices.get(symbol, price)
        
        if current_position > 0:  # Selling from long position
            if quantity >= current_position:
                # Selling entire long position and potentially going short
                self.realized_pnl += current_position * (price - current_avg_price)
                remaining_quantity = quantity - current_position
                new_position = -remaining_quantity if remaining_quantity > 0 else 0
                new_avg_price = price if remaining_quantity > 0 else 0
            else:
                # Partially selling long position
                self.realized_pnl += quantity * (price - current_avg_price)
                new_position = current_position - quantity
                new_avg_price = current_avg_price
        else:  # Adding to short position or starting new short
            new_position = current_position - quantity
            if current_position == 0:
                new_avg_price = price
            else:
                # Calculate weighted average for short position
                total_short_shares = abs(new_position)
                total_short_value = (abs(current_position) * current_avg_price) + (quantity * price)
                new_avg_price = total_short_value / total_short_shares
        
        self.positions[symbol] = new_position
        self.avg_prices[symbol] = new_avg_price
        
        # Log trade
        trade = Trade(timestamp, 'SELL', symbol, quantity, price, self.commission)
        self.trades.append(trade)
        
        return True
    
    def update(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """
        Update portfolio with current market price.
        
        Args:
            symbol: Stock symbol
            price: Current market price
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate current portfolio metrics
        prices = {symbol: price}
        total_equity = self.get_total_equity(prices)
        position_value = self.get_position_value(symbol, price)
        
        # Calculate unrealized P&L for this position
        position_qty = self.get_position(symbol)
        avg_price = self.avg_prices.get(symbol, 0.0)
        
        if position_qty != 0 and avg_price != 0:
            if position_qty > 0:  # Long position
                unrealized_pnl = position_qty * (price - avg_price)
            else:  # Short position
                unrealized_pnl = abs(position_qty) * (avg_price - price)
        else:
            unrealized_pnl = 0.0
        
        # Log equity snapshot
        equity_snapshot = {
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'cash': self.cash,
            'position_qty': position_qty,
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_equity': total_equity,
            'total_return': (total_equity - self.initial_cash) / self.initial_cash,
            'total_commission': self.total_commission_paid
        }
        
        self.equity_history.append(equity_snapshot)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = [trade.to_dict() for trade in self.trades]
        return pd.DataFrame(trade_data)
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity history as DataFrame."""
        if not self.equity_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_history)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary."""
        if not self.equity_history:
            return {}
        
        equity_df = self.get_equity_df()
        final_equity = equity_df['total_equity'].iloc[-1]
        
        return {
            'initial_cash': self.initial_cash,
            'final_equity': final_equity,
            'total_return': (final_equity - self.initial_cash) / self.initial_cash,
            'total_return_pct': ((final_equity - self.initial_cash) / self.initial_cash) * 100,
            'realized_pnl': self.realized_pnl,
            'total_commission': self.total_commission_paid,
            'num_trades': len(self.trades),
            'current_positions': dict(self.positions)
        }
    
    def __str__(self):
        """String representation of portfolio."""
        if self.equity_history:
            latest = self.equity_history[-1]
            return f"Portfolio(Cash: ${self.cash:.2f}, Equity: ${latest['total_equity']:.2f}, Return: {latest['total_return']*100:.2f}%)"
        else:
            return f"Portfolio(Cash: ${self.cash:.2f}, Initial: ${self.initial_cash:.2f})"
