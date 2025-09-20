import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def calculate_total_return(equity_values: List[float], initial_capital: float) -> float:
    """
    Calculate total return from equity values.
    
    Args:
        equity_values: List of portfolio equity values over time
        initial_capital: Starting capital amount
        
    Returns:
        Total return as decimal (e.g., 0.15 for 15% return)
    """
    if not equity_values or initial_capital <= 0:
        return 0.0
    
    final_equity = equity_values[-1]
    return (final_equity - initial_capital) / initial_capital


def calculate_annualized_return(equity_values: List[float], initial_capital: float, 
                              days: int) -> float:
    """
    Calculate annualized return.
    
    Args:
        equity_values: List of portfolio equity values over time
        initial_capital: Starting capital amount
        days: Number of days in the period
        
    Returns:
        Annualized return as decimal
    """
    if not equity_values or initial_capital <= 0 or days <= 0:
        return 0.0
    
    total_return = calculate_total_return(equity_values, initial_capital)
    years = days / 365.25
    
    if years <= 0:
        return 0.0
    
    # Compound annual growth rate (CAGR)
    return (1 + total_return) ** (1 / years) - 1


def calculate_daily_returns(equity_values: List[float]) -> List[float]:
    """
    Calculate daily returns from equity values.
    
    Args:
        equity_values: List of portfolio equity values over time
        
    Returns:
        List of daily returns as decimals
    """
    if len(equity_values) < 2:
        return []
    
    daily_returns = []
    for i in range(1, len(equity_values)):
        if equity_values[i-1] != 0:
            daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            daily_returns.append(daily_return)
        else:
            daily_returns.append(0.0)
    
    return daily_returns


def calculate_sharpe_ratio(equity_values: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio from equity values.
    
    Args:
        equity_values: List of portfolio equity values over time
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sharpe ratio (annualized)
    """
    daily_returns = calculate_daily_returns(equity_values)
    
    if len(daily_returns) == 0:
        return 0.0
    
    # Convert to numpy array for easier calculations
    returns = np.array(daily_returns)
    
    # Calculate excess returns (daily)
    daily_risk_free_rate = risk_free_rate / 252  # Assuming 252 trading days per year
    excess_returns = returns - daily_risk_free_rate
    
    # Calculate Sharpe ratio
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    
    # Annualize the Sharpe ratio
    return sharpe * np.sqrt(252)


def calculate_sortino_ratio(equity_values: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio (similar to Sharpe but only considers downside volatility).
    
    Args:
        equity_values: List of portfolio equity values over time
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sortino ratio (annualized)
    """
    daily_returns = calculate_daily_returns(equity_values)
    
    if len(daily_returns) == 0:
        return 0.0
    
    returns = np.array(daily_returns)
    daily_risk_free_rate = risk_free_rate / 252
    excess_returns = returns - daily_risk_free_rate
    
    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return float('inf')  # No downside risk
    
    downside_deviation = np.std(negative_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / downside_deviation
    
    # Annualize the Sortino ratio
    return sortino * np.sqrt(252)


def calculate_max_drawdown(equity_values: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity values.
    
    Args:
        equity_values: List of portfolio equity values over time
        
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
        max_drawdown: Maximum drawdown as decimal (negative value)
        start_index: Index where drawdown period started
        end_index: Index where maximum drawdown occurred
    """
    if len(equity_values) < 2:
        return 0.0, 0, 0
    
    equity_series = np.array(equity_values)
    
    # Calculate running maximum (peak values)
    running_max = np.maximum.accumulate(equity_series)
    
    # Calculate drawdown at each point
    drawdown = (equity_series - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_index = np.argmin(drawdown)
    max_drawdown = drawdown[max_dd_index]
    
    # Find the start of the drawdown period (last peak before max drawdown)
    start_index = 0
    for i in range(max_dd_index, -1, -1):
        if equity_series[i] == running_max[i]:
            start_index = i
            break
    
    return max_drawdown, start_index, max_dd_index


def calculate_calmar_ratio(equity_values: List[float], initial_capital: float, 
                          days: int) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        equity_values: List of portfolio equity values over time
        initial_capital: Starting capital amount
        days: Number of days in the period
        
    Returns:
        Calmar ratio
    """
    annualized_return = calculate_annualized_return(equity_values, initial_capital, days)
    max_drawdown, _, _ = calculate_max_drawdown(equity_values)
    
    if max_drawdown == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return annualized_return / abs(max_drawdown)


def calculate_win_rate(trade_log: pd.DataFrame) -> float:
    """
    Calculate win rate from trade log.
    
    Args:
        trade_log: DataFrame with columns ['action', 'price', 'quantity', 'timestamp']
        
    Returns:
        Win rate as decimal (e.g., 0.65 for 65% win rate)
    """
    if trade_log.empty:
        return 0.0
    
    # Separate buy and sell trades
    buy_trades = trade_log[trade_log['action'] == 'BUY'].copy()
    sell_trades = trade_log[trade_log['action'] == 'SELL'].copy()
    
    if len(buy_trades) == 0 or len(sell_trades) == 0:
        return 0.0
    
    # Sort by timestamp to match trades properly
    buy_trades = buy_trades.sort_values('timestamp')
    sell_trades = sell_trades.sort_values('timestamp')
    
    wins = 0
    total_trades = 0
    
    # Simple matching: each sell follows a buy
    min_trades = min(len(buy_trades), len(sell_trades))
    
    for i in range(min_trades):
        buy_price = buy_trades.iloc[i]['price']
        sell_price = sell_trades.iloc[i]['price']
        
        if sell_price > buy_price:
            wins += 1
        total_trades += 1
    
    return wins / total_trades if total_trades > 0 else 0.0


def calculate_profit_factor(trade_log: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trade_log: DataFrame with columns ['action', 'price', 'quantity', 'timestamp']
        
    Returns:
        Profit factor
    """
    if trade_log.empty:
        return 0.0
    
    buy_trades = trade_log[trade_log['action'] == 'BUY'].copy()
    sell_trades = trade_log[trade_log['action'] == 'SELL'].copy()
    
    if len(buy_trades) == 0 or len(sell_trades) == 0:
        return 0.0
    
    buy_trades = buy_trades.sort_values('timestamp')
    sell_trades = sell_trades.sort_values('timestamp')
    
    gross_profit = 0.0
    gross_loss = 0.0
    
    min_trades = min(len(buy_trades), len(sell_trades))
    
    for i in range(min_trades):
        buy_price = buy_trades.iloc[i]['price']
        sell_price = sell_trades.iloc[i]['price']
        quantity = buy_trades.iloc[i]['quantity']
        
        pnl = (sell_price - buy_price) * quantity
        
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_average_win_loss(trade_log: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate average winning and losing trade amounts.
    
    Args:
        trade_log: DataFrame with columns ['action', 'price', 'quantity', 'timestamp']
        
    Returns:
        Tuple of (average_win, average_loss)
    """
    if trade_log.empty:
        return 0.0, 0.0
    
    buy_trades = trade_log[trade_log['action'] == 'BUY'].copy()
    sell_trades = trade_log[trade_log['action'] == 'SELL'].copy()
    
    if len(buy_trades) == 0 or len(sell_trades) == 0:
        return 0.0, 0.0
    
    buy_trades = buy_trades.sort_values('timestamp')
    sell_trades = sell_trades.sort_values('timestamp')
    
    wins = []
    losses = []
    
    min_trades = min(len(buy_trades), len(sell_trades))
    
    for i in range(min_trades):
        buy_price = buy_trades.iloc[i]['price']
        sell_price = sell_trades.iloc[i]['price']
        quantity = buy_trades.iloc[i]['quantity']
        
        pnl = (sell_price - buy_price) * quantity
        
        if pnl > 0:
            wins.append(pnl)
        else:
            losses.append(abs(pnl))
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    return avg_win, avg_loss


def calculate_volatility(equity_values: List[float], annualized: bool = True) -> float:
    """
    Calculate portfolio volatility from equity values.
    
    Args:
        equity_values: List of portfolio equity values over time
        annualized: Whether to return annualized volatility
        
    Returns:
        Volatility as decimal
    """
    daily_returns = calculate_daily_returns(equity_values)
    
    if len(daily_returns) == 0:
        return 0.0
    
    volatility = np.std(daily_returns)
    
    if annualized:
        volatility *= np.sqrt(252)  # Annualize assuming 252 trading days
    
    return volatility


def calculate_var(equity_values: List[float], confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) at given confidence level.
    
    Args:
        equity_values: List of portfolio equity values over time
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR as decimal (negative value representing potential loss)
    """
    daily_returns = calculate_daily_returns(equity_values)
    
    if len(daily_returns) == 0:
        return 0.0
    
    return np.percentile(daily_returns, confidence_level * 100)


def calculate_comprehensive_metrics(equity_values: List[float], trade_log: pd.DataFrame,
                                  initial_capital: float, days: int,
                                  risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        equity_values: List of portfolio equity values over time
        trade_log: DataFrame with trade information
        initial_capital: Starting capital amount
        days: Number of days in the period
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Return metrics
    metrics['total_return'] = calculate_total_return(equity_values, initial_capital)
    metrics['total_return_pct'] = metrics['total_return'] * 100
    metrics['annualized_return'] = calculate_annualized_return(equity_values, initial_capital, days)
    metrics['annualized_return_pct'] = metrics['annualized_return'] * 100
    
    # Risk metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(equity_values, risk_free_rate)
    metrics['sortino_ratio'] = calculate_sortino_ratio(equity_values, risk_free_rate)
    metrics['volatility'] = calculate_volatility(equity_values)
    metrics['volatility_pct'] = metrics['volatility'] * 100
    
    # Drawdown metrics
    max_dd, dd_start, dd_end = calculate_max_drawdown(equity_values)
    metrics['max_drawdown'] = max_dd
    metrics['max_drawdown_pct'] = max_dd * 100
    metrics['calmar_ratio'] = calculate_calmar_ratio(equity_values, initial_capital, days)
    
    # Trading metrics
    metrics['win_rate'] = calculate_win_rate(trade_log)
    metrics['win_rate_pct'] = metrics['win_rate'] * 100
    metrics['profit_factor'] = calculate_profit_factor(trade_log)
    
    avg_win, avg_loss = calculate_average_win_loss(trade_log)
    metrics['avg_win'] = avg_win
    metrics['avg_loss'] = avg_loss
    
    # Risk metrics
    metrics['var_95'] = calculate_var(equity_values, 0.05)
    metrics['var_99'] = calculate_var(equity_values, 0.01)
    
    # Trade count
    metrics['total_trades'] = len(trade_log)
    metrics['buy_trades'] = len(trade_log[trade_log['action'] == 'BUY'])
    metrics['sell_trades'] = len(trade_log[trade_log['action'] == 'SELL'])
    
    return metrics


def print_metrics_report(metrics: Dict[str, Any], strategy_name: str = "Strategy"):
    """
    Print a formatted metrics report.
    
    Args:
        metrics: Dictionary of calculated metrics
        strategy_name: Name of the strategy for the report header
    """
    print("=" * 70)
    print(f"PERFORMANCE METRICS - {strategy_name}")
    print("=" * 70)
    
    print(f"\n📈 RETURN METRICS:")
    print(f"   Total Return:        {metrics.get('total_return_pct', 0):.2f}%")
    print(f"   Annualized Return:   {metrics.get('annualized_return_pct', 0):.2f}%")
    
    print(f"\n⚡ RISK METRICS:")
    print(f"   Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}")
    print(f"   Volatility:          {metrics.get('volatility_pct', 0):.2f}%")
    print(f"   Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"   Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}")
    print(f"   VaR (95%):           {metrics.get('var_95', 0)*100:.2f}%")
    
    print(f"\n💰 TRADING METRICS:")
    print(f"   Win Rate:            {metrics.get('win_rate_pct', 0):.2f}%")
    print(f"   Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
    print(f"   Average Win:         ${metrics.get('avg_win', 0):.2f}")
    print(f"   Average Loss:        ${metrics.get('avg_loss', 0):.2f}")
    print(f"   Total Trades:        {metrics.get('total_trades', 0)}")
    
    print("=" * 70)
