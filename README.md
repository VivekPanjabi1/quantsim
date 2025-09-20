# QuantSim - Professional Quantitative Trading Backtesting Engine

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

A comprehensive, high-performance quantitative trading simulation and backtesting framework built in Python. QuantSim provides institutional-grade performance analysis with parallel processing capabilities for strategy optimization.

## 🚀 Features

### Core Capabilities
- **Professional Backtesting Engine** - Realistic trade execution with commission and slippage
- **Parallel Processing** - Multi-core strategy optimization using `concurrent.futures`
- **Comprehensive Metrics** - 20+ performance indicators including Sharpe, Sortino, Calmar ratios
- **Advanced Portfolio Management** - Long/short positions with risk management
- **Professional Visualizations** - High-quality charts with matplotlib integration

### Strategy Framework
- **Modular Strategy Design** - Easy-to-extend abstract base class
- **Built-in Strategies** - Moving Average and Exponential Moving Average crossovers
- **Signal Generation** - Flexible buy/sell/hold signal system
- **Parameter Optimization** - Automated parameter sweeping with ranking

### Performance Analysis
- **Risk Metrics** - VaR, Maximum Drawdown, Volatility analysis
- **Return Analysis** - Total, annualized, and risk-adjusted returns
- **Trade Analytics** - Win rate, profit factor, average win/loss
- **Comparative Analysis** - Multi-strategy performance comparison

## 📊 Sample Results

```
🏆 TOP 5 STRATEGIES RANKED BY SHARPE RATIO
========================================================
Rank Strategy              Total Return (%) Sharpe Ratio Max Drawdown (%)
1    MA_30_100_Very_Slow   4.43            0.558        -5.76
2    MA_50_200_Golden_Cross 3.21           0.445        -4.23
3    MA_20_50_Slow         1.98            0.165        -8.70
4    MA_10_30_Medium       -5.64           -0.325       -12.75
5    MA_5_15_Fast          -9.28           -0.415       -16.52
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/quantsim.git
   cd quantsim
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### Basic Usage
```python
from engine.data_loader import load_data
from engine.backtester import Backtester
from strategies.moving_average import MovingAverageStrategy

# Load data
df = load_data("AAPL")

# Create strategy
strategy = MovingAverageStrategy(short_window=10, long_window=30)

# Initialize backtester
backtester = Backtester(
    initial_cash=100000.0,
    commission=1.0,
    max_position_size=0.25
)

# Run backtest
result = backtester.run(strategy, df, symbol="AAPL")

# Print results
result.print_enhanced_summary()
```

### Parallel Strategy Optimization
```bash
# Run comprehensive parallel backtesting
python run_backtest.py
```

This will:
- Test 18 different MA combinations in parallel
- Generate ranked performance table
- Create professional visualizations
- Export results to CSV

## 📁 Project Structure

```
quantsim/
├── engine/                 # Core backtesting engine
│   ├── __init__.py
│   ├── backtester.py      # Main backtesting logic
│   ├── data_loader.py     # Data loading and caching
│   ├── metrics.py         # Performance metrics calculations
│   └── portfolio.py       # Portfolio management
├── strategies/             # Trading strategies
│   ├── __init__.py
│   ├── base_strategy.py   # Abstract strategy base class
│   └── moving_average.py  # MA and EMA strategies
├── data/                  # Cached market data (auto-generated)
├── results/               # Generated plots and results (auto-generated)
├── tests/                 # Test files
│   ├── test_backtest.py
│   ├── test_enhanced_backtest.py
│   ├── test_loader.py
│   ├── test_metrics.py
│   └── test_strategies.py
├── run_backtest.py        # Main parallel backtesting script
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 📈 Available Strategies

### Moving Average Strategies
- **MovingAverageStrategy** - Simple moving average crossover
- **ExponentialMovingAverageStrategy** - EMA crossover with faster response

### Creating Custom Strategies
```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signal(self, data, i):
        # Your strategy logic here
        # Return: 1 (buy), -1 (sell), 0 (hold)
        return signal
```

## 🔧 Configuration

### Backtester Parameters
- `initial_cash`: Starting capital (default: $100,000)
- `commission`: Per-trade commission (default: $1.00)
- `max_position_size`: Maximum position as % of portfolio (default: 25%)

### Data Sources
- **Primary**: Yahoo Finance API (via direct HTTP requests)
- **Caching**: Local CSV files in `data/` directory
- **Timeframe**: Last 2 years of daily OHLCV data

## 📊 Performance Metrics

### Return Metrics
- Total Return, Annualized Return
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)

### Risk Metrics
- Maximum Drawdown, Volatility
- Value at Risk (VaR) at 95% and 99% confidence levels

### Trading Metrics
- Win Rate, Profit Factor
- Average Win/Loss, Total Trades
- Commission impact analysis

## 🎨 Visualizations

The system generates professional-quality charts including:
- **Price Charts** with buy/sell signals
- **Equity Curves** with embedded performance metrics
- **Drawdown Analysis** with risk visualization
- **Strategy Comparison** plots
- **Performance Rankings** with bar charts

## 🚀 Performance

### Parallel Processing
- **Multi-core optimization** using ProcessPoolExecutor
- **4x speed improvement** over sequential backtesting
- **Real-time progress tracking** during execution

### Benchmarks
- **18 strategies in ~60 seconds** (vs 3+ minutes sequential)
- **Memory efficient** with automatic cleanup
- **Scalable** to hundreds of strategy combinations

## 🧪 Testing

Run the test suite:
```bash
# Test individual components
python test_loader.py        # Data loading
python test_strategies.py    # Strategy signals
python test_metrics.py       # Performance metrics
python test_backtest.py      # Basic backtesting

# Test enhanced features
python test_enhanced_backtest.py  # Full system test
```

## 📝 Example Output

### Console Output
```
🚀 RUNNING 18 BACKTESTS IN PARALLEL...
✅ MA_30_100_Very_Slow    | Return:   4.43% | Sharpe:  0.558
✅ MA_50_200_Golden_Cross | Return:   3.21% | Sharpe:  0.445
✅ MA_20_50_Slow         | Return:   1.98% | Sharpe:  0.165
...
⏱️  Parallel backtesting completed in 45.23 seconds
📊 Successfully completed 18/18 backtests
```

### Generated Files
- `results/backtest_results.csv` - Detailed performance metrics
- `results/Top_8_Strategies_Comparison_AAPL.png` - Visual comparison
- Individual strategy plots for each tested combination

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt  # If you create this file

# Run tests before committing
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data
- **Python community** for excellent libraries (pandas, numpy, matplotlib)
- **Quantitative finance** community for strategy insights

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com

## 🔮 Roadmap

### Upcoming Features
- [ ] **More Strategies** - RSI, Bollinger Bands, MACD
- [ ] **Multi-Asset Support** - Portfolio-level backtesting
- [ ] **Live Trading Integration** - Broker API connections
- [ ] **Web Dashboard** - Interactive strategy management
- [ ] **Machine Learning** - AI-powered strategy optimization
- [ ] **Options Trading** - Derivatives strategy support

### Version History
- **v1.0.0** - Initial release with MA strategies and parallel processing
- **v0.9.0** - Beta release with core backtesting engine
- **v0.8.0** - Alpha release with basic strategy framework

---

⭐ **Star this repository if you find it useful!**

Built with ❤️ for the quantitative trading community.
