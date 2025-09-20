from engine.data_loader import load_data
from strategies.moving_average import MovingAverageStrategy, ExponentialMovingAverageStrategy

# Load Apple stock data
print("Loading AAPL data...")
df = load_data("AAPL")
print(f"Loaded {len(df)} rows of data")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Test MovingAverageStrategy
print("\n" + "="*50)
print("Testing MovingAverageStrategy")
print("="*50)

# Create strategy with 10-day and 30-day moving averages
ma_strategy = MovingAverageStrategy(short_window=10, long_window=30)
print(f"Strategy: {ma_strategy}")

# Generate signals
signals = ma_strategy.generate_signals(df)
print(f"\nGenerated {len(signals)} signals")

# Count signal types
buy_signals = (signals == 1).sum()
sell_signals = (signals == -1).sum()
hold_signals = (signals == 0).sum()

print(f"Buy signals: {buy_signals}")
print(f"Sell signals: {sell_signals}")
print(f"Hold/No signals: {hold_signals}")

# Show some example signals
print(f"\nFirst 5 buy signals:")
buy_dates = signals[signals == 1].head()
for date, signal in buy_dates.items():
    price = df.loc[date, 'Close']
    print(f"  {date.strftime('%Y-%m-%d')}: Buy at ${price:.2f}")

print(f"\nFirst 5 sell signals:")
sell_dates = signals[signals == -1].head()
for date, signal in sell_dates.items():
    price = df.loc[date, 'Close']
    print(f"  {date.strftime('%Y-%m-%d')}: Sell at ${price:.2f}")

# Test ExponentialMovingAverageStrategy
print("\n" + "="*50)
print("Testing ExponentialMovingAverageStrategy")
print("="*50)

ema_strategy = ExponentialMovingAverageStrategy(short_window=10, long_window=30)
print(f"Strategy: {ema_strategy}")

ema_signals = ema_strategy.generate_signals(df)
ema_buy_signals = (ema_signals == 1).sum()
ema_sell_signals = (ema_signals == -1).sum()

print(f"EMA Buy signals: {ema_buy_signals}")
print(f"EMA Sell signals: {ema_sell_signals}")

# Compare strategies
print("\n" + "="*50)
print("Strategy Comparison")
print("="*50)
print(f"Simple MA Strategy: {buy_signals} buys, {sell_signals} sells")
print(f"EMA Strategy: {ema_buy_signals} buys, {ema_sell_signals} sells")

# Get moving averages for analysis
ma_data = ma_strategy.get_moving_averages(df)
print(f"\nMoving Averages (last 5 days):")
print(ma_data.tail())

print("\nStrategy test completed!")
