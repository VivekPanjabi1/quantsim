from engine.data_loader import load_data

df = load_data("AAPL")   # Apple stock
print("Downloaded rows:", len(df))
print(df.head())
