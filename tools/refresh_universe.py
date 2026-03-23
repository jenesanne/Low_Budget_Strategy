"""
refresh_universe.py

Fetches the top 100 most liquid US small/micro-cap stocks from Alpaca and updates config.UNIVERSE_TICKERS.
"""

import config
from strategy.data_fetcher import fetch_snapshot, fetch_tradeable_assets

# Fetch all tradeable assets
assets = fetch_tradeable_assets()

# Filter by market cap and price
# (Assume market cap data is not available from Alpaca, so use price and liquidity)
assets = assets[assets["exchange"].isin(["NYSE", "NASDAQ", "AMEX"])]

# Get latest price and volume for all tickers
snapshots = fetch_snapshot(assets["ticker"].tolist())
assets = assets.merge(snapshots, left_on="ticker", right_on="ticker", how="inner")

# Compute dollar volume
assets["dollar_volume"] = assets["price"] * assets["volume"]

# Filter by price and liquidity
assets = assets[assets["price"] >= config.MIN_PRICE]
assets = assets[assets["dollar_volume"] >= config.MIN_DOLLAR_VOLUME]

# Sort by dollar volume, take top 100
top_liquid = assets.sort_values("dollar_volume", ascending=False).head(100)

# Update config.UNIVERSE_TICKERS
universe = top_liquid["ticker"].tolist()
print(f"Universe size: {len(universe)}")
print(universe)

# Optionally, write to config.py (manual step recommended for transparency)
with open("new_universe.txt", "w") as f:
    for t in universe:
        f.write(f"'{t}',\n")
print(
    "Top 100 liquid tickers written to new_universe.txt. Copy these into config.UNIVERSE_TICKERS."
)
