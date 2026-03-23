"""
filter_small_caps.py

Fetches market cap for each ticker in new_universe.txt using Alpha Vantage and outputs a filtered list of small/micro-caps (<$500M).
"""

import os
import time

import requests

# Auto-load .env using python-dotenv for reliability
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("python-dotenv not installed; proceeding without auto-loading .env")

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    raise EnvironmentError("ALPHA_VANTAGE_API_KEY must be set in your .env file.")

with open("new_universe.txt") as f:
    tickers = [
        line.strip().replace("'", "").replace(",", "") for line in f if line.strip()
    ]

results = []
for t in tickers:
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={t}&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    mc = data.get("MarketCapitalization")
    try:
        mc = int(mc)
    except (TypeError, ValueError):
        mc = None
    results.append({"ticker": t, "market_cap": mc})
    print(f"{t}: {mc}")
    time.sleep(0.9)  # Alpha Vantage premium: up to 75 calls/minute

# Filter for small/micro-caps (<$500M)
filtered = [
    r["ticker"]
    for r in results
    if r["market_cap"] is not None and r["market_cap"] < 500_000_000
]

with open("smallcap_universe.txt", "w") as f:
    for t in filtered:
        f.write(f"'{t}',\n")
print(f"Filtered {len(filtered)} small/micro-caps written to smallcap_universe.txt.")
