"""
Low-Budget Small-Cap Strategy Configuration

All tuneable parameters in one place. Based on peer-reviewed research:
"""

# Load small/micro-cap tickers from filtered universe file
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))

with open(_os.path.join(_HERE, "data", "universe.txt")) as f:
    UNIVERSE_TICKERS = [
        line.strip().replace("'", "").replace(",", "")
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

# ── Portfolio Settings ──────────────────────────────────────────────────────
INITIAL_CAPITAL = 1000  # Starting capital in USD
MAX_POSITIONS = 8  # Concentrated portfolio for $1k capital
MIN_POSITION_SIZE = 50  # Minimum position in USD (below this, skip)
MAX_POSITION_PCT = 0.12  # Max 12% in any single stock
REBALANCE_DAY = 1  # Day of month to rebalance (1 = first trading day)
REBALANCE_FREQUENCY = "quarterly"  # New: quarterly rebalancing for microcaps
REBALANCE_THRESHOLD_PCT = 0.20  # Skip rebalance if position within 20% of target

# ── Stock Universe Filters ──────────────────────────────────────────────────
MIN_MARKET_CAP = 10_000_000  # $10M minimum market cap
MAX_MARKET_CAP = 500_000_000  # $500M maximum market cap
MIN_AVG_VOLUME = 10_000  # Minimum average daily volume (shares)
MIN_DOLLAR_VOLUME = 50_000  # Lower liquidity filter for microcaps
MIN_PRICE = 1.00  # Exclude sub-dollar penny stocks
EXCLUDED_SECTORS = [  # Sectors to exclude
    "Financial Services",
    "Banks",
    "Insurance",
    "Shell Companies",
]

# ── Momentum Parameters (Jegadeesh & Titman 1993) ──────────────────────────
MOMENTUM_LOOKBACK = 126  # 6-month lookback (trading days)
MOMENTUM_SKIP = 10  # Skip most recent 2 weeks
MOMENTUM_WEIGHT = 0.35  # Weight in composite score

# ── Value Parameters ────────────────────────────────────────────────────────
VALUE_WEIGHT = 0.25  # Weight in composite score
# Metrics used: P/E, P/S, EV/EBITDA (lower is better)

# ── Piotroski F-Score Parameters ────────────────────────────────────────────
FSCORE_WEIGHT = 0.40  # Weight in composite score
MIN_FSCORE = 5  # Quality filter — only fundamentally sound companies

# ── Risk Management & Execution ─────────────────────────────────────────────
STOP_LOSS_PCT = -0.25  # -25% trailing stop (from peak price, not entry)
TRAILING_STOP = True  # True = trail from high-water mark; False = fixed from entry
HYSTERESIS_RANK = 30  # Only sell if stock drops below this rank
TRADING_COST_PCT = 0.0025  # 0.25% estimated round-trip spread cost
# Dynamic slippage: additional cost as a function of volatility and dollar volume
SLIPPAGE_VOL_MULT = 0.05  # 5% of 21-day volatility per trade (reduced)
SLIPPAGE_LIQUIDITY_MULT = 0.00025  # 0.025% per $100k below $1M ADV (reduced)

# ── Market Regime Filter ─────────────────────────────────────────────────────
USE_REGIME_FILTER = True  # Only buy when SPY > 200-day MA
REGIME_TICKER = "SPY"  # Benchmark for regime detection
REGIME_MA_PERIOD = 200  # 200-day moving average

# ── Sector Constraints ──────────────────────────────────────────────────────
MAX_SECTOR_PCT = 0.30  # Max 30% of portfolio in any single sector

# ── Backtest Settings ───────────────────────────────────────────────────────
BACKTEST_START = "2020-01-01"
BACKTEST_END = "2025-12-31"
BENCHMARK_TICKER = "IWM"  # iShares Russell 2000 ETF as small-cap benchmark
RISK_FREE_RATE = 0.045  # 4.5% annual risk-free rate (approx US T-bills)

# ── Alpaca API Settings ──────────────────────────────────────────────────────
# Keys are loaded from .env file (see .env.example)
# Set ALPACA_ENVIRONMENT to "paper" or "live"
ALPACA_ENVIRONMENT = "paper"  # "paper" for testing, "live" for real trading

# Alpaca data feed: "iex" (free) or "sip" (paid, real-time consolidated)
ALPACA_DATA_FEED = "iex"

# ── Data Sources ────────────────────────────────────────────────────────────
