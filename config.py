"""
Elite Trader Momentum Strategy — Configuration

Combined strategy drawing from multiple profitable traders:
- Mark Minervini (SEPA Trend Template)     — structural trend filter
- William O'Neil (CANSLIM)                 — earnings growth + relative strength
- Nicolas Darvas (Box Theory)              — volume breakout confirmation
- Stan Weinstein (Stage Analysis)          — Stage 2 / rising 150-day MA
- Turtle Traders (Dennis & Eckhardt)       — ATR-based stops & position sizing
- Piotroski F-Score                        — financial quality gate
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
MAX_POSITIONS = 10  # Diversified enough to reduce blowup risk
MIN_POSITION_SIZE = 50  # Minimum position in USD (below this, skip)
MAX_POSITION_PCT = 0.15  # Max 15% in any single stock at entry
REBALANCE_DAY = 1  # Day of month to rebalance (1 = first trading day)
REBALANCE_FREQUENCY = "monthly"  # Monthly rebalance + daily slot-refill after stop-outs
REBALANCE_THRESHOLD_PCT = 0.25  # Skip rebalance if position within 25% of target

# ── Stock Universe Filters ──────────────────────────────────────────────────
MIN_MARKET_CAP = 10_000_000  # $10M minimum market cap
MAX_MARKET_CAP = 2_000_000_000  # $2B max — includes small-caps, not just micro
MIN_AVG_VOLUME = 10_000  # Minimum average daily volume (shares)
MIN_DOLLAR_VOLUME = 50_000  # Lower liquidity filter for microcaps
MIN_PRICE = 2.00  # Exclude low-priced stocks (Minervini avoids these)
EXCLUDED_SECTORS = [  # Sectors to exclude
    "Financial Services",
    "Banks",
    "Insurance",
    "Shell Companies",
]

# ── Minervini Trend Template Parameters ─────────────────────────────────────
# From Mark Minervini's SEPA — all criteria must pass for a stock to qualify
TREND_SMA_50 = 50  # 50-day simple moving average
TREND_SMA_150 = 150  # 150-day SMA (≈ Weinstein's 30-week MA)
TREND_SMA_200 = 200  # 200-day SMA
TREND_52W_HIGH_PCT = 0.75  # Price must be within 25% of 52-week high
TREND_52W_LOW_PCT = 1.25  # Price must be at least 25% above 52-week low
TREND_200_SLOPE_DAYS = 20  # 200-day MA must be rising over this many days
MIN_TREND_CRITERIA = 5  # Require 5 of 7 criteria (relax from all-7 for small caps)

# ── Momentum Parameters (O'Neil / Jegadeesh & Titman) ──────────────────────
MOMENTUM_LOOKBACK = 126  # 6-month lookback (trading days)
MOMENTUM_SKIP = 10  # Skip most recent 2 weeks (avoid reversal)
MOMENTUM_WEIGHT = 0.35  # Weight in composite score

# ── Trend Strength Parameters (Minervini / Weinstein) ──────────────────────
TREND_WEIGHT = 0.25  # Weight in composite score

# ── Volume Breakout Parameters (Darvas / O'Neil) ───────────────────────────
VOLUME_LOOKBACK = 50  # 50-day average volume baseline
VOLUME_SURGE_MULT = 1.5  # Recent volume must be > 1.5x avg (demand)
VOLUME_WEIGHT = 0.20  # Weight in composite score

# ── Quality / F-Score Parameters (Piotroski / O'Neil) ──────────────────────
FSCORE_WEIGHT = 0.20  # Weight in composite score
MIN_FSCORE = 4  # Quality filter — slightly relaxed for small-cap universe

# ── ATR-Based Risk Management (Turtle Traders / Minervini) ──────────────────
ATR_PERIOD = 20  # 20-day ATR for position sizing reference
ATR_STOP_MULT = 6  # 6× ATR trailing stop (Turtle Traders: adaptive to volatility)
STOP_LOSS_PCT = -0.20  # Emergency exit: if stock drops 20%+ from peak
TRAILING_STOP = True  # True = trail from high-water mark
HYSTERESIS_RANK = 30  # Only sell if stock drops below this rank

# ── Execution Costs ─────────────────────────────────────────────────────────
TRADING_COST_PCT = 0.001  # 0.1% estimated spread cost (Alpaca zero-commission)
SLIPPAGE_VOL_MULT = 0.03  # 3% of 21-day volatility per trade
SLIPPAGE_LIQUIDITY_MULT = 0.0001  # 0.01% per $100k below $1M ADV

# ── Market Regime Filter (O'Neil's "M" / Weinstein) ─────────────────────────
USE_REGIME_FILTER = True  # Only buy when SPY > 200-day MA
REGIME_TICKER = "SPY"  # Benchmark for regime detection
REGIME_MA_PERIOD = 200  # 200-day moving average
REGIME_EXIT_PCT = -0.02  # Go to cash when SPY is 2% below 200-day MA
REGIME_ENTER_PCT = 0.01  # Re-enter when SPY is 1% above 200-day MA

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
