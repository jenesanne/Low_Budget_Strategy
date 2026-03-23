"""
Data fetching utilities for the Low-Budget Small-Cap Strategy.

Uses the Alpaca Markets API for US equity price data and asset metadata.
Alpaca free tier provides IEX data; paid gives SIP consolidated feed.

Docs: https://docs.alpaca.markets/docs/market-data-api
"""

import logging
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

import config

load_dotenv()  # Load .env file for API keys

logger = logging.getLogger(__name__)

# ── Alpaca client setup ─────────────────────────────────────────────────────


def _get_alpaca_clients():
    """Initialise and return Alpaca stock historical data client and trading client."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in your .env file. "
            "See .env.example for the template."
        )

    data_client = StockHistoricalDataClient(api_key, secret_key)
    trading_client = TradingClient(
        api_key, secret_key, paper=(config.ALPACA_ENVIRONMENT == "paper")
    )
    return data_client, trading_client


# ── Price Data ──────────────────────────────────────────────────────────────


def fetch_price_data(
    tickers: list[str],
    start: str = config.BACKTEST_START,
    end: str = config.BACKTEST_END,
) -> pd.DataFrame:
    """
    Download daily close prices for a list of tickers via Alpaca.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    data_client, _ = _get_alpaca_clients()

    logger.info(f"Fetching price data for {len(tickers)} tickers from {start} to {end}")

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        feed=config.ALPACA_DATA_FEED,
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df  # MultiIndex: (symbol, timestamp)

    if df.empty:
        logger.warning("No bar data returned from Alpaca")
        return pd.DataFrame()

    # Pivot to wide format: dates × tickers
    df = df.reset_index()
    prices = df.pivot_table(index="timestamp", columns="symbol", values="close")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    # Drop tickers with insufficient data (need at least 1 year of trading days)
    min_rows = 200
    valid = prices.columns[prices.count() >= min_rows]
    dropped = set(prices.columns) - set(valid)
    if dropped:
        logger.warning(f"Dropped tickers with insufficient data: {dropped}")
    prices = prices[valid]

    # --- Liquidity Filter: Minimum Dollar Volume ---
    # For each ticker, compute average daily dollar volume (close × volume)
    # Remove tickers below config.MIN_DOLLAR_VOLUME
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    # Fetch volume data (same as price fetch)
    request_vol = StockBarsRequest(
        symbol_or_symbols=valid.tolist(),
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        feed=config.ALPACA_DATA_FEED,
    )
    bars_vol = data_client.get_stock_bars(request_vol)
    df_vol = bars_vol.df.reset_index()
    # Compute ADV for each ticker
    adv = (
        df_vol.assign(dollar_volume=df_vol["close"] * df_vol["volume"])
        .groupby("symbol")["dollar_volume"]
        .mean()
    )
    liquid = adv[adv >= config.MIN_DOLLAR_VOLUME].index.tolist()
    illiquid = set(valid) - set(liquid)
    if illiquid:
        logger.warning(
            f"Dropped illiquid tickers (ADV < ${config.MIN_DOLLAR_VOLUME}): {illiquid}"
        )
    prices = prices[liquid]

    return prices


def fetch_latest_prices(tickers: list[str]) -> pd.Series:
    """Fetch the most recent closing price for each ticker."""
    from alpaca.data.requests import StockLatestBarRequest

    data_client, _ = _get_alpaca_clients()
    request = StockLatestBarRequest(
        symbol_or_symbols=tickers,
        feed=config.ALPACA_DATA_FEED,
    )
    latest = data_client.get_stock_latest_bar(request)
    return pd.Series({sym: bar.close for sym, bar in latest.items()})


# ── Asset Metadata & Screening ──────────────────────────────────────────────


def fetch_tradeable_assets() -> pd.DataFrame:
    """
    Fetch all active, tradeable US equities from Alpaca.
    Returns a DataFrame that can be used to dynamically build a universe
    beyond the static ticker list in config.
    """
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    _, trading_client = _get_alpaca_clients()

    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = trading_client.get_all_assets(request)

    records = []
    for a in assets:
        if a.tradable and a.fractionable:
            records.append(
                {
                    "ticker": a.symbol,
                    "name": a.name,
                    "exchange": a.exchange.value if a.exchange else None,
                    "tradable": a.tradable,
                    "fractionable": a.fractionable,
                    "shortable": a.shortable,
                }
            )

    df = pd.DataFrame(records)
    logger.info(f"Found {len(df)} tradeable, fractionable US equities on Alpaca")
    return df


def fetch_snapshot(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch current snapshot (price, volume, prev close) for a list of tickers.
    Useful for live screening and filtering.
    """
    from alpaca.data.requests import StockSnapshotRequest

    data_client, _ = _get_alpaca_clients()
    request = StockSnapshotRequest(
        symbol_or_symbols=tickers,
        feed=config.ALPACA_DATA_FEED,
    )
    snapshots = data_client.get_stock_snapshot(request)

    records = []
    for sym, snap in snapshots.items():
        # Alpaca API: use previous_daily_bar instead of prev_daily_bar
        if snap.daily_bar and getattr(snap, "previous_daily_bar", None):
            prev_bar = snap.previous_daily_bar
            records.append(
                {
                    "ticker": sym,
                    "price": snap.daily_bar.close,
                    "volume": snap.daily_bar.volume,
                    "prev_close": prev_bar.close,
                    "daily_change": (
                        (snap.daily_bar.close - prev_bar.close) / prev_bar.close
                        if prev_bar.close
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ── Universe Filtering ──────────────────────────────────────────────────────


def filter_universe(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Apply basic universe filters from config using snapshot data."""
    df = snapshot.copy()
    initial_count = len(df)

    # Price filter
    df = df[df["price"] >= config.MIN_PRICE]

    # Volume filter
    df = df[df["volume"] >= config.MIN_AVG_VOLUME]

    logger.info(f"Universe filtered: {initial_count} → {len(df)} stocks")
    return df


# ── Return Calculations ────────────────────────────────────────────────────


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from prices."""
    return prices.pct_change().dropna(how="all")


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly returns from daily prices."""
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna(how="all")
