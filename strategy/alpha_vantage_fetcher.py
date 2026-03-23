"""
alpha_vantage_fetcher.py

Fetches fundamental data and historical prices for US stocks from Alpha Vantage.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CACHE = {}
CACHE_TTL = 60 * 60  # 1 hour
DISK_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
DISK_CACHE_TTL = 24 * 60 * 60  # 24 hours — fundamentals don't change fast

logger = logging.getLogger(__name__)


def fetch_monthly_adjusted_close(ticker: str) -> pd.DataFrame:
    """Fetch monthly adjusted close prices for a ticker from Alpha Vantage."""
    params = {
        "function": "TIME_SERIES_MONTHLY_ADJUSTED",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    try:
        resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=10)
        data = resp.json()
        ts = data.get("Monthly Adjusted Time Series", {})
        if not ts:
            return pd.DataFrame(columns=["date", "adj_close"])
        records = []
        for date, values in ts.items():
            records.append(
                {
                    "date": pd.to_datetime(date),
                    "adj_close": float(values["5. adjusted close"]),
                }
            )
        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "adj_close"])


def fetch_fundamentals(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch company overview (fundamentals) for each ticker from Alpha Vantage."""
    results = {}
    for ticker in tickers:
        now = time.time()
        if ticker in CACHE and now - CACHE[ticker]["timestamp"] < CACHE_TTL:
            results[ticker] = CACHE[ticker]["data"]
            continue
        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        try:
            resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=10)
            data = resp.json()
            if "Symbol" in data:
                results[ticker] = data
                CACHE[ticker] = {"data": data, "timestamp": now}
            else:
                results[ticker] = None
        except Exception:
            results[ticker] = None
    return results


def rank_stocks_by_fundamentals(
    fundamentals: Dict[str, Dict[str, Any]], top_n: int = 10
) -> List[str]:
    """Rank stocks by a composite of low P/E, high EPS growth, and high ROE."""
    scored = []
    for ticker, data in fundamentals.items():
        if not data:
            continue
        try:
            pe = float(data.get("PERatio", 0)) or 1000
            eps_growth = float(data.get("QuarterlyEarningsGrowthYOY", 0))
            roe = float(data.get("ReturnOnEquityTTM", 0))
            score = (-pe) + (eps_growth * 10) + (roe * 2)
            scored.append((ticker, score))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in scored[:top_n]]


def _safe_float(val, default=np.nan):
    """Convert Alpha Vantage string to float, returning NaN on failure."""
    if val is None or val == "None" or val == "-":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _load_disk_cache() -> dict:
    """Load cached fundamentals from disk if fresh enough."""
    cache_file = DISK_CACHE_DIR / "fundamentals.json"
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text())
        if time.time() - data.get("timestamp", 0) > DISK_CACHE_TTL:
            return {}
        return data.get("tickers", {})
    except Exception:
        return {}


def _save_disk_cache(ticker_data: dict):
    """Save fundamentals to disk cache."""
    DISK_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = DISK_CACHE_DIR / "fundamentals.json"
    payload = {"timestamp": time.time(), "tickers": ticker_data}
    cache_file.write_text(json.dumps(payload))


def fetch_fundamentals_for_scoring(
    tickers: list[str],
    calls_per_minute: int = 75,
) -> pd.DataFrame:
    """
    Fetch Alpha Vantage OVERVIEW for each ticker and return a DataFrame
    with columns matching what scoring.py expects:
      pe_ratio, ps_ratio, ev_ebitda, roa, operating_cashflow, net_income,
      debt_to_equity, current_ratio, gross_margin, asset_turnover

    Uses disk cache (24h TTL) to avoid hitting the API on every run.
    """
    disk_cache = _load_disk_cache()
    to_fetch = [t for t in tickers if t not in disk_cache]

    if to_fetch:
        logger.info(
            f"Fetching fundamentals for {len(to_fetch)} tickers from Alpha Vantage "
            f"({len(tickers) - len(to_fetch)} cached)..."
        )
        delay = 60.0 / calls_per_minute
        for i, ticker in enumerate(to_fetch):
            if i > 0:
                time.sleep(delay)
            if (i + 1) % 25 == 0 or i == len(to_fetch) - 1:
                logger.info(f"  Progress: {i + 1}/{len(to_fetch)}")
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": ALPHA_VANTAGE_API_KEY,
            }
            try:
                resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
                data = resp.json()
                if "Symbol" in data:
                    disk_cache[ticker] = data
                else:
                    logger.debug(f"No overview data for {ticker}")
            except Exception as e:
                logger.debug(f"Failed to fetch {ticker}: {e}")
        _save_disk_cache(disk_cache)
    else:
        logger.info(f"All {len(tickers)} tickers loaded from disk cache.")

    # Build DataFrame with scoring-compatible columns
    rows = []
    for ticker in tickers:
        raw = disk_cache.get(ticker)
        if not raw:
            continue

        pe = _safe_float(raw.get("PERatio"))
        ps = _safe_float(raw.get("PriceToSalesRatioTTM"))
        ev_ebitda = _safe_float(raw.get("EVToEBITDA"))
        roa = _safe_float(raw.get("ReturnOnAssetsTTM"))
        roe = _safe_float(raw.get("ReturnOnEquityTTM"))
        op_margin = _safe_float(raw.get("OperatingMarginTTM"))
        profit_margin = _safe_float(raw.get("ProfitMargin"))
        gross_profit = _safe_float(raw.get("GrossProfitTTM"))
        revenue = _safe_float(raw.get("RevenueTTM"))
        market_cap = _safe_float(raw.get("MarketCapitalization"))
        book_value = _safe_float(raw.get("BookValue"))
        shares = _safe_float(raw.get("SharesOutstanding"))
        ebitda = _safe_float(raw.get("EBITDA"))

        # Derived metrics for F-Score
        gross_margin = (gross_profit / revenue) if revenue and revenue > 0 else np.nan

        # Proxy operating cashflow: positive operating margin → positive cash flow
        # (Real CF would need CASH_FLOW endpoint, but this proxy works for F-Score binary check)
        operating_cashflow = (
            (op_margin * revenue)
            if not np.isnan(op_margin) and not np.isnan(revenue)
            else np.nan
        )

        # Proxy net income from profit margin × revenue
        net_income = (
            (profit_margin * revenue)
            if not np.isnan(profit_margin) and not np.isnan(revenue)
            else np.nan
        )

        # Debt-to-equity proxy: (MarketCap / BookValue×Shares - 1) is not great;
        # Use EV/EBITDA as a leverage proxy instead. D/E not available from OVERVIEW.
        # We'll compute total equity = book_value * shares, and if EV > MarketCap, debt ≈ EV - MarketCap
        total_equity = (
            (book_value * shares)
            if not np.isnan(book_value) and not np.isnan(shares)
            else np.nan
        )
        ev = (
            (ev_ebitda * ebitda)
            if not np.isnan(ev_ebitda) and not np.isnan(ebitda) and ebitda > 0
            else np.nan
        )
        debt_approx = (
            max(0, ev - market_cap)
            if not np.isnan(ev) and not np.isnan(market_cap)
            else np.nan
        )
        debt_to_equity = (
            (debt_approx / total_equity * 100)
            if total_equity and total_equity > 0 and not np.isnan(debt_approx)
            else np.nan
        )

        # Asset turnover = Revenue / Total Assets; proxy Total Assets ≈ MarketCap (rough)
        asset_turnover = (
            (revenue / market_cap)
            if not np.isnan(revenue) and market_cap and market_cap > 0
            else np.nan
        )

        rows.append(
            {
                "ticker": ticker,
                "pe_ratio": pe,
                "ps_ratio": ps,
                "ev_ebitda": ev_ebitda,
                "roa": roa,
                "operating_cashflow": operating_cashflow,
                "net_income": net_income,
                "debt_to_equity": debt_to_equity,
                "current_ratio": np.nan,  # Not available from OVERVIEW
                "gross_margin": gross_margin,
                "asset_turnover": asset_turnover,
                "sector": raw.get("Sector", ""),
            }
        )

    if not rows:
        logger.warning("No fundamental data retrieved from Alpha Vantage")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")
    valid = df.dropna(how="all", subset=["pe_ratio", "ps_ratio", "ev_ebitda", "roa"])
    logger.info(
        f"Fundamentals ready: {len(valid)} tickers with data "
        f"(of {len(tickers)} requested)"
    )
    return valid


def _load_historical_cache() -> dict:
    """Load cached historical fundamentals from disk."""
    cache_file = DISK_CACHE_DIR / "historical_fundamentals.json"
    if not cache_file.exists():
        return {}
    try:
        return json.loads(cache_file.read_text())
    except Exception:
        return {}


def _save_historical_cache(data: dict):
    """Save historical fundamentals to disk (no TTL — data is immutable)."""
    DISK_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = DISK_CACHE_DIR / "historical_fundamentals.json"
    cache_file.write_text(json.dumps(data))


def _build_quarter_row(ticker: str, inc: dict, bs: dict, cf: dict) -> dict:
    """Build a scoring-compatible row from one quarter's IS/BS/CF data."""
    revenue = _safe_float(inc.get("totalRevenue"))
    gross_profit = _safe_float(inc.get("grossProfit"))
    net_income = _safe_float(inc.get("netIncome"))
    operating_income = _safe_float(inc.get("operatingIncome"))
    ebitda = _safe_float(inc.get("ebitda"))

    total_assets = _safe_float(bs.get("totalAssets"))
    total_equity = _safe_float(bs.get("totalShareholderEquity"))
    total_debt = _safe_float(bs.get("shortLongTermDebtTotal"))
    current_assets = _safe_float(bs.get("totalCurrentAssets"))
    current_liabilities = _safe_float(bs.get("totalCurrentLiabilities"))
    shares = _safe_float(bs.get("commonStockSharesOutstanding"))

    operating_cashflow = _safe_float(cf.get("operatingCashflow"))

    # ROA = Net Income / Total Assets
    roa = net_income / total_assets if total_assets and total_assets > 0 else np.nan

    # Gross margin
    gross_margin = gross_profit / revenue if revenue and revenue > 0 else np.nan

    # Debt-to-equity (as %)
    debt_to_equity = (
        (total_debt / total_equity * 100)
        if total_equity and total_equity > 0 and not np.isnan(total_debt)
        else np.nan
    )

    # Current ratio
    current_ratio = (
        current_assets / current_liabilities
        if current_liabilities and current_liabilities > 0
        else np.nan
    )

    # Asset turnover
    asset_turnover = (
        revenue / total_assets if total_assets and total_assets > 0 else np.nan
    )

    # P/E, P/S, EV/EBITDA — need price, which we don't have here.
    # We leave these as NaN; the backtester can compute them from prices at rebalance time.
    # For now, provide EV/EBITDA from enterprise value approximation.
    ev_ebitda = np.nan  # Will be filled in by backtester if prices available

    return {
        "ticker": ticker,
        "roa": roa,
        "operating_cashflow": operating_cashflow,
        "net_income": net_income,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "gross_margin": gross_margin,
        "asset_turnover": asset_turnover,
        "pe_ratio": np.nan,
        "ps_ratio": np.nan,
        "ev_ebitda": ev_ebitda,
        "revenue": revenue,
        "ebitda": ebitda,
        "total_equity": total_equity,
        "shares_outstanding": shares,
    }


def fetch_historical_fundamentals(
    tickers: list[str],
    calls_per_minute: int = 75,
) -> dict[str, pd.DataFrame]:
    """
    Fetch quarterly IS/BS/CF from Alpha Vantage and return a dict of
    {quarter_end_date_str: DataFrame} where each DataFrame has scoring-compatible columns.

    Results are cached permanently to disk (historical data doesn't change).
    Returns: dict mapping 'YYYY-MM-DD' → pd.DataFrame indexed by ticker.
    """
    cache = _load_historical_cache()
    to_fetch = [t for t in tickers if t not in cache]

    if to_fetch:
        logger.info(
            f"Fetching historical financials for {len(to_fetch)} tickers "
            f"(3 calls each, ~{len(to_fetch) * 3 * 60 / calls_per_minute:.0f}s)..."
        )
        delay = 60.0 / calls_per_minute

        call_count = 0
        for i, ticker in enumerate(to_fetch):
            ticker_data = {}
            for func in ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"]:
                if call_count > 0:
                    time.sleep(delay)
                call_count += 1
                try:
                    resp = requests.get(
                        ALPHA_VANTAGE_URL,
                        params={
                            "function": func,
                            "symbol": ticker,
                            "apikey": ALPHA_VANTAGE_API_KEY,
                        },
                        timeout=15,
                    )
                    data = resp.json()
                    ticker_data[func] = data.get("quarterlyReports", [])
                except Exception as e:
                    logger.debug(f"Failed to fetch {func} for {ticker}: {e}")
                    ticker_data[func] = []

            cache[ticker] = ticker_data
            if (i + 1) % 10 == 0 or i == len(to_fetch) - 1:
                logger.info(f"  Progress: {i + 1}/{len(to_fetch)} tickers")
                _save_historical_cache(cache)  # Save periodically

        _save_historical_cache(cache)
    else:
        logger.info(f"All {len(tickers)} tickers loaded from historical cache.")

    # Build per-quarter DataFrames
    # First, collect all quarter dates across all tickers
    quarter_rows: dict[str, list[dict]] = {}

    for ticker in tickers:
        td = cache.get(ticker, {})
        inc_reports = {r["fiscalDateEnding"]: r for r in td.get("INCOME_STATEMENT", [])}
        bs_reports = {r["fiscalDateEnding"]: r for r in td.get("BALANCE_SHEET", [])}
        cf_reports = {r["fiscalDateEnding"]: r for r in td.get("CASH_FLOW", [])}

        # Use dates where we have at least income statement data
        for qdate in inc_reports:
            inc = inc_reports.get(qdate, {})
            bs = bs_reports.get(qdate, {})
            cf = cf_reports.get(qdate, {})

            if not inc:
                continue

            row = _build_quarter_row(ticker, inc, bs, cf)
            if qdate not in quarter_rows:
                quarter_rows[qdate] = []
            quarter_rows[qdate].append(row)

    # Convert to DataFrames and add sector from OVERVIEW cache
    overview_cache = (
        _load_disk_cache()
    )  # Already cached from fetch_fundamentals_for_scoring
    result: dict[str, pd.DataFrame] = {}
    for qdate, rows in quarter_rows.items():
        df = pd.DataFrame(rows).set_index("ticker")
        # Add sector info from OVERVIEW cache (sector doesn't change historically)
        for ticker in df.index:
            ov = overview_cache.get(ticker)
            if ov:
                df.loc[ticker, "sector"] = ov.get("Sector", "")
        result[qdate] = df

    logger.info(
        f"Historical fundamentals: {len(result)} quarters, " f"{len(tickers)} tickers"
    )
    return result


def get_fundamentals_for_date(
    historical: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """
    Get the most recent fundamentals available as of a given date.
    Uses a 3-month reporting lag (quarter ending Mar 31 available by Jun 30).
    """
    if not historical:
        return None

    # Parse quarter dates and filter to those available before as_of_date
    # Add ~90 day reporting lag
    available = []
    for qdate_str, df in historical.items():
        qdate = pd.Timestamp(qdate_str)
        # Assume data available ~90 days after quarter end
        available_date = qdate + pd.Timedelta(days=90)
        if available_date <= as_of_date:
            available.append((qdate, df))

    if not available:
        return None

    # Return the most recent quarter
    available.sort(key=lambda x: x[0], reverse=True)
    return available[0][1]


if __name__ == "__main__":
    print(f"API Key loaded: {'Yes' if ALPHA_VANTAGE_API_KEY else 'No'}")
    print("Fetching SPY monthly data from Alpha Vantage...")
    df = fetch_monthly_adjusted_close("SPY")
    if df.empty:
        print("ERROR: No data returned for SPY")
    else:
        print(f"Success! {len(df)} months of data retrieved.")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(df.tail())
