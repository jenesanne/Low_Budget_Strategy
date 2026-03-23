"""
Backtester for the Small-Cap Momentum + Value strategy.

Simulates monthly rebalancing on historical price data from Alpaca.
Produces equity curves, risk metrics, and comparison vs benchmark.
"""

import logging

import numpy as np
import pandas as pd

import config
from strategy.risk_management import (
    compute_portfolio_metrics,
)
from strategy.scoring import (
    compute_composite_score,
    compute_fscore,
    compute_momentum_score,
    compute_value_score,
    select_portfolio,
)

logger = logging.getLogger(__name__)


def _enrich_valuations(fund: pd.DataFrame, current_prices: pd.Series) -> pd.DataFrame:
    """Add price-based valuation ratios (P/E, P/S, EV/EBITDA) using current prices."""
    fund = fund.copy()
    for ticker in fund.index:
        if ticker not in current_prices.index:
            continue
        price = float(current_prices[ticker])
        if np.isnan(price) or price <= 0:
            continue

        shares = (
            float(fund.loc[ticker, "shares_outstanding"])
            if "shares_outstanding" in fund.columns
            else np.nan
        )
        if np.isnan(shares) or shares <= 0:
            continue

        market_cap = price * shares
        revenue = (
            float(fund.loc[ticker, "revenue"]) if "revenue" in fund.columns else np.nan
        )
        net_income = (
            float(fund.loc[ticker, "net_income"])
            if "net_income" in fund.columns
            else np.nan
        )
        ebitda = (
            float(fund.loc[ticker, "ebitda"]) if "ebitda" in fund.columns else np.nan
        )

        # P/E = MarketCap / (Net Income × 4) — annualized from quarterly
        if not np.isnan(net_income) and net_income > 0:
            fund.loc[ticker, "pe_ratio"] = market_cap / (net_income * 4)
        # P/S = MarketCap / (Revenue × 4)
        if not np.isnan(revenue) and revenue > 0:
            fund.loc[ticker, "ps_ratio"] = market_cap / (revenue * 4)
        # EV/EBITDA
        total_equity = (
            float(fund.loc[ticker, "total_equity"])
            if "total_equity" in fund.columns
            else np.nan
        )
        d2e = (
            float(fund.loc[ticker, "debt_to_equity"])
            if "debt_to_equity" in fund.columns
            else np.nan
        )
        debt = (
            d2e / 100 * total_equity
            if not np.isnan(d2e) and not np.isnan(total_equity) and total_equity > 0
            else 0
        )
        ev = market_cap + debt
        if not np.isnan(ebitda) and ebitda > 0:
            fund.loc[ticker, "ev_ebitda"] = ev / (ebitda * 4)

    return fund


def run_backtest(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | dict | None = None,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> dict:
    """
    Run a historical backtest of the strategy.

    Args:
        prices: DataFrame of daily close prices (dates × tickers)
        fundamentals: Either:
            - dict[str, DataFrame]: per-quarter historical fundamentals (no look-ahead bias)
            - DataFrame: single snapshot (current fundamentals, has look-ahead bias)
            - None: momentum-only scoring
        initial_capital: Starting capital in USD.

    Returns:
        Dict containing:
        - equity_curve: pd.Series of daily portfolio value
        - returns: pd.Series of daily portfolio returns
        - metrics: dict of performance metrics
        - trades: list of trade records
        - monthly_holdings: dict of {date: [tickers]}
    """
    # Resample to get month-end dates for rebalance points
    monthly_dates = prices.resample("ME").last().index

    # Determine quarterly rebalance months (Mar, Jun, Sep, Dec)
    rebal_freq = getattr(config, "REBALANCE_FREQUENCY", "monthly")
    if rebal_freq == "quarterly":
        rebal_months = {3, 6, 9, 12}
    else:
        rebal_months = set(range(1, 13))

    capital = initial_capital
    holdings = {}  # {ticker: {"shares": float, "entry_price": float}}
    equity_curve = []
    trade_log = []
    monthly_holdings = {}

    # Fetch SPY prices for regime filter if enabled
    regime_bullish = True  # default: always invest
    spy_series = None
    if getattr(config, "USE_REGIME_FILTER", False):
        regime_ticker = getattr(config, "REGIME_TICKER", "SPY")
        if regime_ticker in prices.columns:
            spy_series = prices[regime_ticker]
        else:
            logger.warning(
                f"Regime ticker {regime_ticker} not in price data; regime filter disabled"
            )

    for i, rebal_date in enumerate(monthly_dates):
        # Get price history up to this rebalance date
        hist = prices.loc[:rebal_date]
        if len(hist) < config.MOMENTUM_LOOKBACK // 2:
            # Not enough history yet — stay in cash
            equity_curve.append({"date": rebal_date, "value": capital})
            continue

        current_prices = hist.iloc[-1]

        # ── Market Regime Filter ────────────────────────────────────────
        if spy_series is not None:
            spy_hist = spy_series.loc[:rebal_date].dropna()
            ma_period = getattr(config, "REGIME_MA_PERIOD", 200)
            if len(spy_hist) >= ma_period:
                spy_ma = spy_hist.rolling(ma_period).mean().iloc[-1]
                regime_bullish = spy_hist.iloc[-1] > spy_ma
            else:
                regime_bullish = True  # not enough data yet

        # ── Skip non-rebalance months (just mark-to-market) ─────────────
        if rebal_date.month not in rebal_months:
            end_value = capital
            for ticker, pos in holdings.items():
                if ticker in current_prices.index and not np.isnan(
                    current_prices[ticker]
                ):
                    end_value += pos["shares"] * current_prices[ticker]
            equity_curve.append({"date": rebal_date, "value": end_value})
            monthly_holdings[rebal_date] = list(holdings.keys())
            continue

        # ── Mark-to-market current holdings ──────────────────────────
        portfolio_value = capital  # Uninvested cash
        for ticker, pos in list(holdings.items()):
            if ticker in current_prices.index and not np.isnan(current_prices[ticker]):
                portfolio_value += pos["shares"] * current_prices[ticker]

        # ── Score all stocks ─────────────────────────────────────────
        available_tickers = [t for t in hist.columns if not hist[t].isna().all()]
        momentum = compute_momentum_score(hist[available_tickers])

        # Value & F-Score: resolve fundamentals for this rebalance date
        fund_snapshot = None
        if isinstance(fundamentals, dict):
            # Historical per-quarter fundamentals — pick the right quarter
            from strategy.alpha_vantage_fetcher import get_fundamentals_for_date

            fund_snapshot = get_fundamentals_for_date(fundamentals, rebal_date)

            # Enrich with price-based valuation ratios (P/E, P/S, EV/EBITDA)
            if fund_snapshot is not None:
                fund_snapshot = _enrich_valuations(fund_snapshot, current_prices)
        elif isinstance(fundamentals, pd.DataFrame):
            fund_snapshot = fundamentals

        if fund_snapshot is not None and not fund_snapshot.empty:
            common = [t for t in available_tickers if t in fund_snapshot.index]
            value = (
                compute_value_score(fund_snapshot.loc[common])
                if common
                else pd.Series(dtype=float)
            )
            fscore = (
                compute_fscore(fund_snapshot.loc[common])
                if common
                else pd.Series(dtype=float)
            )
        else:
            value = pd.Series(50, index=momentum.index, name="value_score")
            fscore = pd.Series(50, index=momentum.index, name="fscore_score")

        composite = compute_composite_score(momentum, value, fscore)
        new_portfolio = select_portfolio(composite)

        # ── Sector constraints ───────────────────────────────────────
        max_sector_pct = getattr(config, "MAX_SECTOR_PCT", 1.0)
        if (
            fund_snapshot is not None
            and "sector" in fund_snapshot.columns
            and max_sector_pct < 1.0
        ):
            sector_counts: dict[str, int] = {}
            constrained = []
            for ticker in new_portfolio.index:
                sector = str(
                    fund_snapshot.loc[ticker, "sector"]
                    if ticker in fund_snapshot.index
                    else ""
                )
                if not sector or sector == "":
                    constrained.append(ticker)
                    continue
                max_per_sector = int(len(new_portfolio) * max_sector_pct) + 1
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                if sector_counts[sector] <= max_per_sector:
                    constrained.append(ticker)
            if constrained:
                new_portfolio = new_portfolio.loc[
                    [t for t in constrained if t in new_portfolio.index]
                ]

        # ── Require minimum scored stocks to avoid concentration ────────
        if len(composite) < config.MAX_POSITIONS:
            # Not enough stocks scored — hold current positions
            end_value = capital
            for ticker, pos in holdings.items():
                if ticker in current_prices.index and not np.isnan(
                    current_prices[ticker]
                ):
                    end_value += pos["shares"] * current_prices[ticker]
            equity_curve.append({"date": rebal_date, "value": end_value})
            monthly_holdings[rebal_date] = list(holdings.keys())
            continue

        # ── Regime filter: if bearish, hold current positions but don't rebalance ──
        # (Only applies when we already have positions; otherwise invest normally)
        if not regime_bullish and holdings:
            end_value = capital
            for ticker, pos in holdings.items():
                if ticker in current_prices.index and not np.isnan(
                    current_prices[ticker]
                ):
                    end_value += pos["shares"] * current_prices[ticker]
            equity_curve.append({"date": rebal_date, "value": end_value})
            monthly_holdings[rebal_date] = list(holdings.keys())
            continue

        # ── Determine trades ─────────────────────────────────────────
        current_tickers = set(holdings.keys())
        target_tickers = set(new_portfolio.index)

        # Sell positions no longer in portfolio
        for ticker in list(current_tickers - target_tickers):
            if ticker in current_prices.index and not np.isnan(current_prices[ticker]):
                sell_price = current_prices[ticker]
                proceeds = holdings[ticker]["shares"] * sell_price
                cost = proceeds * config.TRADING_COST_PCT
                capital += proceeds - cost

                trade_log.append(
                    {
                        "date": rebal_date,
                        "ticker": ticker,
                        "action": "SELL",
                        "shares": holdings[ticker]["shares"],
                        "price": sell_price,
                        "value": proceeds,
                        "cost": cost,
                    }
                )
                del holdings[ticker]

        # --- Equal-Weight Position Sizing ---
        weights = pd.Series(1.0 / len(target_tickers), index=list(target_tickers))

        total_value = capital
        for ticker, pos in holdings.items():
            if ticker in current_prices.index:
                total_value += pos["shares"] * current_prices[ticker]

        for ticker in target_tickers:
            if ticker not in current_prices.index or np.isnan(current_prices[ticker]):
                continue
            price = current_prices[ticker]
            if price <= 0:
                continue

            current_value = holdings.get(ticker, {}).get("shares", 0) * price
            target_value = weights[ticker] * total_value
            diff = target_value - current_value

            if abs(diff) < config.MIN_POSITION_SIZE:
                continue  # Skip tiny rebalances

            # Skip if position is within threshold of target (reduces churn)
            rebal_threshold = getattr(config, "REBALANCE_THRESHOLD_PCT", 0.0)
            if target_value > 0 and abs(diff) / target_value < rebal_threshold:
                continue

            if diff > 0 and capital >= config.MIN_POSITION_SIZE:
                # Buy
                buy_amount = min(diff, capital)

                # --- Dynamic Slippage Model ---
                # Slippage = base trading cost + volatility component + liquidity component
                # For investors: This models real-world execution risk, especially in small-caps.
                #   - Volatility: Higher volatility = more slippage
                #   - Liquidity: Lower ADV = more slippage
                lookback = 21
                returns = (
                    hist[ticker].pct_change().dropna()
                    if ticker in hist.columns
                    else pd.Series(dtype=float)
                )
                vol = returns[-lookback:].std() if len(returns) >= lookback else 0.05
                adv = (
                    price * hist[ticker].iloc[-lookback:].mean()
                    if ticker in hist.columns
                    else 1e6
                )
                slippage = (
                    config.TRADING_COST_PCT
                    + config.SLIPPAGE_VOL_MULT * vol
                    + config.SLIPPAGE_LIQUIDITY_MULT * max(0, 1_000_000 - adv) / 100_000
                )
                cost = buy_amount * slippage
                shares_to_buy = (buy_amount - cost) / price
                capital -= buy_amount

                if ticker in holdings:
                    # Average up
                    total_shares = holdings[ticker]["shares"] + shares_to_buy
                    total_cost_basis = (
                        holdings[ticker]["shares"] * holdings[ticker]["entry_price"]
                        + shares_to_buy * price
                    )
                    holdings[ticker] = {
                        "shares": total_shares,
                        "entry_price": total_cost_basis / total_shares,
                    }
                else:
                    holdings[ticker] = {
                        "shares": shares_to_buy,
                        "entry_price": price,
                    }

                trade_log.append(
                    {
                        "date": rebal_date,
                        "ticker": ticker,
                        "action": "BUY",
                        "shares": shares_to_buy,
                        "price": price,
                        "value": buy_amount,
                        "cost": cost,
                        "slippage": slippage,
                        "volatility": vol,
                        "adv": adv,
                    }
                )

        # ── Record equity & holdings ─────────────────────────────────
        end_value = capital
        for ticker, pos in holdings.items():
            if ticker in current_prices.index and not np.isnan(current_prices[ticker]):
                end_value += pos["shares"] * current_prices[ticker]

        equity_curve.append({"date": rebal_date, "value": end_value})
        monthly_holdings[rebal_date] = list(holdings.keys())

    # ── Build results ────────────────────────────────────────────────────
    equity_df = pd.DataFrame(equity_curve).set_index("date")["value"]
    returns = equity_df.pct_change().dropna()
    metrics = compute_portfolio_metrics(returns)

    return {
        "equity_curve": equity_df,
        "returns": returns,
        "metrics": metrics,
        "trades": pd.DataFrame(trade_log),
        "monthly_holdings": monthly_holdings,
    }


def run_benchmark(
    benchmark_prices: pd.Series,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> dict:
    """
    Run a buy-and-hold benchmark for comparison.
    """
    monthly = benchmark_prices.resample("ME").last().dropna()
    shares = initial_capital / monthly.iloc[0]
    equity = monthly * shares
    equity.name = "value"

    returns = equity.pct_change().dropna()
    metrics = compute_portfolio_metrics(returns)

    return {
        "equity_curve": equity,
        "returns": returns,
        "metrics": metrics,
    }


def print_backtest_summary(strategy: dict, benchmark: dict | None = None):
    """Print a formatted summary of backtest results."""
    sm = strategy["metrics"]
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS — Small-Cap Momentum + Value Strategy")
    print("=" * 60)

    print(f"\n  Start Value:       ${config.INITIAL_CAPITAL:,.2f}")
    if strategy["equity_curve"] is not None and len(strategy["equity_curve"]) > 0:
        print(f"  End Value:         ${strategy['equity_curve'].iloc[-1]:,.2f}")
    print(f"  Total Return:      {sm.get('total_return', 0):.1%}")
    print(f"  CAGR:              {sm.get('cagr', 0):.1%}")
    print(f"  Sharpe Ratio:      {sm.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:     {sm.get('sortino_ratio', 0):.2f}")
    print(f"  Max Drawdown:      {sm.get('max_drawdown', 0):.1%}")
    print(f"  Volatility (ann):  {sm.get('annualised_volatility', 0):.1%}")
    print(f"  Win Rate:          {sm.get('win_rate', 0):.1%}")
    print(f"  Trading Days:      {sm.get('n_trading_days', 0)}")

    if strategy["trades"] is not None and len(strategy["trades"]) > 0:
        trades = strategy["trades"]
        print(f"  Total Trades:      {len(trades)}")
        print(f"  Total Costs:       ${trades['cost'].sum():,.2f}")

    if benchmark:
        bm = benchmark["metrics"]
        print(f"\n  {'─' * 40}")
        print(
            f"  Benchmark (IWM):   {bm.get('total_return', 0):.1%} total, "
            f"{bm.get('cagr', 0):.1%} CAGR"
        )
        excess = sm.get("cagr", 0) - bm.get("cagr", 0)
        print(f"  Excess CAGR:       {excess:+.1%}")

    print("=" * 60 + "\n")
