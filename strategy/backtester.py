"""
Backtester for the Small-Cap Momentum + Value strategy.

Daily event-driven simulation with:
- One bar at a time processing (no look-ahead bias)
- Next-bar order execution (orders placed on day T fill at day T+1)
- Trailing stop-loss checked every trading day
- Dynamic volatility-based slippage model
- Daily mark-to-market equity tracking
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
    Run a daily event-driven backtest of the strategy.

    Processes one trading day at a time:
      1. Fill pending orders from yesterday (next-bar execution)
      2. Mark-to-market all positions (daily)
      3. Check trailing stop-losses (daily)
      4. On rebalance days: score stocks, queue orders for tomorrow
      5. Record daily portfolio value

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
    prices = prices.sort_index()
    trading_days = prices.index

    # Determine quarterly rebalance months (Mar, Jun, Sep, Dec)
    rebal_freq = getattr(config, "REBALANCE_FREQUENCY", "monthly")
    rebal_months = {3, 6, 9, 12} if rebal_freq == "quarterly" else set(range(1, 13))
    rebal_day = getattr(config, "REBALANCE_DAY", 1)
    trailing_stop = getattr(config, "TRAILING_STOP", True)

    capital = initial_capital
    # holdings: {ticker: {"shares": float, "cost_basis": float, "high_water": float}}
    holdings = {}
    equity_curve = []
    trade_log = []
    monthly_holdings = {}
    pending_orders = []  # list of dicts: {"ticker", "shares", "created_dt"}

    # Regime filter
    use_regime = getattr(config, "USE_REGIME_FILTER", False)
    regime_ticker = getattr(config, "REGIME_TICKER", "SPY")
    regime_ma_period = getattr(config, "REGIME_MA_PERIOD", 200)

    def portfolio_value(current_prices):
        value = capital
        for tkr, pos in holdings.items():
            if tkr in current_prices.index and not np.isnan(current_prices[tkr]):
                value += pos["shares"] * current_prices[tkr]
        return value

    def is_rebalance_day(dt):
        if dt.month not in rebal_months:
            return False
        month_days = trading_days[
            (trading_days.month == dt.month) & (trading_days.year == dt.year)
        ]
        if len(month_days) == 0:
            return False
        day_idx = min(rebal_day - 1, len(month_days) - 1)
        return dt == month_days[day_idx]

    def check_regime(dt):
        if not use_regime or regime_ticker not in prices.columns:
            return True
        spy = prices[regime_ticker].loc[:dt].dropna()
        if len(spy) < regime_ma_period:
            return True
        ma = spy.rolling(regime_ma_period).mean().iloc[-1]
        return float(spy.iloc[-1]) > float(ma)

    def compute_slippage(ticker, price, dt):
        lookback = 21
        if ticker in prices.columns:
            hist_ticker = prices[ticker].loc[:dt].dropna()
            rets = hist_ticker.pct_change().dropna()
            vol = rets.iloc[-lookback:].std() if len(rets) >= lookback else 0.05
            adv = (
                float(hist_ticker.iloc[-lookback:].mean()) * 50_000
                if len(hist_ticker) >= lookback
                else 1e6
            )
        else:
            vol, adv = 0.05, 1e6
        return (
            config.TRADING_COST_PCT
            + config.SLIPPAGE_VOL_MULT * vol
            + config.SLIPPAGE_LIQUIDITY_MULT * max(0, 1_000_000 - adv) / 100_000
        )

    logger.info(
        f"Backtest: {len(trading_days)} trading days, "
        f"${initial_capital:,.0f} initial capital, "
        f"trailing_stop={'ON' if trailing_stop else 'OFF'}"
    )

    for dt in trading_days:
        current_prices = prices.loc[dt]

        # ── 1. Fill pending orders (next-bar execution) ──────────────
        filled = []
        for order in pending_orders:
            ticker = order["ticker"]
            if ticker not in current_prices.index or np.isnan(current_prices[ticker]):
                continue
            price = float(current_prices[ticker])
            slippage_pct = compute_slippage(ticker, price, dt)

            if order["shares"] > 0:
                # BUY — slippage pushes price up
                fill_price = price * (1 + slippage_pct)
                total_cost = order["shares"] * fill_price
                if total_cost > capital:
                    order["shares"] = (capital * 0.99) / fill_price
                    if order["shares"] < 0.01:
                        filled.append(order)
                        continue
                    total_cost = order["shares"] * fill_price

                capital -= total_cost

                if ticker in holdings:
                    old = holdings[ticker]
                    new_shares = old["shares"] + order["shares"]
                    new_cost = (
                        old["shares"] * old["cost_basis"] + order["shares"] * fill_price
                    )
                    holdings[ticker] = {
                        "shares": new_shares,
                        "cost_basis": new_cost / new_shares,
                        "high_water": max(
                            old.get("high_water", fill_price), fill_price
                        ),
                    }
                else:
                    holdings[ticker] = {
                        "shares": order["shares"],
                        "cost_basis": fill_price,
                        "high_water": fill_price,
                    }

                trade_log.append(
                    {
                        "date": dt,
                        "ticker": ticker,
                        "action": "BUY",
                        "shares": order["shares"],
                        "price": price,
                        "fill_price": round(fill_price, 4),
                        "value": round(total_cost, 2),
                        "cost": round(total_cost - order["shares"] * price, 2),
                        "slippage_pct": round(slippage_pct, 6),
                    }
                )
            else:
                # SELL — slippage pushes price down
                sell_shares = abs(order["shares"])
                fill_price = price * (1 - slippage_pct)
                proceeds = sell_shares * fill_price
                capital += proceeds

                if ticker in holdings:
                    holdings[ticker]["shares"] -= sell_shares
                    if holdings[ticker]["shares"] <= 0.001:
                        del holdings[ticker]

                trade_log.append(
                    {
                        "date": dt,
                        "ticker": ticker,
                        "action": "SELL",
                        "shares": sell_shares,
                        "price": price,
                        "fill_price": round(fill_price, 4),
                        "value": round(proceeds, 2),
                        "cost": round(sell_shares * price - proceeds, 2),
                        "slippage_pct": round(slippage_pct, 6),
                    }
                )

            filled.append(order)

        pending_orders = [o for o in pending_orders if o not in filled]

        # ── 2. Update high-water marks for trailing stops ────────────
        for ticker, pos in holdings.items():
            if ticker in current_prices.index and not np.isnan(current_prices[ticker]):
                price = float(current_prices[ticker])
                if price > pos.get("high_water", 0):
                    pos["high_water"] = price

        # ── 3. Check trailing stop-losses (daily) ───────────────────
        pending_sell_tickers = {o["ticker"] for o in pending_orders if o["shares"] < 0}
        for ticker, pos in list(holdings.items()):
            if ticker in pending_sell_tickers:
                continue
            if ticker not in current_prices.index or np.isnan(current_prices[ticker]):
                continue
            price = float(current_prices[ticker])

            if trailing_stop:
                # Trailing stop: measure decline from high-water mark
                hwm = pos.get("high_water", pos["cost_basis"])
                pct_from_peak = (price - hwm) / hwm
            else:
                # Fixed stop: measure decline from entry cost basis
                pct_from_peak = (price - pos["cost_basis"]) / pos["cost_basis"]

            if pct_from_peak <= config.STOP_LOSS_PCT:
                ref_price = (
                    pos.get("high_water", pos["cost_basis"])
                    if trailing_stop
                    else pos["cost_basis"]
                )
                logger.info(
                    f"{dt.date()}: TRAILING-STOP {ticker} at {pct_from_peak:.1%} "
                    f"(peak=${ref_price:.2f}, now=${price:.2f})"
                )
                pending_orders.append(
                    {
                        "ticker": ticker,
                        "shares": -pos["shares"],
                        "created_dt": dt,
                    }
                )

        # ── 4. Rebalance if scheduled ────────────────────────────────
        if is_rebalance_day(dt):
            hist = prices.loc[:dt]
            if len(hist) >= config.MOMENTUM_LOOKBACK // 2:
                regime_ok = check_regime(dt)

                if not regime_ok and holdings:
                    logger.info(f"{dt.date()}: Bearish regime — skipping rebalance")
                else:
                    # Score stocks
                    scoring_tickers = [
                        t
                        for t in hist.columns
                        if t != regime_ticker and not hist[t].isna().all()
                    ]
                    momentum = compute_momentum_score(hist[scoring_tickers])

                    fund_snapshot = None
                    if isinstance(fundamentals, dict):
                        from strategy.alpha_vantage_fetcher import (
                            get_fundamentals_for_date,
                        )

                        fund_snapshot = get_fundamentals_for_date(fundamentals, dt)
                        if fund_snapshot is not None:
                            fund_snapshot = _enrich_valuations(
                                fund_snapshot, current_prices
                            )
                    elif isinstance(fundamentals, pd.DataFrame):
                        fund_snapshot = fundamentals

                    if fund_snapshot is not None and not fund_snapshot.empty:
                        common = [
                            t for t in scoring_tickers if t in fund_snapshot.index
                        ]
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
                        fscore = pd.Series(
                            50, index=momentum.index, name="fscore_score"
                        )

                    composite = compute_composite_score(momentum, value, fscore)

                    if len(composite) >= config.MAX_POSITIONS:
                        new_portfolio = select_portfolio(composite)

                        # Sector constraints
                        max_sector_pct = getattr(config, "MAX_SECTOR_PCT", 1.0)
                        if (
                            fund_snapshot is not None
                            and "sector" in fund_snapshot.columns
                            and max_sector_pct < 1.0
                        ):
                            sector_counts = {}
                            constrained = []
                            for tkr in new_portfolio.index:
                                sector = str(
                                    fund_snapshot.loc[tkr, "sector"]
                                    if tkr in fund_snapshot.index
                                    else ""
                                )
                                if not sector:
                                    constrained.append(tkr)
                                    continue
                                max_per = int(len(new_portfolio) * max_sector_pct) + 1
                                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                                if sector_counts[sector] <= max_per:
                                    constrained.append(tkr)
                            if constrained:
                                new_portfolio = new_portfolio.loc[
                                    [t for t in constrained if t in new_portfolio.index]
                                ]

                        target_tickers = set(new_portfolio.index)
                        current_tickers = set(holdings.keys())
                        pending_sell_tickers = {
                            o["ticker"] for o in pending_orders if o["shares"] < 0
                        }

                        # Sell positions no longer in target
                        for ticker in current_tickers - target_tickers:
                            if ticker in pending_sell_tickers:
                                continue
                            if ticker in holdings:
                                pending_orders.append(
                                    {
                                        "ticker": ticker,
                                        "shares": -holdings[ticker]["shares"],
                                        "created_dt": dt,
                                    }
                                )

                        # Equal-weight buy/rebalance
                        total_val = portfolio_value(current_prices)
                        target_weight = (
                            1.0 / len(target_tickers) if target_tickers else 0
                        )

                        for ticker in target_tickers:
                            if ticker not in current_prices.index or np.isnan(
                                current_prices[ticker]
                            ):
                                continue
                            price = float(current_prices[ticker])
                            if price <= 0:
                                continue

                            current_shares = (
                                holdings[ticker]["shares"] if ticker in holdings else 0
                            )
                            current_val = current_shares * price
                            target_val = target_weight * total_val
                            diff = target_val - current_val

                            rebal_threshold = getattr(
                                config, "REBALANCE_THRESHOLD_PCT", 0.0
                            )
                            if (
                                target_val > 0
                                and abs(diff) / target_val < rebal_threshold
                            ):
                                continue
                            if abs(diff) < config.MIN_POSITION_SIZE:
                                continue

                            shares_diff = diff / price
                            pending_orders.append(
                                {
                                    "ticker": ticker,
                                    "shares": shares_diff,
                                    "created_dt": dt,
                                }
                            )

                        logger.info(
                            f"{dt.date()}: Queued rebalance → target: {sorted(target_tickers)}"
                        )

                        monthly_holdings[dt] = list(target_tickers)

        # ── 5. Record daily equity ───────────────────────────────────
        pv = portfolio_value(current_prices)
        equity_curve.append({"date": dt, "value": pv})

    # ── Build results ────────────────────────────────────────────────────
    equity_df = pd.DataFrame(equity_curve).set_index("date")["value"]
    returns = equity_df.pct_change().dropna()
    metrics = compute_portfolio_metrics(returns, periods_per_year=252)

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
    Run a buy-and-hold benchmark for comparison (daily granularity).
    """
    daily = benchmark_prices.dropna()
    shares = initial_capital / daily.iloc[0]
    equity = daily * shares
    equity.name = "value"

    returns = equity.pct_change().dropna()
    metrics = compute_portfolio_metrics(returns, periods_per_year=252)

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
