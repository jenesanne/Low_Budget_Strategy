import logging
import sys

import pandas as pd

import config
from strategy.alpha_vantage_fetcher import (
    fetch_fundamentals_for_scoring,
    fetch_historical_fundamentals,
)
from strategy.backtester import print_backtest_summary, run_backtest, run_benchmark
from strategy.data_fetcher import (
    fetch_latest_prices,
    fetch_price_data,
)
from strategy.risk_management import compute_position_sizes
from strategy.scoring import (
    compute_composite_score,
    compute_momentum_score,
    select_portfolio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_backtest():
    """Run historical backtest on the configured universe."""
    logger.info("Starting backtest...")
    tickers = config.UNIVERSE_TICKERS

    # Include regime filter ticker (SPY) in the universe for the backtester
    regime_ticker = getattr(config, "REGIME_TICKER", None)
    if regime_ticker and regime_ticker not in tickers:
        tickers = tickers + [regime_ticker]

    # Fetch price data
    prices = fetch_price_data(tickers)
    if prices.empty:
        logger.error(
            "No price data retrieved — check your Alpaca API keys and ticker list"
        )
        return

    logger.info(f"Price data: {prices.shape[0]} days × {prices.shape[1]} tickers")

    # Fetch historical fundamentals from Alpha Vantage (per-quarter, no look-ahead bias)
    scoring_tickers = [
        t for t in prices.columns if t != getattr(config, "REGIME_TICKER", "SPY")
    ]
    # First fetch OVERVIEW (for sector data, gets cached to disk)
    _ = fetch_fundamentals_for_scoring(scoring_tickers)
    # Then fetch historical IS/BS/CF
    fundamentals = fetch_historical_fundamentals(scoring_tickers)
    if not fundamentals:
        logger.warning(
            "No historical fundamentals retrieved — trying current snapshot..."
        )
        fundamentals = fetch_fundamentals_for_scoring(scoring_tickers)
        if isinstance(fundamentals, pd.DataFrame) and fundamentals.empty:
            fundamentals = None

    # Fetch benchmark
    benchmark_prices = fetch_price_data(
        [config.BENCHMARK_TICKER],
        start=config.BACKTEST_START,
        end=config.BACKTEST_END,
    )

    # Run strategy backtest
    results = run_backtest(
        prices, fundamentals=fundamentals, initial_capital=config.INITIAL_CAPITAL
    )

    # Run benchmark
    benchmark = None
    if (
        not benchmark_prices.empty
        and config.BENCHMARK_TICKER in benchmark_prices.columns
    ):
        benchmark = run_benchmark(benchmark_prices[config.BENCHMARK_TICKER])

    # Print results
    print_backtest_summary(results, benchmark)

    # Save trade log
    if results["trades"] is not None and len(results["trades"]) > 0:
        results["trades"].to_csv("output/backtest_trades.csv", index=False)
        logger.info("Trade log saved to output/backtest_trades.csv")

    # Save equity curve
    results["equity_curve"].to_csv("output/backtest_equity.csv")
    logger.info("Equity curve saved to output/backtest_equity.csv")


def cmd_screen():
    """Screen current universe and display top picks."""
    logger.info("Screening current universe...")
    tickers = config.UNIVERSE_TICKERS

    # Fetch recent price history for momentum scoring
    prices = fetch_price_data(tickers, start="2024-01-01", end="2026-12-31")
    if prices.empty:
        logger.error("No price data — check API keys")
        return

    # Compute momentum scores from price history
    momentum = compute_momentum_score(prices)

    # Try to fetch Alpha Vantage fundamentals
    try:
        from alpha_vantage_fetcher import fetch_fundamentals

        logger.info("Fetching Alpha Vantage fundamentals for value and F-Score...")
        fundamentals = fetch_fundamentals(list(momentum.index))
        # Build value and fscore proxies from Alpha Vantage data
        value = pd.Series(index=momentum.index, dtype=float, name="value_score")
        fscore = pd.Series(index=momentum.index, dtype=float, name="fscore_score")
        for ticker, data in fundamentals.items():
            if not data:
                continue
            try:
                pe = float(data.get("PERatio", 0)) or 1000
                ps = float(data.get("PriceToSalesRatioTTM", 0)) or 1000
                roe = float(data.get("ReturnOnEquityTTM", 0))
                # Value: lower P/E and P/S is better, average percentile
                value.loc[ticker] = (
                    100 - pd.Series([pe, ps]).rank(pct=True).mean() * 100
                )
                # F-Score proxy: high ROE is good, scale to 0-100
                fscore.loc[ticker] = min(max(roe, 0), 50) * 2  # Cap at 50% ROE
            except Exception:
                value.loc[ticker] = 50
                fscore.loc[ticker] = 50
        value = value.fillna(50)
        fscore = fscore.fillna(50)
    except Exception as e:
        logger.warning(f"Alpha Vantage fundamentals unavailable: {e}")
        value = pd.Series(50, index=momentum.index, name="value_score")
        fscore = pd.Series(50, index=momentum.index, name="fscore_score")

    composite = compute_composite_score(momentum, value, fscore)
    portfolio = select_portfolio(composite)

    # Get latest prices for position sizing
    latest = fetch_latest_prices(list(portfolio.index))

    # Position sizing
    sized = compute_position_sizes(config.INITIAL_CAPITAL, portfolio)
    if not latest.empty:
        for ticker in sized.index:
            if ticker in latest.index:
                sized.loc[ticker, "price"] = latest[ticker]
                sized.loc[ticker, "shares"] = (
                    sized.loc[ticker, "dollar_amount"] / latest[ticker]
                )

    print("\n" + "=" * 70)
    print("  TOP PICKS — Small-Cap Momentum + Value + Fundamentals Screen")
    print("=" * 70)
    print(f"  Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    print(f"  Budget: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Positions: {len(sized)}")
    print(f"  {'─' * 50}")

    display_cols = ["composite", "momentum", "value", "fscore"]
    if "price" in sized.columns:
        display_cols.append("price")
    if "shares" in sized.columns:
        display_cols.append("shares")
    display_cols.append("dollar_amount")

    available_cols = [c for c in display_cols if c in sized.columns]
    print(sized[available_cols].to_string())
    print("=" * 70 + "\n")


def cmd_status():
    """Show status of the Alpaca paper trading account."""
    from data_fetcher import _get_alpaca_clients

    _, trading_client = _get_alpaca_clients()
    account = trading_client.get_account()

    print("\n" + "=" * 50)
    print("  ALPACA ACCOUNT STATUS")
    print("=" * 50)
    print(f"  Account ID:     {account.id}")
    print(f"  Status:         {account.status}")
    print(f"  Cash:           ${float(account.cash):,.2f}")
    print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"  Buying Power:   ${float(account.buying_power):,.2f}")
    print(f"  Day Trades:     {account.daytrade_count}/3")
    print(f"  PDT Restricted: {account.pattern_day_trader}")
    print("=" * 50 + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()
    commands = {
        "backtest": cmd_backtest,
        "screen": cmd_screen,
        "status": cmd_status,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(commands.keys())}")


if __name__ == "__main__":
    main()
