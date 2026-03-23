"""
walk_forward.py

Implements rolling walk-forward (out-of-sample) backtesting for the Low-Budget Small-Cap Strategy.
Splits the historical period into sequential train/test windows, runs the strategy, and collects results.
"""

import pandas as pd

import config
from strategy.backtester import run_backtest
from strategy.data_fetcher import fetch_price_data


def walk_forward_backtest(
    tickers,
    start,
    end,
    train_years=3,
    test_years=1,
    initial_capital=1000,
):
    """
    Runs a rolling walk-forward backtest.
    For each window:
      - Train on train_years (for parameter fit, if needed)
      - Test on next test_years (out-of-sample)
    Returns a DataFrame of concatenated equity curves and a summary of each window.
    """
    # Download all price data up front
    prices = fetch_price_data(tickers, start=start, end=end)
    if prices.empty:
        raise ValueError("No price data for walk-forward test.")
    results = []
    window_summaries = []
    dates = prices.index
    min_date = dates.min()
    max_date = dates.max()
    train_days = int(train_years * 252)
    test_days = int(test_years * 252)
    i = 0
    while True:
        train_start = min_date + pd.Timedelta(days=i * test_days)
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > max_date:
            break
        # Use only test window for out-of-sample equity
        test_prices = prices[(prices.index >= test_start) & (prices.index < test_end)]
        if test_prices.empty:
            break
        # Run backtest on test window
        test_result = run_backtest(
            test_prices, fundamentals=None, initial_capital=initial_capital
        )
        eq = test_result["equity_curve"].copy()
        eq.name = f"window_{i+1}"
        results.append(eq)
        window_summaries.append(
            {
                "window": i + 1,
                "test_start": test_start,
                "test_end": test_end,
                "final_value": eq.iloc[-1],
                "total_return": (eq.iloc[-1] / eq.iloc[0]) - 1,
            }
        )
        i += 1
    # Concatenate all equity curves
    all_equity = pd.concat(results, axis=1)
    summary = pd.DataFrame(window_summaries)
    return all_equity, summary


if __name__ == "__main__":
    tickers = config.UNIVERSE_TICKERS
    all_equity, summary = walk_forward_backtest(
        tickers,
        start=config.BACKTEST_START,
        end=config.BACKTEST_END,
        train_years=3,
        test_years=1,
        initial_capital=config.INITIAL_CAPITAL,
    )
    print(summary)
    all_equity.to_csv("output/walk_forward_equity.csv")
    summary.to_csv("output/walk_forward_summary.csv", index=False)
