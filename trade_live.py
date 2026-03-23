"""
trade_live.py — Quarterly paper/live rebalancer.

Runs the full scoring pipeline and submits rebalance orders via Alpaca.
Designed to be invoked by cron (first trading day of March, June, Sept, Dec).

Usage:
    python trade_live.py              # dry-run (prints orders, doesn't submit)
    python trade_live.py --execute    # submits orders to Alpaca
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import config
from strategy.alpha_vantage_fetcher import fetch_fundamentals_for_scoring
from strategy.data_fetcher import fetch_price_data
from strategy.scoring import (
    compute_composite_score,
    compute_fscore,
    compute_momentum_score,
    compute_value_score,
    select_portfolio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trade_live")

DRY_RUN = "--execute" not in sys.argv


def _get_trading_client():
    from alpaca.trading.client import TradingClient

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
    return TradingClient(key, secret, paper=(config.ALPACA_ENVIRONMENT == "paper"))


def rebalance():
    client = _get_trading_client()

    # 1. Account equity -------------------------------------------------------
    account = client.get_account()
    equity = float(account.equity)
    logger.info(f"Account equity: ${equity:,.2f}")

    # 2. Score universe -------------------------------------------------------
    tickers = config.UNIVERSE_TICKERS
    logger.info(f"Fetching prices for {len(tickers)} tickers ...")
    prices = fetch_price_data(tickers)

    logger.info("Fetching fundamentals ...")
    fundamentals = fetch_fundamentals_for_scoring(tickers)

    logger.info("Computing composite scores ...")
    momentum = compute_momentum_score(prices)
    value = compute_value_score(fundamentals)
    fscore = compute_fscore(fundamentals)
    scores = compute_composite_score(momentum, value, fscore)
    portfolio = select_portfolio(scores)

    if portfolio.empty:
        logger.warning("No stocks passed scoring. Exiting without trading.")
        return

    target_tickers = list(portfolio.index)
    logger.info(f"Target portfolio ({len(target_tickers)}): {target_tickers}")

    # 3. Current positions ----------------------------------------------------
    positions = client.get_all_positions()
    current = {p.symbol: float(p.qty) for p in positions}

    # 4. Calculate target shares (equal-weight, fractional) -------------------
    target_value = equity / len(target_tickers)
    latest_prices = prices.iloc[-1]

    targets: dict[str, float] = {}
    for ticker in target_tickers:
        if ticker in latest_prices.index:
            price = float(latest_prices[ticker])
            if price > 0:
                targets[ticker] = round(target_value / price, 4)

    # 5. Compute deltas -------------------------------------------------------
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    sells = []
    buys = []

    # Sell positions no longer in target
    for ticker, qty in current.items():
        if ticker not in targets:
            sells.append((ticker, OrderSide.SELL, qty))

    # Rebalance existing + open new
    for ticker, target_qty in targets.items():
        held = current.get(ticker, 0.0)
        delta = target_qty - held
        threshold = held * config.REBALANCE_THRESHOLD_PCT if held > 0 else 0
        if abs(delta) <= threshold:
            continue
        if delta > 0.001:
            buys.append((ticker, OrderSide.BUY, round(delta, 4)))
        elif delta < -0.001:
            sells.append((ticker, OrderSide.SELL, round(-delta, 4)))

    all_orders = sells + buys  # sells first to free cash
    logger.info(
        f"Orders to submit: {len(all_orders)} ({len(sells)} sells, {len(buys)} buys)"
    )

    if DRY_RUN:
        logger.info("=== DRY RUN — no orders submitted ===")
        for ticker, side, qty in all_orders:
            logger.info(f"  {side.value:4s} {qty:>10.4f} {ticker}")
        return

    # 6. Submit orders --------------------------------------------------------
    submitted = 0
    for ticker, side, qty in all_orders:
        try:
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(req)
            logger.info(f"Submitted: {side.value} {qty:.4f} {ticker}")
            submitted += 1
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Order failed for {ticker}: {e}")

    logger.info(f"Rebalance complete. {submitted}/{len(all_orders)} orders submitted.")


if __name__ == "__main__":
    rebalance()
