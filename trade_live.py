"""
trade_live.py — Monthly paper/live rebalancer.

Runs the full scoring pipeline and submits rebalance orders via Alpaca.
Designed to be invoked by cron on the 1st trading day of each month.

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
    compute_trend_score,
    compute_value_score,
    compute_volume_score,
    passes_trend_template,
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
    capital = config.INITIAL_CAPITAL
    logger.info(f"Account equity: ${equity:,.2f}  |  Strategy capital: ${capital:,.2f}")

    # 2. Regime filter — check if market is bullish -------------------------
    if config.USE_REGIME_FILTER:
        logger.info("Checking market regime (SPY vs 200-day MA) ...")
        spy_prices = fetch_price_data([config.REGIME_TICKER])
        if not spy_prices.empty and config.REGIME_TICKER in spy_prices.columns:
            spy = spy_prices[config.REGIME_TICKER].dropna()
            if len(spy) >= config.REGIME_MA_PERIOD:
                ma = spy.rolling(config.REGIME_MA_PERIOD).mean().iloc[-1]
                current_spy = float(spy.iloc[-1])
                pct_from_ma = (current_spy - ma) / ma
                regime_exit = getattr(config, "REGIME_EXIT_PCT", -0.02)
                if pct_from_ma <= regime_exit:
                    logger.warning(
                        f"BEARISH REGIME: SPY ${current_spy:.2f} is "
                        f"{pct_from_ma:.1%} vs 200-day MA ${ma:.2f}. "
                        f"Selling all positions and going to cash."
                    )
                    # Sell all current positions
                    positions = client.get_all_positions()
                    if positions and not DRY_RUN:
                        from alpaca.trading.enums import OrderSide, TimeInForce
                        from alpaca.trading.requests import MarketOrderRequest

                        for p in positions:
                            try:
                                req = MarketOrderRequest(
                                    symbol=p.symbol,
                                    qty=float(p.qty),
                                    side=OrderSide.SELL,
                                    time_in_force=TimeInForce.DAY,
                                )
                                client.submit_order(req)
                                logger.info(f"REGIME SELL: {p.qty} {p.symbol}")
                                time.sleep(0.5)
                            except Exception as e:
                                logger.error(f"Regime sell failed for {p.symbol}: {e}")
                    elif positions and DRY_RUN:
                        for p in positions:
                            logger.info(f"  DRY-RUN REGIME SELL: {p.qty} {p.symbol}")
                    return
                else:
                    logger.info(
                        f"BULLISH REGIME: SPY ${current_spy:.2f} is "
                        f"{pct_from_ma:+.1%} vs 200-day MA ${ma:.2f}. Proceeding."
                    )

    # 3. Score universe -------------------------------------------------------
    tickers = config.UNIVERSE_TICKERS
    logger.info(f"Fetching prices for {len(tickers)} tickers ...")
    prices = fetch_price_data(tickers)

    logger.info("Fetching fundamentals ...")
    fundamentals = fetch_fundamentals_for_scoring(tickers)

    logger.info("Computing Elite Trader Momentum scores ...")
    momentum = compute_momentum_score(prices)
    trend = compute_trend_score(prices)
    volume = compute_volume_score(prices)
    trend_filter = passes_trend_template(prices)
    value = compute_value_score(fundamentals)
    fscore = compute_fscore(fundamentals)
    scores = compute_composite_score(
        momentum, value, fscore, trend=trend, volume=volume
    )
    portfolio = select_portfolio(scores, trend_filter=trend_filter)

    if portfolio.empty:
        logger.warning("No stocks passed scoring. Exiting without trading.")
        return

    target_tickers = list(portfolio.index)
    logger.info(f"Target portfolio ({len(target_tickers)}): {target_tickers}")

    # 4. Current positions ----------------------------------------------------
    positions = client.get_all_positions()
    current = {p.symbol: float(p.qty) for p in positions}
    # Unrealized P&L pct per symbol (Alpaca reports as decimal, e.g. -0.08 = -8%)
    pnl_pct = {p.symbol: float(p.unrealized_plpc) for p in positions}

    # 4a. Asymmetric cut-losers / let-winners-run -----------------------------
    cut_threshold = getattr(config, "CUT_LOSER_PCT", -0.08)
    keep_threshold = getattr(config, "KEEP_WINNER_PCT", 0.05)
    forced_sells: dict[str, float] = {}  # symbol -> qty (full liquidation)
    protected_winners: set[str] = set()  # held even if not in target list

    for sym, qty in current.items():
        pnl = pnl_pct.get(sym, 0.0)
        if pnl <= cut_threshold:
            forced_sells[sym] = qty
            logger.info(f"CUT LOSER: {sym} P&L {pnl:+.1%} <= {cut_threshold:+.1%}")
        elif pnl >= keep_threshold:
            protected_winners.add(sym)
            logger.info(
                f"LET WINNER RUN: {sym} P&L {pnl:+.1%} >= {keep_threshold:+.1%}"
            )

    # 5. Calculate target shares (equal-weight, fractional) -------------------
    target_value = capital / len(target_tickers)
    latest_prices = prices.iloc[-1]

    targets: dict[str, float] = {}
    for ticker in target_tickers:
        if ticker in latest_prices.index:
            price = float(latest_prices[ticker])
            if price > 0:
                targets[ticker] = round(target_value / price, 4)

    # 6. Compute deltas -------------------------------------------------------
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    sells = []
    buys = []

    # Forced sells (losers) — always liquidate, removes from any further consideration
    for ticker, qty in forced_sells.items():
        sells.append((ticker, OrderSide.SELL, qty))
        targets.pop(ticker, None)

    # Sell positions no longer in target — UNLESS they're protected winners
    for ticker, qty in current.items():
        if ticker in forced_sells:
            continue
        if ticker not in targets and ticker not in protected_winners:
            sells.append((ticker, OrderSide.SELL, qty))

    # Rebalance existing + open new
    for ticker, target_qty in targets.items():
        if ticker in forced_sells:
            continue
        held = current.get(ticker, 0.0)
        delta = target_qty - held
        # Don't trim winners — only allow scale-ups for protected winners
        if ticker in protected_winners and delta < 0:
            continue
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

    # 7. Submit orders --------------------------------------------------------
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
