"""
monitor_stops.py — Daily ATR-based trailing stop checker.

Designed to run every weekday after market close. For each open Alpaca position:
  1. Fetch ~60 days of daily bars (enough for 20-day ATR).
  2. Compute ATR(20) and the high-water mark since entry.
  3. Stop level = HWM - (ATR_STOP_MULT × ATR).
  4. Also check the absolute safety net: STOP_LOSS_PCT from peak.
  5. If current price <= stop, submit a market sell.

This brings the live trader in line with the backtester's risk model so positions
don't free-fall between monthly rebalances.

Usage:
    python monitor_stops.py              # dry-run (logs only)
    python monitor_stops.py --execute    # submit sell orders to Alpaca
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitor_stops")

DRY_RUN = "--execute" not in sys.argv
STATE_FILE = Path(__file__).parent / "output" / "position_state.json"


def _get_trading_client():
    from alpaca.trading.client import TradingClient

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
    return TradingClient(key, secret, paper=(config.ALPACA_ENVIRONMENT == "paper"))


def _fetch_recent_bars(symbols: list[str], lookback_days: int = 60):
    """Fetch ~lookback_days of daily bars for stop calculations."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    data_client = StockHistoricalDataClient(key, secret)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days * 2)  # calendar days, 2x for weekends
    feed = getattr(config, "ALPACA_DATA_FEED", "iex")
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=feed,
    )
    bars = data_client.get_stock_bars(req)
    return bars.df  # MultiIndex (symbol, timestamp)


def _compute_atr(closes, period: int) -> float | None:
    """Close-to-close ATR proxy (matches backtester's compute_atr)."""
    if len(closes) < period + 1:
        return None
    tr = closes.diff().abs().iloc[-period:]
    return float(tr.mean())


def _load_state() -> dict:
    """Load persistent per-position state: {symbol: {hwm: float, entry_date: str}}."""
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception as e:
        logger.warning(f"Could not read state file: {e}")
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def check_stops():
    client = _get_trading_client()

    positions = client.get_all_positions()
    if not positions:
        logger.info("No open positions.")
        return

    # Skip non-equity symbols (crypto, etc.) — this monitor is for stocks only.
    equity_positions = [
        p for p in positions if "/" not in p.symbol and "USD" not in p.symbol
    ]
    skipped = [p.symbol for p in positions if p not in equity_positions]
    if skipped:
        logger.info(f"Skipping non-equity symbols: {skipped}")
    if not equity_positions:
        logger.info("No equity positions to monitor.")
        return

    symbols = [p.symbol for p in equity_positions]
    logger.info(f"Checking stops for {len(symbols)} positions: {symbols}")

    # Pull recent bars for ATR + HWM
    bars = _fetch_recent_bars(symbols, lookback_days=60)
    if bars.empty:
        logger.warning("No bar data returned — cannot evaluate stops.")
        return

    atr_period = getattr(config, "ATR_PERIOD", 20)
    atr_mult = getattr(config, "ATR_STOP_MULT", 6)
    safety_pct = getattr(config, "STOP_LOSS_PCT", -0.20)
    trailing = getattr(config, "TRAILING_STOP", True)

    state = _load_state()
    today = datetime.now(timezone.utc).date().isoformat()

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    sells: list[tuple[str, float, str]] = []  # (symbol, qty, reason)

    for p in equity_positions:
        sym = p.symbol
        try:
            qty = float(p.qty)
            cost_basis = float(p.avg_entry_price)
            current = float(p.current_price)
        except (TypeError, ValueError):
            logger.warning(f"{sym}: missing price/qty fields, skipping")
            continue

        if sym not in bars.index.get_level_values("symbol"):
            logger.warning(f"{sym}: no bar data, skipping")
            continue

        closes = bars.xs(sym, level="symbol")["close"]
        if len(closes) < atr_period + 1:
            logger.warning(f"{sym}: only {len(closes)} bars, need {atr_period + 1}")
            continue

        # Persistent high-water mark: max(stored_hwm, recent closes, current, cost).
        # First time we see a symbol, seed HWM from cost basis (conservative).
        prior = state.get(sym, {})
        prior_hwm = float(prior.get("hwm", cost_basis))
        hwm = float(max(prior_hwm, closes.max(), current, cost_basis))
        state[sym] = {
            "hwm": hwm,
            "entry_date": prior.get("entry_date", today),
            "last_seen": today,
            "cost_basis": cost_basis,
        }
        atr = _compute_atr(closes, atr_period)

        triggered = False
        reason = ""

        if trailing and atr is not None and atr > 0 and atr_mult > 0:
            stop_level = hwm - atr_mult * atr
            if current <= stop_level:
                triggered = True
                reason = (
                    f"ATR-STOP HWM=${hwm:.2f} - {atr_mult}×ATR(${atr:.2f}) "
                    f"= ${stop_level:.2f}, now=${current:.2f}"
                )

        # Safety net from peak
        pct_from_peak = (current - hwm) / hwm if hwm > 0 else 0
        if pct_from_peak <= safety_pct:
            triggered = True
            reason = f"SAFETY-STOP {pct_from_peak:+.1%} from peak (${hwm:.2f} → ${current:.2f})"

        if triggered:
            logger.info(f"{sym}: STOP TRIGGERED — {reason}")
            sells.append((sym, qty, reason))
        else:
            stop_str = f"${hwm - atr_mult * atr:.2f}" if atr else "n/a"
            logger.info(
                f"{sym}: OK (price ${current:.2f}, HWM ${hwm:.2f}, stop {stop_str})"
            )

    if not sells:
        logger.info("No stops triggered.")
        # Prune state for symbols no longer held
        held = {p.symbol for p in equity_positions}
        state = {k: v for k, v in state.items() if k in held}
        _save_state(state)
        return

    if DRY_RUN:
        logger.info("=== DRY RUN — no sell orders submitted ===")
        for sym, qty, reason in sells:
            logger.info(f"  DRY-RUN SELL {qty} {sym}  ({reason})")
        # Still persist updated HWMs even on dry-run
        held = {p.symbol for p in equity_positions}
        state = {k: v for k, v in state.items() if k in held}
        _save_state(state)
        return

    submitted = 0
    for sym, qty, reason in sells:
        try:
            req = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(req)
            logger.info(f"Submitted SELL {qty} {sym}  ({reason})")
            submitted += 1
            # Drop sold symbol from state
            state.pop(sym, None)
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Sell failed for {sym}: {e}")

    # Prune state for symbols no longer held, save
    held = {p.symbol for p in equity_positions} - {s for s, _, _ in sells}
    state = {k: v for k, v in state.items() if k in held}
    _save_state(state)

    logger.info(f"Stop monitor complete. {submitted}/{len(sells)} orders submitted.")


if __name__ == "__main__":
    check_stops()
