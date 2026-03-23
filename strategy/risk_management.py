"""
Risk management module for the Low-Budget Small-Cap Strategy.

Handles:
- Position sizing with Kelly-fraction approximation
- Stop-loss monitoring
- Portfolio-level risk metrics (drawdown, Sharpe, Sortino)
- Rebalance logic with hysteresis to reduce turnover
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Position Sizing ─────────────────────────────────────────────────────────


def equal_weight_allocation(
    capital: float,
    n_positions: int,
) -> float:
    """Simple equal-weight allocation per position."""
    if n_positions <= 0:
        return 0.0
    per_position = capital / n_positions
    # Enforce min/max constraints
    max_allowed = capital * config.MAX_POSITION_PCT
    per_position = min(per_position, max_allowed)
    if per_position < config.MIN_POSITION_SIZE:
        logger.warning(
            f"Position size ${per_position:.2f} below minimum ${config.MIN_POSITION_SIZE}; "
            "reducing number of positions"
        )
    return per_position


def compute_position_sizes(
    capital: float,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute dollar amount per position based on equal weighting.
    Adjusts number of positions if capital is too small.
    """
    n = len(scores)
    per_pos = capital / n if n > 0 else 0

    # Reduce positions if we can't meet minimum size
    while per_pos < config.MIN_POSITION_SIZE and n > 1:
        n -= 1
        per_pos = capital / n

    portfolio = scores.head(n).copy()
    portfolio["dollar_amount"] = per_pos
    portfolio["weight"] = 1.0 / n
    portfolio["shares"] = 0.0  # To be filled when prices are known

    logger.info(f"Allocated ${capital:.2f} across {n} positions (${per_pos:.2f} each)")
    return portfolio


# ── Stop-Loss ───────────────────────────────────────────────────────────────


def check_stop_losses(
    holdings: pd.DataFrame,
    current_prices: pd.Series,
) -> list[str]:
    """
    Check which holdings have hit the stop-loss threshold.

    Args:
        holdings: DataFrame with columns ['entry_price', ...] indexed by ticker
        current_prices: Series of current prices indexed by ticker

    Returns:
        List of tickers that should be sold (hit stop-loss).
    """
    to_sell = []
    for ticker in holdings.index:
        if ticker not in current_prices.index:
            continue
        entry = holdings.loc[ticker, "entry_price"]
        current = current_prices[ticker]
        pct_change = (current - entry) / entry

        if pct_change <= config.STOP_LOSS_PCT:
            logger.info(
                f"STOP-LOSS triggered for {ticker}: "
                f"entry=${entry:.2f}, current=${current:.2f}, "
                f"loss={pct_change:.1%}"
            )
            to_sell.append(ticker)

    return to_sell


# ── Rebalance Logic ─────────────────────────────────────────────────────────


def compute_rebalance_trades(
    current_holdings: set[str],
    new_portfolio: pd.DataFrame,
    all_scores: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Determine which stocks to buy and sell at rebalance.

    Uses hysteresis: only sell a current holding if it drops below
    HYSTERESIS_RANK in the full scoring table. This reduces turnover
    from small rank fluctuations.

    Returns:
        (to_buy, to_sell) — lists of ticker strings
    """
    new_tickers = set(new_portfolio.index)

    # Stocks to definitely add
    to_buy = list(new_tickers - current_holdings)

    # Apply hysteresis — only sell if rank dropped below threshold
    to_sell = []
    for ticker in current_holdings:
        if ticker in new_tickers:
            continue  # Still in top picks, keep
        if ticker in all_scores.index:
            rank = all_scores.index.get_loc(ticker) + 1  # 1-based rank
            if rank > config.HYSTERESIS_RANK:
                to_sell.append(ticker)
                logger.info(
                    f"Selling {ticker}: rank {rank} > hysteresis threshold {config.HYSTERESIS_RANK}"
                )
            else:
                logger.info(f"Keeping {ticker}: rank {rank} within hysteresis band")
        else:
            to_sell.append(ticker)  # No longer in universe at all

    return to_buy, to_sell


# ── Portfolio Risk Metrics ──────────────────────────────────────────────────


def compute_portfolio_metrics(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """
    Compute key portfolio risk/return metrics.

    Args:
        returns: Series of portfolio returns (monthly by default)
        periods_per_year: Number of return observations per year (12=monthly, 252=daily)

    Returns:
        Dict with metrics: total_return, cagr, sharpe, sortino,
        max_drawdown, volatility, win_rate
    """
    if returns.empty or len(returns) < 2:
        return {}

    # Total return
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # CAGR — use actual date range if available, else observation count
    if hasattr(returns.index, "min") and hasattr(returns.index, "max"):
        try:
            delta = returns.index[-1] - returns.index[0]
            n_years = max(delta.days / 365.25, 0.01)
        except Exception:
            n_years = len(returns) / periods_per_year
    else:
        n_years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Annualised volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    excess_period = returns - config.RISK_FREE_RATE / periods_per_year
    sharpe = (
        excess_period.mean() / returns.std() * np.sqrt(periods_per_year)
        if returns.std() > 0
        else 0
    )

    # Sortino ratio (downside deviation only)
    downside = returns[returns < 0]
    downside_std = (
        downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 0.001
    )
    sortino = (returns.mean() * periods_per_year - config.RISK_FREE_RATE) / downside_std

    # Maximum drawdown
    cumulative_max = cumulative.cummax()
    drawdowns = (cumulative - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualised_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_trading_days": len(returns),
    }


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the running drawdown series from daily returns."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown
