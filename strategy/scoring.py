"""
Scoring engine for the Small-Cap Momentum + Value strategy.

Implements the composite scoring system based on:
- Momentum: Jegadeesh & Titman (1993) — 12-month minus last month return
- Value: Percentile ranking on valuation multiples (lower = better)
- Piotroski F-Score: Financial health from accounting data (Piotroski, 2000)

Combined via weighted composite as described in STRATEGY_RESEARCH.md.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Momentum Score ──────────────────────────────────────────────────────────


def compute_momentum_score(prices: pd.DataFrame) -> pd.Series:
    """
    Jegadeesh & Titman (1993) momentum signal.

    Uses 12-month return skipping the most recent month to avoid
    short-term reversal. Returns a percentile rank (0-100) per ticker.
    """
    if len(prices) < config.MOMENTUM_LOOKBACK:
        logger.warning(
            f"Only {len(prices)} rows of price data; need {config.MOMENTUM_LOOKBACK} "
            "for full momentum lookback — using available data"
        )

    # Price at t-skip (skip most recent month)
    recent = (
        prices.iloc[-config.MOMENTUM_SKIP]
        if len(prices) > config.MOMENTUM_SKIP
        else prices.iloc[-1]
    )
    # Price at t-lookback
    past = (
        prices.iloc[-config.MOMENTUM_LOOKBACK]
        if len(prices) >= config.MOMENTUM_LOOKBACK
        else prices.iloc[0]
    )

    momentum_return = (recent - past) / past
    momentum_return = momentum_return.replace([np.inf, -np.inf], np.nan).dropna()

    # Convert to percentile rank (0 = worst momentum, 100 = best)
    rank = momentum_return.rank(pct=True) * 100
    rank.name = "momentum_score"
    return rank


# ── Value Score ─────────────────────────────────────────────────────────────


def compute_value_score(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Composite value score from valuation multiples.

    Lower P/E, P/S, and EV/EBITDA = better value = higher score.
    Each metric is percentile-ranked and averaged.
    Missing metrics are ignored (scored on available data).
    """
    fundamentals = fundamentals[~fundamentals.index.duplicated(keep="first")]
    scores = pd.DataFrame(index=fundamentals.index)

    for col in ["pe_ratio", "ps_ratio", "ev_ebitda"]:
        if col in fundamentals.columns:
            valid = fundamentals[col].replace([np.inf, -np.inf], np.nan).dropna()
            # Negative P/E means losses — penalise by giving worst rank
            if col == "pe_ratio":
                valid = valid[valid > 0]
            # Lower is better → invert the rank
            scores[col] = (1 - valid.rank(pct=True)) * 100

    if scores.empty:
        logger.warning("No value metrics available for scoring")
        return pd.Series(dtype=float, name="value_score")

    value_score = scores.mean(axis=1)
    value_score.name = "value_score"
    return value_score


# ── Piotroski F-Score ───────────────────────────────────────────────────────


def compute_fscore(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Piotroski (2000) F-Score: 9-point financial health score.

    Each criterion adds 1 point:
    Profitability (4 points):
      1. ROA > 0
      2. Operating cash flow > 0
      3. Change in ROA > 0 (proxied by ROA level for single-period)
      4. Cash flow > Net Income (accrual quality)
    Leverage/Liquidity (3 points):
      5. Decrease in debt-to-equity (proxied: D/E < 1.0)
      6. Current ratio > 1.0
      7. No dilution: shares outstanding didn't increase (proxied: available = +1)
    Efficiency (2 points):
      8. Gross margin > 0
      9. Asset turnover > median

    Note: With single-period snapshot data, we proxy year-over-year changes
    with absolute thresholds. Full implementation needs two periods of data.
    """
    df = fundamentals.copy()
    df = df[~df.index.duplicated(keep="first")]
    score = pd.Series(0, index=df.index, dtype=int)

    # 1. ROA > 0
    if "roa" in df.columns:
        score += (df["roa"] > 0).astype(int)

    # 2. Operating cash flow > 0
    if "operating_cashflow" in df.columns:
        score += (df["operating_cashflow"] > 0).astype(int)

    # 3. ROA improvement proxy (ROA > 5% as a proxy for "improving")
    if "roa" in df.columns:
        score += (df["roa"] > 0.05).astype(int)

    # 4. Accrual quality: Operating CF > Net Income
    if "operating_cashflow" in df.columns and "net_income" in df.columns:
        score += (df["operating_cashflow"] > df["net_income"]).astype(int)

    # 5. Low leverage: Debt/Equity < 1.0
    if "debt_to_equity" in df.columns:
        score += (df["debt_to_equity"].fillna(999) < 100).astype(
            int
        )  # D/E in % on some APIs

    # 6. Current ratio > 1.0
    if "current_ratio" in df.columns:
        score += (df["current_ratio"] > 1.0).astype(int)

    # 7. No dilution proxy — give 1 point by default (need two periods for real check)
    score += 1

    # 8. Positive gross margin
    if "gross_margin" in df.columns:
        score += (df["gross_margin"] > 0).astype(int)

    # 9. Asset turnover above median
    if "asset_turnover" in df.columns:
        median_at = df["asset_turnover"].median()
        score += (df["asset_turnover"] > median_at).astype(int)

    score.name = "fscore"
    # Convert to 0-100 scale
    fscore_pct = (score / 9) * 100
    fscore_pct.name = "fscore_score"
    return fscore_pct


# ── Composite Score ─────────────────────────────────────────────────────────


def compute_composite_score(
    momentum: pd.Series,
    value: pd.Series,
    fscore: pd.Series,
) -> pd.DataFrame:
    """
    Weighted composite of momentum, value, and F-Score.

    Weights from config:
    - Momentum: 35%  (Jegadeesh & Titman)
    - F-Score:  40%  (Piotroski)
    - Value:    25%  (Fama-French value factor)

    Returns a DataFrame with individual scores and composite, sorted descending.
    """
    scores = pd.DataFrame(
        {
            "momentum": momentum,
            "value": value,
            "fscore": fscore,
        }
    )

    # Fill NaN with 50 (neutral) so stocks with partial data aren't penalised too harshly
    scores = scores.fillna(50)

    scores["composite"] = (
        scores["momentum"] * config.MOMENTUM_WEIGHT
        + scores["fscore"] * config.FSCORE_WEIGHT
        + scores["value"] * config.VALUE_WEIGHT
    )

    scores = scores.sort_values("composite", ascending=False)
    return scores


# ── Top Picks ───────────────────────────────────────────────────────────────


def select_portfolio(
    scores: pd.DataFrame,
    max_positions: int = config.MAX_POSITIONS,
    min_fscore: int = config.MIN_FSCORE,
) -> pd.DataFrame:
    """
    Select the top-N stocks by composite score.

    Applies F-Score minimum filter before ranking.
    """
    # Filter: require minimum F-Score (converted from 0-100 back to 0-9 scale)
    min_fscore_pct = (min_fscore / 9) * 100
    eligible = scores[scores["fscore"] >= min_fscore_pct]

    if len(eligible) < max_positions:
        logger.warning(
            f"Only {len(eligible)} stocks pass F-Score filter "
            f"(need {max_positions}); relaxing filter"
        )
        eligible = scores.head(max_positions)

    portfolio = eligible.head(max_positions).copy()
    portfolio["weight"] = 1.0 / len(portfolio)  # Equal weight
    return portfolio
