"""
Elite Trader Momentum — Scoring Engine

Combined scoring system from multiple profitable traders:

1. TREND TEMPLATE (Mark Minervini, 3x US Investing Champion)
   - Price > 150-day SMA > 200-day SMA
   - 200-day SMA rising for at least 1 month
   - Price within 25% of 52-week high
   - Price at least 25% above 52-week low
   Hard filter + scored by how well criteria are met.

2. MOMENTUM (William O'Neil / Jegadeesh & Titman)
   - 6-month return skipping most recent 2 weeks
   - Percentile ranked across universe (relative strength)

3. VOLUME BREAKOUT (Nicolas Darvas / Stan Weinstein)
   - Recent volume vs 50-day average volume
   - Demand surge signals institutional accumulation

4. QUALITY / F-SCORE (Piotroski / O'Neil)
   - 9-point financial health score
   - Profitable, low-debt, cash-flow positive companies
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Minervini Trend Template ────────────────────────────────────────────────


def passes_trend_template(prices: pd.DataFrame) -> pd.Series:
    """
    Mark Minervini's Trend Template — hard filter.

    A stock must satisfy ALL of these to be considered:
    1. Price > 150-day SMA
    2. Price > 200-day SMA
    3. 150-day SMA > 200-day SMA
    4. 200-day SMA trending up (current > value N days ago)
    5. Price within 25% of 52-week high
    6. Price at least 25% above 52-week low
    7. Price > 50-day SMA (Weinstein Stage 2 confirmation)

    Returns a boolean Series indexed by ticker.
    """
    if len(prices) < config.TREND_SMA_200:
        logger.warning(
            f"Only {len(prices)} rows; need {config.TREND_SMA_200} for trend template"
        )
        return pd.Series(True, index=prices.columns)

    current_price = prices.iloc[-1]
    sma_50 = prices.iloc[-config.TREND_SMA_50 :].mean()
    sma_150 = prices.iloc[-config.TREND_SMA_150 :].mean()
    sma_200 = prices.iloc[-config.TREND_SMA_200 :].mean()

    # 200-day SMA from N days ago (to check slope)
    slope_days = config.TREND_200_SLOPE_DAYS
    if len(prices) >= config.TREND_SMA_200 + slope_days:
        sma_200_prev = prices.iloc[
            -(config.TREND_SMA_200 + slope_days) : -slope_days
        ].mean()
    else:
        sma_200_prev = sma_200  # Not enough data, pass by default

    # 52-week high and low
    lookback_252 = min(252, len(prices))
    high_52w = prices.iloc[-lookback_252:].max()
    low_52w = prices.iloc[-lookback_252:].min()

    # Apply all 7 criteria
    c1 = current_price > sma_150  # Price > 150-day SMA
    c2 = current_price > sma_200  # Price > 200-day SMA
    c3 = sma_150 > sma_200  # 150-day > 200-day (uptrend structure)
    c4 = sma_200 > sma_200_prev  # 200-day rising
    c5 = current_price >= high_52w * config.TREND_52W_HIGH_PCT  # Within 25% of 52w high
    c6 = current_price >= low_52w * config.TREND_52W_LOW_PCT  # 25%+ above 52w low
    c7 = current_price > sma_50  # Price > 50-day SMA (Stage 2)

    # Apply criteria — require MIN_TREND_CRITERIA of 7 (default 5)
    criteria_sum = (
        c1.astype(int)
        + c2.astype(int)
        + c3.astype(int)
        + c4.astype(int)
        + c5.astype(int)
        + c6.astype(int)
        + c7.astype(int)
    )
    min_criteria = getattr(config, "MIN_TREND_CRITERIA", 7)
    passes = criteria_sum >= min_criteria
    passes = passes.fillna(False)

    n_pass = passes.sum()
    logger.info(
        f"Trend Template: {n_pass}/{len(passes)} stocks pass {min_criteria}/7 criteria"
    )
    return passes


def compute_trend_score(prices: pd.DataFrame) -> pd.Series:
    """
    Continuous trend strength score (0-100) based on Minervini criteria.

    Instead of binary pass/fail, scores how strongly each criterion is met.
    Stocks with stronger uptrend structure score higher.
    """
    if len(prices) < config.TREND_SMA_200:
        return pd.Series(50, index=prices.columns, name="trend_score")

    current_price = prices.iloc[-1]
    sma_50 = prices.iloc[-config.TREND_SMA_50 :].mean()
    sma_150 = prices.iloc[-config.TREND_SMA_150 :].mean()
    sma_200 = prices.iloc[-config.TREND_SMA_200 :].mean()

    lookback_252 = min(252, len(prices))
    high_52w = prices.iloc[-lookback_252:].max()
    low_52w = prices.iloc[-lookback_252:].min()

    # Score each component on a 0-100 scale
    scores = pd.DataFrame(index=prices.columns)

    # How far above 200-day SMA (higher = stronger trend)
    pct_above_200 = ((current_price - sma_200) / sma_200).clip(-0.5, 0.5)
    scores["above_200"] = ((pct_above_200 + 0.5) / 1.0 * 100).clip(0, 100)

    # How far above 150-day SMA
    pct_above_150 = ((current_price - sma_150) / sma_150).clip(-0.5, 0.5)
    scores["above_150"] = ((pct_above_150 + 0.5) / 1.0 * 100).clip(0, 100)

    # SMA alignment: 50 > 150 > 200 (proper uptrend stack)
    sma_align = ((sma_50 - sma_150) / sma_150).clip(-0.2, 0.2) + (
        (sma_150 - sma_200) / sma_200
    ).clip(-0.2, 0.2)
    scores["sma_alignment"] = ((sma_align + 0.4) / 0.8 * 100).clip(0, 100)

    # Proximity to 52-week high (closer = better)
    pct_of_high = (current_price / high_52w).clip(0, 1)
    scores["near_high"] = (pct_of_high * 100).clip(0, 100)

    # Distance above 52-week low (further above = stronger)
    pct_above_low = ((current_price - low_52w) / low_52w).clip(0, 2)
    scores["above_low"] = (pct_above_low / 2.0 * 100).clip(0, 100)

    trend_score = scores.mean(axis=1)
    trend_score.name = "trend_score"
    return trend_score


# ── Momentum Score (O'Neil / Jegadeesh & Titman) ───────────────────────────


def compute_momentum_score(prices: pd.DataFrame) -> pd.Series:
    """
    Relative strength momentum signal.

    Uses 6-month return skipping the most recent 2 weeks to avoid
    short-term reversal (Jegadeesh & Titman 1993).
    Also incorporates 3-month acceleration (O'Neil: accelerating earnings proxy).
    Returns a percentile rank (0-100) per ticker.
    """
    if len(prices) < config.MOMENTUM_LOOKBACK:
        logger.warning(
            f"Only {len(prices)} rows of price data; need {config.MOMENTUM_LOOKBACK} "
            "for full momentum lookback — using available data"
        )

    # Price at t-skip (skip most recent 2 weeks)
    recent = (
        prices.iloc[-config.MOMENTUM_SKIP]
        if len(prices) > config.MOMENTUM_SKIP
        else prices.iloc[-1]
    )
    # Price at t-lookback (6 months ago)
    past_6m = (
        prices.iloc[-config.MOMENTUM_LOOKBACK]
        if len(prices) >= config.MOMENTUM_LOOKBACK
        else prices.iloc[0]
    )

    # 6-month return
    ret_6m = (recent - past_6m) / past_6m
    ret_6m = ret_6m.replace([np.inf, -np.inf], np.nan).dropna()

    # 3-month return (acceleration component — is momentum accelerating?)
    lookback_3m = config.MOMENTUM_LOOKBACK // 2
    past_3m = (
        prices.iloc[-lookback_3m] if len(prices) >= lookback_3m else prices.iloc[0]
    )
    ret_3m = (recent - past_3m) / past_3m
    ret_3m = ret_3m.replace([np.inf, -np.inf], np.nan)

    # Combined: 70% 6-month return + 30% 3-month return (rewards acceleration)
    combined = ret_6m * 0.7 + ret_3m.reindex(ret_6m.index).fillna(0) * 0.3

    # Convert to percentile rank (0 = worst, 100 = best)
    rank = combined.rank(pct=True) * 100
    rank.name = "momentum_score"
    return rank


# ── Volume Breakout Score (Darvas / Weinstein) ──────────────────────────────


def compute_volume_score(prices: pd.DataFrame) -> pd.Series:
    """
    Volume-based demand signal (Darvas / Weinstein / O'Neil).

    Measures recent price-volume action to detect institutional accumulation:
    - Stocks making new highs on above-average volume score highest
    - Uses price range expansion as a proxy when raw volume isn't available
      (price data from Alpaca close prices — we proxy volume via volatility)

    Returns a percentile rank (0-100) per ticker.
    """
    vol_lookback = getattr(config, "VOLUME_LOOKBACK", 50)

    if len(prices) < vol_lookback:
        return pd.Series(50, index=prices.columns, name="volume_score")

    # Proxy for volume/demand: recent price action strength
    # (a) How close to 20-day high vs 20-day low (breakout proximity)
    recent_20 = prices.iloc[-20:]
    high_20 = recent_20.max()
    low_20 = recent_20.min()
    price_range = high_20 - low_20
    price_range = price_range.replace(0, np.nan)
    breakout_position = (prices.iloc[-1] - low_20) / price_range  # 0=at low, 1=at high

    # (b) Volatility expansion: recent 10-day range vs 50-day range
    recent_10_range = prices.iloc[-10:].max() - prices.iloc[-10:].min()
    longer_range = prices.iloc[-vol_lookback:].max() - prices.iloc[-vol_lookback:].min()
    longer_range = longer_range.replace(0, np.nan)
    vol_expansion = (
        recent_10_range / longer_range
    )  # >0.5 means recent action is outsized

    # (c) Consecutive up-days ratio in last 20 days (buying pressure)
    daily_returns = prices.iloc[-21:].pct_change(fill_method=None).iloc[1:]
    up_ratio = (daily_returns > 0).sum() / len(daily_returns)

    # Combine: breakout position (40%) + volatility expansion (30%) + up ratio (30%)
    combined = (
        breakout_position.fillna(0.5) * 0.4
        + vol_expansion.fillna(0.5) * 0.3
        + up_ratio * 0.3
    )

    rank = combined.rank(pct=True) * 100
    rank.name = "volume_score"
    return rank


# ── Piotroski F-Score (Quality Gate) ───────────────────────────────────────


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
        score += (df["debt_to_equity"].fillna(999) < 100).astype(int)

    # 6. Current ratio > 1.0
    if "current_ratio" in df.columns:
        score += (df["current_ratio"] > 1.0).astype(int)

    # 7. No dilution proxy — give 1 point by default
    score += 1

    # 8. Positive gross margin
    if "gross_margin" in df.columns:
        score += (df["gross_margin"] > 0).astype(int)

    # 9. Asset turnover above median
    if "asset_turnover" in df.columns:
        median_at = df["asset_turnover"].median()
        score += (df["asset_turnover"] > median_at).astype(int)

    score.name = "fscore"
    fscore_pct = (score / 9) * 100
    fscore_pct.name = "fscore_score"
    return fscore_pct


# ── Value Score (kept for backward compat, lower weight) ───────────────────


def compute_value_score(fundamentals: pd.DataFrame) -> pd.Series:
    """
    Composite value score from valuation multiples.
    Lower P/E, P/S, and EV/EBITDA = better value = higher score.
    """
    fundamentals = fundamentals[~fundamentals.index.duplicated(keep="first")]
    scores = pd.DataFrame(index=fundamentals.index)

    for col in ["pe_ratio", "ps_ratio", "ev_ebitda"]:
        if col in fundamentals.columns:
            valid = fundamentals[col].replace([np.inf, -np.inf], np.nan).dropna()
            if col == "pe_ratio":
                valid = valid[valid > 0]
            scores[col] = (1 - valid.rank(pct=True)) * 100

    if scores.empty:
        return pd.Series(dtype=float, name="value_score")

    value_score = scores.mean(axis=1)
    value_score.name = "value_score"
    return value_score


# ── Composite Score ─────────────────────────────────────────────────────────


def compute_composite_score(
    momentum: pd.Series,
    value: pd.Series,
    fscore: pd.Series,
    trend: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Weighted composite of all scoring factors.

    Weights from config (Elite Trader Momentum):
    - Momentum:  35%  (O'Neil / Jegadeesh & Titman)
    - Trend:     25%  (Minervini Trend Template)
    - Volume:    20%  (Darvas / Weinstein)
    - Quality:   20%  (Piotroski F-Score)

    If trend/volume not provided, falls back to original 3-factor model.
    """
    scores = pd.DataFrame({"momentum": momentum})

    # Use the new multi-factor model if trend and volume are available
    if trend is not None and volume is not None:
        scores["trend"] = trend
        scores["volume"] = volume
        scores["fscore"] = fscore

        scores = scores.fillna(50)

        scores["composite"] = (
            scores["momentum"] * config.MOMENTUM_WEIGHT
            + scores["trend"] * config.TREND_WEIGHT
            + scores["volume"] * config.VOLUME_WEIGHT
            + scores["fscore"] * config.FSCORE_WEIGHT
        )
    else:
        # Fallback: original 3-factor model
        scores["value"] = value
        scores["fscore"] = fscore
        scores = scores.fillna(50)

        momentum_w = config.MOMENTUM_WEIGHT
        fscore_w = config.FSCORE_WEIGHT
        value_w = getattr(config, "VALUE_WEIGHT", 0.25)
        scores["composite"] = (
            scores["momentum"] * momentum_w
            + scores["fscore"] * fscore_w
            + scores["value"] * value_w
        )

    scores = scores.sort_values("composite", ascending=False)
    return scores


# ── Portfolio Selection ─────────────────────────────────────────────────────


def select_portfolio(
    scores: pd.DataFrame,
    max_positions: int = config.MAX_POSITIONS,
    min_fscore: int = config.MIN_FSCORE,
    trend_filter: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Select the top-N stocks by composite score.

    Applies:
    1. Minervini Trend Template filter (hard gate — must pass all criteria)
    2. Piotroski F-Score minimum (quality gate)
    3. Rank by composite score, take top N
    """
    eligible = scores.copy()

    # Apply trend template filter if provided (Minervini hard gate)
    if trend_filter is not None:
        passing_tickers = trend_filter[trend_filter].index
        before = len(eligible)
        eligible = eligible.loc[eligible.index.intersection(passing_tickers)]
        logger.info(
            f"Trend Template filter: {before} → {len(eligible)} stocks "
            f"({before - len(eligible)} filtered out)"
        )

    # F-Score minimum filter
    min_fscore_pct = (min_fscore / 9) * 100
    if "fscore" in eligible.columns:
        quality_pass = eligible[eligible["fscore"] >= min_fscore_pct]
    else:
        quality_pass = eligible

    if len(quality_pass) >= max_positions:
        eligible = quality_pass
    else:
        logger.warning(
            f"Only {len(quality_pass)} stocks pass both filters "
            f"(need {max_positions}); relaxing F-Score filter"
        )
        # Keep trend template filter but relax F-Score
        if len(eligible) < max_positions:
            logger.warning("Not enough stocks pass trend template — using top scorers")
            eligible = scores.head(max_positions * 2)

    portfolio = eligible.head(max_positions).copy()
    if len(portfolio) > 0:
        portfolio["weight"] = 1.0 / len(portfolio)
    return portfolio
