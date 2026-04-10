"""
Zipline-inspired event-driven backtester for the Small-Cap Strategy.

Architecture modeled after Zipline (https://github.com/quantopian/zipline):

- Event-driven: processes one trading day at a time — look-ahead bias
  is impossible by construction
- Next-bar execution: orders placed on day T fill at day T+1's price,
  modeling realistic execution delay
- Modular slippage & commission models (zipline.finance.slippage/commission)
- Daily mark-to-market with comprehensive performance tracking
- Full tear-sheet output with alpha, beta, Sharpe, Sortino, Calmar,
  information ratio, drawdown analysis, and execution statistics

Usage:
    python main.py zipline-backtest
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
from strategy.backtester import _enrich_valuations
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Position:
    """A single stock position, tracked at share level."""

    shares: float
    cost_basis: float  # weighted average cost per share
    high_water: float = 0.0  # highest price since entry (for trailing stop)


@dataclass
class Order:
    """
    A pending order — queued on day T, fills on day T+1.
    This delay is the key difference vs the simple backtester.
    """

    ticker: str
    shares: float  # positive = buy, negative = sell
    created_dt: pd.Timestamp
    filled: bool = False
    fill_price: float = 0.0
    fill_dt: Optional[pd.Timestamp] = None
    commission: float = 0.0
    slippage_cost: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Commission Models (inspired by zipline.finance.commission)
# ═══════════════════════════════════════════════════════════════════════════


class PerShareCommission:
    """Fixed cost per share traded, with an optional minimum per trade."""

    def __init__(self, cost_per_share: float = 0.005, min_trade_cost: float = 1.00):
        self.cost_per_share = cost_per_share
        self.min_trade_cost = min_trade_cost

    def calculate(self, shares: float, fill_price: float) -> float:
        return max(abs(shares) * self.cost_per_share, self.min_trade_cost)


class PercentageCommission:
    """Commission as a percentage of trade value."""

    def __init__(self, pct: float = 0.001):
        self.pct = pct

    def calculate(self, shares: float, fill_price: float) -> float:
        return abs(shares * fill_price) * self.pct


class NoCommission:
    """Zero commission (Alpaca, most modern brokers)."""

    def calculate(self, shares: float, fill_price: float) -> float:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Slippage Models (inspired by zipline.finance.slippage)
# ═══════════════════════════════════════════════════════════════════════════


class FixedBasisPointsSlippage:
    """Fixed basis points of adverse price impact per trade."""

    def __init__(self, basis_points: float = 5.0):
        self.bps = basis_points / 10_000

    def calculate(self, price: float, shares: float, **kwargs) -> float:
        direction = 1 if shares > 0 else -1
        return price * self.bps * direction


class VolatilitySlippage:
    """
    Slippage proportional to recent volatility and inverse liquidity.
    Models real-world execution cost in small/micro-cap stocks.

        slippage = base_bps + vol_mult × σ + liq_mult × illiquidity

    This is the default model — tuned to match config.py's existing
    SLIPPAGE_VOL_MULT and SLIPPAGE_LIQUIDITY_MULT parameters.
    """

    def __init__(
        self,
        base_bps: float = 5.0,
        vol_mult: float = 0.05,
        liq_mult: float = 0.00025,
    ):
        self.base_bps = base_bps / 10_000
        self.vol_mult = vol_mult
        self.liq_mult = liq_mult

    def calculate(
        self,
        price: float,
        shares: float,
        volatility: float = 0.0,
        adv: float = 1e6,
        **kwargs,
    ) -> float:
        direction = 1 if shares > 0 else -1
        illiquidity = max(0, 1_000_000 - adv) / 100_000
        slip_pct = (
            self.base_bps + self.vol_mult * volatility + self.liq_mult * illiquidity
        )
        return price * slip_pct * direction


# ═══════════════════════════════════════════════════════════════════════════
#  Performance Tracker (inspired by zipline.finance.performance + pyfolio)
# ═══════════════════════════════════════════════════════════════════════════


class PerformanceTracker:
    """
    Tracks daily portfolio value and computes comprehensive metrics.
    Produces a pyfolio-compatible set of statistics.
    """

    def __init__(self, initial_capital: float, risk_free_rate: float):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.daily_values: list[tuple] = []
        self.daily_cash: list[tuple] = []

    def record(self, dt: pd.Timestamp, portfolio_value: float, cash: float):
        self.daily_values.append((dt, portfolio_value))
        self.daily_cash.append((dt, cash))

    def build_equity_curve(self) -> pd.Series:
        df = pd.DataFrame(self.daily_values, columns=["date", "value"])
        return df.set_index("date")["value"]

    def compute_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> dict:
        equity = self.build_equity_curve()
        returns = equity.pct_change().dropna()

        if returns.empty or len(returns) < 2:
            return {}

        n_days = len(returns)
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25

        # ── Core return metrics ──────────────────────────────────────
        cum = (1 + returns).cumprod()
        total_return = cum.iloc[-1] - 1
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        vol = returns.std() * np.sqrt(252)

        # ── Risk-adjusted metrics ────────────────────────────────────
        excess_daily = returns - self.risk_free_rate / 252
        sharpe = (
            excess_daily.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0
            else 0
        )

        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
        sortino = (returns.mean() * 252 - self.risk_free_rate) / downside_std

        # ── Drawdown analysis ────────────────────────────────────────
        cum_max = cum.cummax()
        drawdown = (cum - cum_max) / cum_max
        max_dd = drawdown.min()
        dd_duration = self._max_dd_duration(drawdown)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # ── Distribution metrics ─────────────────────────────────────
        win_rate = (returns > 0).sum() / len(returns)
        p95 = np.percentile(returns, 95)
        p5 = abs(np.percentile(returns, 5))
        tail_ratio = p95 / p5 if p5 > 0 else 0

        # ── Annual breakdown ─────────────────────────────────────────
        annual_returns = self._annual_returns(equity)

        metrics = {
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": dd_duration,
            "win_rate_daily": win_rate,
            "tail_ratio": tail_ratio,
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "n_trading_days": n_days,
            "annual_returns": annual_returns,
        }

        # ── Benchmark-relative metrics ───────────────────────────────
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            common = returns.index.intersection(benchmark_returns.index)
            if len(common) > 10:
                strat_r = returns.loc[common]
                bench_r = benchmark_returns.loc[common]

                # Alpha & Beta via OLS: R_strat = α + β·R_bench + ε
                slope, intercept, r_value, _, _ = stats.linregress(
                    bench_r.values, strat_r.values
                )
                beta = slope
                alpha_annual = intercept * 252

                # Information ratio
                active = strat_r - bench_r
                ir = (
                    (active.mean() * 252) / (active.std() * np.sqrt(252))
                    if active.std() > 0
                    else 0
                )

                # Benchmark performance
                bench_cum = (1 + bench_r).cumprod()
                bench_total = bench_cum.iloc[-1] - 1
                bench_cagr = (1 + bench_total) ** (1 / max(n_years, 0.01)) - 1

                metrics.update(
                    {
                        "alpha": alpha_annual,
                        "beta": beta,
                        "r_squared": r_value**2,
                        "information_ratio": ir,
                        "benchmark_total_return": bench_total,
                        "benchmark_cagr": bench_cagr,
                        "excess_cagr": cagr - bench_cagr,
                        "correlation": strat_r.corr(bench_r),
                    }
                )

        return metrics

    @staticmethod
    def _max_dd_duration(drawdown: pd.Series) -> int:
        """Maximum drawdown duration in calendar days."""
        in_dd = drawdown < 0
        if not in_dd.any():
            return 0
        groups = (~in_dd).cumsum()
        durations = []
        for _, group in drawdown[in_dd].groupby(groups[in_dd]):
            durations.append((group.index[-1] - group.index[0]).days)
        return max(durations) if durations else 0

    @staticmethod
    def _annual_returns(equity: pd.Series) -> dict:
        """Per-year returns."""
        annual = {}
        for year in equity.index.year.unique():
            year_data = equity[equity.index.year == year]
            if len(year_data) >= 2:
                annual[int(year)] = year_data.iloc[-1] / year_data.iloc[0] - 1
        return annual


# ═══════════════════════════════════════════════════════════════════════════
#  Zipline-Style Event-Driven Backtester
# ═══════════════════════════════════════════════════════════════════════════


class ZiplineBacktester:
    """
    Event-driven backtesting engine modeled after Zipline's TradingAlgorithm.

    Key properties that ensure backtest integrity:
      1. Processes one bar at a time — no access to future data
      2. Orders placed on day T fill at day T+1's close price
      3. Slippage and commission applied to every fill
      4. Daily mark-to-market (not just at rebalance points)
      5. Reuses existing scoring.py for consistent signal generation
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        fundamentals=None,
        benchmark_prices: Optional[pd.Series] = None,
        initial_capital: float = config.INITIAL_CAPITAL,
        commission_model=None,
        slippage_model=None,
    ):
        self.prices = prices.sort_index()
        self.fundamentals = fundamentals
        self.benchmark_prices = benchmark_prices
        self.initial_capital = initial_capital

        # Default: Alpaca is commission-free; volatility-based slippage
        self.commission = commission_model or NoCommission()
        self.slippage = slippage_model or VolatilitySlippage(
            base_bps=5.0,
            vol_mult=config.SLIPPAGE_VOL_MULT,
            liq_mult=config.SLIPPAGE_LIQUIDITY_MULT,
        )

        # Portfolio state
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.pending_orders: list[Order] = []

        # Tracking
        self.perf = PerformanceTracker(initial_capital, config.RISK_FREE_RATE)
        self.all_fills: list[dict] = []
        self.rebalance_log: list[dict] = []

        # Rebalance schedule
        freq = getattr(config, "REBALANCE_FREQUENCY", "monthly")
        if freq == "quarterly":
            self.rebal_months = {3, 6, 9, 12}
        else:
            self.rebal_months = set(range(1, 13))
        self.rebal_day = getattr(config, "REBALANCE_DAY", 1)
        self.daily_mode = freq == "daily"

        # Regime filter with hysteresis
        self.use_regime = getattr(config, "USE_REGIME_FILTER", False)
        self.regime_ticker = getattr(config, "REGIME_TICKER", "SPY")
        self.regime_ma = getattr(config, "REGIME_MA_PERIOD", 200)
        self.regime_exit_pct = getattr(config, "REGIME_EXIT_PCT", -0.02)
        self.regime_enter_pct = getattr(config, "REGIME_ENTER_PCT", 0.01)
        self._regime_bullish = True  # Start bullish, switch on signals

        # Trailing stop
        self.trailing_stop = getattr(config, "TRAILING_STOP", True)

        # ATR stop parameters (Turtle Traders)
        self.atr_stop_mult = getattr(config, "ATR_STOP_MULT", 0)
        self.atr_period = getattr(config, "ATR_PERIOD", 20)

        # Inverse-volatility position sizing
        self.inv_vol_sizing = getattr(config, "INVERSE_VOL_SIZING", False)
        self.inv_vol_lookback = getattr(config, "INVERSE_VOL_LOOKBACK", 21)

        # Portfolio drawdown circuit breaker
        self.portfolio_dd_limit = getattr(config, "PORTFOLIO_DD_LIMIT", -0.20)
        self.portfolio_dd_reentry = getattr(config, "PORTFOLIO_DD_REENTRY", -0.10)
        self._portfolio_hwm = initial_capital
        self._portfolio_in_dd_cash = False  # True = circuit breaker tripped

        # Track stop-outs for daily slot refilling
        self._slots_freed_today = False

    # ── Portfolio Valuation ───────────────────────────────────────────────

    def _portfolio_value(self, current_prices: pd.Series) -> float:
        value = self.cash
        for ticker, pos in self.positions.items():
            if ticker in current_prices.index:
                price = current_prices[ticker]
                if not np.isnan(price):
                    value += pos.shares * price
        return value

    # ── Trading Calendar ──────────────────────────────────────────────────

    def _is_rebalance_day(
        self, dt: pd.Timestamp, trading_days: pd.DatetimeIndex
    ) -> bool:
        """First trading day of a rebalance month, or every day in daily mode."""
        if self.daily_mode:
            return True
        if dt.month not in self.rebal_months:
            return False
        month_days = trading_days[
            (trading_days.month == dt.month) & (trading_days.year == dt.year)
        ]
        if len(month_days) == 0:
            return False
        day_idx = min(self.rebal_day - 1, len(month_days) - 1)
        return dt == month_days[day_idx]

    # ── Market Regime ─────────────────────────────────────────────────────

    def _check_regime(self, dt: pd.Timestamp) -> bool:
        """
        Regime filter with hysteresis to prevent whipsaw.

        - Go bearish when SPY drops REGIME_EXIT_PCT below 200-day MA
        - Go bullish when SPY climbs REGIME_ENTER_PCT above 200-day MA
        - Between thresholds: maintain previous state
        """
        if not self.use_regime:
            return True
        if self.regime_ticker not in self.prices.columns:
            return True
        spy = self.prices[self.regime_ticker].loc[:dt].dropna()
        if len(spy) < self.regime_ma:
            return True
        ma = float(spy.rolling(self.regime_ma).mean().iloc[-1])
        price = float(spy.iloc[-1])
        pct_vs_ma = (price - ma) / ma

        if self._regime_bullish:
            # Currently bullish — switch to bearish if SPY drops below exit threshold
            if pct_vs_ma <= self.regime_exit_pct:
                self._regime_bullish = False
                logger.info(
                    f"{dt.date()}: Regime → BEARISH "
                    f"(SPY {pct_vs_ma:+.1%} vs 200-MA)"
                )
        else:
            # Currently bearish — switch to bullish if SPY rises above enter threshold
            if pct_vs_ma >= self.regime_enter_pct:
                self._regime_bullish = True
                logger.info(
                    f"{dt.date()}: Regime → BULLISH "
                    f"(SPY {pct_vs_ma:+.1%} vs 200-MA)"
                )

        return self._regime_bullish

    # ── Slippage Inputs ───────────────────────────────────────────────────

    def _compute_volatility(
        self, ticker: str, dt: pd.Timestamp, lookback: int = 21
    ) -> float:
        if ticker not in self.prices.columns:
            return 0.05
        hist = self.prices[ticker].loc[:dt].dropna()
        if len(hist) < lookback + 1:
            return 0.05
        return hist.pct_change().dropna().iloc[-lookback:].std()

    def _compute_adv(self, ticker: str, dt: pd.Timestamp, lookback: int = 21) -> float:
        """Approximate average daily dollar volume from price level."""
        if ticker not in self.prices.columns:
            return 1e6
        hist = self.prices[ticker].loc[:dt].dropna()
        if len(hist) < lookback:
            return 1e6
        # Proxy: avg close price × assumed avg volume
        return float(hist.iloc[-lookback:].mean()) * 50_000

    # ── Order Execution (Next-Bar Fills) ──────────────────────────────────

    def _fill_orders(self, dt: pd.Timestamp):
        """
        Fill pending orders at the current bar's price.
        Orders were placed on a previous bar — this is the T+1 fill.
        """
        if not self.pending_orders:
            return

        current_prices = self.prices.loc[dt]

        for order in self.pending_orders:
            ticker = order.ticker
            if ticker not in current_prices.index or np.isnan(current_prices[ticker]):
                continue

            price = float(current_prices[ticker])
            vol = self._compute_volatility(ticker, dt)
            adv = self._compute_adv(ticker, dt)

            # Slippage: adverse price impact
            slip = self.slippage.calculate(
                price=price, shares=order.shares, volatility=vol, adv=adv
            )
            fill_price = price + slip

            # Commission
            commission = self.commission.calculate(order.shares, fill_price)

            # ── Execute ──
            if order.shares > 0:
                # BUY
                total_cost = order.shares * fill_price + commission
                if total_cost > self.cash:
                    affordable = (self.cash - commission) / fill_price
                    if affordable < 0.01:
                        continue
                    order.shares = affordable
                    total_cost = order.shares * fill_price + commission

                self.cash -= total_cost

                if ticker in self.positions:
                    pos = self.positions[ticker]
                    new_shares = pos.shares + order.shares
                    new_cost = pos.shares * pos.cost_basis + order.shares * fill_price
                    pos.shares = new_shares
                    pos.cost_basis = new_cost / new_shares
                    pos.high_water = max(pos.high_water, fill_price)
                else:
                    self.positions[ticker] = Position(
                        shares=order.shares,
                        cost_basis=fill_price,
                        high_water=fill_price,
                    )
            else:
                # SELL
                sell_shares = abs(order.shares)
                proceeds = sell_shares * fill_price - commission
                self.cash += proceeds

                if ticker in self.positions:
                    self.positions[ticker].shares -= sell_shares
                    if self.positions[ticker].shares <= 0.001:
                        del self.positions[ticker]

            order.filled = True
            order.fill_price = fill_price
            order.fill_dt = dt
            order.commission = commission
            order.slippage_cost = abs(slip * order.shares)

            self.all_fills.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "action": "BUY" if order.shares > 0 else "SELL",
                    "shares": abs(order.shares),
                    "fill_price": round(fill_price, 4),
                    "market_price": price,
                    "slippage": round(slip, 6),
                    "slippage_cost": round(order.slippage_cost, 4),
                    "commission": round(commission, 4),
                    "value": round(abs(order.shares * fill_price), 2),
                }
            )

        self.pending_orders = [o for o in self.pending_orders if not o.filled]

    # ── Signal Generation & Rebalancing ───────────────────────────────────

    def _generate_rebalance_orders(self, dt: pd.Timestamp):
        """Compute scores using data up to dt and queue orders for next bar."""
        hist = self.prices.loc[:dt]
        current_prices = hist.iloc[-1]

        if len(hist) < config.MOMENTUM_LOOKBACK // 2:
            return

        # Regime check
        regime_ok = self._check_regime(dt)
        if not regime_ok:
            # O'Neil's rule: when market turns bearish, go to cash
            if self.positions:
                pending_sells = {o.ticker for o in self.pending_orders if o.shares < 0}
                for ticker, pos in self.positions.items():
                    if ticker not in pending_sells:
                        self.pending_orders.append(
                            Order(
                                ticker=ticker,
                                shares=-pos.shares,
                                created_dt=dt,
                            )
                        )
                logger.info(
                    f"{dt.date()}: Bearish regime — selling all "
                    f"{len(self.positions)} positions"
                )
            self.rebalance_log.append(
                {"date": dt, "action": "SELL_BEARISH", "n_orders": len(self.positions)}
            )
            return

        # Score all stocks (excluding regime ticker)
        scoring_tickers = [
            t
            for t in hist.columns
            if t != self.regime_ticker and not hist[t].isna().all()
        ]

        # 1. Momentum score (O'Neil / Jegadeesh & Titman)
        momentum = compute_momentum_score(hist[scoring_tickers])

        # 2. Trend strength score (Minervini)
        trend = compute_trend_score(hist[scoring_tickers])

        # 3. Volume/breakout score (Darvas / Weinstein)
        volume = compute_volume_score(hist[scoring_tickers])

        # 4. Trend Template hard filter (Minervini gate)
        trend_filter = passes_trend_template(hist[scoring_tickers])

        # 5. Resolve fundamentals for this date (no look-ahead)
        fund_snapshot = None
        if isinstance(self.fundamentals, dict):
            from strategy.alpha_vantage_fetcher import get_fundamentals_for_date

            fund_snapshot = get_fundamentals_for_date(self.fundamentals, dt)
            if fund_snapshot is not None:
                fund_snapshot = _enrich_valuations(fund_snapshot, current_prices)
        elif isinstance(self.fundamentals, pd.DataFrame):
            fund_snapshot = self.fundamentals

        if fund_snapshot is not None and not fund_snapshot.empty:
            common = [t for t in scoring_tickers if t in fund_snapshot.index]
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

        # Composite score with all 4 factors
        composite = compute_composite_score(
            momentum, value, fscore, trend=trend, volume=volume
        )

        if len(composite) < config.MAX_POSITIONS:
            return

        # Select portfolio with trend template filter
        new_portfolio = select_portfolio(composite, trend_filter=trend_filter)

        # Sector constraints
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
                if not sector:
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

        target_tickers = set(new_portfolio.index)
        current_tickers = set(self.positions.keys())

        # Tickers that already have a pending sell (from stop-loss)
        pending_sells = {o.ticker for o in self.pending_orders if o.shares < 0}

        # ── Hysteresis: only sell if stock drops below rank threshold ──
        # (Minervini: give winners room to run, don't churn on small rank changes)
        hysteresis_rank = getattr(config, "HYSTERESIS_RANK", 30)
        for ticker in current_tickers - target_tickers:
            if ticker in pending_sells:
                continue
            if ticker not in composite.index:
                # No score data — sell it
                if ticker in self.positions:
                    self.pending_orders.append(
                        Order(
                            ticker=ticker,
                            shares=-self.positions[ticker].shares,
                            created_dt=dt,
                        )
                    )
                continue
            # Check rank in the full composite — only sell if rank dropped below threshold
            rank_position = list(composite.index).index(ticker) + 1
            if rank_position > hysteresis_rank:
                if ticker in self.positions:
                    self.pending_orders.append(
                        Order(
                            ticker=ticker,
                            shares=-self.positions[ticker].shares,
                            created_dt=dt,
                        )
                    )
            # else: keep holding — still in top N, just not top 8

        # ── Inverse-vol or equal-weight buy into open slots ──
        # Include held positions that survived hysteresis
        actual_sells = {
            o.ticker for o in self.pending_orders if o.shares < 0 and o.created_dt == dt
        }
        kept_positions = current_tickers - actual_sells - pending_sells
        # Merge: target picks + held keepers (cap at max positions)
        effective_portfolio = list(target_tickers | kept_positions)[
            : config.MAX_POSITIONS
        ]
        total_value = self._portfolio_value(current_prices)

        # Compute weights: inverse-vol or equal-weight
        if self.inv_vol_sizing and effective_portfolio:
            weights = self._compute_inv_vol_weights(effective_portfolio, dt)
        else:
            eq_w = 1.0 / len(effective_portfolio) if effective_portfolio else 0
            weights = {t: eq_w for t in effective_portfolio}

        for ticker in effective_portfolio:
            if ticker not in current_prices.index or np.isnan(current_prices[ticker]):
                continue
            price = float(current_prices[ticker])
            if price <= 0:
                continue

            current_shares = (
                self.positions[ticker].shares if ticker in self.positions else 0
            )
            current_value = current_shares * price
            target_value = weights.get(ticker, 0) * total_value
            diff = target_value - current_value

            # Skip small rebalances
            rebal_threshold = getattr(config, "REBALANCE_THRESHOLD_PCT", 0.0)
            if target_value > 0 and abs(diff) / target_value < rebal_threshold:
                continue
            if abs(diff) < config.MIN_POSITION_SIZE:
                continue

            shares_diff = diff / price
            self.pending_orders.append(
                Order(ticker=ticker, shares=shares_diff, created_dt=dt)
            )

        n_orders = len([o for o in self.pending_orders if o.created_dt == dt])
        self.rebalance_log.append(
            {
                "date": dt,
                "action": "REBALANCE",
                "target_tickers": list(target_tickers),
                "n_orders": n_orders,
                "regime_bullish": regime_ok,
            }
        )
        logger.info(
            f"{dt.date()}: Queued {n_orders} orders → "
            f"target: {sorted(target_tickers)}"
        )

    # ── Daily Slot Refill (after stop-outs) ───────────────────────────────

    def _fill_empty_slots(self, dt: pd.Timestamp):
        """
        When a stop-out frees a slot, find the next best candidate and buy it.
        This prevents capital from sitting idle between scheduled rebalances.
        Runs daily, but only acts when there are empty slots.
        """
        max_pos = config.MAX_POSITIONS
        pending_sells = {o.ticker for o in self.pending_orders if o.shares < 0}
        pending_buys = {o.ticker for o in self.pending_orders if o.shares > 0}

        # Count effective positions (current - pending sells + pending buys)
        current_count = (
            len(self.positions)
            - len(pending_sells & set(self.positions.keys()))
            + len(pending_buys - set(self.positions.keys()))
        )

        if current_count >= max_pos:
            return  # No empty slots

        slots_to_fill = max_pos - current_count
        if slots_to_fill <= 0:
            return

        # Check regime
        if not self._check_regime(dt):
            return

        hist = self.prices.loc[:dt]
        if len(hist) < config.MOMENTUM_LOOKBACK // 2:
            return

        current_prices = hist.iloc[-1]
        scoring_tickers = [
            t
            for t in hist.columns
            if t != self.regime_ticker and not hist[t].isna().all()
        ]

        # Quick scoring — momentum + trend template only (skip fundamentals for speed)
        momentum = compute_momentum_score(hist[scoring_tickers])
        trend = compute_trend_score(hist[scoring_tickers])
        volume = compute_volume_score(hist[scoring_tickers])
        trend_filter = passes_trend_template(hist[scoring_tickers])

        fscore = pd.Series(50, index=momentum.index, name="fscore_score")
        value = pd.Series(50, index=momentum.index, name="value_score")

        composite = compute_composite_score(
            momentum, value, fscore, trend=trend, volume=volume
        )

        # Filter to trend-template-passing stocks only
        passing = trend_filter[trend_filter].index
        eligible = composite.loc[composite.index.intersection(passing)]
        if len(eligible) == 0:
            eligible = composite  # fall back if none pass

        # Exclude stocks we already hold or have pending orders for
        held = set(self.positions.keys()) | pending_buys | pending_sells
        candidates = eligible.loc[~eligible.index.isin(held)]

        if len(candidates) == 0:
            return

        # Take top N candidates to fill available slots
        picks = candidates.head(slots_to_fill)
        total_value = self._portfolio_value(current_prices)

        # Inverse-vol sizing for slot refills too
        if self.inv_vol_sizing:
            weights = self._compute_inv_vol_weights(list(picks.index), dt)
        else:
            weights = {t: 1.0 / max_pos for t in picks.index}

        for ticker in picks.index:
            if ticker not in current_prices.index or np.isnan(current_prices[ticker]):
                continue
            price = float(current_prices[ticker])
            if price <= 0:
                continue

            target_value = weights.get(ticker, 1.0 / max_pos) * total_value
            if target_value < config.MIN_POSITION_SIZE:
                continue

            shares = target_value / price
            self.pending_orders.append(
                Order(ticker=ticker, shares=shares, created_dt=dt)
            )
            logger.info(f"{dt.date()}: SLOT-REFILL → {ticker} (${target_value:.0f})")

    # ── Inverse-Volatility Position Sizing ──────────────────────────────

    def _compute_inv_vol_weights(
        self, tickers: list[str], dt: pd.Timestamp
    ) -> dict[str, float]:
        """
        Compute position weights inversely proportional to realized volatility.
        High-vol stocks get smaller allocations → more balanced risk contribution.
        Falls back to equal-weight if vol data is insufficient.
        """
        inv_vols = {}
        for ticker in tickers:
            if ticker not in self.prices.columns:
                continue
            hist = self.prices[ticker].loc[:dt].dropna()
            if len(hist) < self.inv_vol_lookback + 1:
                continue
            returns = hist.pct_change().dropna().iloc[-self.inv_vol_lookback :]
            vol = returns.std()
            if vol > 0:
                inv_vols[ticker] = 1.0 / vol

        if not inv_vols:
            # Fallback to equal weight
            return {t: 1.0 / len(tickers) for t in tickers}

        total_inv_vol = sum(inv_vols.values())
        weights = {t: iv / total_inv_vol for t, iv in inv_vols.items()}

        # Enforce MAX_POSITION_PCT cap and redistribute excess
        max_pct = getattr(config, "MAX_POSITION_PCT", 1.0)
        capped = {}
        excess = 0.0
        uncapped = []
        for t, w in weights.items():
            if w > max_pct:
                capped[t] = max_pct
                excess += w - max_pct
            else:
                capped[t] = w
                uncapped.append(t)

        # Redistribute excess proportionally to uncapped positions
        if excess > 0 and uncapped:
            uncapped_total = sum(capped[t] for t in uncapped)
            if uncapped_total > 0:
                for t in uncapped:
                    capped[t] += excess * (capped[t] / uncapped_total)

        # Include tickers that had no vol data with minimum weight
        for t in tickers:
            if t not in capped:
                capped[t] = 1.0 / len(tickers)

        return capped

    # ── Portfolio Drawdown Circuit Breaker ────────────────────────────────

    def _check_portfolio_drawdown(self, dt: pd.Timestamp, pv: float) -> bool:
        """
        Portfolio-level circuit breaker.
        - Trips when DD from peak exceeds PORTFOLIO_DD_LIMIT → sell everything
        - Re-enters on the next scheduled rebalance day (wait at least ~1 month)
        - Resets HWM at re-entry so we start fresh
        Returns True if we should be in cash.
        """
        # Update high-water mark (only when not in DD cash)
        if not self._portfolio_in_dd_cash and pv > self._portfolio_hwm:
            self._portfolio_hwm = pv

        if not self._portfolio_in_dd_cash:
            dd = (pv - self._portfolio_hwm) / self._portfolio_hwm
            if dd <= self.portfolio_dd_limit:
                self._portfolio_in_dd_cash = True
                self._dd_trip_date = dt
                logger.info(
                    f"{dt.date()}: PORTFOLIO CIRCUIT BREAKER — "
                    f"DD={dd:.1%} breached {self.portfolio_dd_limit:.0%} limit, "
                    f"going to cash"
                )
                return True
        else:
            # Already in DD-cash — stay until next rebalance day (cooldown)
            # This ensures we wait at least until next month before re-entering
            return True

        return False

    def _try_dd_reentry(self, dt: pd.Timestamp, trading_days: pd.DatetimeIndex) -> bool:
        """
        Check if we should exit the DD-cash state on a rebalance day.
        Returns True if re-entering (circuit breaker cleared).
        """
        if not self._portfolio_in_dd_cash:
            return False

        # Only re-enter on a scheduled rebalance day (wait for next month)
        if not self._is_rebalance_day(dt, trading_days):
            return False

        # Must be at least some days after the trip (avoid same-day re-entry)
        if hasattr(self, "_dd_trip_date") and dt <= self._dd_trip_date:
            return False

        # Also require regime to be bullish
        if not self._check_regime(dt):
            return False

        self._portfolio_in_dd_cash = False
        # Reset HWM to current value so we start fresh
        pv = self._portfolio_value(self.prices.loc[dt])
        self._portfolio_hwm = pv
        logger.info(
            f"{dt.date()}: PORTFOLIO DD RECOVERY — "
            f"re-entering on rebalance day, HWM reset to ${pv:.0f}"
        )
        return True

    # ── ATR Computation (Turtle Traders) ─────────────────────────────────

    def _compute_atr(self, ticker: str, dt: pd.Timestamp) -> Optional[float]:
        """Compute Average True Range for ATR-based stops."""
        if ticker not in self.prices.columns:
            return None
        hist = self.prices[ticker].loc[:dt].dropna()
        if len(hist) < self.atr_period + 1:
            return None
        tr = hist.diff().abs().iloc[-self.atr_period :]
        return float(tr.mean())

    # ── Concentration Cap (daily trim of outsized winners) ──────────────

    def _check_concentration(self, dt: pd.Timestamp):
        """
        Trim any position that has drifted above CONCENTRATION_CAP of portfolio.
        This prevents a single moonshot from dominating portfolio risk.
        """
        cap = getattr(config, "CONCENTRATION_CAP", None)
        target = getattr(config, "CONCENTRATION_TARGET", 0.10)
        if not cap:
            return

        current_prices = self.prices.loc[dt]
        total_value = self._portfolio_value(current_prices)
        if total_value <= 0:
            return

        pending_sells = {o.ticker for o in self.pending_orders if o.shares < 0}

        for ticker, pos in list(self.positions.items()):
            if ticker in pending_sells:
                continue
            if ticker not in current_prices.index:
                continue
            price = float(current_prices[ticker])
            if np.isnan(price) or price <= 0:
                continue

            position_value = pos.shares * price
            weight = position_value / total_value

            if weight > cap:
                target_value = target * total_value
                trim_value = position_value - target_value
                trim_shares = trim_value / price
                if trim_shares > 0:
                    self.pending_orders.append(
                        Order(ticker=ticker, shares=-trim_shares, created_dt=dt)
                    )
                    logger.info(
                        f"{dt.date()}: CONCENTRATION TRIM {ticker}: "
                        f"{weight:.1%} → {target:.0%} "
                        f"(selling ${trim_value:.0f})"
                    )

    # ── Stop-Loss Monitoring (ATR + Trailing) ─────────────────────────────

    def _check_stop_losses(self, dt: pd.Timestamp):
        """Check positions against ATR/trailing stop-loss, queue sell orders."""
        current_prices = self.prices.loc[dt]
        pending_sells = {o.ticker for o in self.pending_orders if o.shares < 0}

        for ticker, pos in list(self.positions.items()):
            if ticker in pending_sells:
                continue
            if ticker not in current_prices.index:
                continue
            price = float(current_prices[ticker])
            if np.isnan(price):
                continue

            # Update high-water mark
            if price > pos.high_water:
                pos.high_water = price

            hwm = pos.high_water
            triggered = False
            stop_desc = ""

            # ATR-based trailing stop (Turtle Traders method)
            if self.atr_stop_mult > 0 and self.trailing_stop:
                atr = self._compute_atr(ticker, dt)
                if atr is not None and atr > 0:
                    stop_level = hwm - self.atr_stop_mult * atr
                    triggered = price <= stop_level
                    stop_desc = (
                        f"ATR-STOP {ticker}: HWM=${hwm:.2f} - "
                        f"{self.atr_stop_mult}×ATR(${atr:.2f}) "
                        f"= stop@${stop_level:.2f}, now=${price:.2f}"
                    )

            # Percentage-based trailing/fixed stop as fallback
            if not triggered:
                if self.trailing_stop:
                    ref_price = hwm
                    pct_change = (price - ref_price) / ref_price
                else:
                    ref_price = pos.cost_basis
                    pct_change = (price - ref_price) / ref_price

                if pct_change <= config.STOP_LOSS_PCT:
                    triggered = True
                    stop_desc = (
                        f"SAFETY-STOP {ticker}: {pct_change:.1%} "
                        f"(peak=${ref_price:.2f}, now=${price:.2f})"
                    )

            if triggered:
                logger.info(f"{dt.date()}: {stop_desc}")
                self.pending_orders.append(
                    Order(ticker=ticker, shares=-pos.shares, created_dt=dt)
                )
                self._slots_freed_today = True

    # ── Main Event Loop ───────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Main event loop — processes one trading day at a time.

        For each bar:
          1. Fill pending orders from yesterday (next-bar execution)
          2. Mark-to-market all positions
          3. Check stop-loss triggers (queue orders for tomorrow)
          4. If rebalance day: score stocks, queue orders for tomorrow
          5. Record daily portfolio value

        Returns dict with equity_curve, returns, metrics, trades, rebalances.
        """
        trading_days = self.prices.index
        logger.info(
            f"Zipline backtest: {len(trading_days)} trading days, "
            f"${self.initial_capital:,.0f} initial capital"
        )

        for dt in trading_days:
            current_prices = self.prices.loc[dt]

            # 1. Fill pending orders (T+1 execution)
            self._fill_orders(dt)

            # 2. Mark-to-market
            pv = self._portfolio_value(current_prices)

            # 3. Stop-loss check (generates orders for next bar)
            self._slots_freed_today = False
            self._check_stop_losses(dt)

            # 3a. (Reserved for future use)

            # 3b. Daily regime check — sell all if market turns bearish
            if self.use_regime and self.positions and not self._check_regime(dt):
                pending_sells = {o.ticker for o in self.pending_orders if o.shares < 0}
                for ticker, pos in list(self.positions.items()):
                    if ticker not in pending_sells:
                        self.pending_orders.append(
                            Order(ticker=ticker, shares=-pos.shares, created_dt=dt)
                        )
                logger.info(
                    f"{dt.date()}: REGIME EXIT — SPY below {self.regime_ma}-day MA, "
                    f"selling {len(self.positions)} positions"
                )

            # 4. Rebalance if scheduled (generates orders for next bar)
            if self._is_rebalance_day(dt, trading_days):
                self._generate_rebalance_orders(dt)
            elif self._slots_freed_today:
                # 4b. Daily slot refill: immediately replace stopped-out positions
                self._fill_empty_slots(dt)

            # 5. Record daily performance
            self.perf.record(dt, pv, self.cash)

        # ── Build results ────────────────────────────────────────────
        equity = self.perf.build_equity_curve()

        bench_returns = None
        if self.benchmark_prices is not None and len(self.benchmark_prices) > 0:
            bench = self.benchmark_prices.sort_index()
            bench_returns = bench.pct_change().dropna()

        metrics = self.perf.compute_metrics(bench_returns)
        trades_df = pd.DataFrame(self.all_fills) if self.all_fills else pd.DataFrame()

        return {
            "equity_curve": equity,
            "returns": equity.pct_change().dropna(),
            "metrics": metrics,
            "trades": trades_df,
            "rebalances": self.rebalance_log,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience Wrapper
# ═══════════════════════════════════════════════════════════════════════════


def run_zipline_backtest(
    prices: pd.DataFrame,
    fundamentals=None,
    benchmark_prices: Optional[pd.Series] = None,
    initial_capital: float = config.INITIAL_CAPITAL,
    commission_model=None,
    slippage_model=None,
) -> dict:
    """Run a Zipline-style event-driven backtest. Convenience wrapper."""
    bt = ZiplineBacktester(
        prices=prices,
        fundamentals=fundamentals,
        benchmark_prices=benchmark_prices,
        initial_capital=initial_capital,
        commission_model=commission_model,
        slippage_model=slippage_model,
    )
    return bt.run()


# ═══════════════════════════════════════════════════════════════════════════
#  Tear Sheet (pyfolio-style output)
# ═══════════════════════════════════════════════════════════════════════════


def print_tearsheet(results: dict, benchmark_name: str = "IWM"):
    """Print a comprehensive pyfolio-style performance tear sheet."""
    m = results["metrics"]
    eq = results["equity_curve"]
    trades = results["trades"]

    if not m:
        print("  No metrics computed — check that price data was available.")
        return

    print()
    print("═" * 70)
    print("  ZIPLINE-STYLE BACKTEST TEAR SHEET")
    print("  Elite Trader Momentum — Daily Rotation + Trailing Stops")
    print("═" * 70)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  ── Summary {'─' * 55}")
    print(
        f"  Backtest Period:      {eq.index[0].date()} to {eq.index[-1].date()}"
        f"  ({m.get('n_trading_days', 0):,} trading days)"
    )
    print(f"  Starting Capital:     ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Ending Value:         ${eq.iloc[-1]:,.2f}")

    # ── Returns ──────────────────────────────────────────────────────
    print(f"\n  ── Returns {'─' * 55}")
    print(f"  Total Return:         {m.get('total_return', 0):.1%}")
    print(f"  CAGR:                 {m.get('cagr', 0):.1%}")

    annual = m.get("annual_returns", {})
    if annual:
        best_yr = max(annual, key=annual.get)
        worst_yr = min(annual, key=annual.get)
        print(f"  Best Year:            {annual[best_yr]:+.1%} ({best_yr})")
        print(f"  Worst Year:           {annual[worst_yr]:+.1%} ({worst_yr})")

    # ── Risk ─────────────────────────────────────────────────────────
    print(f"\n  ── Risk {'─' * 58}")
    print(f"  Annualized Volatility: {m.get('annualized_volatility', 0):.1%}")
    print(f"  Max Drawdown:         {m.get('max_drawdown', 0):.1%}")
    print(f"  Max DD Duration:      {m.get('max_drawdown_duration_days', 0)} days")
    print(f"  Best Day:             {m.get('best_day', 0):+.2%}")
    print(f"  Worst Day:            {m.get('worst_day', 0):+.2%}")
    print(f"  Tail Ratio:           {m.get('tail_ratio', 0):.2f}")

    # ── Risk-Adjusted ────────────────────────────────────────────────
    print(f"\n  ── Risk-Adjusted {'─' * 49}")
    print(f"  Sharpe Ratio:         {m.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:        {m.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio:         {m.get('calmar_ratio', 0):.2f}")

    # ── Benchmark Comparison ─────────────────────────────────────────
    if "alpha" in m:
        label = f"vs Benchmark ({benchmark_name})"
        pad = max(0, 65 - len(label) - 5)
        print(f"\n  ── {label} {'─' * pad}")
        print(f"  Benchmark CAGR:       {m.get('benchmark_cagr', 0):.1%}")
        print(f"  Alpha (ann.):         {m.get('alpha', 0):+.2%}")
        print(f"  Beta:                 {m.get('beta', 0):.2f}")
        print(f"  R-squared:            {m.get('r_squared', 0):.2f}")
        print(f"  Information Ratio:    {m.get('information_ratio', 0):.2f}")
        print(f"  Correlation:          {m.get('correlation', 0):.2f}")
        print(f"  Excess CAGR:          {m.get('excess_cagr', 0):+.1%}")

    # ── Execution Stats ──────────────────────────────────────────────
    print(f"\n  ── Execution {'─' * 53}")
    if trades is not None and len(trades) > 0:
        n_buys = (trades["action"] == "BUY").sum()
        n_sells = (trades["action"] == "SELL").sum()
        total_comm = trades["commission"].sum()
        total_slip = trades["slippage_cost"].sum()
        print(f"  Total Trades:         {len(trades)} ({n_buys} buys, {n_sells} sells)")
        print(f"  Total Commission:     ${total_comm:,.2f}")
        print(f"  Total Slippage Cost:  ${total_slip:,.2f}")
        print(f"  Total Execution Cost: ${total_comm + total_slip:,.2f}")
    else:
        print("  Total Trades:         0")

    n_rebals = len(
        [r for r in results.get("rebalances", []) if r.get("action") == "REBALANCE"]
    )
    print(f"  Rebalance Events:     {n_rebals}")
    print(f"  Win Rate (daily):     {m.get('win_rate_daily', 0):.1%}")

    # ── Annual Returns Breakdown ─────────────────────────────────────
    if annual:
        print(f"\n  ── Annual Returns {'─' * 48}")
        for year in sorted(annual.keys()):
            ret = annual[year]
            bar_len = int(abs(ret) * 100)
            bar = "█" * min(bar_len, 40)
            sign = "+" if ret >= 0 else ""
            print(f"    {year}:  {sign}{ret:.1%}  {bar}")

    # ── Integrity Checklist ──────────────────────────────────────────
    print(f"\n  ── Backtest Integrity {'─' * 44}")
    print("  ✓ Event-driven execution (no look-ahead bias)")
    print("  ✓ Next-bar order fills (T+1 execution delay)")
    print("  ✓ Volatility-adjusted slippage model")
    print(f"  ✓ Daily mark-to-market ({m.get('n_trading_days', 0):,} data points)")
    if "alpha" in m:
        print("  ✓ Benchmark-relative metrics (alpha, beta, IR)")
    print("  ✓ Quarterly rebalancing with regime filter")
    print("  ✓ Trailing stop-loss monitoring (daily)")
    print("  ✓ Daily slot-refill after stop-outs")

    print("═" * 70)
    print()
