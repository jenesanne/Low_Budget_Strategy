# Elite Trader Momentum — Small-Cap Strategy

A quantitative momentum strategy combining methods from six profitable traders: **Minervini** (SEPA trend template), **O'Neil** (CANSLIM), **Darvas** (box theory / volume breakouts), **Weinstein** (stage analysis), **Turtle Traders** (ATR-based stops), and **Piotroski** (F-Score quality gate).

Built for small accounts ($1,000+) using Alpaca's commission-free fractional shares and Alpha Vantage fundamentals data.

## Performance (2020–2025 Zipline-style backtest)

| Metric               | Strategy    | IWM (Russell 2000) |
|----------------------|-------------|-------------------|
| Total Return         | **225.6%**  | 68.1%             |
| CAGR                 | **24.3%**   | 10.0%             |
| Sharpe Ratio         | 0.52        | —                 |
| Sortino Ratio        | 0.73        | —                 |
| Max Drawdown         | -58.3%      | —                 |
| **Excess CAGR**      | **+14.3%**  | —                 |

> Event-driven simulation: next-bar execution (T+1), volatility-adjusted slippage,
> no look-ahead bias. Historical quarterly financials lagged 90 days.

![Equity Curve](output/equity_vs_iwm.png)

## How It Works

1. **Universe** — ~128 US small/micro-caps ($10M–$2B market cap) from `data/universe.txt`
2. **Scoring** — Each stock ranked by a weighted composite:
   - **Momentum (35%)** — 6-month return, skip most recent 2 weeks (Jegadeesh & Titman 1993)
   - **Trend Strength (25%)** — Price position vs 50/150/200-day MAs (Minervini / Weinstein)
   - **Volume (20%)** — Breakout volume, expansion, up-day ratio (Darvas / O'Neil)
   - **F-Score (20%)** — Piotroski 9-point financial health screen (Piotroski 2000)
3. **Filters** — Minervini trend template (5 of 7 criteria), F-Score ≥ 4
4. **Selection** — Top 10 stocks, equal-weight
5. **Rebalance** — Monthly + daily slot-refill after stop-outs
6. **Risk Management**:
   - 6× ATR trailing stops (Turtle Traders)
   - -20% safety stop from high-water mark
   - Sell hysteresis — only sell if rank drops below 30
   - SPY regime filter with hysteresis — go to cash when SPY > 2% below 200-day MA, re-enter when 1% above
   - Go-to-cash on bearish regime (O'Neil's "M" rule)

## Project Structure

```
├── main.py                  # CLI: backtest, zipline-backtest, screen, status
├── trade_live.py            # Automated monthly rebalancer (Alpaca)
├── dashboard.py             # Web dashboard (Flask, dark theme)
├── config.py                # All strategy parameters
├── deploy.sh                # One-command Linode/server setup
│
├── strategy/                # Core engine
│   ├── scoring.py           # Multi-factor composite (6 trader methods)
│   ├── zipline_backtester.py # Event-driven Zipline-style backtester
│   ├── backtester.py        # Original daily backtester
│   ├── risk_management.py   # Position sizing, stop-loss, metrics
│   ├── data_fetcher.py      # Alpaca API (prices, assets)
│   └── alpha_vantage_fetcher.py  # Fundamentals & benchmark data
│
├── tools/                   # Utilities & analysis
│   ├── walk_forward.py      # Out-of-sample rolling backtest
│   ├── sensitivity_analysis.py
│   ├── plot_equity_vs_spy.py
│   ├── plot_equity.py
│   ├── plot_walk_forward.py
│   ├── plot_zipline.py
│   ├── refresh_universe.py
│   ├── filter_small_caps.py
│   └── backtest_summary.py
│
├── data/
│   └── universe.txt         # Ticker universe (~128 small/micro-caps)
│
├── output/                  # Generated (git-ignored)
│   ├── zipline_equity.csv
│   ├── zipline_trades.csv
│   └── equity_vs_iwm.png
│
├── docs/
│   └── STRATEGY_RESEARCH.md # Full academic methodology
│
├── requirements.txt
├── .env.example             # API key template
└── .gitignore
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/jenesanne/Low_Budget_Strategy.git
cd Low_Budget_Strategy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your Alpaca + Alpha Vantage keys

# 3. Run the Zipline-style backtest
python main.py zipline-backtest

# 4. Screen current top picks
python main.py screen

# 5. Check Alpaca account
python main.py status

# 6. Paper trade (dry run)
python trade_live.py

# 7. Paper trade (submit orders)
python trade_live.py --execute

# 8. Launch monitoring dashboard
python dashboard.py
```

## Automated Paper Trading (Linode/Server)

One-command deployment:

```bash
ssh root@YOUR_LINODE_IP 'bash -s' < deploy.sh
# Then edit /opt/low_budget_strategy/.env with your API keys
```

The deploy script sets up a cron job to rebalance monthly on the 1st trading day:

```cron
30 14 1 * * cd /opt/low_budget_strategy && .venv/bin/python trade_live.py --execute >> /var/log/strategy.log 2>&1
```

## Data Sources

| Source | What | Cost |
|--------|------|------|
| **Alpaca** | Daily OHLCV prices, order execution | Free (IEX feed) |
| **Alpha Vantage** | Quarterly financials (IS, BS, CF), company overview | Premium key recommended (75 calls/min) |

First backtest run fetches fundamentals from Alpha Vantage. Data is permanently cached to `.cache/` after the first fetch.

## Configuration

All parameters in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITIONS` | 10 | Portfolio size |
| `MOMENTUM_WEIGHT` | 0.35 | Momentum scoring weight |
| `TREND_WEIGHT` | 0.25 | Trend strength weight |
| `VOLUME_WEIGHT` | 0.20 | Volume breakout weight |
| `FSCORE_WEIGHT` | 0.20 | Piotroski F-Score weight |
| `MIN_TREND_CRITERIA` | 5 | Minervini template (5 of 7) |
| `MIN_FSCORE` | 4 | Quality floor (0–9 scale) |
| `ATR_STOP_MULT` | 6 | 6× ATR trailing stop |
| `STOP_LOSS_PCT` | -0.20 | Safety stop from peak |
| `HYSTERESIS_RANK` | 30 | Sell hysteresis threshold |
| `REBALANCE_FREQUENCY` | monthly | Rebalance cadence |
| `USE_REGIME_FILTER` | True | SPY 200-day MA filter |

## References

- Minervini, M. (2013). *Trade Like a Stock Market Wizard*. McGraw-Hill.
- O'Neil, W. (2009). *How to Make Money in Stocks*. McGraw-Hill. 4th ed.
- Darvas, N. (1960). *How I Made $2,000,000 in the Stock Market*. Lyle Stuart.
- Weinstein, S. (1988). *Secrets for Profiting in Bull and Bear Markets*. McGraw-Hill.
- Covel, M. (2007). *The Complete TurtleTrader*. HarperCollins.
- Piotroski, J. (2000). *Value Investing: The Use of Historical Financial Statement Information*. Journal of Accounting Research, 38, 1–41.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance, 48, 65–91.

## Disclaimer

This project is for research and educational purposes only. Trading involves risk of loss. Past backtest performance does not guarantee future results. Not financial advice.
