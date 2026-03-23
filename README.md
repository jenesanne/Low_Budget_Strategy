# Small-Cap Momentum + Value Strategy

A quantitative, research-backed trading strategy for US small-cap equities. Combines price momentum, fundamental value, and Piotroski F-Score quality screening to select an 8-stock concentrated portfolio, rebalanced quarterly.

Designed for small accounts ($1,000+) using Alpaca's commission-free fractional shares and Alpha Vantage fundamentals data.

## Performance (2020–2025 backtest, no look-ahead bias)

| Metric               | Strategy | IWM (Russell 2000) |
|----------------------|----------|-------------------|
| Total Return         | **96%**  | 68%               |
| CAGR                 | **13.4%**| 10.3%             |
| Sharpe Ratio         | 0.41     | —                 |
| Sortino Ratio        | 0.73     | —                 |
| Max Drawdown         | -33.4%   | —                 |
| **Excess CAGR**      | **+3.2%**| —                 |

> Backtest uses historical quarterly financials with a 90-day reporting lag to
> eliminate look-ahead bias. Trading costs and slippage are modelled.

## How It Works

1. **Universe** — ~150 US small/micro-caps ($10M–$500M market cap)
2. **Scoring** — Each stock ranked by a weighted composite:
   - **Momentum (35%)** — 6-month return, skip most recent 2 weeks (Jegadeesh & Titman 1993)
   - **F-Score (40%)** — Piotroski 9-point financial health screen (Piotroski 2000)
   - **Value (25%)** — P/E, P/S, EV/EBITDA (Fama & French 1993)
3. **Selection** — Top 8 stocks with F-Score ≥ 5, sector-constrained (≤30% per sector)
4. **Rebalance** — Quarterly (Mar, Jun, Sep, Dec), with 20% threshold to reduce churn
5. **Risk** — SPY regime filter (hold cash if SPY < 200-day MA), -25% hard stop-loss

## Project Structure

```
├── main.py                  # CLI: backtest, screen, status
├── trade_live.py            # Automated quarterly rebalancer (Alpaca)
├── config.py                # All strategy parameters
│
├── strategy/                # Core engine
│   ├── scoring.py           # Momentum + Value + F-Score composite
│   ├── backtester.py        # Historical simulation
│   ├── risk_management.py   # Position sizing, stop-loss, metrics
│   ├── data_fetcher.py      # Alpaca API (prices, assets)
│   └── alpha_vantage_fetcher.py  # Fundamentals & benchmark data
│
├── tools/                   # Utilities & analysis
│   ├── walk_forward.py      # Out-of-sample rolling backtest
│   ├── automate.py          # Parameter grid search
│   ├── sensitivity_analysis.py
│   ├── plot_equity_vs_spy.py
│   ├── plot_equity.py
│   ├── plot_walk_forward.py
│   ├── refresh_universe.py
│   ├── filter_small_caps.py
│   └── backtest_summary.py
│
├── data/
│   └── universe.txt         # Ticker universe (~150 small-caps)
│
├── output/                  # Generated (git-ignored)
│   ├── backtest_equity.csv
│   ├── backtest_trades.csv
│   └── equity_vs_spy.png
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

# 3. Run a backtest
python main.py backtest

# 4. Screen current top picks
python main.py screen

# 5. Check Alpaca account
python main.py status

# 6. Paper trade (dry run)
python trade_live.py

# 7. Paper trade (submit orders)
python trade_live.py --execute
```

## Automated Paper Trading (Linode/Server)

Add a cron job to rebalance quarterly on the 1st of March, June, September, December:

```cron
0 14 1 3,6,9,12 * cd /path/to/Low_Budget_Strategy && /path/to/.venv/bin/python trade_live.py --execute >> /var/log/strategy.log 2>&1
```

## Data Sources

| Source | What | Cost |
|--------|------|------|
| **Alpaca** | Daily OHLCV prices, order execution | Free (IEX feed) |
| **Alpha Vantage** | Quarterly financials (IS, BS, CF), company overview | Premium key required |

First backtest run fetches ~370 API calls from Alpha Vantage for historical fundamentals. Data is permanently cached to `.cache/` after the first fetch.

## Configuration

All parameters live in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITIONS` | 8 | Portfolio concentration |
| `MIN_FSCORE` | 5 | Quality floor (0–9 scale) |
| `MOMENTUM_WEIGHT` | 0.35 | Scoring weight |
| `FSCORE_WEIGHT` | 0.40 | Scoring weight |
| `VALUE_WEIGHT` | 0.25 | Scoring weight |
| `REBALANCE_FREQUENCY` | quarterly | Rebalance cadence |
| `REBALANCE_THRESHOLD_PCT` | 0.20 | Skip trades within 20% of target |
| `MAX_SECTOR_PCT` | 0.30 | Sector concentration limit |
| `STOP_LOSS_PCT` | -0.25 | Hard stop-loss per position |

## References

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance, 48, 65–91.
- Fama, E. & French, K. (1993). *Common Risk Factors in the Returns on Stocks and Bonds*. Journal of Financial Economics, 33, 3–56.
- Piotroski, J. (2000). *Value Investing: The Use of Historical Financial Statement Information*. Journal of Accounting Research, 38, 1–41.

## Disclaimer

This project is for research and educational purposes only. Trading involves risk of loss. Past backtest performance does not guarantee future results.
