# Low-Cap Stock Strategy: Peer-Reviewed Research Foundation

> **DISCLAIMER**: This is for educational and research purposes only. This is NOT financial
> advice. Trading stocks involves significant risk of loss. Past performance does not
> guarantee future results. Never invest money you cannot afford to lose.

---

## 1. Executive Summary

This strategy combines **three peer-reviewed market anomalies** that have been shown to
produce excess returns specifically on small/low-cap stocks:

| Factor | Academic Source | Reported Returns |
|--------|---------------|-----------------|
| **Size Premium** | Fama & French (1993) | Small caps outperform large caps by ~3-5% annually |
| **Momentum** | Jegadeesh & Titman (1993) | ~12% annual excess return (winners minus losers) |
| **Value (F-Score)** | Piotroski (2000) | ~23% annual on high-F-Score small-cap value stocks |

The combined **Small-Cap Momentum + Value** strategy targets the intersection of these
anomalies to maximise expected returns on a small portfolio.

---

## 2. Peer-Reviewed Academic Papers

### 2.1 Jegadeesh & Titman (1993) — Momentum
- **Paper**: "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"
- **Journal**: *Journal of Finance*, Vol. 48, pp. 65-91
- **Finding**: Stocks that performed well over the past 3-12 months continue to outperform
  over the next 3-12 months. The strategy of buying past winners and selling past losers
  generates approximately **1% per month** (12% annualised) in excess returns.
- **Relevance**: Momentum is **stronger in small-cap stocks** due to lower analyst coverage
  and slower information diffusion.

### 2.2 Fama & French (1993) — Size Premium (SMB)
- **Paper**: "Common Risk Factors in the Returns on Stocks and Bonds"
- **Journal**: *Journal of Financial Economics*, Vol. 33, pp. 3-56
- **Finding**: Small-cap stocks (bottom decile by market capitalisation) have historically
  outperformed large-cap stocks by 3-5% per year. This "Small Minus Big" (SMB) factor
  is one of the three factors in the Fama-French model.
- **Relevance**: Directly supports focusing on low-cap stocks for higher expected returns.

### 2.3 Piotroski (2000) — F-Score Value Investing
- **Paper**: "Value Investing: The Use of Historical Financial Statement Information to
  Separate Winners from Losers"
- **Journal**: *Journal of Accounting Research*, Vol. 38, pp. 1-41
- **Finding**: A 9-point scoring system (F-Score) based on financial health indicators
  applied to **high book-to-market (value) stocks** generates a **23% annual return**.
  The strategy is most effective on **small-cap value stocks** with low analyst coverage.
- **Relevance**: Provides a systematic way to filter fundamentally strong small-cap stocks.

### 2.4 Asness, Moskowitz & Pedersen (2013) — Value and Momentum Everywhere
- **Paper**: "Value and Momentum Everywhere"
- **Journal**: *Journal of Finance*, Vol. 68, pp. 929-985
- **Finding**: Value and momentum are **negatively correlated** with each other, meaning
  combining them produces a **higher Sharpe ratio** than either alone. This holds across
  multiple asset classes and geographies.
- **Relevance**: Combining momentum ranking with value scoring reduces drawdowns
  while maintaining high returns.

### 2.5 O'Shaughnessy (2011) — What Works on Wall Street
- **Book**: *What Works on Wall Street* (4th Edition)
- **Finding**: The **best-performing composite strategy** over 50+ years was combining
  value factors (low P/S, low P/E) with momentum (6-month price strength) on
  **micro/small-cap stocks**, producing ~20-25% CAGR.
- **Relevance**: Empirical validation of the combined approach on the exact stock universe
  we're targeting.

---

## 3. The Combined Strategy: Small-Cap Momentum + Value

### 3.1 Stock Universe
- **Market cap**: £10M - £500M (AIM-listed UK stocks or US micro/small-caps)
- **Minimum liquidity**: Average daily volume > 10,000 shares
- **Exclude**: Financials, SPACs, shell companies, stocks < £0.05/share

### 3.2 Scoring System (0-100 Composite Score)

#### A. Piotroski F-Score (0-9 points, weighted 40%)
1. **Profitability**
   - ROA > 0 (+1)
   - Operating Cash Flow > 0 (+1)
   - Change in ROA > 0 (+1)
   - Cash flow from operations > Net Income (accrual quality) (+1)

2. **Leverage/Liquidity**
   - Change in leverage (long-term debt/assets ratio decreased) (+1)
   - Change in current ratio > 0 (+1)
   - No new equity issued in last year (+1)

3. **Operating Efficiency**
   - Change in gross margin > 0 (+1)
   - Change in asset turnover > 0 (+1)

#### B. Momentum Score (weighted 35%)
- **12-month return minus last month** (skip most recent month to avoid short-term reversal)
- Rank all stocks in universe, assign percentile score

#### C. Value Score (weighted 25%)
- **Composite of**: Low P/E, Low P/S, Low EV/EBITDA
- Rank all stocks on each metric, average the percentile ranks

### 3.3 Portfolio Construction
- **Top 10-15 stocks** by composite score
- **Equal-weight** positions (£66-100 per stock with £1,000 budget)
- **Rebalance monthly** (on the first trading day of each month)

### 3.4 Entry/Exit Rules
- **Buy**: Top 15 composite scores at rebalance
- **Sell**: Stock drops out of top 30 composite scores at rebalance
- **Stop-loss**: -25% from purchase price (hard stop)
- **Hysteresis band**: Only replace a holding if it drops below top 30
  (prevents excessive turnover from small rank changes)

---

## 4. Risk Management

### 4.1 Position Sizing (Critical for £1,000 Budget)
- Maximum 10% per position (£100)
- Minimum position size: £50 (due to trading costs)
- Account for spread costs on small-caps (often 1-3%)

### 4.2 Expected Drawdowns
Based on backtests of similar strategies:
- **Maximum drawdown**: -40% to -60% (small-cap strategies are volatile)
- **Average drawdown**: -15% to -25%
- **Recovery period**: 6-18 months typically

### 4.3 Trading Cost Budget
- With £1,000 and 10-15 stocks, use a **zero/low-commission broker**
  (e.g., Trading 212, Freetrade, IBKR for UK stocks)
- Budget 0.5-1% per round-trip trade for spreads on small-caps
- Monthly rebalance = ~5-7 trades/month = ~£5-10 in spread costs

---

## 5. Growth Roadmap: £1,000 → £100,000

### Conservative Scenario (15% CAGR — below academic estimates)
| Year | Portfolio Value | Monthly Contribution |
|------|----------------|---------------------|
| 0 | £1,000 | - |
| 1 | £1,150 | +£100/month |
| 2 | £2,725 | +£100/month |
| 3 | £4,534 | +£100/month |
| 5 | £9,091 | +£100/month |
| 7 | £15,254 | +£150/month |
| 10 | £30,000+ | +£200/month |

### Aggressive Scenario (25% CAGR — in line with academic findings)
| Year | Portfolio Value | Monthly Contribution |
|------|----------------|---------------------|
| 0 | £1,000 | - |
| 1 | £1,250 | +£100/month |
| 2 | £3,163 | +£100/month |
| 3 | £5,553 | +£100/month |
| 5 | £12,842 | +£100/month |
| 7 | £25,503 | +£200/month |
| 10 | £70,000+ | +£300/month |

### Reality Check
- **Reaching £100k from £1k purely from trading returns is extremely difficult**
- The most realistic path combines: strategy returns + regular contributions + compounding
- Even at an aggressive 25% CAGR with £200/month contributions, it takes ~8-10 years
- **Adding regular savings is essential** — the strategy alone won't get there quickly

---

## 6. Key Risks & Limitations

1. **Survivorship bias**: Academic studies may overstate returns due to excluding delisted/bankrupt stocks
2. **Capacity constraints**: Small-cap anomalies diminish as more capital enters
3. **Transaction costs**: Higher spreads on small-caps erode returns
4. **Liquidity risk**: Difficulty exiting positions during market stress
5. **Regime changes**: Past patterns may not persist in future markets
6. **Tax implications**: Frequent rebalancing creates taxable events (use ISA wrapper in UK)
7. **Emotional discipline**: -40% drawdowns are psychologically challenging

---

## 7. Recommended Broker & Wrapper (UK)

- **Stocks & Shares ISA**: Shields all gains from capital gains tax (critical for compounding)
- **Broker**: Trading 212 (zero commission, ISA available, fractional shares)
  or Interactive Brokers (wider AIM stock coverage, slightly higher costs)
- **Data source**: Yahoo Finance API (free), or LSE data via various APIs

---

## References

1. Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*, 48, 65-91.
2. Fama, E. & French, K. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33, 3-56.
3. Piotroski, J. (2000). Value Investing: The Use of Historical Financial Statement Information. *Journal of Accounting Research*, 38, 1-41.
4. Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and Momentum Everywhere. *Journal of Finance*, 68, 929-985.
5. O'Shaughnessy, J. (2011). What Works on Wall Street. 4th Edition. McGraw-Hill.
6. Carhart, M. (1997). On Persistence in Mutual Fund Performance. *Journal of Finance*, 52, 57-82.
