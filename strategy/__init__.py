"""
Elite Trader Momentum — Core Strategy Modules

Combined strategy from Mark Minervini, William O'Neil, Nicolas Darvas,
Stan Weinstein, Turtle Traders, and Piotroski F-Score.

Submodules:
    scoring             – Multi-factor scoring (momentum, trend, volume, quality)
    backtester          – Historical simulation engine with ATR-based stops
    risk_management     – Position sizing, stop-loss, portfolio metrics
    data_fetcher        – Alpaca API integration (prices, assets)
    alpha_vantage_fetcher – Alpha Vantage fundamentals & benchmark data
"""
