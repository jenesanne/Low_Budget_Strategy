import matplotlib.pyplot as plt
import pandas as pd

from strategy.alpha_vantage_fetcher import fetch_monthly_adjusted_close


def plot_equity_vs_spy(equity_csv="output/backtest_equity.csv", ticker="SPY"):
    # Load strategy equity
    df = pd.read_csv(equity_csv, parse_dates=["date"])
    df = df.set_index("date")

    # Fetch SPY monthly adjusted close from Alpha Vantage
    spy_df = fetch_monthly_adjusted_close(ticker)
    if spy_df.empty:
        raise ValueError(f"No Alpha Vantage data found for {ticker}")
    spy_df = spy_df.set_index("date")
    # Align SPY to available strategy dates only
    spy_df = spy_df.reindex(df.index, method="ffill")

    # Normalize both to 1000 initial value
    equity_norm = df["value"] / df["value"].iloc[0] * 1000
    spy_norm = spy_df["adj_close"] / spy_df["adj_close"].iloc[0] * 1000

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, equity_norm, label="Strategy", color="blue")
    plt.plot(df.index, spy_norm, label=f"{ticker} (Benchmark)", color="orange")
    plt.title("Equity Curve vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value ($1000 = Start)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/equity_vs_spy.png", dpi=150)
    print("Chart saved to output/equity_vs_spy.png")
    plt.show()


if __name__ == "__main__":
    plot_equity_vs_spy()
