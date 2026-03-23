import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(csv_path="output/backtest_equity.csv"):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["value"], label="Strategy Equity Curve", color="blue")
    plt.title("Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_equity_curve()
