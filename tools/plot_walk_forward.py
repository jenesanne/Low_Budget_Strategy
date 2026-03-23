"""
plot_walk_forward.py

Visualizes the results of the rolling walk-forward backtest.
- Plots each window's equity curve
- Plots final value per window
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_walk_forward_equity(
    equity_csv="output/walk_forward_equity.csv",
    summary_csv="output/walk_forward_summary.csv",
):
    equity = pd.read_csv(equity_csv, index_col=0, parse_dates=True)
    summary = pd.read_csv(summary_csv)
    plt.figure(figsize=(12, 6))
    for col in equity.columns:
        plt.plot(equity.index, equity[col], label=col)
    plt.title("Walk-Forward Out-of-Sample Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(summary["window"], summary["final_value"])
    plt.title("Final Portfolio Value per Walk-Forward Window")
    plt.xlabel("Window")
    plt.ylabel("Final Value ($)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_walk_forward_equity()
