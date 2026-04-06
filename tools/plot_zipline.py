"""
Plot Zipline-style backtest results: equity curve, drawdown, and benchmark comparison.

Usage:
    python tools/plot_zipline.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import config


def plot_zipline_results(
    equity_path: str = "output/zipline_equity.csv",
    benchmark_path: str = "output/backtest_equity.csv",
):
    """Plot the Zipline backtest equity curve with drawdown and benchmark."""
    eq = pd.read_csv(equity_path, parse_dates=["date"], index_col="date")
    if "value" not in eq.columns:
        eq.columns = ["value"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "Zipline-Style Backtest — Small-Cap Momentum + Value + F-Score",
        fontsize=14,
        fontweight="bold",
    )

    # ── Panel 1: Equity Curve ────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(eq.index, eq["value"], color="#2196F3", linewidth=1.5, label="Strategy")

    # Overlay benchmark if available
    try:
        bench = pd.read_csv(benchmark_path, parse_dates=["date"], index_col="date")
        if "value" not in bench.columns:
            bench.columns = ["value"]
        # Normalise benchmark to same starting capital
        bench_norm = bench["value"] / bench["value"].iloc[0] * config.INITIAL_CAPITAL
        ax1.plot(
            bench_norm.index,
            bench_norm.values,
            color="#FF9800",
            linewidth=1.0,
            alpha=0.7,
            label=config.BENCHMARK_TICKER,
        )
    except Exception:
        pass  # no benchmark file, skip

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Daily Returns ───────────────────────────────────────
    ax2 = axes[1]
    returns = eq["value"].pct_change().dropna()
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in returns]
    ax2.bar(returns.index, returns.values, color=colors, alpha=0.6, width=2)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Daily Return")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Drawdown ────────────────────────────────────────────
    ax3 = axes[2]
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    ax3.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4)
    ax3.plot(dd.index, dd.values, color="#D32F2F", linewidth=0.8)
    ax3.set_ylabel("Drawdown")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/zipline_tearsheet.png", dpi=150, bbox_inches="tight")
    print("Saved to output/zipline_tearsheet.png")
    plt.show()


if __name__ == "__main__":
    plot_zipline_results()
