"""
sensitivity_analysis.py

Runs walk-forward backtests with different combinations of enhancements to diagnose which constraint most impacts performance.
"""

import importlib

import pandas as pd

import config
from tools.walk_forward import walk_forward_backtest

# Store original config values
orig = {
    "MIN_DOLLAR_VOLUME": config.MIN_DOLLAR_VOLUME,
    "SLIPPAGE_VOL_MULT": config.SLIPPAGE_VOL_MULT,
    "SLIPPAGE_LIQUIDITY_MULT": config.SLIPPAGE_LIQUIDITY_MULT,
}


def run_scenario(name, min_dollar_volume, slippage_vol_mult, slippage_liq_mult):
    config.MIN_DOLLAR_VOLUME = min_dollar_volume
    config.SLIPPAGE_VOL_MULT = slippage_vol_mult
    config.SLIPPAGE_LIQUIDITY_MULT = slippage_liq_mult
    importlib.reload(config)
    eq, summary = walk_forward_backtest(
        config.UNIVERSE_TICKERS,
        start=config.BACKTEST_START,
        end=config.BACKTEST_END,
        train_years=3,
        test_years=1,
        initial_capital=config.INITIAL_CAPITAL,
    )
    summary["scenario"] = name
    return summary


scenarios = [
    ("Baseline (no new constraints)", 0, 0.0, 0.0),
    ("Volatility Sizing Only", 0, orig["SLIPPAGE_VOL_MULT"], 0.0),
    ("Liquidity Filter Only", orig["MIN_DOLLAR_VOLUME"], 0.0, 0.0),
    (
        "Dynamic Slippage Only",
        0,
        orig["SLIPPAGE_VOL_MULT"],
        orig["SLIPPAGE_LIQUIDITY_MULT"],
    ),
    (
        "All Enhancements",
        orig["MIN_DOLLAR_VOLUME"],
        orig["SLIPPAGE_VOL_MULT"],
        orig["SLIPPAGE_LIQUIDITY_MULT"],
    ),
]

results = []
for name, min_dv, slip_vol, slip_liq in scenarios:
    print(f"Running: {name}")
    try:
        summary = run_scenario(name, min_dv, slip_vol, slip_liq)
        results.append(summary)
    except Exception as e:
        print(f"{name} failed: {e}")

all_results = pd.concat(results, ignore_index=True)
all_results.to_csv("output/sensitivity_results.csv", index=False)
print("Sensitivity analysis complete. Results saved to output/sensitivity_results.csv.")
