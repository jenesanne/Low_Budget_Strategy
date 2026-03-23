"""
automate.py

Automates the full pipeline: parameter search, walk-forward backtesting, and reporting.
"""

import itertools
import json

import numpy as np

import config
from tools.walk_forward import walk_forward_backtest

# Define parameter grid (example: tune train/test years, weights, min F-score)
param_grid = {
    "train_years": [2, 3, 4],
    "test_years": [1],
    "MOMENTUM_WEIGHT": [0.3, 0.35, 0.4],
    "VALUE_WEIGHT": [0.2, 0.25, 0.3],
    "FSCORE_WEIGHT": [0.4, 0.45, 0.5],
    "MIN_FSCORE": [4, 5, 6],
}

# Generate all parameter combinations
grid = list(itertools.product(*param_grid.values()))

results = []
best_result = None
best_return = -np.inf

for params in grid:
    # Unpack params
    (
        train_years,
        test_years,
        MOMENTUM_WEIGHT,
        VALUE_WEIGHT,
        FSCORE_WEIGHT,
        MIN_FSCORE,
    ) = params
    # Patch config (monkeypatch for this run)
    config.MOMENTUM_WEIGHT = MOMENTUM_WEIGHT
    config.VALUE_WEIGHT = VALUE_WEIGHT
    config.FSCORE_WEIGHT = FSCORE_WEIGHT
    config.MIN_FSCORE = MIN_FSCORE
    try:
        all_equity, summary = walk_forward_backtest(
            config.UNIVERSE_TICKERS,
            start=config.BACKTEST_START,
            end=config.BACKTEST_END,
            train_years=train_years,
            test_years=test_years,
            initial_capital=config.INITIAL_CAPITAL,
        )
        avg_return = summary["total_return"].mean()
        result = {
            "params": dict(zip(param_grid.keys(), params)),
            "avg_total_return": avg_return,
            "final_value": summary["final_value"].mean(),
        }
        results.append(result)
        if avg_return > best_return:
            best_return = avg_return
            best_result = result
            all_equity.to_csv("output/best_walk_forward_equity.csv")
            summary.to_csv("output/best_walk_forward_summary.csv", index=False)
    except Exception as e:
        print(f"Params {params} failed: {e}")

# Save all results
with open("output/walk_forward_optimization_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Best result:", best_result)
print("All results saved to walk_forward_optimization_results.json")
