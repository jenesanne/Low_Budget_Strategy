import numpy as np
import pandas as pd

# Load equity curve
eq = pd.read_csv("output/backtest_equity.csv", parse_dates=["date"])
eq.set_index("date", inplace=True)

# Calculate returns
eq["returns"] = eq["value"].pct_change()

# CAGR
start_val = eq["value"].iloc[0]
end_val = eq["value"].iloc[-1]
years = (eq.index[-1] - eq.index[0]).days / 365.25
cagr = (end_val / start_val) ** (1 / years) - 1

# Max drawdown
eq["cummax"] = eq["value"].cummax()
eq["drawdown"] = eq["value"] / eq["cummax"] - 1
max_dd = eq["drawdown"].min()

# Sharpe ratio (annualized, risk-free rate 4.5%)
rf = 0.045
excess_ret = eq["returns"] - (rf / 12)
sharpe = np.sqrt(12) * excess_ret.mean() / excess_ret.std()

print(f"CAGR: {cagr:.2%}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
