"""
dashboard.py — Web monitoring dashboard for the Low-Budget Strategy.

A lightweight Flask app that shows account status, positions, orders,
and portfolio allocation. Mobile-friendly Bootstrap dark theme.

Usage:
    python dashboard.py                   # localhost:5000
    python dashboard.py --host 0.0.0.0    # accessible from network
    python dashboard.py --port 8080       # custom port
"""

import argparse
import csv
import json
import logging
import math
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

load_dotenv()

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

app = Flask(__name__)
app.secret_key = os.environ.get("DASHBOARD_SECRET_KEY", secrets.token_hex(32))

# Simple password protection
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")

# ── Alpaca Client ───────────────────────────────────────────────────────────


def _get_client():
    from alpaca.trading.client import TradingClient

    return TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
        paper=(config.ALPACA_ENVIRONMENT == "paper"),
    )


def _get_account():
    client = _get_client()
    return client.get_account()


def _get_positions():
    client = _get_client()
    return client.get_all_positions()


def _get_orders(limit=20):
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    client = _get_client()
    req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
    return client.get_orders(req)


def _get_portfolio_history():
    """Get portfolio equity history from Alpaca."""
    import requests as req

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    base = (
        "https://paper-api.alpaca.markets"
        if config.ALPACA_ENVIRONMENT == "paper"
        else "https://api.alpaca.markets"
    )
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    resp = req.get(
        f"{base}/v2/account/portfolio/history",
        headers=headers,
        params={"period": "3M", "timeframe": "1D"},
        timeout=10,
    )
    if resp.status_code == 200:
        return resp.json()
    return None


def _last_rebalance_log():
    """Read last rebalance info from the log file."""
    log_path = Path("/var/log/strategy.log")
    if not log_path.exists():
        log_path = Path(__file__).parent / "output" / "strategy.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text().strip().split("\n")
        # Find last "Rebalance complete" or "DRY RUN" line
        for line in reversed(lines):
            if (
                "DRY RUN" in line
                or "Rebalance complete" in line
                or "Account equity" in line
            ):
                return line
        return lines[-1] if lines else None
    except Exception:
        return None


def _next_rebalance():
    """Calculate the next quarterly rebalance date."""
    now = datetime.now(timezone.utc)
    rebal_months = [3, 6, 9, 12]
    for m in rebal_months:
        candidate = datetime(
            now.year, m, config.REBALANCE_DAY, 14, 30, tzinfo=timezone.utc
        )
        if candidate > now:
            return candidate
    # Next year Q1
    return datetime(now.year + 1, 3, config.REBALANCE_DAY, 14, 30, tzinfo=timezone.utc)


# ── Auth ────────────────────────────────────────────────────────────────────


def _requires_auth():
    """Check if auth is enabled and user is logged in."""
    if not DASHBOARD_PASSWORD:
        return False
    return not session.get("authenticated")


# ── Routes ──────────────────────────────────────────────────────────────────


@app.route("/login", methods=["GET", "POST"])
def login():
    if not DASHBOARD_PASSWORD:
        return redirect(url_for("index"))
    error = ""
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), DASHBOARD_PASSWORD):
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Wrong password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/")
def index():
    if _requires_auth():
        return redirect(url_for("login"))
    try:
        account = _get_account()
        positions = _get_positions()
        orders = _get_orders(limit=15)
    except Exception as e:
        return render_template_string(ERROR_HTML, error=str(e))

    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    last_equity = float(account.last_equity)

    # Strategy uses a fixed capital allocation, not the full account
    capital = config.INITIAL_CAPITAL

    pos_data = []
    total_market_value = 0
    total_unrealized_pl = 0
    for p in positions:
        mv = float(p.market_value)
        upl = float(p.unrealized_pl)
        upl_pct = float(p.unrealized_plpc) * 100
        total_market_value += mv
        total_unrealized_pl += upl
        pos_data.append(
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": mv,
                "unrealized_pl": upl,
                "unrealized_pl_pct": upl_pct,
                "side": p.side,
            }
        )
    pos_data.sort(key=lambda x: x["unrealized_pl"], reverse=True)

    # Strategy value = initial capital + unrealised P&L from our positions
    strategy_value = capital + total_unrealized_pl
    strategy_cash = capital - total_market_value + total_unrealized_pl
    day_pnl = equity - last_equity  # approximate from account-level
    day_pnl_pct = (day_pnl / capital * 100) if capital else 0

    # Portfolio allocation for chart
    alloc_labels = [p["symbol"] for p in pos_data]
    alloc_values = [abs(p["market_value"]) for p in pos_data]
    if cash > 0:
        alloc_labels.append("Cash")
        alloc_values.append(cash)

    order_data = []
    for o in orders:
        filled_at = ""
        if o.filled_at:
            filled_at = o.filled_at.strftime("%Y-%m-%d %H:%M")
        elif o.submitted_at:
            filled_at = o.submitted_at.strftime("%Y-%m-%d %H:%M")
        order_data.append(
            {
                "symbol": o.symbol,
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "qty": str(o.qty or o.notional or ""),
                "filled_qty": str(o.filled_qty or "0"),
                "status": (
                    o.status.value if hasattr(o.status, "value") else str(o.status)
                ),
                "time": filled_at,
                "filled_avg_price": str(o.filled_avg_price or ""),
            }
        )

    next_reb = _next_rebalance()
    days_until = (next_reb - datetime.now(timezone.utc)).days
    last_log = _last_rebalance_log()

    return render_template_string(
        DASHBOARD_HTML,
        equity=strategy_value,
        cash=strategy_cash,
        capital=capital,
        day_pnl=day_pnl,
        day_pnl_pct=day_pnl_pct,
        total_unrealized_pl=total_unrealized_pl,
        positions=pos_data,
        num_positions=len(pos_data),
        orders=order_data,
        alloc_labels=json.dumps(alloc_labels),
        alloc_values=json.dumps(alloc_values),
        next_rebalance=next_reb.strftime("%b %d, %Y"),
        days_until=days_until,
        last_log=last_log or "No runs yet",
        env=config.ALPACA_ENVIRONMENT.upper(),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        max_positions=config.MAX_POSITIONS,
    )


@app.route("/api/equity-history")
def api_equity_history():
    if _requires_auth():
        return jsonify({"error": "unauthorized"}), 401
    try:
        data = _get_portfolio_history()
        if data:
            return jsonify(data)
        return jsonify({"error": "no data"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Backtest Helpers ────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "output"


def _load_backtest_equity():
    """Load zipline equity curve CSV into list of (date_str, value) tuples."""
    path = OUTPUT_DIR / "zipline_equity.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["date"][:10], float(row["value"])))
    return rows


def _load_backtest_trades():
    """Load zipline trades CSV."""
    path = OUTPUT_DIR / "zipline_trades.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "date": row["date"][:10],
                    "ticker": row["ticker"],
                    "action": row["action"],
                    "shares": round(float(row["shares"]), 2),
                    "fill_price": round(float(row["fill_price"]), 2),
                    "market_price": round(float(row["market_price"]), 2),
                    "slippage_cost": round(float(row["slippage_cost"]), 4),
                    "commission": round(float(row["commission"]), 4),
                    "value": round(float(row["value"]), 2),
                }
            )
    return rows


def _compute_backtest_metrics(equity_rows):
    """Compute performance metrics from equity time series."""
    if len(equity_rows) < 2:
        return {}

    values = [v for _, v in equity_rows]
    initial = values[0]
    final = values[-1]

    # Daily returns
    returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            returns.append(values[i] / values[i - 1] - 1)

    if not returns:
        return {}

    total_return = final / initial - 1

    # Parse dates for CAGR
    first_date = datetime.strptime(equity_rows[0][0], "%Y-%m-%d")
    last_date = datetime.strptime(equity_rows[-1][0], "%Y-%m-%d")
    n_years = max((last_date - first_date).days / 365.25, 0.01)
    cagr = (1 + total_return) ** (1 / n_years) - 1

    # Volatility
    avg_ret = sum(returns) / len(returns)
    var = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
    daily_vol = math.sqrt(var)
    annual_vol = daily_vol * math.sqrt(252)

    # Sharpe
    risk_free_daily = getattr(config, "RISK_FREE_RATE", 0.05) / 252
    excess = [r - risk_free_daily for r in returns]
    avg_excess = sum(excess) / len(excess)
    sharpe = (avg_excess / daily_vol * math.sqrt(252)) if daily_vol > 0 else 0

    # Sortino
    downside = [r for r in returns if r < 0]
    if downside:
        down_var = sum(r**2 for r in downside) / len(downside)
        down_vol = math.sqrt(down_var) * math.sqrt(252)
        sortino = (
            (avg_ret * 252 - getattr(config, "RISK_FREE_RATE", 0.05)) / down_vol
            if down_vol > 0
            else 0
        )
    else:
        sortino = 0

    # Drawdown
    peak = values[0]
    max_dd = 0
    drawdowns = []
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0
        drawdowns.append(dd)
        if dd < max_dd:
            max_dd = dd

    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0

    # Win rate
    win_days = sum(1 for r in returns if r > 0)
    win_rate = win_days / len(returns) if returns else 0

    # Best / worst
    best_day = max(returns)
    worst_day = min(returns)

    # Annual returns
    annual = {}
    for date_str, val in equity_rows:
        year = int(date_str[:4])
        if year not in annual:
            annual[year] = {"first": val, "last": val}
        annual[year]["last"] = val
    annual_returns = {}
    years_sorted = sorted(annual.keys())
    for i, y in enumerate(years_sorted):
        start_val = annual[y]["first"]
        if i > 0:
            prev_year = years_sorted[i - 1]
            start_val = annual[prev_year]["last"]
        annual_returns[y] = annual[y]["last"] / start_val - 1 if start_val > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "best_day": best_day,
        "worst_day": worst_day,
        "n_days": len(returns),
        "initial_capital": values[0],
        "final_value": values[-1],
        "annual_returns": annual_returns,
        "drawdowns": drawdowns,
    }


# ── Backtest Routes ─────────────────────────────────────────────────────────


@app.route("/backtest")
def backtest():
    if _requires_auth():
        return redirect(url_for("login"))

    equity_rows = _load_backtest_equity()
    trades = _load_backtest_trades()
    metrics = _compute_backtest_metrics(equity_rows)

    if not equity_rows:
        return render_template_string(
            ERROR_HTML,
            error="No backtest data found. Run: python main.py zipline-backtest",
        )

    # Trade summary stats
    buys = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]
    total_slippage = sum(t["slippage_cost"] for t in trades)

    return render_template_string(
        BACKTEST_HTML,
        metrics=metrics,
        trades=trades[-50:],  # last 50 trades
        total_trades=len(trades),
        total_buys=len(buys),
        total_sells=len(sells),
        total_slippage=total_slippage,
        annual_returns=json.dumps(metrics.get("annual_returns", {})),
        env=config.ALPACA_ENVIRONMENT.upper(),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        # Config params for tuning panel
        cfg_stop_loss=config.STOP_LOSS_PCT,
        cfg_trailing_stop=config.TRAILING_STOP,
        cfg_momentum_weight=config.MOMENTUM_WEIGHT,
        cfg_value_weight=config.VALUE_WEIGHT,
        cfg_fscore_weight=config.FSCORE_WEIGHT,
        cfg_min_fscore=config.MIN_FSCORE,
        cfg_max_positions=config.MAX_POSITIONS,
        cfg_rebalance_frequency=config.REBALANCE_FREQUENCY,
        cfg_use_regime_filter=config.USE_REGIME_FILTER,
        cfg_momentum_lookback=config.MOMENTUM_LOOKBACK,
    )


@app.route("/api/backtest-equity")
def api_backtest_equity():
    if _requires_auth():
        return jsonify({"error": "unauthorized"}), 401
    equity_rows = _load_backtest_equity()
    if not equity_rows:
        return jsonify({"error": "no data"}), 404
    dates = [r[0] for r in equity_rows]
    values = [r[1] for r in equity_rows]

    # Drawdown series
    peak = values[0]
    drawdowns = []
    for v in values:
        if v > peak:
            peak = v
        drawdowns.append(round((v - peak) / peak * 100, 2) if peak > 0 else 0)

    return jsonify({"dates": dates, "values": values, "drawdowns": drawdowns})


# ── Backtest Runner (background thread) ─────────────────────────────────────

_backtest_state = {
    "running": False,
    "progress": "",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "params": {},
}
_backtest_lock = threading.Lock()


def _run_backtest_thread(params):
    """Run zipline backtest in background with overridden config params."""

    import pandas as pd

    from strategy.alpha_vantage_fetcher import (
        fetch_fundamentals_for_scoring,
        fetch_historical_fundamentals,
    )
    from strategy.data_fetcher import fetch_price_data
    from strategy.zipline_backtester import run_zipline_backtest

    try:
        with _backtest_lock:
            _backtest_state["progress"] = "Loading price data..."

        # Temporarily override config values
        original_values = {}
        param_map = {
            "stop_loss_pct": ("STOP_LOSS_PCT", float),
            "trailing_stop": ("TRAILING_STOP", lambda v: v == "true"),
            "momentum_weight": ("MOMENTUM_WEIGHT", float),
            "value_weight": ("VALUE_WEIGHT", float),
            "fscore_weight": ("FSCORE_WEIGHT", float),
            "min_fscore": ("MIN_FSCORE", int),
            "max_positions": ("MAX_POSITIONS", int),
            "rebalance_frequency": ("REBALANCE_FREQUENCY", str),
            "use_regime_filter": ("USE_REGIME_FILTER", lambda v: v == "true"),
            "momentum_lookback": ("MOMENTUM_LOOKBACK", int),
        }
        for form_key, (config_attr, converter) in param_map.items():
            if form_key in params and params[form_key] != "":
                original_values[config_attr] = getattr(config, config_attr)
                setattr(config, config_attr, converter(params[form_key]))

        tickers = config.UNIVERSE_TICKERS
        regime_ticker = getattr(config, "REGIME_TICKER", None)
        if regime_ticker and regime_ticker not in tickers:
            tickers = tickers + [regime_ticker]

        prices = fetch_price_data(tickers)
        if prices.empty:
            raise RuntimeError("No price data retrieved — check API keys")

        with _backtest_lock:
            _backtest_state["progress"] = (
                f"Loaded {prices.shape[1]} tickers, {prices.shape[0]} days. Fetching fundamentals..."
            )

        scoring_tickers = [
            t for t in prices.columns if t != getattr(config, "REGIME_TICKER", "SPY")
        ]
        _ = fetch_fundamentals_for_scoring(scoring_tickers)
        fundamentals = fetch_historical_fundamentals(scoring_tickers)
        if not fundamentals:
            fundamentals = fetch_fundamentals_for_scoring(scoring_tickers)
            if isinstance(fundamentals, pd.DataFrame) and fundamentals.empty:
                fundamentals = None

        with _backtest_lock:
            _backtest_state["progress"] = "Running Zipline backtest..."

        benchmark_prices = fetch_price_data(
            [config.BENCHMARK_TICKER],
            start=config.BACKTEST_START,
            end=config.BACKTEST_END,
        )
        bench_series = None
        if (
            not benchmark_prices.empty
            and config.BENCHMARK_TICKER in benchmark_prices.columns
        ):
            bench_series = benchmark_prices[config.BENCHMARK_TICKER]

        results = run_zipline_backtest(
            prices,
            fundamentals=fundamentals,
            benchmark_prices=bench_series,
            initial_capital=config.INITIAL_CAPITAL,
        )

        with _backtest_lock:
            _backtest_state["progress"] = "Saving results..."

        if results["trades"] is not None and len(results["trades"]) > 0:
            results["trades"].to_csv("output/zipline_trades.csv", index=False)
        results["equity_curve"].to_csv("output/zipline_equity.csv")

        with _backtest_lock:
            _backtest_state["running"] = False
            _backtest_state["finished_at"] = time.time()
            _backtest_state["progress"] = "Complete"
            _backtest_state["error"] = None

    except Exception as e:
        logger.exception("Backtest failed")
        with _backtest_lock:
            _backtest_state["running"] = False
            _backtest_state["finished_at"] = time.time()
            _backtest_state["error"] = str(e)
            _backtest_state["progress"] = "Failed"
    finally:
        # Restore original config values
        for config_attr, original_val in original_values.items():
            setattr(config, config_attr, original_val)


@app.route("/backtest/run", methods=["POST"])
def backtest_run():
    if _requires_auth():
        return jsonify({"error": "unauthorized"}), 401

    with _backtest_lock:
        if _backtest_state["running"]:
            return jsonify({"error": "Backtest already running"}), 409

    # Collect parameters from form
    params = {}
    for key in [
        "stop_loss_pct",
        "trailing_stop",
        "momentum_weight",
        "value_weight",
        "fscore_weight",
        "min_fscore",
        "max_positions",
        "rebalance_frequency",
        "use_regime_filter",
        "momentum_lookback",
    ]:
        val = request.form.get(key, "")
        if val != "":
            params[key] = val

    with _backtest_lock:
        _backtest_state["running"] = True
        _backtest_state["started_at"] = time.time()
        _backtest_state["finished_at"] = None
        _backtest_state["error"] = None
        _backtest_state["progress"] = "Starting..."
        _backtest_state["params"] = params

    thread = threading.Thread(target=_run_backtest_thread, args=(params,), daemon=True)
    thread.start()

    return jsonify({"status": "started", "params": params})


@app.route("/api/backtest-status")
def api_backtest_status():
    if _requires_auth():
        return jsonify({"error": "unauthorized"}), 401
    with _backtest_lock:
        elapsed = None
        if _backtest_state["started_at"]:
            end = _backtest_state["finished_at"] or time.time()
            elapsed = round(end - _backtest_state["started_at"], 1)
        return jsonify(
            {
                "running": _backtest_state["running"],
                "progress": _backtest_state["progress"],
                "error": _backtest_state["error"],
                "elapsed_seconds": elapsed,
            }
        )


# ── HTML Templates ──────────────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strategy Login</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="d-flex align-items-center justify-content-center" style="min-height:100vh">
  <div class="card p-4" style="max-width:340px;width:100%">
    <h5 class="card-title text-center mb-3">🔒 Strategy Dashboard</h5>
    {% if error %}<div class="alert alert-danger py-1">{{ error }}</div>{% endif %}
    <form method="post">
      <input type="password" name="password" class="form-control mb-3" placeholder="Password" autofocus>
      <button type="submit" class="btn btn-primary w-100">Login</button>
    </form>
  </div>
</body>
</html>
"""

ERROR_HTML = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard Error</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4">
  <div class="alert alert-danger">
    <h5>API Error</h5>
    <p>{{ error }}</p>
    <a href="/" class="btn btn-outline-light btn-sm mt-2">Retry</a>
  </div>
</body>
</html>
"""

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Low-Budget Strategy</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root { --accent: #00d4aa; }
    body { background: #0d1117; font-family: -apple-system, system-ui, sans-serif; }
    .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1rem; }
    .stat-value { font-size: 1.5rem; font-weight: 700; }
    .stat-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .gain { color: #3fb950; }
    .loss { color: #f85149; }
    .badge-env { background: #1f6feb; font-size: 0.7rem; vertical-align: middle; }
    .table { font-size: 0.85rem; }
    .table th { color: #8b949e; font-weight: 600; border-color: #30363d; }
    .table td { border-color: #21262d; vertical-align: middle; }
    .order-buy { color: #3fb950; }
    .order-sell { color: #f85149; }
    .next-reb { background: linear-gradient(135deg, #161b22, #1c2333); border: 1px solid #30363d; border-radius: 12px; padding: 1rem; }
    .mini-chart { height: 200px; }
    @media (max-width: 576px) { .stat-value { font-size: 1.2rem; } }
    .refresh-btn { position: fixed; bottom: 1rem; right: 1rem; z-index: 100; border-radius: 50%; width: 48px; height: 48px; }
    .nav-link-custom { color: #8b949e; text-decoration: none; font-size: 0.85rem; }
    .nav-link-custom:hover { color: var(--accent); }
    .nav-link-custom.active { color: var(--accent); border-bottom: 2px solid var(--accent); }
  </style>
</head>
<body>
  <div class="container-fluid py-3 px-3">
    <!-- Header -->
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div>
        <h5 class="mb-0">
          <i class="bi bi-graph-up-arrow" style="color:var(--accent)"></i>
          Low-Budget Strategy
          <span class="badge badge-env ms-1">{{ env }}</span>
        </h5>
        <small class="text-muted">{{ now }}</small>
      </div>
      <a href="/" class="btn btn-outline-secondary btn-sm"><i class="bi bi-arrow-clockwise"></i></a>
    </div>

    <!-- Nav -->
    <div class="d-flex gap-3 mb-3 border-bottom" style="border-color:#30363d!important">
      <a href="/" class="nav-link-custom active pb-2">
        <i class="bi bi-speedometer2"></i> Dashboard
      </a>
      <a href="/backtest" class="nav-link-custom pb-2">
        <i class="bi bi-bar-chart-line"></i> Backtest
      </a>
    </div>

    <!-- Account Stats -->
    <div class="row g-2 mb-3">
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Strategy Value</div>
          <div class="stat-value">${{ "{:,.2f}".format(equity) }}</div>
          <div class="stat-label" style="font-size:0.65rem">Capital: ${{ "{:,.0f}".format(capital) }}</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Day P&L</div>
          <div class="stat-value {{ 'gain' if day_pnl >= 0 else 'loss' }}">
            {{ "{:+,.2f}".format(day_pnl) }}
            <small>({{ "{:+.2f}".format(day_pnl_pct) }}%)</small>
          </div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Cash</div>
          <div class="stat-value">${{ "{:,.2f}".format(cash) }}</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Unrealised P&L</div>
          <div class="stat-value {{ 'gain' if total_unrealized_pl >= 0 else 'loss' }}">
            {{ "{:+,.2f}".format(total_unrealized_pl) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Equity Chart + Allocation -->
    <div class="row g-2 mb-3">
      <div class="col-md-8">
        <div class="stat-card">
          <div class="stat-label mb-2">Equity History (3M)</div>
          <div class="mini-chart">
            <canvas id="equityChart"></canvas>
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="stat-card">
          <div class="stat-label mb-2">Allocation</div>
          <div class="mini-chart">
            <canvas id="allocChart"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- Next Rebalance -->
    <div class="next-reb mb-3 d-flex justify-content-between align-items-center">
      <div>
        <div class="stat-label">Next Rebalance</div>
        <div class="fw-bold">{{ next_rebalance }}</div>
      </div>
      <div class="text-end">
        <span class="badge bg-secondary fs-6">{{ days_until }}d</span>
        <div class="stat-label mt-1">{{ num_positions }}/{{ max_positions }} positions</div>
      </div>
    </div>

    <!-- Positions -->
    <div class="stat-card mb-3">
      <div class="stat-label mb-2"><i class="bi bi-briefcase"></i> Positions ({{ num_positions }})</div>
      {% if positions %}
      <div class="table-responsive">
        <table class="table table-sm mb-0">
          <thead>
            <tr>
              <th>Symbol</th>
              <th class="text-end">Qty</th>
              <th class="text-end d-none d-sm-table-cell">Entry</th>
              <th class="text-end">Price</th>
              <th class="text-end">P&L</th>
            </tr>
          </thead>
          <tbody>
            {% for p in positions %}
            <tr>
              <td class="fw-bold">{{ p.symbol }}</td>
              <td class="text-end">{{ "{:.2f}".format(p.qty) }}</td>
              <td class="text-end d-none d-sm-table-cell">${{ "{:.2f}".format(p.avg_entry) }}</td>
              <td class="text-end">${{ "{:.2f}".format(p.current_price) }}</td>
              <td class="text-end {{ 'gain' if p.unrealized_pl >= 0 else 'loss' }}">
                {{ "{:+,.2f}".format(p.unrealized_pl) }}<br>
                <small>{{ "{:+.1f}".format(p.unrealized_pl_pct) }}%</small>
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% else %}
      <p class="text-muted mb-0">No open positions yet. First rebalance hasn't run.</p>
      {% endif %}
    </div>

    <!-- Recent Orders -->
    <div class="stat-card mb-3">
      <div class="stat-label mb-2"><i class="bi bi-clock-history"></i> Recent Orders</div>
      {% if orders %}
      <div class="table-responsive">
        <table class="table table-sm mb-0">
          <thead>
            <tr>
              <th>Time</th>
              <th>Symbol</th>
              <th>Side</th>
              <th class="text-end">Qty</th>
              <th class="text-end d-none d-sm-table-cell">Price</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {% for o in orders %}
            <tr>
              <td class="text-muted" style="font-size:0.75rem">{{ o.time }}</td>
              <td class="fw-bold">{{ o.symbol }}</td>
              <td class="{{ 'order-buy' if o.side == 'buy' else 'order-sell' }}">{{ o.side | upper }}</td>
              <td class="text-end">{{ o.filled_qty }}/{{ o.qty }}</td>
              <td class="text-end d-none d-sm-table-cell">{{ o.filled_avg_price }}</td>
              <td>
                {% if o.status == 'filled' %}
                  <span class="badge bg-success">filled</span>
                {% elif o.status == 'canceled' or o.status == 'cancelled' %}
                  <span class="badge bg-secondary">cancelled</span>
                {% elif o.status == 'new' or o.status == 'accepted' %}
                  <span class="badge bg-info">{{ o.status }}</span>
                {% else %}
                  <span class="badge bg-warning">{{ o.status }}</span>
                {% endif %}
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% else %}
      <p class="text-muted mb-0">No orders yet.</p>
      {% endif %}
    </div>

    <!-- Last Run Log -->
    <div class="stat-card mb-5">
      <div class="stat-label mb-1"><i class="bi bi-terminal"></i> Last Run</div>
      <code class="text-muted" style="font-size:0.75rem; word-break:break-all">{{ last_log }}</code>
    </div>
  </div>

  <!-- Floating refresh -->
  <a href="/" class="btn btn-primary refresh-btn shadow d-md-none">
    <i class="bi bi-arrow-clockwise"></i>
  </a>

  <script>
    // Equity chart
    fetch('/api/equity-history')
      .then(r => r.json())
      .then(data => {
        if (data.error) return;
        const labels = data.timestamp.map(t => {
          const d = new Date(t * 1000);
          return (d.getMonth()+1) + '/' + d.getDate();
        });
        new Chart(document.getElementById('equityChart'), {
          type: 'line',
          data: {
            labels: labels,
            datasets: [{
              data: data.equity,
              borderColor: '#00d4aa',
              backgroundColor: 'rgba(0,212,170,0.1)',
              fill: true,
              tension: 0.3,
              pointRadius: 0,
              borderWidth: 2,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { display: true, grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 6, font: { size: 10 } } },
              y: { display: true, grid: { color: '#21262d' }, ticks: { color: '#8b949e', font: { size: 10 },
                callback: v => '$' + v.toLocaleString() } }
            }
          }
        });
      })
      .catch(() => {
        document.getElementById('equityChart').parentElement.innerHTML +=
          '<p class="text-muted text-center" style="font-size:0.8rem">No history yet</p>';
      });

    // Allocation donut
    const allocLabels = {{ alloc_labels | safe }};
    const allocValues = {{ alloc_values | safe }};
    if (allocLabels.length > 0) {
      const colors = ['#00d4aa','#1f6feb','#f0883e','#a371f7','#3fb950','#f85149','#db61a2','#79c0ff','#d2a8ff','#ffa657','#7ee787'];
      new Chart(document.getElementById('allocChart'), {
        type: 'doughnut',
        data: {
          labels: allocLabels,
          datasets: [{
            data: allocValues,
            backgroundColor: colors.slice(0, allocLabels.length),
            borderWidth: 0,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '60%',
          plugins: {
            legend: { position: 'bottom', labels: { color: '#8b949e', font: { size: 10 }, boxWidth: 10, padding: 6 } }
          }
        }
      });
    }

    // Auto-refresh every 60s
    setTimeout(() => location.reload(), 60000);
  </script>
</body>
</html>
"""

BACKTEST_HTML = r"""
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest — Low-Budget Strategy</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root { --accent: #00d4aa; }
    body { background: #0d1117; font-family: -apple-system, system-ui, sans-serif; }
    .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1rem; }
    .stat-value { font-size: 1.5rem; font-weight: 700; }
    .stat-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .gain { color: #3fb950; }
    .loss { color: #f85149; }
    .neutral { color: #8b949e; }
    .badge-env { background: #1f6feb; font-size: 0.7rem; vertical-align: middle; }
    .table { font-size: 0.85rem; }
    .table th { color: #8b949e; font-weight: 600; border-color: #30363d; }
    .table td { border-color: #21262d; vertical-align: middle; }
    .order-buy { color: #3fb950; }
    .order-sell { color: #f85149; }
    .chart-container { height: 300px; }
    .chart-sm { height: 180px; }
    .nav-link-custom { color: #8b949e; text-decoration: none; font-size: 0.85rem; }
    .nav-link-custom:hover { color: var(--accent); }
    .nav-link-custom.active { color: var(--accent); border-bottom: 2px solid var(--accent); }
    .metric-row { display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #21262d; }
    .metric-row:last-child { border-bottom: none; }
    .metric-name { color: #8b949e; font-size: 0.85rem; }
    .metric-val { font-weight: 600; font-size: 0.85rem; }
    @media (max-width: 576px) { .stat-value { font-size: 1.2rem; } .chart-container { height: 220px; } }
    .param-input { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px;
                   padding: 0.3rem 0.5rem; font-size: 0.8rem; width: 100%; }
    .param-input:focus { border-color: var(--accent); outline: none; }
    .param-label { color: #8b949e; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
    .param-select { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px;
                    padding: 0.3rem 0.5rem; font-size: 0.8rem; width: 100%; }
    .run-btn { background: var(--accent); color: #0d1117; border: none; font-weight: 700; border-radius: 8px; padding: 0.5rem 1.5rem; }
    .run-btn:hover { background: #00b894; color: #0d1117; }
    .run-btn:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; }
    #runStatus { font-size: 0.8rem; }
    .spinner-border-sm { width: 1rem; height: 1rem; border-width: 0.15em; }
    .collapse-toggle { cursor: pointer; user-select: none; }
    .collapse-toggle i { transition: transform 0.2s; }
    .collapse-toggle.collapsed i { transform: rotate(-90deg); }
  </style>
</head>
<body>
  <div class="container-fluid py-3 px-3">
    <!-- Header -->
    <div class="d-flex justify-content-between align-items-center mb-2">
      <div>
        <h5 class="mb-0">
          <i class="bi bi-lightning-charge" style="color:var(--accent)"></i>
          Low-Budget Strategy
          <span class="badge badge-env ms-1">{{ env }}</span>
        </h5>
        <small class="text-muted">{{ now }}</small>
      </div>
    </div>

    <!-- Nav -->
    <div class="d-flex gap-3 mb-3 border-bottom" style="border-color:#30363d!important">
      <a href="/" class="nav-link-custom pb-2">
        <i class="bi bi-speedometer2"></i> Dashboard
      </a>
      <a href="/backtest" class="nav-link-custom active pb-2">
        <i class="bi bi-bar-chart-line"></i> Backtest
      </a>
    </div>

    <!-- Parameter Tuning Panel -->
    <div class="stat-card mb-3">
      <div class="d-flex justify-content-between align-items-center collapse-toggle" data-bs-toggle="collapse" data-bs-target="#tuningPanel">
        <div class="stat-label mb-0"><i class="bi bi-sliders"></i> Strategy Parameters</div>
        <i class="bi bi-chevron-down" style="color:#8b949e; font-size:0.8rem"></i>
      </div>
      <div class="collapse" id="tuningPanel">
        <form id="backtestForm" class="mt-3">
          <div class="row g-2 mb-2">
            <div class="col-6 col-md-3">
              <div class="param-label">Stop Loss %</div>
              <input type="number" name="stop_loss_pct" class="param-input" step="0.05"
                     value="{{ cfg_stop_loss }}" min="-0.80" max="-0.05">
            </div>
            <div class="col-6 col-md-3">
              <div class="param-label">Trailing Stop</div>
              <select name="trailing_stop" class="param-select">
                <option value="true" {{ 'selected' if cfg_trailing_stop else '' }}>Trailing (from peak)</option>
                <option value="false" {{ 'selected' if not cfg_trailing_stop else '' }}>Fixed (from entry)</option>
              </select>
            </div>
            <div class="col-6 col-md-3">
              <div class="param-label">Max Positions</div>
              <input type="number" name="max_positions" class="param-input"
                     value="{{ cfg_max_positions }}" min="3" max="20">
            </div>
            <div class="col-6 col-md-3">
              <div class="param-label">Rebalance</div>
              <select name="rebalance_frequency" class="param-select">
                <option value="monthly" {{ 'selected' if cfg_rebalance_frequency == 'monthly' else '' }}>Monthly</option>
                <option value="quarterly" {{ 'selected' if cfg_rebalance_frequency == 'quarterly' else '' }}>Quarterly</option>
              </select>
            </div>
          </div>
          <div class="row g-2 mb-2">
            <div class="col-4 col-md-2">
              <div class="param-label">Momentum Wt</div>
              <input type="number" name="momentum_weight" class="param-input" step="0.05"
                     value="{{ cfg_momentum_weight }}" min="0" max="1">
            </div>
            <div class="col-4 col-md-2">
              <div class="param-label">Value Wt</div>
              <input type="number" name="value_weight" class="param-input" step="0.05"
                     value="{{ cfg_value_weight }}" min="0" max="1">
            </div>
            <div class="col-4 col-md-2">
              <div class="param-label">F-Score Wt</div>
              <input type="number" name="fscore_weight" class="param-input" step="0.05"
                     value="{{ cfg_fscore_weight }}" min="0" max="1">
            </div>
            <div class="col-4 col-md-2">
              <div class="param-label">Min F-Score</div>
              <input type="number" name="min_fscore" class="param-input"
                     value="{{ cfg_min_fscore }}" min="0" max="9">
            </div>
            <div class="col-4 col-md-2">
              <div class="param-label">Momentum Lookback</div>
              <input type="number" name="momentum_lookback" class="param-input"
                     value="{{ cfg_momentum_lookback }}" min="21" max="252">
            </div>
            <div class="col-4 col-md-2">
              <div class="param-label">Regime Filter</div>
              <select name="use_regime_filter" class="param-select">
                <option value="true" {{ 'selected' if cfg_use_regime_filter else '' }}>On (SPY>200MA)</option>
                <option value="false" {{ 'selected' if not cfg_use_regime_filter else '' }}>Off</option>
              </select>
            </div>
          </div>
          <div class="d-flex align-items-center gap-3 mt-2">
            <button type="submit" class="run-btn" id="runBtn">
              <i class="bi bi-play-fill"></i> Run Backtest
            </button>
            <span id="runStatus"></span>
          </div>
        </form>
      </div>
    </div>

    <!-- Top Metrics Row -->
    <div class="row g-2 mb-3">
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Total Return</div>
          <div class="stat-value {{ 'gain' if metrics.total_return >= 0 else 'loss' }}">
            {{ "{:+.1%}".format(metrics.total_return) }}
          </div>
          <div class="stat-label" style="font-size:0.65rem">
            ${{ "{:,.2f}".format(metrics.initial_capital) }} → ${{ "{:,.2f}".format(metrics.final_value) }}
          </div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">CAGR</div>
          <div class="stat-value {{ 'gain' if metrics.cagr >= 0 else 'loss' }}">
            {{ "{:+.1%}".format(metrics.cagr) }}
          </div>
          <div class="stat-label" style="font-size:0.65rem">{{ metrics.n_days }} trading days</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Max Drawdown</div>
          <div class="stat-value loss">
            {{ "{:.1%}".format(metrics.max_drawdown) }}
          </div>
          <div class="stat-label" style="font-size:0.65rem">Calmar: {{ "{:.2f}".format(metrics.calmar) }}</div>
        </div>
      </div>
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Sharpe Ratio</div>
          <div class="stat-value {{ 'gain' if metrics.sharpe > 0 else 'loss' }}">
            {{ "{:.2f}".format(metrics.sharpe) }}
          </div>
          <div class="stat-label" style="font-size:0.65rem">Sortino: {{ "{:.2f}".format(metrics.sortino) }}</div>
        </div>
      </div>
    </div>

    <!-- Equity Curve + Drawdown -->
    <div class="row g-2 mb-3">
      <div class="col-12">
        <div class="stat-card">
          <div class="stat-label mb-2"><i class="bi bi-graph-up"></i> Equity Curve</div>
          <div class="chart-container">
            <canvas id="equityChart"></canvas>
          </div>
        </div>
      </div>
    </div>
    <div class="row g-2 mb-3">
      <div class="col-12">
        <div class="stat-card">
          <div class="stat-label mb-2"><i class="bi bi-graph-down"></i> Drawdown</div>
          <div class="chart-sm">
            <canvas id="drawdownChart"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- Detailed Metrics + Annual Returns -->
    <div class="row g-2 mb-3">
      <div class="col-md-6">
        <div class="stat-card">
          <div class="stat-label mb-2"><i class="bi bi-clipboard-data"></i> Performance Metrics</div>
          <div class="metric-row">
            <span class="metric-name">Annualized Volatility</span>
            <span class="metric-val">{{ "{:.1%}".format(metrics.volatility) }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Sharpe Ratio</span>
            <span class="metric-val {{ 'gain' if metrics.sharpe > 0 else 'loss' }}">{{ "{:.2f}".format(metrics.sharpe) }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Sortino Ratio</span>
            <span class="metric-val {{ 'gain' if metrics.sortino > 0 else 'loss' }}">{{ "{:.2f}".format(metrics.sortino) }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Calmar Ratio</span>
            <span class="metric-val {{ 'gain' if metrics.calmar > 0 else 'loss' }}">{{ "{:.2f}".format(metrics.calmar) }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Max Drawdown</span>
            <span class="metric-val loss">{{ "{:.1%}".format(metrics.max_drawdown) }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Win Rate (Daily)</span>
            <span class="metric-val">{{ "{:.1%}".format(metrics.win_rate) }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Best Day</span>
            <span class="metric-val gain">{{ "{:+.2%}".format(metrics.best_day) }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Worst Day</span>
            <span class="metric-val loss">{{ "{:+.2%}".format(metrics.worst_day) }}</span>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="stat-card">
          <div class="stat-label mb-2"><i class="bi bi-calendar3"></i> Annual Returns</div>
          <div class="chart-sm">
            <canvas id="annualChart"></canvas>
          </div>
        </div>
        <div class="stat-card mt-2">
          <div class="stat-label mb-2"><i class="bi bi-arrow-left-right"></i> Execution Stats</div>
          <div class="metric-row">
            <span class="metric-name">Total Trades</span>
            <span class="metric-val">{{ total_trades }}</span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Buys / Sells</span>
            <span class="metric-val"><span class="gain">{{ total_buys }}</span> / <span class="loss">{{ total_sells }}</span></span>
          </div>
          <div class="metric-row">
            <span class="metric-name">Total Slippage</span>
            <span class="metric-val loss">${{ "{:,.2f}".format(total_slippage) }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Trades Table -->
    <div class="stat-card mb-5">
      <div class="stat-label mb-2"><i class="bi bi-receipt"></i> Recent Trades (last 50 of {{ total_trades }})</div>
      {% if trades %}
      <div class="table-responsive">
        <table class="table table-sm mb-0">
          <thead>
            <tr>
              <th>Date</th>
              <th>Ticker</th>
              <th>Side</th>
              <th class="text-end">Shares</th>
              <th class="text-end">Fill Price</th>
              <th class="text-end d-none d-sm-table-cell">Mkt Price</th>
              <th class="text-end d-none d-md-table-cell">Slippage</th>
              <th class="text-end">Value</th>
            </tr>
          </thead>
          <tbody>
            {% for t in trades | reverse %}
            <tr>
              <td class="text-muted" style="font-size:0.75rem">{{ t.date }}</td>
              <td class="fw-bold">{{ t.ticker }}</td>
              <td class="{{ 'order-buy' if t.action == 'BUY' else 'order-sell' }}">{{ t.action }}</td>
              <td class="text-end">{{ t.shares }}</td>
              <td class="text-end">${{ "{:.2f}".format(t.fill_price) }}</td>
              <td class="text-end d-none d-sm-table-cell">${{ "{:.2f}".format(t.market_price) }}</td>
              <td class="text-end d-none d-md-table-cell loss">${{ "{:.4f}".format(t.slippage_cost) }}</td>
              <td class="text-end">${{ "{:,.2f}".format(t.value) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% else %}
      <p class="text-muted mb-0">No trades recorded.</p>
      {% endif %}
    </div>
  </div>

  <script>
    // Fetch equity + drawdown data
    fetch('/api/backtest-equity')
      .then(r => r.json())
      .then(data => {
        if (data.error) return;

        // Equity Chart
        new Chart(document.getElementById('equityChart'), {
          type: 'line',
          data: {
            labels: data.dates,
            datasets: [{
              label: 'Portfolio',
              data: data.values,
              borderColor: '#00d4aa',
              backgroundColor: 'rgba(0,212,170,0.08)',
              fill: true,
              tension: 0.2,
              pointRadius: 0,
              borderWidth: 2,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx => '$' + ctx.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2})
                }
              }
            },
            scales: {
              x: { display: true, grid: { color: '#21262d' },
                   ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 10 } } },
              y: { display: true, grid: { color: '#21262d' },
                   ticks: { color: '#8b949e', font: { size: 10 },
                            callback: v => '$' + v.toLocaleString() } }
            }
          }
        });

        // Drawdown Chart
        new Chart(document.getElementById('drawdownChart'), {
          type: 'line',
          data: {
            labels: data.dates,
            datasets: [{
              label: 'Drawdown',
              data: data.drawdowns,
              borderColor: '#f85149',
              backgroundColor: 'rgba(248,81,73,0.15)',
              fill: true,
              tension: 0.2,
              pointRadius: 0,
              borderWidth: 1.5,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + '%' }
              }
            },
            scales: {
              x: { display: true, grid: { color: '#21262d' },
                   ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 10 } } },
              y: { display: true, grid: { color: '#21262d' },
                   ticks: { color: '#8b949e', font: { size: 10 },
                            callback: v => v + '%' } }
            }
          }
        });
      })
      .catch(() => {
        document.getElementById('equityChart').parentElement.innerHTML +=
          '<p class="text-muted text-center" style="font-size:0.8rem">Failed to load equity data</p>';
      });

    // Annual Returns Bar Chart
    const annualData = {{ annual_returns | safe }};
    const years = Object.keys(annualData).sort();
    const annualVals = years.map(y => (annualData[y] * 100).toFixed(1));
    const barColors = annualVals.map(v => parseFloat(v) >= 0 ? '#3fb950' : '#f85149');
    if (years.length > 0) {
      new Chart(document.getElementById('annualChart'), {
        type: 'bar',
        data: {
          labels: years,
          datasets: [{
            data: annualVals,
            backgroundColor: barColors,
            borderRadius: 4,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, ticks: { color: '#8b949e', font: { size: 11 } } },
            y: { grid: { color: '#21262d' },
                 ticks: { color: '#8b949e', font: { size: 10 },
                          callback: v => v + '%' } }
          }
        }
      });
    }

    // ── Run Backtest form ──────────────────────────────────────────────
    const form = document.getElementById('backtestForm');
    const runBtn = document.getElementById('runBtn');
    const runStatus = document.getElementById('runStatus');

    form.addEventListener('submit', function(e) {
      e.preventDefault();
      runBtn.disabled = true;
      runBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Running...';
      runStatus.innerHTML = '<span class="text-muted">Starting backtest...</span>';

      const formData = new FormData(form);

      fetch('/backtest/run', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
          if (data.error) {
            runStatus.innerHTML = '<span class="loss">' + data.error + '</span>';
            runBtn.disabled = false;
            runBtn.innerHTML = '<i class="bi bi-play-fill"></i> Run Backtest';
            return;
          }
          // Start polling for status
          pollStatus();
        })
        .catch(err => {
          runStatus.innerHTML = '<span class="loss">Request failed</span>';
          runBtn.disabled = false;
          runBtn.innerHTML = '<i class="bi bi-play-fill"></i> Run Backtest';
        });
    });

    function pollStatus() {
      fetch('/api/backtest-status')
        .then(r => r.json())
        .then(data => {
          const elapsed = data.elapsed_seconds ? data.elapsed_seconds.toFixed(0) + 's' : '';
          if (data.running) {
            runStatus.innerHTML = '<span class="text-muted">' + data.progress + ' (' + elapsed + ')</span>';
            setTimeout(pollStatus, 2000);
          } else if (data.error) {
            runStatus.innerHTML = '<span class="loss"><i class="bi bi-x-circle"></i> ' + data.error + '</span>';
            runBtn.disabled = false;
            runBtn.innerHTML = '<i class="bi bi-play-fill"></i> Run Backtest';
          } else {
            runStatus.innerHTML = '<span class="gain"><i class="bi bi-check-circle"></i> Done in ' + elapsed + ' — reloading...</span>';
            setTimeout(() => location.reload(), 1500);
          }
        })
        .catch(() => setTimeout(pollStatus, 3000));
    }
  </script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy monitoring dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5000, help="Port")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args()

    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    if DASHBOARD_PASSWORD:
        logger.info("Password protection ENABLED")
    else:
        logger.info("WARNING: No DASHBOARD_PASSWORD set — dashboard is open")

    app.run(host=args.host, port=args.port, debug=args.debug)
