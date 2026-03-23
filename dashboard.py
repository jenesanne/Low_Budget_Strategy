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
import json
import logging
import os
import secrets
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
    day_pnl = equity - last_equity
    day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0

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
        equity=equity,
        cash=cash,
        buying_power=buying_power,
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

    <!-- Account Stats -->
    <div class="row g-2 mb-3">
      <div class="col-6 col-md-3">
        <div class="stat-card">
          <div class="stat-label">Equity</div>
          <div class="stat-value">${{ "{:,.2f}".format(equity) }}</div>
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
