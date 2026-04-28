#!/bin/bash
# deploy.sh — Set up the strategy on a fresh Linode (Ubuntu 22.04+)
#
# Usage:
#   ssh root@YOUR_LINODE_IP 'bash -s' < deploy.sh
#
# Prerequisites: Create a Nanode ($5/mo) on Linode with Ubuntu 22.04 LTS

set -euo pipefail

PROJECT_DIR="/opt/low_budget_strategy"
VENV="$PROJECT_DIR/.venv"
LOG="/var/log/strategy.log"

echo "=== 1. System packages ==="
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3-pip git

echo "=== 2. Clone repo ==="
rm -rf "$PROJECT_DIR"
git clone https://github.com/jenesanne/Low_Budget_Strategy.git "$PROJECT_DIR"

echo "=== 3. Python venv + deps ==="
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "=== 4. Create .env ==="
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > "$PROJECT_DIR/.env" << 'ENVEOF'
ALPACA_API_KEY=YOUR_KEY_HERE
ALPACA_SECRET_KEY=YOUR_SECRET_HERE
ALPACA_ENVIRONMENT=paper
ALPHA_VANTAGE_API_KEY=YOUR_AV_KEY_HERE
ENVEOF
    echo ">>> EDIT $PROJECT_DIR/.env with your real API keys! <<<"
fi

echo "=== 5. Set up cron ==="
# Monthly rebalance: 1st of every month at 14:30 UTC (after US market open)
REBAL_CMD="30 14 1 * * cd $PROJECT_DIR && $VENV/bin/python trade_live.py --execute >> $LOG 2>&1"
# Daily stop-loss monitor: every weekday at 21:05 UTC (~15 min after US close)
STOPS_CMD="5 21 * * 1-5 cd $PROJECT_DIR && $VENV/bin/python monitor_stops.py --execute >> $LOG 2>&1"

# Remove existing strategy cron entries, then add both
(crontab -l 2>/dev/null | grep -v -E "trade_live.py|monitor_stops.py" || true; \
    echo "$REBAL_CMD"; echo "$STOPS_CMD") | crontab -

echo "=== 6. Create log file ==="
touch "$LOG"

echo ""
echo "========================================"
echo "  Deployment complete!"
echo "  Project: $PROJECT_DIR"
echo "  Log:     $LOG"
echo ""
echo "  NEXT STEPS:"
echo "  1. Edit $PROJECT_DIR/.env with your API keys"
echo "  2. Test: cd $PROJECT_DIR && $VENV/bin/python trade_live.py"
echo "  3. Cron will auto-run quarterly"
echo "========================================"
