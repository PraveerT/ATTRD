#!/bin/bash
# Start the sidepanel HTTP server + cloudflared quick tunnel + publisher
# watchdog. Writes PIDs and public URL to /notebooks/sidepanel_api/state/.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/state"
mkdir -p "$STATE"

# Kill any prior instances cleanly
[ -f "$STATE/server.pid" ] && kill "$(cat "$STATE/server.pid")" 2>/dev/null
[ -f "$STATE/cloudflared.pid" ] && kill "$(cat "$STATE/cloudflared.pid")" 2>/dev/null
pkill -f 'python3 .*sidepanel_api/server.py' 2>/dev/null
pkill -f 'cloudflared tunnel --url' 2>/dev/null
pkill -f 'publisher_watchdog.sh' 2>/dev/null
pkill -f 'publisher.py' 2>/dev/null
sleep 1

PORT="${SIDEPANEL_PORT:-8765}"

# Start backend
nohup python3 "$HERE/server.py" > "$STATE/server.log" 2>&1 &
echo $! > "$STATE/server.pid"
echo "[start] server pid=$(cat "$STATE/server.pid") port=$PORT"

# Start cloudflared
nohup /tmp/cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" \
  > "$STATE/cloudflared.log" 2>&1 &
echo $! > "$STATE/cloudflared.pid"
echo "[start] cloudflared pid=$(cat "$STATE/cloudflared.pid")"

# Start publisher watchdog (the "fetcher" that pushes status JSON to viz-qcc).
# Container restarts kill processes; this re-launches the loop cleanly.
if [ -f "$HERE/publisher_watchdog.sh" ]; then
  nohup bash "$HERE/publisher_watchdog.sh" > "$STATE/watchdog.log" 2>&1 &
  echo $! > "$STATE/watchdog.pid"
  echo "[start] publisher watchdog pid=$(cat "$STATE/watchdog.pid")"
else
  echo "[start] WARN: publisher_watchdog.sh missing — fetcher not started"
fi

# Wait for cloudflared to print the public URL (usually 3-8 s)
URL=""
for i in $(seq 1 30); do
  sleep 1
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$STATE/cloudflared.log" 2>/dev/null | head -1)
  [ -n "$URL" ] && break
done

if [ -n "$URL" ]; then
  echo "$URL" > "$STATE/url.txt"
  echo
  echo "============================================================"
  echo "  PWA URL: $URL"
  echo "  Open on phone, then 'Add to Home Screen' (iOS Share / Android menu)"
  echo "============================================================"
else
  echo "[start] cloudflared did not print URL within 30s; check $STATE/cloudflared.log"
fi
