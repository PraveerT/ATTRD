#!/bin/bash
# Start publisher.py in background, unbuffered, writing to state/publisher.log.
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/state"
mkdir -p "$STATE"

# Kill any previous publisher
pkill -f 'sidepanel_api/publisher.py' 2>/dev/null
sleep 1

nohup python3 -u "$HERE/publisher.py" --interval 30 > "$STATE/publisher.log" 2>&1 &
echo $! > "$STATE/publisher.pid"
echo "[publisher_run] launched pid=$(cat "$STATE/publisher.pid")"
