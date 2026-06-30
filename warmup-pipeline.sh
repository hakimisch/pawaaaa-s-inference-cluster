#!/bin/bash
# warmup-pipeline.sh — Pre-compile the Streamlit app so first load is instant
# Run after starting the Streamlit server.

set -euo pipefail

URL="${1:-http://localhost:8501/pipeline/}"
MAX_WAIT=60
INTERVAL=3

echo "→ Warming up Streamlit at $URL..."
echo "   Waiting up to ${MAX_WAIT}s for server..."

# Wait for server to be ready
for i in $(seq 1 $((MAX_WAIT / INTERVAL))); do
    if curl -s -o /dev/null --max-time 5 "$URL" 2>/dev/null; then
        echo "   Server ready after $((i * INTERVAL))s"
        break
    fi
    if [ "$i" -eq "$((MAX_WAIT / INTERVAL))" ]; then
        echo "   ⚠️ Server not ready within ${MAX_WAIT}s"
        exit 1
    fi
    sleep "$INTERVAL"
done

# Hit the page to trigger compilation
echo "→ Compiling app (first load)..."
t1=$(date +%s%N)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 "$URL" 2>/dev/null)
t2=$(date +%s%N)
ELAPSED=$(( (t2 - t1) / 1000000 ))

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ App compiled in ${ELAPSED}ms (HTTP $HTTP_CODE)"
else
    echo "⚠️ App returned HTTP $HTTP_CODE in ${ELAPSED}ms"
fi
