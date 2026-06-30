#!/bin/bash
# stop-cluster.sh — Tear down the compute stack, free all VRAM
# Homelab services (nginx, Open WebUI, Grafana, etc.) stay running.
# Run this when you're done using the LLM cluster.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE="$DIR/docker-compose-compute.yml"

echo "╔══════════════════════════════════════════════╗"
echo "║   LLM Cluster — Stopping Compute Node        ║"
echo "╚══════════════════════════════════════════════╝"

# Check what's running
RUNNING=$(sudo docker compose -f "$COMPOSE" ps --format '{{.Name}} {{.Status}}' 2>/dev/null || true)
if [ -z "$RUNNING" ]; then
    echo "→ No compute services running."
    exit 0
fi

echo "→ Running services:"
echo "$RUNNING" | while read -r line; do echo "   $line"; done

echo ""
echo "→ Stopping and removing all compute containers..."
sudo docker compose -f "$COMPOSE" down

# Verify VRAM is freed
sleep 2
echo ""
echo "→ Checking GPU..."
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "?")
echo "   GPU VRAM: ${VRAM_USED} MiB used (should be under 2000 if truly idle)"
echo ""
echo "✅ Compute stack stopped. GPU should be idle."
echo ""
echo "   Homelab services still running (connect via Tailscale):"
echo "   Open WebUI:  http://{{HOSTNAME}}:3000"
echo "   Grafana:     http://{{HOSTNAME}}:3001"
echo "   Pipeline:    http://{{HOSTNAME}}/pipeline/"
echo ""
echo "   To start again: bash start-cluster.sh"
