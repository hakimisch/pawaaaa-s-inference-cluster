#!/bin/bash
# start-cluster.sh — Bring up the compute stack with your chosen engine
# Usage:
#   ./start-cluster.sh              # Prompts for engine choice
#   ./start-cluster.sh vllm         # Start vLLM only
#   ./start-cluster.sh ollama       # Start Ollama only
#   ./start-cluster.sh both         # Start both (may OOM on 12GB)
#
# Homelab services (nginx, Open WebUI, Grafana, etc.) are always on.
# This script only controls the GPU compute node.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE="$DIR/docker-compose-compute.yml"

echo "╔══════════════════════════════════════════════╗"
echo "║   LLM Cluster — Compute Node Launcher        ║"
echo "╚══════════════════════════════════════════════╝"

# Ensure Docker is running
if ! sudo docker info >/dev/null 2>&1; then
    echo "→ Starting Docker daemon..."
    sudo systemctl start docker
    sleep 3
fi

# Determine engine choice
ENGINE="${1:-}"
if [ -z "$ENGINE" ]; then
    echo ""
    echo "Select engine(s):"
    echo "  1) vLLM (Qwen3-8B-AWQ — high throughput, ~10.8 GB VRAM)"
    echo "  2) Ollama (Gemma 4 12B / LFM2.5-8B — ~9.5 GB VRAM)"
    echo "  3) Both (only if you have >20 GB VRAM)"
    echo ""
    read -rp "Choice [1-3]: " choice
    case "$choice" in
        1) ENGINE="vllm" ;;
        2) ENGINE="ollama" ;;
        3) ENGINE="both" ;;
        *) echo "Invalid choice."; exit 1 ;;
    esac
fi

case "$ENGINE" in
    vllm|ollama|both) ;;
    *) echo "Usage: $0 {vllm|ollama|both}"; exit 1 ;;
esac

echo ""
echo "→ Starting: $ENGINE"

# Stop any existing containers first
sudo docker compose -f "$COMPOSE" down 2>/dev/null || true

if [ "$ENGINE" = "ollama" ] || [ "$ENGINE" = "both" ]; then
    echo "→ Pulling Ollama images (if needed)..."
    ollama pull gemma4:12b-it-q4_K_M 2>/dev/null || true
    ollama pull lfm2.5:8b 2>/dev/null || true
fi

echo ""
echo "→ Booting containers..."
sudo docker compose -f "$COMPOSE" up -d "$ENGINE"

echo ""
echo "→ Waiting for health checks..."
sleep 5

if [ "$ENGINE" = "vllm" ] || [ "$ENGINE" = "both" ]; then
    echo "   Waiting for vLLM (can take 30-90s)..."
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo "   ✅ vLLM ready"
            break
        fi
        sleep 3
    done
fi

if [ "$ENGINE" = "ollama" ] || [ "$ENGINE" = "both" ]; then
    echo "   Waiting for Ollama..."
    for i in $(seq 1 10); do
        if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "   ✅ Ollama ready"
            break
        fi
        sleep 2
    done
fi

echo ""
echo "✅ Compute stack running ($ENGINE)"
echo ""
echo "   vLLM API:     http://localhost:8000/v1"
echo "   Ollama API:   http://localhost:11434"
echo "   Pipeline UI:  http://localhost:8501"
