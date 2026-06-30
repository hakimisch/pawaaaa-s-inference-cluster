# 🧠 Homelab Inference Cluster

> **A distributed AI inference cluster** — serving frontier LLMs (vLLM, Ollama) on consumer GPU hardware, monitored via Prometheus/Grafana, managed through a Streamlit control panel, and connected across machines via Tailscale VPN.

---

## Executive Summary

| Attribute | Detail |
|-----------|--------|
| **Project** | Distributed AI Inference Cluster (compute node + service node) |
| **Compute Node** | WSL2 on Windows 11 — **RTX 4070 12 GB** — Docker-based GPU serving |
| **Service Node** | Homelab server — Prometheus, Grafana, Open WebUI, Nginx reverse proxy |
| **GPU Engines** | **vLLM** (Qwen3-8B-AWQ, 10.8 GB) + **Ollama** (Gemma 4 12B, LFM2.5-8B-A1B) |
| **Monitoring** | DCGM GPU metrics → Prometheus scrape → Grafana dashboards |
| **Orchestration** | Docker Compose on compute node + homelab, linked via Tailscale |
| **Management UI** | Streamlit dashboard — live GPU metrics, inference tests, benchmark runner, fine-tuning launcher |
| **Networking** | Tailscale mesh VPN (100.x.x.x private IPs) — no public ports exposed |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      WSL COMPUTE NODE                           │
│                    (RTX 4070 · 12 GB VRAM)                       │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │   Ollama         │    │   vLLM                              │ │
│  │   :11434         │    │   :8000 (OpenAI-compatible API)     │ │
│  │                  │    │                                     │ │
│  │  Gemma 4 12B    │    │  Qwen3-8B-AWQ                       │ │
│  │  LFM2.5-8B-A1B  │    │  AWQ quantized · Prefix caching     │ │
│  │  (>170 tok/s)   │    │  ~80 tok/s · Thinking mode          │ │
│  └────────┬────────┘    └────────────────┬────────────────────┘ │
│           │                              │                        │
│           └──────────┬───────────────────┘                        │
│                      │ Tailscale ({{COMPUTE_IP}})                  │
│  ┌───────────────────┴──────────────────────────────────────┐    │
│  │  Streamlit Control Panel (:8501)                         │    │
│  │  - Live GPU metrics (temp, util, VRAM, power)           │    │
│  │  - Service status (vLLM, Ollama, Grafana, Prometheus)   │    │
│  │  - Quick inference test (per engine)                    │    │
│  │  - Model benchmark runner → MLflow                      │    │
│  │  - Fine-tuning launcher                                 │    │
│  │  - GGUF deploy → Ollama                                 │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Tailscale
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     HOMELAB SERVICE NODE                         │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────┐ │
│  │  Nginx   │  │  Open WebUI  │  │  Grafana   │  │Prometheus │ │
│  │  :80/443 │──│  :3000       │  │  :3001     │  │  :9090    │ │
│  │  Reverse │  │  Chat UI for │  │  GPU temp  │  │  Scrape   │ │
│  │  Proxy   │  │  vLLM+Ollama │  │  · VRAM    │  │  vLLM+GPU │ │
│  └──────────┘  └──────────────┘  │  · Tok/s   │  └───────────┘ │
│                                  └────────────┘                │
│  Tailscale ({{HOMELAB_IP}})                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Model Inference** → User sends prompt to Open WebUI (or direct API call to vLLM/Ollama)
2. **GPU Processing** → Compute node runs inference on RTX 4070
3. **Monitoring** → DCGM metrics + vLLM/Ollama metrics scraped by Prometheus
4. **Visualization** → Grafana dashboards render GPU temp, VRAM, token throughput
5. **Management** → Streamlit UI queries Prometheus via batched SSH, controls Docker lifecycle

---

## Services

### Compute Node (WSL)

| Service | Port | Purpose | Model(s) | VRAM |
|---------|------|---------|----------|------|
| **vLLM** | `:8000` | High-throughput LLM serving (OpenAI API) | Qwen3-8B-AWQ | ~10.8 GB |
| **Ollama** | `:11434` | Lightweight local LLM runtime | Gemma 4 12B, LFM2.5-8B-A1B | ~9.5 GB (one at a time) |
| **Streamlit** | `:8501` | Control panel & benchmark UI | — | — |

### Service Node (Homelab)

| Service | Port | Purpose |
|---------|------|---------|
| **Nginx** | `:80 / :443` | Reverse proxy for all services (single entry point) |
| **Open WebUI** | `:3000` | Chat UI (connects to vLLM + Ollama backends) |
| **Prometheus** | `:9090` | Metrics collection (GPU, vLLM, system) |
| **Grafana** | `:3001` | Dashboard visualization |
| **Alertmanager** | `:9093` | Alert routing |

---

## Key Design Decisions

### Why Docker + Tailscale?

- **Zero public exposure** — all traffic routes through Tailscale's encrypted mesh. No ports open to the internet.
- **Two-machine split** — GPU container orchestration lives on the WSL machine (owns the GPU), service stack lives on the always-on homelab. Independent lifecycles.
- **SSH batch Prometheus queries** — The Streamlit UI batches all PromQL queries into a single SSH call to avoid Tailscale per-request latency (~500ms round trip). One SSH session, 15 queries returned together.

### Why Two Engines?

| Engine | When to Use |
|--------|-------------|
| **vLLM** | High-throughput batch inference, OpenAI-compatible API, prefix caching |
| **Ollama** | Interactive use (fast cold start), MoE models (LFM2.5 token efficiency), GGUF deploy |

The 12 GB VRAM can't run both simultaneously — `start-cluster.sh` lets you pick one.

### Monitoring Stack

Prometheus scrapes:
- `nvidia_smi_exporter` — GPU temp, utilization, VRAM, power draw
- `vllm` built-in metrics — requests running/waiting, token throughput (prompt + generation)
- `ollama` built-in metrics — loaded models, VRAM usage, context length

Grafana visualizes these on dashboards with per-model breakdowns.

---

## Getting Started

### Prerequisites

- **Hardware**: GPU with ≥8 GB VRAM (tested on RTX 4070 12 GB)
- **Software**: Docker (with nvidia-container-toolkit), WSL2 (Windows), Tailscale
- **Two machines**: One GPU compute node + one always-on service node (or collapse both into one)

### 1. Clone & Configure

```bash
git clone https://github.com/hakimisch/pawaaaa-s-inference-cluster.git
cd pawaaaa-s-inference-cluster
```

Edit `docker-compose-homelab.yml` and replace placeholders:

| Placeholder | Replace With |
|-------------|--------------|
| `{{COMPUTE_IP}}` | Your compute node's Tailscale IP |
| `{{HOMELAB_IP}}` | Your homelab's Tailscale IP |
| `{{HOSTNAME}}` | Your homelab hostname/DNS |

Also update `prometheus.yml`, `stop-cluster.sh`, and `pipeline_app.py` with the same values.

### 2. Deploy Service Node (homelab)

```bash
# Copy docker-compose-homelab.yml to your homelab server
docker compose -f docker-compose-homelab.yml up -d
```

### 3. Deploy Compute Node (WSL)

Create required Docker volumes:

```bash
sudo docker volume create ollama_data
sudo docker volume create vllm_cache
```

Start compute services:

```bash
# Interactive menu
bash start-cluster.sh

# Or pick an engine directly
bash start-cluster.sh vllm
bash start-cluster.sh ollama
```

### 4. Start Control Panel

```bash
# Install deps
pip install streamlit requests

# Run
streamlit run pipeline_app.py --server.port 8501
```

Open `http://{{COMPUTE_IP}}:8501` or `http://{{HOSTNAME}}/pipeline/` (if proxied).

---

## Benchmarking

The Streamlit UI includes a built-in benchmark runner. Click **"Run Benchmark"** in the Benchmarks tab to:

1. Select engine (vLLM / Ollama)
2. Pick model or "all"
3. Choose prompt categories (coding, reasoning, summarization, general)
4. Results are logged to **MLflow** and displayed in the Model Comparison tab

### Benchmark Categories

| Category | Example Prompt |
|----------|---------------|
| Coding | "Write a Python function that merges two sorted lists" |
| Reasoning | "Alice is twice as old as Bob was when Alice was as old as Bob is now." |
| Summarization | "Summarize this technical document in 3 bullet points" |
| General | "What is the capital of Mongolia?" |

Results compare latency (ms), throughput (tok/s), and VRAM usage across models.

---

## Hardware Notes

- **RTX 4070 12 GB** — Good for 7-8B parameter models. Qwen3-8B-AWQ fits at ~10.8 GB with `gpu-memory-utilization=0.85`.
- **vLLM + Ollama simultaneously NOT possible** on 12 GB — the start script prevents this.
- **MoE models** (like LFM2.5-8B-A1B) are very efficient — only 1B active params, so they share the GPU well.
- **WSL limitation**: DCGM exporter doesn't work on WSL (`/sys/bus/pci/` not available). Metrics come from `nvidia-smi` via Prometheus `node_exporter` or custom scripts.

---

## File Manifest

| File | Purpose |
|------|---------|
| `docker-compose-compute.yml` | GPU compute node (vLLM + Ollama) |
| `docker-compose-homelab.yml` | Service node (nginx, Open WebUI, Prometheus, Grafana) |
| `start-cluster.sh` | Interactive launcher for compute stack |
| `stop-cluster.sh` | Graceful shutdown, frees VRAM |
| `warmup-pipeline.sh` | Pre-compile Streamlit app for instant first load |
| `pipeline_app.py` | Streamlit control panel (948 lines) — GPU metrics, benchmarks, training |
| `prometheus.yml` | Prometheus scrape configuration |
| `grafana-provisioning/` | Grafana dashboard provisioning (dashboards, datasources) |

---

## 📊 Prometheus Metrics Dashboard

The Grafana dashboard tracks:

| Metric | Source | Purpose |
|--------|--------|---------|
| GPU Temperature | `nvidia_smi_exporter` | Thermal monitoring |
| GPU Utilization % | `nvidia_smi_exporter` | Compute load |
| VRAM Used / Total | `nvidia_smi_exporter` | Memory pressure |
| Power Draw | `nvidia_smi_exporter` | Power consumption |
| Active Requests | vLLM metrics | Serving load |
| Token Throughput | vLLM metrics | Prompt + generation tok/s |
| Models in VRAM | Ollama metrics | Cache state |

---

## 🔗 Related Projects

- [pawaaaa-s-engram-experiment](https://github.com/hakimisch/pawaaaa-s-engram-experiment) — Replicating DeepSeek's conditional memory research on this cluster
- [ai-wood-image-restoration](https://github.com/hakimisch/ai-wood-image-restoration) — CNN-based wood species classification, trained on 6GB consumer GPU
- [cairo-inventory](https://github.com/hakimisch/cairo-inventory) — Full-stack asset management system for UTM (Laravel + React + AWS)
