#!/usr/bin/env python3
"""
LLM Inference Cluster — Pipeline Control Panel
Streamlit UI for monitoring and managing the distributed AI inference stack.

Runs on: WSL compute node (RTX 4070 machine)
Connects to: Homelab Prometheus/Grafana via Tailscale

Served at: http://{{COMPUTE_IP}}:8501
Proxied via: http://{{HOSTNAME}}/pipeline/
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────

HOMELAB_TAILSCALE = "{{HOMELAB_IP}}"
COMPOSE_FILE = "/mnt/d/Project/docker-compose-compute.yml"
GGUF_DIR = "/mnt/d/Project/gguf-exports_gguf"
ADAPTER_DIR = "/mnt/d/Project/fine-tuned"

PROMETHEUS_URL = f"http://{HOMELAB_TAILSCALE}:9090"
GRAFANA_URL = f"http://{HOMELAB_TAILSCALE}:3001"
ALERTMANAGER_URL = f"http://{HOMELAB_TAILSCALE}:9093"

# ── Model definitions for Benchmark UI ────────────────────────────
MLFLOW_TRACKING_URI = f"http://{HOMELAB_TAILSCALE}:5050"

MODELS_VLLM = [
    {"name": "Qwen/Qwen3-8B-AWQ", "display": "Qwen3-8B-AWQ", "engine": "vllm"},
]
MODELS_OLLAMA = [
    {"name": "gemma4:12b-it-q4_K_M", "display": "Gemma 4 12B", "engine": "ollama"},
    {"name": "lfm2.5:8b", "display": "LFM2.5-8B-A1B", "engine": "ollama"},
]

ICON_HEALTHY = "🟢"
ICON_DOWN = "🔴"
ICON_STOPPED = "⏸️"
ICON_UNKNOWN = "⚪"


# ── Helpers ────────────────────────────────────────────────────────────────

def local_cmd(command: str, timeout: int = 15) -> tuple[str, int]:
    """Run a local shell command."""
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip(), r.returncode
    except subprocess.TimeoutExpired as e:
        return str(e), -1


def prom_batch(queries: list[str]) -> dict[str, list]:
    """Query many Prometheus metrics in one SSH batch call.
    
    Runs all curl queries in a single SSH session on the homelab,
    avoiding per-query Tailscale round-trip latency (~500ms each).
    
    Returns dict of {query_key: list_of_results}.
    """
    # Build a shell script that curls each query and outputs JSON
    script_lines = []
    for i, q in enumerate(queries):
        # Escape the query for safe shell usage
        escaped = q.replace('"', '\\"').replace('$', '\\$')
        script_lines.append(f'curl -s "http://localhost:9090/api/v1/query?query={escaped}" 2>/dev/null')
    
    if not script_lines:
        return {}
    
    # Join all curl commands with a delimiter we can split on
    full_script = "echo '---BATCH---'\n".join(script_lines)
    
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "{{USER}}@{{HOMELAB_IP}}", full_script],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0 or not r.stdout:
            return {}
        
        blocks = r.stdout.split("---BATCH---\n")
        results = {}
        for i, block in enumerate(blocks):
            if i >= len(queries):
                break
            try:
                data = json.loads(block)
                if data.get("status") == "success":
                    results[queries[i]] = data["data"]["result"]
                else:
                    results[queries[i]] = []
            except (json.JSONDecodeError, KeyError):
                results[queries[i]] = []
        return results
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return {}


def prom_query(query: str) -> list:
    """Single Prometheus query (fallback, kept for compatibility)."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return data["data"]["result"]
        return []
    except Exception:
        return []


def container_status(name: str) -> str:
    """Get status of a local Docker container."""
    out, _ = local_cmd(f"sudo docker inspect {name} --format '{{{{.State.Status}}}}' 2>/dev/null")
    if not out:
        return "not found"
    return out


def is_healthy(name: str) -> bool:
    """Check if a container is healthy (has health check and passes it)."""
    out, _ = local_cmd(f"sudo docker inspect {name} --format '{{{{.State.Health.Status}}}}' 2>/dev/null")
    return out == "healthy"


def http_ok(url: str, timeout: int = 3) -> bool:
    """Check if an HTTP endpoint responds OK."""
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=False)
        return r.status_code < 400
    except Exception:
        return False


# ── MLflow helpers ─────────────────────────────────────────────────

def mlflow_search(experiment_name: str = "model-benchmarks",
                  max_results: int = 200) -> list[dict]:
    """Search MLflow benchmark runs and return structured results."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=max_results,
        )

        results = []
        for run in runs:
            data = run.data
            # Skip parent batch runs (no individual prompt data)
            if run.info.run_name and run.info.run_name.startswith("batch_"):
                continue
            # Skip runs without latency (e.g. errored runs from earlier attempt)
            if "latency_ms" not in data.metrics:
                continue

            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "start_time": run.info.start_time,
                "model_name": data.params.get("model_name", "?"),
                "display_name": data.params.get("display_name", "?"),
                "engine": data.params.get("engine", "?"),
                "category": data.params.get("prompt_category", "?"),
                "latency_ms": data.metrics.get("latency_ms", 0),
                "tokens_per_second": data.metrics.get("tokens_per_second", 0),
                "prompt_tokens": data.metrics.get("prompt_tokens", 0),
                "completion_tokens": data.metrics.get("completion_tokens", 0),
                "vram_gb": data.metrics.get("vram_gb", 0),
            })
        return results
    except Exception as e:
        st.caption(f"⚠️ MLflow query failed: {e}")
        return []


def mlflow_training_runs(max_results: int = 20) -> list[dict]:
    """Search MLflow training experiment runs."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        experiment = client.get_experiment_by_name("training-runs")
        if not experiment:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=max_results,
        )

        results = []
        for run in runs:
            data = run.data
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "start_time": run.info.start_time,
                "model_name": data.params.get("model_name", "?").split("/")[-1],
                "dataset": data.params.get("dataset_name", "?").split("/")[-1],
                "num_steps": data.params.get("num_steps", "?"),
                "train_loss": data.metrics.get("train_loss"),
                "train_time_seconds": data.metrics.get("train_time_seconds"),
                "vram_used_gb": data.metrics.get("vram_used_gb"),
            })
        return results
    except Exception as e:
        st.caption(f"⚠️ MLflow training query failed: {e}")
        return []


# ── Page Setup ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Pipeline Control",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<style>#MainMenu {visibility:hidden;} footer {visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

st.title("🧠 LLM Inference Cluster — Control Panel")
st.caption(f"Compute: RTX 4070 (WSL) → Homelab: {HOMELAB_TAILSCALE} (Tailscale)")

col_refresh, col_actions = st.columns([1, 2])
with col_refresh:
    auto = st.checkbox("Auto-refresh (10s)", value=True)


# ── Batch query all Prometheus metrics at once ──────────────────────────

BATCH_QUERIES = [
    "nvidia_gpu_temperature_celsius",
    "nvidia_gpu_utilization_percent",
    "nvidia_gpu_memory_used_bytes",
    "nvidia_gpu_memory_total_bytes",
    "nvidia_gpu_power_watts",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "rate(vllm:prompt_tokens_total[1m])",
    "rate(vllm:generation_tokens_total[1m])",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "ollama_up",
    "ollama_models_loaded",
    "ollama_model_vram_bytes",
    "ollama_model_context_length",
]

# Clear any cached batch data at start of each run
batch_all = prom_batch(BATCH_QUERIES)

def bq(query: str) -> list:
    """Get a single query result from the batch."""
    return batch_all.get(query, [])

# ── Service Status ─────────────────────────────────────────────────────────

st.header("🔌 Service Status")

# WSL services
vllm_up = container_status("vllm")
ollama_up = container_status("ollama")
vllm_healthy = is_healthy("vllm")
ollama_healthy = is_healthy("ollama")

# Homelab services via HTTP
prom_up = http_ok(f"{PROMETHEUS_URL}/-/healthy")
grafana_up = http_ok(f"{GRAFANA_URL}/api/health")
alert_up = http_ok(f"{ALERTMANAGER_URL}/-/healthy")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if "running" in vllm_up and vllm_healthy:
        st.markdown(f"### {ICON_HEALTHY} vLLM")
        st.caption(":8000 · Qwen3-8B-AWQ")
    elif "running" in vllm_up:
        st.markdown(f"### {ICON_DOWN} vLLM")
        st.caption("⚠️ Unhealthy")
    elif "exited" in vllm_up.lower() or "stopped" in vllm_up.lower():
        st.markdown(f"### {ICON_STOPPED} vLLM")
        st.caption("Stopped")
    else:
        st.markdown(f"### {ICON_UNKNOWN} vLLM")
        st.caption(vllm_up[:15])

with c2:
    if "running" in ollama_up and ollama_healthy:
        st.markdown(f"### {ICON_HEALTHY} Ollama")
        st.caption(":11434 · 2 models")
    elif "running" in ollama_up:
        st.markdown(f"### {ICON_DOWN} Ollama")
        st.caption("⚠️ Unhealthy")
    elif "exited" in ollama_up.lower() or "stopped" in ollama_up.lower():
        st.markdown(f"### {ICON_STOPPED} Ollama")
        st.caption("Stopped")
    else:
        st.markdown(f"### {ICON_UNKNOWN} Ollama")
        st.caption(ollama_up[:15])

with c3:
    st.markdown(f"### {'🟢' if grafana_up else '🔴'} Grafana")
    st.caption(f":3001 {'✅' if grafana_up else '❌'}")

with c4:
    st.markdown(f"### {'🟢' if prom_up else '🔴'} Prometheus")
    st.caption(f":9090 {'✅' if prom_up else '❌'}")

with c5:
    st.markdown(f"### {'🟢' if alert_up else '🔴'} Alertmanager")
    st.caption(f":9093 {'✅' if alert_up else '❌'}")

# ── GPU Metrics ────────────────────────────────────────────────────────────

st.header("🎮 GPU Metrics (RTX 4070)")

gpu_temp = bq("nvidia_gpu_temperature_celsius")
gpu_util = bq("nvidia_gpu_utilization_percent")
gpu_mem_used = bq("nvidia_gpu_memory_used_bytes")
gpu_mem_total = bq("nvidia_gpu_memory_total_bytes")
gpu_power = bq("nvidia_gpu_power_watts")

mg1, mg2, mg3, mg4 = st.columns(4)

with mg1:
    temp = float(gpu_temp[0]["value"][1]) if gpu_temp else 0
    st.metric("Temperature", f"{temp:.0f}°C")

with mg2:
    util = float(gpu_util[0]["value"][1]) if gpu_util else 0
    st.metric("GPU Utilization", f"{util:.0f}%")

with mg3:
    used = int(float(gpu_mem_used[0]["value"][1])) / 1e9 if gpu_mem_used else 0
    total = int(float(gpu_mem_total[0]["value"][1])) / 1e9 if gpu_mem_total else 12.9
    pct = (used / total * 100) if total > 0 else 0
    st.metric("VRAM", f"{used:.1f} / {total:.1f} GB")

with mg4:
    power = float(gpu_power[0]["value"][1]) if gpu_power else 0
    st.metric("Power Draw", f"{power:.1f} W")

if gpu_util:
    util_val = float(gpu_util[0]["value"][1])
    st.progress(min(util_val / 100, 1.0), text=f"GPU Utilization: {util_val:.0f}%")

if gpu_mem_used and gpu_mem_total:
    ub = int(float(gpu_mem_used[0]["value"][1]))
    tb = int(float(gpu_mem_total[0]["value"][1]))
    vp = ub / tb if tb > 0 else 0
    st.progress(min(vp, 1.0), text=f"VRAM: {ub/1e9:.1f}/{tb/1e9:.1f} GB ({vp*100:.0f}%)")

# ── vLLM Inference Metrics ──────────────────────────────────────────────

st.header("📊 Inference Metrics")

engine_tabs = st.tabs(["vLLM (Qwen3-8B-AWQ)", "Ollama (Gemma 4 / LFM2.5)"])

with engine_tabs[0]:
    vllm_run = bq("vllm:num_requests_running")
    vllm_wait = bq("vllm:num_requests_waiting")
    vllm_pt = bq("rate(vllm:prompt_tokens_total[1m])")
    vllm_gt = bq("rate(vllm:generation_tokens_total[1m])")
    vllm_pt_t = bq("vllm:prompt_tokens_total")
    vllm_gt_t = bq("vllm:generation_tokens_total")

    mi1, mi2, mi3, mi4 = st.columns(4)
    with mi1:
        st.metric("Active Requests", f"{float(vllm_run[0]['value'][1]) if vllm_run else 0:.0f}")
    with mi2:
        st.metric("Queued", f"{float(vllm_wait[0]['value'][1]) if vllm_wait else 0:.0f}")
    with mi3:
        st.metric("Prompt Tokens/s", f"{float(vllm_pt[0]['value'][1]) if vllm_pt else 0:.1f}")
    with mi4:
        st.metric("Generation Tokens/s", f"{float(vllm_gt[0]['value'][1]) if vllm_gt else 0:.1f}")

    col_a, col_b = st.columns(2)
    with col_a:
        pt_t = float(vllm_pt_t[0]["value"][1]) if vllm_pt_t else 0
        st.caption(f"**Total prompt tokens:** {pt_t:,.0f}")
    with col_b:
        gt_t = float(vllm_gt_t[0]["value"][1]) if vllm_gt_t else 0
        st.caption(f"**Total generation tokens:** {gt_t:,.0f}")

    with st.expander("🧪 Quick Inference Test (vLLM)"):
        test_prompt = st.text_input("Prompt", "Say hello in 5 words.", key="test_prompt_vllm")
        if st.button("Send to vLLM"):
            with st.spinner("Querying vLLM..."):
                try:
                    r = requests.post(
                        "http://localhost:8000/v1/chat/completions",
                        json={
                            "model": "Qwen/Qwen3-8B-AWQ",
                            "messages": [{"role": "user", "content": test_prompt}],
                            "max_tokens": 50,
                        },
                        timeout=30,
                    )
                    if r.status_code == 200:
                        st.success(r.json()["choices"][0]["message"]["content"])
                    else:
                        st.error(f"HTTP {r.status_code}: {r.text[:200]}")
                except Exception as e:
                    st.error(str(e))

with engine_tabs[1]:
    # Ollama metrics from Prometheus
    ollama_up = bq("ollama_up")
    ollama_models = bq("ollama_models_loaded")
    ollama_vram = bq("ollama_model_vram_bytes")
    ollama_ctx = bq("ollama_model_context_length")

    mo1, mo2, mo3 = st.columns(3)
    with mo1:
        up = int(float(ollama_up[0]['value'][1])) if ollama_up else 0
        st.metric("Ollama API", "🟢 Online" if up else "🔴 Offline")
    with mo2:
        cnt = int(float(ollama_models[0]['value'][1])) if ollama_models else 0
        st.metric("Models in VRAM", cnt)
    with mo3:
        vram_total = sum(int(float(m['value'][1])) for m in ollama_vram) / 1e9 if ollama_vram else 0
        st.metric("Ollama VRAM", f"{vram_total:.1f} GB")

    if ollama_vram:
        st.subheader("Loaded Models")
        for m in ollama_vram:
            name = m['metric'].get('model', '?')
            quant = m['metric'].get('quantization', '?')
            vram_gb = int(float(m['value'][1])) / 1e9
            # Find context for this model
            ctx_val = "?"
            for c in ollama_ctx:
                if c['metric'].get('model') == name:
                    ctx_val = c['value'][1]
            st.markdown(f"- **{name.replace('_',':')}**  —  {quant}  —  {vram_gb:.1f} GB VRAM  —  ctx={ctx_val}")

        # Quick test for Ollama
        with st.expander("🧪 Quick Inference Test (Ollama)"):
            ollama_model = st.selectbox(
                "Model", ["gemma4:12b-it-q4_K_M", "lfm2.5:8b"],
                key="ollama_model_select"
            )
            ollama_prompt = st.text_input("Prompt", "Say hello in 5 words.", key="test_prompt_ollama")
            if st.button("Send to Ollama"):
                with st.spinner(f"Querying {ollama_model}..."):
                    try:
                        r = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": ollama_model, "prompt": ollama_prompt, "stream": False},
                            timeout=120,
                        )
                        if r.status_code == 200:
                            st.success(r.json()["response"])
                        else:
                            st.error(f"HTTP {r.status_code}: {r.text[:200]}")
                    except Exception as e:
                        st.error(str(e))
    else:
        st.info("No Ollama models currently loaded in VRAM. Start Ollama and send a request to load a model.")

# ── Quick Actions ─────────────────────────────────────────────────────────

st.header("🎯 Quick Actions")
ac1, ac2, ac3, ac4, ac5 = st.columns(5)

with ac1:
    if st.button("▶️ Start vLLM", use_container_width=True):
        local_cmd(f"sudo docker compose -f {COMPOSE_FILE} start vllm", timeout=30)
        time.sleep(2)
        st.rerun()

with ac2:
    if st.button("⏹️ Stop vLLM", use_container_width=True):
        local_cmd(f"sudo docker compose -f {COMPOSE_FILE} stop vllm")
        time.sleep(2)
        st.rerun()

with ac3:
    if st.button("▶️ Start Ollama", use_container_width=True):
        local_cmd(f"sudo docker compose -f {COMPOSE_FILE} start ollama", timeout=30)
        time.sleep(2)
        st.rerun()

with ac4:
    if st.button("⏹️ Stop Ollama", use_container_width=True):
        local_cmd(f"sudo docker compose -f {COMPOSE_FILE} stop ollama")
        time.sleep(2)
        st.rerun()

with ac5:
    if st.button("🔄 Restart Stack", use_container_width=True):
        local_cmd(f"sudo docker compose -f {COMPOSE_FILE} restart", timeout=60)
        time.sleep(3)
        st.rerun()

# Lifecycle: full on/off
st.markdown("---")
st.subheader("🔄 Full Cluster Control")
lc1, lc2, lc3 = st.columns([1, 1, 2])

with lc1:
    if st.button("🚀 Start Cluster", type="primary", use_container_width=True):
        out, code = local_cmd(f"bash /mnt/d/Project/start-cluster.sh vllm", timeout=120)
        if code == 0:
            st.success("Cluster started with vLLM")
        else:
            st.error(out[:300])
        time.sleep(2)
        st.rerun()

with lc2:
    if st.button("⏹️ Stop Cluster (free VRAM)", type="secondary", use_container_width=True):
        out, code = local_cmd(f"bash /mnt/d/Project/stop-cluster.sh", timeout=30)
        if code == 0:
            st.success("Cluster stopped — GPU idle")
        else:
            st.error(out[:300])
        time.sleep(2)
        st.rerun()

with lc3:
    st.caption(
        "**Stop** removes all compute containers and frees all 12 GB VRAM. "
        "**Start** brings them back with your chosen engine. "
        "Homelab services (Open WebUI, Grafana) stay running."
    )

# Deploy section
with st.expander("📤 Deploy GGUF to Ollama"):
    gguf_dir = Path(GGUF_DIR)
    if gguf_dir.exists():
        ggufs = list(gguf_dir.glob("*.gguf"))
        if ggufs:
            sel = st.selectbox("Select GGUF", [f.name for f in ggufs])
            name = st.text_input("Ollama Model Name", "finetuned-qwen")
            if st.button("Deploy to Ollama"):
                gpath = gguf_dir / sel
                modelfile = gguf_dir / "Modelfile"
                # Copy GGUF into Ollama volume, create model
                out, code = local_cmd(
                    f"sudo docker cp {gpath} ollama:/root/.ollama/{sel} && "
                    f"sudo docker exec ollama ollama create {name} -f /root/.ollama/{sel}",
                    timeout=60,
                )
                if code == 0:
                    st.success(f"✅ Model '{name}' created in Ollama")
                else:
                    st.error(out[:300])
        else:
            st.info("No GGUF files found")
    else:
        st.info("GGUF directory not found")

# ── Training Section ──────────────────────────────────────────────────────

st.header("🧪 Fine-Tuning")

train_col1, train_col2 = st.columns([1, 1])
with train_col1:
    steps = st.number_input("Steps", 10, 500, 50, 10)
    dataset_txt = st.text_input("Dataset", "mlabonne/FineTome-100k")
    export_gguf = st.checkbox("Export to GGUF after training", True)
    if st.button("🏃 Run Training", type="primary", use_container_width=True):
        export_flag = " --export-gguf" if export_gguf else ""
        with st.spinner(f"Training {steps} steps... (this may take a while)"):
            out, code = local_cmd(
                f"cd /mnt/d/Project && source unsloth_env/bin/activate && "
                f"python3 -u train.py --steps {steps} --dataset {dataset_txt}{export_flag} 2>&1 | tail -20",
                timeout=1800,
            )
        if code == 0:
            st.success(f"✅ Training complete! Loss: {out.split('train_loss')[1].split(',')[0].strip() if 'train_loss' in out else '?'}")
        else:
            st.error(f"❌ Training failed (exit {code})")
            if out:
                st.code(out[:1000])

with train_col2:
    st.subheader("📋 Recent Training Info")
    adapter_p = Path(ADAPTER_DIR)
    if adapter_p.exists():
        cfg_f = adapter_p / "adapter_config.json"
        if cfg_f.exists():
            cfg = json.loads(cfg_f.read_text())
            st.json({
                "LoRA Rank": cfg.get("r"),
                "LoRA Alpha": cfg.get("lora_alpha"),
                "Base Model": cfg.get("base_model_name_or_path", "").split("/")[-1],
            })
        # Files in adapter dir
        safetensors = list(adapter_p.rglob("adapter_model.safetensors"))
        if safetensors:
            size_mb = safetensors[0].stat().st_size / 1e6
            st.caption(f"Adapter size: {size_mb:.0f} MB")
            st.caption(f"Last modified: {datetime.fromtimestamp(safetensors[0].stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("No adapter found yet")

# ── Quick Links ────────────────────────────────────────────────────────────

st.header("🔗 Quick Links")
ql1, ql2, ql3, ql4 = st.columns(4)
with ql1:
    st.link_button("📊 Grafana", GRAFANA_URL, use_container_width=True)
with ql2:
    st.link_button("📈 Prometheus", PROMETHEUS_URL, use_container_width=True)
with ql3:
    st.link_button("🌐 Open WebUI", f"http://{HOMELAB_TAILSCALE}:3000", use_container_width=True)
with ql4:
    st.link_button("🏠 Dashy", f"http://{HOMELAB_TAILSCALE}:4000", use_container_width=True)

# ── Model Benchmarks ──────────────────────────────────────────────

st.header("📊 Model Benchmarks")

bench_tabs = st.tabs(["Model Comparison", "Run Benchmark", "MLflow UI"])

with bench_tabs[0]:
    """Model Comparison — aggregate metrics per model"""
    results = mlflow_search()

    if results:
        # Build aggregate summary
        agg = {}
        for r in results:
            key = r["display_name"]
            if key not in agg:
                agg[key] = {"runs": 0, "latencies": [], "tok_speeds": [],
                            "vrams": [], "engine": r["engine"]}
            agg[key]["runs"] += 1
            agg[key]["latencies"].append(r["latency_ms"])
            agg[key]["tok_speeds"].append(r["tokens_per_second"])
            agg[key]["vrams"].append(r["vram_gb"])

        comp_data = {
            "Model": [], "Engine": [], "Runs": [],
            "Avg Latency (ms)": [], "Avg Tok/s": [],
            "Max Tok/s": [], "Avg VRAM (GB)": [],
        }
        for model, vals in sorted(agg.items()):
            comp_data["Model"].append(model)
            comp_data["Engine"].append(vals["engine"])
            comp_data["Runs"].append(vals["runs"])
            comp_data["Avg Latency (ms)"].append(
                round(sum(vals["latencies"]) / len(vals["latencies"]), 1))
            comp_data["Avg Tok/s"].append(
                round(sum(vals["tok_speeds"]) / len(vals["tok_speeds"]), 1))
            comp_data["Max Tok/s"].append(
                round(max(vals["tok_speeds"]), 1))
            comp_data["Avg VRAM (GB)"].append(
                round(sum(vals["vrams"]) / len(vals["vrams"]), 2))

        st.subheader("📈 Model Performance Summary")
        st.dataframe(comp_data, use_container_width=True, hide_index=True)

        # Best-in-class highlights
        best_tok = max(agg.items(),
                       key=lambda x: sum(x[1]["tok_speeds"]) / len(x[1]["tok_speeds"]))
        best_lat = min(agg.items(),
                       key=lambda x: sum(x[1]["latencies"]) / len(x[1]["latencies"]))
        best_vram = min(agg.items(),
                        key=lambda x: sum(x[1]["vrams"]) / len(x[1]["vrams"]))

        c1, c2, c3 = st.columns(3)
        avg_t = sum(best_tok[1]["tok_speeds"]) / len(best_tok[1]["tok_speeds"])
        avg_l = sum(best_lat[1]["latencies"]) / len(best_lat[1]["latencies"])
        avg_v = sum(best_vram[1]["vrams"]) / len(best_vram[1]["vrams"])
        with c1:
            st.metric("⚡ Fastest Output", f"{avg_t:.0f} tok/s", best_tok[0])
        with c2:
            st.metric("⏱️ Lowest Latency", f"{avg_l:.0f} ms", best_lat[0])
        with c3:
            st.metric("💾 Most Efficient VRAM", f"{avg_v:.1f} GB", best_vram[0])

        # Per-category breakdown
        st.subheader("📊 Per-Category Comparison")
        cat_summary = {}
        for r in results:
            key = (r["display_name"], r["category"])
            if key not in cat_summary:
                cat_summary[key] = {"latencies": [], "tok_speeds": []}
            cat_summary[key]["latencies"].append(r["latency_ms"])
            cat_summary[key]["tok_speeds"].append(r["tokens_per_second"])

        cat_rows = []
        for (model, cat), vals in sorted(cat_summary.items()):
            lat_avg = sum(vals["latencies"]) / len(vals["latencies"])
            tok_avg = sum(vals["tok_speeds"]) / len(vals["tok_speeds"])
            cat_rows.append({
                "Model": model,
                "Category": cat,
                "Avg Latency": f"{lat_avg:.0f} ms",
                "Avg Tok/s": f"{tok_avg:.1f}",
            })
        if cat_rows:
            st.dataframe(cat_rows, use_container_width=True, hide_index=True)

        # Recent runs table
        st.subheader("📋 Recent Benchmark Runs")
        recent = results[:20]
        recent_rows = []
        for r in recent:
            ts = (datetime.fromtimestamp(r["start_time"] / 1000)
                  .strftime("%H:%M:%S") if r["start_time"] else "?")
            recent_rows.append({
                "Time": ts,
                "Model": r["display_name"],
                "Category": r["category"],
                "Latency": f"{r['latency_ms']:.0f}ms",
                "Tok/s": f"{r['tokens_per_second']:.1f}",
                "VRAM": f"{r['vram_gb']:.1f}GB",
                "Output": f"{r['completion_tokens']:.0f} tok",
            })
        st.dataframe(recent_rows, use_container_width=True, hide_index=True)

        # ── Before / After Comparison ─────────────────────────────────
        st.subheader("🔄 Model Upgrade Impact")

        # Helper to compute average metric for a model
        def _avg_for(name, metric):
            vals = [r[metric] for r in results if r["display_name"] == name]
            return sum(vals) / len(vals) if vals else 0

        # Identify old vs new models
        old_vllm = "Qwen2.5-7B-AWQ"
        new_vllm = "Qwen3-8B-AWQ"
        have_old = any(r["display_name"] == old_vllm for r in results)
        have_new = any(r["display_name"] == new_vllm for r in results)

        if have_old and have_new:
            old_lat = _avg_for(old_vllm, "latency_ms")
            new_lat = _avg_for(new_vllm, "latency_ms")
            old_tok = _avg_for(old_vllm, "tokens_per_second")
            new_tok = _avg_for(new_vllm, "tokens_per_second")
            old_vram = _avg_for(old_vllm, "vram_gb")
            new_vram = _avg_for(new_vllm, "vram_gb")

            st.markdown("**vLLM Upgrade: Qwen2.5-7B → Qwen3-8B-AWQ**")
            st.caption(
                "Qwen3 adds built-in reasoning (thinking mode) — the latency increase "
                "reflects generating step-by-step thought traces, not slower inference. "
                "Raw throughput (tok/s) is comparable."
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                delta_lat = f"+{((new_lat - old_lat) / old_lat * 100):.0f}%"
                st.metric("⏱️ Latency", f"{new_lat:.0f}ms", delta_lat)
            with c2:
                delta_tok = f"{((new_tok - old_tok) / old_tok * 100):+.1f}%"
                st.metric("⚡ Tok/s", f"{new_tok:.1f}", delta_tok)
            with c3:
                delta_vram = f"{((new_vram - old_vram) / old_vram * 100):+.1f}%"
                st.metric("💾 VRAM", f"{new_vram:.1f} GB", delta_vram)
            with c4:
                st.metric("🧠 Thinking", "✅ Built-in", old_vllm)

            st.caption(
                f"Based on {sum(1 for r in results if r['display_name']==old_vllm)} runs "
                f"({old_vllm}) vs {sum(1 for r in results if r['display_name']==new_vllm)} "
                f"runs ({new_vllm})"
            )
        else:
            st.info(
                "Before/after comparison needs both old (Qwen2.5-7B-AWQ) "
                "and new (Qwen3-8B-AWQ) benchmark data in MLflow.",
                icon="ℹ️",
            )

        # Ollama new-model highlights
        gemma_name = "Gemma 4 12B"
        lfm_name = "LFM2.5-8B-A1B"
        have_gemma = any(r["display_name"] == gemma_name for r in results)
        have_lfm = any(r["display_name"] == lfm_name for r in results)

        if have_gemma or have_lfm:
            st.markdown("**Ollama Upgrade: Frontier + Turbo**")
            st.caption(
                "Replaced previous models with Google's latest frontier model "
                "(Gemma 4 12B) and a high-speed MoE coding specialist "
                "(LFM2.5, 1B active params)."
            )
            cols = st.columns(3)
            col_idx = 0
            if have_gemma:
                g_lat = _avg_for(gemma_name, "latency_ms")
                g_tok = _avg_for(gemma_name, "tokens_per_second")
                g_vram = _avg_for(gemma_name, "vram_gb")
                with cols[col_idx]:
                    st.metric(
                        "🧠 Gemma 4 12B",
                        f"{g_tok:.0f} tok/s",
                        f"{g_lat:.0f}ms avg"
                    )
                    col_idx += 1
            if have_lfm:
                l_lat = _avg_for(lfm_name, "latency_ms")
                l_tok = _avg_for(lfm_name, "tokens_per_second")
                l_vram = _avg_for(lfm_name, "vram_gb")
                with cols[col_idx]:
                    st.metric(
                        "⚡ LFM2.5-8B-A1B",
                        f"{l_tok:.0f} tok/s",
                        f"{l_lat:.0f}ms avg"
                    )
                    col_idx += 1

            st.caption(
                f"LFM2.5 achieves {l_tok:.0f} tok/s at only {l_vram:.1f} GB VRAM "
                f"— ideal for real-time coding. Gemma 4 12B is a multimodal frontier "
                f"model with 128K context and native thinking."
                if have_lfm and have_gemma else ""
            )
    else:
        st.info("No benchmark data yet. Go to the 'Run Benchmark' tab to start.")

with bench_tabs[1]:
    """Run new benchmarks from the UI"""
    st.subheader("🚀 Run New Benchmark")

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        bench_engine = st.selectbox("Engine", ["vllm", "ollama"], key="bench_engine")
    with run_col2:
        model_list = ["all"] + [
            m["name"] for m in (MODELS_VLLM if bench_engine == "vllm" else MODELS_OLLAMA)
        ]
        bench_model = st.selectbox("Model", model_list, key="bench_model")

    bench_quick = st.checkbox("Quick mode (1 prompt per category)", True, key="bench_quick")
    bench_cat = st.selectbox("Category (optional)", [
        "all", "coding", "reasoning", "summarization", "general"
    ], key="bench_cat")

    if st.button("🏃 Run Benchmark", type="primary", use_container_width=True):
        parts = [
            "cd /mnt/d/Project && source unsloth_env/bin/activate",
            "&& python3 benchmark.py",
        ]
        if bench_engine:
            parts.append(f"--engine {bench_engine}")
        if bench_model and bench_model != "all":
            parts.append(f"--model \"{bench_model}\"")
        if bench_quick:
            parts.append("--quick")
        if bench_cat and bench_cat != "all":
            parts.append(f"--category {bench_cat}")

        full = " ".join(parts)
        with st.spinner("Running benchmarks... (may take a few minutes)"):
            out, code = local_cmd(full, timeout=600)
        if code == 0:
            st.success("✅ Benchmark complete! Check Model Comparison or MLflow UI.")
        else:
            st.error(f"❌ Benchmark failed (exit {code})")
        if out:
            st.code(out[:1500])

with bench_tabs[2]:
    """MLflow UI link"""
    st.subheader("🔗 MLflow Tracking Server")
    st.markdown(
        f"Open the [MLflow UI](http://{HOMELAB_TAILSCALE}:5050) "
        "for detailed run comparisons, parameter plots, and artifact browsing."
    )
    st.link_button(
        "📊 Open MLflow UI",
        f"http://{HOMELAB_TAILSCALE}:5050",
        use_container_width=True,
    )

# ── Training History (MLflow) ──────────────────────────────────────

with st.expander("📜 Training History (from MLflow)"):
    try:
        experiments = mlflow_training_runs()
        if experiments:
            exp_rows = []
            for e in experiments:
                ts = (datetime.fromtimestamp(e["start_time"] / 1000)
                      .strftime("%m-%d %H:%M") if e["start_time"] else "?")
                exp_rows.append({
                    "Time": ts,
                    "Model": e["model_name"],
                    "Dataset": e["dataset"],
                    "Steps": e["num_steps"],
                    "Loss": f"{e['train_loss']:.4f}" if e["train_loss"] else "?",
                    "Duration": f"{e['train_time_seconds']:.0f}s" if e["train_time_seconds"] else "?",
                    "VRAM": f"{e['vram_used_gb']:.1f}GB" if e["vram_used_gb"] else "?",
                })
            st.dataframe(exp_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No training experiments logged yet. Run training from Fine-Tuning section.")
    except Exception as ex:
        st.info(f"Could not load training history: {ex}")

# ── Footer ────────────────────────────────────────────────────────────────

st.divider()
st.caption(f"🔄 Last loaded: {datetime.now().strftime('%H:%M:%S')}  ·  🖥️ WSL compute node {'· Auto-refreshes every 10s' if auto else ''}")

# Meta refresh — lightweight, doesn't recompile the script
if auto:
    st.markdown(
        f'<meta http-equiv="refresh" content="10">',
        unsafe_allow_html=True,
    )
