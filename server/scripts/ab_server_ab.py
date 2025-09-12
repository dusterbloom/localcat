#!/usr/bin/env python3
"""
Server-level A/B orchestrator for Localcat on macOS.

Spawns the bot twice (Variant A then B) with different env overrides, waits for health,
polls /api/metrics while you interact via the web client, and saves a metrics log per variant.

This helps you compare real-world latency and resource use on your M4 Pro.

Example:
  uv run server/scripts/ab_server_ab.py \
    --a OPENAI_THINK=false SUMMARIZER_INTERVAL_SECS=30 \
    --b OPENAI_THINK=true SUMMARIZER_INTERVAL_SECS=15 \
    --duration 60 --port 7860
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

import urllib.request
import urllib.error


def parse_kv_pairs(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


def wait_for_health(port: int, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://localhost:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def get_metrics(port: int) -> Dict:
    url = f"http://localhost:{port}/api/metrics"
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


def spawn_bot(port: int, extra_env: Dict[str, str]) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(extra_env)
    # prefer uv if available
    cmd = [sys.executable, "server/core/bot.py", "--port", str(port)]
    return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(__file__)), env=env)


def collect_phase(label: str, port: int, duration: int) -> List[Dict]:
    print(f"\n[{label}] Recording metrics for {duration}s … Interact via the web client now.")
    print("Press Ctrl+C to stop early.")
    start = time.time()
    samples: List[Dict] = []
    try:
        while time.time() - start < duration:
            m = get_metrics(port)
            samples.append({"t": time.time(), "metrics": m})
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    return samples


def save_results(label: str, samples: List[Dict]) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ab_server_{label}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    return path


def summarize(samples: List[Dict]) -> Dict[str, float]:
    # Generic summarizer: pull common numeric fields if present
    # Adjust keys based on your MetricsCollector outputs
    keys = [
        ("cpu", ["current_metrics", "cpu_percent"]),
        ("mem", ["current_metrics", "memory_percent"]),
        ("llm_latency_ms", ["current_metrics", "llm_latency_avg_ms"]),
        ("tts_latency_ms", ["current_metrics", "tts_latency_avg_ms"]),
    ]
    agg: Dict[str, List[float]] = {k: [] for k, _ in keys}

    def get_nested(d, path):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    for s in samples:
        m = s.get("metrics", {})
        for k, path in keys:
            v = get_nested(m, path)
            if isinstance(v, (int, float)):
                agg[k].append(float(v))

    out: Dict[str, float] = {}
    for k, arr in agg.items():
        if arr:
            out[f"avg_{k}"] = sum(arr) / len(arr)
            out[f"max_{k}"] = max(arr)
    return out


def main():
    ap = argparse.ArgumentParser(description="Server-level A/B orchestrator for Localcat")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--duration", type=int, default=60, help="Seconds to record metrics while you interact")
    ap.add_argument("--a", nargs="*", default=[], help="Variant A env overrides: KEY=VAL …")
    ap.add_argument("--b", nargs="*", default=[], help="Variant B env overrides: KEY=VAL …")
    args = ap.parse_args()

    variants = {
        "A": parse_kv_pairs(args.a),
        "B": parse_kv_pairs(args.b),
    }

    summaries: Dict[str, Dict] = {}
    for label, env_over in variants.items():
        print(f"\n=== Variant {label} ===")
        if env_over:
            print("Env overrides:", env_over)
        proc = spawn_bot(args.port, env_over)
        try:
            print("Waiting for /api/health …")
            if not wait_for_health(args.port, timeout=60):
                print("Server did not become healthy in time.")
                proc.terminate()
                proc.wait(timeout=10)
                continue
            print("Ready. Open the web client and converse for the duration.")
            samples = collect_phase(label, args.port, args.duration)
            path = save_results(label, samples)
            summ = summarize(samples)
            summaries[label] = {"file": path, "summary": summ}
            print(f"Saved {label} metrics to {path}")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    print("\n=== A/B Summary ===")
    for label, data in summaries.items():
        print(f"Variant {label}: {data['summary']} (file: {data['file']})")


if __name__ == "__main__":
    main()

