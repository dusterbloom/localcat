#!/usr/bin/env python3
"""
Consolidate bakeoff results across datasets into a single summary JSON and print tables.

Scans server/results/* for known result files or accepts explicit files via args.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def collect_results(paths: List[Path]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in paths:
        data = load_json(p)
        if not data:
            continue
        key = p.stem
        out[key] = data
    return out


def find_default_files() -> List[Path]:
    root = Path(__file__).resolve().parents[1] / "results"
    patterns = [
        "slm_comparison.json",
        "slm_bakeoff_easy_*.json",
        "slm_bakeoff_medium_*.json",
        "slm_bakeoff_hard_*.json",
        "slm_bakeoff_*_qwen_strict.json",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(set(files))


def summarize_entry(name: str, data: Dict) -> Dict:
    s = data.get("summary", {})
    cfg = data.get("config", {})
    def pick(method: str) -> Dict:
        return s.get(method, {}).get("overall", {})
    return {
        "name": name,
        "dataset": cfg.get("dataset"),
        "yaml": {
            "f1": pick("yaml").get("f1_mean"),
            "latency_mean": pick("yaml").get("latency_mean"),
            "latency_p95": pick("yaml").get("latency_p95"),
        },
        "yaml_slm": {
            "f1": pick("yaml_slm").get("f1_mean"),
            "latency_mean": pick("yaml_slm").get("latency_mean"),
            "latency_p95": pick("yaml_slm").get("latency_p95"),
        },
        "methods": list(s.keys()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="Result JSON files to consolidate")
    ap.add_argument("--output", default="server/results/bakeoff_consolidated.json")
    args = ap.parse_args()

    files = [Path(f) for f in args.files] if args.files else find_default_files()
    if not files:
        print("No result files found.")
        return
    results = collect_results(files)

    summaries = [summarize_entry(k, v) for k, v in results.items()]
    out = {"entries": summaries}
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("\nConsolidated Bakeoff Summary:\n")
    header = f"{'name':<30} {'F1 yaml':>8} {'F1 slm':>8} {'mean yaml':>10} {'mean slm':>10} {'p95 yaml':>10} {'p95 slm':>10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        y = s["yaml"]
        ys = s["yaml_slm"]
        print(f"{s['name']:<30} {y['f1']!s:>8} {ys['f1']!s:>8} {y['latency_mean']!s:>10} {ys['latency_mean']!s:>10} {y['latency_p95']!s:>10} {ys['latency_p95']!s:>10}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

