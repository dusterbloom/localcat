#!/usr/bin/env python3
"""
Evaluate staged extraction policy on a dataset and produce summary JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sys
from loguru import logger

# Ensure server root and scripts are importable regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.staged_extraction_policy import StagedExtractionPolicy
from core.memory.eval_graph import prf1
from scripts.eval_extraction_ab import ComplexityAnalyzer


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        # categorized
        out: List[Dict[str, Any]] = []
        for _, rows in data.items():
            out.extend(rows)
        return out
    return data


def main():
    ap = argparse.ArgumentParser(description="Evaluate staged extraction policy")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--yaml", default="server/archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    yaml_path = args.yaml
    data = load_dataset(ds_path)
    logger.info(f"Loaded {len(data)} examples from {ds_path}")

    policy = StagedExtractionPolicy(yaml_path)
    complexity = ComplexityAnalyzer()

    preds: List[Tuple[str, str, str]] = []
    golds: List[Tuple[str, str, str]] = []
    latencies: List[float] = []
    methods: Dict[str, int] = {}

    for row in data:
        text = row["text"]
        gold = [(s, r, d) for s, r, d in row["gold"]]
        start = time.perf_counter()
        triples, meta = policy.extract(text, row.get("lang", "en"))
        lat = (time.perf_counter() - start) * 1000
        latencies.append(lat)
        preds.extend(triples)
        golds.extend(gold)
        methods[meta.get("method", "staged")] = methods.get(meta.get("method", "staged"), 0) + 1

    metrics = prf1(preds, golds)
    import numpy as np
    summary = {
        "staged_policy": {
            "overall": {
                "f1_mean": metrics["f1"],
                "precision_mean": metrics["precision"],
                "recall_mean": metrics["recall"],
                "latency_mean": float(np.mean(latencies) if latencies else 0.0),
                "latency_p95": float(np.percentile(latencies, 95) if latencies else 0.0),
                "timeout_rate": 0.0,
                "error_rate": 0.0,
            },
            "methods_used": methods,
        }
    }

    out = {
        "summary": summary,
        "config": {
            "dataset": str(ds_path),
            "yaml": yaml_path,
            "timestamp": time.time(),
        },
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
