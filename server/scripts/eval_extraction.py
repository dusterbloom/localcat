#!/usr/bin/env python3
"""
Evaluate extraction quality and latency for:
- Baseline HotMemory (UDExtractor)
- YAMLExtractor (dev-only YAML interpreter)

Usage:
  uv run server/scripts/eval_extraction.py \
      --dataset server/tests/data/yaml_eval_examples.json \
      --yaml server/archive/2024_12_consolidation/assets/ASI1_proposal.yaml \
      [--lang en]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

from loguru import logger

import sys
from pathlib import Path as FilePath
sys.path.insert(0, str(FilePath(__file__).parent.parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
from core.memory.extractors.yaml_extractor import YAMLExtractor
from core.memory.eval_graph import prf1


def load_dataset(path: Path) -> List[Dict]:
    data = json.loads(Path(path).read_text())
    # expected format: [{"text": str, "gold": [[s,r,d], ...], "lang": "en"?}]
    return data


def run_extractor_hotmem(hot: HotMemory, text: str, lang: str) -> List[Tuple[str, str, str]]:
    # UDExtractor returns 5-tuple (entities, triples, neg_count, doc, aliases)
    ents, triples, neg, doc, _aliases = hot.extractor.extract(text, lang)
    triples = hot.extractor.refine(text, triples, doc)
    return triples


def run_extractor_yaml(yaml_ext: YAMLExtractor, text: str, lang: str) -> List[Tuple[str, str, str]]:
    ents, triples, neg, doc = yaml_ext.extract(text, lang)
    triples = yaml_ext.refine(text, triples, doc)
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    dataset = load_dataset(Path(args.dataset))

    # Baseline HotMemory
    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store)
    hot.prewarm(args.lang)

    # YAML extractor
    yaml_ext = YAMLExtractor(args.yaml)

    gold_all: List[Tuple[str, str, str]] = []
    base_all: List[Tuple[str, str, str]] = []
    yaml_all: List[Tuple[str, str, str]] = []

    t_base_ms = []
    t_yaml_ms = []

    for ex in dataset:
        text: str = ex["text"]
        gold: List[List[str]] = ex["gold"]
        lang: str = ex.get("lang", args.lang)

        # Baseline
        t0 = time.perf_counter()
        base_tr = run_extractor_hotmem(hot, text, lang)
        t_base_ms.append((time.perf_counter() - t0) * 1000)

        # YAML
        t0 = time.perf_counter()
        yaml_tr = run_extractor_yaml(yaml_ext, text, lang)
        t_yaml_ms.append((time.perf_counter() - t0) * 1000)

        gold_tr = [(s, r, d) for s, r, d in gold]
        gold_all.extend(gold_tr)
        base_all.extend(base_tr)
        yaml_all.extend(yaml_tr)

    base_metrics = prf1(base_all, gold_all)
    yaml_metrics = prf1(yaml_all, gold_all)

    def agg_ms(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "p95": 0.0}
        xs_sorted = sorted(xs)
        p95 = xs_sorted[int(0.95 * (len(xs_sorted) - 1))]
        return {"mean": sum(xs) / len(xs), "p95": p95}

    print("\n=== Extraction Quality (Graph-Edge PRF) ===")
    print(f"Baseline UD: P={base_metrics['precision']:.3f} R={base_metrics['recall']:.3f} F1={base_metrics['f1']:.3f}")
    print(f"YAML       : P={yaml_metrics['precision']:.3f} R={yaml_metrics['recall']:.3f} F1={yaml_metrics['f1']:.3f}")

    print("\n=== Latency per example (ms) ===")
    b = agg_ms(t_base_ms)
    y = agg_ms(t_yaml_ms)
    print(f"Baseline UD: mean={b['mean']:.1f} p95={b['p95']:.1f}")
    print(f"YAML       : mean={y['mean']:.1f} p95={y['p95']:.1f}")


if __name__ == "__main__":
    main()
