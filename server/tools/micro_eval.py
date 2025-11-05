#!/usr/bin/env python3
"""
Micro-eval CLI for A/B retrieval comparisons (graph/convo/summary/semantic).

Runs simple scenarios from a JSONL cases file, builds a fresh in-memory
HotMemory per case, loads setup utterances (to create edges/convo), and then
retrieves bullets for the query under different flag variants (A/B), measuring
precision@k and latency.

Case format (JSONL): one object per line with keys:
{
  "id": "case-1",
  "setup": ["My name is Alice", "I live in San Francisco"],
  "query": "What's my name?",
  "gold": ["name", "alice"],   # substrings expected to appear in bullets
  "session_id": "sess-1"        # optional; defaults to a random session
}

Usage:
  python server/tools/micro_eval.py --cases path/to/cases.jsonl \
    --variants baseline enhanced \
    --out results.json

Variants are predefined below (edit VARIANTS dict). Each entry is a mapping of
environment flags to apply for that run. Default includes:
  - baseline: default behavior
  - enhanced: MEMORY_FTS_ENHANCED_ONLY=true
  - profiles: MEMORY_PROFILES_ENABLED=true
  - catalog: MEMORY_SLOT_CATALOG=true

Outputs a JSON summary (and prints a table) with per-case metrics and totals:
  - bullets, precision_at_k, has_gold, latency_ms
  - per-variant aggregates
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from loguru import logger


# Predefined variants (edit as needed)
VARIANTS: Dict[str, Dict[str, str]] = {
    "baseline": {},
    "enhanced": {"MEMORY_FTS_ENHANCED_ONLY": "true"},
    "profiles": {"MEMORY_PROFILES_ENABLED": "true"},
    "catalog": {"MEMORY_SLOT_CATALOG": "true"},
    "profiles_sem": {
        "MEMORY_PROFILES_ENABLED": "true",
        "MEMORY_SEMANTIC_ENABLED": "true",
        "MEMORY_SOURCES": "graph,convo,summary,semantic"
    },
}


@dataclass
class Case:
    id: str
    setup: List[str]
    query: str
    gold: List[str]
    session_id: Optional[str] = None


def load_cases(path: str) -> List[Case]:
    cases: List[Case] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            cases.append(
                Case(
                    id=j.get("id") or f"case-{len(cases)+1}",
                    setup=list(j.get("setup") or []),
                    query=j.get("query") or "",
                    gold=list(j.get("gold") or []),
                    session_id=j.get("session_id"),
                )
            )
    return cases


def set_env(flags: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Apply flags to environment, return previous values for restoration."""
    prev: Dict[str, Optional[str]] = {}
    for k, v in flags.items():
        prev[k] = os.getenv(k)
        os.environ[k] = str(v)
    return prev


def restore_env(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def build_memory() -> Any:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.memory_hotpath import HotMemory
    try:
        # Provide SlotRouter store/user context so catalog-backed detection can work
        from core.memory.slot_router import SlotRouter
    except Exception:
        SlotRouter = None
    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store)
    hot.prewarm("en")
    try:
        if SlotRouter is not None:
            SlotRouter.set_context(store, getattr(hot, 'current_user_id', 'you'))
    except Exception:
        pass
    return hot


def case_run(hot, case: Case) -> Dict[str, Any]:
    """Load setup utterances, then retrieve bullets for query."""
    # Use provided session or generate one
    sid = case.session_id or f"sess-{int(time.time()*1000)}"
    # Load setup utterances (store facts + convo)
    for i, text in enumerate(case.setup):
        hot.process_turn(text, sid, i)
    hot.store.flush_if_needed(max_ops=1)

    # Retrieval (read-only path)
    start = time.perf_counter()
    bullets = hot.retrieve_bullets(case.query, read_only=True)
    latency_ms = (time.perf_counter() - start) * 1000

    # Precision@k where k=len(gold) (substring match)
    k = max(1, len(case.gold) or 1)
    hits = 0
    bl = [b.lower() for b in bullets]
    for g in case.gold:
        g2 = (g or "").lower()
        if any(g2 in b for b in bl):
            hits += 1
    precision_at_k = hits / float(k)
    has_gold = hits > 0

    # Simple hurt heuristic:
    # - If no gold found but bullets present => hurt
    # - If conflicting graph facts for same relation (e.g., lives_in), hurt
    # - If summary appeared and no gold => hurt
    def _parse_graph_bullet(b: str) -> Optional[tuple]:
        t = (b or "").lower()
        if "[graph]" not in t:
            return None
        # crude patterns
        for key in ("lives in ", "works at ", "is named ", "favorite "):
            if key in t:
                try:
                    idx = t.index(key) + len(key)
                    val = t[idx:].split("(")[0].strip()
                    rel = key.strip()
                    return (rel, val)
                except Exception:
                    continue
        return None

    conflicts = False
    rel_map = {}
    src_summary = any("[summary]" in (b or "").lower() for b in bullets)
    for b in bullets:
        p = _parse_graph_bullet(b)
        if not p:
            continue
        rel, val = p
        if rel not in rel_map:
            rel_map[rel] = set()
        rel_map[rel].add(val)
    for rel, vals in rel_map.items():
        if len(vals) > 1:
            conflicts = True
            break
    hurt = (len(bullets) > 0 and not has_gold) or conflicts or (src_summary and not has_gold)

    return {
        "case_id": case.id,
        "bullets": bullets,
        "precision_at_k": precision_at_k,
        "has_gold": has_gold,
        "latency_ms": latency_ms,
        "k": k,
        "hurt": hurt,
    }


def run_variants(cases: List[Case], variants: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name in variants:
        flags = VARIANTS.get(name, {}).copy()
        # Tag traces with variant if trace logging is enabled
        if os.getenv("MEMORY_TRACE_FILE"):
            flags["MEMORY_TRACE_VARIANT"] = name
        prev = set_env(flags)
        try:
            per_case = []
            for case in cases:
                hot = build_memory()
                r = case_run(hot, case)
                per_case.append(r)
            # Aggregate
            avg_prec = sum(c["precision_at_k"] for c in per_case) / max(1, len(per_case))
            avg_latency = sum(c["latency_ms"] for c in per_case) / max(1, len(per_case))
            has_gold_frac = sum(1 for c in per_case if c["has_gold"]) / max(1, len(per_case))
            hurt_frac = sum(1 for c in per_case if c.get("hurt")) / max(1, len(per_case))
            results[name] = {
                "flags": flags,
                "cases": per_case,
                "avg_precision_at_k": avg_prec,
                "avg_latency_ms": avg_latency,
                "has_gold_rate": has_gold_frac,
                "hurt_rate": hurt_frac,
            }
        finally:
            restore_env(prev)
    return results


def main():
    ap = argparse.ArgumentParser(description="Micro-eval A/B for retrieval")
    ap.add_argument("--cases", required=True, help="Path to JSONL cases file")
    ap.add_argument("--variants", nargs="+", default=["baseline", "enhanced"], help="Variant names to run")
    ap.add_argument("--out", default="", help="Output summary JSON path")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        raise SystemExit("No cases found")

    results = run_variants(cases, args.variants)

    # Print quick summary
    print("Variant\tAvgPrec@k\tHasGold%\tAvgLatency(ms)")
    for name, info in results.items():
        print(f"{name}\t{info['avg_precision_at_k']:.3f}\t{info['has_gold_rate']*100:.1f}\t{info['avg_latency_ms']:.1f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
