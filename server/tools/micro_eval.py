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
  - bullets, precision_at_k, mrr, hit@k, p@k, latency_ms
  - aggregates: has_gold_rate, avg_latency_ms, p95_latency_ms, p99_latency_ms, mrr, hit@k, p@k
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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
    "sem_noise": {
        "MEMORY_PROFILES_ENABLED": "true",
        "MEMORY_SEMANTIC_ENABLED": "true",
        "MEMORY_SOURCES": "graph,convo,summary,semantic",
        "MEMORY_FILTER_QUALITY": "false",
        "MEMORY_MAX_BULLETS": "3",
    },
    "sem_jina_best": {
        "MEMORY_PROFILES_ENABLED": "true",
        "MEMORY_SEMANTIC_ENABLED": "true",
        "MEMORY_SOURCES": "graph,convo,summary,semantic",
        "MEMORY_FILTER_QUALITY": "false",
        "MEMORY_MAX_BULLETS": "3",
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_VERIFIER_MAXLEN": "512",
        "MEMORY_VERIFIER_ENT_T": "0.55",
    },
    # Recommendations A/B variants
    "rec_sem_jina_tiny_rerank025": {
        # Semantic enabled, small bullet cap
        "MEMORY_SEMANTIC_ENABLED": "true",
        "MEMORY_SOURCES": "graph,convo,summary,semantic",
        "MEMORY_FILTER_QUALITY": "false",
        "MEMORY_MAX_BULLETS": "2",
        # Verifier ON (Jina tiny)
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-tiny",
        "MEMORY_VERIFIER_MAXLEN": "384",
        # Jina rerank ON (moderate weight)
        "MEMORY_RERANK_JINA_ENABLED": "true",
        "MEMORY_RERANK_JINA_MODEL": "jinaai/jina-reranker-v3-tiny",
        "MEMORY_RERANK_JINA_MAXLEN": "384",
        "MEMORY_RERANK_JINA_WEIGHT": "0.25",
    },
    "rec_sem_jina_large_rerank025": {
        # Semantic enabled, small bullet cap
        "MEMORY_SEMANTIC_ENABLED": "true",
        "MEMORY_SOURCES": "graph,convo,summary,semantic",
        "MEMORY_FILTER_QUALITY": "false",
        "MEMORY_MAX_BULLETS": "2",
        # Verifier ON (Jina large)
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_VERIFIER_MAXLEN": "512",
        # Jina rerank ON (moderate weight)
        "MEMORY_RERANK_JINA_ENABLED": "true",
        "MEMORY_RERANK_JINA_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_RERANK_JINA_MAXLEN": "512",
        "MEMORY_RERANK_JINA_WEIGHT": "0.25",
    },
    # Verifier A/B toggles
    "verifier_off": {"MEMORY_VERIFIER_ENABLED": "false"},
    "verifier_hf": {
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "hf",
        # Example HF NLI/reranker model (set your local id/cache)
        # "MEMORY_VERIFIER_MODEL": "cross-encoder/nli-deberta-v3-base"
    },
    "verifier_jina_tiny": {
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-tiny",
        "MEMORY_VERIFIER_MAXLEN": "384",
    },
    "verifier_jina_base": {
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-base",
        "MEMORY_VERIFIER_MAXLEN": "512",
        "MEMORY_VERIFIER_ENT_T": "0.55",
    },
    "verifier_jina_best": {
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_VERIFIER_MAXLEN": "512",
        "MEMORY_VERIFIER_ENT_T": "0.55",
    },
    "verifier_jina_best_strict": {
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_VERIFIER_MAXLEN": "512",
        "MEMORY_VERIFIER_ENT_T": "0.50",
        "MEMORY_VERIFIER_CON_T": "0.55",
        "MEMORY_VERIFIER_BOOST": "1.0",
        "MEMORY_VERIFIER_ALLOW_UNKNOWN": "0",
    },
    # Jina reranker variants (pairwise entailment into composite scoring)
    "jina_rerank_best": {
        "MEMORY_RERANK_JINA_ENABLED": "true",
        "MEMORY_RERANK_JINA_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_RERANK_JINA_MAXLEN": "512",
        "MEMORY_RERANK_JINA_WEIGHT": "0.25",
        # keep defaults for other components
    },
    "jina_rerank_best_plus_verifier": {
        "MEMORY_RERANK_JINA_ENABLED": "true",
        "MEMORY_RERANK_JINA_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_RERANK_JINA_MAXLEN": "512",
        "MEMORY_RERANK_JINA_WEIGHT": "0.25",
        "MEMORY_VERIFIER_ENABLED": "true",
        "MEMORY_VERIFIER_BACKEND": "jina",
        "MEMORY_VERIFIER_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_VERIFIER_MAXLEN": "512",
        "MEMORY_VERIFIER_ENT_T": "0.55",
    },
    # Top-1 stress: limit bullets to 1 to expose ranking differences
    "top1": {
        "MEMORY_MAX_BULLETS": "1"
    },
    "top1_jina_rerank": {
        "MEMORY_MAX_BULLETS": "1",
        "MEMORY_RERANK_JINA_ENABLED": "true",
        "MEMORY_RERANK_JINA_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_RERANK_JINA_MAXLEN": "512",
        "MEMORY_RERANK_JINA_WEIGHT": "0.35",
    },
    "top1_jina_rerank_strong": {
        "MEMORY_MAX_BULLETS": "1",
        "MEMORY_RERANK_JINA_ENABLED": "true",
        "MEMORY_RERANK_JINA_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_RERANK_JINA_MAXLEN": "512",
        "MEMORY_RERANK_JINA_WEIGHT": "0.6",
    },
    "sem_top1": {
        "MEMORY_PROFILES_ENABLED": "true",
        "MEMORY_SEMANTIC_ENABLED": "true",
        "MEMORY_SOURCES": "graph,convo,summary,semantic",
        "MEMORY_FILTER_QUALITY": "false",
        "MEMORY_MAX_BULLETS": "1",
    },
    "sem_top1_jina_rerank_strong": {
        "MEMORY_PROFILES_ENABLED": "true",
        "MEMORY_SEMANTIC_ENABLED": "true",
        "MEMORY_SOURCES": "graph,convo,summary,semantic",
        "MEMORY_FILTER_QUALITY": "false",
        "MEMORY_MAX_BULLETS": "1",
        "MEMORY_RERANK_JINA_ENABLED": "true",
        "MEMORY_RERANK_JINA_MODEL": "jinaai/jina-reranker-v3-large",
        "MEMORY_RERANK_JINA_MAXLEN": "512",
        "MEMORY_RERANK_JINA_WEIGHT": "0.6",
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

    # Ranking metrics
    def norm(s: str) -> str:
        return (s or "").strip().lower()
    answers = [norm(a) for a in case.gold if a]
    rb = [norm(b) for b in bullets]
    def hit(text: str) -> bool:
        return any(a and a in text for a in answers)
    pos_hits = [1 if hit(t) else 0 for t in rb]
    # legacy precision_at_k (using number of gold slots)
    k = max(1, len(answers) or 1)
    precision_at_k = (sum(pos_hits[:k]) / float(min(k, len(pos_hits)) or 1))
    has_gold = any(pos_hits)
    # MRR
    mrr = 0.0
    for i, h in enumerate(pos_hits, start=1):
        if h:
            mrr = 1.0 / i
            break
    # hit@k, p@k for common k
    def p_at(K: int) -> float:
        return (sum(pos_hits[:K]) / float(min(K, len(pos_hits)) or 1))
    metrics_k = {f"hit@{K}": 1.0 if any(pos_hits[:K]) else 0.0 for K in (1,3,5)}
    metrics_k.update({f"p@{K}": p_at(K) for K in (1,3,5)})

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
        "mrr": mrr,
        **metrics_k,
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
            latencies: List[float] = []
            for case in cases:
                hot = build_memory()
                r = case_run(hot, case)
                per_case.append(r)
                latencies.append(r["latency_ms"])
            # Aggregate
            avg_prec = sum(c["precision_at_k"] for c in per_case) / max(1, len(per_case))
            avg_latency = sum(c["latency_ms"] for c in per_case) / max(1, len(per_case))
            has_gold_frac = sum(1 for c in per_case if c["has_gold"]) / max(1, len(per_case))
            hurt_frac = sum(1 for c in per_case if c.get("hurt")) / max(1, len(per_case))
            # Ranking aggregates
            def pctile(xs: List[float], p: float) -> float:
                if not xs:
                    return 0.0
                xs = sorted(xs)
                k = (len(xs)-1) * (p/100.0)
                f = int(k)
                c = min(f+1, len(xs)-1)
                if f == c:
                    return xs[f]
                return xs[f] + (xs[c] - xs[f]) * (k - f)
            agg = {
                "flags": flags,
                "cases": per_case,
                "avg_precision_at_k": avg_prec,
                "avg_latency_ms": avg_latency,
                "has_gold_rate": has_gold_frac,
                "hurt_rate": hurt_frac,
                "p95_latency_ms": pctile(latencies, 95),
                "p99_latency_ms": pctile(latencies, 99),
                "mrr": sum(c.get("mrr", 0.0) for c in per_case)/max(1,len(per_case)),
                "hit@1": sum(c.get("hit@1", 0.0) for c in per_case)/max(1,len(per_case)),
                "hit@3": sum(c.get("hit@3", 0.0) for c in per_case)/max(1,len(per_case)),
                "hit@5": sum(c.get("hit@5", 0.0) for c in per_case)/max(1,len(per_case)),
                "p@1": sum(c.get("p@1", 0.0) for c in per_case)/max(1,len(per_case)),
                "p@3": sum(c.get("p@3", 0.0) for c in per_case)/max(1,len(per_case)),
                "p@5": sum(c.get("p@5", 0.0) for c in per_case)/max(1,len(per_case)),
            }
            results[name] = agg
        finally:
            restore_env(prev)
    return results


def check_slo_compliance(results: dict, slo_p95_ms: float = 100.0) -> List[dict]:
    """
    Check if variants meet latency SLO.

    Args:
        results: Results dict from run_variants
        slo_p95_ms: P95 latency SLO threshold in milliseconds

    Returns:
        List of violations (empty if all pass)
    """
    violations = []

    for variant_name, metrics in results.items():
        # Try to get P95, fall back to mean if not available
        p95_latency = metrics.get("latency_p95_ms")
        if p95_latency is None:
            p95_latency = metrics.get("avg_latency_ms", 0.0)

        if p95_latency > slo_p95_ms:
            violations.append({
                "variant": variant_name,
                "p95_latency": p95_latency,
                "slo": slo_p95_ms,
                "excess_ms": p95_latency - slo_p95_ms
            })

    if violations:
        logger.warning("⚠️  SLO VIOLATIONS DETECTED:")
        for v in violations:
            logger.warning(
                f"  {v['variant']}: P95={v['p95_latency']:.1f}ms "
                f"(exceeds {v['slo']:.0f}ms by {v['excess_ms']:.1f}ms)"
            )

        # Optionally fail CI
        if os.getenv("MEMORY_SLO_STRICT", "false").lower() in ("1", "true", "yes"):
            raise AssertionError(f"SLO violated: {len(violations)} variants over budget")
    else:
        logger.info(f"✅ All variants within {slo_p95_ms}ms P95 SLO")

    return violations


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

    # Check SLO compliance
    check_slo_compliance(results, slo_p95_ms=100.0)

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
