#!/usr/bin/env python3
"""
Evaluate LocalCat retrieval on SQuAD questions (A/B variants).

- Assumes contexts have been ingested into Enhanced FTS via ingest_squad.py
- Computes hit@k (has_gold), P@k, MRR, latency mean/p95/p99
- Supports A/B variants by reusing micro_eval VARIANTS or env KEY=VAL overrides
- Modes:
  * --use-raw-fts: query Enhanced FTS paragraphs directly (fair recall)
  * --snippet-bullets: emit Enhanced FTS paragraphs as bullets (no truncation)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def set_env(flags: Dict[str, str]) -> Dict[str, Optional[str]]:
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


def build_hot(db_path: str, lmdb_path: str):
    import sys
    ROOT = Path(__file__).resolve().parents[2] / "server"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.memory.memory_store import MemoryStore, Paths
    from core.memory.memory_hotpath import HotMemory
    from core.memory.slot_router import SlotRouter

    store = MemoryStore(Paths(sqlite_path=db_path, lmdb_dir=lmdb_path))
    hot = HotMemory(store)
    hot.prewarm("en")
    # Provide catalog/user context for slot router if enabled
    try:
        SlotRouter.set_context(store, getattr(hot, 'current_user_id', 'you'))
    except Exception:
        pass
    return hot


def normalize(s: str) -> str:
    return (s or "").strip().lower()


def _percentiles(values: List[float], *ps: float) -> Tuple[float, ...]:
    if not values:
        return tuple(0.0 for _ in ps)
    xs = sorted(values)
    out = []
    for p in ps:
        if p <= 0:
            out.append(xs[0])
            continue
        if p >= 100:
            out.append(xs[-1])
            continue
        k = (len(xs)-1) * (p/100.0)
        f = int(k)
        c = min(f+1, len(xs)-1)
        if f == c:
            out.append(xs[f])
        else:
            out.append(xs[f] + (xs[c]-xs[f]) * (k - f))
    return tuple(out)


def _metrics_for_ranking(ranking: List[str], answers: List[str], ks: List[int]) -> Dict[str, Any]:
    # Normalize
    def norm(s: str) -> str:
        return (s or "").strip().lower()
    rnorm = [norm(x) for x in ranking]
    anorm = [norm(a) for a in answers if a]
    # indicator of whether any answer appears in a text
    def hit(text: str) -> bool:
        return any(a and a in text for a in anorm)
    # hits per position
    hits = [1 if hit(t) else 0 for t in rnorm]
    # first relevant index
    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / i
            break
    out: Dict[str, Any] = {"mrr": rr}
    for k in ks:
        top = hits[:k]
        out[f"hit@{k}"] = 1.0 if any(top) else 0.0
        out[f"p@{k}"] = (sum(top) / max(1, min(k, len(hits))))
    return out


def run_eval(
    cases: List[Dict[str, Any]],
    flags: Dict[str, str],
    db: str,
    lmdb: str,
    *,
    use_raw_fts: bool = False,
    fts_limit: int = 10,
    use_snippet_bullets: bool = False,
) -> Dict[str, Any]:
    prev = set_env(flags)
    try:
        hot = build_hot(db, lmdb)
        # Prepare Enhanced FTS for raw paragraph checks when requested
        fts = None
        store = None
        if use_raw_fts:
            from core.memory.memory_store import MemoryStore, Paths
            from core.memory.enhanced_fts import EnhancedFTS
            store = MemoryStore(Paths(sqlite_path=db, lmdb_dir=lmdb))
            fts = EnhancedFTS(store)
        results: List[Dict[str, Any]] = []
        latencies: List[float] = []
        mrrs: List[float] = []
        ks = [1, 3, 5, 10]
        hits_k = {f"hit@{k}": 0.0 for k in ks}
        p_at_k = {f"p@{k}": 0.0 for k in ks}
        t0 = time.perf_counter()
        for ex in cases:
            q = ex["question"]
            answers = [normalize(a) for a in ex.get("answers", [])]
            start = time.perf_counter()
            if use_raw_fts and fts is not None:
                # Query raw Enhanced FTS paragraphs within the SQuAD session scope
                fts_hits = fts.enhanced_search(q, limit=fts_limit, session_ids=["squad"]) or []
                # fts_hits: List[(score, text, eid, ts, turn_id)]
                paragraphs = [normalize(t[1]) for t in fts_hits]
                latency_ms = (time.perf_counter() - start) * 1000
                has_gold = False
                for a in answers:
                    if not a:
                        continue
                    if any(a in p for p in paragraphs):
                        has_gold = True
                        break
                bullets = [t[1] for t in fts_hits[:min(3, len(fts_hits))]]
            else:
                if use_snippet_bullets:
                    # Fair bullets mode: emit top Enhanced FTS paragraphs as bullets (no truncation),
                    # but keep the rest of pipeline off to measure answer-span recall under bullet budget.
                    if fts is None:
                        from core.memory.memory_store import MemoryStore, Paths
                        from core.memory.enhanced_fts import EnhancedFTS
                        store = MemoryStore(Paths(sqlite_path=db, lmdb_dir=lmdb))
                        fts = EnhancedFTS(store)
                    fts_hits = fts.enhanced_search(q, limit=fts_limit, session_ids=["squad"]) or []
                    bullets = [t[1] for t in fts_hits[:min(fts_limit, len(fts_hits))]]
                else:
                    bullets = hot.retrieve_bullets(q, read_only=True)
                latency_ms = (time.perf_counter() - start) * 1000
                bl = [normalize(b) for b in bullets]
                has_gold = False
                for a in answers:
                    if not a:
                        continue
                    if any(a in b for b in bl):
                        has_gold = True
                        break
            results.append({
                "question": q,
                "answers": answers,
                "bullets": bullets,
                "has_gold": has_gold,
                "latency_ms": latency_ms,
            })
            latencies.append(latency_ms)
            # Ranking metrics on this query
            m = _metrics_for_ranking(bullets, answers, ks)
            mrrs.append(m["mrr"])
            for k in ks:
                hits_k[f"hit@{k}"] += m[f"hit@{k}"]
                p_at_k[f"p@{k}"] += m[f"p@{k}"]
        elapsed = (time.perf_counter() - t0) * 1000
        n = max(1, len(results))
        has_gold_rate = sum(1 for r in results if r["has_gold"]) / n
        avg_latency = sum(latencies) / n
        p95, p99 = _percentiles(latencies, 95, 99)
        agg = {
            "flags": flags,
            "n": n,
            "has_gold_rate": has_gold_rate,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "elapsed_ms": elapsed,
            "mrr": sum(mrrs) / n,
        }
        for k in ks:
            agg[f"hit@{k}"] = hits_k[f"hit@{k}"] / n
            agg[f"p@{k}"] = p_at_k[f"p@{k}"] / n
        return agg
    finally:
        restore_env(prev)


def main():
    ap = argparse.ArgumentParser(description="Evaluate LocalCat retrieval on SQuAD")
    ap.add_argument("--split", default="validation", help="HF datasets split (train/validation)")
    ap.add_argument("--db", default="./data/squad.db", help="SQLite database path")
    ap.add_argument("--lmdb", default="./data/squad.lmdb", help="LMDB directory path")
    ap.add_argument("--limit", type=int, default=200, help="Max QA examples")
    ap.add_argument("--out", default="", help="Output JSON path")
    ap.add_argument("--variants", nargs="*", default=["baseline", "sem_noise"], help="Variant names to run")
    ap.add_argument("--use-raw-fts", action="store_true", help="Query EnhancedFTS directly (raw paragraphs) instead of bullets")
    ap.add_argument("--fts-limit", type=int, default=10, help="Max paragraphs to retrieve per query in raw/snippet modes")
    ap.add_argument("--snippet-bullets", action="store_true", help="Emit Enhanced FTS paragraphs as bullets (no truncation)")
    args = ap.parse_args()

    # Load variants from micro_eval for convenience
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from server.tools.micro_eval import VARIANTS as MICRO_VARIANTS

    # Build a simple baseline if not in micro variants
    VARIANTS: Dict[str, Dict[str, str]] = {"baseline": {}}
    VARIANTS.update(MICRO_VARIANTS)

    from datasets import load_dataset
    ds = load_dataset("squad", split=args.split)
    cases: List[Dict[str, Any]] = []
    for ex in ds:
        q = ex.get("question") or ""
        answers = list((ex.get("answers") or {}).get("text") or [])
        if not q or not answers:
            continue
        cases.append({"question": q, "answers": answers})
        if len(cases) >= int(args.limit):
            break

    results: Dict[str, Any] = {}
    print("Variant\tHasGold%\tAvgLatency(ms)")
    for name in args.variants:
        flags = VARIANTS.get(name, {})
        r = run_eval(
            cases,
            flags,
            args.db,
            args.lmdb,
            use_raw_fts=args.use_raw_fts,
            fts_limit=args.fts_limit,
            use_snippet_bullets=args.snippet_bullets,
        )
        results[name] = r
        print(f"{name}\t{r['has_gold_rate']*100:.1f}\t{r['avg_latency_ms']:.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
