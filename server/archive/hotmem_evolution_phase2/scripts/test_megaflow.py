#!/usr/bin/env python3
"""
MegaFlow End-to-End Test
========================

Consolidates extraction, retrieval, summaries, and correction flows
into a single orchestrated run so you can trace the whole process.

Notes:
- Uses existing test harnesses without modifying core code.
- LM Studio relation/summarization paths are attempted only if available.
"""

import os
import sys
import time
import asyncio
import tempfile
from typing import Any, List, Tuple

# Make server/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory


async def section(title: str):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


async def run_extraction_complex_patterns():
    await section("Section 1: Extraction (Complex Patterns)")
    try:
        from test_complex_patterns import ComplexPatternTester  # type: ignore
    except Exception as e:
        print(f"❌ Could not import ComplexPatternTester: {e}")
        return
    tester = ComplexPatternTester()
    try:
        tester.setup()
        results = tester.test_complex_sentences()
        tester.analyze_results(results)
    finally:
        tester.cleanup()


async def run_extraction_abc():
    await section("Section 2: Extraction A/B/C (Baseline, Stanza, Batch)")
    try:
        from test_abc_extraction import TestFramework  # type: ignore
    except Exception as e:
        print(f"❌ Could not import TestFramework: {e}")
        return
    try:
        tf = TestFramework()
        tf.run_all_tests()
    except Exception as e:
        print(f"❌ A/B/C run failed: {e}")


async def run_storage_retrieval():
    await section("Section 3: Storage, Summaries, and Retrieval")
    try:
        from test_storage_retrieval import StorageRetrievalTester  # type: ignore
    except Exception as e:
        print(f"❌ Could not import StorageRetrievalTester: {e}")
        return
    tester = StorageRetrievalTester()
    try:
        tester.setup()
        storage_results = await tester.simulate_conversation()
        retrieval_results = await tester.test_retrieval_strategies()
        tester.print_results(storage_results, retrieval_results)
    finally:
        tester.cleanup()


def _print_edges(store: MemoryStore, src: str, rel: str):
    cur = store.sql.cursor()
    rows = cur.execute(
        "SELECT src, rel, dst, weight, status, updated_at FROM edge WHERE src=? AND rel=? ORDER BY updated_at DESC",
        (src, rel),
    ).fetchall()
    if not rows:
        print("  (no edges)")
        return
    for (s, r, d, w, st, ts) in rows[:8]:
        print(f"  ({s}) -[{r}]-> ({d})  w={w:.2f} status={st} ts={ts}")


async def run_corrections_flow():
    await section("Section 4: Corrections (Functional Relation Update)")
    with tempfile.TemporaryDirectory() as tdir:
        store = MemoryStore(Paths(
            sqlite_path=os.path.join(tdir, 'memory.db'),
            lmdb_dir=os.path.join(tdir, 'graph.lmdb')
        ))
        hot = HotMemory(store)
        sid = "corr-session"

        # Step 1: Initial name
        text1 = "My name is Alex."
        bullets1, triples1 = hot.process_turn(text1, session_id=sid, turn_id=1)
        print("🔹 After initial fact: My name is Alex")
        _print_edges(store, 'you', 'name')

        # Step 2: Correction / update
        text2 = "Actually, my name is Bob."
        bullets2, triples2 = hot.process_turn(text2, session_id=sid, turn_id=2)
        print("🔹 After correction: my name is Bob")
        _print_edges(store, 'you', 'name')

        # Step 3: Retrieval sanity
        q = "what is my name"
        ents = ["you"]
        bullets = hot._retrieve_context(q, ents, turn_id=999, intent=None)
        print(f"🔎 Retrieval bullets for '{q}':")
        for b in bullets:
            print(f"  • {b}")

async def run_retrieval_quality_eval():
    await section("Section 5: Retrieval Quality Evaluation (LEANN A/B)")
    try:
        from eval_retrieval_quality import main as retrieval_quality_main  # type: ignore
    except Exception as e:
        print(f"❌ Could not import retrieval quality evaluator: {e}")
        return
    try:
        retrieval_quality_main()
    except Exception as e:
        print(f"❌ Retrieval quality evaluation failed: {e}")


async def main():
    start = time.perf_counter()
    await run_extraction_complex_patterns()
    await run_extraction_abc()
    await run_storage_retrieval()
    await run_corrections_flow()
    await run_retrieval_quality_eval()
    # Optional: real conversations with true summaries (requires LM Studio / OpenAI-compatible server)
    await section("Section 6: Real Conversations with True Summaries (Optional)")
    try:
        import test_real_conversations  # type: ignore
        test_real_conversations.run()
    except Exception as e:
        print(f"(skipped) Real conversations test not executed: {e}")
    dur = (time.perf_counter() - start) * 1000
    print("\n" + "=" * 80)
    print(f"MegaFlow completed in {dur:.0f}ms")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
