#!/usr/bin/env python3
"""
Test DSPy-enhanced edge extraction on complex sentences
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Enable DSPy extraction with LM Studio
os.environ["ENABLE_DSPY_EXTRACTION"] = "true"
# Use llama-3.2-3b-instruct (better than 1b for this task)
os.environ["DSPY_MODEL"] = "openai/llama-3.2-3b-instruct"
os.environ["DSPY_BASE_URL"] = "http://127.0.0.1:1234/v1"  # LM Studio default
os.environ["OPENAI_API_KEY"] = "dummy"  # LM Studio doesn't need real key

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
from core.memory.confidence_strategy import RelationTypeConfidence

# Complex test sentences (same as before)
TEST_SENTENCES = [
    # Very complex sentence that spaCy struggled with (4/6 edges = 67%)
    {
        "text": "I'm Alice, a software engineer at Google who loves Python, lives in San Francisco, enjoys hiking on weekends, and has a cat named Whiskers",
        "expected_min": 6,
        "expected_facts": [
            "you --[name]--> alice",
            "you --[is]--> software engineer",
            "you --[works_at]--> google",
            "you --[love]--> python",
            "you --[lives_in]--> san francisco",
            "you --[enjoy]--> hiking",
            "you --[has]--> cat",
            "cat --[name]--> whiskers"
        ]
    },

    # Another complex case from logs (3/4 edges = 75%)
    {
        "text": "In the second case, there are cycles in the payments graph where you are part of, you get refunds back from the liquidity that you save the system",
        "expected_min": 4,
        "expected_facts": [
            "cycles --[in]--> payments graph",
            "you --[is]--> part",
            "you --[get]--> refunds",
            "you --[save]--> system"
        ]
    },

    # Simpler sentence for baseline
    {
        "text": "My name is Bob and I work at Microsoft",
        "expected_min": 2,
        "expected_facts": [
            "you --[name]--> bob",
            "you --[works_at]--> microsoft"
        ]
    }
]


def test_with_dspy():
    """Test with DSPy enhancement"""
    print("\n" + "="*80)
    print("Testing with DSPy-Enhanced Extraction")
    print("="*80)

    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store, enable_dspy_extraction=True)

    results = []

    for i, test_case in enumerate(TEST_SENTENCES):
        text = test_case["text"]
        expected_min = test_case["expected_min"]

        print(f"\n{'='*80}")
        print(f"Test {i+1}: {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"Expected: ≥{expected_min} edges")
        print(f"{'='*80}")

        # Process
        session_id = f"dspy-test-{i}"
        hot.process_turn(text, session_id, turn_id=0)

        # Flush
        store.flush_if_needed(max_ops=1)

        # Get edges
        cur = store.sql.cursor()
        edges = cur.execute("""
            SELECT DISTINCT e.src, e.rel, e.dst, e.weight
            FROM edge e
            JOIN edge_source es ON e.id = es.edge_id
            JOIN conversation_turn ct ON es.turn_id = ct.id
            WHERE ct.session_id = ? AND e.status = 1
        """, (session_id,)).fetchall()

        actual_count = len(edges)

        print(f"\n✓ Extracted {actual_count} edges:")
        for src, rel, dst, weight in edges:
            edge_str = f"  • {src} --[{rel}]--> {dst}"
            print(f"{edge_str:<70} (conf={weight:.3f})")

        # Compare with expected
        status = "✅ GOOD" if actual_count >= expected_min else "⚠️  LOW"
        coverage_pct = (actual_count / expected_min * 100) if expected_min > 0 else 0

        print(f"\n{status}: {actual_count}/{expected_min} edges ({coverage_pct:.0f}% of expected minimum)")

        results.append({
            "text": text,
            "expected": expected_min,
            "actual": actual_count,
            "status": status,
            "edges": edges
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_expected = sum(r["expected"] for r in results)
    total_actual = sum(r["actual"] for r in results)

    print(f"\nTotal edges:")
    print(f"  Expected (minimum): {total_expected}")
    print(f"  Actual extracted:   {total_actual}")
    print(f"  Coverage:           {(total_actual/total_expected*100):.1f}%")

    # Show improvements
    print(f"\n✅ Target: 90%+ coverage on complex sentences")
    print(f"✅ Achieved: {(total_actual/total_expected*100):.1f}% coverage")

    if total_actual >= total_expected * 0.9:
        print(f"\n🎉 SUCCESS: Achieved 90%+ coverage target!")
    else:
        print(f"\n⚠️  Need more work to reach 90% target")

    return results


if __name__ == "__main__":
    try:
        results = test_with_dspy()
        print("\n✅ Test complete!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)