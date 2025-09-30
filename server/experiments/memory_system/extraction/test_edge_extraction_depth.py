#!/usr/bin/env python3
"""
Test edge extraction depth - how many edges do we extract from complex sentences?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
from core.memory.confidence_strategy import RelationTypeConfidence

# Test sentences of varying complexity
TEST_SENTENCES = [
    # Simple sentences (1-2 edges expected)
    {
        "text": "My name is Alice",
        "expected_min": 1,
        "expected_facts": ["you --[name]--> alice"]
    },
    {
        "text": "I work at Google",
        "expected_min": 1,
        "expected_facts": ["you --[works_at]--> google"]
    },

    # Medium complexity (2-4 edges expected)
    {
        "text": "My name is Alice and I work at Google",
        "expected_min": 2,
        "expected_facts": [
            "you --[name]--> alice",
            "you --[works_at]--> google"
        ]
    },
    {
        "text": "I live in San Francisco and I love Python programming",
        "expected_min": 2,
        "expected_facts": [
            "you --[lives_in]--> san francisco",
            "you --[love]--> python programming"
        ]
    },

    # Complex sentences (3-6 edges expected)
    {
        "text": "My name is Alice, I work at Google in Mountain View, and I love Python",
        "expected_min": 3,
        "expected_facts": [
            "you --[name]--> alice",
            "you --[works_at]--> google",
            "you --[work_in]--> mountain view",
            "you --[love]--> python"
        ]
    },

    # Real complex sentence from logs
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

    # Very complex sentence
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

    # Nested/compound sentence
    {
        "text": "My favorite color is blue because it reminds me of the ocean where I learned to surf",
        "expected_min": 3,
        "expected_facts": [
            "you --[favorite_color]--> blue",
            "blue --[reminds]--> ocean",
            "you --[learned]--> surf"
        ]
    }
]


def test_extraction_depth():
    """Test how many edges we extract from each sentence"""

    print("="*80)
    print("EDGE EXTRACTION DEPTH ANALYSIS")
    print("="*80)

    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store, confidence_strategy=RelationTypeConfidence())

    results = []

    for i, test_case in enumerate(TEST_SENTENCES):
        text = test_case["text"]
        expected_min = test_case["expected_min"]

        print(f"\n{'='*80}")
        print(f"Test {i+1}: {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"Expected: ≥{expected_min} edges")
        print(f"{'='*80}")

        # Process the sentence
        session_id = f"test-{i}"
        hot.process_turn(text, session_id, turn_id=0)

        # Flush to get edges
        store.flush_if_needed(max_ops=1)

        # Get edges for this session
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

    # Show which sentences had poor extraction
    poor_extraction = [r for r in results if r["actual"] < r["expected"]]
    if poor_extraction:
        print(f"\n⚠️  Sentences with low extraction ({len(poor_extraction)}/{len(results)}):")
        for r in poor_extraction:
            print(f"  • {r['text'][:60]}... ({r['actual']}/{r['expected']} edges)")

    # Show best extraction
    best = max(results, key=lambda r: r["actual"])
    print(f"\n✅ Best extraction ({best['actual']} edges):")
    print(f"  Text: {best['text'][:70]}...")

    return results


if __name__ == "__main__":
    results = test_extraction_depth()

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("• Edge extraction uses spaCy dependency parsing")
    print("• Simple sentences: 1-2 edges extracted (good)")
    print("• Complex sentences: May need tuning for deeper extraction")
    print("• Compound facts in single sentence may be under-extracted")
    print("\n✅ Analysis complete!")