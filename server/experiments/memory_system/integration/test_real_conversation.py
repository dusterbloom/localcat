#!/usr/bin/env python3
"""
Test edge provenance and confidence with real conversation data from logs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory
from core.memory.confidence_strategy import RelationTypeConfidence, UsageBasedConfidence

# Real conversation sentences from logs.log
REAL_CONVERSATION = [
    "So let's say you have a wallet and you can either pay immediately or pay later, let's say once a day, and if in the",
    "In the second case, there are cycles in the payments graph where you are part of, you get uh refunds back from the liquidity that you save the system.",
    "basically is multilateral netting among agents that coordinate to settle at a given time and there is a provider and a decentralized protocol",
    "You make sure the graph is balanced and the payment can happen.",
    "Absolutely correct. That is the purpose is to save agents liquidity i. e. money and also avoid blockchain conditions.",
    "suggestion by doing what banks have been doing for centuries. Would you actually use something like that or would you pay immediately?",
    "Amazing. Um I gotta go now. Take care. Bye."
]

def test_with_baseline():
    """Test with RelationTypeConfidence (baseline)"""
    print("\n" + "="*80)
    print("Testing with RelationTypeConfidence (Baseline)")
    print("="*80)

    # Create in-memory store
    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store, confidence_strategy=RelationTypeConfidence())

    session_id = "real-conversation-baseline"

    # Process each turn
    for turn_id, text in enumerate(REAL_CONVERSATION):
        print(f"\n[Turn {turn_id}] Processing: {text[:60]}...")
        hot.process_turn(text, session_id, turn_id)

    # Flush to database
    store.flush_if_needed(max_ops=1)

    # Show extracted edges
    cur = store.sql.cursor()
    edges = cur.execute("""
        SELECT src, rel, dst, weight, pos, neg
        FROM edge
        WHERE status=1
        ORDER BY updated_at DESC
    """).fetchall()

    print(f"\n{'='*80}")
    print(f"Extracted {len(edges)} edges with baseline confidence:")
    print(f"{'='*80}")

    for src, rel, dst, weight, pos, neg in edges[:15]:  # Show first 15
        print(f"  ({src:20s} --[{rel:15s}]--> {dst:25s}) conf={weight:.3f} pos={pos} neg={neg}")

    if len(edges) > 15:
        print(f"  ... and {len(edges) - 15} more edges")

    return store, edges


def test_with_usage_based():
    """Test with UsageBasedConfidence (learned)"""
    print("\n" + "="*80)
    print("Testing with UsageBasedConfidence (Learned)")
    print("="*80)

    # Create in-memory store
    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store, confidence_strategy=UsageBasedConfidence())

    session_id = "real-conversation-learned"

    # Process each turn
    for turn_id, text in enumerate(REAL_CONVERSATION):
        print(f"\n[Turn {turn_id}] Processing: {text[:60]}...")
        hot.process_turn(text, session_id, turn_id)

    # Flush to database
    store.flush_if_needed(max_ops=1)

    # Show extracted edges
    cur = store.sql.cursor()
    edges = cur.execute("""
        SELECT src, rel, dst, weight, pos, neg
        FROM edge
        WHERE status=1
        ORDER BY updated_at DESC
    """).fetchall()

    print(f"\n{'='*80}")
    print(f"Extracted {len(edges)} edges with usage-based confidence:")
    print(f"{'='*80}")

    for src, rel, dst, weight, pos, neg in edges[:15]:  # Show first 15
        print(f"  ({src:20s} --[{rel:15s}]--> {dst:25s}) conf={weight:.3f} pos={pos} neg={neg}")

    if len(edges) > 15:
        print(f"  ... and {len(edges) - 15} more edges")

    return store, edges


def compare_strategies():
    """Compare baseline vs learned strategies side-by-side"""
    print("\n" + "="*80)
    print("COMPARISON: Baseline vs Usage-Based Confidence")
    print("="*80)

    baseline_store, baseline_edges = test_with_baseline()
    learned_store, learned_edges = test_with_usage_based()

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Baseline strategy extracted: {len(baseline_edges)} edges")
    print(f"Usage-based strategy extracted: {len(learned_edges)} edges")

    # Compare confidence distributions
    baseline_avg = sum(e[3] for e in baseline_edges) / len(baseline_edges) if baseline_edges else 0
    learned_avg = sum(e[3] for e in learned_edges) / len(learned_edges) if learned_edges else 0

    print(f"\nAverage confidence:")
    print(f"  Baseline: {baseline_avg:.3f}")
    print(f"  Usage-based: {learned_avg:.3f}")

    # Check provenance tracking
    if baseline_edges:
        edge_id = baseline_store.edge_id(baseline_edges[0][0], baseline_edges[0][1], baseline_edges[0][2])
        provenance = baseline_store.get_edge_provenance(edge_id)
        print(f"\n✅ Provenance tracking working: {len(provenance)} source(s) for first edge")
        if provenance:
            print(f"   Source text: {provenance[0][0][:60]}...")


if __name__ == "__main__":
    compare_strategies()
    print("\n✅ Test complete!")