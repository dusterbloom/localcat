#!/usr/bin/env python3
"""
Test DSPy extraction on real sentences from logs.log
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Enable DSPy extraction with LM Studio
os.environ["ENABLE_DSPY_EXTRACTION"] = "true"
os.environ["DSPY_MODEL"] = "openai/llama-3.2-3b-instruct"
os.environ["DSPY_BASE_URL"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "dummy"

from core.memory.memory_store import MemoryStore, Paths
from core.memory.memory_hotpath import HotMemory

# Real complex sentences from logs.log
REAL_LOG_SENTENCES = [
    "So let's say you have a wallet and you can either pay immediately or pay later, let's say once a day, and if in the",

    "In the second case, there are cycles in the payments graph where you are part of, you get uh refunds back from the liquidity that you save the system.",

    "basically is multilateral netting among agents that coordinate to settle at a given time and there is a provider and a decentralized protocol",

    "You make sure the graph is balanced and the payment can happen.",

    "Absolutely correct. That is the purpose is to save agents liquidity i. e. money and also avoid blockchain conditions.",

    "suggestion by doing what banks have been doing for centuries. Would you actually use something like that or would you pay immediately?",
]


def test_real_logs():
    """Test with real log sentences"""
    print("\n" + "="*80)
    print("Testing DSPy Extraction on Real Log Sentences")
    print("="*80)

    store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
    hot = HotMemory(store, enable_dspy_extraction=True)

    total_edges = 0

    for i, text in enumerate(REAL_LOG_SENTENCES):
        print(f"\n{'='*80}")
        print(f"Sentence {i+1}:")
        print(f"  {text}")
        print(f"{'='*80}")

        # Process
        session_id = f"real-log-{i}"
        bullets, triples = hot.process_turn(text, session_id, turn_id=0)

        print(f"\n✓ Extracted {len(triples)} edges:")
        for src, rel, dst in triples:
            edge_str = f"  • {src} --[{rel}]--> {dst}"
            print(edge_str)

        total_edges += len(triples)

        # Flush
        store.flush_if_needed(max_ops=1)

        # Show complexity metrics
        metrics = hot.get_metrics()
        if 'dspy_extraction_ms' in metrics and metrics['dspy_extraction_ms']['count'] > 0:
            dspy_time = metrics['dspy_extraction_ms']['mean']
            print(f"\n  DSPy extraction time: {dspy_time:.0f}ms")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal sentences processed: {len(REAL_LOG_SENTENCES)}")
    print(f"Total edges extracted: {total_edges}")
    print(f"Average edges per sentence: {total_edges/len(REAL_LOG_SENTENCES):.1f}")

    print("\n✅ Real log test complete!")

    return total_edges


if __name__ == "__main__":
    try:
        total = test_real_logs()
        if total > 0:
            print(f"\n🎉 SUCCESS: Extracted {total} edges from real conversation!")
        else:
            print(f"\n⚠️  WARNING: No edges extracted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)