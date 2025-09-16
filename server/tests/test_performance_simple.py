#!/usr/bin/env python
"""Simple performance test to verify extraction and retrieval improvements."""

import time
import sys
from pathlib import Path

# Add the server directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from components.memory.hotmemory_facade import HotMemoryFacade
from components.memory.memory_store import MemoryStore

def test_performance():
    """Test extraction and retrieval performance after fixes."""

    # Initialize store and facade
    store = MemoryStore()
    facade = HotMemoryFacade(store)

    # Test cases
    test_sentences = [
        "I live in Sardinia.",
        "My dog's name is Potola.",
        "I work as a software engineer.",
        "My favorite food is pizza.",
        "I have two brothers and one sister."
    ]

    questions = [
        "Where do I live?",
        "What is my dog's name?",
        "What do I do for work?",
        "What's my favorite food?",
        "How many siblings do I have?"
    ]

    print("\n" + "="*60)
    print("EXTRACTION PERFORMANCE TEST")
    print("="*60)

    extraction_times = []
    extracted_counts = []

    for i, sentence in enumerate(test_sentences, 1):
        start = time.perf_counter()
        result = facade.process_turn(
            text=sentence,
            turn_id=i,
            session_id="test_session"
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        extraction_times.append(elapsed_ms)
        extracted_counts.append(len(result.triples))

        print(f"Turn {i}: '{sentence[:30]}...' " if len(sentence) > 30 else f"Turn {i}: '{sentence}'")
        print(f"  - Time: {elapsed_ms:.1f}ms")
        print(f"  - Extracted: {len(result.triples)} triples")
        if result.triples:
            for t in result.triples[:3]:  # Show first 3
                print(f"    • {t}")

    print("\n" + "="*60)
    print("RETRIEVAL PERFORMANCE TEST")
    print("="*60)

    retrieval_times = []
    bullet_counts = []

    for i, question in enumerate(questions, len(test_sentences) + 1):
        start = time.perf_counter()
        result = facade.process_turn(
            text=question,
            turn_id=i,
            session_id="test_session"
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        retrieval_times.append(elapsed_ms)
        bullet_counts.append(len(result.bullets))

        print(f"Turn {i}: '{question}'")
        print(f"  - Time: {elapsed_ms:.1f}ms")
        print(f"  - Retrieved: {len(result.bullets)} bullets")
        if result.bullets:
            for b in result.bullets[:3]:  # Show first 3
                print(f"    • {b}")

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    avg_extraction = sum(extraction_times) / len(extraction_times) if extraction_times else 0
    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    total_extracted = sum(extracted_counts)
    total_retrieved = sum(bullet_counts)

    print(f"Extraction:")
    print(f"  - Average time: {avg_extraction:.1f}ms")
    print(f"  - Max time: {max(extraction_times):.1f}ms")
    print(f"  - Min time: {min(extraction_times):.1f}ms")
    print(f"  - Total facts extracted: {total_extracted}")

    print(f"\nRetrieval:")
    print(f"  - Average time: {avg_retrieval:.1f}ms")
    print(f"  - Max time: {max(retrieval_times):.1f}ms")
    print(f"  - Min time: {min(retrieval_times):.1f}ms")
    print(f"  - Total bullets retrieved: {total_retrieved}")

    # Performance verdict
    print("\n" + "="*60)
    print("VERDICT:")
    if avg_extraction < 100 and avg_retrieval < 50:
        print("✅ EXCELLENT PERFORMANCE - Both extraction and retrieval are fast!")
    elif avg_extraction < 200 and avg_retrieval < 100:
        print("✅ GOOD PERFORMANCE - Acceptable speeds for both operations")
    else:
        print(f"⚠️ PERFORMANCE NEEDS IMPROVEMENT - Extraction: {avg_extraction:.0f}ms, Retrieval: {avg_retrieval:.0f}ms")

    # Check extraction is working
    if total_extracted >= len(test_sentences):
        print("✅ EXTRACTION WORKING - All facts were extracted")
    else:
        print(f"❌ EXTRACTION ISSUES - Only {total_extracted}/{len(test_sentences)} facts extracted")

if __name__ == "__main__":
    test_performance()