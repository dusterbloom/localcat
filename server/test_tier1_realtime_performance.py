#!/usr/bin/env python3
"""
Demonstrate Tier 1's real-time knowledge graph generation capabilities
Focus on warm-up performance for production usage
"""

import sys
import os
import time
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.extraction.memory_extractor import MemoryExtractor

def test_tier1_realtime_performance():
    """Test Tier 1 extraction after warm-up for real-time scenarios"""

    print("⚡ Tier 1 Real-Time Performance Test")
    print("=" * 60)
    print("Testing warm-up performance for production knowledge graph generation")
    print()

    # Test sentences typical of voice conversations
    test_sentences = [
        "I live in New York and work at Google.",
        "Sarah graduated from Stanford University last year.",
        "My friend John drives a Tesla Model 3.",
        "The company Apple is based in Cupertino.",
        "Maria teaches mathematics at Harvard.",
        "Tom visited Paris last summer for vacation.",
        "Lisa bought a new MacBook Pro yesterday.",
        "David studies computer science at MIT.",
        "Emma works as a software engineer in Seattle.",
        "Mike plays guitar in a local band."
    ]

    # Initialize with warm-up
    print("🔧 Initializing Tier 1 extractor with warm-up...")
    config = {
        'use_glirel': False,  # Focus on Tier 1 only
        'use_gliner': True,   # Enable GLiNER for high-quality entities
        'sqlite_path': ':memory:',
        'session_id': 'test_tier1_realtime'
    }

    extractor = MemoryExtractor(config)

    # Warm-up extraction
    print("🔥 Performing warm-up extraction...")
    warmup_text = "The quick brown fox jumps over the lazy dog."
    warmup_start = time.perf_counter()
    extractor.extract(warmup_text)
    warmup_time = (time.perf_counter() - warmup_start) * 1000
    print(f"   Warm-up completed in {warmup_time:.1f}ms")
    print()

    # Performance testing
    print("🚀 Real-time performance testing...")
    print("-" * 50)

    extraction_times = []
    entity_counts = []
    triple_counts = []

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n  {i:2d}. {sentence}")

        # Time the extraction
        start_time = time.perf_counter()
        result = extractor.extract(sentence)
        extraction_time = (time.perf_counter() - start_time) * 1000

        extraction_times.append(extraction_time)
        entity_counts.append(len(result.entities))
        triple_counts.append(len(result.triples))

        # Color code performance
        if extraction_time < 100:
            status = "🟢 EXCELLENT"
        elif extraction_time < 200:
            status = "🟡 GOOD"
        elif extraction_time < 500:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 SLOW"

        print(f"      ⏱️  {extraction_time:.1f}ms {status}")
        print(f"      👥 {len(result.entities)} entities | 🔗 {len(result.triples)} triples")

        # Show sample knowledge graph
        if result.triples:
            print(f"      📋 Knowledge Graph:")
            for j, (head, rel, tail) in enumerate(result.triples[:3], 1):
                print(f"         {j}. {head} --{rel}--> {tail}")

    # Statistical analysis
    print(f"\n" + "=" * 60)
    print("📊 PERFORMANCE STATISTICS")
    print("=" * 60)

    avg_time = sum(extraction_times) / len(extraction_times)
    min_time = min(extraction_times)
    max_time = max(extraction_times)
    p95_time = sorted(extraction_times)[int(len(extraction_times) * 0.95)]
    avg_entities = sum(entity_counts) / len(entity_counts)
    avg_triples = sum(triple_counts) / len(triple_counts)

    print(f"⏱️  Extraction Time:")
    print(f"   Average: {avg_time:.1f}ms")
    print(f"   Min: {min_time:.1f}ms")
    print(f"   Max: {max_time:.1f}ms")
    print(f"   95th percentile: {p95_time:.1f}ms")

    print(f"\n👥 Entity Extraction:")
    print(f"   Average per sentence: {avg_entities:.1f} entities")
    print(f"   Total extracted: {sum(entity_counts)} entities")

    print(f"\n🔗 Relation Extraction:")
    print(f"   Average per sentence: {avg_triples:.1f} triples")
    print(f"   Total extracted: {sum(triple_counts)} triples")

    # Real-time capability assessment
    print(f"\n⚡ REAL-TIME CAPABILITY ASSESSMENT")
    print("-" * 40)

    real_time_threshold = 200  # <200ms for excellent real-time
    fast_threshold = 500       # <500ms for acceptable real-time

    excellent_count = sum(1 for t in extraction_times if t < real_time_threshold)
    fast_count = sum(1 for t in extraction_times if t < fast_threshold)

    excellent_percent = (excellent_count / len(extraction_times)) * 100
    fast_percent = (fast_count / len(extraction_times)) * 100

    print(f"   🟢 Excellent (<{real_time_threshold}ms): {excellent_count}/{len(extraction_times)} ({excellent_percent:.0f}%)")
    print(f"   🟡 Fast (<{fast_threshold}ms): {fast_count}/{len(extraction_times)} ({fast_percent:.0f}%)")
    print(f"   🔴 Slow (≥{fast_threshold}ms): {len(extraction_times) - fast_count}/{len(extraction_times)} ({100-fast_percent:.0f}%)")

    # Voice conversation suitability
    print(f"\n🎤 VOICE CONVERSATION SUITABILITY")
    print("-" * 40)

    voice_threshold = 300  # Voice conversation needs <300ms
    voice_ready = sum(1 for t in extraction_times if t < voice_threshold)
    voice_percent = (voice_ready / len(extraction_times)) * 100

    if voice_percent >= 90:
        voice_grade = "🟢 EXCELLENT - Ready for production"
    elif voice_percent >= 70:
        voice_grade = "🟡 GOOD - Suitable for voice conversations"
    elif voice_percent >= 50:
        voice_grade = "🟠 ACCEPTABLE - Works with some latency"
    else:
        voice_grade = "🔴 NOT SUITABLE - Too slow for voice"

    print(f"   Voice-ready extractions: {voice_ready}/{len(extraction_times)} ({voice_percent:.0f}%)")
    print(f"   Assessment: {voice_grade}")

    # Knowledge graph quality
    print(f"\n🧠 KNOWLEDGE GRAPH QUALITY")
    print("-" * 40)

    # Sample some high-quality extractions
    print("   High-quality extractions examples:")

    for i, (sentence, entities, triples) in enumerate(zip(test_sentences, entity_counts, triple_counts)):
        if entities >= 3 and triples >= 3:
            print(f"      '{sentence}' → {entities} entities, {triples} relations")
            if i >= 2:  # Show just a few examples
                break

    # Conclusion
    print(f"\n🎉 CONCLUSION")
    print("-" * 40)

    if avg_time < 200:
        print("✅ Tier 1 achieves EXCELLENT real-time performance!")
        print("   • Suitable for production voice conversations")
        print("   • High-quality knowledge graph generation")
        print("   • Ready for deployment")
    elif avg_time < 500:
        print("✅ Tier 1 achieves GOOD real-time performance!")
        print("   • Suitable for most voice applications")
        print("   • Good knowledge graph quality")
        print("   • Production ready with minor optimizations")
    else:
        print("⚠️  Tier 1 needs optimization for real-time usage")
        print("   • Currently too slow for voice conversations")
        print("   • Consider model optimization or caching strategies")

    print(f"\n💡 KEY METRICS FOR PRODUCTION:")
    print(f"   • Average extraction time: {avg_time:.1f}ms")
    print(f"   • Knowledge graph density: {avg_triples:.1f} relations per sentence")
    print(f"   • Entity recognition quality: {avg_entities:.1f} entities per sentence")
    print(f"   • Voice conversation readiness: {voice_percent:.0f}%")

    return {
        'avg_time_ms': avg_time,
        'p95_time_ms': p95_time,
        'voice_ready_percent': voice_percent,
        'avg_entities': avg_entities,
        'avg_triples': avg_triples
    }

if __name__ == "__main__":
    try:
        metrics = test_tier1_realtime_performance()
        print(f"\n🚀 Test completed successfully!")
        print(f"   Tier 1 real-time performance: {metrics['avg_time_ms']:.1f}ms average")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)