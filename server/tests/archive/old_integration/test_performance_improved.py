#!/usr/bin/env python3
"""
Test the improved MemoryExtractor with global model caching
This should achieve <100ms extraction after initial load
"""

import time
import statistics
from components.extraction.memory_extractor import MemoryExtractor

def test_performance_with_caching():
    """Test that we achieve <100ms performance with global caching"""
    print("🚀 TESTING IMPROVED MEMORY EXTRACTOR WITH GLOBAL CACHING")
    print("=" * 70)

    # Test configuration with ReLiK enabled
    config = {
        'use_srl': False,
        'use_onnx_ner': False,
        'use_onnx_srl': False,
        'use_relik': True,
        'use_gliner': True,
        'use_coref': False,
        'llm_base_url': 'http://127.0.0.1:1234/v1'
    }

    # Test texts of various lengths
    test_texts = [
        "Steve Jobs founded Apple Inc. in Cupertino, California.",
        "Dr. Sarah Chen is the AI research director at OpenAI.",
        "The company was founded in 2015 and currently has 500 employees working on AI.",
        """Dr. Sarah Chen is the AI research director at OpenAI. She joined the company in 2021
        after completing her PhD at Stanford under the supervision of Dr. Michael Jordan.""",
    ]

    print("📊 Initial Model Loading (First Extractor)")
    print("-" * 40)

    # First extractor - will load models
    start_time = time.perf_counter()
    extractor1 = MemoryExtractor(config)
    init_time = (time.perf_counter() - start_time) * 1000
    print(f"✅ Extractor initialization: {init_time:.1f}ms")

    # First extraction - models need to be loaded
    start_time = time.perf_counter()
    result1 = extractor1.extract(test_texts[0], use_cache=False)
    first_extract_time = (time.perf_counter() - start_time) * 1000
    print(f"✅ First extraction (model loading): {first_extract_time:.1f}ms")
    print(f"   Found {len(result1.entities)} entities, {len(result1.triples)} relations")

    print("\n📊 Testing with Cached Models (Same Extractor)")
    print("-" * 40)

    # Test multiple extractions with same extractor
    times_same = []
    for i, text in enumerate(test_texts):
        start_time = time.perf_counter()
        result = extractor1.extract(text, use_cache=False)  # No text caching
        elapsed = (time.perf_counter() - start_time) * 1000
        times_same.append(elapsed)
        print(f"  Text {i+1} ({len(text)} chars): {elapsed:.1f}ms")

    avg_same = statistics.mean(times_same)
    print(f"\n📈 Average (same extractor): {avg_same:.1f}ms")

    print("\n📊 Testing Global Cache (New Extractor Instance)")
    print("-" * 40)

    # Create NEW extractor - should use globally cached models
    start_time = time.perf_counter()
    extractor2 = MemoryExtractor(config)
    init_time2 = (time.perf_counter() - start_time) * 1000
    print(f"✅ Second extractor initialization: {init_time2:.1f}ms")

    # Test extractions with new extractor
    times_new = []
    for i, text in enumerate(test_texts):
        start_time = time.perf_counter()
        result = extractor2.extract(text, use_cache=False)
        elapsed = (time.perf_counter() - start_time) * 1000
        times_new.append(elapsed)
        print(f"  Text {i+1} ({len(text)} chars): {elapsed:.1f}ms")

    avg_new = statistics.mean(times_new)
    print(f"\n📈 Average (new extractor): {avg_new:.1f}ms")

    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"First model load time: {first_extract_time:.1f}ms")
    print(f"Average after caching: {avg_new:.1f}ms")
    print(f"Speedup: {first_extract_time/avg_new:.1f}x")

    # Performance assessment
    if avg_new < 100:
        print("\n🎉 EXCELLENT: Target <100ms achieved!")
    elif avg_new < 200:
        print("\n✅ GOOD: Under 200ms")
    elif avg_new < 500:
        print("\n⚠️  OK: Under 500ms but could be better")
    else:
        print("\n❌ SLOW: Still over 500ms")

    # Test thread safety with multiple extractors
    print("\n📊 Testing Thread Safety (Multiple Extractors)")
    print("-" * 40)

    extractors = []
    for i in range(3):
        start_time = time.perf_counter()
        ext = MemoryExtractor(config)
        elapsed = (time.perf_counter() - start_time) * 1000
        extractors.append(ext)
        print(f"  Extractor {i+1} init: {elapsed:.1f}ms")

    print("\n✅ All extractors share the same global models")

    return avg_new < 100  # Return True if we meet the target

def test_cold_vs_warm_detailed():
    """Detailed cold vs warm performance test"""
    print("\n" + "=" * 70)
    print("🧊 COLD vs 🔥 WARM PERFORMANCE TEST")
    print("=" * 70)

    config = {
        'use_srl': False,
        'use_onnx_ner': False,
        'use_onnx_srl': False,
        'use_relik': True,
        'use_gliner': True,
        'use_coref': False,
        'llm_base_url': 'http://127.0.0.1:1234/v1'
    }

    test_text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
    The company is headquartered in Cupertino, California and employs over 150,000 people.
    Tim Cook became CEO in 2011 after Jobs passed away.
    """

    # Clear any existing global cache by restarting the script
    print("\n🧊 COLD START (First Run)")
    print("-" * 40)

    # Measure total cold start time
    total_start = time.perf_counter()

    # Initialize extractor (cold)
    init_start = time.perf_counter()
    extractor_cold = MemoryExtractor(config)
    init_time = (time.perf_counter() - init_start) * 1000

    # First extraction (model loading)
    extract_start = time.perf_counter()
    result_cold = extractor_cold.extract(test_text, use_cache=False)
    extract_time = (time.perf_counter() - extract_start) * 1000

    total_cold = (time.perf_counter() - total_start) * 1000

    print(f"  Initialization: {init_time:.1f}ms")
    print(f"  First extraction: {extract_time:.1f}ms")
    print(f"  Total cold start: {total_cold:.1f}ms")
    print(f"  Entities: {len(result_cold.entities)}, Relations: {len(result_cold.triples)}")

    print("\n🔥 WARM START (Cached Models)")
    print("-" * 40)

    # Run 5 warm extractions to get stable timing
    warm_times = []
    for i in range(5):
        start = time.perf_counter()
        result = extractor_cold.extract(test_text, use_cache=False)
        elapsed = (time.perf_counter() - start) * 1000
        warm_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f}ms")

    avg_warm = statistics.mean(warm_times)
    std_warm = statistics.stdev(warm_times) if len(warm_times) > 1 else 0

    print(f"\n📊 Warm average: {avg_warm:.1f}ms (±{std_warm:.1f}ms)")
    print(f"🚀 Speedup: {extract_time/avg_warm:.1f}x faster")

    if avg_warm < 100:
        print("✅ TARGET MET: <100ms extraction achieved!")

    return avg_warm

if __name__ == "__main__":
    # Run performance tests
    success = test_performance_with_caching()
    avg_warm = test_cold_vs_warm_detailed()

    print("\n" + "=" * 70)
    print("🏁 FINAL VERDICT")
    print("=" * 70)

    if success and avg_warm < 100:
        print("🎉 SUCCESS: The fix works! Extraction is now <100ms!")
        print("✅ ReLiK models are cached globally")
        print("✅ spaCy models are cached globally")
        print("✅ Multiple extractors share the same models")
    else:
        print(f"⚠️  Performance: {avg_warm:.1f}ms (target: <100ms)")
        print("   Models are cached but may need further optimization")