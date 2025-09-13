#!/usr/bin/env python3
"""
Test MPS-optimized ReLiK integration with existing MemoryExtractor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import torch
from components.extraction.memory_extractor import MemoryExtractor

def test_mps_optimized_extractor():
    """Test the MPS-optimized MemoryExtractor with ReLiK"""
    print("🚀 TESTING MPS-OPTIMIZED MEMORY EXTRACTOR")
    print("=" * 60)
    
    # Check device availability
    print(f"📊 Device Info:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    print(f"   MPS built: {torch.backends.mps.is_built()}")
    print()
    
    # Test configuration with ReLiK enabled
    config = {
        'use_srl': False,  # Disable for cleaner test
        'use_onnx_ner': False,
        'use_onnx_srl': False,
        'use_relik': True,  # Enable ReLiK
        'use_gliner': True,
        'use_coref': False,
        'llm_base_url': 'http://127.0.0.1:1234/v1'
    }
    
    # Initialize extractor
    print("🔧 Initializing MemoryExtractor with MPS optimization...")
    extractor = MemoryExtractor(config)
    
    # Test text
    test_text = """
    Dr. Sarah Chen is the AI research director at OpenAI. She joined the company in 2021 
    after completing her PhD at Stanford under the supervision of Dr. Michael Jordan. 
    The researcher previously worked at Google Brain where she collaborated with Dr. Fei-Fei Li. 
    They developed several groundbreaking papers on transformer architectures. 
    The company was founded in 2015 and currently has 500 employees.
    """
    
    print(f"📝 Test text length: {len(test_text)} characters")
    print()
    
    # Test extraction without cache (cold start)
    print("🔄 Testing extraction (cold start)...")
    start_time = time.perf_counter()
    result = extractor.extract(test_text, use_cache=False)
    cold_time = (time.perf_counter() - start_time) * 1000
    
    print(f"✅ Cold start extraction completed in {cold_time:.1f}ms")
    print(f"   Entities found: {len(result.entities)}")
    print(f"   Relations found: {len(result.triples)}")
    print()
    
    # Show some results
    print("📋 Sample entities:")
    for entity in list(result.entities)[:5]:
        print(f"   - {entity}")
    print()
    
    print("📋 Sample relations:")
    for i, triple in enumerate(result.triples[:5]):
        print(f"   {i+1}. {triple}")
    print()
    
    # Test extraction with cache (warm start)
    print("🔄 Testing extraction (warm start with cache)...")
    start_time = time.perf_counter()
    result_cached = extractor.extract(test_text, use_cache=True)
    warm_time = (time.perf_counter() - start_time) * 1000
    
    print(f"✅ Warm start extraction completed in {warm_time:.1f}ms")
    print(f"   Cache hit: {result_cached.entities == result.entities}")
    print()
    
    # Test multiple runs for average performance
    print("📊 Testing average performance (5 runs)...")
    times = []
    for i in range(5):
        start_time = time.perf_counter()
        result = extractor.extract(test_text, use_cache=True)
        run_time = (time.perf_counter() - start_time) * 1000
        times.append(run_time)
        print(f"   Run {i+1}: {run_time:.1f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📈 Performance Summary:")
    print(f"   Cold start: {cold_time:.1f}ms")
    print(f"   Warm start avg: {avg_time:.1f}ms")
    print(f"   Warm start min: {min_time:.1f}ms")
    print(f"   Warm start max: {max_time:.1f}ms")
    print(f"   Speedup: {cold_time/avg_time:.1f}x")
    
    # Get extractor metrics
    metrics = extractor.get_metrics()
    if metrics:
        print(f"\n📊 Extractor Metrics:")
        for key, values in metrics.items():
            if values:
                avg = sum(values) / len(values)
                print(f"   {key}: avg {avg:.1f}ms ({len(values)} runs)")
    
    # Test cache effectiveness
    if hasattr(extractor, '_extraction_cache'):
        cache_size = len(extractor._extraction_cache)
        print(f"\n💾 Cache Info:")
        print(f"   Cache size: {cache_size} entries")
        print(f"   Cache hit rate: {metrics.get('cache_hits', 0) / max(1, extractor.stats.get('total_calls', 1)):.1%}")
    
    print(f"\n🎯 MPS optimization test completed!")
    return result

if __name__ == "__main__":
    result = test_mps_optimized_extractor()