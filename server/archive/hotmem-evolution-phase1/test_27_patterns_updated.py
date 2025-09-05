#!/usr/bin/env python3
"""
Test USGS 27-pattern extraction with current HotMem system
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

os.environ["HOTMEM_SQLITE"] = ":memory:"

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory
import time

def test_27_patterns():
    """Test current HotMem system against comprehensive patterns"""
    
    print("🚀 Testing HotMem Against USGS 27-Pattern Extraction")
    print("="*70)
    
    # Initialize
    paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
    store = MemoryStore(paths)
    hot = HotMemory(store)
    
    # Prewarm models to eliminate first-call latency
    print("🔄 Prewarming spaCy models...")
    prewarm_start = time.perf_counter()
    hot.prewarm("en")
    prewarm_ms = (time.perf_counter() - prewarm_start) * 1000
    print(f"✅ Prewarming completed in {prewarm_ms:.1f}ms")
    
    # Test sentences covering many patterns
    test_cases = [
        # Basic patterns
        "My name is Alex Thompson",
        "I live in Seattle", 
        "I work at Microsoft",
        "My dog's name is Potola",
        
        # Complex patterns from locomo
        "Caroline went to the LGBTQ support group",
        "Melanie painted a sunrise in 2022",
        "Caroline moved from Sweden 4 years ago",
        "Caroline has had her current group of friends for 4 years",
        "Melanie has read Nothing is Impossible and Charlotte's Web",
        "Caroline participated in a pride parade",
        
        # Various dependency types
        "The old red car belongs to me",
        "Sarah and John are friends",
        "I have three pets",
        "My favorite color is blue",
        "I was born in 1995",
        "My son is named Jake",
        
        # Additional challenging cases
        "My favorite number is 77",
        "I enjoy coding and reading",
        "The book is on the table",
        "Caroline is a talented developer"
    ]
    
    total_time = 0
    successful_extractions = 0
    total_triples = 0
    
    for i, text in enumerate(test_cases, 1):
        start_time = time.perf_counter()
        bullets, triples = hot.process_turn(text, "test", i)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_time += elapsed_ms
        
        print(f"\n{i:2d}. '{text}'")
        print(f"     ⏱️  {elapsed_ms:.1f}ms")
        
        if triples:
            successful_extractions += 1
            total_triples += len(triples)
            print(f"     ✅ {len(triples)} triples:")
            for t in triples[:3]:  # Show first 3
                print(f"        {t}")
            if len(triples) > 3:
                print(f"        ... and {len(triples) - 3} more")
        else:
            print("     ❌ No triples extracted")
    
    # Performance summary
    avg_time = total_time / len(test_cases)
    success_rate = (successful_extractions / len(test_cases)) * 100
    avg_triples = total_triples / successful_extractions if successful_extractions > 0 else 0
    
    print("\n" + "="*70)
    print("📊 EXTRACTION PERFORMANCE SUMMARY")
    print("="*70)
    print(f"🎯 Success Rate: {successful_extractions}/{len(test_cases)} ({success_rate:.1f}%)")
    print(f"⚡ Average Time: {avg_time:.1f}ms per sentence")  
    print(f"📈 Total Time: {total_time:.1f}ms for {len(test_cases)} sentences")
    print(f"🔢 Average Triples: {avg_triples:.1f} per successful extraction")
    print(f"🏆 Budget Target: <30ms ({'✅ PASS' if avg_time < 30 else '❌ FAIL'})")
    
    # Test retrieval (non-mutating)
    print("\n" + "="*70)
    print("🔍 RETRIEVAL TEST")
    print("="*70)
    
    queries = [
        "What do you know about Caroline?",
        "Where do I live?",
        "What is my dog's name?",
        "What is my favorite number?",
        "Who are my friends?"
    ]
    
    for query in queries:
        preview = hot.preview_bullets(query)
        bullets = preview.get("bullets", [])
        print(f"\nQ: '{query}'")
        if bullets:
            for bullet in bullets[:3]:  # Show top 3
                print(f"   {bullet}")
            if len(bullets) > 3:
                print(f"   ... and {len(bullets) - 3} more")
        else:
            print("   No memories retrieved")
    
    # Detailed metrics
    print("\n" + "="*70)  
    print("⚙️  DETAILED PERFORMANCE METRICS")
    print("="*70)
    
    metrics = hot.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict) and 'p95' in value:
            print(f"{key}: p95={value['p95']:.1f}ms, mean={value['mean']:.1f}ms")
        else:
            print(f"{key}: {value}")
    
    return {
        'success_rate': success_rate,
        'avg_time_ms': avg_time,
        'total_triples': total_triples,
        'avg_triples': avg_triples
    }

if __name__ == "__main__":
    results = test_27_patterns()
    
    print(f"\n🎉 Test Complete!")
    print(f"Overall Quality Score: {results['success_rate']:.1f}% extraction success")
    print(f"Performance Score: {results['avg_time_ms']:.1f}ms average (target: <30ms)")