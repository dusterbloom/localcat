#!/usr/bin/env python3
"""
Test USGS 27-pattern extraction
"""

import os
os.environ["HOTMEM_SQLITE"] = ":memory:"

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory

# Initialize
paths = Paths(sqlite_path=":memory:", lmdb_dir=None)
store = MemoryStore(paths)
hot = HotMemory(store)

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
    "My son is named Jake"
]

print("Testing USGS 27-Pattern Extraction")
print("="*70)

for i, text in enumerate(test_cases, 1):
    bullets, triples = hot.process_turn(text, "test", i)
    
    print(f"\n{i}. '{text}'")
    if triples:
        print("   Triples:")
        for t in triples:
            print(f"     {t}")
    else:
        print("   ❌ No triples extracted")

# Test retrieval (non-mutating)
print("\n" + "="*70)
print("RETRIEVAL TEST")
print("="*70)

queries = [
    "What do you know about Caroline?",
    "Where do I live?",
    "What is my dog's name?"
]

for query in queries:
    preview = hot.preview_bullets(query)
    bullets = preview.get("bullets", [])
    print(f"\nQuery: '{query}'")
    if bullets:
        for bullet in bullets:
            print(f"  {bullet}")
    else:
        print("  No memories retrieved")

# Metrics
print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

metrics = hot.get_metrics()
for key, value in metrics.items():
    if isinstance(value, dict) and 'p95' in value:
        print(f"{key}: p95={value['p95']:.1f}ms, mean={value['mean']:.1f}ms")
    else:
        print(f"{key}: {value}")
