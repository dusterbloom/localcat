#!/usr/bin/env python3
"""
Test EntityResolver implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.entity.entity_resolver import EntityResolver
from collections import defaultdict

def test_entity_resolver():
    """Test entity resolution functionality"""
    print("🧪 TESTING ENTITY RESOLVER")
    print("=" * 50)
    
    # Test configuration
    config = {
        'entity_resolution_enabled': True,
        'entity_resolution_threshold': 0.85,
        'use_rapidfuzz_fallback': True,
        'force_rapidfuzz': True
    }
    
    # Initialize resolver
    resolver = EntityResolver(config)
    
    # Test entities with variations
    test_entities = [
        "John Smith",
        "john smith", 
        "J. Smith",
        "Smith, John",
        "Dr. John Smith",
        "Sarah Johnson",
        "sarah johnson",
        "S. Johnson",
        "OpenAI",
        "openai", 
        "Open AI",
        "Apple Inc.",
        "apple inc",
        "Apple",
        "New York",
        "new york",
        "NYC",
        "New York City"
    ]
    
    print(f"📊 Input entities: {len(test_entities)}")
    for i, entity in enumerate(test_entities, 1):
        print(f"   {i:2d}. {entity}")
    
    # Test entity resolution
    print(f"\n🔄 Testing entity resolution...")
    result = resolver.resolve_entities(test_entities)
    
    print(f"✅ Resolved entities: {len(set(result.resolved_entities.values()))}")
    
    # Group by resolved entity
    resolved_groups = defaultdict(list)
    for original, resolved in result.resolved_entities.items():
        resolved_groups[resolved].append(original)
    
    print(f"\n📋 Resolution groups:")
    for i, (canonical, variants) in enumerate(resolved_groups.items(), 1):
        if len(variants) > 1:
            print(f"   {i}. {canonical} <- {', '.join(variants)}")
    
    print(f"\n⏱️  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"📈 Method: {result.resolution_stats.get('method', 'unknown')}")
    print(f"📊 Stats: {result.resolution_stats}")
    
    # Test resolver stats
    stats = resolver.get_stats()
    print(f"\n🔧 Resolver stats: {stats}")
    
    return result

if __name__ == "__main__":
    result = test_entity_resolver()
    print(f"\n🎯 Entity resolution test completed!")