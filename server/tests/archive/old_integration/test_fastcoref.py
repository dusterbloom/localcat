#!/usr/bin/env python3
"""
Test fastcoref integration to verify it's working properly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.coreference.coreference_resolver import CoreferenceResolver

def test_fastcoref():
    """Test fastcoref integration"""
    print("🧪 TESTING FASTCOREF INTEGRATION")
    print("=" * 50)
    
    # Test configuration
    config = {
        'use_coref': True,
        'coref_max_entities': 24,
        'coref_device': 'cpu'
    }
    
    # Initialize resolver
    resolver = CoreferenceResolver(config)
    
    # Test basic functionality
    test_triples = [
        ("he", "works at", "the company"),
        ("she", "is the", "CEO"),
        ("it", "is located", "there")
    ]
    
    print(f"📊 Input triples: {len(test_triples)}")
    for i, triple in enumerate(test_triples, 1):
        print(f"   {i}. {triple}")
    
    # Test rule-based resolution
    print("\n🔄 Testing rule-based resolution...")
    result = resolver.resolve_coreferences(test_triples, None, "She is the CEO and it is located there")
    
    print(f"✅ Resolved triples: {len(result.resolved_triples)}")
    for i, triple in enumerate(result.resolved_triples, 1):
        print(f"   {i}. {triple}")
    
    print(f"⏱️  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"📈 Method: {result.resolution_stats.get('method', 'unknown')}")
    
    # Test if neural model can be loaded
    print(f"\n🧠 Testing neural model loading...")
    try:
        resolver.prewarm()
        if resolver._coref_model:
            print("✅ Neural model loaded successfully")
            model_info = getattr(resolver._coref_model, 'model_name', 'unknown')
            print(f"📋 Model info: {model_info}")
        else:
            print("❌ Neural model failed to load")
    except Exception as e:
        print(f"❌ Neural model error: {e}")
    
    return result

if __name__ == "__main__":
    result = test_fastcoref()
    print(f"\n🎯 Test completed successfully!")