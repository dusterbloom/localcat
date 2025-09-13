#!/usr/bin/env python3
"""
Test EnhancedCoreferenceResolver implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.coreference.enhanced_coref_resolver import EnhancedCoreferenceResolver

def test_enhanced_coref_resolver():
    """Test enhanced coreference resolution functionality"""
    print("🧪 TESTING ENHANCED COREFERENCE RESOLVER")
    print("=" * 50)
    
    # Test configuration
    config = {
        'enhanced_coref_enabled': True,
        'use_spacy_coref': True,
        'coref_confidence_threshold': 0.7,
        'max_coref_entities': 50
    }
    
    # Initialize resolver
    resolver = EnhancedCoreferenceResolver(config)
    
    # Test text with coreferences
    test_text = """
    Dr. Sarah Chen is the AI research director at OpenAI. She joined the company in 2021 
    after completing her PhD at Stanford under the supervision of Dr. Michael Jordan. 
    The researcher previously worked at Google Brain where she collaborated with Dr. Fei-Fei Li. 
    They developed several groundbreaking papers together. The company was founded in 2015 
    and currently has 500 employees. It is known for its advanced language models.
    """
    
    # Test triples with pronouns and references
    test_triples = [
        ("Dr. Sarah Chen", "is", "AI research director"),
        ("She", "joined", "OpenAI"),
        ("She", "completed", "PhD at Stanford"),
        ("She", "supervised by", "Dr. Michael Jordan"),
        ("The researcher", "worked at", "Google Brain"),
        ("she", "collaborated with", "Dr. Fei-Fei Li"),
        ("They", "developed", "groundbreaking papers"),
        ("The company", "founded", "2015"),
        ("It", "has", "500 employees"),
        ("It", "known for", "advanced language models")
    ]
    
    print(f"📊 Input relationships: {len(test_triples)}")
    for i, triple in enumerate(test_triples, 1):
        print(f"   {i:2d}. {triple}")
    
    print(f"\n📝 Test text: {test_text.strip()}")
    
    # Test enhanced coreference resolution
    print(f"\n🔄 Testing enhanced coreference resolution...")
    result = resolver.resolve_coreferences(test_triples, test_text)
    
    print(f"✅ Resolved relationships: {len(result.resolved_triples)}")
    print(f"🔗 Coreference chains: {len(result.coreference_chains)}")
    
    print(f"\n📋 RESOLVED relationships:")
    for i, (original, resolved) in enumerate(zip(test_triples, result.resolved_triples), 1):
        if original != resolved:
            print(f"   {i:2d}. {original} → {resolved}")
        else:
            print(f"   {i:2d}. {resolved} (unchanged)")
    
    print(f"\n🔗 COREFERENCE CHAINS:")
    for i, chain in enumerate(result.coreference_chains, 1):
        print(f"   {i}. Main: {chain.main_entity}")
        print(f"      Mentions: {', '.join(chain.mentions)}")
        print(f"      Confidence: {chain.confidence:.2f}")
    
    print(f"\n⏱️  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"📈 Method: {result.resolution_stats.get('method', 'unknown')}")
    print(f"📊 Stats: {result.resolution_stats}")
    
    # Test resolver stats
    stats = resolver.get_stats()
    print(f"\n🔧 Resolver stats: {stats}")
    
    return result

if __name__ == "__main__":
    result = test_enhanced_coref_resolver()
    print(f"\n🎯 Enhanced coreference resolution test completed!")