#!/usr/bin/env python3
"""
Test SemanticRelationshipFilter implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.semantic.semantic_filter import SemanticRelationshipFilter

def test_semantic_filter():
    """Test semantic relationship filtering functionality"""
    print("🧪 TESTING SEMANTIC RELATIONSHIP FILTER")
    print("=" * 50)
    
    # Test configuration
    config = {
        'semantic_filtering_enabled': True,
        'semantic_similarity_threshold': 0.8,
        'min_relationship_confidence': 0.5,
        'use_spacy_fallback': True
    }
    
    # Initialize filter
    filter_obj = SemanticRelationshipFilter(config)
    
    # Test relationships with various issues
    test_triples = [
        # Generic relationships that should be removed
        ("John", "is a", "person"),
        ("Apple", "is an", "company"),
        ("OpenAI", "has a", "CEO"),
        ("Sarah", "works at", "OpenAI"),
        ("OpenAI", "employs", "Sarah"),
        
        # Self-references
        ("John", "knows", "John"),
        ("AI", "improves", "AI"),
        
        # Semantically similar relationships
        ("John", "works for", "OpenAI"),
        ("Sarah", "is employed by", "OpenAI"),
        ("OpenAI", "has employees", "researchers"),
        
        # Good relationships that should be kept
        ("OpenAI", "developed", "GPT"),
        ("GPT", "processes", "natural language"),
        ("John", "researches", "machine learning"),
        ("Sarah", "leads", "AI research team")
    ]
    
    print(f"📊 Input relationships: {len(test_triples)}")
    for i, triple in enumerate(test_triples, 1):
        print(f"   {i:2d}. {triple}")
    
    # Test semantic filtering
    print(f"\n🔄 Testing semantic filtering...")
    result = filter_obj.filter_relationships(test_triples)
    
    print(f"✅ Filtered relationships: {len(result.filtered_triples)}")
    print(f"❌ Removed relationships: {len(result.removed_triples)}")
    
    print(f"\n📋 KEPT relationships:")
    for i, triple in enumerate(result.filtered_triples, 1):
        print(f"   {i:2d}. {triple}")
    
    print(f"\n🗑️  REMOVED relationships:")
    for i, triple in enumerate(result.removed_triples, 1):
        print(f"   {i:2d}. {triple}")
    
    print(f"\n⏱️  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"📈 Method: {result.filter_stats.get('method', 'unknown')}")
    print(f"📊 Stats: {result.filter_stats}")
    
    # Test semantic similarity
    print(f"\n🧠 Testing semantic similarity...")
    test_pairs = [
        ("works for", "employed by"),
        ("developed", "created"),
        ("researches", "studies"),
        ("leads", "manages")
    ]
    
    for text1, text2 in test_pairs:
        similarity = filter_obj.get_semantic_similarity(text1, text2)
        print(f"   '{text1}' vs '{text2}': {similarity:.2f}")
    
    # Test filter stats
    stats = filter_obj.get_stats()
    print(f"\n🔧 Filter stats: {stats}")
    
    return result

if __name__ == "__main__":
    result = test_semantic_filter()
    print(f"\n🎯 Semantic filtering test completed!")