#!/usr/bin/env python3
"""
Test TemporalContextExtractor implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.temporal.temporal_extractor import TemporalContextExtractor

def test_temporal_extractor():
    """Test temporal context extraction functionality"""
    print("🧪 TESTING TEMPORAL CONTEXT EXTRACTOR")
    print("=" * 50)
    
    # Test configuration
    config = {
        'temporal_extraction_enabled': True,
        'temporal_confidence_threshold': 0.5,
        'include_time_in_relationships': True,
        'use_spacy_fallback': True
    }
    
    # Initialize extractor
    extractor = TemporalContextExtractor(config)
    
    # Test text with temporal information
    test_text = """
    John Smith joined OpenAI in 2021 after completing his PhD at Stanford in 2020. 
    He worked on GPT-4 development from 2022 to 2023. 
    Yesterday, Sarah Johnson announced a new AI model that will be released next month.
    The company was founded in 2015 and currently has 500 employees.
    """
    
    # Test triples
    test_triples = [
        ("John Smith", "joined", "OpenAI"),
        ("John Smith", "completed", "PhD at Stanford"),
        ("John Smith", "worked on", "GPT-4 development"),
        ("Sarah Johnson", "announced", "new AI model"),
        ("new AI model", "will be released", "next month"),
        ("company", "founded in", "2015"),
        ("company", "has", "500 employees")
    ]
    
    print(f"📊 Input relationships: {len(test_triples)}")
    for i, triple in enumerate(test_triples, 1):
        print(f"   {i:2d}. {triple}")
    
    print(f"\n📝 Test text: {test_text.strip()}")
    
    # Test temporal extraction
    print(f"\n🔄 Testing temporal context extraction...")
    result = extractor.extract_temporal_context(test_triples, test_text)
    
    print(f"✅ Enhanced relationships: {len(result.enhanced_triples)}")
    print(f"📅 Temporal contexts: {len(result.temporal_contexts)}")
    
    print(f"\n📋 ENHANCED relationships:")
    for i, enhanced_triple in enumerate(result.enhanced_triples, 1):
        subject, predicate, obj, temporal_context = enhanced_triple
        context_info = ""
        if temporal_context:
            expr_count = len(temporal_context.temporal_expressions)
            types = ", ".join(temporal_context.temporal_stats.get('types', []))
            context_info = f" [Temporal: {expr_count} exprs, types: {types}]"
        print(f"   {i:2d}. {subject} {predicate} {obj}{context_info}")
    
    print(f"\n📅 TEMPORAL CONTEXTS:")
    for i, context in enumerate(result.temporal_contexts, 1):
        print(f"   {i}. Type: {context.temporal_stats.get('type', 'unknown')}")
        print(f"      Expressions: {[expr.text for expr in context.temporal_expressions]}")
        print(f"      Confidence: {context.temporal_stats.get('confidence', 0):.2f}")
    
    print(f"\n⏱️  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"📈 Method: {result.extraction_stats.get('method', 'unknown')}")
    print(f"📊 Stats: {result.extraction_stats}")
    
    # Test extractor stats
    stats = extractor.get_stats()
    print(f"\n🔧 Extractor stats: {stats}")
    
    return result

if __name__ == "__main__":
    result = test_temporal_extractor()
    print(f"\n🎯 Temporal extraction test completed!")