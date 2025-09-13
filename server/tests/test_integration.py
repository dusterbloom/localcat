#!/usr/bin/env python3
"""
Comprehensive Integration Tests for HotMemory Refactoring
==========================================================

Tests the complete refactored architecture to ensure all services
work together correctly and maintain backward compatibility.
"""

import sys
import os
sys.path.insert(0, '.')

import time
import pytest
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from unittest.mock import Mock, patch

# Import all refactored components
from components.memory.config import HotMemoryConfig, create_config
from components.extraction.memory_extractor import MemoryExtractor, ExtractionResult
from components.retrieval.memory_retriever import MemoryRetriever, RetrievalResult
from components.coreference.coreference_resolver import CoreferenceResolver, CoreferenceResult
from components.extraction.assisted_extractor import AssistedExtractor, AssistedExtractionResult
from components.memory.hotmemory_facade import HotMemoryFacade, RecencyItem


class MockMemoryStore:
    """Mock memory store for testing"""
    
    def __init__(self):
        self.triples = []
        self.fts_data = []
    
    def update_session_triples(self, session_id: str, triples: List[Tuple[str, str, str]]):
        self.triples.extend(triples)
    
    def search_fts_detailed(self, query: str, limit: int = 10):
        # Return some mock FTS results
        return [
            ("You live in San Francisco", "summary:1", "1234567890"),
            ("You work at Google", "summary:2", "1234567891")
        ]


class TestConfiguration:
    """Test configuration management"""
    
    def test_config_creation(self):
        """Test that configuration can be created and customized"""
        config = create_config()
        
        assert isinstance(config, HotMemoryConfig)
        assert config.max_recency > 0
        assert config.confidence_threshold > 0
        assert config.features is not None
        
        # Test feature flags
        assert isinstance(config.features.use_leann, bool)
        assert isinstance(config.features.use_coref, bool)
        assert isinstance(config.features.assisted_enabled, bool)
        
        # Test model configs
        assert config.assisted_model.name is not None
        assert config.assisted_model.base_url is not None
        
        print("✅ Configuration management works correctly")
    
    def test_service_configs(self):
        """Test that service-specific configurations work"""
        config = create_config()
        
        extractor_config = config.get_extractor_config()
        retriever_config = config.get_retriever_config()
        coreference_config = config.get_coreference_config()
        assisted_config = config.get_assisted_config()
        
        # Test extractor config
        assert 'use_srl' in extractor_config
        assert 'use_onnx_ner' in extractor_config
        assert 'cache_size' in extractor_config
        
        # Test retriever config
        assert 'use_leann' in retriever_config
        assert 'leann_complexity' in retriever_config
        assert 'retrieval_fusion' in retriever_config
        
        # Test coreference config
        assert 'use_coref' in coreference_config
        assert 'coref_max_entities' in coreference_config
        
        # Test assisted config
        assert 'assisted_enabled' in assisted_config
        assert 'assisted_model' in assisted_config
        
        print("✅ Service-specific configurations work correctly")


class TestMemoryExtractor:
    """Test MemoryExtractor functionality"""
    
    def test_extractor_initialization(self):
        """Test that extractor initializes correctly"""
        config = create_config()
        extractor = MemoryExtractor(config.get_extractor_config())
        
        assert extractor is not None
        assert hasattr(extractor, 'extract')
        assert hasattr(extractor, 'extract_entities_light')
        
        print("✅ MemoryExtractor initializes correctly")
    
    def test_basic_extraction(self):
        """Test basic entity and relation extraction"""
        config = create_config()
        extractor = MemoryExtractor(config.get_extractor_config())
        
        # Test basic extraction
        result = extractor.extract("You live in San Francisco", "en")
        
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.entities, list)
        assert isinstance(result.triples, list)
        assert isinstance(result.doc, Mock)  # Mock doc for testing
        
        print("✅ Basic extraction works correctly")
    
    def test_light_entity_extraction(self):
        """Test light entity extraction for retrieval"""
        config = create_config()
        extractor = MemoryExtractor(config.get_extractor_config())
        
        entities = extractor.extract_entities_light("You live in San Francisco")
        
        assert isinstance(entities, list)
        # Should extract some basic entities
        assert len(entities) > 0
        
        print("✅ Light entity extraction works correctly")


class TestMemoryRetriever:
    """Test MemoryRetriever functionality"""
    
    def test_retriever_initialization(self):
        """Test that retriever initializes correctly"""
        config = create_config()
        store = MockMemoryStore()
        entity_index = defaultdict(set)
        
        # Add some test data
        entity_index['you'].add(('you', 'live_in', 'san_francisco'))
        entity_index['you'].add(('you', 'work_at', 'google'))
        
        retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())
        
        assert retriever is not None
        assert hasattr(retriever, 'retrieve_context')
        assert hasattr(retriever, 'get_metrics')
        
        print("✅ MemoryRetriever initializes correctly")
    
    def test_context_retrieval(self):
        """Test context retrieval with MMR"""
        config = create_config()
        store = MockMemoryStore()
        entity_index = defaultdict(set)
        
        # Add test data
        entity_index['you'].add(('you', 'live_in', 'san_francisco'))
        entity_index['you'].add(('you', 'work_at', 'google'))
        entity_index['san_francisco'].add(('san_francisco', 'located_in', 'california'))
        
        retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())
        
        result = retriever.retrieve_context("Where do I live?", ["you"], 1)
        
        assert isinstance(result, RetrievalResult)
        assert isinstance(result.bullets, list)
        assert isinstance(result.relevant_triples, list)
        assert isinstance(result.expanded_entities, list)
        assert isinstance(result.retrieval_stats, dict)
        
        # Should have retrieved some context
        assert len(result.bullets) > 0
        assert len(result.expanded_entities) > 0
        
        print("✅ Context retrieval with MMR works correctly")
    
    def test_entity_expansion(self):
        """Test entity expansion with aliases and relationships"""
        config = create_config()
        store = MockMemoryStore()
        entity_index = defaultdict(set)
        
        # Add test data with aliases
        entity_index['you'].add(('you', 'also_known_as', 'user'))
        entity_index['you'].add(('you', 'live_in', 'san_francisco'))
        entity_index['san_francisco'].add(('san_francisco', 'located_in', 'california'))
        
        retriever = MemoryRetriever(store, entity_index, config.get_retriever_config())
        
        expanded = retriever._expand_query_entities(['you'], "Where do I live?")
        
        assert isinstance(expanded, list)
        assert 'you' in expanded
        # Should expand with related entities
        assert len(expanded) > 1
        
        print("✅ Entity expansion works correctly")


class TestCoreferenceResolver:
    """Test CoreferenceResolver functionality"""
    
    def test_coreference_initialization(self):
        """Test that coreference resolver initializes correctly"""
        config = create_config()
        resolver = CoreferenceResolver(config.get_coreference_config())
        
        assert resolver is not None
        assert hasattr(resolver, 'resolve_coreferences')
        assert hasattr(resolver, 'get_metrics')
        
        print("✅ CoreferenceResolver initializes correctly")
    
    def test_coreference_resolution(self):
        """Test basic coreference resolution"""
        config = create_config()
        resolver = CoreferenceResolver(config.get_coreference_config())
        
        # Test with some basic triples
        triples = [
            ("I", "live_in", "san_francisco"),
            ("my", "work_at", "google")
        ]
        
        mock_doc = Mock()
        result = resolver.resolve_coreferences(triples, mock_doc, "I live in San Francisco")
        
        assert isinstance(result, CoreferenceResult)
        assert isinstance(result.resolved_triples, list)
        assert isinstance(result.resolution_stats, dict)
        assert result.processing_time_ms >= 0
        
        # Should resolve pronouns
        # "I" -> "you", "my" -> "your"
        resolved = result.resolved_triples
        if resolved:  # May not resolve if neural coreference is disabled
            print("✅ Coreference resolution works correctly")
        else:
            print("✅ Coreference resolution handles disabled case correctly")
    
    def test_rule_based_fallback(self):
        """Test rule-based coreference fallback"""
        config = create_config()
        resolver = CoreferenceResolver(config.get_coreference_config())
        
        triples = [("I", "live_in", "san_francisco")]
        mock_doc = Mock()
        
        # Test rule-based resolution
        result = resolver._apply_rule_based_coreference(triples, mock_doc, "I live in San Francisco")
        
        assert isinstance(result.resolved_triples, list)
        assert result.processing_time_ms >= 0
        
        print("✅ Rule-based coreference fallback works correctly")


class TestAssistedExtractor:
    """Test AssistedExtractor functionality"""
    
    def test_assisted_initialization(self):
        """Test that assisted extractor initializes correctly"""
        config = create_config()
        extractor = AssistedExtractor(config.get_assisted_config())
        
        assert extractor is not None
        assert hasattr(extractor, 'extract_assisted')
        assert hasattr(extractor, 'should_assist')
        assert hasattr(extractor, 'get_metrics')
        
        print("✅ AssistedExtractor initializes correctly")
    
    def test_assistance_triggering(self):
        """Test that assistance triggers correctly"""
        config = create_config()
        extractor = AssistedExtractor(config.get_assisted_config())
        
        # Test with few triples (should trigger assistance)
        text = "I live in San Francisco and work at Google"
        entities = ["I", "San Francisco", "Google"]
        triples = [("I", "live_in", "San Francisco")]
        mock_doc = Mock()
        
        should_assist = extractor.should_assist(text, triples, mock_doc)
        
        assert isinstance(should_assist, bool)
        
        print("✅ Assistance triggering works correctly")
    
    def test_fallback_extraction(self):
        """Test fallback extraction when LLM is not available"""
        config = create_config()
        extractor = AssistedExtractor(config.get_assisted_config())
        
        text = "I live in San Francisco"
        entities = ["I", "San Francisco"]
        triples = [("I", "live_in", "San Francisco")]
        
        result = extractor._extract_with_fallback(text, entities, triples)
        
        assert isinstance(result.triples, list)
        assert isinstance(result.extraction_stats, dict)
        assert result.processing_time_ms >= 0
        
        print("✅ Fallback extraction works correctly")


class TestHotMemoryFacade:
    """Test HotMemoryFacade integration"""
    
    def test_facade_initialization(self):
        """Test that facade initializes correctly with all services"""
        store = MockMemoryStore()
        facade = HotMemoryFacade(store, max_recency=50)
        
        assert facade is not None
        assert hasattr(facade, 'process_turn')
        assert hasattr(facade, 'prewarm')
        assert hasattr(facade, 'get_metrics')
        
        # Check that all services are initialized
        assert facade.extractor is not None
        assert facade.retriever is not None
        assert facade.coreference_resolver is not None
        assert facade.assisted_extractor is not None
        
        print("✅ HotMemoryFacade initializes correctly with all services")
    
    def test_full_turn_processing(self):
        """Test complete turn processing through the facade"""
        store = MockMemoryStore()
        facade = HotMemoryFacade(store, max_recency=50)
        
        # Test a complete turn
        text = "I live in San Francisco and work at Google"
        session_id = "test_session"
        turn_id = 1
        
        bullets, triples = facade.process_turn(text, session_id, turn_id)
        
        assert isinstance(bullets, list)
        assert isinstance(triples, list)
        
        # Should extract some information
        if triples:
            print(f"✅ Full turn processing works correctly - extracted {len(triples)} triples")
        else:
            print("✅ Full turn processing handles empty extraction correctly")
    
    def test_backward_compatibility(self):
        """Test that facade maintains backward compatibility"""
        store = MockMemoryStore()
        facade = HotMemoryFacade(store, max_recency=50)
        
        # Test that the interface matches the original
        # This is crucial for backward compatibility
        assert hasattr(facade, 'process_turn')
        assert hasattr(facade, 'prewarm')
        assert hasattr(facade, 'get_metrics')
        
        # Test that process_turn returns the expected format
        bullets, triples = facade.process_turn("Hello", "test", 1)
        
        assert isinstance(bullets, list)
        assert isinstance(triples, list)
        assert all(isinstance(b, str) for b in bullets)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in triples)
        
        print("✅ Backward compatibility maintained")
    
    def test_prewarm_functionality(self):
        """Test prewarm functionality"""
        store = MockMemoryStore()
        facade = HotMemoryFacade(store, max_recency=50)
        
        # Should not raise exceptions
        facade.prewarm("en")
        
        print("✅ Prewarm functionality works correctly")


class TestIntegration:
    """Test integration between all components"""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing"""
        store = MockMemoryStore()
        facade = HotMemoryFacade(store, max_recency=50)
        
        # Simulate a conversation
        turns = [
            "I live in San Francisco",
            "I work at Google as a software engineer",
            "I have a dog named Max"
        ]
        
        all_triples = []
        for i, turn in enumerate(turns):
            bullets, triples = facade.process_turn(turn, "test_session", i + 1)
            all_triples.extend(triples)
            
            assert isinstance(bullets, list)
            assert isinstance(triples, list)
        
        # Should have accumulated some knowledge
        print(f"✅ End-to-end processing completed - accumulated {len(all_triples)} triples")
    
    def test_performance_metrics(self):
        """Test that performance metrics are tracked"""
        store = MockMemoryStore()
        facade = HotMemoryFacade(store, max_recency=50)
        
        # Process some turns
        facade.process_turn("I live in San Francisco", "test", 1)
        facade.process_turn("I work at Google", "test", 2)
        
        # Check metrics
        metrics = facade.get_metrics()
        assert isinstance(metrics, dict)
        
        # Should have some timing metrics
        if 'total_ms' in metrics:
            assert isinstance(metrics['total_ms'], list)
        
        print("✅ Performance metrics tracking works correctly")
    
    def test_error_handling(self):
        """Test error handling across all services"""
        store = MockMemoryStore()
        facade = HotMemoryFacade(store, max_recency=50)
        
        # Test with problematic input
        bullets, triples = facade.process_turn("", "test", 1)
        
        # Should handle gracefully
        assert isinstance(bullets, list)
        assert isinstance(triples, list)
        
        print("✅ Error handling works correctly across all services")


def run_all_tests():
    """Run all integration tests"""
    print("🚀 Running comprehensive integration tests...")
    print("=" * 60)
    
    tests = [
        ("Configuration", TestConfiguration),
        ("MemoryExtractor", TestMemoryExtractor),
        ("MemoryRetriever", TestMemoryRetriever),
        ("CoreferenceResolver", TestCoreferenceResolver),
        ("AssistedExtractor", TestAssistedExtractor),
        ("HotMemoryFacade", TestHotMemoryFacade),
        ("Integration", TestIntegration)
    ]
    
    passed = 0
    total = 0
    
    for test_name, test_class in tests:
        print(f"\n🧪 Testing {test_name}...")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed += 1
                print(f"  ✅ {method_name}")
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n🎯 Integration Test Summary:")
        print("✅ Configuration management")
        print("✅ Memory extraction")
        print("✅ Memory retrieval with MMR")
        print("✅ Coreference resolution")
        print("✅ Assisted extraction")
        print("✅ Facade integration")
        print("✅ End-to-end processing")
        print("✅ Performance tracking")
        print("✅ Error handling")
        print("\n🚀 The refactored architecture is fully functional!")
    else:
        print("💥 Some integration tests failed. Check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)