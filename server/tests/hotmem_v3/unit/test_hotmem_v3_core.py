"""
Test HotMem v3 Core Functionality
Unit tests for the main HotMem v3 class and core components.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path

from components.hotmem_v3.core.hotmem_v3 import HotMemV3, HotMemV3Config


class TestHotMemV3:
    """Test suite for HotMem v3 core functionality"""
    
    def test_hotmem_v3_initialization(self):
        """Test HotMem v3 initialization"""
        config = HotMemV3Config(
            enable_real_time=True,
            confidence_threshold=0.7,
            max_working_memory_entities=100
        )
        
        hotmem = HotMemV3(config)
        
        assert hotmem.config == config
        assert hotmem.initialized == False
        assert hotmem.stats['extractions_processed'] == 0
        assert hotmem.stats['learning_iterations'] == 0
    
    @pytest.mark.asyncio
    async def test_hotmem_v3_async_initialization(self):
        """Test HotMem v3 async initialization"""
        hotmem = HotMemV3()
        
        # Mock the components to avoid actual model loading
        hotmem.streaming_extractor = None
        hotmem.production_integration = None
        
        await hotmem.initialize()
        
        assert hotmem.initialized == True
    
    def test_process_text_basic(self):
        """Test basic text processing"""
        hotmem = HotMemV3()
        
        # Mock initialization
        hotmem.initialized = True
        hotmem.streaming_extractor = None
        hotmem.production_integration = None
        
        result = hotmem.process_text("Hello world", is_final=True)
        
        assert result['success'] == True
        assert result['text'] == "Hello world"
        assert 'entities' in result
        assert 'relations' in result
        assert 'confidence' in result
        assert 'processing_time' in result
    
    def test_process_text_uninitialized(self):
        """Test text processing without initialization"""
        hotmem = HotMemV3()
        
        with pytest.raises(RuntimeError, match="HotMem v3 not initialized"):
            hotmem.process_text("Hello world")
    
    def test_add_user_correction(self):
        """Test user correction functionality"""
        hotmem = HotMemV3()
        hotmem.config.active_learning_enabled = True
        
        # Mock components
        hotmem.active_learning = type('MockActiveLearning', (), {
            'add_user_correction': lambda **kwargs: None
        })()
        hotmem.dual_graph = type('MockDualGraph', (), {
            'add_extraction': lambda **kwargs: None
        })()
        
        original_extraction = {'entities': [], 'relations': []}
        corrected_extraction = {'entities': [{'text': 'Alice'}], 'relations': []}
        
        # Should not raise an exception
        hotmem.add_user_correction(
            original_text="Hello",
            original_extraction=original_extraction,
            corrected_extraction=corrected_extraction
        )
    
    def test_get_knowledge_graph(self):
        """Test knowledge graph retrieval"""
        hotmem = HotMemV3()
        
        # Mock dual graph
        hotmem.dual_graph = type('MockDualGraph', (), {
            'get_combined_graph': lambda: {'entities': [], 'relations': []}
        })()
        
        graph = hotmem.get_knowledge_graph()
        
        assert isinstance(graph, dict)
        assert 'entities' in graph
        assert 'relations' in graph
    
    def test_export_import_knowledge_graph(self):
        """Test knowledge graph export and import"""
        hotmem = HotMemV3()
        
        # Mock dual graph
        hotmem.dual_graph = type('MockDualGraph', (), {
            'add_extraction': lambda **kwargs: None,
            'get_combined_graph': lambda: {
                'entities': [{'text': 'Alice', 'confidence': 0.8}],
                'relations': []
            }
        })()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Export
            hotmem.export_knowledge_graph(filepath)
            
            # Verify file exists and contains data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert 'entities' in data
            assert 'relations' in data
            
            # Import
            hotmem.import_knowledge_graph(filepath)
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_system_stats(self):
        """Test system statistics"""
        hotmem = HotMemV3()
        
        # Mock components
        hotmem.dual_graph = type('MockDualGraph', (), {
            'get_system_stats': lambda: {'working_entities': 5, 'long_term_entities': 10}
        })()
        hotmem.active_learning = type('MockActiveLearning', (), {
            'get_learning_summary': lambda: {'total_corrections': 3}
        })()
        
        stats = hotmem.get_system_stats()
        
        assert isinstance(stats, dict)
        assert 'extractions_processed' in stats
        assert 'learning_iterations' in stats
        assert 'uptime' in stats
        assert 'dual_graph' in stats
        assert 'active_learning' in stats
    
    def test_callbacks(self):
        """Test event callback system"""
        hotmem = HotMemV3()
        
        callback_called = False
        
        def test_callback(data):
            nonlocal callback_called
            callback_called = True
        
        hotmem.add_callback('test_event', test_callback)
        hotmem._trigger_callbacks('test_event', {'test': 'data'})
        
        assert callback_called == True
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality"""
        hotmem = HotMemV3()
        
        # Mock components with cleanup methods
        hotmem.streaming_extractor = type('MockExtractor', (), {
            'cleanup': lambda: None
        })()
        hotmem.production_integration = type('MockIntegration', (), {
            'cleanup': lambda: None
        })()
        
        # Should not raise an exception
        await hotmem.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])