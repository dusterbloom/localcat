#!/usr/bin/env python3
"""
Standalone HotMem v3 Test Suite
Tests the reorganized HotMem v3 components without external dependencies.
"""

import sys
import os
import asyncio
import tempfile
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, '.')


def test_imports():
    """Test all HotMem v3 imports"""
    print("🔍 Testing HotMem v3 Imports...")
    
    try:
        # Test core imports
        from components.hotmem_v3.core.hotmem_v3 import HotMemV3, HotMemV3Config
        from components.hotmem_v3.core.dual_graph_architecture import DualGraphArchitecture
        from components.hotmem_v3.core.interfaces import HotMemV3Interface
        from components.hotmem_v3.core.inference import HotMemInference, InferenceResult
        
        print("✅ Core imports successful")
        
        # Test extraction imports
        from components.hotmem_v3.extraction.streaming_extraction import StreamingExtractor, StreamingChunk
        
        print("✅ Extraction imports successful")
        
        # Test training imports
        from components.hotmem_v3.training.active_learning import ActiveLearningSystem
        from components.hotmem_v3.training.dataset_preparation import HotMemDatasetPreparer
        from components.hotmem_v3.training.model_optimizer import HotMemModelOptimizer
        from components.hotmem_v3.training.training_pipeline import HotMemTrainingPipeline
        
        print("✅ Training imports successful")
        
        # Test integration imports
        from components.hotmem_v3.integration.production_integration import HotMemIntegration
        from components.hotmem_v3.integration.end_to_end_validation import HotMemValidator
        
        print("✅ Integration imports successful")
        
        # Test augmentation imports
        from components.hotmem_v3.augmentation.streaming_augmentation import StreamingAugmentor
        
        print("✅ Augmentation imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic HotMem v3 functionality"""
    print("🔍 Testing HotMem v3 Basic Functionality...")
    
    try:
        # Test HotMem v3 initialization
        from components.hotmem_v3.core.hotmem_v3 import HotMemV3, HotMemV3Config
        
        config = HotMemV3Config(
            enable_real_time=True,
            confidence_threshold=0.7,
            max_working_memory_entities=100
        )
        
        hotmem = HotMemV3(config)
        
        assert hotmem.config == config
        assert hotmem.initialized == False
        assert hotmem.stats['extractions_processed'] == 0
        
        print("✅ HotMem v3 initialization successful")
        
        # Test dual graph architecture
        from components.hotmem_v3.core.dual_graph_architecture import DualGraphArchitecture
        
        dual_graph = DualGraphArchitecture()
        
        print("✅ Dual graph architecture initialization successful")
        
        # Test streaming extraction
        from components.hotmem_v3.extraction.streaming_extraction import StreamingExtractor
        
        extractor = StreamingExtractor()
        
        print("✅ Streaming extraction initialization successful")
        
        # Test active learning
        from components.hotmem_v3.training.active_learning import ActiveLearningSystem
        
        active_learning = ActiveLearningSystem()
        
        print("✅ Active learning system initialization successful")
        
        # Test inference
        from components.hotmem_v3.core.inference import HotMemInference
        
        inference = HotMemInference()
        
        print("✅ Inference system initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_text_processing():
    """Test text processing functionality"""
    print("🔍 Testing Text Processing...")
    
    try:
        from components.hotmem_v3.core.hotmem_v3 import HotMemV3
        
        hotmem = HotMemV3()
        hotmem.initialized = True
        
        # Test basic text processing
        result = hotmem.process_text("Hello world", is_final=True)
        
        assert result['success'] == True
        assert result['text'] == "Hello world"
        assert 'entities' in result
        assert 'relations' in result
        assert 'confidence' in result
        assert 'processing_time' in result
        
        print("✅ Text processing successful")
        
        # Test knowledge graph retrieval
        graph = hotmem.get_knowledge_graph()
        
        assert isinstance(graph, dict)
        assert 'entities' in graph
        assert 'relations' in graph
        
        print("✅ Knowledge graph retrieval successful")
        
        # Test system stats
        stats = hotmem.get_system_stats()
        
        assert isinstance(stats, dict)
        assert 'extractions_processed' in stats
        assert 'uptime' in stats
        
        print("✅ System stats retrieval successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False


def test_knowledge_graph_export_import():
    """Test knowledge graph export and import"""
    print("🔍 Testing Knowledge Graph Export/Import...")
    
    try:
        from components.hotmem_v3.core.hotmem_v3 import HotMemV3
        
        hotmem = HotMemV3()
        hotmem.initialized = True
        
        # Add some data
        hotmem.process_text("Alice lives in London", is_final=True)
        hotmem.process_text("Bob works at Google", is_final=True)
        
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
            
            print("✅ Knowledge graph export/import successful")
            
        finally:
            Path(filepath).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge graph export/import test failed: {e}")
        return False


def test_async_functionality():
    """Test async functionality"""
    print("🔍 Testing Async Functionality...")
    
    try:
        from components.hotmem_v3.core.hotmem_v3 import HotMemV3
        
        hotmem = HotMemV3()
        
        # Test async initialization
        async def test_init():
            await hotmem.initialize()
            assert hotmem.initialized == True
        
        asyncio.run(test_init())
        
        print("✅ Async initialization successful")
        
        # Test async cleanup
        async def test_cleanup():
            await hotmem.cleanup()
        
        asyncio.run(test_cleanup())
        
        print("✅ Async cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False


def main():
    """Main test runner"""
    print("🎯 HotMem v3 Standalone Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Text Processing", test_text_processing),
        ("Knowledge Graph Export/Import", test_knowledge_graph_export_import),
        ("Async Functionality", test_async_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HotMem v3 reorganization is working correctly!")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)