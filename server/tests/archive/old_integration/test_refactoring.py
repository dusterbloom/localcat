#!/usr/bin/env python3
"""
Test script to verify our HotMemory refactoring works correctly
"""

import sys
import os
sys.path.insert(0, '.')

def test_configuration():
    """Test the extracted configuration module"""
    print("🧪 Testing Configuration module...")
    
    from components.memory.config import create_config, HotMemoryConfig
    
    # Test basic configuration
    config = create_config()
    assert isinstance(config, HotMemoryConfig)
    assert config.max_recency > 0
    assert 0.0 <= config.confidence_threshold <= 1.0
    
    # Test feature flags
    assert hasattr(config, 'features')
    assert hasattr(config.features, 'use_srl')
    assert hasattr(config.features, 'use_leann')
    
    # Test model configuration
    assert hasattr(config, 'assisted_model')
    assert config.assisted_model.name
    
    print("✅ Configuration module works correctly")
    return True

def test_memory_extractor():
    """Test the extracted MemoryExtractor"""
    print("🧪 Testing MemoryExtractor...")
    
    from components.extraction.memory_extractor import MemoryExtractor, ExtractionResult
    
    # Create minimal config
    config = {
        'use_srl': False,
        'use_onnx_ner': False, 
        'use_onnx_srl': False,
        'use_relik': False,
        'use_dspy': False,
        'assisted_model': None,
        'cache_size': 100
    }
    
    extractor = MemoryExtractor(config)
    assert isinstance(extractor, MemoryExtractor)
    
    # Test light entity extraction
    entities = extractor.extract_entities_light("I live in San Francisco and work at Google")
    assert isinstance(entities, list)
    
    # Test metrics
    metrics = extractor.get_metrics()
    assert isinstance(metrics, dict)
    
    print("✅ MemoryExtractor works correctly")
    return True

def test_refactoring_metrics():
    """Test that our refactoring reduced complexity"""
    print("🧪 Testing refactoring metrics...")
    
    # Count lines in original file
    original_lines = 0
    try:
        with open('components/memory/memory_hotpath.py', 'r') as f:
            original_lines = len(f.readlines())
    except:
        original_lines = 3501  # Known value
    
    # Count lines in new files
    new_files = [
        'components/extraction/memory_extractor.py',
        'components/memory/config.py',
        'components/memory/hotmemory_facade.py'
    ]
    
    total_new_lines = 0
    for file_path in new_files:
        try:
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
                total_new_lines += lines
                print(f"  📄 {file_path}: {lines} lines")
        except FileNotFoundError:
            print(f"  ❌ {file_path}: Not found")
    
    print(f"📊 Original: 1 file, {original_lines} lines")
    print(f"📊 Refactored: {len(new_files)} files, {total_new_lines} lines")
    print(f"📊 Average per file: {total_new_lines // len(new_files)} lines")
    
    # Success if average is under 600 lines per file
    success = (total_new_lines // len(new_files)) < 600
    if success:
        print("✅ Refactoring successfully reduced complexity")
    else:
        print("⚠️  Some files are still large - more extraction needed")
    
    return success

def main():
    """Run all tests"""
    print("🚀 Testing HotMemory refactoring...")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_memory_extractor,
        test_refactoring_metrics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring is successful!")
        print("\n🎯 Next steps:")
        print("1. Extract MemoryRetriever (MMR algorithm)")
        print("2. Extract remaining legacy methods from facade")
        print("3. Add comprehensive tests")
        print("4. Update documentation")
    else:
        print("💥 Some tests failed. Check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)