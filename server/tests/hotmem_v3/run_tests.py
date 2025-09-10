#!/usr/bin/env python3
"""
HotMem v3 Test Runner
Comprehensive test suite for the revolutionary self-improving AI system.
"""

import sys
import os
import pytest
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, '.')


def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running HotMem v3 Unit Tests...")
    return pytest.main([
        "tests/hotmem_v3/unit/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def run_integration_tests():
    """Run integration tests"""
    print("🔗 Running HotMem v3 Integration Tests...")
    return pytest.main([
        "tests/hotmem_v3/integration/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def run_performance_tests():
    """Run performance tests"""
    print("⚡ Running HotMem v3 Performance Tests...")
    return pytest.main([
        "tests/hotmem_v3/performance/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def run_all_tests():
    """Run all tests"""
    print("🚀 Running HotMem v3 Complete Test Suite...")
    return pytest.main([
        "tests/hotmem_v3/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])


def test_basic_functionality():
    """Test basic HotMem v3 functionality"""
    print("🔍 Testing HotMem v3 Basic Functionality...")
    
    try:
        # Test imports
        from components.hotmem_v3.core.hotmem_v3 import HotMemV3, HotMemV3Config
        from components.hotmem_v3.core.dual_graph_architecture import DualGraphArchitecture
        from components.hotmem_v3.extraction.streaming_extraction import StreamingExtractor
        from components.hotmem_v3.training.active_learning import ActiveLearningSystem
        
        print("✅ All imports successful")
        
        # Test basic initialization
        config = HotMemV3Config()
        hotmem = HotMemV3(config)
        
        print("✅ HotMem v3 initialization successful")
        
        # Test dual graph architecture
        dual_graph = DualGraphArchitecture()
        
        print("✅ Dual graph architecture initialization successful")
        
        # Test streaming extraction (basic)
        extractor = StreamingExtractor()
        
        print("✅ Streaming extraction initialization successful")
        
        # Test active learning
        active_learning = ActiveLearningSystem()
        
        print("✅ Active learning system initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main test runner"""
    print("🎯 HotMem v3 Test Suite")
    print("=" * 50)
    
    # Test basic functionality first
    if not test_basic_functionality():
        print("❌ Basic functionality tests failed. Exiting.")
        return 1
    
    print("\n")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "unit":
            return run_unit_tests()
        elif test_type == "integration":
            return run_integration_tests()
        elif test_type == "performance":
            return run_performance_tests()
        else:
            print(f"Unknown test type: {test_type}")
            return 1
    else:
        # Run all tests
        return run_all_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)