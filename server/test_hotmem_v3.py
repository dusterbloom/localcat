#!/usr/bin/env python3
"""
HotMem v3 Quick Test Script
Verify that all components are working correctly after deployment
"""

import sys
import os
import json
import time
from pathlib import Path

def test_imports():
    """Test that all HotMem v3 modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test HotMem v3 imports
        from hotmem_v3_streaming_extraction import StreamingExtractor, StreamingChunk
        from hotmem_v3_production_integration import HotMemIntegration, LocalCatHotMemIntegration
        from hotmem_v3_active_learning import ActiveLearningSystem
        from hotmem_v3_dual_graph_architecture import DualGraphArchitecture
        from hotmem_v3_model_optimizer import HotMemModelOptimizer
        from hotmem_v3_end_to_end_validation import HotMemValidator
        
        print("✅ All HotMem v3 modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_loading():
    """Test that the model can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        from hotmem_v3_production_integration import HotMemIntegration
        
        # Try to find model in common locations
        model_paths = [
            "~/localcat/models/hotmem_v3/hotmem_v3_int4",
            "./hotmem_v3_int4",
            "../models/hotmem_v3/hotmem_v3_int4"
        ]
        
        for model_path in model_paths:
            try:
                expanded_path = os.path.expanduser(model_path)
                if Path(expanded_path).exists():
                    integration = HotMemIntegration(model_path=expanded_path)
                    print(f"✅ Model loaded from: {model_path}")
                    return integration
            except Exception as e:
                print(f"⚠️ Failed to load from {model_path}: {e}")
                continue
        
        print("❌ No model found. Please ensure model is downloaded and extracted.")
        return None
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return None

def test_streaming_extraction():
    """Test streaming extraction functionality"""
    print("\n🔍 Testing streaming extraction...")
    
    try:
        from hotmem_v3_streaming_extraction import StreamingExtractor, StreamingChunk
        
        # Create extractor (without model for basic test)
        extractor = StreamingExtractor(model_path=None)
        
        # Test chunk processing
        chunk = StreamingChunk(
            text="Steve Jobs founded Apple",
            timestamp=time.time(),
            chunk_id=1,
            is_final=True
        )
        
        result = extractor.process_chunk(chunk)
        
        print(f"✅ Streaming extraction works")
        print(f"  - Entities found: {len(result.entities)}")
        print(f"  - Relations found: {len(result.relations)}")
        print(f"  - Processing time: {result.processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming extraction test failed: {e}")
        return False

def test_dual_graph():
    """Test dual graph architecture"""
    print("\n🔍 Testing dual graph architecture...")
    
    try:
        from hotmem_v3_dual_graph_architecture import DualGraphArchitecture
        
        # Create dual graph system
        dual_graph = DualGraphArchitecture()
        
        # Test adding extractions
        dual_graph.add_extraction(
            text="Sarah works at Google as a software engineer",
            entities=["Sarah", "Google", "software engineer"],
            relations=[
                {"subject": "Sarah", "predicate": "works_at", "object": "Google"},
                {"subject": "Sarah", "predicate": "is", "object": "software engineer"}
            ],
            confidence=0.8,
            extraction_type="conversation"
        )
        
        # Test querying
        results = dual_graph.query_knowledge("Sarah", "entities")
        
        print(f"✅ Dual graph architecture works")
        print(f"  - Working memory entities: {dual_graph.get_system_stats()['working_memory']['entity_count']}")
        print(f"  - Query results: {len(results['combined'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dual graph test failed: {e}")
        return False

def test_active_learning():
    """Test active learning system"""
    print("\n🔍 Testing active learning...")
    
    try:
        from hotmem_v3_active_learning import ActiveLearningSystem
        
        # Create active learning system
        active_learning = ActiveLearningSystem()
        
        # Test adding extraction results
        active_learning.add_extraction_result(
            text="Apple is in California",
            extraction={"entities": ["Apple", "California"], "relations": []},
            confidence=0.9,
            is_correct=True
        )
        
        # Test adding user correction
        active_learning.add_user_correction(
            original_text="Microsoft in Seattle",
            original_extraction={"entities": ["Microsoft"], "relations": []},
            corrected_extraction={"entities": ["Microsoft", "Seattle"], "relations": []},
            confidence=0.5,
            error_type="entity_boundary_error"
        )
        
        # Get learning summary
        summary = active_learning.get_learning_summary()
        
        print(f"✅ Active learning system works")
        print(f"  - Total corrections: {summary['total_corrections']}")
        print(f"  - Significant patterns: {summary['significant_patterns']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Active learning test failed: {e}")
        return False

def test_production_integration():
    """Test production integration"""
    print("\n🔍 Testing production integration...")
    
    try:
        from hotmem_v3_production_integration import HotMemIntegration
        
        # Test integration without model
        integration = HotMemIntegration(model_path=None)
        
        # Test processing transcription
        integration.process_transcription("Hi, I'm John and I work at Microsoft", is_final=True)
        
        # Get knowledge graph
        graph = integration.get_knowledge_graph()
        
        print(f"✅ Production integration works")
        print(f"  - Knowledge graph entities: {len(graph['entities'])}")
        print(f"  - Knowledge graph relations: {len(graph['relations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Production integration test failed: {e}")
        return False

def run_validation():
    """Run validation suite"""
    print("\n🔍 Running validation suite...")
    
    try:
        from hotmem_v3_end_to_end_validation import HotMemValidator
        
        # Create validator (without model for basic tests)
        validator = HotMemValidator(model_path=None)
        
        # Run integration tests
        results = validator.run_integration_tests()
        
        # Calculate pass rate
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0
        
        print(f"✅ Validation suite completed")
        print(f"  - Tests passed: {passed}/{total} ({pass_rate:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation suite failed: {e}")
        return False

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb >= 2:
            print(f"✅ Available memory: {available_gb:.1f}GB")
        else:
            print(f"⚠️ Low memory: {available_gb:.1f}GB (recommended: 2GB+)")
        
    except ImportError:
        print("⚠️ psutil not available - cannot check memory")
    
    # Check for Apple Silicon
    try:
        import platform
        system_info = platform.platform()
        
        if "arm" in system_info.lower():
            print(f"✅ Apple Silicon detected: {system_info}")
        else:
            print(f"⚠️ Non-Apple Silicon: {system_info}")
            
    except Exception as e:
        print(f"⚠️ Could not check system info: {e}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 HotMem v3 Quick Test Suite")
    print("=" * 50)
    
    # Track results
    results = {
        "system_requirements": False,
        "imports": False,
        "model_loading": False,
        "streaming_extraction": False,
        "dual_graph": False,
        "active_learning": False,
        "production_integration": False,
        "validation": False
    }
    
    # Run tests
    results["system_requirements"] = check_system_requirements()
    results["imports"] = test_imports()
    
    if results["imports"]:
        # Only run these tests if imports work
        results["model_loading"] = test_model_loading() is not None
        results["streaming_extraction"] = test_streaming_extraction()
        results["dual_graph"] = test_dual_graph()
        results["active_learning"] = test_active_learning()
        results["production_integration"] = test_production_integration()
        results["validation"] = run_validation()
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! HotMem v3 is ready for deployment!")
    elif passed >= total * 0.7:
        print("✅ Most tests passed. HotMem v3 should work, but check failed tests.")
    else:
        print("⚠️ Many tests failed. Please check the requirements and dependencies.")
    
    # Next steps
    print(f"\n{'='*50}")
    print("📋 Next Steps:")
    
    if not results["system_requirements"]:
        print("1. Fix system requirements issues")
    
    if not results["imports"]:
        print("2. Install missing dependencies from requirements.txt")
    
    if not results["model_loading"]:
        print("3. Download and extract the trained model from Colab")
    
    if results["imports"] and not results["model_loading"]:
        print("4. Run: python hotmem_v3_model_optimizer.py --model_path /path/to/model")
    
    if passed >= total * 0.8:
        print("5. Start using HotMem v3: python bot.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)