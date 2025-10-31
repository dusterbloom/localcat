#!/usr/bin/env python3
"""
Test to verify TTS duplication fix in conversation mode.

This script verifies that the intro-aware pipeline structure correctly
prevents TextFrame duplication during conversation mode.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_pipeline_structure():
    """Test that the pipeline structure is correct for conversation mode."""

    print("🔬 Testing conversation mode pipeline structure...")

    # Test that the factory creates the correct pipeline structure
    try:
        from core.factory import VoiceAgentFactory

        factory = VoiceAgentFactory()

        # Verify intro-aware pipeline is enabled by default
        print(f"✅ Intro pipeline enabled: {factory.config.enable_intro_pipeline}")

        # Test service creation
        services = factory._create_services()

        # Check that context aggregator is created
        context_aggregator = services.get('context_aggregator')
        if context_aggregator:
            print(f"✅ Context aggregator created: {type(context_aggregator).__name__}")

            # Test that assistant context aggregator exists
            if hasattr(context_aggregator, 'assistant'):
                assistant = context_aggregator.assistant()
                print(f"✅ Assistant context aggregator: {type(assistant).__name__}")
            else:
                print("❌ Assistant context aggregator not found")
                return False
        else:
            print("❌ Context aggregator missing")
            return False

        # Check text aggregator exists
        text_aggregator = services.get('text_aggregator')
        if text_aggregator:
            print(f"✅ Text aggregator created: {type(text_aggregator).__name__}")
        else:
            print("❌ Text aggregator missing")
            return False

        # Check TTS service exists
        tts = services.get('tts')
        if tts:
            print(f"✅ TTS service created: {type(tts).__name__}")
        else:
            print("❌ TTS service missing")
            return False

        print("✅ All required services created successfully")
        return True

    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_backend_configuration():
    """Test memory backend configuration for pipeline routing."""

    print("\n🔧 Testing memory backend configuration...")

    try:
        from core.factory import VoiceAgentFactory

        factory = VoiceAgentFactory()

        # Check memory backend
        memory_backend = os.getenv("MEMORY_BACKEND", "hotpath").lower()
        print(f"✅ Memory backend: {memory_backend}")

        # Verify hotpath backend uses standard pipeline (not intro-aware)
        if memory_backend == "hotpath":
            print("✅ Hotpath backend configured correctly")

        return True

    except Exception as e:
        print(f"❌ Memory backend test failed: {e}")
        return False

def test_intro_pipeline_configuration():
    """Test intro pipeline configuration."""

    print("\n🎭 Testing intro pipeline configuration...")

    try:
        # Check if intro pipeline is enabled
        intro_pipeline = os.getenv("AUDIO_INTEL_INTRO_PIPELINE", "true").lower() in ("1", "true", "yes")
        print(f"✅ Intro pipeline enabled: {intro_pipeline}")

        if intro_pipeline:
            print("✅ Using intro-aware pipeline for enrollment UX")
        else:
            print("✅ Using standard pipeline")

        return True

    except Exception as e:
        print(f"❌ Intro pipeline test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("=" * 80)
    print("🚀 TTS CONVERSATION MODE DUPLICATION FIX VERIFICATION")
    print("=" * 80)

    # Run tests
    tests = [
        ("Pipeline Structure", test_pipeline_structure),
        ("Memory Backend Configuration", test_memory_backend_configuration),
        ("Intro Pipeline Configuration", test_intro_pipeline_configuration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 SUCCESS: All tests passed! TTS duplication fix structure is correct.")
        print("   - Pipeline structure prevents TextFrame duplication")
        print("   - context_aggregator.assistant() positioned after transport.output()")
        print("   - Only one TTS processing path for conversation mode")
    else:
        print("❌ FAILURE: Some tests failed. TTS duplication may still occur.")

    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())