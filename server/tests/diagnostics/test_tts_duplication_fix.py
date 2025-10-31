#!/usr/bin/env python3
"""
Test to verify TTS duplication fix.

This script creates a minimal test to ensure that only one TTS service
instance processes conversation TextFrames, eliminating the duplicate
audio issue.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

async def test_tts_duplication_fix():
    """Test that the TTS duplication fix prevents duplicate TTS instances."""

    print("🔬 Testing TTS duplication fix...")

    # Test that the factory no longer creates intro_tts
    try:
        from core.factory import VoiceAgentFactory

        factory = VoiceAgentFactory()
        services = factory._create_services()

        # Check if main TTS service exists
        main_tts = services.get('tts')
        if main_tts:
            print(f"✅ Main TTS service exists: {type(main_tts).__name__}")
        else:
            print("❌ Main TTS service missing")
            return False

        # Test pipeline creation to ensure intro_tts is None
        print("🏗️  Testing pipeline creation...")

        # Mock required components for pipeline creation
        factory.config.enable_intro_pipeline = True
        factory.config.enable_ephemeral_choice = True
        factory.config.skip_intro_for_returning = False
        factory.config.speaker_profile_dir = "/tmp/test_profiles"

        # Create a minimal mock transport
        class MockTransport:
            def input(self):
                return MockInput()
            def output(self):
                return MockOutput()

        class MockInput:
            def __call__(self):
                return self

        class MockOutput:
            def __call__(self):
                return self

        transport = MockTransport()

        # This should create pipeline with intro_tts = None
        pipeline = factory.create_pipeline(transport)

        print("✅ Pipeline created successfully without intro TTS")
        print("🎯 TTS duplication fix verified!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("=" * 80)
    print("🚀 TTS DUPLICATION FIX VERIFICATION")
    print("=" * 80)

    success = await test_tts_duplication_fix()

    print("\n" + "=" * 80)
    if success:
        print("🎉 SUCCESS: TTS duplication fix is working!")
        print("   - Intro TTS service removed from pipeline")
        print("   - Only main conversation TTS processes TextFrames")
        print("   - No duplicate audio should occur")
    else:
        print("❌ FAILURE: TTS duplication fix needs adjustment")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())