#!/usr/bin/env python3
"""
Session 1 Test: Speaker Recognition MVP
Quick test to verify SpeechBrain integration works
"""

import asyncio
import sys
import os
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

async def test_speaker_recognition():
    """Test basic speaker recognition functionality"""
    print("=" * 60)
    print("🧪 Session 1: Testing Audio Intelligence (Speaker Recognition)")
    print("=" * 60)
    
    try:
        # Test 1: Import check
        print("\n[Test 1] Checking imports...")
        from core.audio import AudioIntelligenceProcessor
        print("✅ AudioIntelligenceProcessor imported successfully")
        
        # Test 2: SpeechBrain availability
        print("\n[Test 2] Checking SpeechBrain...")
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            print("✅ SpeechBrain available")
        except ImportError as e:
            print(f"❌ SpeechBrain not available: {e}")
            print("   Install with: pip install speechbrain")
            return False
        
        # Test 3: Create processor
        print("\n[Test 3] Creating AudioIntelligenceProcessor...")
        processor = AudioIntelligenceProcessor(
            profile_dir="data/test_speaker_profiles",
            similarity_threshold=0.75,
            min_utterance_duration_sec=1.0,
            device="cpu",  # Use CPU for testing
        )
        print(f"✅ Processor created: {processor}")
        print(f"   Current speaker: {processor.current_speaker}")
        print(f"   Loaded profiles: {len(processor._speakers)}")
        
        # Test 4: Check model loading
        print("\n[Test 4] Checking SpeechBrain model...")
        if hasattr(processor, '_speaker_model'):
            print("✅ SpeechBrain ECAPA-TDNN model loaded")
            print(f"   Device: {processor._device}")
        else:
            print("❌ Speaker model not loaded")
            return False
        
        # Test 5: Factory integration
        print("\n[Test 5] Testing factory integration...")
        from core.factory import VoiceAgentFactory
        from config import VoiceAgentConfig
        
        config = VoiceAgentConfig()
        factory = VoiceAgentFactory(config)
        
        audio_intel = factory.create_audio_intelligence_processor()
        if audio_intel:
            print("✅ Factory can create AudioIntelligenceProcessor")
        else:
            print("❌ Factory failed to create processor")
            return False
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Session 1 MVP Ready!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install speechbrain")
        print("2. Wire into bot.py parallel pipeline")
        print("3. Test with real audio")
        print("4. Session 2: Add emotion detection")
        
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_speaker_recognition())
    sys.exit(0 if success else 1)
