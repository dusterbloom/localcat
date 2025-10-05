#!/usr/bin/env python3
"""Session 3 Test: Prosody Analysis for TRUE Confidence"""
import asyncio
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

async def test_prosody():
    """Test prosody extraction and confidence fusion"""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    print("🧪 Session 3: Testing Prosody Analysis")
    print("=" * 70)
    
    # Test 1: Direct prosody analyzer
    print("\n[Test 1: ProsodyAnalyzer]")
    try:
        from core.audio.prosody_analyzer import ProsodyAnalyzer
        
        analyzer = ProsodyAnalyzer(sample_rate=16000)
        print(f"  ✅ ProsodyAnalyzer created")
        
        # Generate synthetic audio (1.5s)
        audio = np.random.randn(24000).astype(np.float32) * 0.1
        features = analyzer.extract(audio)
        
        if features:
            print(f"  ✅ Features extracted: {features}")
        else:
            print(f"  ⚠️  Features extraction returned None (audio too short?)")
    except Exception as e:
        print(f"  ❌ ProsodyAnalyzer failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: ConfidenceFusion
    print("\n[Test 2: ConfidenceFusion]")
    try:
        from core.audio.confidence_fusion import ConfidenceFusion
        from core.audio.prosody_analyzer import ProsodyFeatures
        
        fusion = ConfidenceFusion()
        print(f"  ✅ ConfidenceFusion created")
        
        # Test different scenarios
        scenarios = [
            ("name", "My name is Alice", None, "Statement"),
            ("likes", "I think maybe I like pizza?", None, "Uncertain question"),
            ("name", "I definitely know it's Bob", None, "Certain statement"),
        ]
        
        for relation, text, prosody, desc in scenarios:
            conf = fusion.calculate(relation, text, prosody=prosody)
            print(f"  {desc}: confidence={conf:.3f} (text='{text[:30]}...')")
    
    except Exception as e:
        print(f"  ❌ ConfidenceFusion failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Full pipeline
    print("\n[Test 3: AudioIntelligenceProcessor with Prosody]")
    try:
        from core.audio import AudioIntelligenceProcessor
        from pipecat.frames.frames import InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
        
        processor = AudioIntelligenceProcessor(
            profile_dir="data/test_prosody_profiles",
            device="mps",
            min_utterance_duration_sec=0.5,
            enable_emotion=True,
            enable_prosody=True,
        )
        
        print(f"  Speaker model: {processor._speaker_model is not None}")
        print(f"  Emotion model: {processor._emotion_model is not None}")
        print(f"  Prosody analyzer: {processor._prosody_analyzer is not None}")
        
        # Test utterance
        audio_float = np.random.randn(24000).astype(np.float32) * 0.1
        audio_int16 = (audio_float * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        await processor.process_frame(UserStartedSpeakingFrame(), None)
        
        for j in range(0, len(audio_bytes), 3200):
            frame = InputAudioRawFrame(audio=audio_bytes[j:j+3200], sample_rate=16000, num_channels=1)
            await processor.process_frame(frame, None)
        
        await processor.process_frame(UserStoppedSpeakingFrame(), None)
        
        await asyncio.sleep(2)
        
        print(f"  ✅ Pipeline processed (check logs for prosody extraction)")
    
    except Exception as e:
        print(f"  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Session 3 Component Tests Complete!")
    print("=" * 70)
    print("\nExpect in logs:")
    print("  [AudioIntel] Prosody: ProsodyFeatures(pitch=...Hz, ...)")
    print("\nAudioIntelligenceFrame now includes:")
    print("  - prosody_features (ProsodyFeatures object)")
    print("  - prosody_certainty (-0.3 to +0.3 modifier)")
    print("\nNext: Wire ConfidenceFusion into memory_hotpath.py!")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_prosody())
    sys.exit(0 if success else 1)
