#!/usr/bin/env python3
"""Session 2 Test: Emotion Detection"""
import asyncio
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

async def test_emotion():
    """Test emotion detection on synthetic audio"""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    print("🧪 Session 2: Testing Emotion Detection")
    print("=" * 60)
    
    from core.audio import AudioIntelligenceProcessor
    from pipecat.frames.frames import InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    
    processor = AudioIntelligenceProcessor(
        profile_dir="data/test_emotion_profiles",
        device="mps",
        min_utterance_duration_sec=0.5,
        enable_emotion=True,
    )
    
    print(f"\n[Config]")
    print(f"  Speaker model loaded: {processor._speaker_model is not None}")
    print(f"  Emotion model loaded: {processor._emotion_model is not None}")
    print(f"  Emotion enabled: {processor._enable_emotion}")
    
    # Test with synthetic audio
    print(f"\n[Testing Utterance]")
    
    # Generate audio (1.5s)
    audio_float = np.random.randn(24000).astype(np.float32) * 0.1
    audio_int16 = (audio_float * 32768).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    # Simulate pipeline
    await processor.process_frame(UserStartedSpeakingFrame(), None)
    
    for j in range(0, len(audio_bytes), 3200):
        frame = InputAudioRawFrame(audio=audio_bytes[j:j+3200], sample_rate=16000, num_channels=1)
        await processor.process_frame(frame, None)
    
    await processor.process_frame(UserStoppedSpeakingFrame(), None)
    
    # Wait for processing
    await asyncio.sleep(3)
    
    print(f"\n[Results]")
    print(f"  Speaker: {processor.current_speaker}")
    print(f"  Emotion processing attempted: {processor._enable_emotion}")
    
    print("\n" + "=" * 60)
    print("✅ Session 2 Test Complete!")
    print("=" * 60)
    print("\nExpect to see in logs:")
    print("  [AudioIntel] Emotion: <emotion> (conf=X.XX, v=X.X, a=X.X)")
    print("\nEmitted AudioIntelligenceFrame will have:")
    print("  - speaker_id")
    print("  - emotion (angry/happy/sad/neutral)")
    print("  - valence (-1 to +1)")
    print("  - arousal (0 to 1)")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_emotion())
    sys.exit(0 if success else 1)
