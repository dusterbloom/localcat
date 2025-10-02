#!/usr/bin/env python3
"""Test full speaker enrollment (3 utterances)"""
import asyncio
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

async def test_enrollment():
    """Test auto-enrollment with 3 utterances"""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    from core.audio import AudioIntelligenceProcessor
    from pipecat.frames.frames import InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    
    print("🧪 Testing Speaker Auto-Enrollment (3 utterances)")
    print("=" * 60)
    
    processor = AudioIntelligenceProcessor(
        profile_dir="data/test_speaker_profiles_3",
        device="mps",
        min_utterance_duration_sec=0.5,
        auto_enroll_utterances=3,
    )
    
    # Generate 3 similar utterances
    for i in range(1, 4):
        print(f"\n[Utterance {i}/3]")
        
        # Generate audio
        audio_float = np.random.randn(24000).astype(np.float32) * 0.1  # 1.5s
        audio_int16 = (audio_float * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Simulate pipeline
        await processor.process_frame(UserStartedSpeakingFrame(), None)
        
        for j in range(0, len(audio_bytes), 3200):
            frame = InputAudioRawFrame(audio=audio_bytes[j:j+3200], sample_rate=16000, num_channels=1)
            await processor.process_frame(frame, None)
        
        await processor.process_frame(UserStoppedSpeakingFrame(), None)
        
        # Wait for processing
        await asyncio.sleep(1.5)
        
        print(f"  Current speaker: {processor.current_speaker}")
        print(f"  Profiles: {len(processor._speakers)}")
        print(f"  Samples collected: {len(processor._unknown_embeddings)}")
    
    print("\n" + "=" * 60)
    print(f"✅ Final: Speaker = {processor.current_speaker}, Profiles = {len(processor._speakers)}")
    
    if processor.current_speaker and processor.current_speaker != "unknown":
        print(f"🎉 SUCCESS! Auto-enrolled as {processor.current_speaker}")
        return True
    else:
        print(f"❌ FAILED: Still unknown or not enrolled")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enrollment())
    sys.exit(0 if success else 1)
