#!/usr/bin/env python3
"""
Debug speaker recognition device issues with real audio
"""

import asyncio
import sys
import os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

async def test_device_handling():
    """Test device handling with simulated audio"""
    print("=" * 60)
    print("🔧 Testing Audio Intelligence Device Handling")
    print("=" * 60)
    
    try:
        from core.audio import AudioIntelligenceProcessor
        from pipecat.frames.frames import (
            InputAudioRawFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        )
        
        # Check PyTorch device availability
        print("\n[Device Check]")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        # Create processor (will use MPS if available)
        print("\n[Creating Processor]")
        processor = AudioIntelligenceProcessor(
            profile_dir="data/test_speaker_profiles",
            device="mps" if torch.backends.mps.is_available() else "cpu",
            min_utterance_duration_sec=0.5,  # Short for testing
        )
        
        print(f"  Processor device: {processor._device}")
        print(f"  Model device: {next(processor._speaker_model.mods.parameters()).device}")
        
        # Generate synthetic speech-like audio (1.5 seconds)
        print("\n[Generating Test Audio]")
        sample_rate = 16000
        duration = 1.5
        samples = int(sample_rate * duration)
        
        # Generate noise with speech-like envelope
        audio_float = np.random.randn(samples).astype(np.float32) * 0.1
        audio_int16 = (audio_float * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        print(f"  Generated {len(audio_bytes)} bytes ({duration}s)")
        
        # Simulate speech pipeline
        print("\n[Simulating Speech Pipeline]")
        
        # 1. User starts speaking
        print("  1. UserStartedSpeaking...")
        await processor.process_frame(
            UserStartedSpeakingFrame(),
            None
        )
        
        # 2. Feed audio in chunks (simulate real-time)
        chunk_size = 3200  # 100ms chunks
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            frame = InputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=1)
            await processor.process_frame(frame, None)
        
        print(f"  2. Fed {len(audio_bytes)} bytes in chunks")
        
        # 3. User stops speaking (triggers processing)
        print("  3. UserStoppedSpeaking (triggering recognition)...")
        await processor.process_frame(
            UserStoppedSpeakingFrame(),
            None
        )
        
        # Wait for async processing
        print("  4. Waiting for async processing...")
        await asyncio.sleep(2)
        
        # Check results
        print("\n[Results]")
        print(f"  Current speaker: {processor.current_speaker}")
        print(f"  Loaded profiles: {len(processor._speakers)}")
        print(f"  Collecting samples: {processor._collecting_samples}")
        print(f"  Unknown embeddings: {len(processor._unknown_embeddings)}")
        
        # Try one more utterance
        print("\n[Testing Second Utterance]")
        await processor.process_frame(UserStartedSpeakingFrame(), None)
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            frame = InputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=1)
            await processor.process_frame(frame, None)
        
        await processor.process_frame(UserStoppedSpeakingFrame(), None)
        await asyncio.sleep(2)
        
        print("\n[Final State]")
        print(f"  Current speaker: {processor.current_speaker}")
        print(f"  Profiles: {len(processor._speakers)}")
        
        print("\n" + "=" * 60)
        print("✅ Device test completed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Debug device info
        print("\n[Debug Info]")
        try:
            print(f"  Audio tensor device check:")
            audio_array = np.random.randn(16000).astype(np.float32)
            audio_tensor = torch.from_numpy(audio_array)
            print(f"    Created tensor device: {audio_tensor.device}")
            
            if torch.backends.mps.is_available():
                audio_tensor_mps = audio_tensor.to("mps")
                print(f"    Moved to MPS device: {audio_tensor_mps.device}")
        except Exception as e2:
            print(f"    Device test error: {e2}")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(test_device_handling())
    sys.exit(0 if success else 1)
