#!/usr/bin/env python3
"""
Quick test to verify frame initialization fix works correctly.
"""

import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from core.audio.audio_intelligence import (
    UnknownSpeakerDetectedFrame,
    StartEnrollmentFrame,
    SpeakerChangedFrame,
    EnrollmentProgressFrame,
    AudioIntelligenceFrame,
)

def test_frame_creation():
    """Test that all custom frames can be created without errors."""
    
    print("Testing frame initialization fixes...\n")
    
    # Test 1: UnknownSpeakerDetectedFrame
    print("1. Testing UnknownSpeakerDetectedFrame...")
    try:
        frame = UnknownSpeakerDetectedFrame(embedding_hash="abc123def456")
        assert hasattr(frame, 'id'), "Frame missing 'id' attribute"
        assert hasattr(frame, 'name'), "Frame missing 'name' attribute"
        assert frame.embedding_hash == "abc123def456"
        print(f"   ✅ SUCCESS: {frame.name} (id={frame.id})")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: StartEnrollmentFrame
    print("2. Testing StartEnrollmentFrame...")
    try:
        frame = StartEnrollmentFrame(speaker_name="Alice")
        assert hasattr(frame, 'id'), "Frame missing 'id' attribute"
        assert hasattr(frame, 'name'), "Frame missing 'name' attribute"
        assert frame.speaker_name == "Alice"
        print(f"   ✅ SUCCESS: {frame.name} (id={frame.id})")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: SpeakerChangedFrame
    print("3. Testing SpeakerChangedFrame...")
    try:
        frame = SpeakerChangedFrame(
            speaker_id="Speaker_2",
            confidence=0.85,
            auto_enrolled=True
        )
        assert hasattr(frame, 'id'), "Frame missing 'id' attribute"
        assert hasattr(frame, 'name'), "Frame missing 'name' attribute"
        assert frame.speaker_id == "Speaker_2"
        print(f"   ✅ SUCCESS: {frame.name} (id={frame.id})")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 4: EnrollmentProgressFrame
    print("4. Testing EnrollmentProgressFrame...")
    try:
        frame = EnrollmentProgressFrame(
            current_sample=2,
            total_samples=3,
            consistency=0.82
        )
        assert hasattr(frame, 'id'), "Frame missing 'id' attribute"
        assert hasattr(frame, 'name'), "Frame missing 'name' attribute"
        assert frame.current_sample == 2
        assert frame.progress_percentage == 66.66666666666666
        print(f"   ✅ SUCCESS: {frame.name} (id={frame.id}, progress={frame.progress_percentage:.1f}%)")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 5: AudioIntelligenceFrame
    print("5. Testing AudioIntelligenceFrame...")
    try:
        frame = AudioIntelligenceFrame(
            speaker_id="Speaker_1",
            speaker_confidence=0.90
        )
        assert hasattr(frame, 'id'), "Frame missing 'id' attribute"
        assert hasattr(frame, 'name'), "Frame missing 'name' attribute"
        assert frame.speaker_id == "Speaker_1"
        print(f"   ✅ SUCCESS: {frame.name} (id={frame.id})")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print("\n🎉 All frame tests passed!")
    return True


if __name__ == "__main__":
    success = test_frame_creation()
    sys.exit(0 if success else 1)
