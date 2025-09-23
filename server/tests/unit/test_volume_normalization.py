#!/usr/bin/env python
"""
Test volume normalization for Parakeet Streaming STT
"""

import numpy as np
import sys
import os

# Ensure server root is importable
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for p in (_SERVER_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_volume_normalization():
    """Test that volume normalization works correctly"""
    from core.stt.parakeet_streaming import ParakeetStreamingSTT

    # Create STT instance (model won't load in test environment, but we can test the method)
    stt = ParakeetStreamingSTT.__new__(ParakeetStreamingSTT)  # Create without __init__

    # Test 1: Very quiet audio (should be amplified)
    quiet_audio = np.random.normal(0, 0.01, 16000).astype(np.float32)  # Very quiet
    original_rms = np.sqrt(np.mean(quiet_audio**2))

    normalized = stt._normalize_audio(quiet_audio)
    normalized_rms = np.sqrt(np.mean(normalized**2))

    print(".4f")
    print(".4f")
    assert normalized_rms > original_rms, "Quiet audio should be amplified"
    assert normalized_rms <= 0.15, f"RMS too high: {normalized_rms}"  # Should not exceed target + some tolerance

    # Test 2: Already optimal audio (should not change much)
    optimal_audio = np.random.normal(0, 0.08, 16000).astype(np.float32)  # Close to target RMS
    original_rms_opt = np.sqrt(np.mean(optimal_audio**2))

    normalized_opt = stt._normalize_audio(optimal_audio)
    normalized_rms_opt = np.sqrt(np.mean(normalized_opt**2))

    print(".4f")
    print(".4f")
    # Should not change dramatically
    assert abs(normalized_rms_opt - original_rms_opt) < 0.05, "Optimal audio should not change much"

    # Test 3: Silent audio (should remain unchanged)
    silent_audio = np.zeros(16000, dtype=np.float32)
    normalized_silent = stt._normalize_audio(silent_audio)

    assert np.allclose(silent_audio, normalized_silent), "Silent audio should remain unchanged"

    # Test 4: Very loud audio (should be attenuated)
    loud_audio = np.random.normal(0, 0.5, 16000).astype(np.float32)  # Very loud
    original_rms_loud = np.sqrt(np.mean(loud_audio**2))

    normalized_loud = stt._normalize_audio(loud_audio)
    normalized_rms_loud = np.sqrt(np.mean(normalized_loud**2))

    print(".4f")
    print(".4f")
    assert normalized_rms_loud < original_rms_loud, "Loud audio should be attenuated"
    assert np.max(np.abs(normalized_loud)) <= 1.0, "Audio should not clip"

    print("✅ All volume normalization tests passed!")


if __name__ == "__main__":
    test_volume_normalization()