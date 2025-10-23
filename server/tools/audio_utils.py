"""
Audio processing utilities for TTS services.

Provides shared functions for audio format conversion to reduce code duplication
across different TTS implementations.
"""

import numpy as np
from typing import Union, Any


def convert_to_pcm16(
    audio_data: Union[np.ndarray, Any],
    clip: bool = True
) -> np.ndarray:
    """
    Convert audio to int16 PCM format for Pipecat TTSAudioRawFrame.

    This function handles multiple input formats:
    - NumPy arrays (float32, int16)
    - MLX arrays (converted to NumPy)
    - PyTorch tensors (if needed in future)

    Args:
        audio_data: Audio array in various formats
        clip: Whether to clip values to [-1.0, 1.0] range before conversion.
              Recommended True for PyTorch, False for MLX (already normalized)

    Returns:
        int16 numpy array ready for Pipecat TTSAudioRawFrame

    Examples:
        >>> # From MLX (no clipping needed, already normalized)
        >>> audio_int16 = convert_to_pcm16(mlx_audio, clip=False)

        >>> # From PyTorch (clip for safety)
        >>> audio_int16 = convert_to_pcm16(torch_audio, clip=True)

        >>> # From NumPy float32
        >>> audio_int16 = convert_to_pcm16(numpy_audio)
    """
    # Convert non-numpy arrays to numpy (handles MLX, PyTorch, etc.)
    if not isinstance(audio_data, np.ndarray):
        try:
            # Try to convert to numpy (works for MLX arrays)
            audio_np = np.array(audio_data, copy=False)
        except Exception:
            # Fallback: force copy if zero-copy fails
            audio_np = np.array(audio_data)
    else:
        audio_np = audio_data

    # Already int16? Return as-is
    if audio_np.dtype == np.int16:
        return audio_np

    # Convert float32 to int16
    # Assume float audio is in [-1.0, 1.0] range (standard for audio)
    if clip:
        audio_np = np.clip(audio_np, -1.0, 1.0)

    # Scale to int16 range [-32768, 32767]
    return (audio_np * 32767).astype(np.int16)
