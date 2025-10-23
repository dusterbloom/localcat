"""
Configuration constants for Kokoro TTS services.

This module centralizes magic numbers and configuration values used across
different Kokoro TTS implementations (MLX, PyTorch, Professional).

Goals:
- Document the "why" behind each value
- Make tuning easier by having a single source of truth
- Improve code readability by using named constants instead of literals
"""

# ============================================================================
# Thread Pool Sizing
# ============================================================================

# MLX-based TTS requires serialization due to Metal framework constraints
# Single worker + MLX_GLOBAL_LOCK prevents concurrent Metal access
MLX_EXECUTOR_WORKERS = 1

# PyTorch-based TTS can handle limited concurrency
# Two workers provides modest parallelism without resource contention
PYTORCH_EXECUTOR_WORKERS = 2


# ============================================================================
# Text Chunking (for streaming TTS)
# ============================================================================

# Minimum characters per chunk
# Too short: Unnatural pauses between words
# Optimized for natural phrasing and sentence boundaries
CHUNK_MIN_LENGTH = 50

# Maximum characters per chunk
# Too long: Breath artifacts and unnatural pacing
# Optimized for Kokoro's voice quality sweet spot
CHUNK_MAX_LENGTH = 120

# Rationale for 50-120 range:
# - Balances low latency (shorter chunks) with voice quality (natural phrasing)
# - Empirically tested to minimize breath artifacts
# - Allows most sentences to be spoken without mid-sentence breaks


# ============================================================================
# Audio Format
# ============================================================================

# Standard sample rate for Kokoro TTS
# 24kHz provides good voice quality without excessive bandwidth
SAMPLE_RATE = 24000

# Mono audio (single channel)
# Voice synthesis doesn't benefit from stereo
CHANNELS = 1


# ============================================================================
# Thread Name Prefixes (for debugging/profiling)
# ============================================================================

THREAD_PREFIX_MLX = "mlx-kokoro"
THREAD_PREFIX_PYTORCH = "kokoro-pytorch"
THREAD_PREFIX_PROFESSIONAL = "kokoro-pro"
