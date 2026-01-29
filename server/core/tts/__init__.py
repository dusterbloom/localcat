"""
Text-to-Speech (TTS) services.

Production implementations:
- kokoro_professional: Default TTS with artifact-free audio processing
- kokoro_mlx: Backup TTS implementation using MLX
- supertonic: Lightning-fast on-device synthesis (66M params)
- qwen3: Qwen3 TTS with emotional control and voice cloning (1.7B model)
"""