Factories Architecture

Overview
- ServiceFactory is now a thin coordinator. It handles:
  - Concurrency-safe caching (locks + per-service caches)
  - Delegation to specialized builders for STT, TTS, and LLM
  - Siri sidecar resolution for macOS

Structure
- builders/
  - stt_builder.py: explicit, testable chains per engine, powered by FallbackChainManager
  - tts_builder.py: uses strategy classes per engine and defaults to Siri fallback
  - llm_builder.py: selects Direct-MLX or HTTP OpenAI-compatible service
- strategies/
  - tts_strategies.py: Kokoro MLX/Professional/PyTorch + SiriStreaming strategies
- utils/
  - fallback_chain.py: ordered attempt executor with structured error aggregation
  - model_resolver.py: resolves Parakeet model path in bundled environments
  - service_validator.py: lightweight TTS validator used by PyTorch retry

Default fallbacks (macOS-aware)
- STT
  - parakeet_isolated: isolated → streaming → macos_native → Whisper MLX
  - parakeet_streaming: streaming → batch → macos_native → Whisper MLX
  - parakeet_batch: batch → macos_native → Whisper MLX
  - parakeet (legacy): streaming → macos_native → Whisper MLX
  - whisper_mlx_direct: direct → macos_native → Whisper MLX
  - macos_native: macos_native → Whisper MLX
  - unknown: macos_native → Whisper MLX
- TTS
  - kokoro_mlx: MLX → Siri → Professional (last resort)
  - kokoro_professional: Professional → Siri → MLX
  - kokoro_pytorch: PyTorch (with retries) → Siri → MLX
  - unknown: Siri → MLX

Caching flags
- SERVICE_FACTORY_CACHE_STT: true/false (default true)
- SERVICE_FACTORY_CACHE_TTS: true/false (default true)
- LLM is always cached; prewarm applies to HTTP mode unless LLM_USE_DIRECT_MLX=true

Notes
- Builders import Kokoro MLX/Professional via ServiceFactory to play nicely with tests that monkeypatch those classes.
- Chains and strategies are designed to be extended with minimal edit surface.

