# Archived TTS Implementation Files

This directory contains TTS implementations and test files that were archived during the ultra-low latency TTS optimization project.

## Archive Date
September 20, 2025

## Context
After extensive testing and optimization, we consolidated multiple TTS approaches into a single, optimal implementation using native Kokoro ONNX.

## Archived Implementations

### `tts_implementations/`
- **Result**: 375ms TTFB achieved with native ONNX approach
- **Decision**: Keep `tts_native_kokoro.py` as primary implementation
- **Archived approaches**: MLX-based implementations (500-1500ms TTFB), worker processes, token chunking

### `development_tests/`
- **Result**: Comprehensive testing revealed Kokoro's fundamental limitations
- **Decision**: Keep `test_integration.py` and `test_native_kokoro.py`
- **Archived tests**: Development/investigation scripts, performance tests, obsolete approaches

## Performance Summary
- **Target**: 40-80ms TTFB (Kokoro FastAPI best practices)
- **Achieved**: 375ms TTFB (5-10x improvement from original 2.8s)
- **Conclusion**: Best possible with Kokoro ONNX on Apple Silicon

## Key Findings
1. MLX Kokoro generates complete audio (no true streaming)
2. ONNX Kokoro is faster than MLX but still single-chunk
3. 40-80ms TTFB requires different TTS model or server-side optimizations
4. Phrase-level streaming provides best user experience within Kokoro limitations
