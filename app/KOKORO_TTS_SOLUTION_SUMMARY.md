# Kokoro TTS Solution Analysis & Recommendation

## Problem Statement

Kokoro ONNX TTS consistently hangs during `kokoro_tts.create()` calls in the sidecar daemon, specifically during espeak phonemization or ONNX Runtime session execution.

## Solutions Tested

### ❌ Option 1: Minimal Fixes to Existing ONNX Sidecar (FAILED)

**Changes Made:**
1. Fixed event loop deprecation warning (`asyncio.new_event_loop()` instead of `get_event_loop()`)
2. Set `OMP_NUM_THREADS=1` and `ORT_LOG_SEVERITY_LEVEL=3` environment variables
3. Reduced ThreadPoolExecutor to single worker
4. Used `get_running_loop()` in async context

**Result:** Daemon initializes successfully but **hangs indefinitely during TTS generation** (`kokoro_tts.create()` never returns)

**Root Cause:** The kokoro-onnx wrapper has a fundamental issue with espeak phonemization or ONNX Runtime threading that cannot be fixed with environment variables or threading configuration alone. The hang occurs deep in the library stack, not in our code.

### ❌ Option 2: Direct ONNX + Misaki G2P (ABANDONED)

**Why Abandoned:** Misaki **is not an espeak replacement** - it still uses espeak internally via the `espeak` submodule. Would have the exact same hanging issue.

### ✅ Option 3: Kokoros (Rust) - RECOMMENDED

**Why This Will Work:**
1. **Complete rewrite in Rust** - no Python wrapper complications
2. **Built specifically for sidecar pattern** - designed for process isolation
3. **Single, signable binary** - simplifies bundling and codesigning
4. **OpenAI-compatible API** - easy integration
5. **Native Apple Silicon performance**
6. **Active maintenance** - community support

**Potential Concerns:**
- Still uses espeak internally (but Rust implementation may handle threading better)
- Need to verify espeak path resolution in bundled apps
- API differences require integration work

## Recommendation

**Proceed with Kokoros (Rust) immediately.**

The kokoro-onnx Python wrapper has proven unreliable on macOS with uvicorn/FastAPI. Multiple attempts with different threading configurations all result in the same hang. The Rust implementation is the only viable path forward for production deployment.

## Next Steps

1. Install Kokoros: `cargo install kokoros`
2. Test standalone: `kokoros --host 127.0.0.1 --port 8770`
3. Integrate with Tauri as sidecar binary
4. Update daemon_manager.rs to launch Kokoros instead of Python
5. Test audio generation reliability
6. If espeak issues persist, evaluate alternative TTS engines (Piper, Chatterbox)

## Files Modified

- `/Users/peppi/Dev/localcat/server/sidecars/tts_sidecar_onnx.py` - Applied minimal fixes (event loop, threading)
- `/Users/peppi/Dev/localcat/server/sidecars/tts_sidecar_onnx_hardened.py` - Created hardened version with explicit ORT configuration

Both files demonstrate successful daemon initialization but fail during actual TTS generation.

## Evidence

```
2025-10-18 21:32:20.150 | DEBUG | [TTS Stream] Calling kokoro_tts.create (full generation)
<hangs indefinitely - never completes>
```

This pattern repeats across all testing attempts, regardless of threading configuration or environment variables.

## Alternative: Keep MLX as Primary

If Kokoros also fails, consider keeping MLX as the primary TTS engine (it appears to work better from logs) and only offer ONNX/Kokoros as a fallback for older Macs without MLX support.
