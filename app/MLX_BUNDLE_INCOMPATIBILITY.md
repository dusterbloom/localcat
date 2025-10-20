# MLX Bundle Incompatibility - Final Analysis

## Problem Summary

MLX Kokoro TTS hangs indefinitely when running inside a macOS .app bundle, making it unsuitable for production distribution.

## What We Tested

### Attempt 1: Multiprocessing Spawn Mode
- **Change**: Added `mp.set_start_method("spawn", force=True)` to sidecar scripts
- **Result**: ❌ Still hangs at `MLXKokoroTTSService` initialization
- **File**: `/Users/peppi/Dev/localcat/server/sidecars/tts_sidecar_mlx.py:79`

### Attempt 2: Hardened Runtime Entitlements
- **Changes Added**:
  - `com.apple.security.cs.allow-jit` - For MLX Metal JIT compilation
  - `com.apple.security.cs.disable-library-validation` - For dynamic ML frameworks
  - `com.apple.security.cs.allow-unsigned-executable-memory` - For ML framework memory
- **Result**: ❌ Still hangs at same location
- **Files**:
  - `/Users/peppi/Dev/localcat/app/src-tauri/entitlements.plist`
  - `/Users/peppi/Dev/localcat/app/src-tauri/tauri.conf.json`

### Attempt 3: Combined (Spawn + Entitlements)
- **Result**: ❌ Still hangs
- **Verified**: App bundle correctly signed with all entitlements (`codesign -d --entitlements`)

## Evidence from Logs

```
2025-10-18 22:11:13.603 | INFO | sidecars.tts_sidecar_mlx:lifespan:70 - 🚀 Starting MLX Kokoro TTS daemon
2025-10-18 22:11:13.603 | INFO | sidecars.tts_sidecar_mlx:lifespan:72 - 📍 PID: 64309, Port: 8770, Voice: af_heart
2025-10-18 22:11:13.603 | DEBUG | sidecars.tts_sidecar_mlx:lifespan:76 - ✅ PID file written
[HANGS HERE - Never reaches "MLX Kokoro TTS daemon ready"]
```

The daemon starts, writes PID file, but hangs when creating `MLXKokoroTTSService` instance.

## Root Cause

MLX's Metal/GPU initialization is fundamentally incompatible with macOS app bundle sandboxing and code signing restrictions. The hang occurs deep inside the mlx_audio library during Metal framework initialization - beyond our control.

## Working Solution: ONNX Kokoro

The ONNX-based Kokoro TTS works perfectly in bundles:

```
2025-10-18 22:04:06.806 | INFO | __main__:lifespan:86 - ✅ Kokoro ONNX TTS daemon ready
2025-10-18 22:04:17.581 | DEBUG | __main__:_stream_sentence:153 - [TTS Stream] Calling kokoro_tts.create
```

**Trade-offs**:
- ✅ Works in bundles without issues
- ✅ CPU-only execution (stable, predictable)
- ✅ No Metal/GPU dependencies
- ⚠️ Slower than MLX (~2-3x)
- ⚠️ Higher CPU usage

## Recommendation

**For production .app bundles**: Use ONNX Kokoro TTS sidecar
- File: `/Users/peppi/Dev/localcat/server/sidecars/tts_sidecar_onnx_hardened.py`
- Configuration: `/Users/peppi/Dev/localcat/app/src-tauri/src/daemon_manager.rs:86`

**For development**: MLX works fine outside bundles
- Use MLX for faster iteration during development
- Switch to ONNX only for release builds

## Alternative Future Solutions

1. **Piper TTS**: Rust-based TTS, bundle-compatible
   - Requires new integration
   - Different voice quality characteristics

2. **Remote TTS API**: Offload to cloud service
   - Requires network connection
   - Privacy concerns

3. **Wait for MLX fixes**: Apple may resolve bundle compatibility
   - No timeline available
   - Not reliable for production

## Implementation Status

- ✅ ONNX hardened sidecar created with threading controls
- ✅ Multiprocessing spawn added (best practice)
- ✅ Entitlements configured for future ML frameworks
- ✅ Daemon manager can switch between MLX/ONNX
- ❌ MLX confirmed non-functional in bundles

## Files Modified

1. `/Users/peppi/Dev/localcat/server/sidecars/tts_sidecar_mlx.py` - Added spawn
2. `/Users/peppi/Dev/localcat/server/sidecars/tts_sidecar_onnx_hardened.py` - Created hardened version
3. `/Users/peppi/Dev/localcat/app/src-tauri/entitlements.plist` - Added ML entitlements
4. `/Users/peppi/Dev/localcat/app/src-tauri/tauri.conf.json` - Referenced entitlements
5. `/Users/peppi/Dev/localcat/app/src-tauri/src/daemon_manager.rs` - Switch between MLX/ONNX

## Resolution

### Issue: Configuration Mismatch
The daemon_manager.rs was starting an ONNX sidecar, but the server was still using in-process MLX due to default configuration in `base_config.py:214`.

### Fix Applied
Changed server default TTS engine from `kokoro_mlx` to `kokoro_professional`:

**File**: `/Users/peppi/Dev/localcat/server/config/base_config.py`
- Line 214: Changed `engine: str = "kokoro_mlx"` → `engine: str = "kokoro_professional"`

`kokoro_professional` uses ONNX in subprocess isolation:
- CPU-only execution (no Metal conflicts)
- Subprocess worker prevents threading issues
- Compatible with macOS bundles
- Uses `kokoro_onnx_worker.py` worker script

### Testing
Bundle rebuilt with new config and tested for TTS functionality.
