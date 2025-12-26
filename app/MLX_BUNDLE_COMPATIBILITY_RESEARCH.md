# MLX Bundle Compatibility Research (December 2025)

This document summarizes research into potential solutions for the MLX bundle incompatibility issue documented in `MLX_BUNDLE_INCOMPATIBILITY.md`.

## Problem Summary

MLX Kokoro TTS hangs indefinitely when running inside a macOS `.app` bundle due to Metal GPU initialization conflicts with code signing and sandboxing restrictions.

---

## Recent MLX Improvements

### 1. Metal Library Path Resolution Fixed (PR #2061)

**Merged**: April 23, 2025
**Link**: https://github.com/ml-explore/mlx/pull/2061

The MLX team fixed a critical issue where `mlx.metallib` could only be found next to the binary, which conflicted with macOS app bundle structure and code signing.

**Before**:
- `mlx.metallib` had to be next to `libmlx.dylib`
- Xcode wouldn't sign metallib files in `Frameworks/` directory
- App signing failed

**After** (MLX ≥0.30):
- MLX now searches multiple locations:
  1. Same directory as binary
  2. `Resources/mlx.metallib` (standard macOS bundle location)
  3. `default.metallib` in SwiftPM bundles
  4. Compile-time defined `METAL_PATH`

**Potential Fix for LocalCat**:
```bash
# In build-production.sh, ensure mlx.metallib is in Resources/
MLX_PATH=$(python -c "import mlx; print(mlx.__path__[0])")
cp "${MLX_PATH}/lib/mlx.metallib" "LocalCat.app/Contents/Resources/"
```

### 2. MLX_METAL_JIT Flag

MLX supports JIT compilation of Metal kernels at runtime, which reduces the metallib size and may avoid some initialization issues.

**Environment Variable**:
```bash
export MLX_METAL_JIT=1
```

**Trade-offs**:
- ✅ Smaller metallib footprint
- ✅ May bypass some initialization issues
- ⚠️ Cold start penalty: 200ms - 2s (kernels compiled on first use)
- ⚠️ Cached after first run (persists across reboots)

**Implementation**: Add to `daemon_manager.rs` environment setup:
```rust
cmd.env("MLX_METAL_JIT", "1");
```

### 3. Multiprocessing with Spawn Context

GitHub Issue #2457 confirms the recommended approach for MLX multiprocessing:

**Link**: https://github.com/ml-explore/mlx/issues/2457

**Correct Pattern**:
```python
import multiprocessing as mp

# Use spawn context explicitly (not fork)
ctx = mp.get_context("spawn")
queue = ctx.Queue(5)
process = ctx.Process(target=mlx_worker, args=(queue,))
process.start()
```

**Why Spawn?**
- `fork` mode doesn't work with CUDA/Metal runtimes
- `spawn` creates fresh Python interpreter without inherited GPU state
- Already implemented in LocalCat's `tts_sidecar_mlx.py`

---

## What Likely Still Won't Work

The core issue is that MLX hangs at **Metal GPU initialization** inside bundled apps. Research suggests this is related to:

1. **Code Signing Validation**: Even with `disable-library-validation` entitlement, Metal initialization may fail in hardened runtime
2. **Hardened Runtime + Metal JIT**: Security policies may block Metal's JIT compilation
3. **Subprocess Metal Context**: Metal GPU contexts don't transfer across process boundaries cleanly

### LM Studio Evidence

LM Studio (a production MLX app) reports similar issues on macOS Sequoia:
- Loading a second MLX model crashes with "insufficient system resources"
- Warnings about system freeze from resource overloading
- Suggests MLX + bundled apps have known stability issues

**Link**: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/927

---

## Recommendations

| Priority | Action | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| **1. Try First** | Update to MLX ≥0.30, place `mlx.metallib` in `Contents/Resources/` | Low | May resolve metallib path issues |
| **2. Try Second** | Add `MLX_METAL_JIT=1` to environment | Low | Reduces initialization complexity |
| **3. Try Third** | Use native Swift MLX instead of Python mlx-audio | High | Apple's recommended production approach |
| **4. Fallback** | Keep ONNX Kokoro (current solution) | Done | Working, 2-3x slower but stable |

---

## Native Swift MLX Alternative

WWDC 2025 heavily promoted MLX Swift for production apps:
- **Session 315**: "Get started with MLX for Apple silicon"
- **Session 298**: "Explore large language models on Apple silicon with MLX"

**Links**:
- https://developer.apple.com/videos/play/wwdc2025/315/
- https://developer.apple.com/videos/play/wwdc2025/298/

The `mlx-audio` package ships a Swift TTS component that could replace the Python implementation:

**Benefits**:
- Avoids Python subprocess complexity entirely
- Uses Apple's blessed Metal integration path
- Integrates cleanly with Tauri via Swift sidecar (like existing `siri-tts`)
- Native code signing compatibility

**Implementation Path**:
1. Build Swift MLX TTS binary (similar to `siri-tts` sidecar)
2. Bundle as Tauri sidecar in `src-tauri/sidecar/mlx-tts/`
3. Call from `daemon_manager.rs` like other sidecars

---

## Test Script

To verify if recent MLX fixes help with bundle compatibility:

```bash
#!/bin/bash
# test-mlx-bundle-compat.sh

# 1. Update MLX to latest
pip install -U mlx mlx-audio

# 2. Check MLX version (should be ≥0.30)
echo "MLX Version:"
python -c "import mlx; print(mlx.__version__)"

# 3. Find metallib location
echo "Metallib location:"
python -c "import mlx; import os; print(os.path.join(mlx.__path__[0], 'lib', 'mlx.metallib'))"

# 4. Test basic MLX with JIT flag
echo "Testing MLX with JIT..."
MLX_METAL_JIT=1 python -c "import mlx.core as mx; print('Array test:', mx.array([1,2,3]))"

# 5. Test mlx-audio TTS initialization
echo "Testing mlx-audio TTS..."
MLX_METAL_JIT=1 python -c "
from mlx_audio.tts.models import KokoroModel
print('Attempting KokoroModel load...')
# Note: This will download model on first run
"
```

---

## Implementation Checklist

If attempting to re-enable MLX in bundles:

- [ ] Update `requirements.txt` to `mlx>=0.30`
- [ ] Modify `build-production.sh` to copy `mlx.metallib` to `Resources/`
- [ ] Add `MLX_METAL_JIT=1` to `daemon_manager.rs` environment
- [ ] Test bundle with fresh macOS Sequoia installation
- [ ] Verify code signing: `codesign -dv --verbose=4 LocalCat.app`
- [ ] Check entitlements: `codesign -d --entitlements - LocalCat.app`
- [ ] Monitor Console.app for Metal/GPU errors during startup

---

## Current Production Solution

Until MLX bundle compatibility is resolved, the **ONNX Kokoro TTS** remains the recommended production solution:

**File**: `server/sidecars/tts_sidecar_onnx_hardened.py`

**Characteristics**:
- ✅ Works reliably in macOS bundles
- ✅ CPU-only execution (no Metal/GPU conflicts)
- ✅ Subprocess isolation prevents threading issues
- ✅ Hardened with single-threaded ONNX runtime
- ⚠️ 2-3x slower than MLX GPU acceleration
- ⚠️ Higher CPU usage

**Alternative**: Siri TTS (`siri-tts` sidecar)
- Native macOS integration
- Zero model loading time (~50ms startup)
- 25+ language variants
- Works offline

---

## References

- [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- [MLX PR #2061 - Resources directory search](https://github.com/ml-explore/mlx/pull/2061)
- [MLX Issue #2457 - Multiprocessing queues](https://github.com/ml-explore/mlx/issues/2457)
- [MLX Issue #1286 - Xcode integration](https://github.com/ml-explore/mlx/issues/1286)
- [MLX Build Documentation](https://ml-explore.github.io/mlx/build/html/install.html)
- [WWDC25 Session 315 - Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)
- [WWDC25 Session 298 - Explore LLMs with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)
- [mlx-audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [LM Studio MLX Issues](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/927)

---

*Last Updated: December 2025*
