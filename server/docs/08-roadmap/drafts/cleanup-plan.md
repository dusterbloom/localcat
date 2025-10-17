# TTS Code Cleanup Plan

## Files to Remove (Obsolete Implementations)

### 🔴 Primary Obsolete TTS Implementations (1,278 lines total)
- `tts_mlx_ultra_low_latency.py` (380 lines) - Worker-based approach superseded by native ONNX
- `tts_mlx_streaming.py` (248 lines) - Direct MLX approach, still too slow
- `tts_ultra_low_latency.py` (176 lines) - Wrapper approach, adds unnecessary complexity
- `kokoro_worker_optimized.py` (356 lines) - Subprocess worker, superseded by direct approach
- `text_chunker.py` (118 lines) - Complex token chunking, not needed with native approach

### 🟡 Test Files to Remove (Investigation/Development Tests)
- `test_kokoro_fixes.py` - Development debugging file
- `test_kokoro_streaming.py` - Old streaming test
- `test_chunking_quick.py` - Token chunking test (no longer relevant)
- `test_mlx_investigate.py` - Investigation script (served its purpose)
- `test_no_logging.py` - Performance test without logging
- `test_streaming_approach.py` - MLX streaming test
- `test_tts_complete.py` - Comprehensive test for old approach
- `test_worker_direct.py` - Worker testing
- `test_real_performance.py` - Performance testing
- `tts_stress_test_results.json` - Test results file

### ✅ Files to Keep
- `tts_native_kokoro.py` - **PRIMARY IMPLEMENTATION** (375ms TTFB, stable)
- `test_integration.py` - Current integration test
- `test_native_kokoro.py` - Test for primary implementation

## Rationale

**Keep Native ONNX Approach Because:**
- Best performance: 375ms TTFB consistently
- Most stable: No MLX threading issues
- Simplest architecture: Direct ONNX, no workers/processes
- Production ready: Handles model download, voice validation
- Already integrated: Working in bot.py

**Remove Others Because:**
- MLX approaches: Fundamental 500-1500ms TTFB limit
- Worker approach: Unnecessary complexity via subprocess
- Token chunking: Didn't achieve target performance
- Test files: Development artifacts no longer needed

## Impact
- **Reduce codebase by ~2,000 lines**
- **Eliminate 4 competing TTS implementations**
- **Simplify maintenance to single approach**
- **Remove technical debt and confusion**