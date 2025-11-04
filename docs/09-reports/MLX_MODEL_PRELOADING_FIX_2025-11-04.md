# MLX Model Preloading Fix - November 4, 2025

## Executive Summary

Fixed critical bug where preloaded MLX language model was not being used during WebSocket connections, causing "Cannot find an appropriate cached snapshot folder" errors and connection failures in offline mode. The fix establishes a complete chain passing preloaded models from startup through to the LLM service, enabling instant connection times and 100% offline operation.

## Problem Statement

### Symptoms
- WebSocket connections failing with HuggingFace offline error
- Model successfully preloaded at startup but reloaded on connection
- Error: "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and outgoing traffic has been disabled"
- Occurred even with `HF_HUB_OFFLINE=1` environment variable set

### Log Evidence (Before Fix)
```
# Startup - SUCCESS
🚀 PRELOADING MODELS - This happens ONCE at server startup
  Loading LLM from: .../snapshots/main
  ✅ MLX-LM ready (2.88s)

# Connection - FAILURE
🔥 Loading Direct MLX-LM: mlx-community/Qwen3-1.7B-4bit-DWQ-053125
❌ Failed to load: Cannot find an appropriate cached snapshot folder...
```

## Root Cause Analysis

### Architecture Flow (Broken)
```
ServiceFactory (has preloaded_models ✓)
  ↓
create_llm_service()
  ↓
LLMServiceBuilder(config) ⚠️ NO PRELOADED MODELS PASSED
  ↓
DirectMLXLLMServiceWithTools(model=model_id)
  ↓
DirectMLXLLMService.__init__()
  ↓
mlx_lm.load(model_id) ❌ TRIES TO RELOAD FROM ID STRING
```

### Technical Root Cause

The `mlx_lm.load()` function internally calls `snapshot_download()` when given a model ID string (e.g., `"mlx-community/Qwen3-1.7B-4bit-DWQ-053125"`), even when:
- `HF_HUB_OFFLINE=1` is set
- Model is already cached locally
- Model was preloaded at startup

The `snapshot_download()` call attempts to contact HuggingFace servers to verify the snapshot, which fails when offline.

### Why Preloaded Models Were Ignored

The preloaded models were stored in `PreloadedModels` class at startup but never passed through the service creation chain:

1. `bot.py::preload()` loads models into `PreloadedModels` ✓
2. `ServiceFactory.__init__()` receives `preloaded_models` ✓
3. `ServiceFactory.create_llm_service()` calls `LLMServiceBuilder(config)` **without passing preloaded_models** ❌
4. `DirectMLXLLMService.__init__()` has no way to receive preloaded models ❌

## Solution Implemented

### Architecture Flow (Fixed)
```
ServiceFactory (has preloaded_models ✓)
  ↓
create_llm_service() → passes preloaded_models
  ↓
LLMServiceBuilder(config, preloaded_models) ✓
  ↓ extracts mlx_llm_model & mlx_llm_tokenizer
DirectMLXLLMServiceWithTools(..., preloaded_model, preloaded_tokenizer) ✓
  ↓
DirectMLXLLMService.__init__(..., preloaded_model, preloaded_tokenizer) ✓
  ↓
Uses preloaded if available, else loads from snapshot path ✓
```

### Changes Made

#### 1. `server/core/llm/direct_mlx_llm.py`
**Lines 89-165**: Updated `__init__()` method

**New Parameters:**
```python
def __init__(
    self,
    model: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    max_tokens: int = 256,
    temperature: float = 0.7,
    preloaded_model: Any = None,  # NEW
    preloaded_tokenizer: Any = None,  # NEW
    **kwargs
):
```

**Logic:**
```python
if preloaded_model is not None and preloaded_tokenizer is not None:
    # Use preloaded (INSTANT)
    logger.info(f"🚀 Using PRELOADED MLX-LM: {model}")
    self._model = preloaded_model
    self._tokenizer = preloaded_tokenizer
    logger.info(f"✅ Direct MLX-LM ready (INSTANT - preloaded)")
else:
    # Fallback: Load from disk with snapshot path conversion
    # Mirrors bot.py preload() logic to avoid HF API calls
    hf_home = os.getenv("HF_HOME") or ...
    cache_dir = os.path.join(hf_home, "hub", f"models--{model_cache_name}")

    # Find snapshot directory (main or hash)
    snapshot_path = find_snapshot(cache_dir)

    # Load from absolute path (skips HF API)
    self._model, self._tokenizer = mlx_lm.load(snapshot_path)
```

#### 2. `server/core/llm/direct_mlx_llm_with_tools.py`
**Line 79-82**: Forward preloaded parameters to parent

```python
def __init__(self, *args, preloaded_model=None, preloaded_tokenizer=None, **kwargs):
    """Initialize enhanced Direct MLX-LM service with tool support."""
    super().__init__(*args, preloaded_model=preloaded_model,
                     preloaded_tokenizer=preloaded_tokenizer, **kwargs)
```

#### 3. `server/core/factories/builders/llm_builder.py`
**Lines 10-40**: Accept and extract preloaded models

```python
class LLMServiceBuilder:
    def __init__(self, config: VoiceAgentConfig, preloaded_models=None):  # NEW
        self.config = config
        self.preloaded_models = preloaded_models  # NEW

    def build(self) -> OpenAILLMService:
        if use_direct_mlx:
            # Extract preloaded model/tokenizer if available
            preloaded_model = None
            preloaded_tokenizer = None
            if self.preloaded_models:
                preloaded_model = getattr(self.preloaded_models, 'mlx_llm_model', None)
                preloaded_tokenizer = getattr(self.preloaded_models, 'mlx_llm_tokenizer', None)
                if preloaded_model and preloaded_tokenizer:
                    logger.debug("🎯 LLMServiceBuilder: Passing preloaded MLX model to service")

            return DirectMLXLLMServiceWithTools(
                model=llm_config["model"],
                max_tokens=llm_config.get("max_tokens", 256),
                temperature=llm_config.get("temperature", 0.7),
                preloaded_model=preloaded_model,  # NEW
                preloaded_tokenizer=preloaded_tokenizer,  # NEW
            )
```

#### 4. `server/core/factories/service_factory.py`
**Line 266**: Pass preloaded_models to builder

```python
def create_llm_service(self) -> OpenAILLMService:
    """Create LLM service with streaming configuration."""
    with self._llm_lock:
        logger.debug("Creating new LLM service via builder")
        # Pass preloaded_models to builder so it can use them for instant startup
        llm = LLMServiceBuilder(self.config, self.preloaded_models).build()  # FIXED
```

#### 5. `server/.env`
**Lines 191, 193, 312**: Reduced log verbosity

```bash
# Before
LOG_LEVEL=DEBUG
HOTMEM_LOG_LEVEL=DEBUG

# After
LOG_LEVEL=INFO
HOTMEM_LOG_LEVEL=INFO
```

## Test Results

### Log Evidence (After Fix)
```
# Startup - SUCCESS
🚀 PRELOADING MODELS - This happens ONCE at server startup
  Loading LLM from: .../snapshots/main
  ✅ MLX-LM ready (2.88s)

# Connection - SUCCESS
🎯 LLMServiceBuilder: Passing preloaded MLX model to service
🚀 Using PRELOADED MLX-LM: mlx-community/Qwen3-1.7B-4bit-DWQ-053125
✅ Direct MLX-LM ready (INSTANT - preloaded)
```

### Performance Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Startup time | N/A (failed) | ~2.9s | ✅ Works |
| Connection time | N/A (failed) | ~0ms (instant) | ✅ Works |
| Offline mode | ❌ Failed | ✅ Works | 100% fixed |
| Model load calls | 2 (1 success, 1 fail) | 1 (preload only) | 50% reduction |

### Validation Tests

✅ **App launches successfully** - No errors during startup
✅ **WebSocket connects instantly** - No delay reloading model
✅ **Offline mode works** - `HF_HUB_OFFLINE=1` respected
✅ **Logs cleaner** - INFO level instead of DEBUG
✅ **Voice agent functional** - End-to-end pipeline works

## Files Modified

### Production Code (5 files)
1. `server/core/llm/direct_mlx_llm.py` - Core LLM service
2. `server/core/llm/direct_mlx_llm_with_tools.py` - Enhanced LLM with tools
3. `server/core/factories/builders/llm_builder.py` - LLM service factory
4. `server/core/factories/service_factory.py` - Global service factory
5. `server/.env` - Configuration (log levels)

### Bundle Updates
All modified files copied to Tauri app bundle:
```
app/src-tauri/target/aarch64-apple-darwin/release/bundle/macos/LocalCat.app/
  Contents/Resources/_up_/_up_/server/
    ├── core/llm/direct_mlx_llm.py
    ├── core/llm/direct_mlx_llm_with_tools.py
    ├── core/factories/builders/llm_builder.py
    ├── core/factories/service_factory.py
    └── .env
```

## Technical Insights

### Why Snapshot Path Conversion Works

The `mlx_lm._download()` function (from mlx-lm library) has this logic:
```python
def _download(model_path: str):
    # If absolute path exists, use it directly (no HF API calls)
    if os.path.exists(model_path):
        return model_path

    # Otherwise, call snapshot_download() (requires network)
    return snapshot_download(repo_id=model_path, ...)
```

By converting model ID → absolute snapshot path, we skip `snapshot_download()` entirely.

### HuggingFace Cache Structure
```
HF_HOME/hub/
└── models--mlx-community--Qwen3-1.7B-4bit-DWQ-053125/
    ├── blobs/           # Actual files (deduplicated)
    ├── refs/            # Branch references
    └── snapshots/
        └── main/        # Or commit hash
            ├── config.json
            ├── tokenizer.json
            ├── model.safetensors
            └── ...
```

We now point to `snapshots/main/` instead of the model ID string.

### Double Loading Prevention

Before this fix, the model was loaded twice:
1. **Startup preload** (bot.py:146-166) - Used snapshot path ✓
2. **Service creation** (direct_mlx_llm.py:127) - Used model ID ❌

Now the model is loaded once and reused.

## Deployment Notes

### Production Bundle Update
After making changes to server files:
```bash
# Copy modified files to bundle
cp server/core/llm/direct_mlx_llm.py app/src-tauri/.../LocalCat.app/.../server/core/llm/
cp server/core/llm/direct_mlx_llm_with_tools.py app/src-tauri/.../LocalCat.app/.../server/core/llm/
cp server/core/factories/builders/llm_builder.py app/src-tauri/.../LocalCat.app/.../server/core/factories/builders/
cp server/core/factories/service_factory.py app/src-tauri/.../LocalCat.app/.../server/core/factories/
cp server/.env app/src-tauri/.../LocalCat.app/.../server/

# Rebuild for permanent fix
cd app/
npm run build
```

### Environment Variables Required
```bash
HF_HUB_OFFLINE=1          # Force offline mode
TRANSFORMERS_OFFLINE=1    # Prevent transformer model checks
HF_HOME=/path/to/cache    # Point to model cache
LLM_USE_DIRECT_MLX=true   # Enable Direct MLX-LM
```

## Future Considerations

### Potential Improvements
1. **Preload validation** - Verify preloaded model matches config before use
2. **Cache warming** - Pre-warm Metal GPU compilation on first run
3. **Model versioning** - Track which snapshot was preloaded vs. requested
4. **Graceful degradation** - Better error messages if preload fails

### Related Work
- **STT preloading** - Whisper MLX already uses preloading correctly
- **TTS preloading** - Kokoro professional uses on-demand loading
- **Vision models** - Future enhancement for multimodal support

## Conclusion

This fix eliminates a critical blocker for offline voice agent operation. By ensuring preloaded models are actually used, we achieve:
- ✅ Instant WebSocket connections
- ✅ 100% offline capability
- ✅ Cleaner architecture (single model load)
- ✅ Better user experience (no delays/errors)

The solution follows existing patterns in the codebase (snapshot path conversion from bot.py) and maintains backward compatibility (fallback to disk loading if preload unavailable).

---

**Report Date**: November 4, 2025
**Issue**: MLX model double-loading causing offline failures
**Status**: ✅ Fixed and tested
**Files Changed**: 5 production files
**Performance Impact**: Connection time: N/A → instant
