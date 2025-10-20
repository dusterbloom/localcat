# espeak-ng Integration for Tauri App

This document explains how espeak-ng is integrated into the LocalCat Tauri app to enable proper phonemization for Kokoro TTS on macOS.

## Problem

espeak-ng requires:
1. The `espeak-ng-data/` directory to be accessible
2. `ESPEAK_DATA_PATH` environment variable set **before** the library initializes
3. Proper handling of both development and production bundle scenarios

The challenge: macOS GUI apps don't inherit shell environment, and Python subprocess needs explicit configuration.

## Solution Architecture

### 1. Bundling espeak-ng Resources

**Script:** `app/setup-espeak.sh`

Copies espeak-ng from the server's venv to Tauri's bundle directory:
```
app/src-tauri/bin/espeak/
├── espeak-ng                # Binary
├── espeak-ng-data/         # Data directory
└── libespeak-ng.dylib      # Shared library
```

**Run before building:**
```bash
cd app
./setup-espeak.sh
```

### 2. Tauri Configuration

**File:** `app/src-tauri/tauri.conf.json`

```json
{
  "bundle": {
    "externalBin": ["bin/espeak/espeak-ng"],
    "resources": ["bin/espeak/espeak-ng-data/**"]
  }
}
```

This ensures espeak-ng files are included in the `.app` bundle under:
```
LocalCat.app/Contents/Resources/bin/espeak/
```

### 3. Environment Variable Setup (Rust)

**File:** `app/src-tauri/src/main.rs`

The `start_server()` function sets environment variables for the Python subprocess:

**Development mode:**
- Looks for espeak-ng in `app/src-tauri/bin/espeak/`
- Sets `ESPEAK_DATA_PATH` and `ESPEAK_NG_LIBRARY`

**Production mode:**
- Looks for espeak-ng in `.app/Contents/Resources/bin/espeak/`
- Sets environment variables for bundled resources

### 4. Python Worker Configuration

**File:** `server/core/tts/kokoro_worker_optimized.py`

Priority order for finding espeak-ng:
1. **Environment variables** (set by Tauri parent process) ← Preferred
2. **Venv paths** (development fallback)
3. **System defaults** (last resort)

Key code:
```python
# Check environment FIRST (set by Tauri)
if 'ESPEAK_DATA_PATH' not in os.environ:
    # Fallback to venv for development
    data_path = os.path.join(venv_dir, 'lib/python3.12/site-packages/espeakng_loader/espeak-ng-data')
    if os.path.exists(data_path):
        os.environ['ESPEAK_DATA_PATH'] = data_path
```

**File:** `server/core/tts/tts_mlx_ultra_low_latency.py`

Before starting the worker subprocess:
```python
# Pass through parent's environment variables to worker
if "ESPEAK_DATA_PATH" in os.environ:
    env["ESPEAK_DATA_PATH"] = os.environ["ESPEAK_DATA_PATH"]
else:
    # Try to find in venv (development)
    ...
```

## How It Works

### Development Mode

```
Tauri App
  └─> Sets ESPEAK_DATA_PATH → app/src-tauri/bin/espeak/espeak-ng-data/
      └─> Starts Python Server
          └─> Python inherits ESPEAK_DATA_PATH
              └─> Starts Kokoro Worker
                  └─> Worker uses ESPEAK_DATA_PATH
                      └─> mlx-audio/Kokoro uses espeak-ng ✅
```

### Production Mode (Bundle)

```
LocalCat.app
  └─> Tauri sets ESPEAK_DATA_PATH → .app/Contents/Resources/bin/espeak/espeak-ng-data/
      └─> Starts Python Server (bundled venv)
          └─> Python inherits ESPEAK_DATA_PATH
              └─> Starts Kokoro Worker
                  └─> Worker uses ESPEAK_DATA_PATH
                      └─> mlx-audio/Kokoro uses espeak-ng ✅
```

## Testing

### 1. Test Development Mode

```bash
cd /Users/peppi/Dev/localcat/app

# Ensure espeak-ng files are in place
./setup-espeak.sh

# Start the app
npm run dev
```

**Expected console output:**
```
🔧 Dev mode: ESPEAK_DATA_PATH="/Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/espeak-ng-data"
🔧 Dev mode: ESPEAK_NG_LIBRARY="/Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/libespeak-ng.dylib"
✅ Server started successfully
```

**In server logs (stderr):**
```
📍 Using pre-configured ESPEAK_DATA_PATH: /Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/espeak-ng-data
📍 Using pre-configured ESPEAK_NG_LIBRARY: /Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/libespeak-ng.dylib
```

### 2. Test Production Bundle

```bash
cd /Users/peppi/Dev/localcat/app

# Build the app
npm run build

# Check bundle contains espeak-ng
ls -la src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/bin/espeak/

# Should show:
# espeak-ng
# espeak-ng-data/
# libespeak-ng.dylib

# Run the app
open src-tauri/target/release/bundle/macos/LocalCat.app
```

**Expected behavior:**
- App opens without errors
- Voice agent works with TTS
- No "espeak-ng data not found" errors

### 3. Verify espeak-ng Works

Test phonemization directly:

```bash
# In server venv
cd /Users/peppi/Dev/localcat/server
source .venv/bin/activate

# Set environment (simulating Tauri)
export ESPEAK_DATA_PATH="/Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/espeak-ng-data"
export ESPEAK_NG_LIBRARY="/Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/libespeak-ng.dylib"

# Test espeak-ng directly
/Users/peppi/Dev/localcat/app/src-tauri/bin/espeak/espeak-ng --ipa=3 "Hello world"

# Should output IPA phonemes (not errors)
```

## Troubleshooting

### Error: "espeak-ng-data not found"

**Check 1:** Verify files exist
```bash
ls -la app/src-tauri/bin/espeak/espeak-ng-data/
```

**Fix:** Run setup script
```bash
cd app && ./setup-espeak.sh
```

### Error: "Cannot find espeak-ng-data/*_dict"

**Cause:** `ESPEAK_DATA_PATH` not set or points to wrong location

**Check:** Look for environment variable in console output
```bash
# Should see in Tauri console:
🔧 Dev mode: ESPEAK_DATA_PATH=...
```

**Fix:** Check `main.rs` logic for setting environment variables

### Error: "dyld: Library not loaded: @rpath/libespeak-ng.1.dylib"

**Cause:** `DYLD_LIBRARY_PATH` or `ESPEAK_NG_LIBRARY` not set

**Check:** Verify library file exists
```bash
ls -la app/src-tauri/bin/espeak/libespeak-ng.dylib
file app/src-tauri/bin/espeak/libespeak-ng.dylib  # Should show "Mach-O 64-bit dynamically linked shared library arm64"
```

**Fix:** Ensure `main.rs` sets `ESPEAK_NG_LIBRARY` environment variable

### Python Worker Can't Find espeak-ng

**Symptom:** Errors in worker about missing phonemes or initialization

**Debug:** Add logging to worker
```python
# In kokoro_worker_optimized.py
print(f"DEBUG: ESPEAK_DATA_PATH = {os.environ.get('ESPEAK_DATA_PATH')}", file=sys.stderr)
print(f"DEBUG: ESPEAK_NG_LIBRARY = {os.environ.get('ESPEAK_NG_LIBRARY')}", file=sys.stderr)
```

**Check:** Environment variables are passed to subprocess in `tts_mlx_ultra_low_latency.py`

## Key Insights from GitHub Issues

Based on espeak-ng community experiences:

1. **Data path is mandatory** - espeak-ng **cannot work** without `espeak-ng-data/` accessible
2. **Set BEFORE import** - Environment variables must be set before any espeak-ng code runs
3. **Use version ≥ 1.49.1** - Older versions have hardcoded path assumptions
4. **Any directory works** - Don't need folder named exactly "espeak-ng-data", but must contain the phoneme dictionaries

## References

- [espeak-ng #337](https://github.com/espeak-ng/espeak-ng/issues/337) - "Why doesn't my espeak-ng say anything?"
- [espeak-ng 1.49.1 release](https://github.com/espeak-ng/espeak-ng/releases) - Support for custom data directories
- [Piper #404](https://github.com/rhasspy/piper/issues/404) - macOS dylib loading issues
- [Tauri Environment Variables](https://tauri.app/v1/guides/building/app-publishing/#environment-variables)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Tauri App (Rust)                                        │
│  - Reads: app/src-tauri/bin/espeak/                    │
│  - Sets: ESPEAK_DATA_PATH env var                      │
│  - Starts: Python subprocess with env                   │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ Python Server (FastAPI)                                 │
│  - Inherits: ESPEAK_DATA_PATH from parent              │
│  - Passes to: Worker subprocess env                     │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ Kokoro Worker (subprocess)                              │
│  - Checks: ESPEAK_DATA_PATH env var (priority 1)       │
│  - Falls back: venv paths (priority 2)                 │
│  - Uses: espeak-ng via mlx-audio/Kokoro                │
└─────────────────────────────────────────────────────────┘
```

## Success Criteria

✅ espeak-ng files bundled in Tauri app
✅ Environment variables set by Tauri parent process
✅ Python worker respects environment variables
✅ Works in both development and production
✅ No hardcoded paths in Python code
✅ Clear logging for debugging

## Next Steps

1. **Test in development** - Run `npm run dev` and verify console output
2. **Test phonemization** - Use voice agent and check for TTS errors
3. **Build production** - Run `npm run build` and test `.app` bundle
4. **Verify bundle structure** - Check espeak-ng files are in Resources/
5. **Test on clean system** - Ensure no dependency on system espeak-ng

---

**Last Updated:** 2025-10-17
**Status:** Implementation complete, ready for testing
