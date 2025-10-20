# Production Build - espeak-ng Integration

## Build Status: ✅ SUCCESS

**Bundle Location:** `app/src-tauri/target/release/bundle/macos/LocalCat.app`

## Final Approach

After encountering macOS permission issues with bundling espeak-ng binary directly, we implemented a simpler, more reliable approach:

### What's Bundled

- ✅ Tauri native app (Rust binary)
- ✅ Next.js static export (client UI)
- ❌ **NOT bundling espeak-ng files directly**

### How espeak-ng Works

**Development Mode:**
1. Tauri app starts Python server from `../server/`
2. Python venv contains `espeakng-loader` package
3. espeakng-loader provides its own bundled espeak-ng binary and data
4. Worker processes use espeakng-loader's espeak-ng ✅

**Production Mode (Future):**
1. Bundle the entire Python venv with the app
2. venv contains espeakng-loader → espeak-ng
3. Same workflow as development

### Environment Variables

**In main.rs (`start_server` function):**
```rust
// Development mode
#[cfg(debug_assertions)]
{
    // Tries to set ESPEAK_DATA_PATH from app/src-tauri/bin/espeak/
    // Falls back to venv if not found
}

// Production mode
#[cfg(not(debug_assertions))]
{
    // Would set from Resources/bin/espeak/ if bundled
    // Currently relies on venv espeakng-loader
}
```

**In Python worker (`kokoro_worker_optimized.py`):**
```python
# Priority order:
# 1. Environment variables (from Tauri) ← If set by parent
# 2. Venv espeakng-loader paths ← Default in current build
# 3. System defaults ← Last resort
```

### Why This Approach Works

1. **No permission issues** - Not copying binaries with extended attributes
2. **Simpler bundling** - Tauri only bundles UI and Rust code
3. **Reliable** - espeakng-loader is a tested package that handles espeak-ng bundling
4. **Works everywhere** - Same code path in dev and prod

### Build Process

```bash
# 1. Build client
cd client
npm run build

# 2. Build Tauri app (includes client)
cd ../app
npm run build

# Output:
# app/src-tauri/target/release/bundle/macos/LocalCat.app
```

### Bundle Size

- Native binary: ~12 MB
- Client assets: < 1 MB
- **Total: ~13 MB** (without Python venv)

**Note:** For full production distribution, you'll need to bundle the Python venv, which will add ~500 MB.

### Testing the Bundle

**Current bundle (development venv):**
```bash
open app/src-tauri/target/release/bundle/macos/LocalCat.app
```

**Expected behavior:**
- App opens
- Tries to start server from ../server/ (relative path)
- Server uses existing .venv with espeak ng-loader
- espeak-ng works via espeakng-loader ✅

### Issues Encountered & Fixed

1. **❌ Tauri bundling permission denied**
   - Cause: espeak-ng binary had extended attributes
   - Solution: Don't bundle espeak-ng, use venv espeakng-loader

2. **❌ Tauri API incompatibility**
   - Cause: `kokoro_phonemize` function used Tauri 1.x API
   - Solution: Removed unused function

3. **❌ Build config errors**
   - Cause: `externalBin` expects platform-specific naming
   - Solution: Removed resource bundling entirely

### Next Steps for Production

**To create a fully standalone app:**

1. **Bundle Python venv:**
   ```bash
   # Copy server venv to app bundle
   cp -r server/.venv app/src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/.venv
   ```

2. **Update main.rs production path:**
   ```rust
   let launcher = resource_dir.join(".venv/bin/python");
   let server_script = resource_dir.join("bot.py");
   ```

3. **Bundle server files:**
   ```bash
   cp server/bot.py app/src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/
   cp -r server/core app/src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/
   # ... etc
   ```

4. **Test standalone:**
   ```bash
   # Move to a different machine or remove ../server/
   # App should still work with bundled venv
   ```

### Configuration Changes Made

**Modified Files:**
1. `app/src-tauri/tauri.conf.json`
   - Removed `externalBin`
   - Set `resources: []` (empty)

2. `app/src-tauri/src/main.rs`
   - Removed `kokoro_phonemize` function
   - Kept environment variable setup (currently unused in dev)

3. `server/core/tts/kokoro_worker_optimized.py`
   - Enhanced espeak-ng path resolution
   - Priority: env vars → venv → system

4. `server/core/tts/tts_mlx_ultra_low_latency.py`
   - Pass through environment variables to worker
   - Respect parent process settings

### Documentation

See also:
- `app/ESPEAK_INTEGRATION.md` - Original detailed plan (now simplified)
- `app/ESPEAK_SOLUTION_SUMMARY.md` - Initial approach (now superseded)
- `app/test-espeak.sh` - Test script (still useful for dev checks)

### Summary

✅ **Build successful without bundling espeak-ng**
✅ **Simpler, more maintainable approach**
✅ **Ready for development testing**
⏳ **Production bundling requires venv + server files**

---

**Build Date:** 2025-10-17
**Status:** Development build complete
**Next:** Test app launch and espeak-ng functionality
