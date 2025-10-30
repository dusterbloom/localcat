# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Tauri-based native macOS application wrapper** for LocalCat, a local voice agent built with Pipecat framework. The app provides a polished, integrated desktop experience that automatically manages the Python server lifecycle and presents the voice UI in a native window.

## Architecture

### Three-Layer Structure

```
LocalCat Native App
├── Tauri Shell (Rust) - Process management, native window, resource bundling
├── Client UI (Next.js) - Voice interface, visualizers, transcripts
└── Server (Python) - FastAPI bot.py with Pipecat voice pipeline
```

The Tauri app acts as a **thin orchestration layer** that:
- Manages Python server lifecycle (start on launch, stop on quit)
- Handles TTS sidecar process coordination
- Configures environment for offline model usage
- Sets up espeak-ng paths for TTS phonemization
- Bundles all resources for distribution

**Important architectural principle:** The Tauri wrapper does NOT modify server or client logic. It only manages process lifecycle and environment configuration.

## Development Commands

### Prerequisites

All commands require specific versions:
```bash
# Node.js 22 (required for client)
nvm use 22

# Verify Rust installation
rustc --version  # Should be 1.82.0+

# Verify Python 3.12 (used by server)
python3.12 --version
```

### Primary Development Workflow

```bash
# Start development mode (recommended)
cd app/
nvm use 22
npm run dev
```

This single command:
1. Starts Tauri in dev mode
2. Auto-starts Python server from `../server/bot.py` on port 7860
3. Launches Next.js dev server on port 3000
4. Opens native window showing client UI
5. Enables hot-reload for both client and Rust changes

### Build Production Bundle

```bash
# Build macOS app (.app + .dmg)
cd app/
nvm use 22
npm run build
```

Output locations:
- `src-tauri/target/release/bundle/macos/LocalCat.app`
- `src-tauri/target/release/bundle/dmg/LocalCat_1.0.0_aarch64.dmg`

### Client-Only Development

```bash
# Run Next.js dev server separately (testing UI changes)
cd app/
npm run client:dev

# Build client for production
cd app/
npm run client:build
```

### Server-Only Development

```bash
# Run Python server manually (testing bot.py changes)
cd server/
source .venv/bin/activate
python bot.py --host 127.0.0.1 --port 7860
```

## Key Code Locations

### Process Management (`src-tauri/src/main.rs`)

**Core responsibilities:**
- `start_server()` - Launches Python bot.py with correct venv and environment
- `stop_server()` - Gracefully terminates Python process
- `ensure_tts_sidecar()` - Manages optional MLX TTS sidecar (macOS only)
- `get_server_paths()` - Resolves paths for dev vs production mode
- `setup_espeak_symlink()` - Creates `/tmp/espeak-ng-data` symlink for TTS

**Development vs Production paths:**
- Dev: Uses `../server/.venv/bin/python3` (existing venv)
- Prod: Uses bundled venv in `Resources/_up_/_up_/server/.venv/`

**Environment configuration:**
The Rust code sets critical environment variables before spawning Python:
- `HF_HUB_OFFLINE=1` - Forces offline mode for HuggingFace models
- `TRANSFORMERS_OFFLINE=1` - Prevents network model checks
- `HF_HOME` - Points to bundled model cache
- `ESPEAK_DATA_PATH` - Points to bundled espeak-ng-data
- `SKIP_TTS_VALIDATION=true` - Skips TTS voice validation in production

**Database path resolution:**
LocalCat uses a two-tier strategy to ensure database paths work correctly across development and production:

**Development Mode** (running `server/bot.py` directly):
- Reads `server/.env` for database paths
- Uses `~/Library/Application Support/LocalCat/data/` (portable across users)
- Python's `os.path.expanduser()` expands `~` to current user's home
- Falls back to `server/data/` if env vars not set

**Production Mode** (running from LocalCat.app bundle):
- Rust code in `daemon_manager.rs:316-338` **overrides** .env paths at runtime
- Uses `$HOME/Library/Application Support/LocalCat/data/` (current user's home)
- Creates directories automatically if they don't exist
- Environment variables set:
  - `MEMORY_SQLITE_PATH` → `memory.db` (main conversation memory)
  - `MEMORY_LMDB_PATH` → `memory.lmdb/` (graph adjacency index)
  - `SESSION_DB_PATH` → `sessions.db` (conversation sessions)
  - `SPEAKER_PROFILE_DIR` → `speaker_profiles/` (voice enrollments)

**Per-User Data Locations:**
```
~/Library/Application Support/LocalCat/data/
├── memory.db           (SQLite - conversation memory)
├── memory.lmdb/        (LMDB - fast graph lookups)
├── sessions.db         (SQLite - session metadata)
├── speaker_profiles/   (Voice enrollment data)
└── semantic_index/     (Vector embeddings)
```

**Why This Design:**
1. **Portability**: `~` in .env works for all users (no hardcoded `/Users/username/`)
2. **Safety**: Rust override ensures production always uses correct paths
3. **Fallback**: Python auto-detects environment if paths not explicitly set
4. **macOS Standard**: Follows Apple's Application Support conventions

### Build Process (`src-tauri/build.rs`)

The build script runs before compilation to **hydrate voice symlinks**:
- Converts symlinked `.pt` voice files to actual files
- Required for bundling voice models in app bundle
- Only runs during production builds

### Tauri Configuration (`src-tauri/tauri.conf.json`)

**Key settings:**
- `build.devUrl: "http://localhost:3000"` - Points to Next.js dev server
- `build.frontendDist: "../../client/out"` - Points to static Next.js export
- `bundle.resources` - Lists all server files to bundle
- `macOS.signingIdentity` - Apple Developer certificate for distribution

## Server Connection Flow

1. **Tauri starts** → `main.rs::setup()` runs
2. **Python server auto-starts** → `start_server()` spawns `bot.py`
3. **Next.js dev server starts** → Tauri's `beforeDevCommand` runs
4. **Window opens** → Shows `http://localhost:3000` (dev) or static files (prod)
5. **Client connects** → Uses `NEXT_PUBLIC_SERVER_URL=http://127.0.0.1:7860`

## Configuration

### Server Port

Default: `7860` (hardcoded in `main.rs:335`)

To change:
1. Edit `src-tauri/src/main.rs` line 335: `.arg("7860")`
2. Update `../client/.env.local`: `NEXT_PUBLIC_SERVER_URL=http://127.0.0.1:NEW_PORT`

### TTS Engine Selection

**Development mode:**
- Uses `server/.env` configuration
- Respects `VOICE_AGENT_TTS_ENGINE` setting

**Production mode:**
- Forces `VOICE_AGENT_TTS_ENGINE=kokoro_mlx` (in-process MLX)
- Disables sidecar mode for stability on older macOS/WebKit
- See `main.rs:422-423`

### HuggingFace Model Cache

**Critical for offline operation:**
- Models must be pre-cached in `server/models/hf_cache/`
- Production bundle includes entire cache directory
- First run downloads models if cache is empty

## Common Development Tasks

### Testing Server Startup

```bash
# Check if Python is found correctly
cd app/src-tauri
cargo run  # Will print detected Python path

# Expected output:
# ✅ Server started on http://127.0.0.1:7860
# 💡 Server starting, please wait 10-30 seconds...
```

### Debugging espeak-ng Issues

```bash
# Verify espeak-ng paths in dev mode
ls app/src-tauri/bin/espeak/espeak-ng-data
ls app/src-tauri/bin/espeak/libespeak-ng.dylib

# Check symlink in production
ls -la /tmp/espeak-ng-data
# Should point to bundled espeak-ng-data
```

### Verifying Model Bundle

```bash
# After build, check bundled models
cd src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/
ls -la _up_/_up_/server/models/hf_cache/hub/

# Should contain model directories like:
# models--mlx-community--Kokoro-82M-bf16/
# models--nvidia--parakeet-tdt-1.1b/
```

### Running Tests

There are no automated tests for the Tauri wrapper currently. Testing is manual:

1. **Dev mode test:** `npm run dev` → verify window opens, server starts
2. **Production build test:** `npm run build` → verify .app runs without errors
3. **Server lifecycle test:** Launch app → quit app → verify Python process stops

## Important Implementation Notes

### Why No TTS Sidecar in Production?

The MLX TTS sidecar (`ensure_tts_sidecar()`) is **disabled in production** (see `main.rs:296`):
- Older macOS WebKit environments can cause mid-stream termination
- In-process Kokoro MLX is more stable for bundled apps
- Sidecar still useful in development for experimentation

### Why Symlink for espeak-ng?

The `kokoro-onnx` library has a **hardcoded CI build path** for espeak-ng-data:
- Expects `/tmp/espeak-ng-data` (from GitHub Actions builds)
- Creating symlink avoids needing to patch binary
- Symlink in `/tmp` doesn't require sudo

### Resource Bundling Path Transformation

Tauri converts `../` in resource paths to `_up_/`:
- Config: `../../server/` → Bundle: `_up_/_up_/server/`
- This is why production code uses `_up_/_up_/server` paths

## Troubleshooting

### "Python not found"

**Check detection logic** (`main.rs:66-99`):
- Dev: Looks for `../server/.venv/bin/python3`
- Prod: Looks for `Resources/_up_/_up_/server/.venv/bin/python3`

**Fix:**
```bash
# Verify venv exists
ls -la server/.venv/bin/python3

# Recreate venv if needed
cd server/
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### "Port 7860 already in use"

```bash
# Find process using port
lsof -i :7860

# Kill it
kill -9 <PID>

# Or use the built-in cleanup (for sidecar only)
# The main server uses a fixed port and won't auto-cleanup
```

### "Models downloading on first run"

This is **expected behavior** if model cache is empty:
- Parakeet STT: ~1.1GB download
- Kokoro TTS: ~160MB download
- Silero VAD: ~100MB download

**Speed up subsequent runs:**
```bash
# After first successful run, models are cached
# Offline mode prevents re-checking model versions
# Already set in production: HF_HUB_OFFLINE=1
```

### "Client can't connect to server"

**Verify server is running:**
```bash
curl http://127.0.0.1:7860/
# Should return response (not connection refused)
```

**Check client configuration:**
```bash
# Verify environment variable
cat ../client/.env.local
# Should contain: NEXT_PUBLIC_SERVER_URL=http://127.0.0.1:7860
```

**Check browser console:**
- Open DevTools in app window (if enabled)
- Look for WebRTC connection errors
- Check for CORS issues (shouldn't occur with localhost)

## Distribution Checklist

Before releasing a .dmg:

- [ ] **Code signing:** Configure Apple Developer certificate in `tauri.conf.json`
- [ ] **Notarization:** Submit to Apple for Gatekeeper approval
- [ ] **Model bundling:** Verify all models cached in `server/models/hf_cache/`
- [ ] **Virtual environment:** Bundle complete `.venv` with all dependencies
- [ ] **espeak-ng:** Include `espeakng_loader` in venv site-packages
- [ ] **Icons:** Generate proper app icons (`npm run tauri icon path/to/icon.png`)
- [ ] **Test on clean macOS:** Verify no external dependencies required
- [ ] **First run experience:** Add loading screen during model initialization

## Performance Characteristics

**Startup times:**
- Cold start (models not cached): 30-60 seconds
- Warm start (models cached): 10-30 seconds
- With `HF_HUB_OFFLINE=1`: 5-15 seconds

**Memory usage:**
- Tauri wrapper: ~50 MB
- Python server: ~500 MB (with loaded models)
- Next.js client: ~100 MB
- **Total:** ~650 MB

**Voice latency:**
- Same as standalone: <800ms voice-to-voice
- No overhead from native wrapper

## Architecture Decisions

### Why Tauri instead of Electron?

- **Binary size:** Tauri uses system WebView (smaller bundle)
- **Performance:** Lower memory footprint than Chromium
- **Security:** Rust-based with minimal attack surface
- **macOS integration:** Better native feel on Apple Silicon

### Why Auto-Start Server?

- **User experience:** Single click to launch
- **Process management:** Guaranteed cleanup on quit
- **Configuration:** No need to run separate terminal commands

### Why Static Next.js Export?

- **Bundle size:** No need for Next.js server at runtime
- **Simplicity:** Just serve static files from Tauri
- **Performance:** Instant page loads from local filesystem

## Related Documentation

- Parent project: See `/Users/peppi/Dev/localcat/CLAUDE.md` for server details
- Tauri docs: https://tauri.app/
- Client details: See `/Users/peppi/Dev/localcat/client/` for UI components
- Server details: See `/Users/peppi/Dev/localcat/server/` for voice pipeline
