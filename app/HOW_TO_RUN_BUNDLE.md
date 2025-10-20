# How to Build and Run the Tauri Bundle with Client-Side TTS

## Quick Start

### 1. Build the Bundle

```bash
cd /Users/peppi/Dev/localcat/app
nvm use 22
npm run build
```

This will:
- Build the Next.js client (with client-side TTS support)
- Build the Rust Tauri app
- Create a `.app` bundle and `.dmg` installer

**Build time:** ~2-5 minutes depending on your machine

### 2. Run the Bundle

After the build completes, run the app:

```bash
# Option 1: Run the .app directly
/Users/peppi/Dev/localcat/app/src-tauri/target/release/bundle/macos/LocalCat.app/Contents/MacOS/localcat

# Option 2: Open the .app via macOS
open /Users/peppi/Dev/localcat/app/src-tauri/target/release/bundle/macos/LocalCat.app

# Option 3: Use the shortcut script (if it exists)
./run-bundle.sh
```

### 3. What Happens on First Run

When you run the bundle for the first time:

1. **Server Starts** (Python bot.py)
   - Located at: `../server/bot.py`
   - Starts automatically on port 7860
   - May take 10-30s to initialize

2. **App Opens**
   - Shows loading screen
   - Detects it's running in Tauri
   - **Enables client-side TTS automatically**

3. **TTS Initializes**
   - Shows "Initializing Voice Engine..." screen
   - Downloads Kokoro-82M model (~300MB)
   - **This happens ONCE** (30-60s download)
   - Subsequent runs load from cache (5-10s)

4. **Ready!**
   - Green "✓ WebGPU TTS" badge appears
   - You can now connect and use voice features

## Build Commands Explained

### Full Build (Recommended)
```bash
npm run build
```
Does:
1. `npm run client:build` - Builds Next.js static export
2. `npm run tauri build` - Builds Rust + bundles everything

### Build Components Separately

```bash
# Just build the client
npm run client:build

# Just build Tauri (requires client already built)
npm run tauri build

# Development mode (no bundle, just runs)
npm run dev
```

## File Locations

After building, you'll find:

```
app/src-tauri/target/release/bundle/macos/
├── LocalCat.app           # The macOS application
└── dmg/
    └── LocalCat_1.0.0_aarch64.dmg  # Installer for distribution
```

## Verification Steps

### 1. Check TTS Mode
When the app starts, open the browser dev console (if enabled in Tauri config):

```
🚀 Running in Tauri bundle
✅ Client-side TTS will be enabled
[ClientTTS] Initializing WebGPU TTS...
[ClientTTS] TTS initialized successfully!
```

### 2. Visual Confirmation
Look for the green badge in the top-right corner:
```
✓ WebGPU TTS
```

This confirms client-side TTS is active.

### 3. Test Voice
1. Click "Connect"
2. Allow microphone permissions
3. Speak to the bot
4. Bot should respond using **client-side TTS** (not server TTS)

Check console logs for:
```
[ClientTTS] Synthesizing: Hello! How can I help you?
```

## Troubleshooting

### Build Fails

**Error: `nvm: command not found`**
```bash
# Install nvm first, then:
nvm install 22
nvm use 22
```

**Error: `npm: command not found`**
```bash
# Ensure you're using Node 22
nvm use 22
npm --version  # Should show 10.x.x
```

**Error: `cargo: command not found`**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Bundle Won't Run

**Error: "LocalCat" can't be opened because the developer cannot be verified**
```bash
# Right-click the app → Open → Open again
# Or disable Gatekeeper (not recommended):
sudo spctl --master-disable
```

**Error: Python server doesn't start**
Check that the Python server venv exists:
```bash
ls -la /Users/peppi/Dev/localcat/server/.venv/bin/python3

# If missing, recreate it:
cd /Users/peppi/Dev/localcat/server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### TTS Doesn't Initialize

**Check browser support:**
Open Safari/Chrome and check:
```javascript
if ('gpu' in navigator) {
  console.log('✅ WebGPU supported');
} else {
  console.log('❌ WebGPU not supported');
}
```

**Required:** Chrome 113+, Edge 113+, or Safari 18+

**Model download failed:**
- Check internet connection
- Verify HuggingFace CDN is accessible
- Check available disk space (~500MB needed)

### Server TTS Still Being Used

If you see Python TTS logs instead of `[ClientTTS]`:

1. Check Tauri detection:
```javascript
console.log('__TAURI__' in window);  // Should be true
```

2. Check VoiceApp prop:
```tsx
// Should show: useClientTTS={true} in Tauri
<VoiceApp videoEnabled={false} useClientTTS={isTauri} />
```

## Development vs Production

### Development Mode (`npm run dev`)
- Uses server-side MLX TTS (faster development)
- Python server runs separately
- Hot reload enabled
- `useClientTTS={false}`

### Production Bundle (`npm run build`)
- Uses client-side WebGPU TTS (no Python bundling issues!)
- Python server auto-starts in background
- No hot reload
- `useClientTTS={true}` (auto-detected)

## Performance Comparison

| Metric | First Run | Subsequent Runs |
|--------|-----------|-----------------|
| Bundle startup | 10-30s (server) | 10-30s (server) |
| TTS initialization | 30-60s (download) | 5-10s (cache) |
| **Total** | **40-90s** | **15-40s** |
| Voice latency | ~200-500ms | ~200-500ms |

After the first run, the experience is much faster!

## Distribution

To distribute the app:

### Option 1: Share the .app
```bash
# Compress the .app
cd app/src-tauri/target/release/bundle/macos/
zip -r LocalCat.app.zip LocalCat.app

# Share LocalCat.app.zip
```

**Recipients need:**
- macOS with Apple Silicon
- Server running at `../server/bot.py`
- Internet connection (first run only)

### Option 2: Share the .dmg
```bash
# Find the .dmg
ls app/src-tauri/target/release/bundle/dmg/

# Share LocalCat_1.0.0_aarch64.dmg
```

**Better for distribution:**
- Self-contained installer
- Professional installation experience
- Same requirements as .app

### For Production Distribution

You'll need to:
1. Sign the app with Apple Developer certificate
2. Notarize with Apple
3. Bundle the server code (or deploy separately)
4. Handle auto-updates (optional)

See: [Tauri Distribution Guide](https://tauri.app/v1/guides/distribution/)

## Summary

**To build and run:**
```bash
cd /Users/peppi/Dev/localcat/app
nvm use 22
npm run build
open src-tauri/target/release/bundle/macos/LocalCat.app
```

**What you'll see:**
1. Loading screen
2. "Initializing Voice Engine..." (first run: 30-60s)
3. App ready with green "✓ WebGPU TTS" badge
4. Voice features work with client-side TTS!

**No more:**
- espeak-ng issues
- ONNX bundling nightmares
- Python TTS packaging headaches

Just pure, browser-based TTS magic! 🎉
