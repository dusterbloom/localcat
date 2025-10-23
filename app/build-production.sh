#!/bin/bash
# Production build script for LocalCat Tauri app
# Builds .app, adds server files, then creates DMG manually to avoid stack overflow
#
# Platform-Intelligent Bundling:
#   - macOS: Uses Kokoro MLX TTS (~160MB models) → ~3.8GB bundle
#   - Windows/Linux: Uses Kokoro TTS (includes models) → ~4.6GB bundle
#
# Optional Environment Variables:
#   LIGHTWEIGHT=1     Skip venv (requires internet on first run)
#   SLIM_VENV=1       Remove unused ML packages (saves ~200-500MB)
#
# Expected Bundle Sizes (macOS):
#   - Parakeet STT: 2.3GB
#   - Smart-turn: 362MB
#   - Speaker recognition (ECAPA-TDNN): 85MB ← CRITICAL for voice enrollment
#   - Kokoro MLX TTS: 160MB ← Better voice quality than Siri
#   - Python venv: ~1.8GB (or ~1.3GB with SLIM_VENV=1)
#   - Server code: ~10MB
#   Total: ~3.8GB (or ~4.6GB with Kokoro ONNX on Windows/Linux)

set -e  # Exit on error

echo "🏗️  Building LocalCat production app..."

# Step 1: Build the Tauri .app bundle
# Note: Client build happens automatically via beforeBuildCommand in tauri.conf.json
echo "📦 Step 1/5: Building Tauri .app bundle..."
echo "  Client will be built automatically by Tauri's beforeBuildCommand"
export NEXT_PUBLIC_SERVER_URL=${NEXT_PUBLIC_SERVER_URL:-"http://127.0.0.1:7860"}
echo "  Using NEXT_PUBLIC_SERVER_URL=$NEXT_PUBLIC_SERVER_URL"

# Step 2: Run Tauri build
echo "📦 Step 2/6: Building Tauri .app..."

# Clean Python bytecode from source to avoid bundling stale caches
echo "  Cleaning Python bytecode from source (../server)..."
find ../server -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find ../server -name "*.pyc" -delete 2>/dev/null || true

# Pre-build sidecars required by tauri.conf.json resources so paths exist at compile time
echo "  Pre-building sidecars required by resources..."
(
  cd src-tauri/sidecar/macos-stt 2>/dev/null || exit 1
  if [ -x "macos-stt" ]; then
    echo "    ✅ macos-stt already built"
  else
    if [ -f "build.sh" ]; then
      bash build.sh || { echo "    ❌ Failed to build macos-stt"; exit 1; }
    else
      echo "    ❌ build.sh missing in sidecar/macos-stt"; exit 1
    fi
  fi
) || { echo "  ❌ Pre-build of macos-stt failed"; exit 1; }

cd src-tauri

# Ensure a default PNG icon exists for Tauri v2 toolchain
if [ ! -f "icons/icon.png" ]; then
  if [ -f "icons/128x128.png" ]; then
    echo "  Preparing default icon.png for Tauri..."
    cp icons/128x128.png icons/icon.png || true
  fi
fi

# Verify frontendDist exists (../../client/out relative to src-tauri)
if [ ! -d "../../client/out" ]; then
  echo "❌ Error: frontend assets not found at ../../client/out"
  echo "   Make sure client/build produced 'out/' (Next.js output: 'export')."
  echo "   You can inspect with: ls -la ../client/out from the repo root."
  exit 1
fi

# Build with explicit --bundles app flag (required for .app creation)
echo "  Running: npx tauri build --bundles app --target aarch64-apple-darwin"
if ! npx tauri build --bundles app --target aarch64-apple-darwin; then
    echo "❌ Error: tauri build failed"
    echo "Debug info:"
    echo "  Current directory: $(pwd)"
    echo "  Binary exists: $(ls -la target/release/localcat 2>/dev/null || echo 'NO')"
    echo "  Bundle exists: $(ls -la target/release/bundle/macos/ 2>/dev/null || echo 'NO')"
    exit 1
fi

cd ..

# Use target-specific path (aarch64-apple-darwin creates subdirectory)
TARGET_DIR="src-tauri/target/aarch64-apple-darwin/release"
BUNDLE_PATH="$TARGET_DIR/bundle/macos/LocalCat.app/Contents/Resources"
TAURI_SERVER_DIR="$BUNDLE_PATH/_up_/_up_/server"

# Check if .app bundle was created
if [ ! -d "$TARGET_DIR/bundle/macos/LocalCat.app" ]; then
    echo "❌ Error: .app bundle not found at $TARGET_DIR/bundle/macos/LocalCat.app"
    echo "Debug info:"
    echo "  Checking what was built..."
    ls -la "$TARGET_DIR/" 2>/dev/null || echo "  No target release directory"
    ls -la "$TARGET_DIR/bundle/" 2>/dev/null || echo "  No bundle directory"

    # Check if at least the binary was built
    if [ -f "$TARGET_DIR/localcat" ]; then
        echo ""
        echo "ℹ️  Binary was built but .app bundle was not created."
        echo "   Binary location: $TARGET_DIR/localcat"
        echo "   You can run it directly, but bundling into .app failed."
    fi
    exit 1
fi

echo "  ✅ .app bundle created successfully"

# Step 3: Copy server files to the .app bundle
echo "📂 Step 3/6: Copying server files and sidecars to bundle..."

# Ensure Tauri server dir exists (Tauri places ../../server under Resources/_up_/_up_/server)
mkdir -p "$TAURI_SERVER_DIR"

# 3.a Build and bundle Siri TTS sidecar (native macOS voices)
SIDEcar_SRC="src-tauri/sidecar/siri-tts"
SIDECAR_BIN="$SIDEcar_SRC/siri-tts"
SIDECAR_DST="src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/sidecar/siri-tts"
SIDECAR_DST_BIN="src-tauri/target/release/sidecar/siri-tts" # for running target/release/localcat directly
echo "  Building Siri TTS sidecar (swift)..."
if [ -x "$SIDECAR_BIN" ]; then
  echo "  ✅ siri-tts already built"
else
  if [ -f "$SIDEcar_SRC/build.sh" ]; then
    bash "$SIDEcar_SRC/build.sh" || { echo "  ❌ Failed to build siri-tts"; exit 1; }
  else
    echo "  ❌ Missing sidecar build script: $SIDEcar_SRC/build.sh"; exit 1
  fi
fi
mkdir -p "$SIDECAR_DST"
cp -f "$SIDECAR_BIN" "$SIDECAR_DST/" || { echo "  ❌ Failed to copy siri-tts into bundle"; exit 1; }
chmod +x "$SIDECAR_DST/siri-tts"
echo "  ✅ Bundled siri-tts → $(/usr/bin/dirname "$SIDECAR_DST/siri-tts")"

# Also place sidecar near compiled binary for direct runs
mkdir -p "$SIDECAR_DST_BIN"
cp -f "$SIDECAR_BIN" "$SIDECAR_DST_BIN/" 2>/dev/null || true
chmod +x "$SIDECAR_DST_BIN/siri-tts" 2>/dev/null || true
echo "  ✅ siri-tts available for direct binary: $SIDECAR_DST_BIN/siri-tts"

## 3.b Build and bundle macOS STT sidecar (native Speech framework)
STT_SIDECAR_SRC="src-tauri/sidecar/macos-stt"
STT_SIDECAR_BIN="$STT_SIDECAR_SRC/macos-stt"
STT_SIDECAR_DST="src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/sidecar/macos-stt"
STT_SIDECAR_DST_BIN="src-tauri/target/release/sidecar/macos-stt" # for running target/release/localcat directly
echo "  Building macos-stt sidecar (swift)..."
if [ -x "$STT_SIDECAR_BIN" ]; then
  echo "  ✅ macos-stt already built"
else
  if [ -f "$STT_SIDECAR_SRC/build.sh" ]; then
    bash "$STT_SIDECAR_SRC/build.sh" || { echo "  ❌ Failed to build macos-stt"; exit 1; }
  else
    echo "  ❌ Missing sidecar build script: $STT_SIDECAR_SRC/build.sh"; exit 1
  fi
fi
mkdir -p "$STT_SIDECAR_DST"
cp -f "$STT_SIDECAR_BIN" "$STT_SIDECAR_DST/" || { echo "  ❌ Failed to copy macos-stt into bundle"; exit 1; }
chmod +x "$STT_SIDECAR_DST/macos-stt"
echo "  ✅ Bundled macos-stt → $(/usr/bin/dirname "$STT_SIDECAR_DST/macos-stt")"

# Also place sidecar near compiled binary for direct runs
mkdir -p "$STT_SIDECAR_DST_BIN"
cp -f "$STT_SIDECAR_BIN" "$STT_SIDECAR_DST_BIN/" 2>/dev/null || true
chmod +x "$STT_SIDECAR_DST_BIN/macos-stt" 2>/dev/null || true
echo "  ✅ macos-stt available for direct binary: $STT_SIDECAR_DST_BIN/macos-stt"

LIGHTWEIGHT=${LIGHTWEIGHT:-0}

# Copy virtual environment unless LIGHTWEIGHT=1
if [ "$LIGHTWEIGHT" = "1" ]; then
  echo "  ⚠️  LIGHTWEIGHT=1 → Skipping .venv copy to minimize bundle size."
  echo "     First-run will require internet to set up dependencies."
else
  echo "  Checking Python venv..."
  if [ -d "../server/.venv" ]; then
    if [ -d "$TAURI_SERVER_DIR/.venv" ]; then
      echo "  ✅ venv already present in bundle server dir. Skipping copy."
    else
      echo "  Copying .venv into bundle server dir (this may take a minute)..."
      cp -R "../server/.venv" "$TAURI_SERVER_DIR/" || {
        echo "  ❌ Failed to copy server/.venv"; exit 1; }
      echo "  ✅ venv copied to: $TAURI_SERVER_DIR/.venv"
      # Prune caches and bytecode to shrink bundle size
      echo "  Pruning venv caches and bytecode..."
      find "$TAURI_SERVER_DIR/.venv" -name "__pycache__" -type d -prune -exec rm -rf {} +
      find "$TAURI_SERVER_DIR/.venv" -name "*.pyc" -delete -o -name "*.pyo" -delete || true
      find "$TAURI_SERVER_DIR/.venv" -type d -name "tests" -prune -exec rm -rf {} + || true
      echo "  ✅ venv pruning complete"

      # CRITICAL: Sign eSpeak dylib to prevent macOS Sequoia SIGKILL
      # This fixes code signature violation when Kokoro MLX loads eSpeak for phonemization
      ESPEAK_DYLIB="$TAURI_SERVER_DIR/.venv/lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib"
      if [ -f "$ESPEAK_DYLIB" ]; then
        echo "  🔐 Signing eSpeak dylib for production bundle..."
        if codesign --force --deep --sign - "$ESPEAK_DYLIB" 2>/dev/null; then
          echo "  ✅ eSpeak dylib signed successfully"
        else
          echo "  ⚠️  Failed to sign eSpeak dylib (continuing, may cause runtime issues)"
        fi
      else
        echo "  ⚠️  eSpeak dylib not found at: $ESPEAK_DYLIB"
        echo "     Kokoro MLX TTS may fail if this library is missing!"
      fi

      # Optional: aggressively slim venv for platform-specific builds
      if [ "${SLIM_VENV:-0}" = "1" ]; then
        echo "  ⚠️  SLIM_VENV=1 → Removing unnecessary ML packages from venv."

        # Detect build target (use same logic as model bundling)
        BUILD_TARGET="macos"
        if [[ "$TAURI_SERVER_DIR" == *"x86_64-pc-windows"* ]]; then
          BUILD_TARGET="windows"
        elif [[ "$TAURI_SERVER_DIR" == *"x86_64-unknown-linux"* ]]; then
          BUILD_TARGET="linux"
        fi

        VENV_SITE=$(python3 - <<'PY'
import sysconfig, sys
print(sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib') or '')
PY
)
        if [ -z "$VENV_SITE" ]; then
          VENV_SITE="$TAURI_SERVER_DIR/.venv/lib/python3.12/site-packages"
        fi
        echo "     Site-packages: $VENV_SITE"
        echo "     Target: $BUILD_TARGET"

        # Platform-specific package removal
        if [ "$BUILD_TARGET" = "macos" ]; then
          # macOS with Kokoro MLX TTS - remove heavy ONNX packages
          # NOTE: Keep espeakng_loader (required by Kokoro MLX for phonemization)
          echo "     Removing ONNX packages (using MLX instead)..."
          for pkg in onnxruntime onnxruntime_gpu; do
            if [ -d "$VENV_SITE/$pkg" ] || ls "$VENV_SITE" 2>/dev/null | grep -q "^$pkg[-_]"; then
              echo "       - Removing $pkg"
              rm -rf "$VENV_SITE/$pkg" "$VENV_SITE/${pkg}-"* 2>/dev/null || true
            fi
          done
        fi

        # All platforms: remove unused heavy ML frameworks
        echo "     Removing unused ML frameworks..."
        for pkg in torch torchvision torchaudio tensorflow jax flax opt_einsum opencv_python cv2 speechbrain; do
          if [ -d "$VENV_SITE/$pkg" ] || ls "$VENV_SITE" 2>/dev/null | grep -q "^$pkg[-_]"; then
            echo "       - Removing $pkg"
            rm -rf "$VENV_SITE/$pkg" "$VENV_SITE/${pkg}-"* 2>/dev/null || true
          fi
        done

        echo "  ✅ venv slimming complete for $BUILD_TARGET"
      fi
    fi
  else
    if [ -d "$TAURI_SERVER_DIR/.venv" ]; then
      echo "  ✅ Bundle already contains a venv in server dir"
    else
      echo "  ⚠️  No venv found at ../../server/.venv and none in bundle server dir."
      echo "     Create it with:"
      echo "       cd server && python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install espeakng-loader && bash fix_shebangs.sh"
    fi
  fi
fi

echo "  ✅ Python files already bundled via tauri.conf.json"

echo "  ✅ All server directories already bundled via tauri.conf.json (config/, core/, tools/, sidecars/, utils/)"

echo "  Cleaning up development artifacts from bundle..."
# Remove .logs directories (development artifacts)
find "$TAURI_SERVER_DIR" -type d -name ".logs" -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Removed .logs directories"

# Remove any Python bytecode from bundled server code to prevent stale imports
echo "  Pruning Python bytecode from bundled server code..."
find "$TAURI_SERVER_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$TAURI_SERVER_DIR" -name "*.pyc" -delete 2>/dev/null || true
echo "  ✅ Removed __pycache__ and *.pyc from bundle"

# Platform-intelligent model bundling
if [ "$LIGHTWEIGHT" != "1" ]; then
  echo "  Copying required models (platform-intelligent bundling)..."

  # Detect build target platform
  BUILD_TARGET="macos"  # Default for this script
  if [[ "$TARGET_DIR" == *"x86_64-pc-windows"* ]]; then
    BUILD_TARGET="windows"
  elif [[ "$TARGET_DIR" == *"x86_64-unknown-linux"* ]]; then
    BUILD_TARGET="linux"
  fi

  echo "    Target platform: $BUILD_TARGET"

  # Always copy HuggingFace cache for STT models (Parakeet, Smart-turn)
  echo "    Copying HuggingFace cache (STT models)..."
  mkdir -p "$TAURI_SERVER_DIR/models/hf_cache/hub"

  # Copy Parakeet STT model (~2.3GB - required for all platforms)
  if [ -d "../server/models/hf_cache/hub/models--mlx-community--parakeet-tdt-0.6b-v3" ]; then
    # Use rsync for reliable copy with extended attributes and proper error checking
    if rsync -a --exclude='.cache' "../server/models/hf_cache/hub/models--mlx-community--parakeet-tdt-0.6b-v3/" \
      "$TAURI_SERVER_DIR/models/hf_cache/hub/models--mlx-community--parakeet-tdt-0.6b-v3/"; then
      echo "      ✅ Parakeet STT (~2.3GB)"
    else
      echo "      ❌ Failed to copy Parakeet STT model"
      exit 1
    fi
  fi

  # Copy Smart-turn model (~362MB - required for all platforms)
  if [ -d "../server/models/hf_cache/hub/models--pipecat-ai--smart-turn-v2" ]; then
    if rsync -a --exclude='.cache' "../server/models/hf_cache/hub/models--pipecat-ai--smart-turn-v2/" \
      "$TAURI_SERVER_DIR/models/hf_cache/hub/models--pipecat-ai--smart-turn-v2/"; then
      echo "      ✅ Smart-turn (~362MB)"
    else
      echo "      ❌ Failed to copy Smart-turn model"
      exit 1
    fi
  fi

  # Copy SpeechBrain speaker recognition model (~85MB - CRITICAL for voice enrollment)
  if [ -d "../server/models/hf_cache/hub/models--speechbrain--spkrec-ecapa-voxceleb" ]; then
    if rsync -a --exclude='.cache' "../server/models/hf_cache/hub/models--speechbrain--spkrec-ecapa-voxceleb/" \
      "$TAURI_SERVER_DIR/models/hf_cache/hub/models--speechbrain--spkrec-ecapa-voxceleb/"; then
      echo "      ✅ SpeechBrain ECAPA-TDNN speaker recognition (~85MB)"
    else
      echo "      ❌ Failed to copy SpeechBrain model"
      exit 1
    fi
  fi

  # Copy emotion recognition model if present (optional feature)
  if [ -d "../server/models/hf_cache/hub/models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP" ]; then
    if rsync -a --exclude='.cache' "../server/models/hf_cache/hub/models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP/" \
      "$TAURI_SERVER_DIR/models/hf_cache/hub/models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP/"; then
      echo "      ✅ SpeechBrain emotion recognition (optional)"
    else
      echo "      ⚠️  Failed to copy emotion recognition model (optional, continuing...)"
    fi
  fi

  # Platform-specific TTS models
  if [ "$BUILD_TARGET" = "macos" ]; then
    echo "    Copying Kokoro MLX TTS models for macOS..."

    # Copy Kokoro MLX model from HF cache (required for in-process TTS)
    if [ -d "../server/models/hf_cache/hub/models--mlx-community--Kokoro-82M-bf16" ]; then
      if rsync -a --exclude='.cache' "../server/models/hf_cache/hub/models--mlx-community--Kokoro-82M-bf16/" \
        "$TAURI_SERVER_DIR/models/hf_cache/hub/models--mlx-community--Kokoro-82M-bf16/"; then
        echo "      ✅ Kokoro MLX (~160MB)"
      else
        echo "      ⚠️  Failed to copy Kokoro MLX model"
      fi
    else
      echo "      ⚠️  Kokoro MLX model not found in cache"
      echo "         Run server once to download: cd ../server && python bot.py"
    fi

    # Also copy prince-canuma variant if present (alternative model)
    if [ -d "../server/models/hf_cache/hub/models--prince-canuma--Kokoro-82M" ]; then
      if rsync -a --exclude='.cache' "../server/models/hf_cache/hub/models--prince-canuma--Kokoro-82M/" \
        "$TAURI_SERVER_DIR/models/hf_cache/hub/models--prince-canuma--Kokoro-82M/"; then
        echo "      ✅ Kokoro MLX (prince-canuma variant)"
      fi
    fi

    echo "      Note: Using Kokoro MLX for better voice quality than Siri"
  else
    echo "    Copying Kokoro TTS (required for $BUILD_TARGET - no Siri available)..."

    # Copy Kokoro ONNX model for cross-platform TTS
    mkdir -p "$TAURI_SERVER_DIR/models/kokoro"
    if [ -f "../server/models/kokoro/kokoro-v1.0.onnx" ]; then
      cp "../server/models/kokoro/kokoro-v1.0.onnx" "$TAURI_SERVER_DIR/models/kokoro/" || true
      echo "      ✅ Kokoro ONNX (~337MB)"
    fi
    if [ -f "../server/models/kokoro/voices-v1.0.bin" ]; then
      cp "../server/models/kokoro/voices-v1.0.bin" "$TAURI_SERVER_DIR/models/kokoro/" || true
    fi

    # Also copy Kokoro from HF cache if present
    if [ -d "../server/models/hf_cache/hub/models--prince-canuma--Kokoro-82M" ]; then
      if rsync -a --exclude='.cache' "../server/models/hf_cache/hub/models--prince-canuma--Kokoro-82M/" \
        "$TAURI_SERVER_DIR/models/hf_cache/hub/models--prince-canuma--Kokoro-82M/"; then
        echo "      ✅ Kokoro HF model (~312MB)"
      else
        echo "      ⚠️  Failed to copy Kokoro HF model (continuing...)"
      fi
    fi
  fi
fi

# Step 3.5: .env is already bundled via tauri.conf.json resources
# No need to copy it post-build (signed .app is read-only anyway)
echo "  ✅ .env bundled via tauri.conf.json (preserving STT=parakeet_batch, TTS=siri_streaming, LLM config)"

# Quick verification summary
if [ -x "$TAURI_SERVER_DIR/.venv/bin/python3" ]; then
  echo "  ✅ venv python found: $TAURI_SERVER_DIR/.venv/bin/python3"
else
  echo "  ⚠️  venv python missing in bundle under _up_/_up_/server/.venv"
fi

if [ -f "$TAURI_SERVER_DIR/.env" ]; then
  ENV_SIZE=$(wc -l < "$TAURI_SERVER_DIR/.env" | tr -d ' ')
  echo "  ✅ .env found in bundle ($ENV_SIZE lines)"
  # Verify critical settings are present
  if grep -q "VOICE_AGENT_STT_ENGINE" "$TAURI_SERVER_DIR/.env"; then
    STT_ENGINE=$(grep "^VOICE_AGENT_STT_ENGINE=" "$TAURI_SERVER_DIR/.env" | cut -d'=' -f2)
    echo "  ✅ STT Engine: $STT_ENGINE"
  fi
  if grep -q "LLM_MODEL" "$TAURI_SERVER_DIR/.env"; then
    LLM_MODEL=$(grep "^LLM_MODEL=" "$TAURI_SERVER_DIR/.env" | cut -d'=' -f2 | cut -d'#' -f1 | tr -d ' ')
    echo "  ✅ LLM Model: $LLM_MODEL"
  fi
else
  echo "  ⚠️  .env missing in bundle ($TAURI_SERVER_DIR/.env)"
fi

# Step 4: Verify the bundle
echo "✅ Step 4/5: Verifying bundle..."
if [ -x "$TAURI_SERVER_DIR/.venv/bin/python3" ]; then
    echo "  ✅ Python interpreter found in server venv"
else
    echo "  ❌ Python interpreter missing in server venv ($TAURI_SERVER_DIR/.venv/bin/python3)"
fi

if [ -f "$TAURI_SERVER_DIR/bot.py" ]; then
    echo "  ✅ bot.py found in server dir"
else
    echo "  ❌ bot.py missing in server dir ($TAURI_SERVER_DIR)"
fi

# Step 5: Create DMG from the complete .app
echo "📦 Step 5/6: Creating DMG installer..."
DMG_DIR="$TARGET_DIR/bundle/dmg"
DMG_PATH="$DMG_DIR/LocalCat_1.0.0_aarch64.dmg"
mkdir -p "$DMG_DIR"

# Remove old DMG if it exists
rm -f "$DMG_PATH"

# Create temporary mount point for DMG creation
TMP_DMG="/tmp/LocalCat_temp_$$.dmg"  # Use PID to avoid conflicts
rm -f "$TMP_DMG"

# Create DMG with the complete .app (calculate size with 15% headroom for filesystem overhead)
echo "  Creating disk image..."

# Calculate required DMG size with overhead
APP_SIZE_KB=$(du -sk "$TARGET_DIR/bundle/macos/LocalCat.app" | cut -f1)
DMG_SIZE_KB=$((APP_SIZE_KB * 150 / 100))  # Add 50% overhead
DMG_SIZE_MB=$((DMG_SIZE_KB / 1024))
echo "    App size: $((APP_SIZE_KB / 1024))MB, DMG size with overhead: ${DMG_SIZE_MB}MB"

# Try creating DMG with explicit size (prevents "no space left" errors)
if hdiutil create -volname "LocalCat_Install" \
    -srcfolder "$TARGET_DIR/bundle/macos/LocalCat.app" \
    -size "${DMG_SIZE_MB}m" \
    -ov -format UDZO -imagekey zlib-level=9 "$TMP_DMG" 2>/dev/null; then
  echo "  ✅ Created compressed DMG"
  mv -f "$TMP_DMG" "$DMG_PATH"
else
  echo "  ⚠️  DMG creation failed (permission issue)"
  echo "  This is usually due to missing Full Disk Access for Terminal/iTerm."
  echo ""
  echo "  To fix:"
  echo "    1. Open System Settings → Privacy & Security → Full Disk Access"
  echo "    2. Add Terminal.app or iTerm.app"
  echo "    3. Restart your terminal and run this script again"
  echo ""
  echo "  Or create DMG manually after build:"
  echo "    hdiutil create -volname LocalCat_Install -srcfolder $TARGET_DIR/bundle/macos/LocalCat.app -ov -format UDZO LocalCat.dmg"
  echo ""
  echo "  ℹ️  The .app bundle is ready and can be distributed without DMG"
  DMG_PATH=""  # Clear DMG path so we don't show it in summary
fi

# Get bundle size
BUNDLE_SIZE=$(du -sh "$TARGET_DIR/bundle/macos/LocalCat.app" | cut -f1)

echo ""
echo "🎉 Build complete!"
echo "📦 Bundle size: $BUNDLE_SIZE"
echo "📍 .app: $(pwd)/$TARGET_DIR/bundle/macos/LocalCat.app"

if [ -n "$DMG_PATH" ] && [ -f "$DMG_PATH" ]; then
  DMG_SIZE=$(du -sh "$DMG_PATH" | cut -f1)
  echo "📦 DMG size: $DMG_SIZE"
  echo "📍 .dmg: $(pwd)/$DMG_PATH"
  echo ""
  echo "To test DMG installation:"
  echo "  1. Open the DMG: open $DMG_PATH"
  echo "  2. Drag LocalCat to Applications"
  echo "  3. Open from Applications folder"
else
  echo ""
  echo "To test the .app bundle:"
  echo "  open $TARGET_DIR/bundle/macos/LocalCat.app"
fi

if [ "$LIGHTWEIGHT" = "1" ]; then
  echo ""
  echo "ℹ️  This is a LIGHTWEIGHT build. On first run, the app will need internet"
  echo "    access to create a Python venv and download required models."
fi
