#!/bin/bash
# Production build script for LocalCat Tauri app
# Builds .app, adds server files, then creates DMG manually to avoid stack overflow

set -e  # Exit on error

echo "🏗️  Building LocalCat production app..."

# Step 1: Build the Tauri .app bundle
# Note: Client build happens automatically via beforeBuildCommand in tauri.conf.json
echo "📦 Step 1/5: Building Tauri .app bundle..."
echo "  Client will be built automatically by Tauri's beforeBuildCommand"
export NEXT_PUBLIC_SERVER_URL=${NEXT_PUBLIC_SERVER_URL:-"http://127.0.0.1:7860"}
echo "  Using NEXT_PUBLIC_SERVER_URL=$NEXT_PUBLIC_SERVER_URL"

# Step 2: Run Tauri build
echo "📦 Step 2/5: Building Tauri .app..."
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
echo "📂 Step 3/5: Copying server files and sidecars to bundle..."

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

      # Optional: aggressively slim venv for Siri-only builds
      if [ "${SLIM_VENV:-0}" = "1" ]; then
        echo "  ⚠️  SLIM_VENV=1 → Removing heavy ML packages from venv (Siri-only)."
        VENV_SITE=$(python3 - <<'PY'
import sysconfig, sys
print(sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib') or '')
PY
)
        if [ -z "$VENV_SITE" ]; then
          VENV_SITE="$TAURI_SERVER_DIR/.venv/lib/python3.12/site-packages"
        fi
        echo "     Site-packages: $VENV_SITE"
        for pkg in torch torchvision torchaudio onnxruntime onnxruntime_gpu transformers spacy sentence_transformers mlx_lm mlx_audio tensorflow jax flax opt_einsum flax_core opencv_python cv2 speechbrain parselmouth; do
          if [ -d "$VENV_SITE/$pkg" ] || ls "$VENV_SITE" | grep -q "^$pkg[-_]"; then
            echo "     - Removing $pkg"
            rm -rf "$VENV_SITE/$pkg" "$VENV_SITE/${pkg}-"* 2>/dev/null || true
          fi
        done
        echo "  ✅ venv slimming complete (be sure TTS=\"siri_streaming\" in .env)"
      fi
    fi
  else
    if [ -d "$TAURI_SERVER_DIR/.venv" ]; then
      echo "  ✅ Bundle already contains a venv in server dir"
    else
      echo "  ⚠️  No venv found at ../server/.venv and none in bundle server dir."
      echo "     Create it with:"
      echo "       cd server && python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install espeakng-loader && bash fix_shebangs.sh"
    fi
  fi
fi

echo "  Copying Python files to server dir..."
cp ../server/*.py "$TAURI_SERVER_DIR/"

echo "  Copying directories to server dir..."
cp -r ../server/config "$TAURI_SERVER_DIR/" 2>/dev/null || true
cp -r ../server/pipecat "$TAURI_SERVER_DIR/" 2>/dev/null || true
cp -r ../server/core "$TAURI_SERVER_DIR/" 2>/dev/null || true
cp -r ../server/tools "$TAURI_SERVER_DIR/" 2>/dev/null || true
cp -r ../server/sidecars "$TAURI_SERVER_DIR/" 2>/dev/null || true

# Copy only Kokoro ONNX model assets (omit HF cache, Kokoro-MLX, etc.)
if [ "$LIGHTWEIGHT" != "1" ]; then
  echo "  Copying Kokoro ONNX model (saves hundreds of MB vs full models dir)..."
  mkdir -p "$TAURI_SERVER_DIR/models/kokoro"
  if [ -f "../server/models/kokoro/kokoro-v1.0.onnx" ]; then
    cp "../server/models/kokoro/kokoro-v1.0.onnx" "$TAURI_SERVER_DIR/models/kokoro/" || true
  fi
  if [ -f "../server/models/kokoro/voices-v1.0.bin" ]; then
    cp "../server/models/kokoro/voices-v1.0.bin" "$TAURI_SERVER_DIR/models/kokoro/" || true
  fi
fi

# Hydrate Kokoro MLX voices from HF cache into bundled voices directory
# Remove Kokoro-MLX hydration (Siri + ONNX only build)

# Step 3.5: Copy complete .env configuration to bundle (preserves all settings)
echo "  Copying .env configuration to server dir..."
if [ -f "../server/.env" ]; then
  cp ../server/.env "$TAURI_SERVER_DIR/" || {
    echo "  ❌ Failed to copy .env"
    exit 1
  }
  echo "  ✅ .env copied successfully (preserving STT=parakeet_batch, TTS=siri_streaming, LLM config)"
  # Also copy .env to the compiled-binary resource dir so running `target/release/localcat` picks it up
  RELEASE_RES_DIR="src-tauri/target/release/_up_/_up_/server"
  if [ -d "$RELEASE_RES_DIR" ]; then
    cp ../server/.env "$RELEASE_RES_DIR/" 2>/dev/null || true
    echo "  ✅ .env also placed in $RELEASE_RES_DIR for direct binary runs"
  fi
else
  echo "  ❌ No .env found at ../server/.env - cannot build without configuration"
  exit 1
fi

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
echo "📦 Step 5/5: Creating DMG installer..."
DMG_DIR="$TARGET_DIR/bundle/dmg"
DMG_PATH="$DMG_DIR/LocalCat_1.0.0_aarch64.dmg"
mkdir -p "$DMG_DIR"

# Remove old DMG if it exists
rm -f "$DMG_PATH"

# Create temporary mount point for DMG creation
TMP_DMG="/tmp/LocalCat_temp.dmg"
rm -f "$TMP_DMG"

# Create DMG with the complete .app (try LZFSE first, fallback to zlib)
echo "  Creating disk image..."
if hdiutil create -volname "LocalCat" \
    -srcfolder "$TARGET_DIR/bundle/macos/LocalCat.app" \
    -ov -format ULFO -imagekey lzfse-level=19 "$TMP_DMG"; then
  echo "  ✅ Created LZFSE-compressed DMG"
else
  echo "  ⚠️  LZFSE not available; falling back to zlib (UDZO)"
  hdiutil create -volname "LocalCat" \
    -srcfolder "$TARGET_DIR/bundle/macos/LocalCat.app" \
    -ov -format UDZO -imagekey zlib-level=9 "$TMP_DMG"
fi

# Move to final DMG path
echo "  Finalizing DMG..."
mv -f "$TMP_DMG" "$DMG_PATH"

# Get bundle size
BUNDLE_SIZE=$(du -sh "$TARGET_DIR/bundle/macos/LocalCat.app" | cut -f1)
DMG_SIZE=$(du -sh "$DMG_PATH" | cut -f1)

echo ""
echo "🎉 Build complete!"
echo "📦 Bundle size: $BUNDLE_SIZE"
echo "📦 DMG size: $DMG_SIZE"
echo "📍 .app: $(pwd)/$TARGET_DIR/bundle/macos/LocalCat.app"
echo "📍 .dmg: $(pwd)/$DMG_PATH"
echo ""
echo "To test DMG installation:"
echo "  1. Open the DMG: open $DMG_PATH"
echo "  2. Drag LocalCat to Applications"
echo "  3. Open from Applications folder"

if [ "$LIGHTWEIGHT" = "1" ]; then
  echo ""
  echo "ℹ️  This is a LIGHTWEIGHT build. On first run, the app will need internet"
  echo "    access to create a Python venv and download required models."
fi
