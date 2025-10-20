#!/bin/bash
# Monitor production build progress and verify .env bundling

echo "📊 Monitoring production build..."
echo ""

BUILD_LOG="/tmp/production-rebuild.log"
BUNDLE_PATH="src-tauri/target/release/bundle/macos/LocalCat.app/Contents/Resources/_up_/_up_/server"

# Monitor build progress
while true; do
  if [ -f "$BUILD_LOG" ]; then
    # Show last 3 lines of build output
    tail -3 "$BUILD_LOG" 2>/dev/null | grep -v "^$"

    # Check if build completed
    if grep -q "Build complete" "$BUILD_LOG" 2>/dev/null; then
      echo ""
      echo "✅ Build completed!"

      # Verify .env bundling
      if [ -f "$BUNDLE_PATH/.env" ]; then
        ENV_LINES=$(wc -l < "$BUNDLE_PATH/.env" | tr -d ' ')
        echo "✅ .env bundled successfully ($ENV_LINES lines)"

        # Show critical settings
        echo ""
        echo "📋 Configuration Verification:"
        grep "^VOICE_AGENT_STT_ENGINE=" "$BUNDLE_PATH/.env" | sed 's/^/  /'
        grep "^LLM_MODEL=" "$BUNDLE_PATH/.env" | sed 's/^/  /'
        grep "^VOICE_AGENT_TTS_ENGINE=" "$BUNDLE_PATH/.env" | sed 's/^/  /'
      else
        echo "⚠️  .env not found in bundle!"
      fi

      break
    fi

    # Check for errors
    if grep -q "Error\|failed" "$BUILD_LOG" 2>/dev/null; then
      echo "⚠️  Build errors detected, check $BUILD_LOG"
    fi
  fi

  sleep 5
done

echo ""
echo "📍 Bundle location: $(pwd)/$BUNDLE_PATH"
echo "📍 Build log: $BUILD_LOG"
