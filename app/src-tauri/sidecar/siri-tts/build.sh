#!/bin/bash
# Simple build script for siri-tts sidecar

SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)

echo "Building siri-tts sidecar..."
echo "Using SDK: $SDK_PATH"

cd "$(dirname "$0")"

swiftc -O \
  -sdk "$SDK_PATH" \
  -target arm64-apple-macos12.0 \
  -o siri-tts \
  main-streaming-hardened.swift \
  2>&1 | grep -v "using sysroot"

if [ -f siri-tts ]; then
  echo "✅ Build successful: $(ls -lh siri-tts | awk '{print $5}')"
  file siri-tts
else
  echo "❌ Build failed"
  exit 1
fi
