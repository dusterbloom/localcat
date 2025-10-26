#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "🔧 Building macos-stt sidecar (Swift Speech framework)"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "❌ xcrun not found. Please install Xcode Command Line Tools: xcode-select --install"
  exit 1
fi

SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
echo "   Using SDK: $SDK_PATH"

# Choose a conservative minimum target to maximize compatibility
TARGET_TRIPLE="arm64-apple-macos12.0"

# Build with explicit SDK and target to avoid iPhoneOS sysroot issues
xcrun swiftc -O \
  -sdk "$SDK_PATH" \
  -target "$TARGET_TRIPLE" \
  -framework Speech \
  -framework AVFoundation \
  -o macos-stt \
  main.swift 2>&1 | grep -v "using sysroot" || true

if [ -f macos-stt ]; then
  chmod +x macos-stt

  # Codesign with entitlements for microphone access
  echo "🔐 Signing macos-stt with entitlements..."
  if [ -f entitlements.plist ]; then
    codesign --force --sign - --entitlements entitlements.plist macos-stt
    echo "✅ Signed with microphone entitlements"
  else
    echo "⚠️ No entitlements.plist found, signing without entitlements"
    codesign --force --sign - macos-stt
  fi

  echo "✅ Built: $(pwd)/macos-stt"
  file macos-stt || true
  codesign -dv macos-stt 2>/dev/null || echo "No signature info available"
else
  echo "❌ Build failed"
  exit 1
fi
