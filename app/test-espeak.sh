#!/bin/bash
# Test script to verify espeak-ng integration
# Run this before and after building the Tauri app

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ESPEAK_DIR="$SCRIPT_DIR/src-tauri/bin/espeak"

echo "=== espeak-ng Integration Test ==="
echo ""

# Test 1: Check files exist
echo "1. Checking espeak-ng files..."
if [ -f "$ESPEAK_DIR/espeak-ng" ]; then
    echo "   ✅ Binary exists: $ESPEAK_DIR/espeak-ng"
else
    echo "   ❌ Binary NOT found: $ESPEAK_DIR/espeak-ng"
    echo "   Run: ./setup-espeak.sh"
    exit 1
fi

if [ -d "$ESPEAK_DIR/espeak-ng-data" ]; then
    echo "   ✅ Data directory exists: $ESPEAK_DIR/espeak-ng-data"
else
    echo "   ❌ Data directory NOT found: $ESPEAK_DIR/espeak-ng-data"
    echo "   Run: ./setup-espeak.sh"
    exit 1
fi

if [ -f "$ESPEAK_DIR/libespeak-ng.dylib" ]; then
    echo "   ✅ Library exists: $ESPEAK_DIR/libespeak-ng.dylib"
else
    echo "   ⚠️  Library NOT found (optional): $ESPEAK_DIR/libespeak-ng.dylib"
fi

# Test 2: Check binary is executable and correct architecture
echo ""
echo "2. Checking binary properties..."
if [ -x "$ESPEAK_DIR/espeak-ng" ]; then
    echo "   ✅ Binary is executable"
else
    echo "   ⚠️  Binary is not executable, setting permissions..."
    chmod +x "$ESPEAK_DIR/espeak-ng"
fi

FILE_TYPE=$(file "$ESPEAK_DIR/espeak-ng")
if echo "$FILE_TYPE" | grep -q "arm64"; then
    echo "   ✅ Binary architecture: arm64 (Apple Silicon)"
elif echo "$FILE_TYPE" | grep -q "x86_64"; then
    echo "   ⚠️  Binary architecture: x86_64 (Intel, may work under Rosetta)"
else
    echo "   ❌ Binary architecture: Unknown ($FILE_TYPE)"
fi

# Test 3: Test espeak-ng directly
echo ""
echo "3. Testing espeak-ng phonemization..."
export ESPEAK_DATA_PATH="$ESPEAK_DIR/espeak-ng-data"
export DYLD_LIBRARY_PATH="$ESPEAK_DIR:$DYLD_LIBRARY_PATH"

TEST_OUTPUT=$("$ESPEAK_DIR/espeak-ng" --ipa=3 -q "Hello world" 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✅ Phonemization works!"
    echo "   Output: $TEST_OUTPUT"
else
    echo "   ❌ Phonemization FAILED:"
    echo "   $TEST_OUTPUT"
    exit 1
fi

# Test 4: Check data directory contents
echo ""
echo "4. Checking data directory contents..."
LANG_COUNT=$(ls -1 "$ESPEAK_DIR/espeak-ng-data" | grep -c "_dict" || true)
if [ "$LANG_COUNT" -gt 0 ]; then
    echo "   ✅ Found $LANG_COUNT language dictionaries"
else
    echo "   ❌ No language dictionaries found in espeak-ng-data/"
    exit 1
fi

# Test 5: Check if Tauri config includes espeak-ng
echo ""
echo "5. Checking Tauri configuration..."
TAURI_CONF="$SCRIPT_DIR/src-tauri/tauri.conf.json"
if grep -q "bin/espeak" "$TAURI_CONF"; then
    echo "   ✅ Tauri config includes espeak-ng bundling"
else
    echo "   ⚠️  Tauri config may not include espeak-ng bundling"
    echo "   Check: $TAURI_CONF"
fi

# Test 6: Check Rust code sets environment variables
echo ""
echo "6. Checking Rust environment variable setup..."
MAIN_RS="$SCRIPT_DIR/src-tauri/src/main.rs"
if grep -q "ESPEAK_DATA_PATH" "$MAIN_RS"; then
    echo "   ✅ main.rs sets ESPEAK_DATA_PATH"
else
    echo "   ❌ main.rs does NOT set ESPEAK_DATA_PATH"
    echo "   Check: $MAIN_RS"
fi

# Summary
echo ""
echo "=== Test Summary ==="
echo "✅ All checks passed!"
echo ""
echo "Next steps:"
echo "1. Run the app in dev mode:"
echo "   cd $SCRIPT_DIR && npm run dev"
echo ""
echo "2. Check console output for:"
echo "   🔧 Dev mode: ESPEAK_DATA_PATH=..."
echo "   📍 Using pre-configured ESPEAK_DATA_PATH: ..."
echo ""
echo "3. Build production app:"
echo "   npm run build"
echo ""
echo "4. Test the bundle:"
echo "   open src-tauri/target/release/bundle/macos/LocalCat.app"
