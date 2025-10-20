#!/bin/bash
# Setup script to prepare espeak-ng for Tauri bundling
# Run this script before building the Tauri app

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ESPEAK_DIR="$SCRIPT_DIR/src-tauri/bin/espeak"

echo "Setting up espeak-ng for Tauri bundling..."

# Create directory structure
mkdir -p "$ESPEAK_DIR"

# Check if we have espeakng_loader in the server venv
SERVER_VENV="$SCRIPT_DIR/../server/.venv"
if [ -d "$SERVER_VENV" ]; then
    echo "Found server venv at $SERVER_VENV"

    # Copy espeak-ng binary
    ESPEAK_BIN="$SERVER_VENV/lib/python3.12/site-packages/espeakng_loader/espeak-ng"
    if [ -f "$ESPEAK_BIN" ]; then
        echo "Copying espeak-ng binary..."
        cp "$ESPEAK_BIN" "$ESPEAK_DIR/"
        chmod +x "$ESPEAK_DIR/espeak-ng"
    else
        echo "ERROR: espeak-ng binary not found at $ESPEAK_BIN"
        echo "Make sure espeakng_loader is installed in server venv:"
        echo "  cd ../server && source .venv/bin/activate && pip install espeakng-loader"
        exit 1
    fi

    # Copy espeak-ng-data directory
    ESPEAK_DATA="$SERVER_VENV/lib/python3.12/site-packages/espeakng_loader/espeak-ng-data"
    if [ -d "$ESPEAK_DATA" ]; then
        echo "Copying espeak-ng-data..."
        rsync -a --delete "$ESPEAK_DATA/" "$ESPEAK_DIR/espeak-ng-data/"
    else
        echo "ERROR: espeak-ng-data not found at $ESPEAK_DATA"
        exit 1
    fi

    # Copy libespeak-ng.dylib
    ESPEAK_LIB="$SERVER_VENV/lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib"
    if [ -f "$ESPEAK_LIB" ]; then
        echo "Copying libespeak-ng.dylib..."
        cp "$ESPEAK_LIB" "$ESPEAK_DIR/"
    else
        echo "WARNING: libespeak-ng.dylib not found, may use system version"
    fi
else
    echo "ERROR: Server venv not found at $SERVER_VENV"
    echo "Please create and activate the server venv first"
    exit 1
fi

echo "✅ espeak-ng setup complete!"
echo "Files prepared in: $ESPEAK_DIR"
echo ""
echo "Directory structure:"
tree -L 2 "$ESPEAK_DIR" 2>/dev/null || ls -R "$ESPEAK_DIR"
