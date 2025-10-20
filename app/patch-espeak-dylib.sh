#!/bin/bash
#
# Patch libespeak-ng.dylib to use /tmp/espeak-ng-data instead of hardcoded CI path
# This eliminates the need for sudo to create symlinks in /Users/runner/...
#

set -e

DYLIB_PATH="$1"

if [[ ! -f "$DYLIB_PATH" ]]; then
    echo "Error: dylib not found at: $DYLIB_PATH"
    exit 1
fi

echo "🔧 Patching espeak-ng dylib for Tauri bundle..."

# Hardcoded CI build path (90 characters)
OLD_PATH="/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data"

# New path (18 characters + 72 null bytes for padding to 90 total)
NEW_PATH="/tmp/espeak-ng-data"

# Calculate lengths
OLD_LEN=${#OLD_PATH}
NEW_LEN=${#NEW_PATH}
echo "Old path length: $OLD_LEN"
echo "New path length: $NEW_LEN"

# Create a backup
cp "$DYLIB_PATH" "$DYLIB_PATH.backup"
echo "✅ Created backup: $DYLIB_PATH.backup"

# Use perl to do binary-safe replacement with null padding
# We replace the OLD_PATH with NEW_PATH followed by enough null bytes to match the original length
PADDING_LEN=$((OLD_LEN - NEW_LEN - 1))
export OLD_PATH NEW_PATH PADDING_LEN
perl -pi -e 's|\Q$ENV{OLD_PATH}\E|$ENV{NEW_PATH}\x00" . ("\x00" x $ENV{PADDING_LEN}) . "|ge' "$DYLIB_PATH"

# Verify the patch worked
if strings "$DYLIB_PATH" | grep -q "/tmp/espeak-ng-data"; then
    echo "✅ Successfully patched dylib"
    echo "   Old: $OLD_PATH"
    echo "   New: $NEW_PATH (null-padded to $OLD_LEN bytes)"

    # Verify old path is gone
    if strings "$DYLIB_PATH" | grep -q "/Users/runner/work"; then
        echo "⚠️  Warning: Old path still found in dylib (may have multiple occurrences)"
    else
        echo "✅ Old hardcoded path completely removed"
    fi
else
    echo "❌ Patch failed - reverting"
    mv "$DYLIB_PATH.backup" "$DYLIB_PATH"
    exit 1
fi

echo "🎉 Dylib patching complete!"
