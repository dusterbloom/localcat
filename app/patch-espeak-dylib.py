#!/usr/bin/env python3
"""
Patch libespeak-ng.dylib to use /tmp/espeak-ng-data instead of hardcoded CI path.
This eliminates the need for sudo to create symlinks in /Users/runner/...
"""

import sys
import shutil
from pathlib import Path

def patch_dylib(dylib_path: Path):
    """Binary-patch the dylib to replace hardcoded path."""

    # Hardcoded CI build path (90 bytes)
    old_path = b"/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data"

    # New path (19 bytes + 71 null bytes = 90 bytes total)
    new_path = b"/tmp/espeak-ng-data"
    padding_len = len(old_path) - len(new_path)
    new_path_padded = new_path + b"\x00" * padding_len

    print(f"🔧 Patching espeak-ng dylib: {dylib_path}")
    print(f"   Old path: {old_path.decode('utf-8')} ({len(old_path)} bytes)")
    print(f"   New path: {new_path.decode('utf-8')} (padded to {len(new_path_padded)} bytes)")

    # Read binary
    with open(dylib_path, "rb") as f:
        data = f.read()

    # Check if already patched (idempotent operation)
    if new_path in data and old_path not in data:
        print("✅ Dylib already patched to /tmp/espeak-ng-data - skipping")
        return True

    # Count occurrences of old path
    occurrences = data.count(old_path)
    if occurrences == 0:
        print("⚠️  Hardcoded path not found in dylib (may be pre-patched or different version)")
        # Check if it contains the new path anyway
        if new_path in data:
            print("✅ Dylib contains /tmp/espeak-ng-data path - treating as success")
            return True
        print("❌ Dylib does not contain expected paths")
        return False

    # Create backup before patching
    backup_path = dylib_path.with_suffix(dylib_path.suffix + ".backup")
    shutil.copy2(dylib_path, backup_path)
    print(f"✅ Created backup: {backup_path}")

    print(f"📍 Found {occurrences} occurrence(s) of hardcoded path")

    # Replace all occurrences
    patched_data = data.replace(old_path, new_path_padded)

    # Verify replacement worked
    if patched_data.count(old_path) > 0:
        print("❌ Replacement failed - old path still present")
        return False

    if patched_data.count(new_path) != occurrences:
        print(f"❌ Replacement failed - expected {occurrences} new paths, found {patched_data.count(new_path)}")
        return False

    # Write patched binary
    with open(dylib_path, "wb") as f:
        f.write(patched_data)

    print(f"✅ Successfully patched {occurrences} occurrence(s)")
    print(f"   New path: /tmp/espeak-ng-data")
    print(f"🎉 Dylib patching complete!")

    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: patch-espeak-dylib.py <path-to-libespeak-ng.dylib>")
        sys.exit(1)

    dylib_path = Path(sys.argv[1])

    if not dylib_path.exists():
        print(f"Error: dylib not found at: {dylib_path}")
        sys.exit(1)

    success = patch_dylib(dylib_path)
    sys.exit(0 if success else 1)
