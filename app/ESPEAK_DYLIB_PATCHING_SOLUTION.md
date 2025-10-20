# espeak-ng Dylib Patching Solution for Tauri

## Problem

The `libespeak-ng.dylib` bundled in `espeakng-loader` Python package has a **hardcoded CI build path** compiled into the binary:

```
/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data
```

This path cannot be overridden by environment variables because it's compiled into the C binary. When running in a Tauri macOS bundle, this causes TTS to fail with:

```
Error processing file '/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data/phontab': No such file or directory.
```

## Solution: Binary Patching + /tmp Symlink

We solve this by:

1. **Binary-patching the dylib** during Tauri build to replace the hardcoded path with `/tmp/espeak-ng-data` (only 19 bytes vs 90)
2. **Creating a runtime symlink** from `/tmp/espeak-ng-data` to the actual bundled data (no sudo required!)

### Why This Works

- `/tmp` is writable without sudo permissions
- Binary patching replaces the hardcoded string with proper null-byte padding
- Symlinks are automatically cleaned up on reboot (ephemeral)
- No user intervention required!

## Implementation

### 1. Binary Patching Script (`patch-espeak-dylib.py`)

```python
#!/usr/bin/env python3
"""
Patch libespeak-ng.dylib to use /tmp/espeak-ng-data instead of hardcoded CI path.
"""

import sys
import shutil
from pathlib import Path

def patch_dylib(dylib_path: Path):
    # Hardcoded CI build path (90 bytes)
    old_path = b"/Users/runner/work/espeakng-loader/espeakng-loader/espeak-ng/_dynamic/share/espeak-ng-data"

    # New path (19 bytes + 71 null bytes = 90 bytes total)
    new_path = b"/tmp/espeak-ng-data"
    padding_len = len(old_path) - len(new_path)
    new_path_padded = new_path + b"\x00" * padding_len

    print(f"🔧 Patching espeak-ng dylib: {dylib_path}")

    # Create backup
    backup_path = dylib_path.with_suffix(dylib_path.suffix + ".backup")
    shutil.copy2(dylib_path, backup_path)

    # Read, patch, and write binary
    with open(dylib_path, "rb") as f:
        data = f.read()

    patched_data = data.replace(old_path, new_path_padded)

    with open(dylib_path, "wb") as f:
        f.write(patched_data)

    print("✅ Successfully patched dylib")
    return True
```

### 2. Runtime Symlink Creation (`main.rs`)

```rust
/// Create symlink for patched espeak-ng path in libespeak-ng.dylib
/// The patched library looks for: /tmp/espeak-ng-data
/// We symlink this to our bundled espeak-ng-data (no sudo required!)
fn setup_espeak_symlink(espeak_data_path: &PathBuf) -> Result<(), String> {
    let tmp_symlink = PathBuf::from("/tmp/espeak-ng-data");

    // Check if symlink already exists and points to correct location
    if tmp_symlink.exists() {
        if let Ok(target) = fs::read_link(&tmp_symlink) {
            if target == *espeak_data_path {
                println!("🔗 espeak-ng symlink already exists");
                return Ok(());
            } else {
                // Remove incorrect symlink
                let _ = fs::remove_file(&tmp_symlink);
            }
        }
    }

    // Create the symlink in /tmp (no sudo needed!)
    match symlink(espeak_data_path, &tmp_symlink) {
        Ok(_) => {
            println!("✅ Created espeak-ng symlink: /tmp/espeak-ng-data");
            Ok(())
        }
        Err(e) => {
            Err(format!("Failed to create symlink in /tmp: {}", e))
        }
    }
}
```

### 3. Tauri Build Integration

Add a `beforeBuildCommand` in `tauri.conf.json` or create a custom build script:

```json
{
  "build": {
    "beforeBuildCommand": "python3 patch-espeak-dylib.py ../server/.venv/lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib"
  }
}
```

Or integrate into the existing Rust build process using a `build.rs` file.

## Testing

### 1. Test the Patching Script

```bash
cd /Users/peppi/Dev/localcat/app

# Patch the dylib in development venv
python3 patch-espeak-dylib.py ../server/.venv/lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib

# Verify the patch worked
strings ../server/.venv/lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib | grep espeak
# Should show: /tmp/espeak-ng-data
# Should NOT show: /Users/runner/work/...
```

### 2. Create the Symlink

```bash
ln -sf /Users/peppi/Dev/localcat/server/.venv/lib/python3.12/site-packages/espeakng_loader/espeak-ng-data /tmp/espeak-ng-data

# Verify
ls -la /tmp/espeak-ng-data
```

### 3. Test TTS

```bash
cd ../server
source .venv/bin/activate
python3 -c "from kokoro_onnx import Kokoro; k = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin'); audio, sr = k.create('Hello world', voice='af_heart'); print(f'Generated {len(audio)} samples')"
```

This should now work WITHOUT the espeak-ng error!

## Production Build Steps

1. **Patch the venv dylib** before bundling:
   ```bash
   python3 app/patch-espeak-dylib.py server/.venv/lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib
   ```

2. **Bundle normally** with Tauri:
   ```bash
   cd app && npm run build
   ```

3. **Runtime**: The Rust code automatically creates `/tmp/espeak-ng-data` symlink on app startup

## Advantages Over Previous Approaches

| Approach | Requires Sudo | Portable | User Action | Complexity |
|----------|--------------|----------|-------------|------------|
| **Original CI path** | ✅ Yes | ❌ No | Manual setup | High |
| **Binary Patching + /tmp** | ❌ No | ✅ Yes | None | Medium |
| **Rebuild espeak-ng** | ❌ No | ✅ Yes | None | Very High |
| **DYLD_INSERT_LIBRARIES** | Special entitlement | ❌ No | None | Very High |

## Files Modified

- `app/patch-espeak-dylib.py` - Binary patching script (NEW)
- `app/src-tauri/src/main.rs` - Simplified symlink creation
- `server/.venv/lib/python3.12/site-packages/espeakng_loader/libespeak-ng.dylib` - Patched at build time

## Notes

- The patched dylib is **binary-compatible** with the original
- Null-byte padding preserves binary structure
- `/tmp` symlinks are ephemeral and automatically cleaned on reboot
- Works with both ONNX and MLX Kokoro implementations
- No runtime performance impact

## Verification Commands

```bash
# Check if dylib is patched
strings <path-to-dylib> | grep -E "(Users/runner|/tmp/espeak)"

# Check symlink
ls -la /tmp/espeak-ng-data

# Test TTS generation
python3 -c "from kokoro_onnx import Kokoro; k = Kokoro('<model>', '<voices>'); k.create('test', voice='af_heart')"
```

## Troubleshooting

**Problem**: Symlink creation fails
**Solution**: Check `/tmp` is writable: `touch /tmp/test && rm /tmp/test`

**Problem**: Old hardcoded path still appears
**Solution**: Verify patching worked: `strings dylib | grep runner`

**Problem**: TTS still errors
**Solution**: Check both symlink exists AND dylib is patched
