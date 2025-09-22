#!/usr/bin/env python3
"""
Fix all test import issues by updating imports to use the actual module paths.
"""

import os
import re
from pathlib import Path

# Define the import mappings
IMPORT_MAPPINGS = {
    # TTS services that don't exist -> use what's actually available
    r"from tts_native_kokoro import": "from core.tts.kokoro_professional import KokoroProfessionalTTSService as",
    r"from core\.tts\.tts_native_kokoro import": "from core.tts.kokoro_professional import KokoroProfessionalTTSService as",
    r"from tts_mlx_simple import": "# Skipped - module doesn't exist: ",
    r"from core\.tts\.tts_mlx_simple import": "# Skipped - module doesn't exist: ",
    r"from tts_mlx_kokoro import": "from core.tts.kokoro_mlx import KokoroMLXTTSService as",
    r"from tts_mlx_isolated import": "# Skipped - module doesn't exist: ",
    r"from tts_piper_streaming import": "# Skipped - module doesn't exist: ",
    r"from fastapi_streaming_tts import": "# Skipped - module doesn't exist: ",
    r"from tts_mlx_ultra_low_latency import": "# Skipped - module doesn't exist: ",
}

# Also need to fix the sys.path additions
PATH_FIX_PATTERN = r"sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\)\)"
PATH_FIX_REPLACEMENT = "sys.path.insert(0, str(Path(__file__).parent.parent.parent))"

def fix_file(filepath: Path):
    """Fix imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Fix import statements
    for pattern, replacement in IMPORT_MAPPINGS.items():
        content = re.sub(pattern, replacement, content)

    # Fix sys.path additions
    content = re.sub(PATH_FIX_PATTERN, PATH_FIX_REPLACEMENT, content)

    # Also fix if they use append instead of insert
    content = re.sub(
        r"sys\.path\.append\(os\.path\.dirname\(__file__\)\)",
        "sys.path.insert(0, str(Path(__file__).parent.parent.parent))",
        content
    )

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Fixed {filepath.name}")
        return True
    return False

def main():
    """Fix all test files."""
    test_dir = Path(__file__).parent / "tests" / "unit"

    print("🔧 Fixing test imports...")
    print("=" * 50)

    fixed_count = 0
    for test_file in test_dir.glob("*.py"):
        if fix_file(test_file):
            fixed_count += 1

    print("=" * 50)
    print(f"✅ Fixed {fixed_count} files")

if __name__ == "__main__":
    main()