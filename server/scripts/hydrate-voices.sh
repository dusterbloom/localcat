#!/bin/bash
set -euo pipefail

# Resolve voices directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VOICES_DIR="${SCRIPT_DIR}/../models/kokoro-mlx/voices"

if [ ! -d "$VOICES_DIR" ]; then
  echo "[hydrate-voices] Voices directory not found: $VOICES_DIR" >&2
  exit 0
fi

echo "[hydrate-voices] Hydrating symlinked voice files in: $VOICES_DIR"

# Replace symlinks with actual files by copying targets (cp -L follows symlink)
find "$VOICES_DIR" -type l -name "*.pt" -print0 | while IFS= read -r -d '' f; do
  tmp_file="${f}.tmp"
  if cp -L "$f" "$tmp_file" 2>/dev/null; then
    rm -f "$f"
    mv -f "$tmp_file" "$f"
    echo "[hydrate-voices] Replaced symlink with file: $(basename "$f")"
  else
    echo "[hydrate-voices] Warning: failed to copy target for $(basename "$f")" >&2
  fi
done

echo "[hydrate-voices] Done"

