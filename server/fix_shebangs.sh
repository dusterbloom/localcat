#!/bin/bash
# Fix shebangs in all venv binaries to make them relocatable

cd "$(dirname "$0")"

echo "Fixing shebangs in .venv/bin..."

fixed_count=0
for file in .venv/bin/*; do
    if [ -f "$file" ] && [ -x "$file" ]; then
        first_line=$(head -1 "$file" 2>/dev/null)
        if echo "$first_line" | grep -q "^#!.*python"; then
            # Replace absolute python path with env lookup
            perl -pi -e 's|^#!/.*python.*|#!/usr/bin/env python3|' "$file"
            ((fixed_count++))
        fi
    fi
done

echo "✅ Fixed $fixed_count shebangs"

# Verify a few key files
echo "Verifying..."
echo "pip: $(head -1 .venv/bin/pip)"
echo "pytest: $(head -1 .venv/bin/pytest 2>/dev/null || echo 'N/A')"
echo "python: $(head -1 .venv/bin/python3 2>/dev/null)"
