#!/usr/bin/env python3
"""
normalize_yaml.py

Goal: Convert our legacy, non-standard extraction YAML into strict YAML that is
easier to validate and compile. This script performs conservative text-level
normalizations without changing semantics.

Transforms:
- Convert inline unions like: key: "A" | "B" | "C" -> key_in: ["A", "B", "C"]
  Applies to keys: pos, dep, tag, mood, position
- Convert single-line variant conditionals:
    - if: "COND": "TEMPLATE"
  into
    - when: "COND"\n      template: "TEMPLATE"
- Drop large example blocks (examples:) to keep files compact and valid
  (runtime does not consume examples, keeps YAML simple/parseable).

Usage:
  uv run --project server --directory server -m scripts.normalize_yaml \
      --in archive/2024_12_consolidation/assets/ASI1_proposal.yaml \
      --out archive/2024_12_consolidation/assets/ASI1_proposal.normalized.yaml
"""

from __future__ import annotations

import argparse
import io
import os
import re
from typing import List

import yaml


UNION_KEYS = {"pos", "dep", "tag", "mood", "position", "semantic_relation"}


def normalize_unions(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        # Fast path
        if ' | ' not in line:
            out.append(line)
            continue

        # Handle key-pair unions like: keyA: [..] | keyB: [..]
        m_pair = re.match(r'^(\s*)([A-Za-z_][\w:-]*)\s*:\s*(\[[^\]]*\]|"[^"]*")\s*\|\s*([A-Za-z_][\w:-]*)\s*:\s*(\[[^\]]*\]|"[^"]*")\s*$', line)
        if m_pair:
            indent, key1, val1, key2, val2 = m_pair.groups()
            key1_in = f"{key1.split(':')[0]}_in"
            key2_in = f"{key2.split(':')[0]}_in"
            out.append(f"{indent}{key1_in}: {val1}\n")
            out.append(f"{indent}{key2_in}: {val2}\n")
            continue

        # Detect patterns like: <indent><key>: "A" | "B" | "C"
        m = re.match(r'^(\s*)([A-Za-z_][\w:-]*)\s*:\s*(.+?)\s*$', line)
        if not m:
            out.append(line)
            continue

        indent, key, rhs = m.groups()
        base_key = key.split(":")[0] if ":" in key else key

        if base_key not in UNION_KEYS:
            out.append(line)
            continue

        # Extract quoted tokens
        tokens = re.findall(r'"([^"]*)"', rhs)
        if len(tokens) < 2:
            out.append(line)
            continue

        key_in = f"{base_key}_in"
        joined = ", ".join([f'"{t}"' for t in tokens])
        out.append(f"{indent}{key_in}: [{joined}]\n")
    return out


def normalize_variants(lines: List[str]) -> List[str]:
    out: List[str] = []
    # Pattern for: - if: "COND": "TEMPLATE"
    pat = re.compile(r'^(\s*)-\s*if:\s*"([^"]+)"\s*:\s*"([^"]*)"\s*$')
    for line in lines:
        m = pat.match(line)
        if m:
            indent, cond, tmpl = m.groups()
            out.append(f"{indent}- when: \"{cond}\"\n")
            out.append(f"{indent}  template: \"{tmpl}\"\n")
        else:
            out.append(line)
    return out


def drop_examples_blocks(lines: List[str]) -> List[str]:
    out: List[str] = []
    drop = False
    drop_indent = 0
    for line in lines:
        # Start dropping at lines that begin an examples: section
        if not drop:
            m = re.match(r'^(\s*)examples\s*:\s*$', line)
            if m:
                drop = True
                drop_indent = len(m.group(1))
                continue
            out.append(line)
            continue

        # We are dropping; stop when indentation dedents to <= examples indentation
        indent = len(line) - len(line.lstrip(" "))
        if line.strip() == "":
            # Skip blank lines within examples block
            continue
        if indent <= drop_indent:
            drop = False
            out.append(line)
        else:
            # Skip example content
            continue
    return out


def drop_semantics_blocks(lines: List[str]) -> List[str]:
    out: List[str] = []
    drop = False
    drop_indent = 0
    pat = re.compile(r'^(\s*)([A-Za-z_]+_semantics)\s*:\s*$')
    for line in lines:
        if not drop:
            m = pat.match(line)
            if m:
                drop = True
                drop_indent = len(m.group(1))
                continue
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if line.strip() == "":
            continue
        if indent <= drop_indent:
            drop = False
            out.append(line)
        else:
            continue
    return out


def drop_named_blocks(lines: List[str], key_names: List[str]) -> List[str]:
    out: List[str] = []
    drop = False
    drop_indent = 0
    pat = re.compile(r'^(\s*)(' + "|".join(re.escape(k) for k in key_names) + r')\s*:\s*$')
    for line in lines:
        if not drop:
            m = pat.match(line)
            if m:
                drop = True
                drop_indent = len(m.group(1))
                continue
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if line.strip() == "":
            continue
        if indent <= drop_indent:
            drop = False
            out.append(line)
        else:
            continue
    return out


def normalize_yaml_text(text: str) -> str:
    lines = text.splitlines(True)
    lines = drop_examples_blocks(lines)
    lines = drop_semantics_blocks(lines)
    lines = drop_named_blocks(lines, ["case_mapping"])  # contains prose arrows →
    lines = normalize_variants(lines)
    lines = normalize_unions(lines)
    buf = "".join(lines)
    # Convert comparisons like: key: >0.7 -> key_gt: 0.7 (YAML-safe)
    buf = re.sub(r'^(\s*)([A-Za-z_][\w:]*)\s*:\s*>\s*([0-9]+(?:\.[0-9]+)?)(.*)$',
                 r"\1\2_gt: \3\4",
                 buf, flags=re.MULTILINE)
    buf = re.sub(r'^(\s*)([A-Za-z_][\w:]*)\s*:\s*>=\s*([0-9]+(?:\.[0-9]+)?)(.*)$',
                 r"\1\2_gte: \3\4",
                 buf, flags=re.MULTILINE)
    buf = re.sub(r'^(\s*)-\s*([A-Za-z_][\w:]*)\s*:\s*>\s*([0-9]+(?:\.[0-9]+)?)(.*)$',
                 r"\1- \2_gt: \3\4",
                 buf, flags=re.MULTILINE)
    buf = re.sub(r'^(\s*)-\s*([A-Za-z_][\w:]*)\s*:\s*>=\s*([0-9]+(?:\.[0-9]+)?)(.*)$',
                 r"\1- \2_gte: \3\4",
                 buf, flags=re.MULTILINE)
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input YAML path")
    ap.add_argument("--out", dest="out", required=True, help="Output YAML path")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        raw = f.read()

    normalized = normalize_yaml_text(raw)

    # Validate it is at least parseable YAML
    try:
        _ = yaml.safe_load(normalized)
    except Exception as e:
        print(f"[WARN] Normalized YAML still not strictly parseable: {e}")
        # Still write it out to inspect/improve iteratively
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(normalized)

    print(f"Wrote normalized YAML -> {args.out}")


if __name__ == "__main__":
    main()
