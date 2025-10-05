#!/usr/bin/env python3
"""
validate_yaml.py

Lightweight structural validator for the extraction YAML. This does not require
jsonschema; it checks key shapes we rely on in the runtime and future codegen.

Checks:
- Top-level optional sections: core_patterns, coreference_system, language_extensions
- core_patterns: list of rules; each rule requires name, pattern, output
- pattern.anchor exists when present
- language_extensions: mapping lang -> list[rules] with name required

Usage:
  uv run --project server --directory server -m scripts.validate_yaml \
      --path archive/2024_12_consolidation/assets/ASI1_proposal.normalized.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import yaml


def err(msg: str) -> None:
    print(f"[ERROR] {msg}")


def validate_core_patterns(sp: List[Dict[str, Any]], relaxed: bool = False) -> int:
    errors = 0
    if not isinstance(sp, list):
        err("core_patterns must be a list")
        return 1
    for i, rule in enumerate(sp):
        if not isinstance(rule, dict):
            err(f"core_patterns[{i}] must be a mapping")
            errors += 1
            continue
        name = rule.get("name") or rule.get("rule_id")
        if not name:
            err(f"core_patterns[{i}] missing 'name' or 'rule_id'")
            errors += 1
        if not relaxed:
            if "pattern" not in rule:
                err(f"core_patterns[{i}] missing 'pattern'")
                errors += 1
            if "output" not in rule:
                err(f"core_patterns[{i}] missing 'output'")
                errors += 1
            pattern = rule.get("pattern", {})
            if not isinstance(pattern, dict):
                err(f"core_patterns[{i}].pattern must be a mapping")
                errors += 1
            else:
                anchor = pattern.get("anchor") or pattern.get("verb") or pattern.get("matrix_verb")
                if not isinstance(anchor, dict):
                    err(f"core_patterns[{i}].pattern missing a primary anchor (anchor/verb/matrix_verb)")
                    errors += 1
    return errors


def validate_language_extensions(le: Dict[str, Any]) -> int:
    errors = 0
    if not isinstance(le, dict):
        err("language_extensions must be a mapping")
        return 1
    for lang, rules in le.items():
        if not isinstance(rules, list):
            err(f"language_extensions.{lang} must be a list of rules")
            errors += 1
            continue
        for i, r in enumerate(rules):
            if not isinstance(r, dict):
                err(f"language_extensions.{lang}[{i}] must be a mapping")
                errors += 1
                continue
            if not r.get("name"):
                err(f"language_extensions.{lang}[{i}] missing 'name'")
                errors += 1
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to YAML spec")
    ap.add_argument("--relaxed", action="store_true", help="Relax checks (allow name-only index rules)")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    total_errors = 0
    if "core_patterns" in spec:
        total_errors += validate_core_patterns(spec.get("core_patterns"), relaxed=args.relaxed)

    if "language_extensions" in spec:
        total_errors += validate_language_extensions(spec.get("language_extensions"))

    # coreference_system is optional in this lightweight pass

    if total_errors:
        print(f"Validation FAILED with {total_errors} error(s)")
        sys.exit(1)
    print("Validation OK")


if __name__ == "__main__":
    main()
