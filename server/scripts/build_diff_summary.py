#!/usr/bin/env python3
"""
Build a consolidated diff summary for medium and long datasets.

Generates:
  - docs/reports/diff_l1_medium.md (via eval_extraction_diff)
  - docs/reports/diff_l1_long.md   (via eval_extraction_diff)
  - docs/reports/diff_l1_summary.md (this script)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.memory.extractors.yaml_extractor import YAMLExtractor
from core.memory.eval_graph import prf1


def setenv(name: str, val: str) -> None:
    os.environ[name] = val


def metrics(dataset: Path, yaml_path: Path, lang: str, judge_model: Path, caps_off: bool) -> Dict[str, float]:
    import json
    data = json.loads(dataset.read_text())
    # Build extractor once per run
    ext = YAMLExtractor(str(yaml_path))
    ext.prewarm(lang)
    gold_all: List[Tuple[str, str, str]] = []
    raw_all: List[Tuple[str, str, str]] = []
    judged_all: List[Tuple[str, str, str]] = []

    # Density caps toggle per profile
    setenv("YAML_DENSITY_CAPS", "off" if caps_off else "on")

    for ex in data:
        text = ex["text"]
        gold = [(s, r, d) for s, r, d in ex["gold"]]
        lang_ex = ex.get("lang", lang)

        # A: judge off
        setenv("YAML_GRAPH_JUDGE", "off")
        _, trA, _, docA = ext.extract(text, lang_ex)
        trA = ext.refine(text, trA, docA)

        # B: judge on
        setenv("YAML_GRAPH_JUDGE", "on")
        setenv("YAML_GRAPH_JUDGE_MODEL", str(judge_model))
        _, trB, _, docB = ext.extract(text, lang_ex)
        trB = ext.refine(text, trB, docB)

        gold_all.extend(gold)
        raw_all.extend(trA)
        judged_all.extend(trB)

    mA = prf1(raw_all, gold_all)
    mB = prf1(judged_all, gold_all)
    return {"raw_p": mA["precision"], "raw_r": mA["recall"], "raw_f": mA["f1"],
            "judge_p": mB["precision"], "judge_r": mB["recall"], "judge_f": mB["f1"]}


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    server_dir = base / "server"
    docs_dir = base / "docs" / "reports"
    docs_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = server_dir / "archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml"
    judge_model = server_dir / "models/graph_judge.json"
    medium_ds = server_dir / "tests/data/yaml_eval_l1_en_medium.json"
    long_ds = server_dir / "tests/data/yaml_eval_l1_en_long.json"

    # Regenerate per-example diffs
    import sys
    subprocess.run([
        sys.executable, "-m", "scripts.eval_extraction_diff",
        "--dataset", str(medium_ds), "--yaml", str(yaml_path), "--lang", "en",
        "--judge_model", str(judge_model), "--out", str(docs_dir / "diff_l1_medium.md")
    ], check=True, cwd=str(server_dir))
    subprocess.run([
        sys.executable, "-m", "scripts.eval_extraction_diff",
        "--dataset", str(long_ds), "--yaml", str(yaml_path), "--lang", "en",
        "--judge_model", str(judge_model), "--out", str(docs_dir / "diff_l1_long.md")
    ], check=True, cwd=str(server_dir))

    # Aggregate metrics (caps off for medium; caps on for long)
    med = metrics(medium_ds, yaml_path, "en", judge_model, caps_off=True)
    lng = metrics(long_ds, yaml_path, "en", judge_model, caps_off=False)

    md = []
    md.append("# L1 Diff Summary (Judge vs Raw)\n")
    md.append("## Medium (caps off)\n")
    md.append(f"- Raw   PRF: P={med['raw_p']:.3f} R={med['raw_r']:.3f} F1={med['raw_f']:.3f}")
    md.append(f"- Judge PRF: P={med['judge_p']:.3f} R={med['judge_r']:.3f} F1={med['judge_f']:.3f}")
    md.append(f"- Report: diff_l1_medium.md")
    md.append("\n## Long (caps on)\n")
    md.append(f"- Raw   PRF: P={lng['raw_p']:.3f} R={lng['raw_r']:.3f} F1={lng['raw_f']:.3f}")
    md.append(f"- Judge PRF: P={lng['judge_p']:.3f} R={lng['judge_r']:.3f} F1={lng['judge_f']:.3f}")
    md.append(f"- Report: diff_l1_long.md\n")

    (docs_dir / "diff_l1_summary.md").write_text("\n".join(md))
    print("Wrote summary to", docs_dir / "diff_l1_summary.md")


if __name__ == "__main__":
    main()
