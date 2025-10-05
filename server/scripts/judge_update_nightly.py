#!/usr/bin/env python3
"""
Judge Nightly Update Orchestrator

Combines gray-zone logging, LLM labeling (optional), and distillation into
an updated distilled judge model. Designed to run as a nightly cron.

Scenarios:
  1) If --labels is provided → train from labels (fastest path)
  2) Else if --dataset is provided → supervised distill using dataset gold
  3) Else → export features (requires labeling before training)

Examples:
  # Train from LM-labeled gray-zone batches
  python -m scripts.judge_update_nightly \
    --log data/judge_grayzone.jsonl --labels data/judge_labels.jsonl \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml \
    --out models/graph_judge.json --keep_rate 0.35

  # Supervised distillation (no labels yet)
  python -m scripts.judge_update_nightly \
    --log data/judge_grayzone.jsonl \
    --dataset tests/data/yaml_eval_l1_en_medium.json \
    --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml \
    --out models/graph_judge.json --auto_calibrate
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--dataset")
    ap.add_argument("--auto_calibrate", action="store_true")
    ap.add_argument("--keep_rate", type=float, default=0.0)
    args = ap.parse_args()

    # Ensure log exists for visibility (but some flows may train off labels only)
    if not Path(args.log).exists():
        print(f"Gray-zone log not found: {args.log}. Proceeding based on provided flags...")

    if args.labels:
        # Train from labels
        cmd = [
            "python", "-m", "scripts.train_graph_judge_from_labels",
            "--labels", args.labels,
            "--yaml", args.yaml,
            "--out", args.out,
        ]
        if args.keep_rate > 0:
            cmd += ["--keep_rate", str(args.keep_rate)]
        run(cmd)
        return

    if args.dataset:
        # Supervised distillation from dataset gold
        cmd = [
            "python", "-m", "scripts.judge_collect_and_distill",
            "--log", args.log,
            "--dataset", args.dataset,
            "--yaml", args.yaml,
            "--out", args.out,
        ]
        if args.auto_calibrate:
            cmd.append("--auto_calibrate")
        run(cmd)
        return

    # Default: export features; manual labeling needed before training
    cmd = [
        "python", "-m", "scripts.judge_collect_and_distill",
        "--log", args.log,
        "--yaml", args.yaml,
        "--out", args.out,
    ]
    run(cmd)
    print("Exported features for labeling. Once labeled, run train_graph_judge_from_labels.")


if __name__ == "__main__":
    main()

