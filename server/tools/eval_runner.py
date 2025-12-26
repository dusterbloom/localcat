#!/usr/bin/env python3
"""
Versioned evaluation runner with automatic result archiving.

Usage:
    python server/tools/eval_runner.py \\
        --cases evals/ragas/test_queries.jsonl \\
        --variant baseline \\
        --save

Creates timestamped directory in evals/runs/ with all artifacts.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent  # Go to repo root
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def create_run_directory(variant: str) -> Path:
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    git_sha = get_git_sha()

    run_name = f"{timestamp}_{variant}_{git_sha}"
    # Relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    run_dir = repo_root / "evals" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_metadata(run_dir: Path, variant: str, config: dict):
    """Save run metadata."""
    metadata = {
        "variant": variant,
        "timestamp": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "config": config,
        "user": os.getenv("USER", "unknown")
    }

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "commit.txt").write_text(get_git_sha())


def run_micro_eval(cases: str, variant: str, run_dir: Path) -> dict:
    """Run micro_eval with instrumentation."""
    results_file = run_dir / "results.json"
    timing_file = run_dir / "timing.ndjson"
    trace_file = run_dir / "trace.ndjson"

    # Get variant config from micro_eval
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from micro_eval import VARIANTS
        variant_config = VARIANTS.get(variant, {})
    except ImportError:
        print(f"Warning: Could not import VARIANTS from micro_eval, using empty config")
        variant_config = {}

    # Set environment
    env = os.environ.copy()
    env.update(variant_config)
    env["MEMORY_TRACK_TIMING"] = "true"
    env["MEMORY_INSTRUMENTATION_FILE"] = str(timing_file)
    env["MEMORY_TRACE_FILE"] = str(trace_file)
    env["MEMORY_TRACE_VARIANT"] = variant

    # Run micro_eval
    repo_root = Path(__file__).parent.parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "server" / "tools" / "micro_eval.py"),
        "--cases", cases,
        "--variants", variant,
        "--out", str(results_file)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

    # Save config
    (run_dir / "config.json").write_text(json.dumps(variant_config, indent=2))

    # Load results
    return json.loads(results_file.read_text())


def update_leaderboard():
    """Update leaderboard.md with latest runs."""
    repo_root = Path(__file__).parent.parent.parent
    runs_dir = repo_root / "evals" / "runs"

    if not runs_dir.exists():
        return

    # Collect all runs
    runs = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name in ["archive", ".DS_Store"]:
            continue

        metadata_file = run_dir / "metadata.json"
        results_file = run_dir / "results.json"

        if not (metadata_file.exists() and results_file.exists()):
            continue

        try:
            metadata = json.loads(metadata_file.read_text())
            results = json.loads(results_file.read_text())

            # Extract key metrics
            variants = results.get("variants", {})
            variant_name = metadata["variant"]
            variant_results = variants.get(variant_name, {})

            # Get latency percentiles if available
            latency_p95 = variant_results.get("latency_p95_ms")
            if latency_p95 is None:
                # Fallback to avg if p95 not available
                latency_p95 = variant_results.get("avg_latency_ms", 0.0)

            runs.append({
                "timestamp": metadata["timestamp"],
                "variant": variant_name,
                "git_sha": metadata["git_sha"],
                "precision": variant_results.get("precision_at_k", 0.0),
                "has_gold": variant_results.get("has_gold_rate", 0.0),
                "latency_p95": latency_p95,
                "latency_mean": variant_results.get("avg_latency_ms", 0.0),
                "over_budget": latency_p95 > 100.0 if latency_p95 else False
            })
        except Exception as e:
            print(f"Warning: Failed to process {run_dir.name}: {e}")
            continue

    # Generate markdown table
    leaderboard = ["# Evaluation Leaderboard\n", "\n"]
    leaderboard.append("Last updated: {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    leaderboard.append("| Timestamp | Variant | Git SHA | Precision@K | Has Gold | P95 Latency | Mean Latency | Over Budget |\n")
    leaderboard.append("|-----------|---------|---------|-------------|----------|-------------|--------------|-------------|\n")

    for run in runs[:20]:  # Top 20
        leaderboard.append(
            f"| {run['timestamp'][:16]} | {run['variant']} | {run['git_sha']} | "
            f"{run['precision']:.3f} | {run['has_gold']:.3f} | "
            f"{run['latency_p95']:.1f}ms | {run['latency_mean']:.1f}ms | "
            f"{'❌' if run['over_budget'] else '✅'} |\n"
        )

    leaderboard_file = runs_dir / "leaderboard.md"
    leaderboard_file.write_text("".join(leaderboard))
    print(f"\n📊 Leaderboard updated: {leaderboard_file}")


def main():
    parser = argparse.ArgumentParser(description="Versioned evaluation runner")
    parser.add_argument("--cases", required=True, help="Path to test cases JSONL")
    parser.add_argument("--variant", required=True, help="Variant name (from micro_eval.py)")
    parser.add_argument("--save", action="store_true", help="Save to versioned directory")

    args = parser.parse_args()

    if args.save:
        run_dir = create_run_directory(args.variant)
        print(f"📁 Created run directory: {run_dir}")

        print(f"🏃 Running micro_eval for variant '{args.variant}'...")
        results = run_micro_eval(args.cases, args.variant, run_dir)

        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from micro_eval import VARIANTS
            save_metadata(run_dir, args.variant, VARIANTS.get(args.variant, {}))
        except ImportError:
            save_metadata(run_dir, args.variant, {})

        print(f"✅ Results saved to {run_dir}")
        print(f"📊 Updating leaderboard...")
        update_leaderboard()
        print(f"✅ Done!")
    else:
        # Quick run without saving
        print("⚠️  Running without --save flag (results will not be archived)")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            run_micro_eval(args.cases, args.variant, Path(tmpdir))


if __name__ == "__main__":
    main()
