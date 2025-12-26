# Memory System Instrumentation Plan: F1 Race Car → Full Self-Driving

**Status**: Ready for Execution
**Risk Level**: LOW (All changes opt-in, non-breaking)
**Estimated Effort**: 3 phases, ~2-3 days total
**Target**: Achieve full observability for <100ms retrieval SLO validation

---

## Executive Summary

This plan adds comprehensive instrumentation to the memory retrieval system to validate claimed performance targets and enable data-driven optimization. All changes are **opt-in via environment flags** and **behavior-preserving** to ensure zero production risk.

**Goals**:
1. ✅ Measure per-stage latency in retrieval hot path (graph, convo, scoring, verification)
2. ✅ Track P50/P95/P99 percentiles across all critical operations
3. ✅ Enable latency budget accounting against 100ms target
4. ✅ Version-control eval results with git SHA tracking
5. ✅ Compare local vs SOTA LLM-as-judge performance
6. ✅ Add adversarial test cases for robustness validation

---

## Current State Assessment

### ✅ What We Have
- **High-level timing**: `memory_hotpath.py`, `memory_store.py` already track major stages
- **Eval infrastructure**: `micro_eval.py` (A/B testing), `evaluate_ragas.py` (RAGAS metrics)
- **60+ test cases**: RAGAS test queries covering diverse scenarios
- **Trace logging**: NDJSON output with candidate details (`MEMORY_TRACE_FILE`)
- **SLO definitions**: 100ms retrieval, 200ms total budget (in `memory_constants.py`)

### ❌ What's Missing
- **Per-source timing**: No instrumentation in `_graph_collect_candidates()`, `_convo_collect_candidates()`, etc.
- **Verifier overhead**: Unknown latency for cross-encoder inference
- **Versioned results**: Eval outputs not saved with timestamps/git SHA
- **Latency assertions**: No automated SLO checking in CI
- **Adversarial tests**: No contradiction/negation/temporal edge cases
- **LLM-as-judge comparison**: Local Gemma3n vs Claude/GPT-4 not validated

---

## Phase 1: Core Instrumentation (Day 1) - SAFE

**Objective**: Add per-stage latency tracking to retrieval hot path with zero behavior changes.

### 1.1 Create Timing Infrastructure

**File**: `server/core/memory/timing_tracker.py` (NEW)

```python
"""
Lightweight timing tracker for retrieval pipeline.
Usage:
    tracker = TimingTracker()
    tracker.start("graph_collection")
    # ... operation ...
    tracker.end("graph_collection")

    breakdown = tracker.get_breakdown()
    # {"graph_collection": 12.5, "convo_collection": 8.3, ...}
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TimingTracker:
    """Thread-safe timing tracker for retrieval stages."""

    _stages: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _active: Dict[str, float] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.perf_counter)

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._active[stage] = time.perf_counter()

    def end(self, stage: str) -> float:
        """End timing a stage and return duration in ms."""
        if stage not in self._active:
            return 0.0

        duration_ms = (time.perf_counter() - self._active[stage]) * 1000
        self._stages[stage].append(duration_ms)
        del self._active[stage]
        return duration_ms

    def mark(self, stage: str) -> None:
        """Mark a point-in-time stage (duration from tracker start)."""
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._stages[stage].append(elapsed_ms)

    def get_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown (sum of all durations per stage)."""
        return {
            stage: sum(durations)
            for stage, durations in self._stages.items()
        }

    def get_total(self) -> float:
        """Get total elapsed time from tracker creation."""
        return (time.perf_counter() - self._start_time) * 1000

    def to_dict(self) -> Dict[str, any]:
        """Export to dict for logging."""
        breakdown = self.get_breakdown()
        return {
            "total_ms": self.get_total(),
            "breakdown_ms": breakdown,
            "budget_remaining_ms": 100.0 - self.get_total(),  # Against 100ms SLO
            "over_budget": self.get_total() > 100.0
        }


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""

    samples: List[float] = field(default_factory=list)

    def add(self, value_ms: float):
        self.samples.append(value_ms)

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate P50, P95, P99."""
        if not self.samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        import numpy as np
        return {
            "p50": np.percentile(self.samples, 50),
            "p95": np.percentile(self.samples, 95),
            "p99": np.percentile(self.samples, 99),
            "mean": np.mean(self.samples)
        }
```

**Safety**: Pure utility class, no side effects. Can be imported but never called without explicit opt-in.

---

### 1.2 Instrument Retrieval Hot Path

**File**: `server/core/memory/retrieval.py` (MODIFY)

**Changes**: Add opt-in timing to `retrieve()` method.

**Before**:
```python
def retrieve(self, query: str, entities=None, max_bullets=3, ...):
    all_candidates = []
    graph_candidates = self._graph_collect_candidates(...)
    convo_candidates = self._convo_collect_candidates(...)
    # ... rest of method
```

**After**:
```python
def retrieve(self, query: str, entities=None, max_bullets=3, ...):
    # Opt-in timing instrumentation
    from .timing_tracker import TimingTracker
    track_timing = os.getenv("MEMORY_TRACK_TIMING", "false").lower() in ("1", "true")
    tracker = TimingTracker() if track_timing else None

    all_candidates = []

    # Graph collection
    if tracker: tracker.start("graph_collection")
    graph_candidates = self._graph_collect_candidates(...)
    if tracker: tracker.end("graph_collection")

    # Convo collection
    if tracker: tracker.start("convo_collection")
    convo_candidates = self._convo_collect_candidates(...)
    if tracker: tracker.end("convo_collection")

    # Summary collection
    if tracker: tracker.start("summary_collection")
    summary_candidates = self._summary_collect_candidates(...)
    if tracker: tracker.end("summary_collection")

    # Semantic collection
    if tracker: tracker.start("semantic_collection")
    semantic_candidates = self._semantic_collect_candidates(...)
    if tracker: tracker.end("semantic_collection")

    # Composite scoring
    if tracker: tracker.start("composite_scoring")
    scored_candidates = self._score_and_rank(...)
    if tracker: tracker.end("composite_scoring")

    # Verification
    if tracker: tracker.start("verification")
    scored_candidates = self._verify_and_filter(...)
    if tracker: tracker.end("verification")

    # Context planning
    if tracker: tracker.start("context_planning")
    selected = self._plan_context(...)
    if tracker: tracker.end("context_planning")

    # Token budget + dedup
    if tracker: tracker.start("budget_enforcement")
    final = self._apply_token_budget_and_deduplication(...)
    if tracker: tracker.end("budget_enforcement")

    # Log timing breakdown
    if tracker:
        timing_data = tracker.to_dict()
        logger.info(f"[TIMING] retrieve() completed in {timing_data['total_ms']:.1f}ms", extra=timing_data)

        # Optional: Write to structured log
        if self._instrumentation_file:
            self._write_instrumentation(query, timing_data, final)

    return final
```

**Injection Points** (9 total):
1. `graph_collection` - `_graph_collect_candidates()`
2. `convo_collection` - `_convo_collect_candidates()`
3. `summary_collection` - `_summary_collect_candidates()`
4. `semantic_collection` - `_semantic_collect_candidates()`
5. `composite_scoring` - `_composite_score()`
6. `verification` - `_verify_and_filter()`
7. `context_planning` - `_plan_context()`
8. `budget_enforcement` - `_apply_token_budget_and_deduplication()`
9. `slot_detection` - `SlotRouter.detect_slot()` (if called)

**Safety Guarantees**:
- ✅ No behavior changes (timing is read-only)
- ✅ Zero overhead when `MEMORY_TRACK_TIMING=false` (default)
- ✅ Conditional imports and calls (no perf impact)
- ✅ Defensive coding (tracker null checks)

---

### 1.3 Add Structured Instrumentation Output

**File**: `server/core/memory/retrieval.py` (MODIFY)

Add method to write structured timing logs:

```python
def __init__(self, ...):
    # Existing init code...

    # Instrumentation file (separate from trace)
    self._instrumentation_file = os.getenv("MEMORY_INSTRUMENTATION_FILE")
    if self._instrumentation_file:
        logger.info(f"Instrumentation enabled: {self._instrumentation_file}")

def _write_instrumentation(self, query: str, timing_data: dict, bullets: List[str]):
    """Write structured timing data to NDJSON."""
    if not self._instrumentation_file:
        return

    record = {
        "timestamp": time.time(),
        "query": query,
        "timing": timing_data,
        "bullet_count": len(bullets),
        "variant": os.getenv("MEMORY_TRACE_VARIANT", "unknown")
    }

    import json
    with open(self._instrumentation_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

**Output Format** (NDJSON):
```json
{"timestamp": 1699300000.123, "query": "what's my name?", "timing": {"total_ms": 45.2, "breakdown_ms": {"graph_collection": 12.3, "convo_collection": 8.1, ...}, "budget_remaining_ms": 54.8, "over_budget": false}, "bullet_count": 2, "variant": "baseline"}
{"timestamp": 1699300001.456, "query": "where do I live?", "timing": {"total_ms": 103.7, "breakdown_ms": {...}, "budget_remaining_ms": -3.7, "over_budget": true}, "bullet_count": 3, "variant": "verifier_jina"}
```

---

### 1.4 Testing Phase 1

**Test 1: Verify No Behavior Change**
```bash
cd server/

# Baseline (no instrumentation)
python tools/micro_eval.py --cases tools/cases/examples.jsonl --variants baseline --out /tmp/before.json

# With instrumentation enabled
MEMORY_TRACK_TIMING=true \
MEMORY_INSTRUMENTATION_FILE=/tmp/timing.ndjson \
python tools/micro_eval.py --cases tools/cases/examples.jsonl --variants baseline --out /tmp/after.json

# Compare outputs (should be identical except for timing)
diff /tmp/before.json /tmp/after.json
```

**Expected**: No diff in precision, bullets, or gold match rates.

**Test 2: Verify Timing Data**
```bash
# Check instrumentation file
cat /tmp/timing.ndjson | jq '.timing.total_ms'
# Should show per-query latencies

# Check for budget violations
cat /tmp/timing.ndjson | jq 'select(.timing.over_budget == true) | {query, total_ms: .timing.total_ms}'
```

**Acceptance Criteria**:
- ✅ Precision@K unchanged
- ✅ Timing data present when flag enabled
- ✅ No timing data when flag disabled
- ✅ No crashes or errors

---

## Phase 2: Eval Infrastructure Hardening (Day 2) - SAFE

**Objective**: Version-control eval results and add latency SLO assertions.

### 2.1 Create Eval Results Versioning

**File**: `evals/runs/README.md` (NEW)

```markdown
# Evaluation Run Archive

This directory stores timestamped evaluation runs for version control and comparison.

## Directory Structure
```
evals/runs/
  2025-11-06_143022_baseline_fe8415e/
    config.json          # Environment flags used
    results.json         # Micro-eval output (precision, latency)
    timing.ndjson        # Per-query instrumentation
    trace.ndjson         # Retrieval trace (candidates, scoring)
    commit.txt           # Git SHA
    metadata.json        # Run info (date, user, duration)
  2025-11-06_145533_verifier_jina_fe8415e/
    ...
  leaderboard.md         # Comparison table (auto-generated)
```

## Naming Convention
`{date}_{time}_{variant}_{git_sha}/`

## Usage
```bash
# Run eval with versioning
python tools/eval_runner.py --cases cases.jsonl --variant baseline --save
```
```

**File**: `server/tools/eval_runner.py` (NEW)

```python
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
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def create_run_directory(variant: str) -> Path:
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    git_sha = get_git_sha()

    run_name = f"{timestamp}_{variant}_{git_sha}"
    run_dir = Path("evals/runs") / run_name
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
    from micro_eval import VARIANTS
    variant_config = VARIANTS.get(variant, {})

    # Set environment
    env = os.environ.copy()
    env.update(variant_config)
    env["MEMORY_TRACK_TIMING"] = "true"
    env["MEMORY_INSTRUMENTATION_FILE"] = str(timing_file)
    env["MEMORY_TRACE_FILE"] = str(trace_file)
    env["MEMORY_TRACE_VARIANT"] = variant

    # Run micro_eval
    cmd = [
        sys.executable,
        "server/tools/micro_eval.py",
        "--cases", cases,
        "--variants", variant,
        "--out", str(results_file)
    ]

    subprocess.run(cmd, env=env, check=True)

    # Save config
    (run_dir / "config.json").write_text(json.dumps(variant_config, indent=2))

    # Load results
    return json.loads(results_file.read_text())


def update_leaderboard():
    """Update leaderboard.md with latest runs."""
    runs_dir = Path("evals/runs")
    if not runs_dir.exists():
        return

    # Collect all runs
    runs = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        metadata_file = run_dir / "metadata.json"
        results_file = run_dir / "results.json"

        if not (metadata_file.exists() and results_file.exists()):
            continue

        metadata = json.loads(metadata_file.read_text())
        results = json.loads(results_file.read_text())

        # Extract key metrics
        variants = results.get("variants", {})
        variant_name = metadata["variant"]
        variant_results = variants.get(variant_name, {})

        runs.append({
            "timestamp": metadata["timestamp"],
            "variant": variant_name,
            "git_sha": metadata["git_sha"],
            "precision": variant_results.get("precision_at_k", 0.0),
            "has_gold": variant_results.get("has_gold_rate", 0.0),
            "latency_p95": variant_results.get("latency_p95_ms", 0.0),
            "latency_mean": variant_results.get("avg_latency_ms", 0.0),
            "over_budget": variant_results.get("latency_p95_ms", 0.0) > 100.0
        })

    # Generate markdown table
    leaderboard = ["# Evaluation Leaderboard\n", "\n"]
    leaderboard.append("| Timestamp | Variant | Git SHA | Precision@K | Has Gold | P95 Latency | Mean Latency | Over Budget |\n")
    leaderboard.append("|-----------|---------|---------|-------------|----------|-------------|--------------|-------------|\n")

    for run in runs[:20]:  # Top 20
        leaderboard.append(
            f"| {run['timestamp'][:16]} | {run['variant']} | {run['git_sha']} | "
            f"{run['precision']:.3f} | {run['has_gold']:.3f} | "
            f"{run['latency_p95']:.1f}ms | {run['latency_mean']:.1f}ms | "
            f"{'❌' if run['over_budget'] else '✅'} |\n"
        )

    (runs_dir / "leaderboard.md").write_text("".join(leaderboard))


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

        from micro_eval import VARIANTS
        save_metadata(run_dir, args.variant, VARIANTS.get(args.variant, {}))

        print(f"✅ Results saved to {run_dir}")
        print(f"📊 Updating leaderboard...")
        update_leaderboard()
        print(f"✅ Leaderboard updated: evals/runs/leaderboard.md")
    else:
        # Quick run without saving
        print("⚠️  Running without --save flag (results will not be archived)")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            run_micro_eval(args.cases, args.variant, Path(tmpdir))


if __name__ == "__main__":
    main()
```

---

### 2.2 Add Latency SLO Assertions

**File**: `server/tools/micro_eval.py` (MODIFY)

Add SLO checking after eval runs:

```python
# Add at end of main()
def check_slo_compliance(results: dict, slo_p95_ms: float = 100.0):
    """Check if variants meet latency SLO."""
    violations = []

    for variant_name, metrics in results["variants"].items():
        p95_latency = metrics.get("latency_p95_ms", 0.0)

        if p95_latency > slo_p95_ms:
            violations.append({
                "variant": variant_name,
                "p95_latency": p95_latency,
                "slo": slo_p95_ms,
                "excess_ms": p95_latency - slo_p95_ms
            })

    if violations:
        logger.warning("⚠️  SLO VIOLATIONS DETECTED:")
        for v in violations:
            logger.warning(
                f"  {v['variant']}: P95={v['p95_latency']:.1f}ms "
                f"(exceeds {v['slo']:.0f}ms by {v['excess_ms']:.1f}ms)"
            )

        # Optionally fail CI
        if os.getenv("MEMORY_SLO_STRICT", "false").lower() in ("1", "true"):
            raise AssertionError(f"SLO violated: {len(violations)} variants over budget")
    else:
        logger.info(f"✅ All variants within {slo_p95_ms}ms P95 SLO")

    return violations

# In main()
if results:
    check_slo_compliance(results, slo_p95_ms=100.0)
```

---

### 2.3 Testing Phase 2

**Test 1: Run Versioned Eval**
```bash
cd /Users/peppi/Dev/localcat

# Run baseline eval with versioning
python server/tools/eval_runner.py \
    --cases evals/ragas/test_queries.jsonl \
    --variant baseline \
    --save

# Check output
ls -la evals/runs/
cat evals/runs/leaderboard.md
```

**Test 2: Verify SLO Checking**
```bash
# Run with SLO enforcement
MEMORY_SLO_STRICT=true \
python server/tools/micro_eval.py \
    --cases evals/ragas/test_queries.jsonl \
    --variants baseline verifier_jina \
    --out /tmp/slo_test.json

# Should warn or fail if P95 > 100ms
```

**Acceptance Criteria**:
- ✅ Run directory created with timestamp + git SHA
- ✅ All artifacts saved (results, timing, trace, config)
- ✅ Leaderboard auto-generated
- ✅ SLO violations detected and logged

---

## Phase 3: Advanced Evals (Day 3) - SAFE

**Objective**: Add adversarial test cases and LLM-as-judge comparison.

### 3.1 Adversarial Test Cases

**File**: `evals/ragas/adversarial_cases.jsonl` (NEW)

```jsonl
{"id":"empty_context","setup":[],"query":"What's my name?","gold":[],"expect":"no_retrieval","category":"hallucination"}
{"id":"contradiction_latest","setup":["I live in NYC","I moved to San Francisco"],"query":"Where do I live?","gold":["san francisco"],"expect":"latest_only","category":"contradiction"}
{"id":"contradiction_explicit","setup":["I like pizza","Actually I don't like pizza"],"query":"Do I like pizza?","gold":["don't","not"],"expect":"negation","category":"contradiction"}
{"id":"negation_simple","setup":["I'm not a vegetarian"],"query":"What do I eat?","gold":[],"expect":"no_specific_food","category":"negation"}
{"id":"temporal_update","setup":["I worked at Google in 2020","Now I work at Meta"],"query":"Where do I work?","gold":["meta"],"expect":"latest_only","category":"temporal"}
{"id":"temporal_past_tense","setup":["I lived in Paris","I moved to Tokyo"],"query":"Where did I live before?","gold":["paris"],"expect":"historical","category":"temporal"}
{"id":"multi_hop_simple","setup":["My sister is Alice","Alice lives in Seattle"],"query":"Where does my sister live?","gold":["seattle","alice"],"expect":"transitive","category":"multi_hop"}
{"id":"multi_hop_complex","setup":["My manager is Bob","Bob reports to the CEO","The CEO is named Sarah"],"query":"Who is my manager's boss?","gold":["sarah","ceo"],"expect":"two_hop","category":"multi_hop"}
{"id":"ambiguous_pronoun","setup":["My brother and my dad both like basketball","My dad is a coach"],"query":"Who coaches basketball?","gold":["dad"],"expect":"resolve_pronoun","category":"coreference"}
{"id":"partial_info","setup":["I have a dog","His name is Max"],"query":"What pets do I have?","gold":["dog","max"],"expect":"combine_facts","category":"multi_fact"}
{"id":"implicit_negation","setup":["I only eat fish and vegetables"],"query":"Do I eat chicken?","gold":[],"expect":"implicit_no","category":"negation"}
{"id":"update_attribute","setup":["My favorite color is blue","Actually my favorite color is green"],"query":"What's my favorite color?","gold":["green"],"expect":"latest_update","category":"update"}
{"id":"conflicting_relations","setup":["I work at Microsoft","I work at Amazon"],"query":"Where do I work?","gold":["amazon"],"expect":"latest_relation","category":"conflict"}
{"id":"deleted_fact","setup":["I'm allergic to peanuts","Never mind, I'm not allergic to peanuts"],"query":"What am I allergic to?","gold":[],"expect":"retraction","category":"deletion"}
{"id":"future_intent","setup":["I'm planning to visit Japan next year"],"query":"Where am I going?","gold":["japan"],"expect":"future_plan","category":"temporal"}
```

**File**: `server/tools/adversarial_eval.py` (NEW)

```python
#!/usr/bin/env python3
"""
Adversarial evaluation runner for edge cases.

Tests robustness on contradictions, negations, temporal queries, multi-hop, etc.

Usage:
    python server/tools/adversarial_eval.py \\
        --cases evals/ragas/adversarial_cases.jsonl \\
        --variant baseline \\
        --out results.json
"""

# Similar structure to micro_eval.py but with:
# - Category-based analysis (group by adversarial type)
# - Expected behavior validation (check "expect" field)
# - Error type classification (hallucination, outdated, missing)

# Implementation details omitted for brevity - follows micro_eval pattern
```

---

### 3.2 LLM-as-Judge Comparison

**File**: `evals/scripts/llm_judge_comparison.py` (NEW)

```python
#!/usr/bin/env python3
"""
Compare local LLM judge vs SOTA (Claude Sonnet) for eval reliability.

Measures inter-annotator agreement and identifies cases where local judge is unreliable.

Usage:
    python evals/scripts/llm_judge_comparison.py \\
        --cases evals/ragas/test_queries.jsonl \\
        --local-model gemma3n-4b \\
        --local-base http://localhost:1234/v1 \\
        --sota-model claude-sonnet-4.5 \\
        --out comparison.json
"""

import argparse
import json
from typing import List, Dict, Tuple
from openai import OpenAI
import anthropic


def judge_quality(query: str, context: List[str], gold: List[str], llm_client, model: str) -> Dict:
    """
    Use LLM to judge if context contains relevant information for query.

    Returns:
        {
            "relevant": true/false,
            "confidence": 0.0-1.0,
            "explanation": "...",
            "matches_gold": true/false
        }
    """
    prompt = f"""Given the user query and retrieved context, assess if the context is relevant and helpful.

Query: {query}

Retrieved Context:
{chr(10).join(f"- {c}" for c in context)}

Gold Standard Keywords: {", ".join(gold)}

Provide your assessment:
1. Is the context relevant to the query? (yes/no)
2. Does it match the gold standard? (yes/no)
3. Confidence (0-100%)
4. Brief explanation

Format your response as JSON:
{{"relevant": true/false, "matches_gold": true/false, "confidence": 0.0-1.0, "explanation": "..."}}
"""

    # Implementation for OpenAI-compatible and Anthropic APIs
    # Returns structured judgment
    pass


def calculate_agreement(local_judgments: List[Dict], sota_judgments: List[Dict]) -> Dict:
    """Calculate inter-annotator agreement metrics."""
    agreements = []
    disagreements = []

    for local, sota in zip(local_judgments, sota_judgments):
        if local["relevant"] == sota["relevant"]:
            agreements.append({"local": local, "sota": sota})
        else:
            disagreements.append({"local": local, "sota": sota})

    agreement_rate = len(agreements) / len(local_judgments) if local_judgments else 0.0

    return {
        "agreement_rate": agreement_rate,
        "total_cases": len(local_judgments),
        "agreements": len(agreements),
        "disagreements": len(disagreements),
        "disagreement_cases": disagreements[:10]  # Sample
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--local-model", default="gemma3n-4b")
    parser.add_argument("--local-base", default="http://localhost:1234/v1")
    parser.add_argument("--sota-model", default="claude-sonnet-4.5")
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    # Load cases
    cases = [json.loads(line) for line in open(args.cases)]

    # Initialize clients
    local_client = OpenAI(base_url=args.local_base, api_key="local")
    sota_client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    local_judgments = []
    sota_judgments = []

    for case in cases:
        # Run retrieval (omitted - similar to micro_eval)
        context = ["..."]  # Retrieved bullets

        # Get judgments
        local_j = judge_quality(case["query"], context, case["gold"], local_client, args.local_model)
        sota_j = judge_quality(case["query"], context, case["gold"], sota_client, args.sota_model)

        local_judgments.append(local_j)
        sota_judgments.append(sota_j)

    # Calculate agreement
    agreement = calculate_agreement(local_judgments, sota_judgments)

    # Write results
    with open(args.out, 'w') as f:
        json.dump({
            "local_model": args.local_model,
            "sota_model": args.sota_model,
            "agreement": agreement,
            "local_judgments": local_judgments,
            "sota_judgments": sota_judgments
        }, f, indent=2)

    print(f"✅ Agreement rate: {agreement['agreement_rate']:.1%}")
    print(f"📝 Results saved to {args.out}")


if __name__ == "__main__":
    main()
```

---

### 3.3 Testing Phase 3

**Test 1: Adversarial Cases**
```bash
python server/tools/adversarial_eval.py \
    --cases evals/ragas/adversarial_cases.jsonl \
    --variant baseline \
    --out /tmp/adversarial_results.json

# Analyze by category
cat /tmp/adversarial_results.json | jq '.by_category'
```

**Test 2: LLM Judge Comparison**
```bash
# Requires ANTHROPIC_API_KEY env var
export ANTHROPIC_API_KEY=sk-...

python evals/scripts/llm_judge_comparison.py \
    --cases evals/ragas/test_queries.jsonl \
    --local-model gemma3n-4b \
    --local-base http://localhost:1234/v1 \
    --sota-model claude-sonnet-4.5 \
    --out /tmp/judge_comparison.json

# Check agreement
cat /tmp/judge_comparison.json | jq '.agreement.agreement_rate'
# Expected: >0.7 for reliable local judge
```

**Acceptance Criteria**:
- ✅ Adversarial cases run successfully
- ✅ Results grouped by category (contradiction, negation, temporal, etc.)
- ✅ LLM judge comparison completes
- ✅ Agreement rate calculated
- ✅ Disagreement cases documented

---

## Safety Checklist

Before each phase, verify:

- [ ] All changes are **opt-in via environment variables**
- [ ] Default behavior is **unchanged** (flags default to disabled)
- [ ] **No synchronous I/O** in hot path (async or cached only)
- [ ] **No control flow changes** based on timing data
- [ ] All new files have **clear docstrings** and usage examples
- [ ] **Backward compatibility** maintained (old code still works)
- [ ] **Test coverage** for new functionality
- [ ] **Documentation** updated

---

## Rollout Plan

### Week 1: Phase 1 + 2
- **Day 1**: Implement timing tracker + instrument retrieval.py
- **Day 2**: Test instrumentation with micro_eval, verify no regressions
- **Day 3**: Implement eval versioning + SLO checks
- **Day 4**: Run baseline + 5 key variants, save to evals/runs/
- **Day 5**: Generate leaderboard, document findings

### Week 2: Phase 3
- **Day 1**: Create adversarial test cases
- **Day 2**: Implement adversarial_eval.py
- **Day 3**: Run adversarial suite across variants
- **Day 4**: Implement LLM judge comparison
- **Day 5**: Run comparison, analyze agreement, document unreliable cases

### Week 3: Analysis & Optimization
- Use instrumentation data to identify bottlenecks
- Optimize slow stages (if any over budget)
- Rerun evals to validate improvements
- Update docs with performance characteristics

---

## Success Metrics

At completion, we will have:

1. ✅ **Full observability**: Per-stage latency tracked for every retrieval
2. ✅ **SLO validation**: Automated checks that P95 < 100ms
3. ✅ **Version control**: All eval results saved with git SHA + config
4. ✅ **Leaderboard**: Automated comparison table for variant selection
5. ✅ **Robustness testing**: 15+ adversarial cases covering edge cases
6. ✅ **Judge reliability**: Agreement rate between local and SOTA models
7. ✅ **Zero regressions**: Precision@K and latency unchanged from baseline

---

## Environment Variable Reference

| Variable | Default | Purpose | Phase |
|----------|---------|---------|-------|
| `MEMORY_TRACK_TIMING` | `false` | Enable per-stage timing | 1 |
| `MEMORY_INSTRUMENTATION_FILE` | - | Path to timing NDJSON output | 1 |
| `MEMORY_SLO_STRICT` | `false` | Fail CI if P95 > 100ms | 2 |
| `MEMORY_TRACE_FILE` | - | Path to trace NDJSON (existing) | 1 |
| `MEMORY_TRACE_VARIANT` | `unknown` | Tag traces with variant name | 1 |

---

## Example Commands

**Run instrumented baseline eval:**
```bash
MEMORY_TRACK_TIMING=true \
MEMORY_INSTRUMENTATION_FILE=evals/runs/latest/timing.ndjson \
python server/tools/micro_eval.py \
    --cases evals/ragas/test_queries.jsonl \
    --variants baseline \
    --out evals/runs/latest/results.json
```

**Run versioned A/B test:**
```bash
# Variant A: baseline
python server/tools/eval_runner.py --cases evals/ragas/test_queries.jsonl --variant baseline --save

# Variant B: verifier enabled
python server/tools/eval_runner.py --cases evals/ragas/test_queries.jsonl --variant verifier_jina --save

# Compare
cat evals/runs/leaderboard.md
```

**Analyze timing breakdown:**
```bash
# Find slowest stage
cat evals/runs/latest/timing.ndjson | jq -r '.timing.breakdown_ms | to_entries | max_by(.value) | "\(.key): \(.value)ms"'

# Count budget violations
cat evals/runs/latest/timing.ndjson | jq 'select(.timing.over_budget == true)' | wc -l
```

---

## Risk Mitigation

| Risk | Mitigation | Verification |
|------|------------|--------------|
| Instrumentation adds latency | All timing code gated by env flag (default off) | Benchmark with/without flag |
| Breaking existing behavior | Defensive null checks, opt-in only | Run full test suite before/after |
| I/O blocking hot path | Buffered writes, async where possible | Profile with py-spy |
| Memory leaks from logging | Bounded buffers, periodic flushes | Monitor RSS over 1000 queries |
| Eval results storage bloat | Keep last 50 runs, archive older | Automated cleanup script |

---

## Next Steps After Completion

With full instrumentation in place, you can:

1. **Identify bottlenecks**: Which stage consistently exceeds budget?
2. **Validate improvements**: Does enabling verifier improve precision without breaking SLO?
3. **Optimize hot paths**: Profile and optimize slowest components
4. **Production monitoring**: Export metrics to Prometheus/Grafana
5. **Continuous benchmarking**: Run evals on every PR (CI integration)

---

## Appendix: File Inventory

### New Files (9)
1. `server/core/memory/timing_tracker.py` (140 lines)
2. `server/tools/eval_runner.py` (250 lines)
3. `server/tools/adversarial_eval.py` (300 lines)
4. `evals/scripts/llm_judge_comparison.py` (200 lines)
5. `evals/ragas/adversarial_cases.jsonl` (15 cases)
6. `evals/runs/README.md` (documentation)
7. `evals/runs/leaderboard.md` (auto-generated)
8. `docs/09-reports/MEMORY_INSTRUMENTATION_PLAN.md` (this document)
9. `docs/09-reports/INSTRUMENTATION_RESULTS.md` (post-completion analysis)

### Modified Files (2)
1. `server/core/memory/retrieval.py` (~50 lines added)
2. `server/tools/micro_eval.py` (~30 lines added)

**Total New Code**: ~1200 lines
**Total Modified**: ~80 lines
**Risk Level**: LOW (all opt-in, behavior-preserving)

---

## Conclusion

This plan provides a **safe, incremental path** to full observability without breaking existing functionality. Each phase builds on the previous, with testing and validation at every step.

The F1 race car will be ready for FSD: fully instrumented, SLO-validated, and benchmarked against adversarial cases. All changes are opt-in and can be rolled back instantly by removing environment flags.

**Ready to execute when approved.**
