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

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add server root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger


@dataclass
class Case:
    """Adversarial test case."""
    id: str
    setup: List[str]
    query: str
    gold: List[str]
    expect: str
    category: str


@dataclass
class CaseResult:
    """Result for a single adversarial case."""
    case_id: str
    query: str
    category: str
    expect: str
    retrieved_bullets: List[str]
    bullet_count: int
    latency_ms: float
    gold_found: bool
    expectation_met: bool
    error_type: Optional[str] = None


def load_cases(path: str) -> List[Case]:
    """Load adversarial cases from JSONL file."""
    cases = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    case = Case(
                        id=data["id"],
                        setup=data["setup"],
                        query=data["query"],
                        gold=data["gold"],
                        expect=data["expect"],
                        category=data["category"]
                    )
                    cases.append(case)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed case on line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"Adversarial cases file not found: {path}")
        return []

    logger.info(f"Loaded {len(cases)} adversarial cases")
    return cases


def check_expectation(case: Case, retrieved_bullets: List[str]) -> tuple[bool, Optional[str]]:
    """
    Check if retrieved bullets match expected behavior.

    Returns:
        (expectation_met, error_type)
    """
    retrieved_text = " ".join(retrieved_bullets).lower()
    query_text = case.query.lower()

    # Check for gold keywords
    gold_found = any(gold.lower() in retrieved_text for gold in case.gold)

    # Handle different expectation types
    if case.expect == "no_retrieval":
        if retrieved_bullets:
            return False, "hallucination"  # Retrieved something when should be empty
        return True, None

    elif case.expect == "latest_only":
        if not retrieved_bullets:
            return False, "missing"
        # Should retrieve latest fact
        if not gold_found:
            return False, "outdated"
        return True, None

    elif case.expect == "negation":
        if not retrieved_bullets:
            return False, "missing"
        # Should find negation keywords
        negation_words = ["don't", "not", "never", "no", "cannot", "won't"]
        has_negation = any(neg in retrieved_text for neg in negation_words)
        if not has_negation:
            return False, "missing_negation"
        return True, None

    elif case.expect == "latest_update":
        if not retrieved_bullets:
            return False, "missing"
        if not gold_found:
            return False, "outdated"
        return True, None

    elif case.expect == "historical":
        if not retrieved_bullets:
            return False, "missing"
        if not gold_found:
            return False, "wrong_timeframe"
        return True, None

    elif case.expect in ["transitive", "two_hop"]:
        if not retrieved_bullets:
            return False, "missing"
        if not gold_found:
            return False, "incomplete_reasoning"
        return True, None

    elif case.expect == "resolve_pronoun":
        if not retrieved_bullets:
            return False, "missing"
        if not gold_found:
            return False, "ambiguous_pronoun"
        return True, None

    elif case.expect == "combine_facts":
        if not retrieved_bullets:
            return False, "missing"
        # Check if all gold keywords are present
        if not all(gold.lower() in retrieved_text for gold in case.gold):
            return False, "incomplete_combination"
        return True, None

    elif case.expect == "implicit_no":
        if retrieved_bullets:
            return False, "implicit_yes"  # Retrieved when shouldn't
        return True, None

    elif case.expect == "latest_relation":
        if not retrieved_bullets:
            return False, "missing"
        if not gold_found:
            return False, "conflicting_relation"
        return True, None

    elif case.expect == "retraction":
        if retrieved_bullets:
            return False, "retracted_fact"  # Retrieved retracted fact
        return True, None

    elif case.expect == "future_plan":
        if not retrieved_bullets:
            return False, "missing"
        if not gold_found:
            return False, "wrong_temporal"
        return True, None

    elif case.expect in ["no_specific_food", "implicit_no"]:
        if retrieved_bullets:
            # Check if retrieved something inappropriate
            food_words = ["eat", "food", "diet", "vegetarian", "meat"]
            has_food = any(food in retrieved_text for food in food_words)
            if has_food:
                return False, "contradicts_negation"
        return True, None

    # Default: check if gold keywords found
    if case.gold and not gold_found:
        return False, "missing_gold"

    return True, None


def run_case(case: Case, variant_config: Dict[str, str]) -> CaseResult:
    """Run a single adversarial case."""
    # Import HotMemory
    try:
        from core.memory.memory_hotpath import HotMemory
        from core.memory.memory_store import MemoryStore, Paths
    except ImportError as e:
        logger.error(f"Failed to import memory modules: {e}")
        raise

    # Create fresh HotMemory instance
    try:
        store = MemoryStore(Paths(sqlite_path=":memory:", lmdb_dir=None))
        memory = HotMemory(store)
        memory.prewarm("en")
    except Exception as e:
        logger.error(f"Failed to create HotMemory instance: {e}")
        raise

    # Configure variant environment BEFORE processing setup so extraction enhancers run
    prev_env = {}
    for key, value in variant_config.items():
        prev_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Tag utility logs with case id
    prev_env_case = os.environ.get('EXTRACT_CONTEXT_TAG')
    os.environ['EXTRACT_CONTEXT_TAG'] = case.id

    # Set up the case context (with variant flags in effect)
    for i, utterance in enumerate(case.setup):
        try:
            memory.process_turn(utterance, f"test-session-{case.id}", i)
        except Exception as e:
            logger.warning(f"Failed to process setup utterance {i} for case {case.id}: {e}")
            continue

    try:
        # Run retrieval with timing
        start_time = time.perf_counter()
        retrieved_bullets = memory.retrieve_bullets(case.query, read_only=True)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Check expectation
        expectation_met, error_type = check_expectation(case, retrieved_bullets)

        # Check gold keywords
        retrieved_text = " ".join(retrieved_bullets).lower()
        gold_found = any(gold.lower() in retrieved_text for gold in case.gold)

        return CaseResult(
            case_id=case.id,
            query=case.query,
            category=case.category,
            expect=case.expect,
            retrieved_bullets=retrieved_bullets,
            bullet_count=len(retrieved_bullets),
            latency_ms=latency_ms,
            gold_found=gold_found,
            expectation_met=expectation_met,
            error_type=error_type
        )

    except Exception as e:
        logger.error(f"Failed to run case {case.id}: {e}")
        return CaseResult(
            case_id=case.id,
            query=case.query,
            category=case.category,
            expect=case.expect,
            retrieved_bullets=[],
            bullet_count=0,
            latency_ms=0.0,
            gold_found=False,
            expectation_met=False,
            error_type="execution_error"
        )

    finally:
        # Restore environment
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if prev_env_case is None:
            os.environ.pop('EXTRACT_CONTEXT_TAG', None)
        else:
            os.environ['EXTRACT_CONTEXT_TAG'] = prev_env_case


def run_variant(cases: List[Case], variant_name: str, variant_config: Dict[str, str]) -> Dict[str, Any]:
    """Run all cases for a specific variant."""
    logger.info(f"Running {len(cases)} cases for variant '{variant_name}'")

    results = []
    categories = defaultdict(list)

    # Tag utility logs with variant name during this run
    prev_env_variant = os.environ.get('EXTRACT_CONTEXT_VARIANT')
    os.environ['EXTRACT_CONTEXT_VARIANT'] = variant_name

    for case in cases:
        result = run_case(case, variant_config)
        results.append(result)
        categories[result.category].append(result)

    # Restore variant tag
    if prev_env_variant is None:
        os.environ.pop('EXTRACT_CONTEXT_VARIANT', None)
    else:
        os.environ['EXTRACT_CONTEXT_VARIANT'] = prev_env_variant

    # Calculate overall metrics
    total_cases = len(results)
    expectation_met_count = sum(1 for r in results if r.expectation_met)
    gold_found_count = sum(1 for r in results if r.gold_found)

    # Calculate category-wise metrics
    category_metrics = {}
    for category, cat_results in categories.items():
        cat_total = len(cat_results)
        cat_passed = sum(1 for r in cat_results if r.expectation_met)
        cat_gold = sum(1 for r in cat_results if r.gold_found)

        category_metrics[category] = {
            "total": cat_total,
            "pass_rate": cat_passed / cat_total if cat_total > 0 else 0.0,
            "gold_rate": cat_gold / cat_total if cat_total > 0 else 0.0,
            "avg_latency_ms": sum(r.latency_ms for r in cat_results) / cat_total if cat_total > 0 else 0.0
        }

    # Analyze error types
    error_counts = defaultdict(int)
    for result in results:
        if result.error_type:
            error_counts[result.error_type] += 1

    return {
        "variant": variant_name,
        "total_cases": total_cases,
        "expectation_met_rate": expectation_met_count / total_cases if total_cases > 0 else 0.0,
        "gold_found_rate": gold_found_count / total_cases if total_cases > 0 else 0.0,
        "avg_latency_ms": sum(r.latency_ms for r in results) / total_cases if total_cases > 0 else 0.0,
        "max_latency_ms": max(r.latency_ms for r in results) if results else 0.0,
        "by_category": category_metrics,
        "error_analysis": dict(error_counts),
        "results": [r.__dict__ for r in results]
    }


def load_variants() -> Dict[str, Dict[str, str]]:
    """Load variant configurations from micro_eval.py."""
    try:
        from tools.micro_eval import VARIANTS
        return VARIANTS
    except ImportError:
        logger.warning("Could not import VARIANTS from micro_eval.py, using empty dict")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Adversarial evaluation runner")
    parser.add_argument("--cases", required=True, help="Path to adversarial cases JSONL")
    parser.add_argument("--variant", default="baseline", help="Variant name to test")
    parser.add_argument("--out", default="", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Load cases
    cases = load_cases(args.cases)
    if not cases:
        logger.error("No cases loaded, exiting")
        return

    # Load variant configurations
    variants = load_variants()
    variant_config = variants.get(args.variant, {})
    logger.info(f"Using variant '{args.variant}' with config: {variant_config}")

    # Run evaluation
    start_time = time.time()
    results = run_variant(cases, args.variant, variant_config)
    elapsed = time.time() - start_time

    # Add summary
    results["total_time_seconds"] = elapsed

    # Print summary
    print(f"\n🧪 Adversarial Evaluation Results: {args.variant}")
    print(f"   Cases: {results['total_cases']}")
    print(f"   Expectation Met Rate: {results['expectation_met_rate']:.1%}")
    print(f"   Gold Found Rate: {results['gold_found_rate']:.1%}")
    print(f"   Avg Latency: {results['avg_latency_ms']:.1f}ms")
    print(f"   Max Latency: {results['max_latency_ms']:.1f}ms")

    print(f"\n📊 Category Breakdown:")
    for category, metrics in results["by_category"].items():
        print(f"   {category:15}: {metrics['pass_rate']:.1%} pass ({metrics['total']} cases)")

    if results["error_analysis"]:
        print(f"\n❌ Error Analysis:")
        for error_type, count in results["error_analysis"].items():
            print(f"   {error_type}: {count}")

    # Save results
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")

    print(f"\n✅ Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
