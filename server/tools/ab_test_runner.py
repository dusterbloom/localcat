#!/usr/bin/env python3
"""
A/B Test Runner: BERT-NER + MiniLM vs Current Pattern System

Rapid evaluation to measure exactly what value BERT-NER and MiniLM add.
Uses existing adversarial evaluation framework for A/B comparison.

Usage:
    python server/tools/ab_test_runner.py \
        --cases evals/ragas/adversarial_cases.jsonl \
        --baseline baseline \
        --variants bert_only,bert_minilm \
        --out results/ab_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add server root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger
from tools.adversarial_eval import load_cases, run_variant, Case, CaseResult


@dataclass
class ABConfig:
    """Configuration for A/B test variant."""
    name: str
    use_bert: bool = False
    use_minilm: bool = False
    description: str = ""


@dataclass
class ABResult:
    """A/B test result for a variant."""
    config: ABConfig
    case_results: List[CaseResult]
    total_time_ms: float
    variant_data: Dict[str, Any]


@dataclass
class ABComparison:
    """Complete A/B comparison results."""
    baseline_result: ABResult
    variant_results: List[ABResult]
    improvements: Dict[str, Dict[str, float]]
    recommendations: List[str]


class ABTestRunner:
    """A/B test runner for comparing extraction approaches."""

    def __init__(self):
        self.configs = {
            'baseline': ABConfig('baseline', False, False, "Current pattern-based system"),
            'bert_only': ABConfig('bert_only', True, False, "Current + BERT-NER for entities"),
            'bert_minilm': ABConfig('bert_minilm', True, True, "Current + BERT-NER + MiniLM semantic similarity"),
        }

    def run_ab_test(self, cases_file: str, baseline_name: str = 'baseline',
                    variant_names: List[str] = None) -> ABComparison:
        """Run complete A/B test."""

        if variant_names is None:
            variant_names = ['bert_only', 'bert_minilm']

        logger.info(f"🧪 Starting A/B Test: Baseline='{baseline_name}' vs Variants={variant_names}")
        logger.info(f"📝 Loading cases from: {cases_file}")

        # Load test cases
        cases = load_cases(cases_file)
        logger.info(f"✅ Loaded {len(cases)} adversarial cases")

        # Run baseline
        logger.info(f"🏁 Running baseline: {baseline_name}")
        baseline_result = self.run_configuration(cases, self.configs[baseline_name])

        # Run variants
        variant_results = []
        for variant_name in variant_names:
            logger.info(f"🚀 Running variant: {variant_name}")
            variant_result = self.run_configuration(cases, self.configs[variant_name])
            variant_results.append(variant_result)

        # Generate comparison
        logger.info("📊 Analyzing results...")
        comparison = self.compare_results(baseline_result, variant_results)

        # Generate recommendations
        comparison.recommendations = self.generate_recommendations(comparison)

        return comparison

    def run_configuration(self, cases: List[Case], config: ABConfig) -> ABResult:
        """Run a single configuration against all cases."""

        start_time = time.perf_counter()
        case_results = []
        variant_data = {}

        # Set environment variables for this configuration
        old_env = {}
        if config.use_bert:
            old_env['USE_BERT_NER'] = os.environ.get('USE_BERT_NER')
            os.environ['USE_BERT_NER'] = 'true'
            logger.info("🤖 Enabling BERT-NER")

        if config.use_minilm:
            old_env['USE_MINILM'] = os.environ.get('USE_MINILM')
            os.environ['USE_MINILM'] = 'true'
            logger.info("🧠 Enabling MiniLM semantic similarity")

        try:
            # Run the variant using existing adversarial_eval infrastructure
            variant_config = {}
            evaluation_results = run_variant(cases, config.name, variant_config)

            # Convert evaluation results to CaseResult list
            for i, (case, result) in enumerate(zip(cases, evaluation_results['results'])):
                case_result = CaseResult(
                    case_id=case.id,
                    query=case.query,
                    category=case.category,
                    expect=case.expect,
                    retrieved_bullets=result.get('retrieved_bullets', []),
                    bullet_count=result.get('bullet_count', 0),
                    latency_ms=result.get('latency_ms', 0),
                    gold_found=result.get('gold_found', False),
                    expectation_met=result.get('expectation_met', False),
                    error_type=result.get('error_type')
                )
                case_results.append(case_result)

            total_time_ms = (time.perf_counter() - start_time) * 1000

            # Calculate aggregate metrics
            variant_data = self.calculate_metrics(case_results)
            variant_data['description'] = config.description
            variant_data['config'] = {
                'use_bert': config.use_bert,
                'use_minilm': config.use_minilm
            }

            logger.info(f"✅ Configuration '{config.name}' completed in {total_time_ms:.1f}ms")
            logger.info(f"   Success Rate: {variant_data['success_rate']:.1%}")
            logger.info(f"   Avg Latency: {variant_data['avg_latency_ms']:.1f}ms")

        finally:
            # Restore environment
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        return ABResult(config, case_results, total_time_ms, variant_data)

    def calculate_metrics(self, case_results: List[CaseResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics for a set of case results."""

        if not case_results:
            return {}

        success_count = sum(1 for r in case_results if r.expectation_met)
        gold_found_count = sum(1 for r in case_results if r.gold_found)
        latencies = [r.latency_ms for r in case_results if r.latency_ms > 0]

        # Category breakdown
        category_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'gold': 0, 'latencies': []})

        for result in case_results:
            cat = result.category
            category_stats[cat]['total'] += 1
            if result.expectation_met:
                category_stats[cat]['success'] += 1
            if result.gold_found:
                category_stats[cat]['gold'] += 1
            if result.latency_ms > 0:
                category_stats[cat]['latencies'].append(result.latency_ms)

        # Calculate category metrics
        category_breakdown = {}
        for cat, stats in category_stats.items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            gold_rate = stats['gold'] / stats['total'] if stats['total'] > 0 else 0
            avg_latency = sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0

            category_breakdown[cat] = {
                'total': stats['total'],
                'success_rate': success_rate,
                'gold_rate': gold_rate,
                'avg_latency_ms': avg_latency
            }

        return {
            'total_cases': len(case_results),
            'success_rate': success_count / len(case_results),
            'gold_found_rate': gold_found_count / len(case_results),
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'category_breakdown': category_breakdown
        }

    def compare_results(self, baseline: ABResult, variants: List[ABResult]) -> ABComparison:
        """Compare variant results against baseline."""

        improvements = {}

        for variant in variants:
            variant_name = variant.config.name
            baseline_metrics = baseline.variant_data
            variant_metrics = variant.variant_data

            # Calculate improvements
            improvements[variant_name] = {
                'success_rate_improvement': (
                    variant_metrics['success_rate'] - baseline_metrics['success_rate']
                ) * 100,  # Convert to percentage points
                'gold_rate_improvement': (
                    variant_metrics['gold_found_rate'] - baseline_metrics['gold_found_rate']
                ) * 100,
                'latency_overhead_ms': (
                    variant_metrics['avg_latency_ms'] - baseline_metrics['avg_latency_ms']
                ),
                'category_improvements': {}
            }

            # Category-specific improvements
            for cat in baseline_metrics['category_breakdown']:
                if cat in variant_metrics['category_breakdown']:
                    baseline_cat = baseline_metrics['category_breakdown'][cat]
                    variant_cat = variant_metrics['category_breakdown'][cat]

                    improvements[variant_name]['category_improvements'][cat] = {
                        'success_rate_improvement': (
                            variant_cat['success_rate'] - baseline_cat['success_rate']
                        ) * 100,
                        'gold_rate_improvement': (
                            variant_cat['gold_rate'] - baseline_cat['gold_rate']
                        ) * 100,
                        'latency_impact_ms': variant_cat['avg_latency_ms'] - baseline_cat['avg_latency_ms']
                    }

        return ABComparison(baseline, variants, improvements, [])

    def generate_recommendations(self, comparison: ABComparison) -> List[str]:
        """Generate recommendations based on A/B test results."""

        recommendations = []
        baseline = comparison.baseline_result.variant_data

        for variant in comparison.variant_results:
            variant_name = variant.config.name
            variant_metrics = variant.variant_data
            improvements = comparison.improvements[variant_name]

            # Success rate improvement
            success_improvement = improvements['success_rate_improvement']
            latency_overhead = improvements['latency_overhead_ms']

            if success_improvement > 10 and latency_overhead < 10:
                recommendations.append(
                    f"🚀 {variant_name}: STRONG WIN - +{success_improvement:.1f}% success rate, "
                    f"+{latency_overhead:.1f}ms latency"
                )
            elif success_improvement > 5 and latency_overhead < 20:
                recommendations.append(
                    f"✅ {variant_name}: Good improvement - +{success_improvement:.1f}% success rate, "
                    f"+{latency_overhead:.1f}ms latency"
                )
            elif success_improvement > 0:
                recommendations.append(
                    f"⚠️ {variant_name}: Marginal improvement - +{success_improvement:.1f}% success rate, "
                    f"+{latency_overhead:.1f}ms latency (evaluate if worth it)"
                )
            else:
                recommendations.append(
                    f"❌ {variant_name}: No improvement - {success_improvement:.1f}% success rate change, "
                    f"+{latency_overhead:.1f}ms latency overhead"
                )

            # Category-specific insights
            cat_improvements = improvements['category_improvements']
            big_wins = [cat for cat, imp in cat_improvements.items() if imp['success_rate_improvement'] > 20]
            if big_wins:
                recommendations.append(
                    f"🎯 {variant_name}: Major improvements in categories: {', '.join(big_wins)}"
                )

        return recommendations

    def save_results(self, comparison: ABComparison, output_file: str):
        """Save A/B test results to file."""

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON serialization
        results_data = {
            'baseline': {
                'config': asdict(comparison.baseline_result.config),
                'metrics': comparison.baseline_result.variant_data,
                'cases': [r.__dict__ for r in comparison.baseline_result.case_results],
                'total_time_ms': comparison.baseline_result.total_time_ms
            },
            'variants': [
                {
                    'config': asdict(variant.config),
                    'metrics': variant.variant_data,
                    'cases': [r.__dict__ for r in variant.case_results],
                    'total_time_ms': variant.total_time_ms
                }
                for variant in comparison.variant_results
            ],
            'improvements': comparison.improvements,
            'recommendations': comparison.recommendations,
            'timestamp': time.time()
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"💾 Results saved to: {output_path}")

    def print_summary(self, comparison: ABComparison):
        """Print A/B test summary."""

        print("\n" + "="*80)
        print("🧪 A/B TEST RESULTS SUMMARY")
        print("="*80)

        baseline = comparison.baseline_result.variant_data
        print(f"\nBASELINE ({comparison.baseline_result.config.name}):")
        print(f"   Success Rate: {baseline['success_rate']:.1%}")
        print(f"   Gold Found Rate: {baseline['gold_found_rate']:.1%}")
        print(f"   Avg Latency: {baseline['avg_latency_ms']:.1f}ms")

        print(f"\nVARIANTS:")
        for variant in comparison.variant_results:
            variant_name = variant.config.name
            variant_metrics = variant.variant_data
            improvements = comparison.improvements[variant_name]

            print(f"\n📊 {variant_name.upper()}:")
            print(f"   Success Rate: {variant_metrics['success_rate']:.1%} "
                  f"({improvements['success_rate_improvement']:+.1f}%)")
            print(f"   Gold Found Rate: {variant_metrics['gold_found_rate']:.1%} "
                  f"({improvements['gold_rate_improvement']:+.1f}%)")
            print(f"   Avg Latency: {variant_metrics['avg_latency_ms']:.1f}ms "
                  f"({improvements['latency_overhead_ms']:+.1f}ms)")

            # Top 3 category improvements
            cat_improvements = improvements['category_improvements']
            top_improvements = sorted(
                [(cat, imp['success_rate_improvement']) for cat, imp in cat_improvements.items()],
                key=lambda x: x[1], reverse=True
            )[:3]

            if top_improvements:
                print(f"   Top improvements: {', '.join([f'{cat}+{imp:.0f}%' for cat, imp in top_improvements])}")

        print(f"\nRECOMMENDATIONS:")
        for rec in comparison.recommendations:
            print(f"   {rec}")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="A/B Test Runner for BERT-NER + MiniLM")
    parser.add_argument("--cases", required=True, help="Adversarial cases file")
    parser.add_argument("--baseline", default="baseline", help="Baseline configuration")
    parser.add_argument("--variants", default="bert_only,bert_minilm", help="Comma-separated variant names")
    parser.add_argument("--out", default="results/ab_comparison.json", help="Output results file")

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    try:
        # Create A/B test runner
        runner = ABTestRunner()

        # Parse variants
        variant_names = [v.strip() for v in args.variants.split(',')]

        # Run A/B test
        comparison = runner.run_ab_test(args.cases, args.baseline, variant_names)

        # Save results
        runner.save_results(comparison, args.out)

        # Print summary
        runner.print_summary(comparison)

        logger.info("✅ A/B test completed successfully!")

    except KeyboardInterrupt:
        logger.info("⏹️ A/B test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 A/B test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
