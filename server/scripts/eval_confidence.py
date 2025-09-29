#!/usr/bin/env python3
"""
Confidence Strategy Evaluation Script

Compare different confidence scoring strategies on real conversation data.

Usage:
    # Evaluate on existing database
    python scripts/eval_confidence.py --db data/memory.db

    # Compare specific strategies
    python scripts/eval_confidence.py --strategies relation_type usage_based

    # Limit evaluation set size
    python scripts/eval_confidence.py --limit 100

    # Save results to JSON
    python scripts/eval_confidence.py --output results.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

# Add server root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.memory_store import MemoryStore, Paths
from core.memory.confidence_strategy import (
    RelationTypeConfidence,
    UsageBasedConfidence,
    create_confidence_strategy
)
from core.memory.evaluation import (
    build_eval_dataset,
    compare_strategies,
    print_evaluation_report,
    confidence_distribution,
    EvalExample
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate confidence scoring strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on existing database
  python scripts/eval_confidence.py --db data/memory.db

  # Compare specific strategies
  python scripts/eval_confidence.py --strategies relation_type usage_based

  # Quick test on small dataset
  python scripts/eval_confidence.py --limit 50

  # Save results
  python scripts/eval_confidence.py --output results.json
        """
    )

    parser.add_argument(
        "--db",
        default="data/memory.db",
        help="Path to SQLite database (relative to server/)"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["relation_type", "usage_based"],
        choices=["relation_type", "usage_based"],
        help="Strategies to evaluate (default: both)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum evaluation examples (default: 500)"
    )

    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-example analysis"
    )

    args = parser.parse_args()

    # Resolve database path
    db_path = Path(__file__).parent.parent / args.db

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Please check the path or run the system to generate data")
        return 1

    print(f"Loading database: {db_path}")

    # Load store
    store = MemoryStore(Paths(sqlite_path=str(db_path), lmdb_dir=None))

    # Build evaluation dataset
    print(f"Building evaluation dataset (limit={args.limit})...")
    test_set = build_eval_dataset(store, limit=args.limit)

    if not test_set:
        print("❌ No evaluation data found!")
        print("   The database needs conversation history with provenance.")
        print("   Run the system and have conversations to generate data.")
        return 1

    print(f"✅ Built dataset: {len(test_set)} examples")

    # Analyze dataset
    print("\nDataset Statistics:")
    correct_count = sum(1 for ex in test_set if ex.is_correct)
    print(f"  Total examples:     {len(test_set)}")
    print(f"  Labeled correct:    {correct_count} ({correct_count/len(test_set)*100:.1f}%)")
    print(f"  Labeled incorrect:  {len(test_set)-correct_count} ({(len(test_set)-correct_count)/len(test_set)*100:.1f}%)")

    avg_sources = sum(ex.source_count for ex in test_set) / len(test_set)
    avg_reinf = sum(ex.reinforcements for ex in test_set) / len(test_set)
    avg_age = sum(ex.age_days for ex in test_set) / len(test_set)

    print(f"  Avg source count:   {avg_sources:.2f}")
    print(f"  Avg reinforcements: {avg_reinf:.2f}")
    print(f"  Avg age (days):     {avg_age:.1f}")

    # Create strategies
    print(f"\nEvaluating strategies: {', '.join(args.strategies)}")
    strategies = {}
    for name in args.strategies:
        strategies[name] = create_confidence_strategy(name)

    # Evaluate
    print("\nRunning evaluation...")
    results = compare_strategies(strategies, test_set, store)

    # Print results
    print_evaluation_report(results, title="Confidence Strategy Comparison")

    # Show confidence distribution for each strategy
    print("\nConfidence Distribution:")
    for name, strategy in strategies.items():
        # Re-score to get predictions
        predictions = []
        from core.memory.confidence_strategy import Edge, Context
        import time as time_module

        for ex in test_set:
            edge = Edge(
                src=ex.fact[0], rel=ex.fact[1], dst=ex.fact[2],
                pos=ex.reinforcements, neg=ex.negations,
                updated_at=int((time_module.time() - ex.age_days * 86400) * 1000),
                id=ex.edge_id
            )
            context = Context(store=store, text=ex.text)
            predictions.append(strategy.score(edge, context))

        dist = confidence_distribution(predictions)

        print(f"\n{name}:")
        for bin_label, count in dist.items():
            pct = count / len(predictions) * 100
            bar = "█" * int(pct / 2)
            print(f"  {bin_label}: {count:4d} ({pct:5.1f}%) {bar}")

    # Verbose mode: show per-example analysis
    if args.verbose:
        print("\n" + "="*80)
        print("Per-Example Analysis (first 10)")
        print("="*80)

        for i, ex in enumerate(test_set[:10]):
            print(f"\nExample {i+1}:")
            print(f"  Text: {ex.text[:60]}...")
            print(f"  Fact: {ex.fact[0]} --[{ex.fact[1]}]--> {ex.fact[2]}")
            print(f"  Ground Truth: {'CORRECT' if ex.is_correct else 'INCORRECT'}")
            print(f"  Sources: {ex.source_count}, Reinforcements: {ex.reinforcements}, Negations: {ex.negations}")
            print(f"  Age: {ex.age_days:.1f} days")

            from core.memory.confidence_strategy import Edge, Context
            import time as time_module

            edge = Edge(
                src=ex.fact[0], rel=ex.fact[1], dst=ex.fact[2],
                pos=ex.reinforcements, neg=ex.negations,
                updated_at=int((time_module.time() - ex.age_days * 86400) * 1000),
                id=ex.edge_id
            )
            context = Context(store=store, text=ex.text)

            print(f"  Predictions:")
            for name, strategy in strategies.items():
                pred = strategy.score(edge, context)
                match = "✓" if (pred >= 0.7) == ex.is_correct else "✗"
                print(f"    {name:20s}: {pred:.3f} {match}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'dataset': {
                    'size': len(test_set),
                    'correct': correct_count,
                    'incorrect': len(test_set) - correct_count,
                    'avg_sources': avg_sources,
                    'avg_reinforcements': avg_reinf,
                    'avg_age_days': avg_age
                },
                'results': results
            }, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    best_strategy = max(results.items(), key=lambda x: x[1]['correlation'])
    print(f"Best strategy (by correlation): {best_strategy[0]}")
    print(f"  Correlation: {best_strategy[1]['correlation']:.4f}")
    print(f"  MAE: {best_strategy[1]['mae']:.4f}")
    print(f"  Accuracy@0.7: {best_strategy[1]['accuracy_at_70']:.4f}")

    # Improvement calculation
    if "relation_type" in results and "usage_based" in results:
        baseline = results["relation_type"]
        improved = results["usage_based"]

        # Calculate improvements (handle zero division)
        if abs(baseline['correlation']) > 0.001:
            corr_improvement = (improved['correlation'] - baseline['correlation']) / abs(baseline['correlation']) * 100
        else:
            corr_improvement = (improved['correlation'] - baseline['correlation']) * 100

        if baseline['mae'] > 0:
            mae_improvement = (baseline['mae'] - improved['mae']) / baseline['mae'] * 100
        else:
            mae_improvement = 0.0

        if baseline['accuracy_at_70'] > 0:
            acc_improvement = (improved['accuracy_at_70'] - baseline['accuracy_at_70']) / baseline['accuracy_at_70'] * 100
        else:
            acc_improvement = 0.0

        print(f"\nUsage-Based vs Baseline:")
        print(f"  Correlation: {corr_improvement:+.1f}%")
        print(f"  MAE:         {mae_improvement:+.1f}% (lower is better)")
        print(f"  Accuracy:    {acc_improvement:+.1f}%")

    print("\n✅ Evaluation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())