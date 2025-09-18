#!/usr/bin/env python3
"""
A/B Comparison: ASI1 vs Fallback Extraction Strategies
Tests speed and quality of triple extraction
"""

import os
import sys
import time
from typing import List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import extraction components
from processors.extraction_processor import ExtractionProcessor, ExtractionProcessorConfig
from components.extraction.extraction_registry import ExtractionRegistry as ExtractorRegistry

@dataclass
class ExtractionComparison:
    strategy: str
    text: str
    num_triples: int
    extraction_time_ms: float
    triples: List[Tuple[str, str, str]]
    quality_score: float

def calculate_quality_score(triples: List[Tuple[str, str, str]]) -> float:
    """Calculate quality score based on triple characteristics"""
    if not triples:
        return 0.0

    score = 0.0
    for subj, pred, obj in triples:
        # Penalize redundant/trivial relations
        if pred in ['has_attribute', 'modifies', 'type', 'is', 'has']:
            score += 0.2
        # Reward meaningful predicates
        elif pred in ['chased', 'played', 'watched', 'responded', 'caused', 'saved',
                      'enable', 'process', 'extract', 'optimize', 'reveals', 'seeks',
                      'posits', 'argues', 'challenges']:
            score += 1.0
        else:
            score += 0.5

        # Reward longer, specific entities
        if len(subj) > 15 and len(obj) > 15:
            score += 0.3

        # Penalize pronouns and generic terms
        if subj.lower() in ['it', 'they', 'he', 'she', 'this', 'that']:
            score -= 0.2

    return score / len(triples)

def test_extraction_strategy(text: str, strategy: str) -> ExtractionComparison:
    """Test a single extraction strategy"""
    # Set the extraction strategy via environment
    os.environ['DEFAULT_EXTRACTION_STRATEGY'] = strategy
    os.environ['HOTMEM_ROUTE_TO_REGISTRY'] = 'true'

    # Create processor
    processor = ExtractionProcessor()

    # Measure extraction time
    start_time = time.perf_counter()
    result = processor.extract(text)
    extraction_time = (time.perf_counter() - start_time) * 1000

    # Calculate quality score
    quality = calculate_quality_score(result.triples)

    return ExtractionComparison(
        strategy=strategy,
        text=text[:50] + "..." if len(text) > 50 else text,
        num_triples=len(result.triples),
        extraction_time_ms=round(extraction_time, 2),
        triples=result.triples[:5],  # First 5 for display
        quality_score=round(quality, 2)
    )

def run_ab_comparison():
    """Run A/B comparison between ASI1 and fallback strategies"""

    # Test sentences from Level1to3_text.md
    test_texts = [
        # Easy (10 words)
        "The cat chased the ball across the sunny yard.",

        # Simple (20 words)
        "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees.",

        # Complex (30 words)
        "Yesterday, firefighters quickly responded to a small kitchen fire caused by an unattended stove, saving the family's home and ensuring no one was injured in the timely rescue operation.",

        # Technical (50 words)
        "In the quantum computing algorithm, qubits entangled through superposition states enable parallel processing, where error correction codes, such as surface codes implemented via lattice surgery, mitigate decoherence effects by repeatedly measuring stabilizers to preserve computational fidelity across multiple logical gates.",

        # Philosophical (110 words - truncated for testing)
        "In contemplating the existential dialectic between freedom and determinism, Sartre's notion of 'bad faith' reveals how individuals, ensnared in the gaze of the Other, often deny their radical liberty by assuming inauthentic roles."
    ]

    # Strategies to test
    strategies = ['asi1', 'asi2', 'enhanced_hotmem', 'lightweight']

    print("=" * 80)
    print("A/B COMPARISON: ASI1 vs Fallback Extraction Strategies")
    print("=" * 80)

    for idx, text in enumerate(test_texts, 1):
        print(f"\n📝 TEST {idx}: {text[:60]}...")
        print("-" * 70)

        results = []
        for strategy in strategies:
            try:
                result = test_extraction_strategy(text, strategy)
                results.append(result)
            except Exception as e:
                print(f"  ❌ {strategy}: Failed - {str(e)}")
                continue

        # Display results
        print(f"{'Strategy':<20} {'Time(ms)':<10} {'#Triples':<10} {'Quality':<10}")
        print("-" * 60)

        for result in results:
            print(f"{result.strategy:<20} {result.extraction_time_ms:<10.2f} {result.num_triples:<10} {result.quality_score:<10.2f}")

        # Show sample triples from best performer
        if results:
            best = max(results, key=lambda r: r.quality_score)
            print(f"\n  🏆 Best Quality: {best.strategy}")
            print("  Sample triples:")
            for i, (s, p, o) in enumerate(best.triples[:3], 1):
                print(f"    {i}. ({s} | {p} | {o})")

            fastest = min(results, key=lambda r: r.extraction_time_ms)
            if fastest != best:
                print(f"\n  ⚡ Fastest: {fastest.strategy} ({fastest.extraction_time_ms}ms)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Overall statistics
    all_results = {}
    for text in test_texts[:3]:  # Test on first 3 for summary
        for strategy in strategies:
            try:
                result = test_extraction_strategy(text, strategy)
                if strategy not in all_results:
                    all_results[strategy] = []
                all_results[strategy].append(result)
            except:
                pass

    print(f"{'Strategy':<20} {'Avg Time(ms)':<15} {'Avg Triples':<15} {'Avg Quality':<15}")
    print("-" * 65)

    for strategy, results in all_results.items():
        if results:
            avg_time = sum(r.extraction_time_ms for r in results) / len(results)
            avg_triples = sum(r.num_triples for r in results) / len(results)
            avg_quality = sum(r.quality_score for r in results) / len(results)
            print(f"{strategy:<20} {avg_time:<15.2f} {avg_triples:<15.1f} {avg_quality:<15.2f}")

if __name__ == "__main__":
    # Check if extraction registry exists
    try:
        from components.extraction.extraction_strategies import ExtractorRegistry
        registry = ExtractorRegistry.get_instance()
        print(f"✅ Extraction Registry loaded with {len(registry.strategies)} strategies")
    except Exception as e:
        print(f"⚠️ Warning: Extraction registry not fully configured: {e}")

    run_ab_comparison()
