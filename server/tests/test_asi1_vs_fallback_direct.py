#!/usr/bin/env python3
"""
Direct A/B Comparison: ASI1 vs Other Extraction Methods
Tests speed and quality of triple extraction
"""

import os
import sys
import time
from typing import List, Tuple, Dict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_asi1():
    """Test ASI1 extraction"""
    from asi1_processor import ULTRAGROKSpacyV821Processor

    processor = ULTRAGROKSpacyV821Processor(
        yaml_file="ULTRAGROK_V8.2.1_SPACY.yaml",
        model_name="en_core_web_sm"
    )
    return processor

def test_level3():
    """Test Level3 Universal KG extraction"""
    from level3_universal_kg import UniversalKGExtractor

    extractor = UniversalKGExtractor()
    return extractor

def test_enhanced_level3():
    """Test Enhanced Level3 extraction"""
    from enhanced_level3_extractor import EnhancedLevel3Extractor

    extractor = EnhancedLevel3Extractor()
    return extractor

def extract_with_asi1(text: str):
    """Extract using ASI1"""
    try:
        processor = test_asi1()
        start = time.perf_counter()
        result = processor.process_spacy_semantics(text)
        elapsed = (time.perf_counter() - start) * 1000

        # Convert ASI1 format to standard triples
        triples = []
        for triple in result.get('triples', []):
            triples.append((triple.subj, triple.pred, triple.obj))

        return triples, elapsed
    except Exception as e:
        print(f"ASI1 Error: {e}")
        return [], 0

def extract_with_level3(text: str):
    """Extract using Level3 Universal KG"""
    try:
        extractor = test_level3()
        start = time.perf_counter()
        kg = extractor.extract_universal_kg(text)
        elapsed = (time.perf_counter() - start) * 1000

        # Convert to standard triples
        triples = []
        for rel in kg.relations:
            triples.append((rel.subject, rel.predicate, rel.object))

        return triples, elapsed
    except Exception as e:
        print(f"Level3 Error: {e}")
        return [], 0

def extract_with_enhanced(text: str):
    """Extract using Enhanced Level3"""
    try:
        extractor = test_enhanced_level3()
        start = time.perf_counter()
        result = extractor.extract(text)
        elapsed = (time.perf_counter() - start) * 1000

        # Result is already in triples format
        return result['relations'], elapsed
    except Exception as e:
        print(f"Enhanced Error: {e}")
        return [], 0

def calculate_quality_score(triples: List[Tuple[str, str, str]]) -> float:
    """Calculate quality score based on triple characteristics"""
    if not triples:
        return 0.0

    score = 0.0
    meaningful_count = 0

    for subj, pred, obj in triples:
        # Skip trivial/redundant relations
        if pred in ['has_attribute', 'modifies', 'type', 'is', 'has', 'located_in', 'participates_in']:
            continue

        # Count meaningful relations
        meaningful_count += 1

        # Reward action predicates
        if any(action in pred.lower() for action in ['chase', 'play', 'watch', 'respond', 'save', 'cause', 'enable']):
            score += 2.0
        # Reward specific predicates
        elif len(pred) > 5 and pred not in ['being', 'having', 'doing']:
            score += 1.0
        else:
            score += 0.5

        # Reward specific entities (not pronouns)
        if subj.lower() not in ['it', 'they', 'he', 'she', 'this', 'that'] and len(subj) > 2:
            score += 0.5
        if obj.lower() not in ['it', 'they', 'he', 'she', 'this', 'that'] and len(obj) > 2:
            score += 0.5

    return score / max(len(triples), 1), meaningful_count

def run_comparison():
    """Run direct comparison"""

    # Test sentences
    test_texts = [
        ("Easy", "The cat chased the ball across the sunny yard."),

        ("Simple", "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees."),

        ("Complex", "Yesterday, firefighters quickly responded to a small kitchen fire caused by an unattended stove, saving the family's home and ensuring no one was injured in the timely rescue operation."),

        ("Technical", "In the quantum computing algorithm, qubits entangled through superposition states enable parallel processing, where error correction codes mitigate decoherence effects."),
    ]

    print("=" * 80)
    print("DIRECT A/B COMPARISON: ASI1 vs Level3 vs Enhanced")
    print("=" * 80)

    for label, text in test_texts:
        print(f"\n📝 {label}: {text[:60]}...")
        print("-" * 70)

        # Test each method
        methods = [
            ("ASI1", extract_with_asi1),
            ("Level3", extract_with_level3),
            ("Enhanced", extract_with_enhanced)
        ]

        results = []
        for name, extract_func in methods:
            triples, time_ms = extract_func(text)
            quality, meaningful = calculate_quality_score(triples)
            results.append({
                'name': name,
                'triples': triples,
                'count': len(triples),
                'meaningful': meaningful,
                'time_ms': round(time_ms, 2),
                'quality': round(quality, 2)
            })

        # Display comparison
        print(f"{'Method':<15} {'Time(ms)':<10} {'Total':<8} {'Meaningful':<12} {'Quality':<10}")
        print("-" * 60)

        for r in results:
            print(f"{r['name']:<15} {r['time_ms']:<10.2f} {r['count']:<8} {r['meaningful']:<12} {r['quality']:<10.2f}")

        # Show best triples
        best = max(results, key=lambda x: x['quality'])
        print(f"\n🏆 Best Quality: {best['name']}")
        print("Sample meaningful triples:")

        count = 0
        for s, p, o in best['triples']:
            if p not in ['has_attribute', 'modifies', 'type', 'participates_in']:
                print(f"  • {s} | {p} | {o}")
                count += 1
                if count >= 3:
                    break

        # Show speed winner if different
        fastest = min(results, key=lambda x: x['time_ms'])
        if fastest['name'] != best['name']:
            print(f"\n⚡ Fastest: {fastest['name']} ({fastest['time_ms']}ms)")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. ASI1: Pure YAML pattern matching (no GLiNER, no GLiREL, no LLM)
2. Level3: SpaCy patterns with comprehensive extraction
3. Enhanced: Level3 with quality filtering

WHAT WE'RE MEASURING:
- Speed: How fast extraction completes
- Total: Total number of triples extracted
- Meaningful: Non-trivial relations (excludes has_attribute, modifies, etc.)
- Quality: Score based on predicate meaningfulness and entity specificity
""")

if __name__ == "__main__":
    run_comparison()