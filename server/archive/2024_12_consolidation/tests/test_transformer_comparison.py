#!/usr/bin/env python3
"""Test transformer vs statistical model comparison"""

from level3_universal_kg import UniversalKGExtractor
import time

def test_transformer_performance():
    extractor = UniversalKGExtractor()

    # Test cases with varying complexity
    test_cases = [
        ("Simple", "The cat chased the ball across the sunny yard."),
        ("Complex", "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees."),
        ("Technical", "Engineers designing the neural network architecture incorporated convolutional layers to extract hierarchical features from input images."),
        ("Scientific", "Evolutionary biologists posit that Darwin's finches exemplify punctuated equilibrium through rapid speciation events driven by ecological niches."),
    ]

    print("🎯 TRANSFORMER MODEL PERFORMANCE TEST")
    print("=" * 60)

    total_start = time.time()

    for i, (category, text) in enumerate(test_cases, 1):
        print(f"\n📝 TEST {i}: {category}")
        print("─" * 40)
        print(f"Input: \"{text[:60]}...\"")

        # Extract with timing
        start_time = time.time()
        kg = extractor.extract_universal_kg(text)
        extraction_time = (time.time() - start_time) * 1000

        # Performance metrics
        beautiful_relations = [r for r in kg.relations
                             if ('has_attribute' not in r.predicate and
                                 'modifies' not in r.predicate and
                                 'participates_in' not in r.predicate and
                                 'type' not in r.predicate)]

        print(f"⚡ Time: {extraction_time:.1f}ms")
        print(f"📊 Results: {len(kg.entities)} entities, {len(kg.relations)} total relations")
        print(f"🌟 Beautiful: {len(beautiful_relations)} core semantic relations")

        # Show coreference if any
        if hasattr(kg, 'coreference_clusters') and kg.coreference_clusters:
            print(f"🔗 Coreference: {len(kg.coreference_clusters)} clusters")

        # Show best relations
        if beautiful_relations:
            print("🎯 Top semantic relations:")
            for j, relation in enumerate(beautiful_relations[:3], 1):
                print(f"  {j}. {relation.subject} | {relation.predicate} | {relation.object}")

    total_time = (time.time() - total_start) * 1000
    print("\n" + "=" * 60)
    print(f"🏆 TOTAL TIME: {total_time:.1f}ms for {len(test_cases)} sentences")
    print(f"📈 AVERAGE: {total_time/len(test_cases):.1f}ms per sentence")

if __name__ == "__main__":
    test_transformer_performance()