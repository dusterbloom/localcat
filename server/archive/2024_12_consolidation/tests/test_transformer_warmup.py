#!/usr/bin/env python3
"""Test transformer performance with proper model warm-up"""

from level3_universal_kg import UniversalKGExtractor
import time

def test_warmed_up_performance():
    print("🔥 WARMING UP TRANSFORMER MODEL...")

    # Create extractor and warm up with a simple sentence
    extractor = UniversalKGExtractor()

    # Warm-up runs (models load on first use)
    warmup_text = "The quick brown fox jumps."
    print("Running 3 warm-up extractions...")
    for i in range(3):
        start = time.time()
        kg = extractor.extract_universal_kg(warmup_text)
        warmup_time = (time.time() - start) * 1000
        print(f"  Warm-up {i+1}: {warmup_time:.1f}ms")

    print("\n🎯 TESTING WARMED-UP PERFORMANCE")
    print("=" * 60)

    # Test cases
    test_cases = [
        ("Simple", "The cat chased the ball across the sunny yard."),
        ("Complex", "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees."),
        ("Technical", "Engineers designing the neural network architecture incorporated convolutional layers to extract hierarchical features from input images."),
        ("Scientific", "Evolutionary biologists posit that Darwin's finches exemplify punctuated equilibrium through rapid speciation events driven by ecological niches."),
    ]

    total_start = time.time()
    times = []

    for i, (category, text) in enumerate(test_cases, 1):
        print(f"\n📝 TEST {i}: {category}")
        print("─" * 40)
        print(f"Input: \"{text[:60]}...\"")

        # Run multiple times for consistent measurement
        run_times = []
        for run in range(3):
            start_time = time.time()
            kg = extractor.extract_universal_kg(text)
            extraction_time = (time.time() - start_time) * 1000
            run_times.append(extraction_time)

        # Average the runs
        avg_time = sum(run_times) / len(run_times)
        times.append(avg_time)

        # Quality metrics
        beautiful_relations = [r for r in kg.relations
                             if ('has_attribute' not in r.predicate and
                                 'modifies' not in r.predicate and
                                 'participates_in' not in r.predicate and
                                 'type' not in r.predicate)]

        print(f"⚡ Average time (3 runs): {avg_time:.1f}ms (range: {min(run_times):.1f}-{max(run_times):.1f}ms)")
        print(f"📊 Results: {len(kg.entities)} entities, {len(kg.relations)} total relations")
        print(f"🌟 Beautiful: {len(beautiful_relations)} core semantic relations")

        # Show coreference if any
        if hasattr(kg, 'coreference_clusters') and kg.coreference_clusters:
            print(f"🔗 Coreference: {len(kg.coreference_clusters)} clusters")

        # Show best relations
        if beautiful_relations:
            print("🎯 Top semantic relations:")
            for j, relation in enumerate(beautiful_relations[:2], 1):
                print(f"  {j}. {relation.subject} | {relation.predicate} | {relation.object}")

    avg_overall = sum(times) / len(times)
    print("\n" + "=" * 60)
    print(f"🏆 WARMED-UP AVERAGE: {avg_overall:.1f}ms per sentence")
    print(f"📈 RANGE: {min(times):.1f}ms - {max(times):.1f}ms")

    # Performance assessment
    if avg_overall < 500:
        print("✅ MEETS <500ms TARGET!")
    else:
        print(f"⚠️  EXCEEDS 500ms target by {avg_overall-500:.1f}ms")

if __name__ == "__main__":
    test_warmed_up_performance()