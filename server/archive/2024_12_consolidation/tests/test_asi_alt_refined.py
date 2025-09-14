#!/usr/bin/env python3
"""
Test the ASI_ALT_REFINED.yaml system and compare with existing systems
"""

import time
from components.extraction.yaml_ud_loader import YAMLUDExtractor
from components.processing.semantic_roles import EnhancedSemanticRoleExtractor
import spacy

def test_asi_alt_refined():
    """Test ASI_ALT_REFINED.yaml system"""

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install: python -m spacy download en_core_web_sm")
        return

    # Initialize systems
    asi_alt_extractor = YAMLUDExtractor("ASI_ALT_REFINED.yaml")
    semantic_extractor = EnhancedSemanticRoleExtractor()

    test_sentences = [
        "My name is Alex Thompson.",
        "John works at Google.",
        "Mary lives in Paris.",
        "John gave Mary a book.",
        "She likes chocolate.",
        "The company announced profits.",
        "John and Mary gave books to students.",
        "She will have been working at Google.",
        "After John graduated, he worked at Apple.",
        "The CEO of Microsoft announced profits.",
        "This solution is better than the previous one.",
        "All students passed the exam.",
        "I think that she knows the answer.",
        "John does not like Mary.",
        "The book was written by the author."
    ]

    print("🔥 A/B/C/D SRL QUALITY TEST")
    print("=" * 60)
    print("A = Existing semantic_roles.py system")
    print("B = Enhanced UD system (18 rules)")
    print("C = ASI_ALT_REFINED system (44 rules)")
    print("=" * 60)

    results_a = []
    results_b = []
    results_c = []

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: {sentence}")
        print("-" * 50)

        doc = nlp(sentence)

        # System A: Semantic roles
        start_time = time.time()
        triples_a = semantic_extractor.extract_semantic_triples(doc)
        time_a = (time.time() - start_time) * 1000

        # System C: ASI_ALT_REFINED
        start_time = time.time()
        triples_c = asi_alt_extractor.extract_triples(doc)
        time_c = (time.time() - start_time) * 1000

        print(f"🅰️  System A (Semantic): {len(triples_a)} triples in {time_a:.1f}ms")
        for triple in triples_a:
            print(f"   • {triple}")

        print(f"🅲  System C (ASI_ALT): {len(triples_c)} triples in {time_c:.1f}ms")
        for triple in triples_c:
            print(f"   • {triple}")

        # Simple quality scoring
        quality_a = min(1.0, len([t for t in triples_a if len(t[0]) > 1 and len(t[1]) > 1]) * 0.3)
        quality_c = min(1.0, len([t for t in triples_c if len(t[0]) > 1 and len(t[1]) > 1]) * 0.3)

        print(f"📊 Quality: A={quality_a:.2f} | C={quality_c:.2f}")

        if quality_a > quality_c:
            print("🏆 Winner: System A (Semantic)")
        elif quality_c > quality_a:
            print("🏆 Winner: System C (ASI_ALT)")
        else:
            print("🤝 Tie")

        results_a.append((len(triples_a), time_a, quality_a))
        results_c.append((len(triples_c), time_c, quality_c))

    # Final results
    print("\n" + "=" * 60)
    print("📊 FINAL A/C TEST RESULTS")
    print("=" * 60)

    avg_quality_a = sum(r[2] for r in results_a) / len(results_a)
    avg_time_a = sum(r[1] for r in results_a) / len(results_a)
    total_triples_a = sum(r[0] for r in results_a)

    avg_quality_c = sum(r[2] for r in results_c) / len(results_c)
    avg_time_c = sum(r[1] for r in results_c) / len(results_c)
    total_triples_c = sum(r[0] for r in results_c)

    print(f"🅰️  System A (Semantic):")
    print(f"   Quality Score: {avg_quality_a:.3f}/1.000")
    print(f"   Avg Time: {avg_time_a:.1f}ms")
    print(f"   Total Triples: {total_triples_a}")

    print(f"🅲  System C (ASI_ALT):")
    print(f"   Quality Score: {avg_quality_c:.3f}/1.000")
    print(f"   Avg Time: {avg_time_c:.1f}ms")
    print(f"   Total Triples: {total_triples_c}")

    if avg_quality_a > avg_quality_c:
        advantage = ((avg_quality_a - avg_quality_c) / avg_quality_c) * 100
        print(f"\n🏆 VERDICT:")
        print(f"   System A WINS by {advantage:.1f}% quality advantage")
    elif avg_quality_c > avg_quality_a:
        advantage = ((avg_quality_c - avg_quality_a) / avg_quality_a) * 100
        print(f"\n🏆 VERDICT:")
        print(f"   System C WINS by {advantage:.1f}% quality advantage")
    else:
        print(f"\n🤝 VERDICT: TIE")

    print(f"\n💡 Key Insights:")
    print(f"   • System C extracts {total_triples_c - total_triples_a:+d} more triples")
    if avg_time_a > 0:
        speed_ratio = avg_time_c / avg_time_a
        print(f"   • System A is {speed_ratio:.1f}x faster")

if __name__ == "__main__":
    test_asi_alt_refined()