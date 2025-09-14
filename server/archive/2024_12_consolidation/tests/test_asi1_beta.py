#!/usr/bin/env python3
"""
Test ASI1_BETA.yaml precision system
"""

import time
from components.extraction.yaml_ud_loader import YAMLUDExtractor
import spacy

def test_asi1_beta():
    """Test ASI1_BETA.yaml system"""

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install: python -m spacy download en_core_web_sm")
        return

    # Initialize system
    print("🔥 TESTING ASI1_BETA.yaml (Precision-Focused)")
    print("=" * 60)

    try:
        asi1_extractor = YAMLUDExtractor("ASI1_BETA.yaml")
        print("✅ ASI1_BETA.yaml loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load ASI1_BETA.yaml: {e}")
        return

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

    total_triples = 0
    total_time = 0
    coverage_count = 0
    high_quality_count = 0

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: {sentence}")
        print("-" * 50)

        doc = nlp(sentence)

        start_time = time.time()
        triples = asi1_extractor.extract_triples(doc)
        elapsed_time = (time.time() - start_time) * 1000

        total_time += elapsed_time
        total_triples += len(triples)

        if triples:
            coverage_count += 1

        print(f"🔗 {len(triples)} triples extracted in {elapsed_time:.1f}ms:")
        for triple in triples:
            print(f"   • {triple}")

            # Quality assessment
            subj, pred, obj = triple
            if len(subj) > 1 and len(pred) > 1 and obj and len(obj) > 1:
                if pred not in ["be", "exist", "have"]:
                    high_quality_count += 1

    print("\n" + "=" * 60)
    print("📊 ASI1_BETA PRECISION RESULTS")
    print("=" * 60)

    avg_triples = total_triples / len(test_sentences) if test_sentences else 0
    avg_time = total_time / len(test_sentences) if test_sentences else 0
    coverage_pct = (coverage_count / len(test_sentences)) * 100 if test_sentences else 0
    quality_ratio = (high_quality_count / total_triples) if total_triples > 0 else 0

    print(f"🎯 Coverage: {coverage_count}/15 sentences ({coverage_pct:.1f}%)")
    print(f"🔗 Total triples: {total_triples}")
    print(f"💡 Avg triples per sentence: {avg_triples:.1f}")
    print(f"⚡ Avg extraction time: {avg_time:.1f}ms")
    print(f"💎 High-quality triples: {high_quality_count}/{total_triples} ({quality_ratio:.1%})")

    # Precision assessment
    if avg_triples <= 3.0:
        print("✅ PRECISION TARGET MET: ≤3 triples per sentence")
    else:
        print(f"⚠️  PRECISION MISS: {avg_triples:.1f} > 3.0 triples per sentence")

    if coverage_pct >= 90:
        print("✅ COVERAGE TARGET MET: ≥90% sentences")
    else:
        print(f"⚠️  COVERAGE MISS: {coverage_pct:.1f}% < 90%")

    # Overall verdict
    print(f"\n🏆 ASI1_BETA VERDICT:")
    if avg_triples <= 3.0 and coverage_pct >= 90 and quality_ratio >= 0.7:
        print("🟢 PRECISION SUCCESS - Ready for production!")
    elif avg_triples <= 3.5 and coverage_pct >= 80:
        print("🟡 GOOD PRECISION - Minor tweaks needed")
    else:
        print("🔴 NEEDS WORK - Precision/coverage issues")

if __name__ == "__main__":
    test_asi1_beta()