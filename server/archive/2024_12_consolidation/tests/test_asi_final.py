#!/usr/bin/env python3
"""
Test ASI_ALT_REFINED.yaml final performance
"""

import time
from components.extraction.yaml_ud_loader import YAMLUDExtractor
import spacy

def test_asi_alt_refined():
    """Test ASI_ALT_REFINED.yaml system"""

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install: python -m spacy download en_core_web_sm")
        return

    # Initialize system
    print("🔥 TESTING ASI_ALT_REFINED.yaml (Final Version)")
    print("=" * 60)

    asi_extractor = YAMLUDExtractor("ASI_ALT_REFINED.yaml")

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
    key_patterns_found = 0

    # Track specific patterns we care about
    key_patterns = {
        "name_extraction": False,
        "work_at": False,
        "live_in": False,
        "gave_action": False
    }

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: {sentence}")
        print("-" * 50)

        doc = nlp(sentence)

        start_time = time.time()
        triples = asi_extractor.extract_triples(doc)
        elapsed_time = (time.time() - start_time) * 1000

        total_time += elapsed_time
        total_triples += len(triples)

        if triples:
            coverage_count += 1

        print(f"🔗 {len(triples)} triples extracted in {elapsed_time:.1f}ms:")
        for triple in triples:
            print(f"   • {triple}")

            # Check for key patterns
            subj, pred, obj = triple
            if sentence.startswith("My name is") and pred == "has_name":
                key_patterns["name_extraction"] = True
            elif "works at" in sentence and "work" in pred.lower():
                key_patterns["work_at"] = True
            elif "lives in" in sentence and ("live" in pred.lower() or "location" in pred):
                key_patterns["live_in"] = True
            elif "gave" in sentence and "give" in pred.lower():
                key_patterns["gave_action"] = True

        # Simple quality assessment
        quality_score = 0
        for triple in triples:
            subj, pred, obj = triple
            # Quality indicators: reasonable length, no empty parts, meaningful predicates
            if len(subj) > 1 and len(pred) > 1 and pred not in ["be", "exist", "have"]:
                quality_score += 0.5
            # Bonus for semantic predicates over syntactic
            if any(semantic_pred in pred for semantic_pred in ["work_at", "live_in", "has_name", "part_of"]):
                quality_score += 0.3

        print(f"📊 Quality estimate: {quality_score:.2f}")

    # Count found key patterns
    key_patterns_found = sum(key_patterns.values())

    print("\n" + "=" * 60)
    print("📊 ASI_ALT_REFINED FINAL RESULTS")
    print("=" * 60)
    print(f"🎯 Coverage: {coverage_count}/15 sentences ({coverage_count/15*100:.1f}%)")
    print(f"🔗 Total triples: {total_triples}")
    print(f"⚡ Avg extraction time: {total_time/len(test_sentences):.1f}ms")
    print(f"🎮 Key patterns found: {key_patterns_found}/4")
    print(f"   • Name extraction: {'✅' if key_patterns['name_extraction'] else '❌'}")
    print(f"   • Work location: {'✅' if key_patterns['work_at'] else '❌'}")
    print(f"   • Live location: {'✅' if key_patterns['live_in'] else '❌'}")
    print(f"   • Give action: {'✅' if key_patterns['gave_action'] else '❌'}")

    avg_triples = total_triples / len(test_sentences)
    print(f"\n💡 Average triples per sentence: {avg_triples:.1f}")

    if avg_triples > 5:
        print("⚠️  HIGH EXTRACTION - May be over-extracting")
    elif avg_triples > 2:
        print("✅ MODERATE EXTRACTION - Good balance")
    else:
        print("🔍 LOW EXTRACTION - May be missing patterns")

    # Performance assessment
    if total_time/len(test_sentences) < 1.0:
        print("🚀 PERFORMANCE: Excellent (<1ms avg)")
    elif total_time/len(test_sentences) < 5.0:
        print("⚡ PERFORMANCE: Good (<5ms avg)")
    else:
        print("🐌 PERFORMANCE: Needs optimization")

    # Overall recommendation
    print(f"\n🏆 OVERALL ASSESSMENT:")
    overall_score = (coverage_count/15) * 0.4 + (key_patterns_found/4) * 0.6
    if overall_score > 0.8:
        print("🟢 EXCELLENT - Ready for production")
    elif overall_score > 0.6:
        print("🟡 GOOD - Minor tweaks needed")
    else:
        print("🔴 NEEDS WORK - Major improvements required")

if __name__ == "__main__":
    test_asi_alt_refined()