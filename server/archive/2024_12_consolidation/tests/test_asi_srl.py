#!/usr/bin/env python3
"""
Test ASI-Enhanced YAML UD→SRL System
===================================

Testing the sophisticated ASI-optimized semantic role labeling system.
"""

import spacy
from components.extraction.yaml_ud_loader import YAMLUDExtractor
import time

def test_asi_srl_system():
    """Test ASI's enhanced UD→SRL system"""

    # Test sentences covering different constructions
    test_sentences = [
        # Basic coordination
        "John and Mary gave books to students.",

        # Modal/aspectual chains
        "She will have been working at Google.",

        # Complex nominals
        "The CEO of Microsoft announced profits.",

        # Comparative
        "This solution is better than the previous one.",

        # Temporal chains
        "After John graduated, he worked at Apple.",

        # Quantifiers
        "All students passed the exam.",

        # Clause embedding
        "I think that she knows the answer.",

        # Psych verb (if we had Italian)
        "I like chocolate.",

        # Negation
        "John does not like Mary.",

        # Passive
        "The book was written by the author.",
    ]

    print("🚀 Testing ASI-Enhanced YAML UD→SRL System")
    print("=" * 60)

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("❌ Please install: python -m spacy download en_core_web_sm")
        return

    try:
        # Initialize with enhanced rules (simplified version)
        extractor = YAMLUDExtractor("enhanced_fastlane_rules.ud.yaml")

        total_triples = 0
        total_time = 0

        for i, sentence in enumerate(test_sentences):
            print(f"\n📝 Sentence {i+1}: {sentence}")

            start_time = time.time()
            doc = nlp(sentence)
            triples = extractor.extract_triples(doc)
            end_time = time.time()

            extraction_time = (end_time - start_time) * 1000
            total_time += extraction_time
            total_triples += len(triples)

            print(f"⏱️  Extraction time: {extraction_time:.1f}ms")
            print(f"🔗 Extracted {len(triples)} triples:")

            for triple in triples:
                subj, pred, obj = triple
                print(f"   • (\"{subj}\", \"{pred}\", \"{obj}\")")

            print("-" * 50)

        # Performance summary
        avg_time = total_time / len(test_sentences)
        print(f"\n📊 Performance Summary:")
        print(f"   • Total sentences: {len(test_sentences)}")
        print(f"   • Total triples: {total_triples}")
        print(f"   • Average time per sentence: {avg_time:.1f}ms")
        print(f"   • Target achieved: {'✅ YES' if avg_time < 500 else '❌ NO'} (<500ms)")

    except Exception as e:
        print(f"❌ Error testing ASI system: {e}")
        import traceback
        traceback.print_exc()

        # Fallback test with basic rules
        print("\n🔄 Falling back to basic YAML test...")
        try:
            extractor = YAMLUDExtractor("enhanced_fastlane_rules.ud.yaml")
            doc = nlp("John gave Mary a book.")
            triples = extractor.extract_triples(doc)
            print(f"Basic test result: {triples}")
        except Exception as e2:
            print(f"Basic test also failed: {e2}")

if __name__ == "__main__":
    test_asi_srl_system()