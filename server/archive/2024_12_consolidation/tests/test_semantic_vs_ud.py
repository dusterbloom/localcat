#!/usr/bin/env python3
"""
🔬 DIRECT COMPARISON: Semantic SRL vs UD Patterns
Shows exactly what each approach extracts from the same sentences
"""

import spacy
from components.processing.semantic_roles import SRLExtractor
from services.ud_utils import UDPatternMatcher

# Load models once
nlp = spacy.load("en_core_web_trf")
srl = SRLExtractor(use_normalizer=False)
ud_matcher = UDPatternMatcher()

def compare_approaches(text: str):
    """Compare what each approach extracts"""
    print(f"\n🔍 SENTENCE: '{text}'")
    print("="*60)

    doc = nlp(text)

    # SEMANTIC APPROACH
    print("\n🧠 SEMANTIC SRL:")
    try:
        predications = srl.doc_to_predications(doc)
        semantic_triples = srl.predications_to_triples(predications)
        if semantic_triples:
            for i, triple in enumerate(semantic_triples, 1):
                print(f"  {i}. {triple}")
        else:
            print("  ❌ No extractions")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

    # UD PATTERNS APPROACH
    print("\n📐 UD PATTERNS:")
    try:
        ud_relations = ud_matcher.match(doc)
        if ud_relations:
            for i, rel in enumerate(ud_relations, 1):
                print(f"  {i}. ({rel.subject}, {rel.relation}, {rel.object}) [conf: {rel.confidence:.2f}]")
        else:
            print("  ❌ No extractions")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

def main():
    print("🔬 SEMANTIC SRL vs UD PATTERNS COMPARISON")
    print("="*60)

    # Test the cases where semantic failed
    failed_cases = [
        "My name is Alex Thompson",           # Copula failure
        "My dog's name is Potola",           # Copula + possessive failure
        "Sarah and John are friends",        # Copula + coordination failure
        "My favorite color is blue",         # Copula + attribute failure
        "I was born in 1995",                # Passive voice failure
        "My son is named Jake",              # Copula + passive failure
    ]

    # Test cases where semantic worked
    working_cases = [
        "Alice feeds the cat in the morning",
        "I live in Seattle",
        "I work at Microsoft",
        "Caroline went to the LGBTQ support group",
    ]

    print("\n🚨 FAILED CASES (Semantic missed, UD should catch):")
    for case in failed_cases:
        compare_approaches(case)

    print("\n\n✅ WORKING CASES (Both should extract):")
    for case in working_cases:
        compare_approaches(case)

if __name__ == "__main__":
    main()