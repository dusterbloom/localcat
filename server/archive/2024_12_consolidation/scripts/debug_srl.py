#!/usr/bin/env python3
"""
Debug why SRL extractor returns 0 relations
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import spacy
from components.processing.semantic_roles import SRLExtractor

def debug_srl():
    """Debug SRL extraction step by step"""
    print("🐛 DEBUGGING SRL EXTRACTION")
    print("=" * 50)

    text = "The CEO announced that the company would restructure after declining profits."
    print(f"📝 Text: {text}")

    # Load spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    print(f"\n🔍 SPACY ANALYSIS:")
    print("-" * 20)
    for sent in doc.sents:
        print(f"Sentence: {sent.text}")
        print(f"Root: {sent.root.text} ({sent.root.pos_}) - {sent.root.lemma_}")

        for token in sent:
            print(f"  {token.text:12} | {token.pos_:6} | {token.dep_:10} | head: {token.head.text}")

    print(f"\n🧠 SRL EXTRACTION:")
    print("-" * 20)

    # Test SRL
    srl = SRLExtractor(use_normalizer=True)
    predications = srl.doc_to_predications(doc)

    print(f"Predications found: {len(predications)}")
    for i, pred in enumerate(predications, 1):
        print(f"  {i}. Predicate: '{pred.predicate}'")
        print(f"     Roles: {pred.roles}")
        print(f"     Lang: {pred.lang}")
        print(f"     Sentence: {pred.sent_text}")

    print(f"\n⚡ TRIPLE CONVERSION:")
    print("-" * 20)

    triples = srl.predications_to_triples(predications)
    print(f"Triples: {len(triples)}")
    for i, triple in enumerate(triples, 1):
        print(f"  {i}. {triple}")

    return predications, triples

if __name__ == "__main__":
    debug_srl()