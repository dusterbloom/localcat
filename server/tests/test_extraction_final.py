#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import spacy
nlp = spacy.load("en_core_web_sm")

from components.extraction.enhanced_level3_extractor import QualityExtractor

# Patch to add debugging
original_extract = QualityExtractor._extract_candidate_relations

def debug_extract(self, doc):
    print("\n[DEBUG] _extract_candidate_relations called")
    result = original_extract(self, doc)
    print(f"[DEBUG] Returned {len(result)} relations")
    for r in result:
        print(f"  - {r.subject} | {r.predicate} | {r.object}")
    return result

QualityExtractor._extract_candidate_relations = debug_extract

extractor = QualityExtractor()

test_text = "I live in Sardinia."
doc = nlp(test_text)

print(f"Testing: '{test_text}'")
result = extractor.extract_quality_kg(doc)

print(f"\nFinal Results:")
print(f"  Entities: {len(result['entities'])}")  
print(f"  Relations: {len(result['relations'])}")

if not result['relations']:
    print("\n❌ Relations were extracted but filtered out!")
    print(f"  Confidence threshold: {extractor.RELATION_CONFIDENCE_THRESHOLD}")
