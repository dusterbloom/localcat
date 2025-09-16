#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Bypass the strategy and test extractor directly
print("Testing extraction bug...")

# 1. Import spacy
import spacy
nlp = spacy.load("en_core_web_sm")
print("✅ Loaded spaCy")

# 2. Import the extractor directly
from components.extraction.enhanced_level3_extractor import QualityExtractor
print("✅ Imported QualityExtractor")

# 3. Create extractor
extractor = QualityExtractor()
print(f"✅ Created extractor (DSPy available: {hasattr(extractor, 'dspy_extractor')})")

# 4. Test sentence
test_text = "I live in Sardinia."
doc = nlp(test_text)
print(f"\nTesting: '{test_text}'")

# 5. Extract
result = extractor.extract_quality_kg(doc)

print(f"\nResults:")
print(f"  Entities: {len(result['entities'])}")
print(f"  Relations: {len(result['relations'])}")

for rel in result['relations']:
    print(f"    {rel.subject} | {rel.predicate} | {rel.object} (conf={rel.confidence})")

if not result['relations']:
    print("\n❌ NO RELATIONS EXTRACTED! Debugging...")
    
    # Debug parse
    for token in doc:
        print(f"  {token.text:10} pos={token.pos_:6} dep={token.dep_:10} lemma={token.lemma_}")
    
    print(f"\n'live' in core_verbs: {'live' in extractor.core_verbs}")
