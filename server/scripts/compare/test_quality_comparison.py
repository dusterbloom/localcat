#!/usr/bin/env python3
import os
import sys
import time

# Suppress debug output
os.environ['DEBUG'] = ''
os.environ['ASI_DEBUG'] = ''

# Redirect stdout during initialization to suppress warnings
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with SuppressOutput():
    from asi1_processor import ULTRAGROKSpacyV821Processor
    from level3_universal_kg import UniversalKGExtractor

text = "John Smith works at Google in San Francisco. He manages the AI team and develops new products."

print("ASI1 vs Level3 Quality Comparison")
print("="*50)

# ASI1
print("\nASI1:")
asi1 = ULTRAGROKSpacyV821Processor()
result1 = asi1.process_spacy_semantics(text)
print(f"Extracted: {len(result1.get('triples', []))} triples")
for t in result1.get('triples', []):
    print(f"  {t.subj} | {t.pred} | {t.obj}")

# Level3
print("\nLevel3:")
level3 = UniversalKGExtractor()
result3 = level3.extract_universal_kg(text)
print(f"Extracted: {len(result3.relations)} relations")
for r in result3.relations[:5]:
    print(f"  {r.subject} | {r.predicate} | {r.object}")

print("\n⚠️ FINDING:")
print(f"ASI1: {len(result1.get('triples', []))} triples")
print(f"Level3: {len(result3.relations)} relations")
if len(result3.relations) > len(result1.get('triples', [])) * 5:
    print("Level3 extracts MANY MORE relations than ASI1!")
    print("They are NOT the same system!")

