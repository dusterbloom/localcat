#!/usr/bin/env python3
"""Debug ASI1 to see why it produces incomplete triples"""

import os
os.environ['ASI_DEBUG'] = 'false'

from asi1_processor import ULTRAGROKSpacyV821Processor

# Test text
text = "John Smith works at Google in San Francisco."

print("ASI1 Debug Analysis")
print("=" * 50)
print(f"Text: {text}\n")

# Initialize and process
processor = ULTRAGROKSpacyV821Processor()
result = processor.process_spacy_semantics(text)

print(f"Raw result keys: {result.keys()}")
print(f"Total raw relations: {result.get('total_raw_relations', 0)}")
print(f"Quality filtered: {result.get('quality_filtered', 0)}")
print(f"Final validated: {result.get('final_validated', 0)}")

print("\nTriples structure:")
triples = result.get('triples', [])
for i, t in enumerate(triples):
    print(f"Triple {i}:")
    print(f"  Type: {type(t)}")
    print(f"  Attrs: {dir(t) if hasattr(t, '__dict__') else 'N/A'}")
    if hasattr(t, 'subj'):
        print(f"  subj: '{t.subj}'")
        print(f"  pred: '{t.pred}'")
        print(f"  obj: '{t.obj}'")
        print(f"  confidence: {getattr(t, 'confidence', 'N/A')}")
        print(f"  pattern_name: {getattr(t, 'pattern_name', 'N/A')}")
