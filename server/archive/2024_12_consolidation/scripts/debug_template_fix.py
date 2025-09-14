#!/usr/bin/env python3
"""Debug template resolution for ASI1 V8.2.3"""

from asi1_processor import ULTRAGROKSpacyV821Processor

def debug_template_resolution():
    processor = ULTRAGROKSpacyV821Processor('ASI1_8_2_3.yaml')

    text = "John works at Google"
    print(f'\nTesting: "{text}"')
    print('=' * 40)

    result = processor.process_spacy_semantics(text)
    triples = result.get('triples', [])

    print(f'\nFinal result: {len(triples)} triplets')
    for i, triple in enumerate(triples, 1):
        subj = getattr(triple, 'subj', 'N/A')
        pred = getattr(triple, 'pred', 'N/A')
        obj = getattr(triple, 'obj', 'N/A')
        print(f'  {i}. {subj} | {pred} | {obj}')

if __name__ == "__main__":
    debug_template_resolution()