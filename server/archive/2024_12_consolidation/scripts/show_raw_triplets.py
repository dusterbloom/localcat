#!/usr/bin/env python3
"""Show raw extracted triplets before any filtering"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Completely disable all print statements
import builtins
def no_print(*args, **kwargs):
    pass
builtins.print = no_print

from asi1_processor import ULTRAGROKSpacyV821Processor

# Monkey patch the processor to capture raw triples
original_filter_quality_signal = ULTRAGROKSpacyV821Processor._filter_quality_signal

def capture_raw_triples(self, triples):
    self._captured_triples = triples.copy()  # Capture before filtering
    return original_filter_quality_signal(self, triples)

ULTRAGROKSpacyV821Processor._filter_quality_signal = capture_raw_triples

def show_raw_triplets():
    # Restore print for our output only
    builtins.print = print

    test_cases = [
        'John works at Google',
        'Mary gave the book to her friend',
        'The CEO announced quarterly results'
    ]

    print('🎯 RAW EXTRACTED TRIPLETS (Before Filtering)')
    print('=' * 60)

    # ASI2 (Original ULTRAGROK V8.2.1)
    try:
        processor_asi2 = ULTRAGROKSpacyV821Processor('ULTRAGROK_V8.2.1_SPACY.yaml')
        asi2_available = True
    except:
        asi2_available = False

    # ASI1 V8.2.3
    try:
        processor_asi1 = ULTRAGROKSpacyV821Processor('ASI1_8_2_3.yaml')
        asi1_available = True
    except:
        asi1_available = False

    for i, text in enumerate(test_cases, 1):
        print(f'\n{i}. "{text}"')
        print('-' * 40)

        # ASI2 Test
        if asi2_available:
            try:
                processor_asi2.process_spacy_semantics(text)
                raw_triples = getattr(processor_asi2, '_captured_triples', [])

                print(f'  ASI2: {len(raw_triples)} raw triplets')
                for j, triple in enumerate(raw_triples[:3], 1):
                    subj = getattr(triple, 'subj', 'N/A')
                    pred = getattr(triple, 'pred', 'N/A')
                    obj = getattr(triple, 'obj', 'N/A')
                    print(f'    {j}. {subj} | {pred} | {obj}')
            except Exception as e:
                print(f'  ASI2: ERROR - {str(e)[:50]}...')
        else:
            print('  ASI2: Not Available')

        # ASI1 Test
        if asi1_available:
            try:
                processor_asi1.process_spacy_semantics(text)
                raw_triples = getattr(processor_asi1, '_captured_triples', [])

                print(f'  ASI1: {len(raw_triples)} raw triplets')
                for j, triple in enumerate(raw_triples[:3], 1):
                    subj = getattr(triple, 'subj', 'N/A')
                    pred = getattr(triple, 'pred', 'N/A')
                    obj = getattr(triple, 'obj', 'N/A')
                    print(f'    {j}. {subj} | {pred} | {obj}')
            except Exception as e:
                print(f'  ASI1: ERROR - {str(e)[:50]}...')
        else:
            print('  ASI1: Not Available')

if __name__ == "__main__":
    show_raw_triplets()