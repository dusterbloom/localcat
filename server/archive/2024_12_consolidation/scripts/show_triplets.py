#!/usr/bin/env python3
"""Show actual extracted triplets - clean output only"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Completely silence all debug output
import builtins
original_print = print
def silent_print(*args, **kwargs):
    return
builtins.print = silent_print

from asi1_processor import ULTRAGROKSpacyV821Processor

def show_triplets():
    # Restore print for our output only
    builtins.print = original_print

    # Initialize processors
    try:
        processor_asi2 = ULTRAGROKSpacyV821Processor('ULTRAGROK_V8.2.1_SPACY.yaml')
        asi2_available = True
    except:
        asi2_available = False

    try:
        processor_asi1 = ULTRAGROKSpacyV821Processor('ASI1_8_2_3.yaml')
        asi1_available = True
    except:
        asi1_available = False

    # Test sentences
    test_cases = [
        'John works at Google',
        'Mary gave the book to her friend',
        'The CEO announced quarterly results'
    ]

    print('🎯 EXTRACTED TRIPLETS COMPARISON')
    print('=' * 60)

    for i, text in enumerate(test_cases, 1):
        print(f'\n{i}. "{text}"')
        print('-' * 40)

        # ASI2 (Original ULTRAGROK V8.2.1)
        if asi2_available:
            try:
                # Patch the processor to bypass quality filtering
                result = processor_asi2.process_spacy_semantics(text)

                # Try to get raw triples before filtering
                if hasattr(processor_asi2, '_last_raw_triples'):
                    triples = processor_asi2._last_raw_triples
                else:
                    triples = result.get('triples', [])

                print(f'  ASI2: {len(triples)} triplets')
                for j, triple in enumerate(triples[:3], 1):
                    if hasattr(triple, 'subject'):
                        subj = getattr(triple, 'subject', 'N/A')
                        pred = getattr(triple, 'predicate', 'N/A')
                        obj = getattr(triple, 'object', 'N/A')
                    else:
                        subj = getattr(triple, 'subj', 'N/A')
                        pred = getattr(triple, 'pred', 'N/A')
                        obj = getattr(triple, 'obj', 'N/A')
                    print(f'    {j}. {subj} | {pred} | {obj}')
            except Exception as e:
                print(f'  ASI2: ERROR - {str(e)[:50]}...')
        else:
            print('  ASI2: Not Available')

        # ASI1 V8.2.3
        if asi1_available:
            try:
                result = processor_asi1.process_spacy_semantics(text)

                # Try to get raw triples before filtering
                if hasattr(processor_asi1, '_last_raw_triples'):
                    triples = processor_asi1._last_raw_triples
                else:
                    triples = result.get('triples', [])

                print(f'  ASI1: {len(triples)} triplets')
                for j, triple in enumerate(triples[:3], 1):
                    if hasattr(triple, 'subject'):
                        subj = getattr(triple, 'subject', 'N/A')
                        pred = getattr(triple, 'predicate', 'N/A')
                        obj = getattr(triple, 'object', 'N/A')
                    else:
                        subj = getattr(triple, 'subj', 'N/A')
                        pred = getattr(triple, 'pred', 'N/A')
                        obj = getattr(triple, 'obj', 'N/A')
                    print(f'    {j}. {subj} | {pred} | {obj}')
            except Exception as e:
                print(f'  ASI1: ERROR - {str(e)[:50]}...')
        else:
            print('  ASI1: Not Available')

if __name__ == "__main__":
    show_triplets()