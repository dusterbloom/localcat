#!/usr/bin/env python3
"""Debug quality filtering to see what's being filtered out"""

import sys
import warnings
warnings.filterwarnings('ignore')

from asi1_processor import ULTRAGROKSpacyV821Processor

def debug_quality_filtering():
    processor = ULTRAGROKSpacyV821Processor('ASI1_8_2_3.yaml')

    test_cases = [
        'John works at Google',
        'Mary gave the book to her friend'
    ]

    print('🔍 QUALITY FILTERING DEBUG')
    print('=' * 40)

    for i, text in enumerate(test_cases, 1):
        print(f'\n{i}. INPUT: "{text}"')

        # Get the full result with raw relations
        result = processor.process_spacy_semantics(text)

        print(f'   Raw relations extracted: {result.get("raw_relations", 0)}')
        print(f'   Quality relations: {result.get("quality_relations", 0)}')
        print(f'   Final relations: {len(result.get("relations", []))}')

        # Show the actual relations that were extracted (before filtering)
        if 'relations' in result:
            print(f'   Relations that passed all filters:')
            for j, rel in enumerate(result['relations'][:3], 1):
                subj = rel.get('subj', 'N/A')
                pred = rel.get('pred', 'N/A')
                obj = rel.get('obj', 'N/A')
                print(f'     {j}. {subj} | {pred} | {obj}')

        # Check what the result structure looks like
        print(f'   Result keys: {list(result.keys())}')

if __name__ == "__main__":
    debug_quality_filtering()