#!/usr/bin/env python3
"""SHOW THE ACTUAL TRIPLETS - NO FLUFF"""

import spacy
from level3_extractor import Level3Extractor
from asi1_precision_postprocessor import ASI1PrecisionProcessor

def show_triplets():
    """Show actual extracted triplets from all systems"""

    nlp = spacy.load('en_core_web_sm')
    level3_extractor = Level3Extractor()
    asi1_processor = ASI1PrecisionProcessor()

    test_cases = [
        'John works at Google',
        'Mary gave the book to her friend',
        'The CEO announced quarterly results',
        'John works at Google. He announced quarterly results. However, the company faced challenges.'
    ]

    print('🔥 ACTUAL EXTRACTED TRIPLETS - NO BS')
    print('=' * 70)

    for i, text in enumerate(test_cases, 1):
        print(f'\n{i}. TEXT: "{text}"')
        print('=' * 50)

        # Level 3 Extractor
        level3_triples = level3_extractor.extract(text)
        print(f'LEVEL 3 EXTRACTOR ({len(level3_triples)} triples):')
        for j, triple in enumerate(level3_triples, 1):
            print(f'  {j}. {triple.subject} | {triple.predicate} | {triple.object}')

        # ASI1 + Level 3 Processing
        doc = nlp(text)
        raw_triples = []
        for triple in level3_triples:
            raw_triples.append({
                'subj': triple.subject,
                'pred': triple.predicate,
                'obj': triple.object,
                'confidence': triple.confidence
            })

        asi1_triples = asi1_processor.process_level3(raw_triples, doc)
        print(f'\nASI1 LEVEL 3 PROCESSED ({len(asi1_triples)} triples):')
        for j, triple in enumerate(asi1_triples, 1):
            print(f'  {j}. {triple.subj} | {triple.pred} | {triple.obj}')

        print()

if __name__ == "__main__":
    show_triplets()