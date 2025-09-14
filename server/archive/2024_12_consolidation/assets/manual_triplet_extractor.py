#!/usr/bin/env python3
"""Manual triplet extraction to show what we should be getting"""

import spacy

def manual_extract_triplets(text):
    """Extract basic subject-verb-object triplets manually using spaCy"""
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    triplets = []

    for sent in doc.sents:
        # Find main verb (ROOT)
        root_verb = None
        for token in sent:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                root_verb = token
                break

        if root_verb:
            # Find subject
            subject = None
            for child in root_verb.children:
                if child.dep_ in ['nsubj', 'csubj']:
                    subject = child.text
                    break

            # Find direct object
            obj = None
            for child in root_verb.children:
                if child.dep_ in ['dobj', 'pobj']:
                    obj = child.text
                    break

            # Find prepositional object (for "at Google")
            prep_obj = None
            for child in root_verb.children:
                if child.dep_ == 'prep':
                    for grandchild in child.children:
                        if grandchild.dep_ == 'pobj':
                            prep_obj = f"{child.text} {grandchild.text}"
                            break

            # Add triplet
            if subject:
                verb = root_verb.lemma_

                # Primary triplet
                if obj:
                    triplets.append((subject, verb, obj))
                elif prep_obj:
                    triplets.append((subject, verb, prep_obj))
                else:
                    triplets.append((subject, verb, ""))

                # Additional prepositional relation
                if prep_obj and obj:
                    triplets.append((subject, verb, prep_obj))

    return triplets

def show_manual_comparison():
    """Show what triplets should look like vs ASI systems"""

    test_cases = [
        'John works at Google',
        'Mary gave the book to her friend',
        'The CEO announced quarterly results'
    ]

    print('🎯 MANUAL vs ASI TRIPLET COMPARISON')
    print('=' * 60)

    for i, text in enumerate(test_cases, 1):
        print(f'\n{i}. "{text}"')
        print('-' * 40)

        # Manual extraction
        manual_triplets = manual_extract_triplets(text)
        print(f'  MANUAL: {len(manual_triplets)} triplets')
        for j, (s, p, o) in enumerate(manual_triplets, 1):
            print(f'    {j}. {s} | {p} | {o}')

        # Try ASI2 (if available)
        try:
            from asi1_processor import ULTRAGROKSpacyV821Processor

            # Redirect debug output to null
            import builtins
            original_print = builtins.print
            builtins.print = lambda *args, **kwargs: None

            processor = ULTRAGROKSpacyV821Processor('ULTRAGROK_V8.2.1_SPACY.yaml')
            result = processor.process_spacy_semantics(text)

            # Restore print
            builtins.print = original_print

            triples = result.get('triples', [])
            print(f'  ASI2: {len(triples)} triplets')
            for j, triple in enumerate(triples[:3], 1):
                subj = getattr(triple, 'subj', getattr(triple, 'subject', 'N/A'))
                pred = getattr(triple, 'pred', getattr(triple, 'predicate', 'N/A'))
                obj = getattr(triple, 'obj', getattr(triple, 'object', 'N/A'))
                print(f'    {j}. {subj} | {pred} | {obj}')

        except Exception as e:
            print(f'  ASI2: ERROR - {str(e)[:50]}...')

        # Try ASI1 V8.2.3 (if available)
        try:
            from asi1_processor import ULTRAGROKSpacyV821Processor

            # Redirect debug output to null
            import builtins
            original_print = builtins.print
            builtins.print = lambda *args, **kwargs: None

            processor = ULTRAGROKSpacyV821Processor('ASI1_8_2_3.yaml')
            result = processor.process_spacy_semantics(text)

            # Restore print
            builtins.print = original_print

            triples = result.get('triples', [])
            print(f'  ASI1: {len(triples)} triplets')
            for j, triple in enumerate(triples[:3], 1):
                subj = getattr(triple, 'subj', getattr(triple, 'subject', 'N/A'))
                pred = getattr(triple, 'pred', getattr(triple, 'predicate', 'N/A'))
                obj = getattr(triple, 'obj', getattr(triple, 'object', 'N/A'))
                print(f'    {j}. {subj} | {pred} | {obj}')

        except Exception as e:
            print(f'  ASI1: ERROR - {str(e)[:50]}...')

if __name__ == "__main__":
    show_manual_comparison()