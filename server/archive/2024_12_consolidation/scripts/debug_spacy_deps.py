#!/usr/bin/env python3
"""Debug spaCy dependencies for 'John works at Google'"""

import spacy

def debug_spacy_deps():
    nlp = spacy.load('en_core_web_sm')
    text = "John works at Google"
    doc = nlp(text)

    print(f'Text: "{text}"')
    print('=' * 40)
    print('spaCy Dependencies:')

    for token in doc:
        print(f'{token.text:8} | {token.dep_:12} | {token.pos_:8} | head: {token.head.text}')

    print('\nDependency Tree:')
    for token in doc:
        if token.dep_ == 'ROOT':
            print(f'ROOT: {token.text} ({token.pos_})')
            for child in token.children:
                print(f'  └─ {child.text} ({child.dep_}, {child.pos_})')
                for grandchild in child.children:
                    print(f'      └─ {grandchild.text} ({grandchild.dep_}, {grandchild.pos_})')

    print('\nFor ASI1 spatial pattern:')
    print('Looking for: ^prep and ^pobj from prep')

    for token in doc:
        if token.dep_ == 'ROOT':
            print(f'Root verb: {token.text}')
            for child in token.children:
                if child.dep_ == 'prep':
                    print(f'  Found prep: {child.text} (rel: {child.dep_})')
                    for grandchild in child.children:
                        if grandchild.dep_ == 'pobj':
                            print(f'    Found pobj: {grandchild.text} (rel: {grandchild.dep_})')

if __name__ == "__main__":
    debug_spacy_deps()