#!/usr/bin/env python3
"""Test filtered quality extraction on news example"""

from components.extraction.level3_universal_kg import UniversalKGExtractor

def test_news_quality():
    extractor = UniversalKGExtractor()

    # The news example you loved
    news_text = "In the bustling city park, a group of children played tag while their parents watched from wooden benches under tall oak trees."

    kg = extractor.extract_universal_kg(news_text)

    print('🎯 FILTERED QUALITY TEST - NEWS EXAMPLE')
    print('=' * 50)
    print(f'Input: "{news_text}"')
    print()

    print(f'📊 RESULTS: {len(kg.entities)} entities, {len(kg.relations)} relations')
    print()

    print(f'🔥 ALL EXTRACTED RELATIONS:')
    for i, relation in enumerate(kg.relations, 1):
        if 'has_attribute' not in relation.predicate and 'modifies' not in relation.predicate:
            print(f'  ✅ {i:2d}. {relation.subject} | {relation.predicate} | {relation.object}')
        else:
            print(f'  📝 {i:2d}. {relation.subject} | {relation.predicate} | {relation.object} (filtered)')

    print(f'\n🎯 BEAUTIFUL CORE RELATIONS (no noise):')
    beautiful_relations = [r for r in kg.relations
                          if 'has_attribute' not in r.predicate
                          and 'modifies' not in r.predicate
                          and 'participates_in' not in r.predicate
                          and 'type' not in r.predicate]

    for i, relation in enumerate(beautiful_relations, 1):
        print(f'  🌟 {i}. {relation.subject} | {relation.predicate} | {relation.object}')

if __name__ == "__main__":
    test_news_quality()
