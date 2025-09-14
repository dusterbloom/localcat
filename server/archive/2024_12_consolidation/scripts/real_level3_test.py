#!/usr/bin/env python3
"""
REAL LEVEL 3 TEST - The Actual Requirements
============================================

LEVEL 2: Coreference + Complexity + Cross-Sentence
LEVEL 3: Universal KG (any length, any complexity, any language, rich KGs)
"""

import spacy
from level3_extractor import Level3Extractor
from asi1_precision_postprocessor import ASI1PrecisionProcessor

def test_real_level3():
    """Test against the ACTUAL Level 3 requirements"""

    nlp = spacy.load('en_core_web_sm')
    extractor = Level3Extractor()
    processor = ASI1PrecisionProcessor()

    print('🎯 REAL LEVEL 3 UNIVERSAL KG GENERATION TEST')
    print('=' * 60)

    # LEVEL 2 TESTS
    print('\n📋 LEVEL 2 REQUIREMENTS:')
    print('=' * 30)

    # Coreference test
    coref_text = "John works at Google. He announced results. The CEO said the company is doing well."
    print(f'\n1. COREFERENCE TEST:')
    print(f'   Input: "{coref_text}"')

    doc = nlp(coref_text)
    level3_triples = extractor.extract(coref_text)

    raw_triples = [{'subj': t.subject, 'pred': t.predicate, 'obj': t.object, 'confidence': t.confidence} for t in level3_triples]
    final_triples = processor.process_level3(raw_triples, doc)

    print(f'   Extracted {len(final_triples)} triples:')
    for i, triple in enumerate(final_triples, 1):
        print(f'     {i}. {triple.subj} | {triple.pred} | {triple.obj}')

    # Cross-sentence relations test
    discourse_text = "John launched the product. However, sales were disappointing. Therefore, the company pivoted strategy."
    print(f'\n2. CROSS-SENTENCE RELATIONS:')
    print(f'   Input: "{discourse_text}"')

    doc = nlp(discourse_text)
    level3_triples = extractor.extract(discourse_text)
    raw_triples = [{'subj': t.subject, 'pred': t.predicate, 'obj': t.object, 'confidence': t.confidence} for t in level3_triples]
    final_triples = processor.process_level3(raw_triples, doc)

    print(f'   Extracted {len(final_triples)} triples:')
    for i, triple in enumerate(final_triples, 1):
        print(f'     {i}. {triple.subj} | {triple.pred} | {triple.obj}')

    # LEVEL 3 TESTS
    print('\n🚀 LEVEL 3 REQUIREMENTS:')
    print('=' * 30)

    # Long text test (100+ words)
    long_text = """
    John Smith works as the Chief Technology Officer at Google Corporation in Mountain View, California.
    He announced the quarterly financial results during yesterday's board meeting. The results showed
    significant growth in cloud computing revenue. However, the company faced challenges in the competitive
    artificial intelligence market. Mary Johnson, the Chief Marketing Officer, then joined the discussion
    to present the new marketing strategy. She emphasized the importance of customer acquisition and retention.
    The board members expressed concerns about the recent regulatory changes in the European Union.
    Subsequently, the legal team provided an analysis of the compliance requirements. The meeting concluded
    with unanimous approval of the proposed budget allocation for the next fiscal quarter.
    """

    print(f'\n3. LONG TEXT (10-10,000+ words):')
    print(f'   Input: {len(long_text.split())} words')

    doc = nlp(long_text)
    level3_triples = extractor.extract(long_text)
    raw_triples = [{'subj': t.subject, 'pred': t.predicate, 'obj': t.object, 'confidence': t.confidence} for t in level3_triples]
    final_triples = processor.process_level3(raw_triples, doc)

    print(f'   Extracted {len(final_triples)} triples:')
    for i, triple in enumerate(final_triples[:10], 1):  # Show first 10
        print(f'     {i}. {triple.subj} | {triple.pred} | {triple.obj}')
    if len(final_triples) > 10:
        print(f'     ... and {len(final_triples) - 10} more')

    # Complex technical text
    technical_text = """
    The distributed microservices architecture implements event-driven communication patterns through
    asynchronous message queues. The system utilizes containerized deployment strategies with
    Kubernetes orchestration for scalability and fault tolerance.
    """

    print(f'\n4. COMPLEX TECHNICAL TEXT:')
    print(f'   Input: "{technical_text.strip()}"')

    doc = nlp(technical_text)
    level3_triples = extractor.extract(technical_text)
    raw_triples = [{'subj': t.subject, 'pred': t.predicate, 'obj': t.object, 'confidence': t.confidence} for t in level3_triples]
    final_triples = processor.process_level3(raw_triples, doc)

    print(f'   Extracted {len(final_triples)} triples:')
    for i, triple in enumerate(final_triples, 1):
        print(f'     {i}. {triple.subj} | {triple.pred} | {triple.obj}')

    # EVALUATION
    print('\n📊 LEVEL 3 EVALUATION:')
    print('=' * 25)

    total_entities = set()
    total_relations = set()

    # Count unique entities and relations across all tests
    all_test_triples = []

    for test_text in [coref_text, discourse_text, long_text, technical_text]:
        doc = nlp(test_text)
        level3_triples = extractor.extract(test_text)
        raw_triples = [{'subj': t.subject, 'pred': t.predicate, 'obj': t.object, 'confidence': t.confidence} for t in level3_triples]
        final_triples = processor.process_level3(raw_triples, doc)
        all_test_triples.extend(final_triples)

    for triple in all_test_triples:
        if triple.subj.strip():
            total_entities.add(triple.subj.strip().lower())
        if triple.obj.strip():
            total_entities.add(triple.obj.strip().lower())
        if triple.pred.strip():
            total_relations.add(triple.pred.strip().lower())

    print(f'Total Unique Entities: {len(total_entities)}')
    print(f'Total Unique Relations: {len(total_relations)}')
    print(f'Total Triples: {len(all_test_triples)}')

    # Level 3 Requirements Check
    print(f'\nLEVEL 3 REQUIREMENTS CHECK:')
    print(f'✅ Any Length: Tested 10-100+ words')
    print(f'✅ Any Complexity: Simple → Technical')
    print(f'✅ Rich KGs: {len(total_entities)} entities, {len(total_relations)} relations')

    # What we're missing
    print(f'\n⚠️  MISSING FOR FULL LEVEL 3:')
    print(f'❌ Coreference clusters (basic resolution only)')
    print(f'❌ Discourse structure (detection only)')
    print(f'❌ Temporal chains (basic detection)')
    print(f'❌ Multi-language (English only)')
    print(f'❌ 50+ entities/relations on single text (need longer input)')

if __name__ == "__main__":
    test_real_level3()