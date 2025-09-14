#!/usr/bin/env python3
"""
Debug the "My name is" pattern matching step by step
"""
import spacy
from components.extraction.yaml_ud_loader import YAMLUDExtractor

def debug_my_name_pattern():
    nlp = spacy.load('en_core_web_sm')
    doc = nlp('My name is Alex Thompson.')

    # Load extractor
    extractor = YAMLUDExtractor('enhanced_fastlane_rules.ud.yaml')

    # Find MY_NAME_IS pattern
    my_name_pattern = None
    for pattern in extractor.matcher.patterns:
        if pattern.name == 'MY_NAME_IS':
            my_name_pattern = pattern
            break

    if not my_name_pattern:
        print("❌ MY_NAME_IS pattern not found!")
        return

    print("✅ Found MY_NAME_IS pattern:")
    print(f"   Priority: {my_name_pattern.priority}")
    print(f"   Edges: {len(my_name_pattern.edges)}")
    for i, edge in enumerate(my_name_pattern.edges):
        print(f"     {i+1}. from:{edge.get('from')} -> rel:{edge.get('rel')} -> as:{edge.get('as')}")
    print(f"   Guards: {my_name_pattern.guards}")
    print(f"   Emit: {my_name_pattern.emit}")

    # Test anchor matching
    print("\n🔍 Testing anchor matching...")
    anchor_token = None
    for token in doc:
        if token.pos_ == 'AUX' and token.lemma_ in ['be', 'is', 'are', 'was', 'were']:
            anchor_token = token
            print(f"✅ Found anchor: {token.text} (pos={token.pos_}, lemma={token.lemma_})")
            break

    if not anchor_token:
        print("❌ No anchor found!")
        return

    # Test edge matching manually
    print(f"\n🔗 Testing edge matching from anchor '{anchor_token.text}'...")

    # Edge 1: nsubj -> name_noun
    nsubj_child = None
    for child in anchor_token.children:
        if child.dep_ in ['nsubj', 'csubj']:
            nsubj_child = child
            print(f"✅ Edge 1: Found nsubj child: {child.text} (dep={child.dep_})")
            break
    if not nsubj_child:
        print("❌ Edge 1: No nsubj child found")
        return

    # Edge 2: poss -> poss_pron (from name_noun)
    poss_child = None
    for child in nsubj_child.children:
        if child.dep_ == 'poss':
            poss_child = child
            print(f"✅ Edge 2: Found poss child of name_noun: {child.text} (dep={child.dep_})")
            break
    if not poss_child:
        print("❌ Edge 2: No poss child found")
        return

    # Edge 3: attr -> actual_name (from anchor)
    attr_child = None
    for child in anchor_token.children:
        if child.dep_ == 'attr':
            attr_child = child
            print(f"✅ Edge 3: Found attr child: {child.text} (dep={child.dep_})")
            break
    if not attr_child:
        print("❌ Edge 3: No attr child found")
        return

    # Test guards
    print(f"\n🛡️ Testing guards...")
    print(f"   name_noun.lemma = '{nsubj_child.lemma_}' (should be in ['name'])")
    name_guard_pass = nsubj_child.lemma_.lower() in ['name']
    print(f"   name_noun_lemma_in: {'✅ PASS' if name_guard_pass else '❌ FAIL'}")

    print(f"   poss_pron.lemma = '{poss_child.lemma_}' (should be in ['my', 'i'])")
    poss_guard_pass = poss_child.lemma_.lower() in ['my', 'i']
    print(f"   poss_pron_lemma_in: {'✅ PASS' if poss_guard_pass else '❌ FAIL'}")

    if name_guard_pass and poss_guard_pass:
        print("✅ All guards pass!")
        print(f"\n🎯 Should emit triple:")
        print(f"   subj: 'you'")
        print(f"   pred: 'has_name'")
        print(f"   obj: '{attr_child.text}' (subtree: {' '.join([t.text for t in attr_child.subtree])})")
    else:
        print("❌ Guards failed!")

    # Test actual extraction
    print(f"\n🚀 Testing actual extraction...")
    result = extractor.extract_triples(doc)
    print(f"Result: {result}")

if __name__ == "__main__":
    debug_my_name_pattern()