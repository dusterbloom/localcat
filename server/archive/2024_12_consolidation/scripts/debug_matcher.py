#!/usr/bin/env python3
"""
Debug the actual pattern matching process
"""
import spacy
from components.extraction.yaml_ud_loader import YAMLUDExtractor

# Monkey patch the matcher to add debug output
def debug_match_pattern(self, anchor, pattern, context):
    print(f"\n🔍 Testing pattern '{pattern.name}' on anchor '{anchor.text}'")

    # Check anchor conditions first
    anchor_matches = True
    if hasattr(pattern, 'anchor') and pattern.anchor:
        for key, value in pattern.anchor.items():
            if key == 'pos':
                if not self._match_pos(anchor, value):
                    print(f"   ❌ Anchor POS mismatch: {anchor.pos_} vs {value}")
                    anchor_matches = False
            elif key == 'lemma':
                lemma_options = [v.strip() for v in value.split('|')]
                if anchor.lemma_ not in lemma_options:
                    print(f"   ❌ Anchor lemma mismatch: {anchor.lemma_} vs {lemma_options}")
                    anchor_matches = False

    if not anchor_matches:
        return []

    print(f"   ✅ Anchor matches")

    # Original method
    matches = [{'anchor': anchor}]

    for i, edge in enumerate(pattern.edges):
        print(f"   🔗 Edge {i+1}: from={edge['from']} rel={edge['rel']} as={edge['as']}")
        new_matches = []
        for match in matches:
            edge_matches = self._match_edge(match, edge, context)
            print(f"      Found {len(edge_matches)} edge matches")
            new_matches.extend(edge_matches)
        matches = new_matches

        if not matches:
            print(f"   ❌ Edge {i+1} failed - no matches")
            break
        else:
            print(f"   ✅ Edge {i+1} passed - {len(matches)} total matches")

    # Apply guards
    if matches and pattern.guards:
        print(f"   🛡️ Testing {len(pattern.guards)} guards...")
        original_count = len(matches)
        matches = [m for m in matches if self._check_guards(m, pattern.guards, context)]
        print(f"   Guards: {original_count} -> {len(matches)} matches")

    print(f"   Final: {len(matches)} matches for pattern '{pattern.name}'")
    return matches

def debug_extraction():
    nlp = spacy.load('en_core_web_sm')
    doc = nlp('My name is Alex Thompson.')

    extractor = YAMLUDExtractor('enhanced_fastlane_rules.ud.yaml')

    # Monkey patch for debugging
    extractor.matcher._match_pattern = lambda anchor, pattern, context: debug_match_pattern(extractor.matcher, anchor, pattern, context)

    print("🚀 Starting extraction with debug...")
    result = extractor.extract_triples(doc)
    print(f"\n🎯 Final result: {result}")

if __name__ == "__main__":
    debug_extraction()