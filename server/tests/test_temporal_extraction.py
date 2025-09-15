#!/usr/bin/env python3
"""Test temporal/date extraction capabilities"""

from level3_universal_kg import UniversalKGExtractor
import time

def test_temporal_extraction():
    extractor = UniversalKGExtractor()

    # Test cases with various temporal expressions
    test_cases = [
        ("Basic temporal", "Yesterday, firefighters quickly responded to the emergency call."),
        ("Date mentions", "The meeting is scheduled for March 15th, 2024 at 3:30 PM."),
        ("Time expressions", "Last week, the project was completed ahead of schedule."),
        ("Complex temporal", "On Monday morning at 9 AM, after the weekend break, the team reconvened for the quarterly review."),
        ("Relative times", "Three hours ago, before the deadline, she submitted the final report."),
        ("Real dates", "The conference will be held on January 20, 2025, from 2:00 to 5:00 PM EST."),
    ]

    print("🕐 TEMPORAL/DATE EXTRACTION TEST")
    print("=" * 60)

    for i, (category, text) in enumerate(test_cases, 1):
        print(f"\n📅 TEST {i}: {category}")
        print("─" * 40)
        print(f"Input: \"{text}\"")
        print()

        # Extract knowledge graph
        start_time = time.time()
        kg = extractor.extract_universal_kg(text)
        extraction_time = (time.time() - start_time) * 1000

        print(f"⚡ Performance: {extraction_time:.1f}ms")

        # Show temporal entities
        temporal_entities = [e for e in kg.entities if 'temporal' in e.entity_type.lower() or
                           any(word in e.text.lower() for word in ['yesterday', 'today', 'tomorrow',
                                                                   'morning', 'afternoon', 'pm', 'am',
                                                                   'monday', 'tuesday', 'march', 'january',
                                                                   'ago', 'week', 'month', 'year', 'hour'])]

        if temporal_entities:
            print(f"🕐 TEMPORAL ENTITIES ({len(temporal_entities)}):")
            for j, entity in enumerate(temporal_entities, 1):
                print(f"  {j}. \"{entity.text}\" (type: {entity.entity_type})")
        else:
            print("❌ No temporal entities found")

        # Show temporal relations
        temporal_relations = [r for r in kg.relations if
                             'temporal' in r.predicate.lower() or
                             'yesterday' in r.subject.lower() or 'yesterday' in r.object.lower() or
                             any(word in r.predicate for word in ['has_temporal', 'occurs_on', 'scheduled_for'])]

        if temporal_relations:
            print(f"🕐 TEMPORAL RELATIONS ({len(temporal_relations)}):")
            for j, relation in enumerate(temporal_relations, 1):
                print(f"  {j}. {relation.subject} | {relation.predicate} | {relation.object}")
        else:
            print("❌ No temporal relations found")

        # Show all beautiful relations for context
        beautiful_relations = [r for r in kg.relations
                             if ('has_attribute' not in r.predicate and
                                 'modifies' not in r.predicate and
                                 'participates_in' not in r.predicate and
                                 'type' not in r.predicate)]

        if beautiful_relations:
            print(f"\n🌟 ALL BEAUTIFUL RELATIONS ({len(beautiful_relations)}):")
            for j, relation in enumerate(beautiful_relations, 1):
                print(f"  {j}. {relation.subject} | {relation.predicate} | {relation.object}")

        print()

if __name__ == "__main__":
    test_temporal_extraction()