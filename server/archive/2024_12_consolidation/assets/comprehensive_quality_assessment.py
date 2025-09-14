#!/usr/bin/env python3
"""
Comprehensive Quality Assessment - 10 Difficulty Levels
Using existing proven test cases to evaluate when UD vs GLiREL vs Full Pipeline excels

Focus: Quality over quantity - real semantic understanding and coreference resolution
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.extraction.memory_extractor import MemoryExtractor

@dataclass
class QualityMetrics:
    level: int
    description: str
    entities: List[str]
    triples: List[Tuple[str, str, str]]
    time_ms: float
    semantic_quality: float  # 0.0-1.0 score for semantic understanding
    coreference_quality: float  # 0.0-1.0 score for coreference resolution
    temporal_quality: float  # 0.0-1.0 score for temporal relations
    duplicate_penalty: float  # 0.0-1.0 penalty for duplicates

def comprehensive_quality_assessment():
    """
    Progressive difficulty assessment using existing proven test cases
    10 levels from simple to highly complex
    """

    print("🧠 COMPREHENSIVE QUALITY ASSESSMENT")
    print("=" * 80)
    print("Progressive difficulty: 10 levels from simple to complex")
    print("Focus: Semantic understanding, coreference resolution, temporal relations")
    print("Goal: Determine when UD vs GLiREL vs Full Pipeline excels")
    print()

    # 10 Difficulty Levels - using proven test cases from existing tests
    difficulty_levels = [
        {
            'level': 1,
            'description': 'Simple Person-Location',
            'text': 'I live in New York.',
            'complexity': 'Basic entity recognition',
            'expected_quality': {
                'entities': ['I', 'New York'],
                'semantic_relations': ['lives_in'],
                'coreferences': [],
                'temporal': []
            }
        },
        {
            'level': 2,
            'description': 'Person-Organization-Action',
            'text': 'Sarah graduated from Stanford University last year.',
            'complexity': 'Simple past action with temporal',
            'expected_quality': {
                'entities': ['Sarah', 'Stanford University', 'last year'],
                'semantic_relations': ['graduated_from'],
                'coreferences': [],
                'temporal': ['last year']
            }
        },
        {
            'level': 3,
            'description': 'Multi-Entity Professional',
            'text': 'Steve Jobs founded Apple Inc. in Cupertino.',
            'complexity': 'Professional relationship with location',
            'expected_quality': {
                'entities': ['Steve Jobs', 'Apple Inc.', 'Cupertino'],
                'semantic_relations': ['founded', 'located_in'],
                'coreferences': [],
                'temporal': []
            }
        },
        {
            'level': 4,
            'description': 'Product-Company Relationship',
            'text': 'Tesla Model S is produced by Tesla Motors in Fremont, California.',
            'complexity': 'Product attribution with nested location',
            'expected_quality': {
                'entities': ['Tesla Model S', 'Tesla Motors', 'Fremont', 'California'],
                'semantic_relations': ['produced_by', 'located_in'],
                'coreferences': ['Tesla -> Tesla Motors'],
                'temporal': []
            }
        },
        {
            'level': 5,
            'description': 'Temporal Succession',
            'text': 'Tim Cook became CEO of Apple in 2011 after Steve Jobs resigned.',
            'complexity': 'Temporal sequence with role transition',
            'expected_quality': {
                'entities': ['Tim Cook', 'Apple', 'Steve Jobs', '2011'],
                'semantic_relations': ['became', 'ceo_of', 'resigned', 'after'],
                'coreferences': [],
                'temporal': ['in 2011', 'after']
            }
        },
        {
            'level': 6,
            'description': 'Complex Discovery Narrative',
            'text': 'Marie Curie discovered radium in Paris while working at the Sorbonne University. She won the Nobel Prize in Physics.',
            'complexity': 'Multi-sentence with coreference and achievements',
            'expected_quality': {
                'entities': ['Marie Curie', 'radium', 'Paris', 'Sorbonne University', 'Nobel Prize', 'Physics'],
                'semantic_relations': ['discovered', 'located_in', 'worked_at', 'won'],
                'coreferences': ['She -> Marie Curie'],
                'temporal': ['while working']
            }
        },
        {
            'level': 7,
            'description': 'Corporate Acquisition Chain',
            'text': 'Elon Musk, the CEO of Tesla and SpaceX, was born in South Africa and later moved to Silicon Valley where he founded PayPal with Peter Thiel.',
            'complexity': 'Multiple roles, locations, and collaborations',
            'expected_quality': {
                'entities': ['Elon Musk', 'Tesla', 'SpaceX', 'South Africa', 'Silicon Valley', 'PayPal', 'Peter Thiel'],
                'semantic_relations': ['ceo_of', 'born_in', 'moved_to', 'founded', 'collaborated_with'],
                'coreferences': ['he -> Elon Musk'],
                'temporal': ['later']
            }
        },
        {
            'level': 8,
            'description': 'Technology Evolution',
            'text': 'The iPhone 15 Pro, manufactured by Apple Inc. in Cupertino, California, costs 999 dollars and competes with Samsung Galaxy S24 Ultra produced in South Korea.',
            'complexity': 'Product competition with pricing and geographic manufacturing',
            'expected_quality': {
                'entities': ['iPhone 15 Pro', 'Apple Inc.', 'Cupertino', 'California', '999 dollars', 'Samsung Galaxy S24 Ultra', 'South Korea'],
                'semantic_relations': ['manufactured_by', 'located_in', 'costs', 'competes_with', 'produced_in'],
                'coreferences': [],
                'temporal': []
            }
        },
        {
            'level': 9,
            'description': 'Multi-Company Market Analysis',
            'text': 'Microsoft Corporation acquired GitHub for 7.5 billion dollars in 2018, while Amazon Web Services competed with Google Cloud Platform and Microsoft Azure for the enterprise cloud market.',
            'complexity': 'Financial acquisition with competitive landscape',
            'expected_quality': {
                'entities': ['Microsoft Corporation', 'GitHub', '7.5 billion dollars', '2018', 'Amazon Web Services', 'Google Cloud Platform', 'Microsoft Azure', 'enterprise cloud market'],
                'semantic_relations': ['acquired', 'for_amount', 'in_year', 'competed_with', 'for_market'],
                'coreferences': ['Microsoft appears twice'],
                'temporal': ['in 2018', 'while']
            }
        },
        {
            'level': 10,
            'description': 'Complex Professional Transition',
            'text': 'Dr. Sarah Chen, the former research director at Google DeepMind, left her position at the London office to join OpenAI in San Francisco, where she now leads the GPT-4 development team alongside Sam Altman and Greg Brockman.',
            'complexity': 'Career transition with geographic move, role change, and team composition',
            'expected_quality': {
                'entities': ['Dr. Sarah Chen', 'Google DeepMind', 'London office', 'OpenAI', 'San Francisco', 'GPT-4 development team', 'Sam Altman', 'Greg Brockman'],
                'semantic_relations': ['former_director_at', 'left_position', 'joined', 'leads', 'works_alongside'],
                'coreferences': ['she -> Dr. Sarah Chen'],
                'temporal': ['former', 'now']
            }
        }
    ]

    # Test configurations
    configurations = [
        {
            'name': 'UD Patterns Only',
            'config': {
                'use_glirel': False,
                'use_gliner': False,
                'sqlite_path': ':memory:',
                'session_id': 'test_ud_only'
            },
            'strength': 'Grammatical relations, speed'
        },
        {
            'name': 'Tier 1 (UD + GLiNER)',
            'config': {
                'use_glirel': False,
                'use_gliner': True,
                'sqlite_path': ':memory:',
                'session_id': 'test_tier1'
            },
            'strength': 'Entity recognition + grammatical relations'
        },
        {
            'name': 'Full Pipeline (UD + GLiNER + GLiREL)',
            'config': {
                'use_glirel': True,
                'use_gliner': True,
                'sqlite_path': ':memory:',
                'session_id': 'test_full'
            },
            'strength': 'Semantic relations + entity recognition + grammar'
        }
    ]

    all_results = {}

    for config_info in configurations:
        config_name = config_info['name']
        config = config_info['config']
        strength = config_info['strength']

        print(f"\n🔬 Testing: {config_name}")
        print(f"   Strength: {strength}")
        print("-" * 60)

        try:
            extractor = MemoryExtractor(config)
            config_results = []

            for test_case in difficulty_levels:
                level = test_case['level']
                description = test_case['description']
                text = test_case['text']
                complexity = test_case['complexity']
                expected = test_case['expected_quality']

                print(f"\n  📊 Level {level}: {description}")
                print(f"      Text: '{text}'")
                print(f"      Complexity: {complexity}")

                # Time the extraction
                start_time = time.perf_counter()
                result = extractor.extract(text)
                extraction_time = (time.perf_counter() - start_time) * 1000

                # Analyze quality
                entities = [str(e).lower().strip() for e in result.entities]
                triples = [(str(h).lower().strip(), str(r).lower().strip(), str(t).lower().strip())
                          for h, r, t in result.triples]

                # Calculate quality scores
                semantic_quality = calculate_semantic_quality(triples, expected['semantic_relations'])
                coreference_quality = calculate_coreference_quality(triples, expected['coreferences'])
                temporal_quality = calculate_temporal_quality(triples, expected['temporal'])
                duplicate_penalty = calculate_duplicate_penalty(entities, triples)

                overall_quality = (semantic_quality + coreference_quality + temporal_quality) / 3 * (1 - duplicate_penalty)

                metrics = QualityMetrics(
                    level=level,
                    description=description,
                    entities=entities,
                    triples=triples,
                    time_ms=extraction_time,
                    semantic_quality=semantic_quality,
                    coreference_quality=coreference_quality,
                    temporal_quality=temporal_quality,
                    duplicate_penalty=duplicate_penalty
                )

                config_results.append(metrics)

                # Display results
                print(f"      ⏱️  Time: {extraction_time:.1f}ms")
                print(f"      👥 Entities: {len(entities)}")
                print(f"      🔗 Relations: {len(triples)}")
                print(f"      🧠 Semantic Quality: {semantic_quality:.1%}")
                print(f"      🔄 Coreference Quality: {coreference_quality:.1%}")
                print(f"      ⏰ Temporal Quality: {temporal_quality:.1%}")
                print(f"      ⚠️  Duplicate Penalty: {duplicate_penalty:.1%}")
                print(f"      📈 Overall Quality: {overall_quality:.1%}")

                # Show top 3 meaningful relations
                meaningful_relations = get_meaningful_relations(triples)
                if meaningful_relations:
                    print(f"      📋 Top Relations:")
                    for i, (h, r, t) in enumerate(meaningful_relations[:3], 1):
                        print(f"         {i}. {h} --{r}--> {t}")

            all_results[config_name] = config_results

        except Exception as e:
            print(f"❌ Configuration {config_name} failed: {e}")
            continue

    # Comprehensive Analysis
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE QUALITY ANALYSIS")
    print("=" * 80)

    analyze_configuration_performance(all_results)
    analyze_difficulty_progression(all_results)
    provide_usage_recommendations(all_results)

    return all_results

def calculate_semantic_quality(triples: List[Tuple[str, str, str]], expected_relations: List[str]) -> float:
    """Calculate how well semantic relations are captured"""
    if not expected_relations:
        return 1.0  # No expectations, full score

    semantic_relations = ['founded', 'discovered', 'works_at', 'produced_by', 'acquired', 'competed_with', 'leads']
    found_semantic = sum(1 for _, rel, _ in triples if any(sem in rel for sem in semantic_relations))

    # Score based on semantic relation coverage
    expected_count = len(expected_relations)
    semantic_coverage = min(found_semantic / expected_count, 1.0) if expected_count > 0 else 0

    return semantic_coverage

def calculate_coreference_quality(triples: List[Tuple[str, str, str]], expected_coreferences: List[str]) -> float:
    """Calculate coreference resolution quality"""
    if not expected_coreferences:
        return 1.0  # No coreferences expected

    # Look for pronoun resolution in entities/relations
    pronouns = ['he', 'she', 'it', 'they', 'his', 'her', 'its', 'their']
    found_pronouns = sum(1 for h, r, t in triples if any(pron in h or pron in t for pron in pronouns))

    # Fewer unresolved pronouns = better coreference resolution
    if found_pronouns == 0:
        return 1.0  # Perfect - no unresolved pronouns
    else:
        return max(0, 1.0 - (found_pronouns * 0.2))  # Penalty for unresolved pronouns

def calculate_temporal_quality(triples: List[Tuple[str, str, str]], expected_temporal: List[str]) -> float:
    """Calculate temporal relation quality"""
    if not expected_temporal:
        return 1.0  # No temporal expectations

    temporal_indicators = ['in', 'after', 'before', 'during', 'while', 'when', 'since', 'until']
    temporal_relations = ['temporal', 'before', 'after', 'during', 'when']

    found_temporal = sum(1 for _, rel, _ in triples
                        if any(temp in rel for temp in temporal_relations)
                        or any(temp in rel for temp in temporal_indicators))

    expected_count = len(expected_temporal)
    temporal_coverage = min(found_temporal / expected_count, 1.0) if expected_count > 0 else 1.0

    return temporal_coverage

def calculate_duplicate_penalty(entities: List[str], triples: List[Tuple[str, str, str]]) -> float:
    """Calculate penalty for duplicates and low-quality extractions"""
    total_items = len(entities) + len(triples)
    if total_items == 0:
        return 0.0

    # Entity duplicates
    unique_entities = len(set(entities))
    entity_duplicate_ratio = 1.0 - (unique_entities / len(entities)) if entities else 0.0

    # Triple duplicates
    unique_triples = len(set(triples))
    triple_duplicate_ratio = 1.0 - (unique_triples / len(triples)) if triples else 0.0

    # Very short entity penalty (likely tokenization errors)
    short_entities = sum(1 for e in entities if len(e.strip()) <= 2)
    short_entity_ratio = short_entities / len(entities) if entities else 0.0

    # Combine penalties
    total_penalty = (entity_duplicate_ratio + triple_duplicate_ratio + short_entity_ratio) / 3
    return min(total_penalty, 0.5)  # Cap at 50% penalty

def get_meaningful_relations(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Filter to show only meaningful, non-trivial relations"""
    meaningful = []
    skip_relations = ['compound_with', 'compound', 'det', 'aux', 'case']

    for h, r, t in triples:
        # Skip very short tokens (likely parsing errors)
        if len(h.strip()) <= 2 or len(t.strip()) <= 2:
            continue
        # Skip trivial grammatical relations
        if any(skip in r for skip in skip_relations):
            continue
        meaningful.append((h, r, t))

    return meaningful

def analyze_configuration_performance(all_results: Dict):
    """Analyze how each configuration performs across difficulty levels"""
    print("\n🎯 CONFIGURATION PERFORMANCE ANALYSIS")
    print("-" * 60)

    for config_name, results in all_results.items():
        if not results:
            continue

        avg_time = sum(r.time_ms for r in results) / len(results)
        avg_semantic = sum(r.semantic_quality for r in results) / len(results)
        avg_coreference = sum(r.coreference_quality for r in results) / len(results)
        avg_temporal = sum(r.temporal_quality for r in results) / len(results)
        avg_penalty = sum(r.duplicate_penalty for r in results) / len(results)

        overall_quality = (avg_semantic + avg_coreference + avg_temporal) / 3 * (1 - avg_penalty)

        print(f"\n📊 {config_name}:")
        print(f"   ⏱️  Average Time: {avg_time:.1f}ms")
        print(f"   🧠 Semantic Quality: {avg_semantic:.1%}")
        print(f"   🔄 Coreference Quality: {avg_coreference:.1%}")
        print(f"   ⏰ Temporal Quality: {avg_temporal:.1%}")
        print(f"   ⚠️  Duplicate Penalty: {avg_penalty:.1%}")
        print(f"   📈 Overall Quality: {overall_quality:.1%}")

def analyze_difficulty_progression(all_results: Dict):
    """Analyze how quality degrades with difficulty"""
    print(f"\n📈 DIFFICULTY PROGRESSION ANALYSIS")
    print("-" * 60)

    for level in range(1, 11):
        print(f"\n🔢 Level {level}:")
        for config_name, results in all_results.items():
            level_result = next((r for r in results if r.level == level), None)
            if level_result:
                overall_quality = (level_result.semantic_quality +
                                 level_result.coreference_quality +
                                 level_result.temporal_quality) / 3 * (1 - level_result.duplicate_penalty)
                print(f"   {config_name}: {overall_quality:.1%} quality, {level_result.time_ms:.1f}ms")

def provide_usage_recommendations(all_results: Dict):
    """Provide specific recommendations for when to use each approach"""
    print(f"\n💡 USAGE RECOMMENDATIONS")
    print("-" * 60)

    print(f"\n🎯 When to use UD Patterns Only:")
    print(f"   • Simple grammatical analysis needed")
    print(f"   • Speed is critical (<50ms)")
    print(f"   • Levels 1-3: Basic entity-relation extraction")

    print(f"\n🎯 When to use Tier 1 (UD + GLiNER):")
    print(f"   • Balanced speed and quality needed")
    print(f"   • Good entity recognition required")
    print(f"   • Levels 1-6: Most conversational scenarios")

    print(f"\n🎯 When to use Full Pipeline (UD + GLiNER + GLiREL):")
    print(f"   • Maximum semantic understanding needed")
    print(f"   • Complex business relationships")
    print(f"   • Levels 7-10: Complex multi-entity scenarios")
    print(f"   • When semantic relations outweigh speed concerns")

if __name__ == "__main__":
    try:
        results = comprehensive_quality_assessment()
        print(f"\n🎉 Comprehensive quality assessment completed!")
        print(f"📊 Use results to make informed decisions about extraction strategy")

    except Exception as e:
        print(f"❌ Assessment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)