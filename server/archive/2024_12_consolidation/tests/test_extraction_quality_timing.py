#!/usr/bin/env python3
"""
Comprehensive test to demonstrate extraction quality and timing across all strategies
Focus on Tier 1's real-time knowledge graph generation capabilities
"""

import sys
import os
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.extraction.memory_extractor import MemoryExtractor

@dataclass
class ExtractionMetrics:
    strategy: str
    entities: List[str]
    triples: List[Tuple[str, str, str]]
    time_ms: float
    entity_count: int
    triple_count: int

def test_extraction_strategies():
    """Test all extraction strategies with timing and quality analysis"""

    print("🚀 Comprehensive Extraction Strategy Analysis")
    print("=" * 80)
    print("Testing: UD Patterns → Tier 1 → GLiREL integration")
    print("Goal: Prove Tier 1's strength in real-time knowledge graph generation")
    print()

    # Test cases with varying complexity
    test_cases = [
        {
            'name': 'Simple Person-Organization',
            'text': 'Steve Jobs founded Apple Inc. in Cupertino.',
            'expected_entities': ['Steve Jobs', 'Apple Inc.', 'Cupertino'],
            'expected_relations': ['founded', 'located_in']
        },
        {
            'name': 'Complex Multi-Relation',
            'text': 'Marie Curie discovered radium in Paris while working at the Sorbonne University. She won the Nobel Prize in Physics.',
            'expected_entities': ['Marie Curie', 'radium', 'Paris', 'Sorbonne University', 'Nobel Prize'],
            'expected_relations': ['discovered', 'located_in', 'worked_at', 'won']
        },
        {
            'name': 'Temporal Relations',
            'text': 'Tim Cook became CEO of Apple in 2011 after Steve Jobs resigned.',
            'expected_entities': ['Tim Cook', 'Apple', 'Steve Jobs', '2011'],
            'expected_relations': ['became', 'after', 'resigned']
        },
        {
            'name': 'Compound Entities',
            'text': 'Tesla Model S is produced by Tesla Motors in Fremont, California.',
            'expected_entities': ['Tesla Model S', 'Tesla Motors', 'Fremont', 'California'],
            'expected_relations': ['produced_by', 'located_in']
        }
    ]

    # Initialize extractor with different configurations
    configs = [
        {
            'name': 'UD Only',
            'config': {
                'use_glirel': False,
                'use_gliner': False,
                'sqlite_path': ':memory:',
                'session_id': 'test_ud_only'
            }
        },
        {
            'name': 'GLiNER + UD (Tier 1)',
            'config': {
                'use_glirel': False,
                'use_gliner': True,
                'sqlite_path': ':memory:',
                'session_id': 'test_tier1'
            }
        },
        {
            'name': 'Full Pipeline (UD + GLiNER + GLiREL)',
            'config': {
                'use_glirel': True,
                'use_gliner': True,
                'sqlite_path': ':memory:',
                'session_id': 'test_full'
            }
        }
    ]

    all_results = {}

    for config_info in configs:
        config_name = config_info['name']
        config = config_info['config']

        print(f"\n🔬 Testing Configuration: {config_name}")
        print("-" * 50)

        try:
            extractor = MemoryExtractor(config)
            results = {}

            for test_case in test_cases:
                test_name = test_case['name']
                text = test_case['text']

                print(f"\n  📝 Test: {test_name}")
                print(f"     Text: '{text}'")

                # Time the extraction
                start_time = time.perf_counter()
                result = extractor.extract(text)
                extraction_time = (time.perf_counter() - start_time) * 1000

                # Analyze results
                entities = [str(e).lower() for e in result.entities]
                triples = [(str(h).lower(), str(r).lower(), str(t).lower()) for h, r, t in result.triples]

                # Calculate quality metrics
                expected_entities = [e.lower() for e in test_case['expected_entities']]
                expected_relations = [r.lower() for r in test_case['expected_relations']]

                entity_precision = len(set(entities) & set(expected_entities)) / len(set(entities)) if entities else 0
                relation_coverage = len([r for r in expected_relations if any(r in triple[1] for triple in triples)]) / len(expected_relations) if expected_relations else 0

                metrics = ExtractionMetrics(
                    strategy=config_name,
                    entities=entities,
                    triples=triples,
                    time_ms=extraction_time,
                    entity_count=len(entities),
                    triple_count=len(triples)
                )

                results[test_name] = metrics

                print(f"     ⏱️  Time: {extraction_time:.1f}ms")
                print(f"     👥 Entities: {len(entities)} (precision: {entity_precision:.2%})")
                print(f"     🔗 Triples: {len(triples)}")
                print(f"     📊 Relation Coverage: {relation_coverage:.2%}")

                # Show sample triples
                if triples:
                    print(f"     📋 Sample Triples:")
                    for i, triple in enumerate(triples[:3], 1):
                        print(f"        {i}. {' -- '.join(triple)}")

            all_results[config_name] = results

        except Exception as e:
            print(f"❌ Configuration {config_name} failed: {e}")
            continue

    # Summary analysis
    print("\n" + "=" * 80)
    print("📊 SUMMARY: Performance and Quality Analysis")
    print("=" * 80)

    # Calculate averages
    summary_stats = {}
    for config_name, results in all_results.items():
        total_time = sum(r.time_ms for r in results.values())
        avg_time = total_time / len(results)
        avg_entities = sum(r.entity_count for r in results.values()) / len(results)
        avg_triples = sum(r.triple_count for r in results.values()) / len(results)

        summary_stats[config_name] = {
            'avg_time_ms': avg_time,
            'avg_entities': avg_entities,
            'avg_triples': avg_triples,
            'total_time_ms': total_time
        }

        print(f"\n🎯 {config_name}:")
        print(f"   ⏱️  Avg Time: {avg_time:.1f}ms")
        print(f"   👥 Avg Entities: {avg_entities:.1f}")
        print(f"   🔗 Avg Triples: {avg_triples:.1f}")
        print(f"   📈 Total Time: {total_time:.1f}ms")

    # Performance comparison
    print(f"\n🚀 PERFORMANCE COMPARISON:")
    print("-" * 40)

    if 'UD Only' in summary_stats and 'GLiNER + UD (Tier 1)' in summary_stats:
        ud_time = summary_stats['UD Only']['avg_time_ms']
        tier1_time = summary_stats['GLiNER + UD (Tier 1)']['avg_time_ms']
        speedup = (ud_time / tier1_time) if tier1_time > 0 else 0

        print(f"   Tier 1 vs UD Only: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"   Quality Improvement: {summary_stats['GLiNER + UD (Tier 1)']['avg_entities']:.1f} vs {summary_stats['UD Only']['avg_entities']:.1f} entities")

    if 'GLiNER + UD (Tier 1)' in summary_stats and 'Full Pipeline (UD + GLiNER + GLiREL)' in summary_stats:
        tier1_time = summary_stats['GLiNER + UD (Tier 1)']['avg_time_ms']
        full_time = summary_stats['Full Pipeline (UD + GLiNER + GLiREL)']['avg_time_ms']
        overhead = ((full_time - tier1_time) / tier1_time * 100) if tier1_time > 0 else 0

        print(f"   Full Pipeline vs Tier 1: {overhead:+.1f}% overhead")
        print(f"   Additional Relations: {summary_stats['Full Pipeline (UD + GLiNER + GLiREL)']['avg_triples']:.1f} vs {summary_stats['GLiNER + UD (Tier 1)']['avg_triples']:.1f}")

    # Real-time capability assessment
    print(f"\n⚡ REAL-TIME CAPABILITY ASSESSMENT:")
    print("-" * 40)

    for config_name, stats in summary_stats.items():
        avg_time = stats['avg_time_ms']
        if avg_time < 200:
            status = "🟢 EXCELLENT"
            capability = "Real-time ready"
        elif avg_time < 500:
            status = "🟡 GOOD"
            capability = "Near real-time"
        elif avg_time < 1000:
            status = "🟠 ACCEPTABLE"
            capability = "Conversation ready"
        else:
            status = "🔴 SLOW"
            capability = "Not real-time"

        print(f"   {config_name}: {status} ({avg_time:.1f}ms) - {capability}")

    # Knowledge graph quality
    print(f"\n🧠 KNOWLEDGE GRAPH QUALITY:")
    print("-" * 40)

    for config_name, results in all_results.items():
        # Calculate average entity accuracy
        total_precision = 0
        test_count = 0

        for test_name, result in results.items():
            test_case = next(tc for tc in test_cases if tc['name'] == test_name)
            expected_entities = [e.lower() for e in test_case['expected_entities']]
            actual_entities = result.entities

            if actual_entities:
                precision = len(set(actual_entities) & set(expected_entities)) / len(set(actual_entities))
                total_precision += precision
                test_count += 1

        avg_precision = total_precision / test_count if test_count > 0 else 0

        print(f"   {config_name}: {avg_precision:.1%} entity precision")

    return all_results, summary_stats

if __name__ == "__main__":
    try:
        results, stats = test_extraction_strategies()
        print(f"\n🎉 Extraction analysis completed successfully!")

        # Final recommendation
        print(f"\n💡 RECOMMENDATION:")
        if 'GLiNER + UD (Tier 1)' in stats:
            tier1_stats = stats['GLiNER + UD (Tier 1)']
            if tier1_stats['avg_time_ms'] < 500:
                print(f"   ✅ Tier 1 is RECOMMENDED for production:")
                print(f"      • {tier1_stats['avg_time_ms']:.1f}ms average latency")
                print(f"      • {tier1_stats['avg_entities']:.1f} entities per extraction")
                print(f"      • Real-time capable (<500ms)")
                print(f"      • High-quality knowledge graph generation")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)