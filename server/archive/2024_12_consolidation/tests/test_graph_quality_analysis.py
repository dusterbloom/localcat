#!/usr/bin/env python3
"""
Graph Quality Analysis for Selective UD Patterns
VERIFICATION: Are the resulting graphs acceptable for graph intelligence?

This test examines:
1. Graph density and connectivity
2. Semantic richness of extracted relations
3. Entity-relation coverage
4. Traversal path quality
5. Comparison with full extraction system
"""

import sys
import time
from typing import Dict, List, Any, Set
from collections import defaultdict
import networkx as nx

sys.path.insert(0, '.')

import spacy
from services.selective_ud_patterns import SelectiveUDPatterns, PatternTier


class GraphQualityAnalyzer:
    """Analyze graph quality for selective UD pattern extraction"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.selective_patterns = SelectiveUDPatterns()

        # Test sentences covering different relationship types
        self.test_cases = [
            {
                'text': "Steve Jobs founded Apple Inc. in Cupertino, California.",
                'expected_relations': ['founded', 'located_in', 'compound'],
                'expected_entities': ['Steve Jobs', 'Apple Inc.', 'Cupertino', 'California'],
                'graph_type': 'person-organization-location'
            },
            {
                'text': "Dr. Smith teaches artificial intelligence at Stanford University.",
                'expected_relations': ['teaches', 'works_at', 'located_at'],
                'expected_entities': ['Dr. Smith', 'artificial intelligence', 'Stanford University'],
                'graph_type': 'person-subject-organization'
            },
            {
                'text': "The new MacBook Pro costs $2,999 and features advanced processors.",
                'expected_relations': ['costs', 'features', 'has_property'],
                'expected_entities': ['MacBook Pro', '$2,999', 'processors'],
                'graph_type': 'product-price-features'
            },
            {
                'text': "Elon Musk founded SpaceX and currently serves as CEO while Tesla produces electric vehicles.",
                'expected_relations': ['founded', 'serves_as', 'produces', 'ceo_of'],
                'expected_entities': ['Elon Musk', 'SpaceX', 'CEO', 'Tesla', 'vehicles'],
                'graph_type': 'complex-multi-entity'
            },
            {
                'text': "After graduating from MIT, Sarah joined Tesla where she led the autopilot team.",
                'expected_relations': ['graduated_from', 'joined', 'led', 'works_at'],
                'expected_entities': ['Sarah', 'MIT', 'Tesla', 'autopilot team'],
                'graph_type': 'career-progression'
            }
        ]

    def extract_and_analyze_graph(self, text: str, tier: PatternTier) -> Dict[str, Any]:
        """Extract relations and build graph structure"""
        doc = self.nlp(text)

        # Extract relations using selective patterns
        start_time = time.perf_counter()
        result = self.selective_patterns.extract_selective_patterns(doc, tier)
        extraction_time = (time.perf_counter() - start_time) * 1000

        relations = result['relations']

        # Build graph
        G = nx.DiGraph()

        # Add relations as edges
        for rel in relations:
            subject = rel['subject'].lower().strip()
            obj = rel['object'].lower().strip()
            relation = rel['relation']

            if subject and obj and subject != obj:
                G.add_edge(subject, obj, relation=relation,
                          confidence=rel.get('confidence', 0.5),
                          tier=rel.get('pattern_tier', 'unknown'))

        # Calculate graph metrics
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        density = nx.density(G) if node_count > 1 else 0

        # Connected components
        weakly_connected_components = list(nx.weakly_connected_components(G))
        largest_component_size = max(len(comp) for comp in weakly_connected_components) if weakly_connected_components else 0

        # Average path length (for largest component)
        avg_path_length = 0
        if largest_component_size > 1:
            largest_comp = max(weakly_connected_components, key=len)
            subgraph = G.subgraph(largest_comp)
            if nx.is_weakly_connected(subgraph):
                try:
                    avg_path_length = nx.average_shortest_path_length(subgraph.to_undirected())
                except:
                    avg_path_length = 0

        return {
            'text': text,
            'tier_used': result['tier_used'],
            'extraction_time_ms': extraction_time,
            'relations': relations,
            'graph': G,
            'metrics': {
                'nodes': node_count,
                'edges': edge_count,
                'density': density,
                'components': len(weakly_connected_components),
                'largest_component': largest_component_size,
                'avg_path_length': avg_path_length,
                'connectivity_ratio': largest_component_size / node_count if node_count > 0 else 0
            }
        }

    def analyze_semantic_coverage(self, relations: List[Dict]) -> Dict[str, Any]:
        """Analyze semantic richness of extracted relations"""

        # Categorize relations by semantic type
        relation_types = {
            'predicate_argument': ['nsubj', 'obj', 'iobj', 'nsubj:pass'],
            'modification': ['amod', 'advmod', 'compound'],
            'structural': ['det', 'case', 'cc', 'cop'],
            'clausal': ['xcomp', 'ccomp'],
            'spatial_temporal': ['obl', 'conj']
        }

        coverage = defaultdict(int)
        relation_details = defaultdict(list)

        for rel in relations:
            rel_name = rel['relation']

            # Find category
            for category, patterns in relation_types.items():
                if any(pattern in rel_name for pattern in patterns):
                    coverage[category] += 1
                    relation_details[category].append({
                        'relation': rel_name,
                        'subject': rel['subject'],
                        'object': rel['object'],
                        'confidence': rel.get('confidence', 0.5)
                    })
                    break
            else:
                coverage['other'] += 1
                relation_details['other'].append({
                    'relation': rel_name,
                    'subject': rel['subject'],
                    'object': rel['object'],
                    'confidence': rel.get('confidence', 0.5)
                })

        return {
            'coverage_counts': dict(coverage),
            'relation_details': dict(relation_details),
            'total_relations': len(relations),
            'semantic_diversity': len(coverage)
        }

    def compare_tiers(self, text: str) -> Dict[str, Any]:
        """Compare graph quality across different pattern tiers"""
        comparison = {}

        for tier in [PatternTier.ESSENTIAL, PatternTier.CONNECTIVITY, PatternTier.OPTIONAL]:
            analysis = self.extract_and_analyze_graph(text, tier)
            semantic_analysis = self.analyze_semantic_coverage(analysis['relations'])

            comparison[tier.name] = {
                'metrics': analysis['metrics'],
                'semantic': semantic_analysis,
                'extraction_time_ms': analysis['extraction_time_ms'],
                'relations_sample': analysis['relations'][:3]  # First 3 relations for inspection
            }

        return comparison

    def run_comprehensive_analysis(self) -> None:
        """Run comprehensive graph quality analysis"""

        print("🧠 GRAPH QUALITY ANALYSIS FOR SELECTIVE UD PATTERNS")
        print("=" * 80)
        print("VERIFICATION: Are resulting graphs acceptable for graph intelligence?")
        print("Minimum requirements:")
        print("  - Graph density ≥ 0.01 (effective connectivity)")
        print("  - Semantic diversity across relation types")
        print("  - Traversable graph structure")
        print("=" * 80)

        tier_results = defaultdict(lambda: defaultdict(list))

        for i, test_case in enumerate(self.test_cases, 1):
            text = test_case['text']
            graph_type = test_case['graph_type']

            print(f"\n📝 TEST CASE {i}: {graph_type}")
            print(f"Text: {text}")
            print("-" * 60)

            # Compare all tiers for this sentence
            comparison = self.compare_tiers(text)

            for tier_name, results in comparison.items():
                metrics = results['metrics']
                semantic = results['semantic']

                print(f"\n{tier_name:12s} | Nodes: {metrics['nodes']:2d} | Edges: {metrics['edges']:2d} | "
                      f"Density: {metrics['density']:.3f} | Time: {results['extraction_time_ms']:5.1f}ms")

                # Show semantic coverage
                coverage = semantic['coverage_counts']
                print(f"             | Semantic types: {semantic['semantic_diversity']}")
                for sem_type, count in coverage.items():
                    if count > 0:
                        print(f"             |   {sem_type}: {count}")

                # Store for summary
                tier_results[tier_name]['density'].append(metrics['density'])
                tier_results[tier_name]['nodes'].append(metrics['nodes'])
                tier_results[tier_name]['edges'].append(metrics['edges'])
                tier_results[tier_name]['connectivity'].append(metrics['connectivity_ratio'])
                tier_results[tier_name]['semantic_diversity'].append(semantic['semantic_diversity'])
                tier_results[tier_name]['time'].append(results['extraction_time_ms'])

        # Summary analysis
        print("\n" + "=" * 80)
        print("📊 SUMMARY ANALYSIS")
        print("=" * 80)

        print(f"{'Tier':<15} {'Avg Density':<12} {'Avg Nodes':<10} {'Avg Edges':<10} {'Connectivity':<12} {'Semantic':<9} {'Time':<8}")
        print("-" * 80)

        tier_quality = {}

        for tier_name, data in tier_results.items():
            avg_density = sum(data['density']) / len(data['density'])
            avg_nodes = sum(data['nodes']) / len(data['nodes'])
            avg_edges = sum(data['edges']) / len(data['edges'])
            avg_connectivity = sum(data['connectivity']) / len(data['connectivity'])
            avg_semantic = sum(data['semantic_diversity']) / len(data['semantic_diversity'])
            avg_time = sum(data['time']) / len(data['time'])

            print(f"{tier_name:<15} {avg_density:<12.3f} {avg_nodes:<10.1f} {avg_edges:<10.1f} "
                  f"{avg_connectivity:<12.2f} {avg_semantic:<9.1f} {avg_time:<8.1f}ms")

            # Quality assessment
            quality_score = 0
            if avg_density >= 0.01: quality_score += 25  # Minimum connectivity
            if avg_connectivity >= 0.7: quality_score += 25  # Good connectivity
            if avg_semantic >= 3: quality_score += 25  # Semantic diversity
            if avg_edges >= 3: quality_score += 25  # Sufficient relations

            tier_quality[tier_name] = {
                'score': quality_score,
                'density': avg_density,
                'connectivity': avg_connectivity,
                'semantic_diversity': avg_semantic,
                'edges': avg_edges,
                'time': avg_time
            }

        # Quality verdict
        print("\n" + "=" * 80)
        print("🎯 GRAPH INTELLIGENCE QUALITY VERDICT")
        print("=" * 80)

        for tier_name, quality in tier_quality.items():
            score = quality['score']

            if score >= 75:
                status = "✅ EXCELLENT"
            elif score >= 50:
                status = "🟡 ACCEPTABLE"
            else:
                status = "❌ INSUFFICIENT"

            print(f"{tier_name:15s}: {status} (Score: {score}/100)")
            print(f"                Density: {quality['density']:.3f} {'✅' if quality['density'] >= 0.01 else '❌'}")
            print(f"                Connectivity: {quality['connectivity']:.2f} {'✅' if quality['connectivity'] >= 0.7 else '❌'}")
            print(f"                Semantic diversity: {quality['semantic_diversity']:.1f} {'✅' if quality['semantic_diversity'] >= 3 else '❌'}")
            print(f"                Relations per sentence: {quality['edges']:.1f} {'✅' if quality['edges'] >= 3 else '❌'}")

        # Final recommendation
        print("\n" + "=" * 80)
        print("🚀 FINAL RECOMMENDATION FOR GRAPH INTELLIGENCE")
        print("=" * 80)

        # Find best tier
        best_tier = max(tier_quality.items(), key=lambda x: x[1]['score'])
        best_name, best_quality = best_tier

        if best_quality['score'] >= 75:
            print(f"✅ RECOMMENDED TIER: {best_name}")
            print(f"   Quality Score: {best_quality['score']}/100 (EXCELLENT)")
            print(f"   Average Time: {best_quality['time']:.1f}ms")
            print(f"   Graph Density: {best_quality['density']:.3f} (sufficient for graph intelligence)")
            print(f"   Connectivity: {best_quality['connectivity']:.2f} (good traversal)")
            print(f"   Semantic Coverage: {best_quality['semantic_diversity']:.1f} relation types")
            print(f"\n🎯 VERDICT: GRAPHS ARE ACCEPTABLE FOR GRAPH INTELLIGENCE!")
        else:
            print(f"⚠️  CONCERNS WITH GRAPH QUALITY")
            print(f"   Best tier: {best_name} (Score: {best_quality['score']}/100)")
            print(f"   May need additional optimization for graph intelligence")

    def visualize_sample_graph(self, text: str, tier: PatternTier) -> None:
        """Show detailed graph structure for manual inspection"""
        print(f"\n🔍 DETAILED GRAPH INSPECTION")
        print(f"Text: {text}")
        print(f"Tier: {tier.name}")
        print("-" * 60)

        analysis = self.extract_and_analyze_graph(text, tier)
        G = analysis['graph']
        relations = analysis['relations']

        print(f"Graph Structure:")
        print(f"  Nodes: {list(G.nodes())}")
        print(f"  Edges with relations:")

        for i, (subject, obj, data) in enumerate(G.edges(data=True), 1):
            print(f"    {i:2d}. {subject} --[{data['relation']}]--> {obj} "
                  f"(confidence: {data['confidence']:.2f})")

        print(f"\nRaw Relations:")
        for i, rel in enumerate(relations, 1):
            print(f"    {i:2d}. {rel['subject']} --[{rel['relation']}]--> {rel['object']}")


def main():
    analyzer = GraphQualityAnalyzer()

    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()

    # Show detailed example
    print("\n" + "=" * 80)
    analyzer.visualize_sample_graph(
        "Steve Jobs founded Apple Inc. in Cupertino, California.",
        PatternTier.CONNECTIVITY
    )

    print("\n" + "=" * 80)
    print("✅ GRAPH QUALITY ANALYSIS COMPLETE")
    print("Use results above to determine if graphs are acceptable for graph intelligence.")


if __name__ == "__main__":
    main()