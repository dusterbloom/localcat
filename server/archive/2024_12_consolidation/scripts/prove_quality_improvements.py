"""
Proof of Quality Improvements
=============================

Show concrete evidence of:
1. Entity deduplication working
2. Coreference resolution working
3. Graph connectivity improvements
4. Performance staying under targets
"""

from hotpath_tier1_extractor import HotPathTier1Extractor
import networkx as nx

def create_graph_from_extraction(result):
    """Convert extraction to NetworkX for analysis"""
    G = nx.DiGraph()

    # Add entity nodes
    for i, entity in enumerate(result.entities):
        G.add_node(f"e_{i}", label=entity, type="entity")

    # Add relation edges
    entity_to_id = {entity: f"e_{i}" for i, entity in enumerate(result.entities)}

    for subj, rel, obj in result.relations:
        subj_id = entity_to_id.get(subj, f"n_{subj}")
        obj_id = entity_to_id.get(obj, f"n_{obj}")

        if subj_id not in G:
            G.add_node(subj_id, label=subj, type="concept")
        if obj_id not in G:
            G.add_node(obj_id, label=obj, type="concept")

        G.add_edge(subj_id, obj_id, relation=rel)

    return G

def prove_quality_improvements():
    print("🧪 PROOF OF QUALITY IMPROVEMENTS")
    print("=" * 60)

    # Initialize extractor
    extractor = HotPathTier1Extractor()
    extractor.warmup()

    # Test cases that prove specific improvements
    test_cases = [
        {
            "name": "Entity Deduplication Test",
            "text": "Apple Inc. was founded by Steve Jobs. Jobs created Apple Inc in Cupertino.",
            "expected_improvement": "Should merge 'Jobs'/'Steve Jobs' and 'Apple Inc.'/'Apple Inc'"
        },
        {
            "name": "Pronoun Resolution Test",
            "text": "Steve Jobs founded Apple. He was a visionary. His company changed everything.",
            "expected_improvement": "Should resolve 'He'→'Steve Jobs', 'His'→'Steve Jobs'"
        },
        {
            "name": "Complex Sentence Test",
            "text": "The CEO announced that the company would restructure after declining profits.",
            "expected_improvement": "Should extract structured relations with good connectivity"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test['name']}")
        print(f"Text: '{test['text']}'")
        print(f"Expected: {test['expected_improvement']}")
        print("-" * 50)

        # Run extraction
        result = extractor.extract(test['text'])

        # Create graph for analysis
        graph = create_graph_from_extraction(result)

        print(f"⏱️  PERFORMANCE:")
        print(f"   Time: {result.extraction_time_ms:.1f}ms")
        print(f"   Under 100ms: {'✅' if result.extraction_time_ms < 100 else '❌'}")

        print(f"\n📊 RAW EXTRACTION:")
        print(f"   Entities ({result.entity_count}):")
        for j, entity in enumerate(result.entities, 1):
            print(f"      {j}. '{entity}'")

        print(f"   Relations ({result.relation_count}):")
        for j, (s, r, o) in enumerate(result.relations, 1):
            print(f"      {j}. '{s}' --[{r}]--> '{o}'")

        print(f"\n🔍 QUALITY ANALYSIS:")
        density = nx.density(graph) if graph.nodes() else 0
        components = nx.number_weakly_connected_components(graph) if graph.nodes() else 0

        print(f"   Graph density: {density:.3f}")
        print(f"   Connected components: {components}")
        print(f"   Nodes: {graph.number_of_nodes()}")
        print(f"   Edges: {graph.number_of_edges()}")

        # Specific quality checks per test
        if "Deduplication" in test['name']:
            print(f"\n✅ DEDUPLICATION PROOF:")
            entities_lower = [e.lower() for e in result.entities]

            # Check for duplicates
            apple_mentions = [e for e in entities_lower if 'apple' in e]
            jobs_mentions = [e for e in entities_lower if 'jobs' in e or 'steve' in e]

            print(f"   Apple mentions: {apple_mentions}")
            print(f"   Jobs mentions: {jobs_mentions}")

            if len(apple_mentions) <= 1:
                print("   ✅ Apple entities successfully deduplicated!")
            else:
                print("   ❌ Apple entities still duplicated")

            if len(jobs_mentions) <= 1:
                print("   ✅ Jobs entities successfully deduplicated!")
            else:
                print("   ❌ Jobs entities still duplicated")

        elif "Pronoun" in test['name']:
            print(f"\n✅ COREFERENCE PROOF:")
            has_pronouns = any(p in [r[0].lower() for r in result.relations] + [r[2].lower() for r in result.relations]
                             for p in ['he', 'his', 'him', 'she', 'her', 'they', 'their'])

            if has_pronouns:
                print("   ❌ Pronouns still present - coreference not fully working")
                print("   Unresolved pronouns found in relations")
            else:
                print("   ✅ No pronouns in relations - coreference working!")

        elif "Complex" in test['name']:
            print(f"\n✅ COMPLEXITY HANDLING PROOF:")
            if density > 0.05:
                print("   ✅ Good graph connectivity achieved!")
            else:
                print("   ❌ Low graph connectivity")

            if components < len(result.entities) * 0.5:
                print("   ✅ Well-connected graph components!")
            else:
                print("   ❌ Too many disconnected components")

        print("\n" + "=" * 60)

    # Overall performance proof
    perf_stats = extractor.get_performance_stats()
    print(f"\n🏆 OVERALL PERFORMANCE PROOF:")
    print(f"   Average time: {perf_stats['average_time_ms']:.1f}ms")
    print(f"   Under 200ms target: {'✅' if perf_stats['average_time_ms'] < 200 else '❌'}")
    print(f"   Components loaded: {sum(perf_stats['components'].values())}/5")
    print(f"   System ready: {'✅' if all(perf_stats['components'].values()) else '❌'}")

if __name__ == "__main__":
    prove_quality_improvements()