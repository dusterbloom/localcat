"""
HotPath Tier1 Quality Assessment
===============================

Test varied sentences (short→long, simple→complex) and show FULL graph quality
"""

import json
import networkx as nx
from hotpath_tier1_extractor import HotPathTier1Extractor

def create_graph_from_extraction(result):
    """Convert extraction result to NetworkX graph for analysis"""
    G = nx.DiGraph()

    # Add entity nodes
    for i, entity in enumerate(result.entities):
        G.add_node(f"e_{i}", label=entity, type="entity")

    # Add relation edges
    entity_to_id = {entity: f"e_{i}" for i, entity in enumerate(result.entities)}

    for subj, rel, obj in result.relations:
        subj_id = entity_to_id.get(subj)
        obj_id = entity_to_id.get(obj)

        # Create nodes for subjects/objects not in entities
        if not subj_id:
            subj_id = f"n_{len(G.nodes())}"
            G.add_node(subj_id, label=subj, type="concept")

        if not obj_id:
            obj_id = f"n_{len(G.nodes())}"
            G.add_node(obj_id, label=obj, type="concept")

        G.add_edge(subj_id, obj_id, relation=rel)

    return G

def analyze_graph_quality(G):
    """Analyze graph structure and quality metrics"""
    if G.number_of_nodes() == 0:
        return {"quality": "empty", "metrics": {}}

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "connected_components": nx.number_weakly_connected_components(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.nodes() else 0,
        "has_cycles": len(list(nx.simple_cycles(G))) > 0,
        "graph_diameter": nx.diameter(G.to_undirected()) if nx.is_connected(G.to_undirected()) else "disconnected"
    }

    # Quality assessment
    if metrics["density"] > 0.1 and metrics["nodes"] > 3:
        quality = "excellent"
    elif metrics["density"] > 0.05 and metrics["nodes"] > 2:
        quality = "good"
    elif metrics["edges"] > 0:
        quality = "basic"
    else:
        quality = "poor"

    return {"quality": quality, "metrics": metrics}

def test_hotpath_quality():
    """Test HotPath with varied, creative sentences"""

    # Varied test sentences: short→long, simple→complex
    test_cases = [
        # SHORT & SIMPLE
        ("Cat sleeps.", "minimal"),
        ("Dogs bark loudly.", "simple_action"),

        # MEDIUM & MODERATELY COMPLEX
        ("The talented musician played beautiful jazz at midnight.", "descriptive_action"),
        ("While studying, Maria discovered quantum physics principles.", "temporal_complex"),
        ("My grandmother, who lived in Paris, taught me French cooking.", "relative_clause"),

        # LONG & COMPLEX
        ("Despite the heavy rain that started suddenly, the dedicated team of researchers continued their fieldwork in the remote Amazon rainforest, hoping to discover new species before the funding deadline.", "complex_subordinate"),

        ("The CEO announced that the company, which had struggled with declining profits for three consecutive quarters, would implement a comprehensive restructuring plan involving layoffs, office closures, and a strategic pivot toward artificial intelligence technologies.", "business_complex"),

        ("When the ancient manuscript, discovered by archaeologists in a hidden chamber beneath the monastery, was finally decoded by the linguistic expert who had spent decades studying dead languages, it revealed surprising connections between medieval trade routes and modern economic patterns.", "academic_complex"),

        # CHALLENGING CASES
        ("She said he said they would come tomorrow.", "nested_reported_speech"),
        ("Running quickly, the athlete, breathing heavily, crossed the finish line.", "participial_phrases"),
        ("Neither the manager nor the employees understood the new policy that HR implemented without consulting anyone.", "coordination_negation")
    ]

    print("🧪 HotPath Tier1 Quality Assessment")
    print("=" * 60)

    extractor = HotPathTier1Extractor()
    extractor.warmup()  # One-time warmup

    results = []

    for i, (text, complexity) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {complexity.upper()}")
        print(f"Text: '{text}'")
        print(f"Length: {len(text)} chars, {len(text.split())} words")

        # Extract
        result = extractor.extract(text)

        # Create graph
        graph = create_graph_from_extraction(result)
        graph_analysis = analyze_graph_quality(graph)

        # Full output
        print(f"\n✅ EXTRACTION RESULTS:")
        print(f"   ⏱️  Time: {result.extraction_time_ms:.1f}ms")
        print(f"   🎯 Entities ({result.entity_count}):")
        for j, entity in enumerate(result.entities):
            print(f"      {j+1}. '{entity}'")

        print(f"   🔗 Relations ({result.relation_count}):")
        for j, (s, r, o) in enumerate(result.relations):
            print(f"      {j+1}. '{s}' --[{r}]--> '{o}'")

        print(f"\n📊 GRAPH ANALYSIS:")
        print(f"   Quality: {graph_analysis['quality'].upper()}")
        print(f"   Nodes: {graph_analysis['metrics']['nodes']}")
        print(f"   Edges: {graph_analysis['metrics']['edges']}")
        print(f"   Density: {graph_analysis['metrics']['density']:.3f}")
        print(f"   Avg Degree: {graph_analysis['metrics']['avg_degree']:.2f}")
        print(f"   Connected: {graph_analysis['metrics']['connected_components']} component(s)")

        # Store for summary
        results.append({
            "text": text,
            "complexity": complexity,
            "time_ms": result.extraction_time_ms,
            "entities": result.entity_count,
            "relations": result.relation_count,
            "graph_quality": graph_analysis['quality'],
            "graph_density": graph_analysis['metrics']['density']
        })

        print("-" * 60)

    # SUMMARY
    print(f"\n🏆 PERFORMANCE SUMMARY")
    print("=" * 60)

    avg_time = sum(r['time_ms'] for r in results) / len(results)
    quality_distribution = {}
    for r in results:
        quality_distribution[r['graph_quality']] = quality_distribution.get(r['graph_quality'], 0) + 1

    print(f"Total tests: {len(results)}")
    print(f"Average time: {avg_time:.1f}ms")
    print(f"Under 100ms: {sum(1 for r in results if r['time_ms'] < 100)}/{len(results)}")
    print(f"Quality distribution: {quality_distribution}")
    print(f"Average entities per extraction: {sum(r['entities'] for r in results) / len(results):.1f}")
    print(f"Average relations per extraction: {sum(r['relations'] for r in results) / len(results):.1f}")

    # Performance stats
    perf_stats = extractor.get_performance_stats()
    print(f"\nSystem performance: {perf_stats}")

    return results

if __name__ == "__main__":
    test_hotpath_quality()