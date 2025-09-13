#!/usr/bin/env python3
"""
Test KnowledgeGraphAnalyzer with NetworkX for community detection and graph analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_graph_analyzer():
    """Test graph analyzer with sample knowledge graph data"""
    print("🎯 TESTING KNOWLEDGE GRAPH ANALYZER")
    print("=" * 50)
    
    # Import the analyzer
    from components.graph.graph_analyzer import KnowledgeGraphAnalyzer
    
    # Test configuration
    config = {
        'graph_analysis_enabled': True,
        'use_louvain': True,
        'min_community_size': 2,
        'max_communities': 10,
        'calculate_centrality': True,
        'detect_communities': True
    }
    
    # Initialize analyzer
    analyzer = KnowledgeGraphAnalyzer(config)
    
    # Sample knowledge graph triples from previous tests
    test_triples = [
        ("Dr. Sarah Chen", "is", "AI research director"),
        ("Dr. Sarah Chen", "works_at", "OpenAI"),
        ("Dr. Sarah Chen", "joined", "OpenAI"),
        ("Dr. Sarah Chen", "completed", "PhD at Stanford"),
        ("Dr. Sarah Chen", "supervised_by", "Dr. Michael Jordan"),
        ("OpenAI", "founded_in", "2015"),
        ("OpenAI", "has", "500 employees"),
        ("OpenAI", "known_for", "advanced language models"),
        ("Stanford", "is", "university"),
        ("Dr. Michael Jordan", "is", "professor"),
        ("Dr. Michael Jordan", "works_at", "Stanford"),
        ("Dr. Fei-Fei Li", "works_at", "Stanford"),
        ("Dr. Fei-Fei Li", "collaborated_with", "Dr. Sarah Chen"),
        ("Google Brain", "is", "research division"),
        ("Dr. Sarah Chen", "worked_at", "Google Brain"),
        ("Dr. Sarah Chen", "collaborated_with", "Dr. Fei-Fei Li"),
        ("Dr. Sarah Chen", "developed", "groundbreaking papers"),
        ("Dr. Fei-Fei Li", "developed", "groundbreaking papers")
    ]
    
    print(f"📊 Input triples: {len(test_triples)}")
    for i, triple in enumerate(test_triples, 1):
        print(f"   {i:2d}. {triple}")
    
    # Test graph analysis
    print(f"\n🔄 Testing graph analysis...")
    
    result = analyzer.analyze_knowledge_graph(test_triples)
    
    print(f"✅ Analysis completed in {result.processing_time_ms:.1f}ms")
    
    # Display communities
    print(f"\n🔗 COMMUNITIES DETECTED: {len(result.communities)}")
    for i, community in enumerate(result.communities, 1):
        print(f"   {i}. Community {community.community_id} ({community.size} entities)")
        print(f"      Key entities: {', '.join(community.key_entities)}")
        print(f"      Description: {community.description}")
        if community.modularity > 0:
            print(f"      Modularity: {community.modularity:.3f}")
        print()
    
    # Display centrality metrics
    print(f"📈 CENTRALITY METRICS:")
    for metric_name, scores in result.centrality_metrics.items():
        print(f"   {metric_name.upper()}:")
        top_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for entity, score in top_entities:
            print(f"      {entity}: {score:.3f}")
        print()
    
    # Display graph statistics
    print(f"📊 GRAPH STATISTICS:")
    for key, value in result.graph_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Test top entities method
    print(f"\n🏆 TOP ENTITIES BY PAGERANK:")
    top_entities = analyzer.get_top_entities(result.centrality_metrics, 'pagerank', 5)
    for entity, score in top_entities:
        print(f"   {entity}: {score:.3f}")
    
    # Test analyzer stats
    stats = analyzer.get_stats()
    print(f"\n🔧 ANALYZER STATS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return result

if __name__ == "__main__":
    result = test_graph_analyzer()
    print(f"\n🎯 Knowledge graph analysis test completed successfully!")