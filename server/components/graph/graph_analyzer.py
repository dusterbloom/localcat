"""
KnowledgeGraphAnalyzer: Advanced Graph Analysis with NetworkX
==========================================================

SOTA knowledge graph analysis using NetworkX for community detection,
clustering, and graph analytics.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from loguru import logger

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except Exception as e:
    NETWORKX_AVAILABLE = False
    logger.warning(f"[KnowledgeGraphAnalyzer] networkx not available: {e}")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except Exception as e:
    LOUVAIN_AVAILABLE = False
    logger.warning(f"[KnowledgeGraphAnalyzer] python-louvain not available: {e}")


@dataclass
class CommunityInfo:
    """Information about a detected community"""
    community_id: int
    nodes: List[str]
    size: int
    modularity: float
    key_entities: List[str]
    description: str


@dataclass
class GraphAnalysisResult:
    """Result of graph analysis"""
    communities: List[CommunityInfo]
    centrality_metrics: Dict[str, Dict[str, float]]
    graph_stats: Dict[str, Any]
    processing_time_ms: float


class KnowledgeGraphAnalyzer:
    """
    Advanced knowledge graph analysis using NetworkX.
    Handles community detection, centrality analysis, and graph statistics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize graph analyzer with configuration"""
        self.enabled = config.get('graph_analysis_enabled', False) and NETWORKX_AVAILABLE
        self.use_louvain = config.get('use_louvain', True) and LOUVAIN_AVAILABLE
        self.min_community_size = config.get('min_community_size', 2)
        self.max_communities = config.get('max_communities', 20)
        
        # Analysis options
        self.calculate_centrality = config.get('calculate_centrality', True)
        self.detect_communities = config.get('detect_communities', True)
        
        logger.info(f"[KnowledgeGraphAnalyzer] Initialized with enabled={'✓' if self.enabled else '✗'}, "
                   f"louvain={'✓' if self.use_louvain else '✗'}")
    
    def analyze_knowledge_graph(self, triples: List[Tuple[str, str, str]]) -> GraphAnalysisResult:
        """
        Main entry point for knowledge graph analysis
        """
        start = time.perf_counter()
        
        try:
            if not triples or not self.enabled:
                return GraphAnalysisResult([], {}, {'method': 'none'}, 0.0)
            
            # Build NetworkX graph
            G = self._build_graph(triples)
            
            if len(G.nodes()) == 0:
                return GraphAnalysisResult([], {}, {'method': 'empty_graph'}, 0.0)
            
            # Analyze communities
            communities = []
            if self.detect_communities:
                communities = self._detect_communities(G)
            
            # Calculate centrality metrics
            centrality_metrics = {}
            if self.calculate_centrality:
                centrality_metrics = self._calculate_centrality(G)
            
            # Calculate graph statistics
            graph_stats = self._calculate_graph_stats(G, communities)
            
            processing_time = (time.perf_counter() - start) * 1000
            
            return GraphAnalysisResult(
                communities=communities,
                centrality_metrics=centrality_metrics,
                graph_stats=graph_stats,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"[KnowledgeGraphAnalyzer] Graph analysis failed: {e}")
            processing_time = (time.perf_counter() - start) * 1000
            return GraphAnalysisResult([], {}, {'error': str(e)}, processing_time)
    
    def _build_graph(self, triples: List[Tuple[str, str, str]]) -> nx.Graph:
        """Build NetworkX graph from triples"""
        
        G = nx.Graph()
        
        # Add nodes and edges
        for subject, predicate, obj in triples:
            # Add nodes
            G.add_node(subject)
            G.add_node(obj)
            
            # Add edge with predicate as attribute
            G.add_edge(subject, obj, relation=predicate)
        
        return G
    
    def _detect_communities(self, G: nx.Graph) -> List[CommunityInfo]:
        """Detect communities in the graph"""
        
        communities = []
        
        try:
            if self.use_louvain:
                # Use Louvain method (faster and often better)
                partition = community_louvain.best_partition(G)
                
                # Group nodes by community
                community_groups = defaultdict(list)
                for node, community_id in partition.items():
                    community_groups[community_id].append(node)
                
                # Calculate modularity
                modularity = community_louvain.modularity(partition, G)
                
                # Create CommunityInfo objects
                for community_id, nodes in community_groups.items():
                    if len(nodes) >= self.min_community_size:
                        key_entities = self._find_key_entities(G, nodes)
                        description = self._describe_community(G, nodes)
                        
                        community_info = CommunityInfo(
                            community_id=community_id,
                            nodes=nodes,
                            size=len(nodes),
                            modularity=modularity,
                            key_entities=key_entities,
                            description=description
                        )
                        communities.append(community_info)
                
            else:
                # Use built-in NetworkX community detection
                # Label propagation (fast but less accurate)
                communities_result = community.label_propagation_communities(G)
                
                for i, community_nodes in enumerate(communities_result):
                    if len(community_nodes) >= self.min_community_size:
                        key_entities = self._find_key_entities(G, list(community_nodes))
                        description = self._describe_community(G, list(community_nodes))
                        
                        community_info = CommunityInfo(
                            community_id=i,
                            nodes=list(community_nodes),
                            size=len(community_nodes),
                            modularity=0.0,  # Not available for label propagation
                            key_entities=key_entities,
                            description=description
                        )
                        communities.append(community_info)
            
            # Sort communities by size
            communities.sort(key=lambda x: x.size, reverse=True)
            
            # Limit number of communities
            if len(communities) > self.max_communities:
                communities = communities[:self.max_communities]
            
        except Exception as e:
            logger.debug(f"[KnowledgeGraphAnalyzer] Community detection failed: {e}")
            # Fallback: create one community with all nodes
            if len(G.nodes()) > 0:
                all_nodes = list(G.nodes())
                key_entities = self._find_key_entities(G, all_nodes)
                description = self._describe_community(G, all_nodes)
                
                community_info = CommunityInfo(
                    community_id=0,
                    nodes=all_nodes,
                    size=len(all_nodes),
                    modularity=0.0,
                    key_entities=key_entities,
                    description=description
                )
                communities.append(community_info)
        
        return communities
    
    def _find_key_entities(self, G: nx.Graph, nodes: List[str]) -> List[str]:
        """Find key entities in a community using centrality"""
        
        if len(nodes) <= 3:
            return nodes
        
        try:
            # Calculate degree centrality for community nodes
            subgraph = G.subgraph(nodes)
            centrality = nx.degree_centrality(subgraph)
            
            # Get top 3 most central nodes
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            key_entities = [node for node, score in sorted_nodes[:3]]
            
            return key_entities
            
        except Exception:
            # Fallback: return first few nodes
            return nodes[:3]
    
    def _describe_community(self, G: nx.Graph, nodes: List[str]) -> str:
        """Generate a description for a community"""
        
        if len(nodes) == 0:
            return "Empty community"
        
        try:
            # Get unique predicates in this community
            subgraph = G.subgraph(nodes)
            predicates = set()
            for u, v, data in subgraph.edges(data=True):
                if 'relation' in data:
                    predicates.add(data['relation'])
            
            # Count entity types (simplified)
            entity_count = len(nodes)
            relation_count = len(predicates)
            
            if relation_count == 0:
                return f"Isolated group of {entity_count} entities"
            
            # Generate description
            top_predicates = list(predicates)[:3]
            if len(top_predicates) == 1:
                desc = f"Community connected by '{top_predicates[0]}'"
            elif len(top_predicates) == 2:
                desc = f"Community connected by '{top_predicates[0]}' and '{top_predicates[1]}'"
            else:
                desc = f"Diverse community with {relation_count} relation types"
            
            return f"{desc} ({entity_count} entities)"
            
        except Exception:
            return f"Community with {len(nodes)} entities"
    
    def _calculate_centrality(self, G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Calculate centrality metrics for all nodes"""
        
        centrality_metrics = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            centrality_metrics['degree'] = degree_centrality
            
            # Betweenness centrality (for larger graphs)
            if len(G.nodes()) <= 100:  # Avoid expensive calculation for large graphs
                betweenness_centrality = nx.betweenness_centrality(G)
                centrality_metrics['betweenness'] = betweenness_centrality
            
            # Closeness centrality
            if nx.is_connected(G):
                closeness_centrality = nx.closeness_centrality(G)
                centrality_metrics['closeness'] = closeness_centrality
            
            # PageRank (for directed graphs, but can work on undirected)
            pagerank = nx.pagerank(G)
            centrality_metrics['pagerank'] = pagerank
            
        except Exception as e:
            logger.debug(f"[KnowledgeGraphAnalyzer] Centrality calculation failed: {e}")
        
        return centrality_metrics
    
    def _calculate_graph_stats(self, G: nx.Graph, communities: List[CommunityInfo]) -> Dict[str, Any]:
        """Calculate comprehensive graph statistics"""
        
        stats = {}
        
        try:
            # Basic stats
            stats['nodes'] = len(G.nodes())
            stats['edges'] = len(G.edges())
            stats['density'] = nx.density(G)
            stats['is_connected'] = nx.is_connected(G)
            
            # Connected components
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                stats['connected_components'] = len(components)
                stats['largest_component_size'] = len(max(components, key=len))
            else:
                stats['connected_components'] = 1
                stats['largest_component_size'] = len(G.nodes())
            
            # Average clustering coefficient
            stats['average_clustering'] = nx.average_clustering(G)
            
            # Community stats
            if communities:
                stats['communities_detected'] = len(communities)
                stats['average_community_size'] = sum(c.size for c in communities) / len(communities)
                stats['largest_community_size'] = max(c.size for c in communities)
                
                if self.use_louvain and communities:
                    stats['modularity'] = communities[0].modularity  # All communities have same modularity
            
            # Path stats (for connected graphs)
            if nx.is_connected(G) and len(G.nodes()) > 1:
                stats['average_path_length'] = nx.average_shortest_path_length(G)
                stats['diameter'] = nx.diameter(G)
            
            # Degree distribution
            degrees = [d for n, d in G.degree()]
            if degrees:
                stats['average_degree'] = sum(degrees) / len(degrees)
                stats['max_degree'] = max(degrees)
                stats['min_degree'] = min(degrees)
            
        except Exception as e:
            logger.debug(f"[KnowledgeGraphAnalyzer] Graph stats calculation failed: {e}")
        
        return stats
    
    def get_top_entities(self, centrality_metrics: Dict[str, Dict[str, float]], 
                        metric: str = 'pagerank', top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top entities by centrality metric"""
        
        if metric not in centrality_metrics:
            return []
        
        scores = centrality_metrics[metric]
        sorted_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities[:top_n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            'enabled': self.enabled,
            'networkx_available': NETWORKX_AVAILABLE,
            'louvain_available': LOUVAIN_AVAILABLE,
            'use_louvain': self.use_louvain,
            'min_community_size': self.min_community_size,
            'max_communities': self.max_communities
        }


logger.info("🎯 KnowledgeGraphAnalyzer initialized - advanced graph analysis with NetworkX")
logger.info("📊 Features: Community detection, centrality analysis, graph statistics")