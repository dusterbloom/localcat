"""
NetworkX Graph Optimizer for Memory Extraction
Prunes junk, infers transitive relations, detects communities
"""

import networkx as nx
from community import community_louvain
from typing import List, Dict, Any, Tuple
from loguru import logger
from collections import defaultdict

class GraphOptimizer:
    def __init__(self):
        self.G = nx.Graph()
        logger.info("[GraphOptimizer] Initialized")

    def build_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> nx.Graph:
        """Build graph from entities and relationships"""
        # Add nodes
        for entity in entities:
            self.G.add_node(entity['text'], 
                            type=entity.get('label', 'ENTITY'),
                            confidence=entity.get('confidence', 1.0))
        
        # Add edges
        for rel in relationships:
            weight = rel.get('confidence', 1.0)
            self.G.add_edge(rel['subject'], rel['object'], 
                            predicate=rel['predicate'],
                            weight=weight)
        
        logger.info(f"[GraphOptimizer] Built graph with {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G

    def prune_junk(self, min_weight: float = 0.6, exclude_vague: bool = True) -> nx.Graph:
        """Prune low-confidence edges and vague nodes"""
        # Prune low-weight edges
        low_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get('weight', 1.0) < min_weight]
        self.G.remove_edges_from(low_edges)
        
        # Prune vague/pronoun nodes
        vague_nodes = set()
        for node in list(self.G.nodes()):
            if exclude_vague and node.lower() in ['he', 'she', 'it', 'they', 'the', 'a', 'an', 'in', 'at', 'on']:
                vague_nodes.add(node)
        
        self.G.remove_nodes_from(vague_nodes)
        logger.info(f"[GraphOptimizer] Pruned {len(low_edges)} edges, {len(vague_nodes)} nodes")
        return self.G

    def detect_communities(self) -> Dict[str, Any]:
        """Detect communities using Louvain"""
        if len(self.G) < 3:
            return {}
        
        partition = community_louvain.best_partition(self.G)
        communities = defaultdict(list)
        for node, comm in partition.items():
            communities[comm].append(node)
        
        modularity = community_louvain.modularity(partition, self.G)
        logger.info(f"[GraphOptimizer] Detected {len(communities)} communities (modularity: {modularity:.3f})")
        return dict(communities)

    def add_transitive_relations(self, max_distance: int = 2) -> List[Dict[str, Any]]:
        """Infer transitive relationships"""
        inferred = []
        for node in self.G.nodes():
            distances = nx.single_source_shortest_path_length(self.G, node, cutoff=max_distance)
            for target, dist in distances.items():
                if dist == 2 and not self.G.has_edge(node, target):
                    # Find path
                    paths = list(nx.all_simple_paths(self.G, node, target, cutoff=2))
                    if paths:
                        intermediate = paths[0][1]
                        inferred.append({
                            'subject': node,
                            'predicate': f'related_via_{intermediate}',
                            'object': target,
                            'inferred': True,
                            'confidence': 0.7
                        })
                        self.G.add_edge(node, target, predicate=f'related_via_{intermediate}', weight=0.7, inferred=True)
        
        logger.info(f"[GraphOptimizer] Added {len(inferred)} transitive relations")
        return inferred

    def remove_duplicates(self) -> nx.Graph:
        """Deduplicate nodes based on similarity"""
        nodes_to_merge = defaultdict(list)
        for node1 in list(self.G.nodes()):
            for node2 in list(self.G.nodes()):
                if node1 != node2 and node1.lower() in node2.lower() or node2.lower() in node1.lower():
                    nodes_to_merge[min(node1, node2)].append(max(node1, node2))
        
        for main, dups in nodes_to_merge.items():
            for dup in dups:
                if dup in self.G:
                    # Transfer edges
                    for neighbor in list(self.G.neighbors(dup)):
                        if neighbor != main and not self.G.has_edge(main, neighbor):
                            pred = self.G[dup][neighbor].get('predicate', 'related_to')
                            self.G.add_edge(main, neighbor, predicate=pred, weight=0.8)
                    self.G.remove_node(dup)
        
        logger.info(f"[GraphOptimizer] Merged {sum(len(d) for d in nodes_to_merge.values())} duplicates")
        return self.G

    def optimize(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Full optimization pipeline"""
        self.build_graph(entities, relationships)
        self.prune_junk()
        communities = self.detect_communities()
        inferred = self.add_transitive_relations()
        self.remove_duplicates()
        
        return {
            'graph': self.G,
            'communities': communities,
            'inferred_relations': inferred,
            'stats': {
                'nodes': self.G.number_of_nodes(),
                'edges': self.G.number_of_edges(),
                'density': nx.density(self.G)
            }
        }

# Global instance
graph_optimizer = GraphOptimizer()

if __name__ == "__main__":
    entities = [{'text': 'maria', 'label': 'PERSON'}, {'text': 'paris', 'label': 'LOC'}]
    rels = [{'subject': 'maria', 'predicate': 'moved_to', 'object': 'paris', 'confidence': 0.8}]
    result = graph_optimizer.optimize(entities, rels)
    print(result)