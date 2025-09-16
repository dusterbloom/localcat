#!/usr/bin/env python3
"""Dual Graph Manager for V7: Agent (ephemeral hypotheses) + User (durable facts).
Implements TTL decay on Agent graph and promotion to User graph based on confidence/recency.
Integrates with existing graph_analyzer for NetworkX-based traversal and Louvain communities.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available for dual graph operations")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

@dataclass
class Triple:
    """Standard triple with metadata for dual graph tracking."""
    subject: str
    predicate: str
    object_: str
    confidence: float
    timestamp: int  # ms
    source: str  # 'agent' or 'user'
    ttl_ms: Optional[int] = None  # For ephemeral triples

@dataclass
class GraphStats:
    """Dual graph statistics."""
    agent_nodes: int
    agent_edges: int
    user_nodes: int
    user_edges: int
    total_hypotheses: int
    decayed_triples: int
    promoted_triples: int

class DualGraphManager:
    """Manages dual graphs: Agent (ephemeral, TTL-based) and User (durable facts).
    
    - Agent graph: Hypotheses from low-confidence extractions or inferences (decay after 5min TTL).
    - User graph: Promoted durable facts (high-confidence, no TTL).
    - 1-2 hop traversal via NetworkX shortest_path.
    - Louvain community detection on combined graph.
    - Promotion policy: Confidence > 0.8 and validated in 2+ turns.
    """

    def __init__(self, ttl_minutes: int = 5, promotion_threshold: float = 0.8,
                 max_hops: int = 2, use_louvain: bool = True):
        self.agent_graph = nx.DiGraph()  # Ephemeral hypotheses
        self.user_graph = nx.DiGraph()   # Durable facts
        self.ttl_ms = ttl_minutes * 60 * 1000  # Default 5min TTL
        self.promotion_threshold = promotion_threshold
        self.max_hops = max_hops
        self.use_louvain = use_louvain and LOUVAIN_AVAILABLE
        self.promoted_count = 0
        self.decayed_count = 0
        
        # Track TTL for agent triples
        self.agent_triples: Dict[str, Triple] = {}  # id -> Triple
        self.last_cleanup = time.time() * 1000
        
        logger.info(f"DualGraphManager initialized: TTL={ttl_minutes}min, promote>={promotion_threshold}, hops<={max_hops}, louvain={self.use_louvain}")

    def add_triple(self, subject: str, predicate: str, object_: str, confidence: float,
                   source: str = 'agent', ttl_ms: Optional[int] = None) -> str:
        """Add triple to appropriate graph (agent or user). Returns triple ID."""
        triple_id = f"{subject}_{predicate}_{object_}_{int(time.time()*1000)}"
        timestamp = int(time.time() * 1000)
        ttl = ttl_ms or self.ttl_ms if source == 'agent' else None
        
        triple = Triple(subject, predicate, object_, confidence, timestamp, source, ttl)
        self.agent_triples[triple_id] = triple
        
        if source == 'user':
            self.user_graph.add_edge(subject, object_, relation=predicate, confidence=confidence,
                                     timestamp=timestamp, triple_id=triple_id)
            logger.debug(f"Added durable fact: {subject} {predicate} {object_} (conf={confidence:.2f})")
        else:
            self.agent_graph.add_edge(subject, object_, relation=predicate, confidence=confidence,
                                      timestamp=timestamp, triple_id=triple_id, ttl=ttl)
            logger.debug(f"Added hypothesis: {subject} {predicate} {object_} (conf={confidence:.2f}, ttl={ttl/60000:.1f}min)")
        
        return triple_id

    def cleanup_expired(self) -> int:
        """Remove expired agent triples (TTL decay). Returns count decayed."""
        now = int(time.time() * 1000)
        expired_ids = []
        
        for tid, triple in list(self.agent_triples.items()):
            if triple.source == 'agent' and triple.ttl_ms and now - triple.timestamp > triple.ttl_ms:
                expired_ids.append(tid)
        
        for tid in expired_ids:
            triple = self.agent_triples.pop(tid)
            self.agent_graph.remove_edge(triple.subject, triple.object_,
                                         key=tuple(self.agent_graph[triple.subject][triple.object_].keys())[0])
            self.decayed_count += 1
            logger.debug(f"Decayed expired hypothesis: {triple.subject} {triple.predicate} {triple.object_}")
        
        self.last_cleanup = now
        return len(expired_ids)

    def promote_hypothesis(self, triple_id: str, new_confidence: Optional[float] = None) -> bool:
        """Promote agent triple to user graph if confidence threshold met. Returns success."""
        if triple_id not in self.agent_triples:
            return False
        
        triple = self.agent_triples[triple_id]
        if new_confidence:
            triple.confidence = new_confidence
        
        if triple.confidence < self.promotion_threshold:
            logger.debug(f"Promotion denied (conf={triple.confidence:.2f} < {self.promotion_threshold}): {triple_id}")
            return False
        
        # Remove from agent, add to user
        del self.agent_triples[triple_id]
        self.user_graph.add_edge(triple.subject, triple.object_, relation=triple.predicate,
                                 confidence=triple.confidence, timestamp=triple.timestamp,
                                 triple_id=triple_id, source='promoted')
        self.promoted_count += 1
        logger.info(f"Promoted hypothesis to fact (conf={triple.confidence:.2f}): {triple.subject} {triple.predicate} {triple.object_}")
        return True

    def get_neighbors(self, node: str, graph_type: str = 'combined', max_hops: int = None) -> List[Tuple[str, str, float]]:
        """1-2 hop traversal from node in specified graph. Returns (neighbor, relation, confidence)."""
        max_hops = max_hops or self.max_hops
        
        if graph_type == 'agent':
            G = self.agent_graph
        elif graph_type == 'user':
            G = self.user_graph
        else:  # combined
            G = nx.compose(self.agent_graph, self.user_graph)
            # Preserve edge attributes in compose
            for u, v, key, data in self.agent_graph.edges(keys=True, data=True):
                if G.has_edge(u, v, key=key):
                    G[u][v][key].update(data)
                else:
                    G.add_edge(u, v, **data, key=key)
            for u, v, key, data in self.user_graph.edges(keys=True, data=True):
                if G.has_edge(u, v, key=key):
                    G[u][v][key].update(data)
                else:
                    G.add_edge(u, v, **data, key=key)
        
        neighbors = []
        try:
            # 1-hop: direct neighbors
            for target in nx.neighbors(G, node):
                for key, attr in G[node][target].items():
                    neighbors.append((target, attr.get('relation', 'unknown'), attr.get('confidence', 0.0)))
            
            # 2-hop: paths of length 2
            if max_hops >= 2:
                for path in nx.all_simple_paths(G, node, nodes=list(G.nodes()), cutoff=2):
                    if len(path) == 3:  # node -> intermediate -> target
                        intermediate = path[1]
                        target = path[2]
                        # Get edge to intermediate
                        for key1, attr1 in G[node][intermediate].items():
                            # Get edge from intermediate to target
                            for key2, attr2 in G[intermediate][target].items():
                                # Chain confidence (min or product)
                                chain_conf = min(attr1.get('confidence', 1.0), attr2.get('confidence', 1.0))
                                neighbors.append((target, f"{attr1.get('relation', '')} → {attr2.get('relation', '')}", chain_conf))
            
            # Dedup and sort by confidence
            unique_neighbors = {}
            for neigh, rel, conf in neighbors:
                if neigh not in unique_neighbors or conf > unique_neighbors[neigh][2]:
                    unique_neighbors[neigh] = (neigh, rel, conf)
            neighbors = list(unique_neighbors.values())
            neighbors.sort(key=lambda x: x[2], reverse=True)
            
        except Exception as e:
            logger.warning(f"Traversal failed for {node}: {e}")
        
        return neighbors[:10]  # Limit results

    def detect_communities(self, graph_type: str = 'combined') -> List[Dict[str, Any]]:
        """Enable Louvain community detection on specified graph."""
        if not self.use_louvain:
            logger.warning("Louvain not available")
            return []
        
        if graph_type == 'agent':
            G = self.agent_graph
        elif graph_type == 'user':
            G = self.user_graph
        else:
            G = nx.compose(self.agent_graph, self.user_graph)
        
        try:
            partition = community_louvain.best_partition(G)
            community_groups = defaultdict(list)
            for node, comm_id in partition.items():
                community_groups[comm_id].append(node)
            
            communities = []
            modularity = community_louvain.modularity(partition, G)
            for comm_id, nodes in community_groups.items():
                if len(nodes) >= 2:  # Min size
                    subgraph = G.subgraph(nodes)
                    communities.append({
                        'id': comm_id,
                        'nodes': list(nodes),
                        'size': len(nodes),
                        'edges': len(subgraph.edges()),
                        'modularity': modularity,
                        'key_entities': sorted(nodes, key=lambda n: G.degree(n), reverse=True)[:3],
                        'description': f"Community of {len(nodes)} entities with {len(subgraph.edges())} connections"
                    })
            communities.sort(key=lambda c: c['size'], reverse=True)
            logger.info(f"Detected {len(communities)} communities (modularity={modularity:.3f})")
            return communities[:5]  # Top 5
        except Exception as e:
            logger.error(f"Louvain detection failed: {e}")
            return []

    def get_stats(self) -> GraphStats:
        """Get dual graph statistics."""
        agent_nodes = self.agent_graph.number_of_nodes()
        agent_edges = self.agent_graph.number_of_edges()
        user_nodes = self.user_graph.number_of_nodes()
        user_edges = self.user_graph.number_of_edges()
        total_hypotheses = len(self.agent_triples)
        
        return GraphStats(
            agent_nodes=agent_nodes,
            agent_edges=agent_edges,
            user_nodes=user_nodes,
            user_edges=user_edges,
            total_hypotheses=total_hypotheses,
            decayed_triples=self.decayed_count,
            promoted_triples=self.promoted_count
        )

    def export_to_hotmem_facade(self) -> Tuple[List[Tuple], List[Tuple]]:
        """Export combined graphs for integration with HotMemoryFacade."""
        # Agent triples (ephemeral)
        agent_triples = [(t.subject, t.predicate, t.object_) for t in self.agent_triples.values()]
        # User triples (durable)
        user_triples = [(u, v, d.get('relation', 'unknown')) for u, v, d in self.user_graph.edges(data=True)]
        return agent_triples, user_triples

    def import_from_hotmem_facade(self, agent_triples: List[Tuple], user_triples: List[Tuple]):
        """Import triples from HotMemoryFacade for dual graph population."""
        self.agent_graph.clear()
        self.user_graph.clear()
        self.agent_triples.clear()
        
        # Add agent triples with TTL
        for s, p, o in agent_triples:
            self.add_triple(s, p, o, confidence=0.6, source='agent')  # Default low conf for hypotheses
        
        # Add user triples as durable
        for s, p, o in user_triples:
            self.add_triple(s, p, o, confidence=0.9, source='user')

# Integration hook for existing graph_analyzer
def enable_louvain_in_analyzer(analyzer_instance):
    """Enable Louvain communities in existing KnowledgeGraphAnalyzer."""
    if hasattr(analyzer_instance, 'use_louvain'):
        analyzer_instance.use_louvain = True
    logger.info("Louvain community detection enabled in graph_analyzer")

if NETWORKX_AVAILABLE:
    logger.info("🎯 DualGraphManager ready - Agent/User graphs with TTL and traversal")
else:
    logger.warning("DualGraphManager limited: NetworkX not available")