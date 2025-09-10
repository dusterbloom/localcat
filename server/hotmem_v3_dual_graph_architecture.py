"""
HotMem v3 Dual Graph Architecture
Implements the revolutionary dual graph system from HOT MEM V3:
- Working Memory Graph: Fast, temporary relationships for conversation flow
- Long-term Memory Graph: Persistent, high-confidence knowledge storage

This architecture enables:
1. Real-time conversation flow with working memory
2. Persistent knowledge storage with long-term memory
3. Intelligent promotion/demotion between graphs
4. Context-aware relationship weighting
5. Temporal relationship tracking
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import asyncio
from pathlib import Path
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphType(Enum):
    """Types of graphs in the dual architecture"""
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"

class RelationshipType(Enum):
    """Types of relationships with different weights and lifetimes"""
    CONVERSATIONAL = "conversational"      # Low weight, short lifetime
    CONTEXTUAL = "contextual"            # Medium weight, medium lifetime  
    FACTUAL = "factual"                  # High weight, long lifetime
    INFERRED = "inferred"                # Medium weight, depends on confidence

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    subject: str
    predicate: str
    object: str
    relationship_type: RelationshipType
    confidence: float
    weight: float
    created_at: float
    last_accessed: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'relationship_type': self.relationship_type.value,
            'confidence': self.confidence,
            'weight': self.weight,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create from dictionary"""
        return cls(
            subject=data['subject'],
            predicate=data['predicate'],
            object=data['object'],
            relationship_type=RelationshipType(data['relationship_type']),
            confidence=data['confidence'],
            weight=data['weight'],
            created_at=data['created_at'],
            last_accessed=data['last_accessed'],
            access_count=data['access_count'],
            metadata=data.get('metadata', {})
        )

@dataclass
class Entity:
    """Represents an entity in the graph"""
    name: str
    entity_type: str = "unknown"
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'entity_type': self.entity_type,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            entity_type=data.get('entity_type', 'unknown'),
            confidence=data.get('confidence', 1.0),
            created_at=data.get('created_at', time.time()),
            last_accessed=data.get('last_accessed', time.time()),
            access_count=data.get('access_count', 0),
            metadata=data.get('metadata', {})
        )

class MemoryGraph:
    """Base class for memory graphs"""
    
    def __init__(self, graph_type: GraphType, max_entities: int = 1000, max_relationships: int = 5000):
        self.graph_type = graph_type
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        
        # NetworkX graphs for efficient operations
        self.entity_graph = nx.DiGraph()
        self.relationship_graph = nx.MultiDiGraph()
        
        # Entity and relationship storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
        # Performance tracking
        self.creation_time = time.time()
        self.last_update_time = time.time()
        self.access_count = 0
    
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph"""
        
        if entity.name in self.entities:
            # Update existing entity
            existing = self.entities[entity.name]
            existing.confidence = max(existing.confidence, entity.confidence)
            existing.last_accessed = time.time()
            existing.access_count += 1
            return False
        
        # Check capacity
        if len(self.entities) >= self.max_entities:
            self._evict_entities()
        
        # Add new entity
        self.entities[entity.name] = entity
        self.entity_graph.add_node(entity.name, **entity.to_dict())
        
        self.last_update_time = time.time()
        return True
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to the graph"""
        
        # Ensure entities exist
        if relationship.subject not in self.entities:
            self.add_entity(Entity(name=relationship.subject))
        if relationship.object not in self.entities:
            self.add_entity(Entity(name=relationship.object))
        
        # Check for duplicates
        for existing in self.relationships:
            if (existing.subject == relationship.subject and 
                existing.object == relationship.object and
                existing.predicate == relationship.predicate):
                # Update existing relationship
                existing.confidence = max(existing.confidence, relationship.confidence)
                existing.weight = max(existing.weight, relationship.weight)
                existing.last_accessed = time.time()
                existing.access_count += 1
                return False
        
        # Check capacity
        if len(self.relationships) >= self.max_relationships:
            self._evict_relationships()
        
        # Add new relationship
        self.relationships.append(relationship)
        
        # Add to NetworkX graphs
        self.entity_graph.add_edge(
            relationship.subject, 
            relationship.object,
            predicate=relationship.predicate,
            weight=relationship.weight,
            relationship_type=relationship.relationship_type.value
        )
        
        self.relationship_graph.add_edge(
            relationship.subject,
            relationship.object,
            key=relationship.predicate,
            **relationship.to_dict()
        )
        
        self.last_update_time = time.time()
        return True
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name"""
        entity = self.entities.get(name)
        if entity:
            entity.last_accessed = time.time()
            entity.access_count += 1
            self.access_count += 1
        return entity
    
    def get_relationships(self, subject: Optional[str] = None, object: Optional[str] = None,
                         predicate: Optional[str] = None) -> List[Relationship]:
        """Get relationships matching criteria"""
        
        relationships = []
        
        for rel in self.relationships:
            # Update access stats
            rel.last_accessed = time.time()
            rel.access_count += 1
            
            # Check filters
            if subject and rel.subject != subject:
                continue
            if object and rel.object != object:
                continue
            if predicate and rel.predicate != predicate:
                continue
            
            relationships.append(rel)
        
        self.access_count += len(relationships)
        return relationships
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between entities"""
        
        try:
            paths = list(nx.all_simple_paths(
                self.entity_graph, 
                source=source, 
                target=target, 
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_entity_neighbors(self, entity_name: str, direction: str = "both") -> Dict[str, List[str]]:
        """Get neighboring entities"""
        
        if entity_name not in self.entity_graph:
            return {}
        
        neighbors = {"incoming": [], "outgoing": []}
        
        if direction in ["both", "outgoing"]:
            neighbors["outgoing"] = list(self.entity_graph.successors(entity_name))
        
        if direction in ["both", "incoming"]:
            neighbors["incoming"] = list(self.entity_graph.predecessors(entity_name))
        
        return neighbors
    
    def _evict_entities(self):
        """Evict least recently used entities"""
        
        # Sort by last accessed time
        sorted_entities = sorted(
            self.entities.values(), 
            key=lambda e: e.last_accessed
        )
        
        # Remove oldest 10%
        to_remove = sorted_entities[:max(1, len(self.entities) // 10)]
        
        for entity in to_remove:
            del self.entities[entity.name]
            self.entity_graph.remove_node(entity.name)
        
        # Remove relationships involving evicted entities
        self.relationships = [
            rel for rel in self.relationships
            if rel.subject in self.entities and rel.object in self.entities
        ]
    
    def _evict_relationships(self):
        """Evict least recently used relationships"""
        
        # Sort by last accessed time
        sorted_relationships = sorted(
            self.relationships,
            key=lambda r: r.last_accessed
        )
        
        # Remove oldest 10%
        to_remove = sorted_relationships[:max(1, len(self.relationships) // 10)]
        
        for rel in to_remove:
            self.relationships.remove(rel)
        
        # Rebuild relationship graph
        self.relationship_graph.clear()
        for rel in self.relationships:
            self.relationship_graph.add_edge(
                rel.subject,
                rel.object,
                key=rel.predicate,
                **rel.to_dict()
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        
        return {
            'graph_type': self.graph_type.value,
            'entity_count': len(self.entities),
            'relationship_count': len(self.relationships),
            'creation_time': self.creation_time,
            'last_update_time': self.last_update_time,
            'access_count': self.access_count,
            'graph_density': nx.density(self.entity_graph) if self.entity_graph.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_weakly_connected_components(self.entity_graph)
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export graph data"""
        
        return {
            'graph_type': self.graph_type.value,
            'entities': [e.to_dict() for e in self.entities.values()],
            'relationships': [r.to_dict() for r in self.relationships],
            'stats': self.get_stats()
        }

class WorkingMemoryGraph(MemoryGraph):
    """Working memory graph for fast, temporary relationships"""
    
    def __init__(self, max_entities: int = 100, max_relationships: int = 500):
        super().__init__(GraphType.WORKING_MEMORY, max_entities, max_relationships)
        
        # Working memory specific settings
        self.default_relationship_type = RelationshipType.CONVERSATIONAL
        self.decay_rate = 0.1  # 10% decay per hour
        self.max_lifetime = 3600  # 1 hour maximum lifetime
    
    def add_conversation_extraction(self, text: str, entities: List[str], 
                                 relations: List[Dict[str, Any]], confidence: float):
        """Add extraction from conversation to working memory"""
        
        # Add entities
        for entity_name in entities:
            entity = Entity(
                name=entity_name,
                entity_type="conversation",
                confidence=confidence * 0.8,  # Slightly lower confidence for conversational entities
                metadata={'source': 'conversation', 'text': text}
            )
            self.add_entity(entity)
        
        # Add relationships
        for rel_data in relations:
            relationship = Relationship(
                subject=rel_data.get('subject', ''),
                predicate=rel_data.get('predicate', ''),
                object=rel_data.get('object', ''),
                relationship_type=self.default_relationship_type,
                confidence=confidence * 0.7,  # Lower confidence for conversational relationships
                weight=0.5,  # Lower weight for working memory
                created_at=time.time(),
                last_accessed=time.time(),
                metadata={'source': 'conversation', 'text': text}
            )
            self.add_relationship(relationship)
    
    def decay_relationships(self):
        """Apply decay to old relationships"""
        
        current_time = time.time()
        to_remove = []
        
        for rel in self.relationships:
            age = current_time - rel.created_at
            
            # Remove if too old
            if age > self.max_lifetime:
                to_remove.append(rel)
                continue
            
            # Apply decay
            decay_factor = 1.0 - (self.decay_rate * age / 3600)
            rel.weight *= decay_factor
            rel.confidence *= decay_factor
            
            # Remove if too weak
            if rel.weight < 0.1 or rel.confidence < 0.1:
                to_remove.append(rel)
        
        # Remove decayed relationships
        for rel in to_remove:
            self.relationships.remove(rel)
    
    def get_active_entities(self, min_access: int = 1) -> List[Entity]:
        """Get entities that have been accessed recently"""
        
        return [
            entity for entity in self.entities.values()
            if entity.access_count >= min_access
        ]

class LongTermMemoryGraph(MemoryGraph):
    """Long-term memory graph for persistent knowledge storage"""
    
    def __init__(self, max_entities: int = 10000, max_relationships: int = 50000):
        super().__init__(GraphType.LONG_TERM_MEMORY, max_entities, max_relationships)
        
        # Long-term memory specific settings
        self.promotion_threshold = 0.8  # Confidence threshold for promotion
        self.consolidation_interval = 86400  # 24 hours
        self.last_consolidation = time.time()
    
    def add_factual_extraction(self, text: str, entities: List[str],
                             relations: List[Dict[str, Any]], confidence: float):
        """Add factual extraction to long-term memory"""
        
        # Add entities with higher confidence
        for entity_name in entities:
            entity = Entity(
                name=entity_name,
                entity_type="factual",
                confidence=confidence,
                metadata={'source': 'factual', 'text': text}
            )
            self.add_entity(entity)
        
        # Add relationships as factual
        for rel_data in relations:
            relationship = Relationship(
                subject=rel_data.get('subject', ''),
                predicate=rel_data.get('predicate', ''),
                object=rel_data.get('object', ''),
                relationship_type=RelationshipType.FACTUAL,
                confidence=confidence,
                weight=1.0,  # High weight for factual relationships
                created_at=time.time(),
                last_accessed=time.time(),
                metadata={'source': 'factual', 'text': text}
            )
            self.add_relationship(relationship)
    
    def promote_from_working_memory(self, working_memory: WorkingMemoryGraph):
        """Promote high-confidence relationships from working memory"""
        
        promoted_count = 0
        
        for rel in working_memory.relationships:
            # Check if relationship meets promotion criteria
            if (rel.confidence >= self.promotion_threshold and
                rel.access_count >= 3 and  # Accessed multiple times
                rel.weight >= 0.7):  # High weight
                
                # Create promoted relationship
                promoted_rel = Relationship(
                    subject=rel.subject,
                    predicate=rel.predicate,
                    object=rel.object,
                    relationship_type=RelationshipType.FACTUAL,
                    confidence=min(rel.confidence * 1.1, 1.0),  # Slight confidence boost
                    weight=1.0,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    metadata={
                        'promoted_from': 'working_memory',
                        'original_confidence': rel.confidence,
                        'original_weight': rel.weight
                    }
                )
                
                if self.add_relationship(promoted_rel):
                    promoted_count += 1
        
        logger.info(f"Promoted {promoted_count} relationships from working memory to long-term memory")
        return promoted_count
    
    def consolidate_relationships(self):
        """Consolidate similar relationships"""
        
        current_time = time.time()
        
        # Check if consolidation is needed
        if current_time - self.last_consolidation < self.consolidation_interval:
            return
        
        # Group similar relationships
        relationship_groups = defaultdict(list)
        
        for rel in self.relationships:
            key = (rel.subject, rel.predicate, rel.object)
            relationship_groups[key].append(rel)
        
        # Consolidate groups
        consolidated_count = 0
        
        for key, group in relationship_groups.items():
            if len(group) > 1:
                # Keep the relationship with highest confidence
                best_rel = max(group, key=lambda r: r.confidence)
                
                # Remove others
                for rel in group:
                    if rel != best_rel:
                        self.relationships.remove(rel)
                        consolidated_count += 1
        
        self.last_consolidation = current_time
        logger.info(f"Consolidated {consolidated_count} relationships")

class DualGraphArchitecture:
    """Main dual graph architecture system"""
    
    def __init__(self, working_memory_size: int = 100, long_term_memory_size: int = 10000):
        self.working_memory = WorkingMemoryGraph(max_entities=working_memory_size)
        self.long_term_memory = LongTermMemoryGraph(max_entities=long_term_memory_size)
        
        # Integration state
        self.last_promotion_time = time.time()
        self.promotion_interval = 1800  # 30 minutes
        self.conversation_sessions = {}
        
        # Performance tracking
        self.total_extractions = 0
        self.total_promotions = 0
    
    def add_extraction(self, text: str, entities: List[str], 
                      relations: List[Dict[str, Any]], confidence: float,
                      session_id: Optional[str] = None, 
                      extraction_type: str = "conversation"):
        """Add extraction to appropriate memory graph"""
        
        self.total_extractions += 1
        
        if extraction_type == "conversation":
            # Add to working memory
            self.working_memory.add_conversation_extraction(text, entities, relations, confidence)
        elif extraction_type == "factual":
            # Add to long-term memory
            self.long_term_memory.add_factual_extraction(text, entities, relations, confidence)
        
        # Update session tracking
        if session_id:
            if session_id not in self.conversation_sessions:
                self.conversation_sessions[session_id] = {
                    'start_time': time.time(),
                    'extractions': 0,
                    'entities': set(),
                    'relations': set()
                }
            
            session = self.conversation_sessions[session_id]
            session['extractions'] += 1
            session['entities'].update(entities)
            session['relations'].update(
                f"{r.get('subject', '')}_{r.get('predicate', '')}_{r.get('object', '')}"
                for r in relations
            )
        
        # Periodic maintenance
        self._perform_maintenance()
    
    def query_knowledge(self, query: str, query_type: str = "entities",
                        session_context: Optional[str] = None) -> Dict[str, Any]:
        """Query knowledge across both memory graphs"""
        
        results = {
            'working_memory': [],
            'long_term_memory': [],
            'combined': [],
            'query_stats': {
                'working_memory_entities': len(self.working_memory.entities),
                'long_term_memory_entities': len(self.long_term_memory.entities),
                'total_entities': len(self.working_memory.entities) + len(self.long_term_memory.entities)
            }
        }
        
        if query_type == "entities":
            # Search for entities
            working_entities = [e for e in self.working_memory.entities.values() if query.lower() in e.name.lower()]
            long_term_entities = [e for e in self.long_term_memory.entities.values() if query.lower() in e.name.lower()]
            
            results['working_memory'] = [e.to_dict() for e in working_entities]
            results['long_term_memory'] = [e.to_dict() for e in long_term_entities]
            results['combined'] = results['working_memory'] + results['long_term_memory']
        
        elif query_type == "relations":
            # Search for relations
            working_relations = self.working_memory.get_relationships()
            long_term_relations = self.long_term_memory.get_relationships()
            
            # Filter by query
            working_filtered = [r.to_dict() for r in working_relations 
                             if query.lower() in r.subject.lower() or 
                                query.lower() in r.object.lower() or
                                query.lower() in r.predicate.lower()]
            
            long_term_filtered = [r.to_dict() for r in long_term_relations
                                if query.lower() in r.subject.lower() or
                                   query.lower() in r.object.lower() or
                                   query.lower() in r.predicate.lower()]
            
            results['working_memory'] = working_filtered
            results['long_term_memory'] = long_term_filtered
            results['combined'] = working_filtered + long_term_filtered
        
        elif query_type == "paths":
            # Find paths between entities
            if len(query.split()) == 2:
                source, target = query.split()
                
                working_paths = self.working_memory.find_paths(source, target)
                long_term_paths = self.long_term_memory.find_paths(source, target)
                
                results['working_memory'] = working_paths
                results['long_term_memory'] = long_term_paths
                results['combined'] = working_paths + long_term_paths
        
        return results
    
    def get_enhanced_context(self, base_context: str, session_id: Optional[str] = None) -> str:
        """Enhance context with relevant knowledge from both graphs"""
        
        enhanced_context = base_context + "\n\nRelevant Knowledge:\n"
        
        # Get active entities from working memory
        active_entities = self.working_memory.get_active_entities(min_access=2)
        
        if active_entities:
            enhanced_context += "Active Conversation Entities:\n"
            for entity in active_entities[:5]:  # Limit to top 5
                enhanced_context += f"- {entity.name} (accessed {entity.access_count} times)\n"
        
        # Get high-confidence relationships from long-term memory
        high_confidence_relations = [
            r for r in self.long_term_memory.relationships
            if r.confidence >= 0.9
        ]
        
        if high_confidence_relations:
            enhanced_context += "\nHigh-Confidence Knowledge:\n"
            for rel in high_confidence_relations[:3]:  # Limit to top 3
                enhanced_context += f"- {rel.subject} {rel.predicate} {rel.object}\n"
        
        return enhanced_context
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        
        current_time = time.time()
        
        # Decay working memory relationships
        self.working_memory.decay_relationships()
        
        # Promote from working to long-term memory
        if current_time - self.last_promotion_time >= self.promotion_interval:
            promoted = self.long_term_memory.promote_from_working_memory(self.working_memory)
            self.total_promotions += promoted
            self.last_promotion_time = current_time
        
        # Consolidate long-term memory
        self.long_term_memory.consolidate_relationships()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            'total_extractions': self.total_extractions,
            'total_promotions': self.total_promotions,
            'active_sessions': len(self.conversation_sessions),
            'working_memory': self.working_memory.get_stats(),
            'long_term_memory': self.long_term_memory.get_stats(),
            'last_promotion_time': self.last_promotion_time,
            'system_uptime': time.time() - self.working_memory.creation_time
        }
    
    def export_system(self) -> Dict[str, Any]:
        """Export entire dual graph system"""
        
        return {
            'working_memory': self.working_memory.export_graph(),
            'long_term_memory': self.long_term_memory.export_graph(),
            'system_stats': self.get_system_stats(),
            'conversation_sessions': self.conversation_sessions,
            'export_timestamp': time.time()
        }
    
    def save_to_disk(self, filepath: str):
        """Save system state to disk"""
        
        export_data = self.export_system()
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Dual graph system saved to {filepath}")
    
    def load_from_disk(self, filepath: str):
        """Load system state from disk"""
        
        with open(filepath, 'r') as f:
            export_data = json.load(f)
        
        # Reconstruct graphs (simplified - in practice you'd need full reconstruction)
        logger.info(f"Dual graph system loaded from {filepath}")

def main():
    """Test the dual graph architecture"""
    
    print("🧠 HotMem v3 Dual Graph Architecture Test")
    print("=" * 50)
    
    # Initialize dual graph system
    dual_graph = DualGraphArchitecture()
    
    # Simulate conversation extractions
    conversation_extractions = [
        {
            'text': "Hi, I'm Sarah and I work at Google",
            'entities': ['Sarah', 'Google'],
            'relations': [{'subject': 'Sarah', 'predicate': 'works_at', 'object': 'Google'}],
            'confidence': 0.8,
            'type': 'conversation'
        },
        {
            'text': "I'm a software engineer in the AI department",
            'entities': ['Sarah', 'software engineer', 'AI department'],
            'relations': [
                {'subject': 'Sarah', 'predicate': 'is', 'object': 'software engineer'},
                {'subject': 'Sarah', 'predicate': 'works_in', 'object': 'AI department'}
            ],
            'confidence': 0.7,
            'type': 'conversation'
        },
        {
            'text': "My manager is John who leads the machine learning team",
            'entities': ['John', 'machine learning team'],
            'relations': [
                {'subject': 'John', 'predicate': 'leads', 'object': 'machine learning team'},
                {'subject': 'John', 'predicate': 'is_manager_of', 'object': 'Sarah'}
            ],
            'confidence': 0.9,
            'type': 'conversation'
        }
    ]
    
    print("Processing conversation extractions...")
    
    for i, extraction in enumerate(conversation_extractions):
        print(f"\nExtraction {i+1}: '{extraction['text']}'")
        
        dual_graph.add_extraction(
            text=extraction['text'],
            entities=extraction['entities'],
            relations=extraction['relations'],
            confidence=extraction['confidence'],
            session_id="session_1",
            extraction_type=extraction['type']
        )
        
        # Show current state
        stats = dual_graph.get_system_stats()
        print(f"  Working Memory: {stats['working_memory']['entity_count']} entities, "
              f"{stats['working_memory']['relationship_count']} relationships")
        print(f"  Long-term Memory: {stats['long_term_memory']['entity_count']} entities, "
              f"{stats['long_term_memory']['relationship_count']} relationships")
    
    # Add some factual knowledge
    print(f"\n{'='*50}")
    print("Adding factual knowledge...")
    
    factual_extractions = [
        {
            'text': "Google is a technology company founded by Larry Page and Sergey Brin",
            'entities': ['Google', 'Larry Page', 'Sergey Brin'],
            'relations': [
                {'subject': 'Google', 'predicate': 'is', 'object': 'technology company'},
                {'subject': 'Larry Page', 'predicate': 'founded', 'object': 'Google'},
                {'subject': 'Sergey Brin', 'predicate': 'founded', 'object': 'Google'}
            ],
            'confidence': 0.95,
            'type': 'factual'
        },
        {
            'text': "Machine learning is a subset of artificial intelligence",
            'entities': ['Machine learning', 'artificial intelligence'],
            'relations': [
                {'subject': 'Machine learning', 'predicate': 'is_subset_of', 'object': 'artificial intelligence'}
            ],
            'confidence': 0.99,
            'type': 'factual'
        }
    ]
    
    for extraction in factual_extractions:
        dual_graph.add_extraction(
            text=extraction['text'],
            entities=extraction['entities'],
            relations=extraction['relations'],
            confidence=extraction['confidence'],
            extraction_type=extraction['type']
        )
    
    # Test knowledge queries
    print(f"\n{'='*50}")
    print("Testing Knowledge Queries:")
    
    # Query for Google
    google_results = dual_graph.query_knowledge("Google", "entities")
    print(f"\nQuery for 'Google' entities:")
    print(f"  Working memory: {len(google_results['working_memory'])} results")
    print(f"  Long-term memory: {len(google_results['long_term_memory'])} results")
    
    # Query for Sarah
    sarah_relations = dual_graph.query_knowledge("Sarah", "relations")
    print(f"\nQuery for 'Sarah' relations:")
    print(f"  Working memory: {len(sarah_relations['working_memory'])} relations")
    print(f"  Long-term memory: {len(sarah_relations['long_term_memory'])} relations")
    
    # Test context enhancement
    print(f"\n{'='*50}")
    print("Testing Context Enhancement:")
    base_context = "User is talking about their job."
    enhanced = dual_graph.get_enhanced_context(base_context, "session_1")
    print(f"Base: {base_context}")
    print(f"Enhanced: {enhanced}")
    
    # Show final stats
    print(f"\n{'='*50}")
    print("Final System Statistics:")
    final_stats = dual_graph.get_system_stats()
    for key, value in final_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()