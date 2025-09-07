"""
Memory Storage Module

Handles all storage operations and persistence for the HotMem system.
Manages entity storage, edge storage, and alias management.

Author: SOLID Refactoring
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict
import json

from loguru import logger
from memory_store import MemoryStore

class MemoryStorage:
    """
    Handles all storage operations for the HotMem system.
    
    Responsibilities:
    - Entity storage and retrieval
    - Edge (triple) storage and management
    - Alias management and resolution
    - Batch operations and performance optimization
    """
    
    def __init__(self, store: MemoryStore, config: Optional[Dict[str, Any]] = None):
        self.store = store
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 100)
        self.flush_interval = self.config.get('flush_interval', 5.0)  # seconds
        
        # Performance tracking
        self.operation_counts = defaultdict(int)
        self.last_flush_time = time.time()
        
    def store_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> bool:
        """
        Store an entity with its properties.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity (PERSON, ORG, etc.)
            properties: Dictionary of entity properties
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add entity to store
            self.store.enqueue_entity(entity_id, entity_type, properties)
            self.operation_counts['entity_stores'] += 1
            
            # Check if we need to flush
            self._check_flush()
            
            return True
        except Exception as e:
            logger.error(f"Failed to store entity {entity_id}: {e}")
            return False
    
    def store_edge(self, subject: str, relation: str, object_: str, 
                   confidence: float, timestamp: int, metadata: Dict[str, Any] = None) -> bool:
        """
        Store an edge (triple) in the knowledge graph.
        
        Args:
            subject: Subject entity
            relation: Relation type
            object_: Object entity
            confidence: Confidence score (0.0-1.0)
            timestamp: Unix timestamp
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add edge to store
            self.store.observe_edge(subject, relation, object_, confidence, timestamp, metadata or {})
            self.operation_counts['edge_stores'] += 1
            
            # Check if we need to flush
            self._check_flush()
            
            return True
        except Exception as e:
            logger.error(f"Failed to store edge ({subject}, {relation}, {object_}): {e}")
            return False
    
    def store_alias(self, entity_id: str, alias: str) -> bool:
        """
        Store an alias mapping for an entity.
        
        Args:
            entity_id: Original entity ID
            alias: Alias to map to the entity
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add alias to store
            self.store.enqueue_alias(alias, entity_id)
            self.operation_counts['alias_stores'] += 1
            
            # Check if we need to flush
            self._check_flush()
            
            return True
        except Exception as e:
            logger.error(f"Failed to store alias {alias} -> {entity_id}: {e}")
            return False
    
    def store_triples_batch(self, triples: List[Tuple[str, str, str, float]], 
                           timestamp: int) -> int:
        """
        Store multiple triples in a batch operation.
        
        Args:
            triples: List of (subject, relation, object, confidence) tuples
            timestamp: Unix timestamp for all triples
            
        Returns:
            Number of successfully stored triples
        """
        success_count = 0
        
        for subject, relation, object_, confidence in triples:
            if self.store_edge(subject, relation, object_, confidence, timestamp):
                success_count += 1
        
        logger.info(f"Stored {success_count}/{len(triples)} triples in batch")
        return success_count
    
    def resolve_alias(self, alias: str) -> Optional[str]:
        """
        Resolve an alias to its canonical entity ID.
        
        Args:
            alias: Alias to resolve
            
        Returns:
            Canonical entity ID if found, None otherwise
        """
        try:
            return self.store.resolve_alias(alias)
        except Exception as e:
            logger.error(f"Failed to resolve alias {alias}: {e}")
            return None
    
    def get_entity_properties(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get properties for an entity.
        
        Args:
            entity_id: Entity ID to retrieve
            
        Returns:
            Dictionary of entity properties if found, None otherwise
        """
        try:
            return self.store.get_entity(entity_id)
        except Exception as e:
            logger.error(f"Failed to get entity properties for {entity_id}: {e}")
            return None
    
    def get_entity_edges(self, entity_id: str, relation: Optional[str] = None) -> List[Tuple[str, str, str, float, int]]:
        """
        Get edges connected to an entity.
        
        Args:
            entity_id: Entity ID
            relation: Optional relation filter
            
        Returns:
            List of (subject, relation, object, confidence, timestamp) tuples
        """
        try:
            return self.store.get_edges(entity_id, relation)
        except Exception as e:
            logger.error(f"Failed to get edges for entity {entity_id}: {e}")
            return []
    
    def flush_if_needed(self) -> bool:
        """
        Flush pending operations if needed.
        
        Returns:
            True if flush was performed, False otherwise
        """
        try:
            self.store.flush_if_needed()
            self.last_flush_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to flush operations: {e}")
            return False
    
    def force_flush(self) -> bool:
        """
        Force flush all pending operations.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.store.flush_if_needed()
            self.last_flush_time = time.time()
            logger.info("Forced flush of all pending operations")
            return True
        except Exception as e:
            logger.error(f"Failed to force flush: {e}")
            return False
    
    def _check_flush(self) -> None:
        """Check if we need to flush based on batch size or time interval"""
        current_time = time.time()
        
        # Flush if batch size reached
        if (self.operation_counts['entity_stores'] + 
            self.operation_counts['edge_stores'] + 
            self.operation_counts['alias_stores']) >= self.batch_size:
            self.flush_if_needed()
            return
        
        # Flush if time interval reached
        if current_time - self.last_flush_time >= self.flush_interval:
            self.flush_if_needed()
            return
    
    def get_operation_stats(self) -> Dict[str, int]:
        """
        Get statistics about storage operations.
        
        Returns:
            Dictionary of operation counts
        """
        return dict(self.operation_counts)
    
    def reset_operation_stats(self) -> None:
        """Reset operation statistics."""
        self.operation_counts.clear()
        self.last_flush_time = time.time()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics from the underlying store.
        
        Returns:
            Dictionary of storage statistics
        """
        try:
            return self.store.get_stats()
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def cleanup_old_entities(self, max_age_days: int = 30) -> int:
        """
        Clean up old entities to manage storage size.
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            Number of entities cleaned up
        """
        try:
            cutoff_time = int(time.time()) - (max_age_days * 24 * 60 * 60)
            cleaned_count = self.store.cleanup_entities_before(cutoff_time)
            logger.info(f"Cleaned up {cleaned_count} old entities")
            return cleaned_count
        except Exception as e:
            logger.error(f"Failed to cleanup old entities: {e}")
            return 0