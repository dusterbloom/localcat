"""
Memory Interfaces Module

Abstract interfaces for memory system components following SOLID principles.
Enables dependency injection, testability, and multiple implementations.

Author: SOLID Refactoring
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class Triple:
    """Represents a knowledge graph triple."""
    subject: str
    relation: str
    object_: str
    confidence: float
    timestamp: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any]
    created_at: int
    updated_at: int
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class ExtractionResult:
    """Result of an extraction operation."""
    entities: List[str]
    triples: List[Tuple[str, str, str]]
    extraction_time_ms: int
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    results: List[Tuple[float, str, str, str]]  # (score, subject, relation, object)
    query_time_ms: int
    total_found: int
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class QualityAssessment:
    """Quality assessment result for a triple."""
    triple: Tuple[str, str, str]
    confidence: float
    quality_score: float
    issues: List[str]
    should_filter: bool
    explanation: Dict[str, Any]

class IMemoryExtractor(ABC):
    """Interface for memory extraction components."""
    
    @abstractmethod
    def extract_triples(self, text: str, lang: str = "en") -> ExtractionResult:
        """
        Extract entities and triples from text.
        
        Args:
            text: Input text to process
            lang: Language code
            
        Returns:
            ExtractionResult containing entities and triples
        """
        pass
    
    @abstractmethod
    def get_extraction_strategies(self) -> List[str]:
        """Get list of available extraction strategies."""
        pass
    
    @abstractmethod
    def is_strategy_available(self, strategy: str) -> bool:
        """Check if a specific extraction strategy is available."""
        pass
    
    @abstractmethod
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        pass

class IMemoryStorage(ABC):
    """Interface for memory storage components."""
    
    @abstractmethod
    def store_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> bool:
        """
        Store an entity with its properties.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            properties: Dictionary of entity properties
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def store_alias(self, entity_id: str, alias: str) -> bool:
        """
        Store an alias mapping for an entity.
        
        Args:
            entity_id: Original entity ID
            alias: Alias to map to the entity
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def resolve_alias(self, alias: str) -> Optional[str]:
        """
        Resolve an alias to its canonical entity ID.
        
        Args:
            alias: Alias to resolve
            
        Returns:
            Canonical entity ID if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_entity_properties(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get properties for an entity.
        
        Args:
            entity_id: Entity ID to retrieve
            
        Returns:
            Dictionary of entity properties if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_entity_edges(self, entity_id: str, relation: Optional[str] = None) -> List[Tuple[str, str, str, float, int]]:
        """
        Get edges connected to an entity.
        
        Args:
            entity_id: Entity ID
            relation: Optional relation filter
            
        Returns:
            List of (subject, relation, object, confidence, timestamp) tuples
        """
        pass
    
    @abstractmethod
    def flush_if_needed(self) -> bool:
        """
        Flush pending operations if needed.
        
        Returns:
            True if flush was performed, False otherwise
        """
        pass
    
    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

class IMemoryRetriever(ABC):
    """Interface for memory retrieval components."""
    
    @abstractmethod
    def retrieve(self, query: str, session_id: str = None, 
                strategies: List[str] = None) -> RetrievalResult:
        """
        Retrieve relevant facts based on the query.
        
        Args:
            query: Query string
            session_id: Optional session ID for context
            strategies: List of retrieval strategies to use
            
        Returns:
            RetrievalResult containing ranked results
        """
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available retrieval strategies."""
        pass
    
    @abstractmethod
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the query cache."""
        pass
    
    @abstractmethod
    def explain_retrieval(self, query: str, results: List[Tuple[float, str, str, str]]) -> Dict[str, Any]:
        """
        Provide explanation for retrieval results.
        
        Args:
            query: Original query
            results: Retrieved results
            
        Returns:
            Dictionary with explanation details
        """
        pass

class IMemoryQuality(ABC):
    """Interface for memory quality assessment components."""
    
    @abstractmethod
    def assess_triple_quality(self, subject: str, relation: str, object_: str,
                            context: Optional[Dict[str, Any]] = None) -> QualityAssessment:
        """
        Assess the quality of a triple.
        
        Args:
            subject: Subject entity
            relation: Relation type
            object_: Object entity
            context: Optional context information
            
        Returns:
            QualityAssessment with detailed assessment
        """
        pass
    
    @abstractmethod
    def filter_triples(self, triples: List[Tuple[str, str, str]], 
                      context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str, str, float]]:
        """
        Filter a list of triples based on quality.
        
        Args:
            triples: List of (subject, relation, object) tuples
            context: Optional context information
            
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        pass
    
    @abstractmethod
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality assessment statistics."""
        pass
    
    @abstractmethod
    def explain_quality_decision(self, subject: str, relation: str, object_: str) -> Dict[str, Any]:
        """
        Provide detailed explanation for quality decision.
        
        Args:
            subject: Subject entity
            relation: Relation type
            object_: Object entity
            
        Returns:
            Dictionary with detailed explanation
        """
        pass

class IMemoryMetrics(ABC):
    """Interface for memory metrics collection components."""
    
    @abstractmethod
    def record_extraction_time(self, duration_ms: int, success: bool = True) -> None:
        """Record extraction operation time."""
        pass
    
    @abstractmethod
    def record_storage_time(self, duration_ms: int, success: bool = True) -> None:
        """Record storage operation time."""
        pass
    
    @abstractmethod
    def record_retrieval_time(self, duration_ms: int, success: bool = True) -> None:
        """Record retrieval operation time."""
        pass
    
    @abstractmethod
    def record_quality_score(self, score: float) -> None:
        """Record quality assessment score."""
        pass
    
    @abstractmethod
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        pass
    
    @abstractmethod
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        pass
    
    @abstractmethod
    def get_recent_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics from the last N minutes."""
        pass
    
    @abstractmethod
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in various formats."""
        pass

class IMemoryService(ABC):
    """Main interface for the complete memory service."""
    
    @abstractmethod
    def process_turn(self, text: str, session_id: str, turn_id: int) -> Dict[str, Any]:
        """
        Process a conversational turn and store extracted information.
        
        Args:
            text: Input text
            session_id: Session identifier
            turn_id: Turn identifier
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    @abstractmethod
    def retrieve_facts(self, query: str, session_id: str = None, 
                      limit: int = 10) -> List[Tuple[float, str, str, str]]:
        """
        Retrieve relevant facts based on query.
        
        Args:
            query: Query string
            session_id: Optional session context
            limit: Maximum number of results
            
        Returns:
            List of (score, subject, relation, object) tuples
        """
        pass
    
    @abstractmethod
    def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health status."""
        pass
    
    @abstractmethod
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        pass
    
    @abstractmethod
    def reset_service(self) -> None:
        """Reset service state and clear data."""
        pass

class IExtractionStrategy(ABC):
    """Interface for extraction strategies."""
    
    @abstractmethod
    def extract(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """
        Extract triples from text using this strategy.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            List of (subject, relation, object) triples
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this extraction strategy."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy is available and ready to use."""
        pass
    
    @abstractmethod
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get configuration for this strategy."""
        pass

class IRetrievalStrategy(ABC):
    """Interface for retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, triples: List[Tuple[str, str, str, float]]) -> List[Tuple[float, str, str, str]]:
        """
        Retrieve relevant triples using this strategy.
        
        Args:
            query: Query string
            triples: Available triples with confidence scores
            
        Returns:
            List of (score, subject, relation, object) tuples
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this retrieval strategy."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy is available and ready to use."""
        pass

class IConfigurationProvider(ABC):
    """Interface for configuration providers."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        pass
    
    @abstractmethod
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        pass