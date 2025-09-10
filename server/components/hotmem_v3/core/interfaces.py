"""
HotMem v3 Interfaces

Defines the core interfaces and contracts for HotMem v3 components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import asyncio

@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    label: str
    confidence: float
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Relation:
    """Represents a extracted relation"""
    subject: str
    predicate: str
    object: str
    confidence: float
    subject_entity: Optional[Entity] = None
    object_entity: Optional[Entity] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExtractionResult:
    """Result of text extraction"""
    entities: List[Entity]
    relations: List[Relation]
    confidence: float
    processing_time: float
    text: str
    metadata: Optional[Dict[str, Any]] = None

class HotMemV3Interface(ABC):
    """Main interface for HotMem v3 systems"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the HotMem v3 system"""
        pass
    
    @abstractmethod
    async def process_text(self, text: str, is_final: bool = True, 
                          session_id: Optional[str] = None) -> ExtractionResult:
        """Process text and return extraction results"""
        pass
    
    @abstractmethod
    async def add_user_correction(self, original_text: str,
                                original_extraction: ExtractionResult,
                                corrected_extraction: ExtractionResult,
                                confidence: float = 0.5,
                                error_type: str = "user_correction",
                                session_id: Optional[str] = None) -> None:
        """Add user correction for learning"""
        pass
    
    @abstractmethod
    async def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get current knowledge graph state"""
        pass
    
    @abstractmethod
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class StreamingExtractorInterface(ABC):
    """Interface for streaming text extractors"""
    
    @abstractmethod
    async def process_chunk(self, text: str, timestamp: float, 
                           chunk_id: int, is_final: bool = False) -> ExtractionResult:
        """Process a streaming text chunk"""
        pass
    
    @abstractmethod
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current streaming state"""
        pass
    
    @abstractmethod
    async def reset_stream(self) -> None:
        """Reset the current stream state"""
        pass

class DualGraphInterface(ABC):
    """Interface for dual graph architecture"""
    
    @abstractmethod
    async def add_extraction(self, text: str, entities: List[str],
                           relations: List[Dict[str, Any]], confidence: float,
                           session_id: Optional[str] = None,
                           extraction_type: str = "conversation") -> None:
        """Add extraction to appropriate memory graph"""
        pass
    
    @abstractmethod
    async def query_knowledge(self, query: str, query_type: str = "entities") -> Dict[str, Any]:
        """Query knowledge from graphs"""
        pass
    
    @abstractmethod
    async def promote_to_long_term(self, session_id: str) -> None:
        """Promote working memory to long-term memory"""
        pass
    
    @abstractmethod
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get dual graph statistics"""
        pass

class ActiveLearningInterface(ABC):
    """Interface for active learning systems"""
    
    @abstractmethod
    async def add_extraction_result(self, text: str, extraction: Dict[str, Any],
                                   confidence: float, is_correct: bool) -> None:
        """Add extraction result for learning"""
        pass
    
    @abstractmethod
    async def add_user_correction(self, original_text: str,
                                original_extraction: Dict[str, Any],
                                corrected_extraction: Dict[str, Any],
                                confidence: float, error_type: str) -> None:
        """Add user correction for learning"""
        pass
    
    @abstractmethod
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get learning progress summary"""
        pass
    
    @abstractmethod
    async def generate_training_examples(self) -> List[Dict[str, Any]]:
        """Generate training examples from learning data"""
        pass

class HotMemIntegrationInterface(ABC):
    """Interface for production integration"""
    
    @abstractmethod
    async def process_transcription(self, text: str, is_final: bool = False) -> None:
        """Process transcription through HotMem"""
        pass
    
    @abstractmethod
    async def get_enhanced_context(self, text: str) -> Dict[str, Any]:
        """Get enhanced context with HotMem knowledge"""
        pass
    
    @abstractmethod
    async def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get current knowledge graph"""
        pass
    
    @abstractmethod
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        pass

class EventCallbackInterface(ABC):
    """Interface for event callbacks"""
    
    @abstractmethod
    async def on_extraction_complete(self, result: ExtractionResult) -> None:
        """Called when extraction is complete"""
        pass
    
    @abstractmethod
    async def on_learning_update(self, summary: Dict[str, Any]) -> None:
        """Called when learning system updates"""
        pass
    
    @abstractmethod
    async def on_graph_updated(self, graph: Dict[str, Any]) -> None:
        """Called when knowledge graph is updated"""
        pass
    
    @abstractmethod
    async def on_error(self, error: Dict[str, Any]) -> None:
        """Called when an error occurs"""
        pass