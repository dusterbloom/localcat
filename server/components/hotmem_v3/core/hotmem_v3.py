"""
HotMem v3 - Revolutionary Self-Improving AI System

Main HotMem v3 class that integrates all components:
- Dual graph architecture (working + long-term memory)
- Active learning system for continuous improvement
- Real-time streaming extraction
- Production-ready optimization
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import time
import threading
from dataclasses import dataclass, field

# Import HotMem v3 components
from .dual_graph_architecture import DualGraphArchitecture
from ..extraction.streaming_extraction import StreamingExtractor, StreamingChunk
from ..training.active_learning import ActiveLearningSystem
from ..integration.production_integration import HotMemIntegration

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class HotMemV3Config:
    """Configuration for HotMem v3"""
    model_path: Optional[str] = None
    enable_real_time: bool = True
    confidence_threshold: float = 0.7
    max_working_memory_entities: int = 100
    max_long_term_memory_entities: int = 10000
    active_learning_enabled: bool = True
    promotion_interval: int = 1800  # seconds
    enable_streaming: bool = True
    
class HotMemV3:
    """
    Main HotMem v3 class - Revolutionary Self-Improving AI System
    
    This class integrates all HotMem v3 components into a unified system:
    - Real-time knowledge graph construction
    - Dual memory architecture with intelligent promotion/demotion
    - Active learning with pattern detection
    - Streaming extraction for voice conversations
    - Production-ready optimization
    """
    
    def __init__(self, config: Optional[HotMemV3Config] = None):
        """Initialize HotMem v3 with configuration"""
        self.config = config or HotMemV3Config()
        self.initialized = False
        self.callbacks = {}
        
        # Initialize components
        self.dual_graph = DualGraphArchitecture()
        self.active_learning = ActiveLearningSystem()
        self.streaming_extractor = None
        self.production_integration = None
        
        # Performance tracking
        self.stats = {
            'extractions_processed': 0,
            'learning_iterations': 0,
            'last_update': time.time(),
            'uptime': time.time()
        }
        
        logger.info("HotMem v3 initialized")
    
    async def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return
            
        try:
            # Initialize streaming extractor if enabled
            if self.config.enable_streaming:
                self.streaming_extractor = StreamingExtractor(
                    model_path=self.config.model_path
                )
                logger.info("Streaming extractor initialized")
            
            # Initialize production integration
            self.production_integration = HotMemIntegration(
                model_path=self.config.model_path,
                enable_real_time=self.config.enable_real_time,
                confidence_threshold=self.config.confidence_threshold
            )
            logger.info("Production integration initialized")
            
            # Configure dual graph
            self.dual_graph.configure(
                max_working_entities=self.config.max_working_memory_entities,
                max_long_term_entities=self.config.max_long_term_memory_entities,
                promotion_interval=self.config.promotion_interval
            )
            
            # Configure active learning
            if self.config.active_learning_enabled:
                self.active_learning.configure(
                    enable_pattern_detection=True,
                    enable_uncertainty_sampling=True,
                    confidence_threshold=self.config.confidence_threshold
                )
            
            self.initialized = True
            logger.info("HotMem v3 fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize HotMem v3: {e}")
            raise
    
    def process_text(self, text: str, is_final: bool = True, 
                    session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process text through HotMem v3 pipeline
        
        Args:
            text: Input text to process
            is_final: Whether this is the final text
            session_id: Optional session identifier
            
        Returns:
            Processing results with entities, relations, and confidence
        """
        if not self.initialized:
            raise RuntimeError("HotMem v3 not initialized. Call initialize() first.")
        
        start_time = time.time()
        self.stats['extractions_processed'] += 1
        
        try:
            # Extract entities and relations
            if self.streaming_extractor:
                chunk = StreamingChunk(
                    text=text,
                    timestamp=time.time(),
                    chunk_id=self.stats['extractions_processed'],
                    is_final=is_final
                )
                extraction_result = self.streaming_extractor.process_chunk(chunk)
            else:
                # Fallback to production integration
                if self.production_integration:
                    self.production_integration.process_transcription(text, is_final)
                    graph = self.production_integration.get_knowledge_graph()
                    extraction_result = type('ExtractionResult', (), {
                        'entities': graph.get('entities', []),
                        'relations': graph.get('relations', []),
                        'confidence': 0.8,
                        'processing_time': time.time() - start_time
                    })()
                else:
                    extraction_result = type('ExtractionResult', (), {
                        'entities': [],
                        'relations': [],
                        'confidence': 0.0,
                        'processing_time': time.time() - start_time
                    })()
            
            # Add to dual graph architecture
            self.dual_graph.add_extraction(
                text=text,
                entities=[e.get('text', str(e)) if isinstance(e, dict) else str(e) for e in extraction_result.entities],
                relations=[
                    {
                        'subject': r.get('subject', ''),
                        'predicate': r.get('predicate', ''),
                        'object': r.get('object', '')
                    }
                    for r in extraction_result.relations
                ],
                confidence=extraction_result.confidence,
                session_id=session_id,
                extraction_type="conversation"
            )
            
            # Add to active learning if enabled
            if self.config.active_learning_enabled:
                self.active_learning.add_extraction_result(
                    text=text,
                    extraction={
                        'entities': extraction_result.entities,
                        'relations': extraction_result.relations
                    },
                    confidence=extraction_result.confidence,
                    is_correct=True  # Assume correct until corrected
                )
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['last_update'] = time.time()
            
            # Trigger callbacks
            self._trigger_callbacks('extraction_complete', {
                'text': text,
                'entities': extraction_result.entities,
                'relations': extraction_result.relations,
                'confidence': extraction_result.confidence,
                'processing_time': processing_time,
                'session_id': session_id
            })
            
            return {
                'text': text,
                'entities': extraction_result.entities,
                'relations': extraction_result.relations,
                'confidence': extraction_result.confidence,
                'processing_time': processing_time,
                'session_id': session_id,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                'text': text,
                'entities': [],
                'relations': [],
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'session_id': session_id,
                'success': False,
                'error': str(e)
            }
    
    def add_user_correction(self, original_text: str, 
                          original_extraction: Dict[str, Any],
                          corrected_extraction: Dict[str, Any],
                          confidence: float = 0.5,
                          error_type: str = "user_correction",
                          session_id: Optional[str] = None):
        """
        Add user correction for active learning
        
        Args:
            original_text: Original text that was processed
            original_extraction: Original extraction result
            corrected_extraction: Corrected extraction result
            confidence: Confidence in the correction
            error_type: Type of error/correction
            session_id: Optional session identifier
        """
        if not self.config.active_learning_enabled:
            return
            
        self.active_learning.add_user_correction(
            original_text=original_text,
            original_extraction=original_extraction,
            corrected_extraction=corrected_extraction,
            confidence=confidence,
            error_type=error_type,
            user_id=None,  # Could be added in future
            session_id=session_id
        )
        
        # Update dual graph with corrected information
        self.dual_graph.add_extraction(
            text=original_text,
            entities=corrected_extraction.get('entities', []),
            relations=corrected_extraction.get('relations', []),
            confidence=confidence,
            session_id=session_id,
            extraction_type="correction"
        )
        
        self.stats['learning_iterations'] += 1
        logger.info(f"User correction added: {error_type}")
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """
        Get current knowledge graph state
        
        Returns:
            Combined knowledge graph from working and long-term memory
        """
        return self.dual_graph.get_combined_graph()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        
        Returns:
            System performance and usage statistics
        """
        stats = self.stats.copy()
        stats['uptime'] = time.time() - stats['uptime']
        
        # Add dual graph stats
        dual_graph_stats = self.dual_graph.get_system_stats()
        stats['dual_graph'] = dual_graph_stats
        
        # Add active learning stats
        if self.config.active_learning_enabled:
            learning_stats = self.active_learning.get_learning_summary()
            stats['active_learning'] = learning_stats
        
        # Add performance metrics
        if self.stats['extractions_processed'] > 0:
            avg_processing_time = stats.get('avg_processing_time', 0)
            stats['performance'] = {
                'extractions_per_second': self.stats['extractions_processed'] / stats['uptime'],
                'average_processing_time': avg_processing_time,
                'learning_rate': self.stats['learning_iterations'] / stats['uptime'] if stats['uptime'] > 0 else 0
            }
        
        return stats
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Add event callback
        
        Args:
            event_type: Type of event ('extraction_complete', 'learning_update', etc.)
            callback: Callback function to call when event occurs
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Trigger callbacks for specific event type"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")
    
    def export_knowledge_graph(self, filepath: str):
        """
        Export knowledge graph to file
        
        Args:
            filepath: Path to save the knowledge graph
        """
        graph = self.get_knowledge_graph()
        with open(filepath, 'w') as f:
            json.dump(graph, f, indent=2)
        logger.info(f"Knowledge graph exported to {filepath}")
    
    def import_knowledge_graph(self, filepath: str):
        """
        Import knowledge graph from file
        
        Args:
            filepath: Path to load the knowledge graph from
        """
        with open(filepath, 'r') as f:
            graph = json.load(f)
        
        # Add imported data to dual graph
        for entity in graph.get('entities', []):
            self.dual_graph.add_extraction(
                text=entity.get('text', ''),
                entities=[entity.get('text', str(entity))],
                relations=[],
                confidence=entity.get('confidence', 0.8),
                extraction_type="import"
            )
        
        for relation in graph.get('relations', []):
            self.dual_graph.add_extraction(
                text=relation.get('text', ''),
                entities=[relation.get('subject', ''), relation.get('object', '')],
                relations=[relation],
                confidence=relation.get('confidence', 0.8),
                extraction_type="import"
            )
        
        logger.info(f"Knowledge graph imported from {filepath}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.streaming_extractor:
            await self.streaming_extractor.cleanup()
        
        if self.production_integration:
            await self.production_integration.cleanup()
        
        logger.info("HotMem v3 cleanup completed")