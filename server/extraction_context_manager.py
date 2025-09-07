"""
Extraction Context Manager

Unified context management for extraction operations.
Provides session management, configuration state, and extraction lifecycle control.

Author: SOLID Refactoring
"""

import time
import uuid
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from loguru import logger

@dataclass
class ExtractionContext:
    """Context for a single extraction operation."""
    session_id: str
    extraction_id: str
    text: str
    language: str = "en"
    timestamp: float = field(default_factory=time.time)
    
    # Configuration
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    quality_config: Dict[str, Any] = field(default_factory=dict)
    performance_config: Dict[str, Any] = field(default_factory=dict)
    
    # State tracking
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Results
    entities: Set[str] = field(default_factory=set)
    triples: List[tuple] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    strategy_times: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.extraction_id:
            self.extraction_id = str(uuid.uuid4())
    
    @property
    def duration(self) -> float:
        """Get extraction duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def success(self) -> bool:
        """Check if extraction was successful."""
        return len(self.errors) == 0 and len(self.triples) > 0
    
    def add_error(self, error: str) -> None:
        """Add an error to the context."""
        self.errors.append(error)
        logger.warning(f"Extraction error [{self.extraction_id}]: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the context."""
        self.warnings.append(warning)
        logger.debug(f"Extraction warning [{self.extraction_id}]: {warning}")
    
    def record_strategy_time(self, strategy: str, duration_ms: float) -> None:
        """Record execution time for a strategy."""
        self.strategy_times[strategy] = duration_ms
    
    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return {
            'extraction_id': self.extraction_id,
            'session_id': self.session_id,
            'text_length': len(self.text),
            'language': self.language,
            'duration': self.duration,
            'success': self.success,
            'entities_count': len(self.entities),
            'triples_count': len(self.triples),
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'strategies_used': list(self.strategy_times.keys()),
            'total_strategy_time_ms': sum(self.strategy_times.values())
        }

@dataclass
class SessionContext:
    """Context for a conversation session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Session configuration
    default_language: str = "en"
    user_profile: Dict[str, Any] = field(default_factory=dict)
    session_config: Dict[str, Any] = field(default_factory=dict)
    
    # Session state
    extraction_history: List[str] = field(default_factory=list)
    accumulated_entities: Set[str] = field(default_factory=set)
    accumulated_triples: List[tuple] = field(default_factory=list)
    
    # Performance tracking
    total_extractions: int = 0
    total_extraction_time: float = 0.0
    average_quality_score: float = 0.0
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def add_extraction(self, extraction_id: str) -> None:
        """Add extraction to session history."""
        self.extraction_history.append(extraction_id)
        self.total_extractions += 1
        self.update_activity()
    
    def add_entities(self, entities: Set[str]) -> None:
        """Add entities to session accumulation."""
        self.accumulated_entities.update(entities)
        self.update_activity()
    
    def add_triples(self, triples: List[tuple]) -> None:
        """Add triples to session accumulation."""
        self.accumulated_triples.extend(triples)
        self.update_activity()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        duration = self.last_activity - self.created_at
        return {
            'session_id': self.session_id,
            'duration_seconds': duration,
            'total_extractions': self.total_extractions,
            'unique_entities': len(self.accumulated_entities),
            'total_triples': len(self.accumulated_triples),
            'avg_extraction_time': self.total_extraction_time / max(self.total_extractions, 1),
            'average_quality_score': self.average_quality_score
        }

class ExtractionContextManager:
    """
    Manages extraction contexts and session state.
    
    Responsibilities:
    - Context lifecycle management
    - Session tracking and state
    - Configuration management
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Context storage
        self.active_contexts: Dict[str, ExtractionContext] = {}
        self.session_contexts: Dict[str, SessionContext] = {}
        self.completed_contexts: Dict[str, ExtractionContext] = {}
        
        # Configuration
        self.default_session_config = self.config.get('default_session_config', {})
        self.default_extraction_config = self.config.get('default_extraction_config', {})
        self.context_retention_hours = self.config.get('context_retention_hours', 24)
        self.max_completed_contexts = self.config.get('max_completed_contexts', 1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cleanup
        self.last_cleanup = time.time()
        self.cleanup_interval = self.config.get('cleanup_interval', 3600)  # 1 hour
        
        logger.info("ExtractionContextManager initialized")
    
    def create_extraction_context(self, text: str, session_id: str, 
                                language: str = "en",
                                config: Optional[Dict[str, Any]] = None) -> ExtractionContext:
        """
        Create a new extraction context.
        
        Args:
            text: Text to extract from
            session_id: Session identifier
            language: Language code
            config: Optional configuration overrides
            
        Returns:
            New extraction context
        """
        with self._lock:
            # Ensure session exists
            if session_id not in self.session_contexts:
                self.create_session_context(session_id)
            
            # Create extraction context
            context_config = {**self.default_extraction_config}
            if config:
                context_config.update(config)
            
            context = ExtractionContext(
                session_id=session_id,
                text=text,
                language=language,
                strategy_config=context_config.get('strategies', {}),
                quality_config=context_config.get('quality', {}),
                performance_config=context_config.get('performance', {})
            )
            
            # Store context
            self.active_contexts[context.extraction_id] = context
            
            # Update session
            self.session_contexts[session_id].add_extraction(context.extraction_id)
            
            logger.debug(f"Created extraction context: {context.extraction_id}")
            return context
    
    def get_extraction_context(self, extraction_id: str) -> Optional[ExtractionContext]:
        """Get an extraction context by ID."""
        with self._lock:
            return self.active_contexts.get(extraction_id)
    
    def complete_extraction(self, extraction_id: str, 
                          entities: Set[str], 
                          triples: List[tuple],
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Complete an extraction and move to completed contexts.
        
        Args:
            extraction_id: Extraction context ID
            entities: Extracted entities
            triples: Extracted triples
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            context = self.active_contexts.get(extraction_id)
            if not context:
                logger.warning(f"Extraction context not found: {extraction_id}")
                return False
            
            # Update context
            context.entities = entities
            context.triples = triples
            context.end_time = time.time()
            if metadata:
                context.metadata.update(metadata)
            
            # Move to completed
            self.completed_contexts[extraction_id] = context
            del self.active_contexts[extraction_id]
            
            # Update session
            session = self.session_contexts.get(context.session_id)
            if session:
                session.add_entities(entities)
                session.add_triples(triples)
                session.total_extraction_time += context.duration
            
            # Cleanup if needed
            self._cleanup_if_needed()
            
            logger.debug(f"Completed extraction: {extraction_id}")
            return True
    
    def fail_extraction(self, extraction_id: str, error: str) -> bool:
        """
        Mark an extraction as failed.
        
        Args:
            extraction_id: Extraction context ID
            error: Error message
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            context = self.active_contexts.get(extraction_id)
            if not context:
                return False
            
            context.add_error(error)
            context.end_time = time.time()
            
            # Move to completed
            self.completed_contexts[extraction_id] = context
            del self.active_contexts[extraction_id]
            
            logger.debug(f"Failed extraction: {extraction_id} - {error}")
            return True
    
    def create_session_context(self, session_id: str, 
                             config: Optional[Dict[str, Any]] = None) -> SessionContext:
        """
        Create a new session context.
        
        Args:
            session_id: Session identifier
            config: Optional session configuration
            
        Returns:
            New session context
        """
        with self._lock:
            if session_id in self.session_contexts:
                return self.session_contexts[session_id]
            
            session_config = {**self.default_session_config}
            if config:
                session_config.update(config)
            
            session = SessionContext(
                session_id=session_id,
                session_config=session_config
            )
            
            self.session_contexts[session_id] = session
            logger.info(f"Created session context: {session_id}")
            return session
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Get a session context by ID."""
        with self._lock:
            return self.session_contexts.get(session_id)
    
    def get_session_extractions(self, session_id: str) -> List[ExtractionContext]:
        """Get all extraction contexts for a session."""
        with self._lock:
            extractions = []
            for extraction_id in self.session_contexts.get(session_id, {}).extraction_history:
                if extraction_id in self.completed_contexts:
                    extractions.append(self.completed_contexts[extraction_id])
            return extractions
    
    def cleanup_expired_sessions(self, max_age_hours: float = None) -> int:
        """
        Clean up expired session contexts.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of sessions cleaned up
        """
        if max_age_hours is None:
            max_age_hours = self.context_retention_hours
        
        with self._lock:
            cutoff_time = time.time() - (max_age_hours * 3600)
            expired_sessions = []
            
            for session_id, session in self.session_contexts.items():
                if session.last_activity < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.session_contexts[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
            
            return len(expired_sessions)
    
    def _cleanup_if_needed(self) -> None:
        """Perform cleanup if needed."""
        current_time = time.time()
        
        # Clean up completed contexts if too many
        if len(self.completed_contexts) > self.max_completed_contexts:
            # Remove oldest contexts
            sorted_contexts = sorted(
                self.completed_contexts.items(),
                key=lambda x: x[1].end_time or x[1].start_time
            )
            
            to_remove = len(self.completed_contexts) - self.max_completed_contexts
            for i in range(to_remove):
                extraction_id, context = sorted_contexts[i]
                del self.completed_contexts[extraction_id]
        
        # Periodic cleanup
        if current_time - self.last_cleanup > self.cleanup_interval:
            cleaned_sessions = self.cleanup_expired_sessions()
            if cleaned_sessions > 0:
                logger.info(f"Periodic cleanup: {cleaned_sessions} sessions removed")
            self.last_cleanup = current_time
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        with self._lock:
            return {
                'active_contexts': len(self.active_contexts),
                'session_contexts': len(self.session_contexts),
                'completed_contexts': len(self.completed_contexts),
                'context_retention_hours': self.context_retention_hours,
                'max_completed_contexts': self.max_completed_contexts,
                'last_cleanup': self.last_cleanup,
                'cleanup_interval': self.cleanup_interval
            }
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session summary."""
        with self._lock:
            session = self.session_contexts.get(session_id)
            if not session:
                return None
            
            extractions = self.get_session_extractions(session_id)
            
            return {
                'session': session.get_session_stats(),
                'extractions': [ctx.get_summary() for ctx in extractions],
                'total_unique_entities': len(session.accumulated_entities),
                'total_triples': len(session.accumulated_triples)
            }
    
    def update_session_config(self, session_id: str, config: Dict[str, Any]) -> bool:
        """Update session configuration."""
        with self._lock:
            session = self.session_contexts.get(session_id)
            if session:
                session.session_config.update(config)
                return True
            return False
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session and all its data."""
        with self._lock:
            if session_id not in self.session_contexts:
                return False
            
            # Remove session
            del self.session_contexts[session_id]
            
            # Remove associated extractions from completed contexts
            extraction_ids = self.session_contexts.get(session_id, {}).extraction_history
            for extraction_id in extraction_ids:
                if extraction_id in self.completed_contexts:
                    del self.completed_contexts[extraction_id]
            
            logger.info(f"Cleared session: {session_id}")
            return True
    
    def shutdown(self) -> None:
        """Shutdown the context manager."""
        with self._lock:
            self.active_contexts.clear()
            self.session_contexts.clear()
            self.completed_contexts.clear()
            logger.info("ExtractionContextManager shutdown complete")