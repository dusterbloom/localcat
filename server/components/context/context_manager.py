"""
Context Manager - Unified context management across pipeline stages
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from processors.context_processor import ContextItem, ContextType, ContextWindow
from components.memory.memory_store import MemoryStore


class ContextState(Enum):
    """Context lifecycle states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class ContextSession:
    """Session-specific context data"""
    session_id: str
    user_id: str
    start_time: float
    last_activity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_windows: Dict[ContextType, ContextWindow] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds"""
        return time.time() - self.start_time
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds"""
        return time.time() - self.last_activity


@dataclass
class ContextManagerConfig:
    """Configuration for context manager"""
    max_session_duration: int = 3600  # 1 hour
    max_idle_time: int = 1800  # 30 minutes
    max_sessions_per_user: int = 5
    enable_cross_session_context: bool = True
    enable_persistence: bool = True
    cleanup_interval: int = 300  # 5 minutes
    enable_metrics: bool = True


class ContextManager:
    """
    Unified context management across pipeline stages.
    Manages context lifecycle, state, and performance monitoring.
    """
    
    def __init__(self, config: ContextManagerConfig, memory_store: Optional[MemoryStore] = None):
        self.config = config
        self.memory_store = memory_store
        self.state = ContextState.INITIALIZING
        
        # Session management
        self.active_sessions: Dict[str, ContextSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}
        
        # Performance monitoring
        self._metrics = {
            'total_sessions': 0,
            'active_sessions': 0,
            'context_operations': 0,
            'avg_operation_time_ms': 0,
            'context_hits': 0,
            'context_misses': 0,
            'session_timeouts': 0
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("🎛️ Context manager initialized")
    
    async def initialize(self):
        """Initialize the context manager"""
        try:
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Load persisted sessions if enabled
            if self.config.enable_persistence and self.memory_store:
                await self._load_persisted_sessions()
            
            self.state = ContextState.ACTIVE
            logger.info("🎛️ Context manager initialized successfully")
            
        except Exception as e:
            logger.error(f"🎛️ Error initializing context manager: {e}")
            self.state = ContextState.CLOSED
            raise
    
    async def create_session(self, user_id: str, session_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new context session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}_{user_id}"
        
        # Check session limits
        if user_id in self.user_sessions:
            user_session_list = self.user_sessions[user_id]
            if len(user_session_list) >= self.config.max_sessions_per_user:
                # Remove oldest session
                oldest_session_id = user_session_list[0]
                await self.close_session(oldest_session_id)
        
        # Create new session
        session = ContextSession(
            session_id=session_id,
            user_id=user_id,
            start_time=time.time(),
            last_activity=time.time(),
            metadata=metadata or {}
        )
        
        # Initialize context windows
        for context_type in ContextType:
            session.context_windows[context_type] = ContextWindow(
                max_items=50,
                max_tokens=2000
            )
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Update user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        # Update metrics
        self._metrics['total_sessions'] += 1
        self._metrics['active_sessions'] += 1
        
        logger.info(f"🎛️ Created session: {session_id} for user: {user_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[ContextSession]:
        """Get a session by ID"""
        session = self.active_sessions.get(session_id)
        if session:
            session.last_activity = time.time()
        return session
    
    async def close_session(self, session_id: str):
        """Close a session and save context if enabled"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        try:
            # Save session context if persistence enabled
            if self.config.enable_persistence and self.memory_store:
                await self._save_session_context(session)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            # Remove from user sessions
            if session.user_id in self.user_sessions:
                user_sessions = self.user_sessions[session.user_id]
                if session_id in user_sessions:
                    user_sessions.remove(session_id)
            
            # Update metrics
            self._metrics['active_sessions'] -= 1
            
            logger.info(f"🎛️ Closed session: {session_id}")
            
        except Exception as e:
            logger.error(f"🎛️ Error closing session {session_id}: {e}")
    
    async def add_context_item(self, session_id: str, item: ContextItem):
        """Add a context item to a session"""
        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"🎛️ Session not found: {session_id}")
            return
        
        session.context_windows[item.context_type].add_item(item)
        session.last_activity = time.time()
        
        self._metrics['context_operations'] += 1
        
        logger.debug(f"🎛️ Added context item to session {session_id}: {item.context_type.value}")
    
    async def get_context_items(self, session_id: str, context_type: ContextType, 
                              query: str = "", limit: int = 10) -> List[ContextItem]:
        """Get context items from a session"""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        window = session.context_windows.get(context_type)
        if not window:
            return []
        
        items = window.get_relevant_items(query, limit)
        
        if items:
            self._metrics['context_hits'] += 1
        else:
            self._metrics['context_misses'] += 1
        
        return items
    
    async def get_cross_session_context(self, user_id: str, query: str, 
                                       limit: int = 5) -> List[ContextItem]:
        """Get context across all user sessions"""
        if not self.config.enable_cross_session_context:
            return []
        
        all_items = []
        
        # Get all user sessions
        user_session_ids = self.user_sessions.get(user_id, [])
        
        for session_id in user_session_ids:
            session = self.active_sessions.get(session_id)
            if session:
                # Get relevant items from all context types
                for context_type, window in session.context_windows.items():
                    items = window.get_relevant_items(query, limit=2)
                    all_items.extend(items)
        
        # Sort by priority and timestamp
        all_items.sort(key=lambda x: (x.priority, x.timestamp), reverse=True)
        
        return all_items[:limit]
    
    async def update_context_relevance(self, session_id: str, context_type: ContextType,
                                      item_content: str, relevance_delta: float):
        """Update relevance score for context items"""
        session = await self.get_session(session_id)
        if not session:
            return
        
        window = session.context_windows.get(context_type)
        if not window:
            return
        
        # Find and update matching items
        for item in window.items:
            if item.content == item_content:
                item.relevance_score = max(0.0, min(1.0, item.relevance_score + relevance_delta))
                break
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session context"""
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        summary = {
            'session_id': session_id,
            'user_id': session.user_id,
            'duration': session.duration,
            'idle_time': session.idle_time,
            'context_windows': {}
        }
        
        for context_type, window in session.context_windows.items():
            summary['context_windows'][context_type.value] = {
                'item_count': len(window.items),
                'estimated_tokens': window._estimate_tokens(),
                'expired_items': len([item for item in window.items if item.is_expired])
            }
        
        return summary
    
    async def get_user_context_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's context across all sessions"""
        user_session_ids = self.user_sessions.get(user_id, [])
        
        summary = {
            'user_id': user_id,
            'active_sessions': len(user_session_ids),
            'sessions': []
        }
        
        for session_id in user_session_ids:
            session_summary = await self.get_session_summary(session_id)
            summary['sessions'].append(session_summary)
        
        return summary
    
    @asynccontextmanager
    async def session_context(self, user_id: str, session_id: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Context manager for session lifecycle"""
        session_id = await self.create_session(user_id, session_id, metadata)
        
        try:
            yield session_id
        finally:
            await self.close_session(session_id)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.state == ContextState.ACTIVE:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🎛️ Error in cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # Check session duration
            if current_time - session.start_time > self.config.max_session_duration:
                expired_sessions.append(session_id)
                continue
            
            # Check idle time
            if current_time - session.last_activity > self.config.max_idle_time:
                expired_sessions.append(session_id)
                continue
        
        # Close expired sessions
        for session_id in expired_sessions:
            await self.close_session(session_id)
            self._metrics['session_timeouts'] += 1
        
        if expired_sessions:
            logger.info(f"🎛️ Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _save_session_context(self, session: ContextSession):
        """Save session context to persistent storage"""
        if not self.memory_store:
            return
        
        try:
            # Prepare session data for storage
            session_data = {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'start_time': session.start_time,
                'end_time': time.time(),
                'metadata': session.metadata,
                'context_summary': await self.get_session_summary(session.session_id)
            }
            
            # Save to memory store
            await self.memory_store.add_session_context(session_data)
            
            logger.debug(f"🎛️ Saved session context: {session.session_id}")
            
        except Exception as e:
            logger.error(f"🎛️ Error saving session context: {e}")
    
    async def _load_persisted_sessions(self):
        """Load persisted sessions from storage"""
        if not self.memory_store:
            return
        
        try:
            # This would be implemented based on the memory store interface
            # For now, just log that we would load sessions
            logger.debug("🎛️ Loading persisted sessions (not implemented)")
            
        except Exception as e:
            logger.error(f"🎛️ Error loading persisted sessions: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get context manager metrics"""
        metrics = self._metrics.copy()
        
        # Calculate additional metrics
        if metrics['context_operations'] > 0:
            metrics['context_hit_rate'] = (
                metrics['context_hits'] / (metrics['context_hits'] + metrics['context_misses'])
            )
        else:
            metrics['context_hit_rate'] = 0
        
        metrics['current_state'] = self.state.value
        metrics['cleanup_task_running'] = self._cleanup_task and not self._cleanup_task.done()
        
        return metrics
    
    async def cleanup(self):
        """Cleanup context manager resources"""
        self.state = ContextState.CLOSING
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.close_session(session_id)
        
        self.state = ContextState.CLOSED
        logger.info("🎛️ Context manager cleanup completed")


class ContextManagerFactory:
    """Factory for creating context managers"""
    
    @staticmethod
    def create_default_manager(memory_store: Optional[MemoryStore] = None) -> ContextManager:
        """Create a context manager with default configuration"""
        config = ContextManagerConfig()
        return ContextManager(config, memory_store)
    
    @staticmethod
    def create_lightweight_manager() -> ContextManager:
        """Create a lightweight context manager"""
        config = ContextManagerConfig(
            max_session_duration=1800,  # 30 minutes
            max_idle_time=600,  # 10 minutes
            max_sessions_per_user=2,
            enable_cross_session_context=False,
            enable_persistence=False,
            cleanup_interval=600  # 10 minutes
        )
        return ContextManager(config)
    
    @staticmethod
    def create_persistent_manager(memory_store: MemoryStore) -> ContextManager:
        """Create a persistent context manager"""
        config = ContextManagerConfig(
            max_session_duration=7200,  # 2 hours
            max_idle_time=3600,  # 1 hour
            max_sessions_per_user=10,
            enable_cross_session_context=True,
            enable_persistence=True,
            cleanup_interval=300  # 5 minutes
        )
        return ContextManager(config, memory_store)