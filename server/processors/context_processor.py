"""
Context Processor - Manages conversation context and state
"""

import time
import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, TextFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class ContextType(Enum):
    """Types of context information"""
    CONVERSATION = "conversation"
    MEMORY = "memory"
    EXTRACTION = "extraction"
    SYSTEM = "system"
    USER_PROFILE = "user_profile"
    SESSION = "session"


@dataclass
class ContextItem:
    """Individual context item"""
    content: str
    context_type: ContextType
    timestamp: float
    priority: float = 0.5
    relevance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[float] = None  # Time to live in seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if context item has expired"""
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl


@dataclass
class ContextWindow:
    """Sliding window of context items"""
    max_items: int = 50
    max_tokens: int = 2000
    items: List[ContextItem] = field(default_factory=list)
    
    def add_item(self, item: ContextItem):
        """Add item to context window"""
        self.items.append(item)
        self._cleanup_expired()
        self._enforce_limits()
    
    def get_relevant_items(self, query: str = "", limit: int = 10) -> List[ContextItem]:
        """Get most relevant context items"""
        relevant_items = self.items.copy()
        
        # Filter by query if provided
        if query:
            relevant_items = [item for item in relevant_items if self._is_relevant(item, query)]
        
        # Sort by priority and relevance
        relevant_items.sort(key=lambda x: (x.priority, x.relevance_score), reverse=True)
        
        return relevant_items[:limit]
    
    def _cleanup_expired(self):
        """Remove expired items"""
        self.items = [item for item in self.items if not item.is_expired]
    
    def _enforce_limits(self):
        """Enforce item and token limits"""
        # Remove oldest items if over limit
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        
        # Estimate tokens and remove if over limit
        estimated_tokens = self._estimate_tokens()
        if estimated_tokens > self.max_tokens:
            # Remove lowest priority items first
            self.items.sort(key=lambda x: (x.priority, x.relevance_score))
            while estimated_tokens > self.max_tokens and self.items:
                removed = self.items.pop(0)
                estimated_tokens -= self._estimate_item_tokens(removed)
    
    def _is_relevant(self, item: ContextItem, query: str) -> bool:
        """Check if item is relevant to query"""
        if not query:
            return True
        
        query_lower = query.lower()
        content_lower = item.content.lower()
        
        # Simple keyword matching
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        # Check for any overlap
        return len(query_words.intersection(content_words)) > 0
    
    def _estimate_tokens(self) -> int:
        """Estimate total tokens in context window"""
        return sum(self._estimate_item_tokens(item) for item in self.items)
    
    def _estimate_item_tokens(self, item: ContextItem) -> int:
        """Estimate tokens for a single item"""
        # Rough estimate: 4 chars per token
        return max(1, len(item.content) // 4)


@dataclass
class ContextProcessorConfig:
    """Configuration for context processor"""
    max_context_items: int = 50
    max_context_tokens: int = 2000
    context_ttl: int = 3600  # 1 hour default
    enable_memory_context: bool = True
    enable_extraction_context: bool = True
    enable_session_context: bool = True
    enable_user_profile: bool = True
    relevance_threshold: float = 0.3
    enable_metrics: bool = True


class ContextProcessor(FrameProcessor):
    """
    Context processor that manages conversation context and state.
    Provides context aggregation, filtering, and injection capabilities.
    """
    
    def __init__(self, config: ContextProcessorConfig):
        super().__init__()
        self.config = config
        
        # Initialize context windows
        self.context_windows = {
            ContextType.CONVERSATION: ContextWindow(
                max_items=config.max_context_items,
                max_tokens=config.max_context_tokens
            ),
            ContextType.MEMORY: ContextWindow(
                max_items=20,
                max_tokens=500
            ),
            ContextType.EXTRACTION: ContextWindow(
                max_items=30,
                max_tokens=800
            ),
            ContextType.SYSTEM: ContextWindow(
                max_items=10,
                max_tokens=300
            ),
            ContextType.USER_PROFILE: ContextWindow(
                max_items=5,
                max_tokens=200
            ),
            ContextType.SESSION: ContextWindow(
                max_items=15,
                max_tokens=400
            )
        }
        
        # Session state
        self.session_id = None
        self.user_id = None
        self.conversation_history = []
        
        # Performance tracking
        self._metrics = {
            'total_processed': 0,
            'context_injections': 0,
            'context_items_added': 0,
            'avg_processing_time_ms': 0,
            'context_hits': 0,
            'context_misses': 0
        }
        
        logger.info(f"🎯 Context processor initialized with "
                   f"max_items={config.max_context_items}, "
                   f"max_tokens={config.max_context_tokens}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames"""
        await super().process_frame(frame, direction)
        
        start_time = time.time()
        
        if isinstance(frame, StartFrame):
            await self._handle_start_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription_frame(frame, direction)
        elif isinstance(frame, TextFrame):
            await self._handle_text_frame(frame, direction)
        elif isinstance(frame, LLMMessagesFrame):
            await self._handle_llm_messages_frame(frame, direction)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
        
        # Update metrics
        if self.config.enable_metrics:
            self._update_metrics(time.time() - start_time)
    
    async def _handle_start_frame(self, frame: StartFrame, direction: FrameDirection):
        """Handle start frame to initialize session"""
        # Extract session and user info
        self.session_id = getattr(frame, 'session_id', 'default-session')
        self.user_id = getattr(frame, 'user_id', 'default-user')
        
        # Initialize session context
        await self._initialize_session_context()
        
        await self.push_frame(frame, direction)
    
    async def _handle_transcription_frame(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Handle transcription frames"""
        try:
            # Add to conversation context
            await self._add_conversation_context(frame.text, role='user')
            
            # Pass through frame
            await self.push_frame(frame, direction)
            
        except Exception as e:
            logger.error(f"🎯 Error handling transcription frame: {e}")
            await self.push_frame(frame, direction)
    
    async def _handle_text_frame(self, frame: TextFrame, direction: FrameDirection):
        """Handle text frames"""
        try:
            # Check for extraction metadata
            extraction_result = getattr(frame, 'metadata', {}).get('extraction_result', None)
            
            if extraction_result:
                # Add extraction context
                await self._add_extraction_context(extraction_result)
            
            # Add to conversation context if from assistant
            if hasattr(frame, 'role') and frame.role == 'assistant':
                await self._add_conversation_context(frame.text, role='assistant')
            
            # Pass through frame
            await self.push_frame(frame, direction)
            
        except Exception as e:
            logger.error(f"🎯 Error handling text frame: {e}")
            await self.push_frame(frame, direction)
    
    async def _handle_llm_messages_frame(self, frame: LLMMessagesFrame, direction: FrameDirection):
        """Handle LLM messages frames - inject context"""
        try:
            # Get relevant context
            context_items = await self._gather_relevant_context(frame.messages)
            
            if context_items:
                # Inject context into messages
                enhanced_messages = await self._inject_context(frame.messages, context_items)
                enhanced_frame = LLMMessagesFrame(enhanced_messages)
                await self.push_frame(enhanced_frame, direction)
                self._metrics['context_injections'] += 1
            else:
                # Pass through original frame
                await self.push_frame(frame, direction)
                
        except Exception as e:
            logger.error(f"🎯 Error handling LLM messages frame: {e}")
            await self.push_frame(frame, direction)
    
    async def _initialize_session_context(self):
        """Initialize session-specific context"""
        if self.config.enable_session_context:
            session_item = ContextItem(
                content=f"Session started: {self.session_id}",
                context_type=ContextType.SESSION,
                timestamp=time.time(),
                priority=0.9,
                metadata={'session_id': self.session_id, 'user_id': self.user_id}
            )
            self.context_windows[ContextType.SESSION].add_item(session_item)
    
    async def _add_conversation_context(self, text: str, role: str):
        """Add conversation context"""
        if not text or not text.strip():
            return
        
        # Create context item
        context_item = ContextItem(
            content=f"{role}: {text}",
            context_type=ContextType.CONVERSATION,
            timestamp=time.time(),
            priority=0.7 if role == 'user' else 0.6,
            metadata={'role': role, 'session_id': self.session_id},
            ttl=self.config.context_ttl
        )
        
        self.context_windows[ContextType.CONVERSATION].add_item(context_item)
        self._metrics['context_items_added'] += 1
        
        # Add to conversation history
        self.conversation_history.append({
            'role': role,
            'text': text,
            'timestamp': time.time()
        })
    
    async def _add_extraction_context(self, extraction_result: Dict[str, Any]):
        """Add extraction context"""
        if not self.config.enable_extraction_context:
            return
        
        # Add entities as context
        entities = extraction_result.get('entities', [])
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_label = entity.get('label', 'UNKNOWN')
            
            if entity_text:
                context_item = ContextItem(
                    content=f"Entity: {entity_text} ({entity_label})",
                    context_type=ContextType.EXTRACTION,
                    timestamp=time.time(),
                    priority=0.5,
                    metadata={'entity': entity, 'session_id': self.session_id},
                    ttl=self.config.context_ttl
                )
                self.context_windows[ContextType.EXTRACTION].add_item(context_item)
        
        # Add facts as context
        facts = extraction_result.get('facts', [])
        for fact in facts:
            fact_text = fact.get('text', '')
            if fact_text:
                context_item = ContextItem(
                    content=f"Fact: {fact_text}",
                    context_type=ContextType.EXTRACTION,
                    timestamp=time.time(),
                    priority=0.6,
                    metadata={'fact': fact, 'session_id': self.session_id},
                    ttl=self.config.context_ttl
                )
                self.context_windows[ContextType.EXTRACTION].add_item(context_item)
    
    async def _gather_relevant_context(self, messages: List[Dict[str, Any]]) -> List[ContextItem]:
        """Gather relevant context for current messages"""
        all_context_items = []
        
        # Get recent conversation context
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        query_text = " ".join([msg.get('content', '') for msg in recent_messages])
        
        # Collect context from different sources
        for context_type, window in self.context_windows.items():
            if context_type == ContextType.CONVERSATION:
                # Always include recent conversation
                items = window.get_relevant_items(query_text, limit=5)
            elif context_type == ContextType.EXTRACTION and self.config.enable_extraction_context:
                # Include extraction context
                items = window.get_relevant_items(query_text, limit=3)
            elif context_type == ContextType.SESSION and self.config.enable_session_context:
                # Include session context
                items = window.get_relevant_items(query_text, limit=2)
            else:
                # Other context types
                items = window.get_relevant_items(query_text, limit=2)
            
            all_context_items.extend(items)
        
        # Sort by priority and relevance
        all_context_items.sort(key=lambda x: (x.priority, x.relevance_score), reverse=True)
        
        # Apply relevance threshold
        filtered_items = [
            item for item in all_context_items 
            if item.relevance_score >= self.config.relevance_threshold
        ]
        
        if filtered_items:
            self._metrics['context_hits'] += 1
        else:
            self._metrics['context_misses'] += 1
        
        return filtered_items[:10]  # Limit total context items
    
    async def _inject_context(self, messages: List[Dict[str, Any]], 
                            context_items: List[ContextItem]) -> List[Dict[str, Any]]:
        """Inject context into messages"""
        if not context_items:
            return messages
        
        # Create context message
        context_parts = []
        
        for item in context_items:
            context_parts.append(f"• {item.content}")
        
        context_message = {
            'role': 'system',
            'content': f"Context Information:\n" + "\n".join(context_parts)
        }
        
        # Insert after system message or at beginning
        enhanced_messages = messages.copy()
        system_index = next((i for i, msg in enumerate(enhanced_messages) if msg.get('role') == 'system'), -1)
        
        if system_index >= 0:
            enhanced_messages.insert(system_index + 1, context_message)
        else:
            enhanced_messages.insert(0, context_message)
        
        return enhanced_messages
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self._metrics['total_processed'] += 1
        
        if processing_time > 0:
            self._metrics['avg_processing_time_ms'] = (
                (self._metrics['avg_processing_time_ms'] * (self._metrics['total_processed'] - 1) + 
                 processing_time * 1000) / self._metrics['total_processed']
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = self._metrics.copy()
        
        # Calculate context hit rate
        total_context_requests = metrics['context_hits'] + metrics['context_misses']
        if total_context_requests > 0:
            metrics['context_hit_rate'] = metrics['context_hits'] / total_context_requests
        else:
            metrics['context_hit_rate'] = 0
        
        # Add context window sizes
        metrics['context_window_sizes'] = {
            context_type.value: len(window.items)
            for context_type, window in self.context_windows.items()
        }
        
        return metrics
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'conversation_length': len(self.conversation_history),
            'context_windows': {
                context_type.value: {
                    'item_count': len(window.items),
                    'estimated_tokens': window._estimate_tokens()
                }
                for context_type, window in self.context_windows.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🎯 Context processor cleanup completed")