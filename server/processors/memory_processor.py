"""
Memory Processor - Handles memory operations in the pipeline
"""

import time
from typing import List, Optional, Dict, Any
from loguru import logger
from dataclasses import dataclass

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from components.memory.memory_store import MemoryStore, Paths
from components.memory.memory_hotpath import HotMemory


@dataclass
class MemoryProcessorConfig:
    """Configuration for memory processor"""
    sqlite_path: Optional[str] = None
    lmdb_dir: Optional[str] = None
    user_id: str = "default-user"
    enable_metrics: bool = True
    max_facts_per_injection: int = 3
    enable_real_time_corrections: bool = True


class MemoryProcessor(FrameProcessor):
    """
    Memory processor that handles memory operations in the pipeline.
    Replaces the hotpath_processor with a cleaner, more modular approach.
    """
    
    def __init__(self, config: MemoryProcessorConfig):
        super().__init__()
        self.config = config
        
        # Initialize memory components
        self.memory_store = MemoryStore(
            sqlite_path=config.sqlite_path,
            lmdb_dir=config.lmdb_dir
        )
        self.hot_memory = HotMemory(
            store=self.memory_store,
            user_id=config.user_id
        )
        
        # Metrics
        self._metrics = {
            'total_processed': 0,
            'memory_hits': 0,
            'corrections_applied': 0,
            'avg_latency_ms': 0
        }
        
        logger.info(f"🧠 Memory processor initialized for user: {config.user_id}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames"""
        await super().process_frame(frame, direction)
        
        start_time = time.time()
        
        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame, direction)
        elif isinstance(frame, LLMMessagesFrame):
            await self._handle_llm_messages(frame, direction)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
        
        # Update metrics
        if self.config.enable_metrics:
            self._update_metrics(time.time() - start_time)
    
    async def _handle_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Handle transcription frames - extract and store memory"""
        try:
            # Extract memory from transcription
            memory_facts = await self._extract_memory_facts(frame.text)
            
            if memory_facts:
                # Store memory facts
                await self._store_memory_facts(memory_facts)
                logger.debug(f"🧠 Stored {len(memory_facts)} memory facts")
            
            # Pass through the frame
            await self.push_frame(frame, direction)
            
        except Exception as e:
            logger.error(f"🧠 Error processing transcription: {e}")
            await self.push_frame(frame, direction)
    
    async def _handle_llm_messages(self, frame: LLMMessagesFrame, direction: FrameDirection):
        """Handle LLM message frames - inject memory context"""
        try:
            # Get relevant memory for context
            memory_context = await self._get_memory_context(frame.messages)
            
            if memory_context:
                # Inject memory context into messages
                enhanced_messages = self._inject_memory_context(frame.messages, memory_context)
                enhanced_frame = LLMMessagesFrame(enhanced_messages)
                await self.push_frame(enhanced_frame, direction)
            else:
                # Pass through original frame
                await self.push_frame(frame, direction)
                
        except Exception as e:
            logger.error(f"🧠 Error handling LLM messages: {e}")
            await self.push_frame(frame, direction)
    
    async def _extract_memory_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract memory facts from text"""
        # Use the hot memory system to extract facts
        facts = await self.hot_memory.extract_facts(text)
        return facts
    
    async def _store_memory_facts(self, facts: List[Dict[str, Any]]):
        """Store memory facts"""
        for fact in facts:
            await self.memory_store.add_fact(
                user_id=self.config.user_id,
                fact_text=fact['text'],
                fact_type=fact.get('type', 'general'),
                confidence=fact.get('confidence', 0.8),
                metadata=fact.get('metadata', {})
            )
    
    async def _get_memory_context(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Get relevant memory context for current conversation"""
        # Extract entities and context from recent messages
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        context_text = " ".join([msg.get('content', '') for msg in recent_messages])
        
        # Get relevant memory facts
        memory_facts = await self.hot_memory.retrieve_relevant_facts(
            context_text, 
            limit=self.config.max_facts_per_injection
        )
        
        # Format as memory bullets
        return [fact['text'] for fact in memory_facts]
    
    def _inject_memory_context(self, messages: List[Dict[str, Any]], memory_context: List[str]) -> List[Dict[str, Any]]:
        """Inject memory context into messages"""
        if not memory_context:
            return messages
        
        # Create memory message
        memory_message = {
            'role': 'user',
            'content': f"Memory Context:\n" + "\n".join(f"• {fact}" for fact in memory_context)
        }
        
        # Insert after system message or at beginning
        enhanced_messages = messages.copy()
        system_index = next((i for i, msg in enumerate(enhanced_messages) if msg.get('role') == 'system'), -1)
        
        if system_index >= 0:
            enhanced_messages.insert(system_index + 1, memory_message)
        else:
            enhanced_messages.insert(0, memory_message)
        
        return enhanced_messages
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self._metrics['total_processed'] += 1
        self._metrics['avg_latency_ms'] = (
            (self._metrics['avg_latency_ms'] * (self._metrics['total_processed'] - 1) + 
             processing_time * 1000) / self._metrics['total_processed']
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self._metrics.copy()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.memory_store.close()
        logger.info("🧠 Memory processor cleanup completed")