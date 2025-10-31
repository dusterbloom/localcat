"""
HotMemService: Pipecat-compatible memory service with tool-based interface.

Drop-in replacement for Mem0MemoryService that:
- Leverages existing HotPath performance (<200ms)
- Provides explicit tool-based interface (no intent classification)
- Maintains full Pipecat compatibility
- Uses existing storage systems
"""

import asyncio
import json
import time
import os
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMMessagesFrame,
    TranscriptionFrame,
    ErrorFrame
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .memory_store import MemoryStore, Paths
from .memory_hotpath import HotMemory
from .session_tracker import SessionTracker
from .confidence_strategy import ConfidenceStrategy


class HotMemService(FrameProcessor):
    """
    Pipecat-compatible memory service with tool-based interface.

    Drop-in replacement for Mem0MemoryService that combines:
    - HotPath's ultra-fast performance (<200ms)
    - Tool-based explicit interface (no intent classification)
    - Full Mem0 compatibility (for existing Pipecat users)
    """

    # Core tool definitions based on design document
    TOOL_DEFINITIONS = [
        {
            "name": "hotmem_remember",
            "description": "Store information in memory for future recall",
            "parameters": {
                "type": "object",
                "properties": {
                    "information": {
                        "type": "string",
                        "description": "Information to remember"
                    }
                },
                "required": ["information"]
            }
        },
        {
            "name": "hotmem_recall",
            "description": "Retrieve specific information from memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to recall from memory"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "hotmem_forget",
            "description": "Remove information from memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to forget/remove from memory"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "hotmem_search",
            "description": "Search memory with optional search type",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["conversation", "graph", "context", "related", "entity", "temporal", "semantic"],
                        "description": "Type of search to perform"
                    }
                },
                "required": ["query"]
            }
        }
    ]

    def __init__(
        self,
        *,
        # Mem0MemoryService compatibility parameters
        api_key: Optional[str] = None,
        local_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        host: Optional[str] = None,
        # HotPath-specific parameters
        sqlite_path: Optional[str] = None,
        lmdb_dir: Optional[str] = None,
        session_tracker: Optional[SessionTracker] = None,
        confidence_strategy: Optional[ConfidenceStrategy] = None,
        enable_dspy_extraction: Optional[bool] = None,
        context_aggregator: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize HotMemService with Mem0 compatibility.

        Args:
            api_key: Ignored (for Mem0 compatibility)
            local_config: Ignored (for Mem0 compatibility)
            user_id: User identifier for memory context
            agent_id: Agent identifier for memory context
            run_id: Run identifier for memory context
            params: Additional parameters (for Mem0 compatibility)
            host: Ignored (for Mem0 compatibility)
            sqlite_path: Path to SQLite database (default: from env)
            lmdb_dir: Path to LMDB directory (default: from env)
            session_tracker: Optional session tracker
            confidence_strategy: Optional confidence scoring strategy
        """
        # REQUIRED: Call parent constructor first
        super().__init__()

        # Mem0 compatibility: At least one ID must be provided
        if not any([user_id, agent_id, run_id]):
            raise ValueError("At least one of user_id, agent_id, or run_id must be provided")

        # Store IDs for compatibility
        self.user_id = user_id or "default-user"
        self.agent_id = agent_id or "hotmem-agent"
        self.run_id = run_id

        # Initialize HotPath storage system
        paths = Paths(
            sqlite_path=sqlite_path,
            lmdb_dir=lmdb_dir
        )
        self.store = MemoryStore(paths)
        self.hot = HotMemory(
            self.store,
            confidence_strategy=confidence_strategy,
            enable_dspy_extraction=enable_dspy_extraction
        )

        # Pre-warm NLP to avoid first-turn latency
        try:
            self.hot.prewarm("en")
        except Exception as e:
            logger.warning(f"HotMem prewarm failed: {e}")

        # Rebuild RAM indices from persistent store
        try:
            self.hot.rebuild_from_store()
        except Exception as e:
            logger.warning(f"Could not rebuild from store (starting fresh): {e}")

        # Session management
        self._session_id = f"{self.user_id}_{int(time.time())}_{os.urandom(4).hex()}"
        self._turn_id = 0
        self._session_tracker = session_tracker
        self.context_aggregator = context_aggregator  # For context enhancement integration
        self.last_query = None  # Mem0 compatibility

        # Performance settings
        self._bullets_max = int(os.getenv("MEMORY_BULLETS_MAX", "3"))
        self._inject_role = os.getenv("MEMORY_INJECT_ROLE", "system").strip().lower()
        self._inject_header = os.getenv("MEMORY_INJECT_HEADER", "[HotMem Context]")

        logger.info(f"HotMemService initialized: user_id={user_id}, agent_id={agent_id}, session_id={self._session_id}")

    def _store_messages(self, messages: List[Dict[str, Any]]):
        """
        Store messages using HotPath backend (maintains automatic storage).

        Compatible with Mem0MemoryService interface but uses HotPath storage.
        """
        try:
            logger.debug(f"Storing {len(messages)} messages in HotPath")

            # Extract user messages for processing
            for message in messages:
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    content = message.get("content", "").strip()
                    if content:
                        # Store conversation text for retrieval
                        now_ts = int(time.time() * 1000)
                        self.store.enqueue_mention(
                            self._session_id,
                            content,
                            now_ts,
                            self._session_id,
                            self._turn_id
                        )

                        # Process through HotPath extraction pipeline
                        try:
                            self._turn_id += 1
                            bullets, triples = self.hot.process_turn(
                                content,
                                self._session_id,
                                self._turn_id
                            )
                            logger.debug(f"HotPath processed: {len(triples)} facts, {len(bullets)} bullets")
                        except Exception as e:
                            logger.warning(f"HotPath processing failed: {e}")

            # Flush to ensure persistence
            self.store.flush_if_needed()

        except Exception as e:
            logger.error(f"Error storing messages in HotPath: {e}")

    def _retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve memories using HotPath backend.

        Compatible with Mem0MemoryService interface but uses HotPath retrieval.
        """
        try:
            logger.debug(f"Retrieving memories for query: {query}")

            # Use HotPath retrieval system
            bullets = self.hot.retrieve_bullets(query, read_only=True)

            # Convert to Mem0-compatible format
            memories = []
            for i, bullet in enumerate(bullets[:self._bullets_max]):
                memories.append({
                    "memory": bullet,
                    "score": 1.0 - (i * 0.1),  # Simple scoring
                    "metadata": {
                        "source": "hotpath",
                        "session_id": self._session_id,
                        "timestamp": int(time.time())
                    }
                })

            logger.debug(f"Retrieved {len(memories)} memories from HotPath")
            return {"results": memories} if memories else {"results": []}

        except Exception as e:
            logger.error(f"Error retrieving memories from HotPath: {e}")
            return {"results": []}

    def _enhance_context_with_memories(self, context: Union[LLMContext, OpenAILLMContext], query: str):
        """
        Enhanced context building: Add memories and tool availability notice.

        This combines traditional memory retrieval with tool availability.
        """
        logger.debug(f"[HotMemService] _enhance_context_with_memories called with query: '{query[:50]}...'")

        # Skip if same query (Mem0 compatibility)
        if self.last_query == query:
            logger.debug(f"[HotMemService] Skipping duplicate query")
            return

        self.last_query = query

        # Skip enhancement if a memory header is already present (avoids double-injection)
        try:
            existing = False
            for m in context.get_messages():
                if isinstance(m, dict) and m.get('role') == 'system':
                    content = m.get('content', '')
                    if isinstance(content, str) and content.startswith(self._inject_header):
                        existing = True
                        break
            if existing:
                logger.debug("Memory header already present; skipping HotMemService enhancement")
                return
        except Exception:
            pass

        # First, get relevant memories using HotPath
        memories = self._retrieve_memories(query)
        if memories.get("results"):
            # Format memories as a message (similar to Mem0MemoryService)
            memory_text = self._inject_header + "\n"
            for i, memory in enumerate(memories["results"], 1):
                memory_text += f"{i}. {memory.get('memory', '')}\n"

            # Add tool availability notice
            memory_text += "\nMemory tools available: hotmem_remember, hotmem_recall, hotmem_forget, hotmem_search\n (use only when asked explicitly by the user)"

            # Add as system message
            context.add_message({"role": "system", "content": memory_text})
            logger.debug(f"Enhanced context with {len(memories['results'])} memories and tool notice")
        else:
            # No memories found, but still add tool availability
            tool_text = self._inject_header + "\nMemory tools available: hotmem_remember, hotmem_recall, hotmem_forget, hotmem_search (use only when asked explicitly by the user)\n"
            context.add_message({"role": "system", "content": tool_text})
            logger.debug("Enhanced context with memory tool availability notice")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process frames with memory enhancement (similar to Mem0MemoryService).

        Handles both TranscriptionFrame (from UserTranscriptProcessor) and context frames.
        """
        await super().process_frame(frame, direction)

        # Debug: Log all frame types received
        # logger.debug(f"[HotMemService] Received frame: {type(frame).__name__} from {direction.name}")

        # Handle TranscriptionFrame (from UserTranscriptProcessor) - convert to context and enhance
        if isinstance(frame, TranscriptionFrame):
            try:
                # Extract transcription text
                transcription_text = getattr(frame, 'text', '') or getattr(frame, 'content', '')
                if not transcription_text:
                    await self.push_frame(frame, direction)
                    return

                logger.debug(f"[HotMemService] Processing TranscriptionFrame: '{transcription_text[:50]}...'")

                # Create user message from transcription
                user_message = {"role": "user", "content": transcription_text}
                messages = [user_message]

                # Store message automatically
                self._store_messages(messages)

                # Check if this is a question (avoid extraction from questions)
                is_question = self._is_question_from_text(transcription_text)

                if not is_question:
                    # Extract facts from non-questions for memory storage
                    try:
                        bullets, triples = self.hot.process_turn(
                            transcription_text,
                            self._session_id,
                            self._turn_id
                        )
                        self._turn_id += 1
                        logger.debug(f"[HotMemService] Extracted {len(triples)} facts, {len(bullets)} bullets")
                    except Exception as e:
                        logger.warning(f"[HotMemService] Memory extraction failed: {e}")

                # Pass through the TranscriptionFrame
                # The context_aggregator will create the context frame next
                await self.push_frame(frame, direction)

            except Exception as e:
                logger.error(f"[HotMemService] Error processing TranscriptionFrame: {e}")
                await self.push_frame(frame, direction)  # Still pass original frame

        # Handle context frames (same as Mem0MemoryService)
        elif isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            logger.debug(f"[HotMemService] Processing context frame: {type(frame).__name__}")
            context = frame.context
            messages = None

            try:
                # Get latest user message for memory enhancement
                context_messages = context.get_messages()
                latest_user_message = None

                for message in reversed(context_messages):
                    if message.get("role") == "user" and isinstance(message.get("content"), str):
                        latest_user_message = message.get("content")
                        break

                if latest_user_message:
                    logger.debug(f"[HotMemService] Latest user message: '{latest_user_message[:50]}...'")
                    # Enhanced with memories and tool availability notice
                    self._enhance_context_with_memories(context, latest_user_message)
                    logger.debug(f"[HotMemService] Enhanced context with memories and tools")

                # Forward enhanced context
                await self.push_frame(frame, direction)

            except Exception as e:
                logger.error(f"[HotMemService] Error processing context frame: {str(e)}")
                await self.push_frame(frame, direction)  # Still pass original frame

        elif isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            context = LLMContext(messages)

            try:
                # Get latest user message for memory enhancement
                context_messages = context.get_messages()
                latest_user_message = None

                for message in reversed(context_messages):
                    if message.get("role") == "user" and isinstance(message.get("content"), str):
                        latest_user_message = message.get("content")
                        break

                if latest_user_message:
                    # Enhanced with memories and tool availability notice
                    self._enhance_context_with_memories(context, latest_user_message)

                # Forward enhanced context
                await self.push_frame(LLMMessagesFrame(context.get_messages()), direction)

            except Exception as e:
                logger.error(f"[HotMemService] Error processing messages frame: {str(e)}")
                await self.push_frame(frame, direction)  # Still pass original frame
        else:
            # For other frame types, just pass through
            await self.push_frame(frame, direction)


    def _is_question_from_text(self, text: str) -> bool:
        """Check if text is a question to avoid fact extraction from questions."""
        if not text or not text.strip():
            return False

        text_lower = text.strip().lower()

        # Check for question marks
        if '?' in text:
            return True

        # Check for question words at the beginning
        question_starters = [
            'what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose',
            'are', 'is', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'will', 'shall', 'may', 'might', 'must', 'have', 'has', 'had'
        ]

        words = text_lower.split()
        if words and words[0] in question_starters:
            return True

        return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics (HotPath compatibility)."""
        return {
            'session_id': self._session_id,
            'turn_id': self._turn_id,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'hot_metrics': self.hot.get_metrics() if hasattr(self.hot, 'get_metrics') else {},
            'store_metrics': self.store.get_metrics() if hasattr(self.store, 'get_metrics') else {}
        }

    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.store.flush()
            logger.debug("HotMemService cleanup complete")
        except Exception as e:
            logger.error(f"HotMemService cleanup error: {e}")


# Alias for backward compatibility
Mem0MemoryService = HotMemService
