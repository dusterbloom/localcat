"""
Mem0 Memory Service V2 - Let mem0 work as designed

This service removes all complexity and lets mem0 handle everything internally:
- Single LLM call (no dual schemas)
- mem0 handles fact extraction automatically
- mem0 manages ADD/UPDATE/DELETE using semantic similarity
- No manual operation management
"""

import os
import hashlib
import uuid
import time
from typing import Any, Dict, List, Optional
from collections import deque
from loguru import logger

from pipecat.services.mem0.memory import Mem0MemoryService as BaseMem0MemoryService

try:
    from mem0 import Memory
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Mem0, you need to `pip install mem0ai`")
    raise Exception(f"Missing module: {e}")


class Mem0ServiceV2(BaseMem0MemoryService):
    """
    Mem0 memory service V2 that lets mem0 work as designed
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        local_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        params: Optional[BaseMem0MemoryService.InputParams] = None,
        host: Optional[str] = None,
    ):
        """Initialize with minimal configuration - let mem0 do its job"""
        
        # Initialize parent normally
        super().__init__(
            api_key=api_key,
            local_config=local_config,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            params=params,
            host=host
        )
        
        # Only add basic deduplication
        self.seen_memories = set()
        
        logger.info("Initialized Mem0ServiceV2 - letting mem0 work as designed")
    
    def _retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """Simple memory retrieval with basic deduplication"""
        try:
            memories = super()._retrieve_memories(query)
            
            if not memories or 'results' not in memories:
                return memories
            
            # Simple deduplication only
            unique_results = []
            seen_hashes = set()
            
            for mem in memories['results']:
                mem_text = mem.get('memory', '').strip().lower()
                if len(mem_text) < 5:  # Skip very short memories
                    continue
                    
                mem_hash = hashlib.md5(mem_text.encode()).hexdigest()
                if mem_hash not in seen_hashes:
                    seen_hashes.add(mem_hash)
                    unique_results.append(mem)
                    
                    # Limit to reasonable number
                    if len(unique_results) >= 5:
                        break
            
            memories['results'] = unique_results
            logger.debug(f"Mem0ServiceV2: Retrieved {len(unique_results)} unique memories")
            return memories
            
        except Exception as e:
            logger.error(f"Mem0ServiceV2: Error retrieving memories: {e}")
            return {"results": []}
    
    def _enhance_context_with_memories(self, context, memories: List[Dict[str, Any]]):
        """Simple context enhancement - let parent handle it properly"""
        try:
            # Just call parent method without any modifications
            return super()._enhance_context_with_memories(context, memories)
        except Exception as e:
            logger.error(f"Mem0ServiceV2: Error enhancing context: {e}")
            # Return original context if enhancement fails
            return context
    
    def _store_messages(self, messages: List[Dict[str, str]]) -> None:
        """Simple message storage - let mem0 handle everything"""
        
        # Only store recent relevant messages 
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        
        # Filter out system-only conversations that don't contain facts
        meaningful_messages = []
        for msg in recent_messages:
            content = msg.get('content', '').lower()
            role = msg.get('role', '')
            
            # Skip empty or very short messages
            if len(content.strip()) < 5:
                continue
                
            # Skip pure system messages without facts
            if role == 'system' and 'based on previous conversations' in content:
                continue
                
            meaningful_messages.append(msg)
        
        if not meaningful_messages:
            logger.debug("Mem0ServiceV2: No meaningful messages to store")
            return
        
        logger.debug(f"Mem0ServiceV2: Storing {len(meaningful_messages)} meaningful messages")
        
        try:
            # Reset LM Studio context before each memory operation to prevent accumulation
            self._reset_lm_studio_context()
            
            # Let mem0 do everything - no interference
            super()._store_messages(meaningful_messages)
            logger.debug("Mem0ServiceV2: Successfully stored messages")
        except Exception as e:
            logger.error(f"Mem0ServiceV2: Error storing messages: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_lm_studio_context(self):
        """Reset LM Studio context using session_id rotation to prevent accumulation"""
        try:
            # Generate unique session_id for context isolation
            session_id = f"mem0_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            logger.info(f"Mem0ServiceV2: Resetting LM Studio context with session_id: {session_id}")
            
            # Patch mem0's OpenAI client to inject session_id
            if hasattr(self, 'memory_client') and hasattr(self.memory_client, 'llm'):
                llm = self.memory_client.llm
                
                # Find the OpenAI client and patch it
                if hasattr(llm, 'client') and hasattr(llm.client, 'chat'):
                    original_create = llm.client.chat.completions.create
                    
                    def create_with_session_reset(*args, **kwargs):
                        # Inject session_id to reset LM Studio context
                        kwargs['extra_body'] = kwargs.get('extra_body', {})
                        kwargs['extra_body']['session_id'] = session_id
                        
                        logger.debug(f"Mem0ServiceV2: Injecting session_id {session_id} for context reset")
                        return original_create(*args, **kwargs)
                    
                    # Apply the patch for this memory operation
                    llm.client.chat.completions.create = create_with_session_reset
                    
                    logger.info("Mem0ServiceV2: Successfully applied LM Studio context reset patch")
                else:
                    logger.debug("Mem0ServiceV2: Could not find OpenAI client to patch for context reset")
            else:
                logger.debug("Mem0ServiceV2: Could not access memory_client.llm for context reset")
                
        except Exception as e:
            logger.warning(f"Mem0ServiceV2: Context reset failed (memory operations will continue): {e}")
            # Don't raise - let memory operations continue even if context reset fails