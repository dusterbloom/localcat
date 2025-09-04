"""
Custom Mem0 Memory Service for LM Studio Compatibility

This module provides a custom implementation of Pipecat's Mem0MemoryService
that fixes compatibility issues with LM Studio and other local LLM servers.

Issues fixed:
1. Removes async_mode parameter that doesn't exist in mem0ai API
2. Removes response_format requirement incompatible with LM Studio
3. Adds better error handling for empty/malformed JSON responses
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from pipecat.services.mem0.memory import Mem0MemoryService as BaseMem0MemoryService

try:
    from mem0 import Memory, MemoryClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Mem0, you need to `pip install mem0ai`")
    raise Exception(f"Missing module: {e}")


class LMStudioCompatibleMemory(Memory):
    """
    Wrapper for mem0.Memory that removes incompatible parameters for LM Studio
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Patch the underlying LLM to remove response_format
        self._patch_llm_for_compatibility()
    
    def _patch_llm_for_compatibility(self):
        """
        Patch the LLM's generate_response method to remove response_format parameter
        """
        logger.debug(f"Attempting to patch LLM. Has llm: {hasattr(self, 'llm')}")
        if hasattr(self, 'llm'):
            logger.debug(f"LLM type: {type(self.llm)}, has generate_response: {hasattr(self.llm, 'generate_response')}")
            
        if hasattr(self, 'llm') and hasattr(self.llm, 'generate_response'):
            original_generate = self.llm.generate_response
            
            def patched_generate_response(messages, **kwargs):
                logger.debug(f"Original kwargs: {kwargs.get('response_format', {})}")
                
                # Convert mem0's json_object to LM Studio's json_schema format
                if kwargs.get('response_format', {}).get('type') == 'json_object':
                    
                    # Detect which type of call this is based on the prompt content
                    prompt_text = ""
                    if messages and len(messages) > 0:
                        # Get the last user message to determine call type
                        for msg in reversed(messages):
                            if msg.get('role') == 'user':
                                prompt_text = msg.get('content', '').lower()
                                break
                    
                    # Determine schema based on prompt content
                    if any(keyword in prompt_text for keyword in ['update', 'existing memories', 'add', 'delete', 'modify']):
                        # Memory update call - needs memory schema
                        schema = {
                            "type": "object", 
                            "properties": {
                                "memory": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "text": {"type": "string"}, 
                                            "event": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NONE"]},
                                            "old_memory": {"type": "string"}
                                        },
                                        "required": ["text", "event"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["memory"],
                            "additionalProperties": False
                        }
                        schema_name = "memory_update"
                        logger.debug("Using memory update schema")
                    else:
                        # Fact extraction call - needs facts schema
                        schema = {
                            "type": "object",
                            "properties": {
                                "facts": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": ["facts"],
                            "additionalProperties": False
                        }
                        schema_name = "fact_extraction" 
                        logger.debug("Using fact extraction schema")
                    
                    kwargs['response_format'] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema
                        }
                    }
                    logger.debug(f"Converted response_format to json_schema: {schema_name}")
                
                try:
                    response = original_generate(messages, **kwargs)
                    logger.debug(f"Memory extraction model response: {response[:200]}...")
                    return response
                except Exception as e:
                    logger.warning(f"Memory extraction model failed with JSON schema: {e}. Trying without response_format.")
                    # If JSON schema fails, try without any response_format
                    kwargs_no_format = {k: v for k, v in kwargs.items() if k != 'response_format'}
                    try:
                        response = original_generate(messages, **kwargs_no_format)
                        logger.debug("Memory extraction succeeded without response_format")
                        return response
                    except Exception as e2:
                        logger.error(f"Memory extraction failed completely: {e2}. Returning appropriate empty JSON.")
                        # Return appropriate empty response based on detected schema
                        if 'memory_update' in str(kwargs.get('response_format', {})):
                            return '{"memory": []}'
                        else:
                            return '{"facts": []}'
            
            self.llm.generate_response = patched_generate_response
            logger.info("Patched LLM generate_response for LM Studio compatibility")
    
    def add(self, messages, **kwargs):
        """
        Override add method to handle LM Studio compatibility issues
        """
        # Remove async_mode if present (doesn't exist in mem0ai)
        kwargs.pop('async_mode', None)
        
        # Remove output_format if it's v1.1 (causes issues with some versions)
        if kwargs.get('output_format') == 'v1.1':
            kwargs.pop('output_format', None)
        
        try:
            # Call original add method without problematic parameters
            return super().add(messages, **kwargs)
        except Exception as e:
            logger.warning(f"Memory add failed with error: {e}. Attempting without inference.")
            # If it fails, try without inference (bypasses JSON parsing)
            kwargs['infer'] = False
            try:
                return super().add(messages, **kwargs)
            except Exception as e2:
                logger.error(f"Memory add failed even without inference: {e2}")
                # Return empty result if all fails
                return {"results": []}


class CustomMem0MemoryService(BaseMem0MemoryService):
    """
    Custom Mem0 Memory Service that works with LM Studio and local LLMs
    
    This service extends Pipecat's Mem0MemoryService to fix compatibility issues:
    - Removes async_mode parameter that causes errors
    - Handles empty/malformed JSON responses from local LLMs
    - Provides fallback mechanisms for memory storage
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        local_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Custom Mem0 Memory Service
        
        Args:
            api_key: API key for cloud-based Mem0 (optional)
            local_config: Configuration for local Mem0 setup
            user_id: Unique identifier for the user
            agent_id: Optional identifier for the agent
            run_id: Optional identifier for the run
            **kwargs: Additional parameters for the base service
        """
        # Intercept local_config to ensure lmstudio provider is configured correctly
        if local_config and local_config.get('llm', {}).get('provider') == 'lmstudio':
            # Configure LM Studio specific settings for memory extraction
            import os
            mem0_base_url = os.getenv('MEM0_BASE_URL', 'http://localhost:1234/v1')  # LM Studio for memory extraction
            mem0_model = os.getenv('MEM0_MODEL', 'qwen2.5-7b-instruct')  # Memory extraction model in LM Studio
            
            logger.info(f"Configuring LM Studio for memory extraction with model: {mem0_model}")
            # Add LM Studio specific configuration for memory extraction
            local_config['llm']['config']['model'] = mem0_model
            logger.info(f"Memory extraction will use: {mem0_base_url} with model {mem0_model}")
        
        # Initialize the base service
        super().__init__(
            api_key=api_key,
            local_config=local_config,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            **kwargs
        )
        
        # Replace memory client with our compatible version if using local Memory
        if isinstance(self.memory_client, Memory):
            logger.info("Wrapping Memory client with LM Studio compatible version")
            # Create new compatible memory with same config
            compatible_memory = LMStudioCompatibleMemory(config=self.memory_client.config)
            self.memory_client = compatible_memory
    
    def _store_messages(self, messages: List[Dict[str, str]]):
        """
        Override _store_messages to handle LM Studio compatibility
        
        Removes async_mode and handles errors gracefully
        """
        try:
            logger.debug(f"Storing {len(messages)} messages in Mem0")
            params = {
                # Remove async_mode - doesn't exist in mem0ai
                "messages": messages,
                "metadata": {"platform": "pipecat"},
                # Remove output_format if causing issues
            }
            
            # Add user_id, agent_id, run_id if available
            for id_field in ["user_id", "agent_id", "run_id"]:
                if getattr(self, id_field):
                    params[id_field] = getattr(self, id_field)
            
            # For local Memory instances, remove output_format
            if isinstance(self.memory_client, Memory):
                params.pop("output_format", None)
            
            # For local models, start with infer=False to avoid JSON format issues
            # Based on GitHub issue #3391: Ollama/LM Studio have JSON parsing issues with inference
            if isinstance(self.memory_client, Memory):
                logger.info("Using local Memory client - setting infer=False to avoid JSON issues")
                params['infer'] = False
                self.memory_client.add(**params)
            else:
                # Try to add with inference first for cloud clients
                try:
                    self.memory_client.add(**params)
                except Exception as e:
                    if "json" in str(e).lower() or "response_format" in str(e).lower():
                        # JSON parsing or format error - try without inference
                        logger.warning(f"Memory inference failed: {e}. Storing raw messages.")
                        params['infer'] = False
                        self.memory_client.add(**params)
                    else:
                        raise
                    
        except Exception as e:
            logger.error(f"Error storing messages in Mem0: {e}")
            # Continue processing even if memory storage fails
    
    def _retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Override _retrieve_memories with better error handling
        """
        try:
            return super()._retrieve_memories(query)
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            # Return empty list to continue processing
            return []
    
    def _enhance_context_with_memories(
        self,
        context_messages: List[Dict[str, str]],
        memories: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Override _enhance_context_with_memories with better error handling
        """
        try:
            return super()._enhance_context_with_memories(context_messages, memories)
        except Exception as e:
            logger.error(f"Error enhancing context with memories: {e}")
            # Return original context if enhancement fails
            return context_messages


def patch_mem0_for_lmstudio():
    """
    Utility function to patch mem0 library for LM Studio compatibility
    
    This can be called at startup to monkey-patch the mem0 library
    if you prefer that approach over using the custom service.
    
    WARNING: This modifies the installed library globally for this process.
    """
    try:
        import mem0.memory.main as mem0_main
        
        # Patch the Memory class's add method
        original_add = mem0_main.Memory.add
        
        def patched_add(self, messages, **kwargs):
            # Remove problematic parameters
            kwargs.pop('async_mode', None)
            if kwargs.get('output_format') == 'v1.1':
                kwargs.pop('output_format', None)
            return original_add(self, messages, **kwargs)
        
        mem0_main.Memory.add = patched_add
        logger.info("Successfully patched mem0 for LM Studio compatibility")
        
    except Exception as e:
        logger.error(f"Failed to patch mem0: {e}")


# Example usage documentation
"""
Usage in bot.py:

Instead of:
    from pipecat.services.mem0.memory import Mem0MemoryService
    
Use:
    from custom_mem0_service import CustomMem0MemoryService as Mem0MemoryService

Or if you need both:
    from custom_mem0_service import CustomMem0MemoryService
    
    # Then use CustomMem0MemoryService instead of Mem0MemoryService
    memory = CustomMem0MemoryService(
        local_config=local_config,
        user_id=os.getenv("USER_ID"),
        agent_id=os.getenv("AGENT_ID"),
        params=Mem0MemoryService.InputParams(
            search_limit=10,
            search_threshold=0.3,
            api_version="v2",
        )
    )

Alternative - Monkey Patching (not recommended for production):
    from custom_mem0_service import patch_mem0_for_lmstudio
    
    # Call this once at startup before importing mem0
    patch_mem0_for_lmstudio()
    
    # Then use normal imports
    from pipecat.services.mem0.memory import Mem0MemoryService
"""