"""
Dual-Model Mem0 Memory Service for LM Studio

This module uses two specialized models:
1. Fact extraction model (Qwen 3 4B) - Extracts facts from conversations
2. Memory update model (Qwen 3 4B Instruct) - Handles ADD/UPDATE/DELETE operations

This approach provides better JSON compliance than using a single small model.
"""

import os
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional
from collections import deque
from loguru import logger
from openai import OpenAI

from pipecat.services.mem0.memory import Mem0MemoryService as BaseMem0MemoryService

try:
    from mem0 import Memory, MemoryClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Mem0, you need to `pip install mem0ai`")
    raise Exception(f"Missing module: {e}")


class DualModelMemory(Memory):
    """
    Memory class that uses two different models for different tasks
    """
    
    def __init__(self, config=None):
        # Extract dual model configuration from config if provided
        # Make a copy to preserve original config
        if isinstance(config, dict):
            self.fact_model = config.get('fact_model', None)
            self.update_model = config.get('update_model', None)
            self.lm_studio_base_url = config.get('lm_studio_base_url', None)
            # Remove from config copy for base class
            config_copy = config.copy()
            config_copy.pop('fact_model', None)
            config_copy.pop('update_model', None)
            config_copy.pop('lm_studio_base_url', None)
        else:
            self.fact_model = None
            self.update_model = None
            self.lm_studio_base_url = None
            config_copy = config
        
        super().__init__(config_copy)
        
        # Set up dual model clients if configured
        logger.debug(f"DualModelMemory init - fact_model: {self.fact_model}, update_model: {self.update_model}, base_url: {self.lm_studio_base_url}")
        if self.fact_model and self.update_model and self.lm_studio_base_url:
            self._setup_dual_models()
        else:
            # Fall back to patching for single model
            logger.debug(f"Falling back to single model patching - missing config")
            self._patch_llm_for_compatibility()
    
    def _setup_dual_models(self):
        """Set up two separate OpenAI clients for different models"""
        logger.info(f"Setting up dual models: fact={self.fact_model}, update={self.update_model}")
        
        # Create clients for each model
        self.fact_client = OpenAI(
            base_url=self.lm_studio_base_url,
            api_key="lm-studio"
        )
        self.update_client = OpenAI(
            base_url=self.lm_studio_base_url,
            api_key="lm-studio"
        )
        
        # Patch the generate_response to use appropriate model
        if hasattr(self, 'llm') and hasattr(self.llm, 'generate_response'):
            original_generate = self.llm.generate_response
            
            def dual_model_generate(messages, **kwargs):
                # Detect which operation this is based on the messages
                is_fact_extraction = False
                is_memory_update = False
                
                if messages and len(messages) > 0:
                    for msg in messages:
                        content = msg.get('content', '').lower()
                        if 'extract facts' in content or '"facts"' in content:
                            is_fact_extraction = True
                            break
                        elif 'memory' in content and ('add' in content or 'update' in content or 'delete' in content):
                            is_memory_update = True
                            break
                
                # Choose model based on operation
                if is_fact_extraction:
                    model = self.fact_model
                    client = self.fact_client
                    logger.debug(f"Using fact extraction model: {model}")
                elif is_memory_update:
                    model = self.update_model
                    client = self.update_client
                    logger.debug(f"Using memory update model: {model}")
                else:
                    # Default to fact extraction model
                    model = self.fact_model
                    client = self.fact_client
                    logger.debug(f"Using default model: {model}")
                    return original_generate(messages, **kwargs)
                
                # Make request with appropriate model
                # LM Studio uses simpler JSON format without schema enforcement
                try:
                    # Add JSON instruction to the last message
                    modified_messages = messages.copy()
                    if modified_messages and len(modified_messages) > 0:
                        last_msg = modified_messages[-1].copy()
                        if is_fact_extraction:
                            last_msg['content'] += '\n\nRespond ONLY with valid JSON in this format: {"facts": ["fact1", "fact2", ...]}'
                        elif is_memory_update:
                            last_msg['content'] += '\n\nRespond ONLY with valid JSON in this format: {"memory": [{"text": "...", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "..."}]}'
                        modified_messages[-1] = last_msg
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=modified_messages,
                        max_tokens=kwargs.get('max_tokens', 300),
                        temperature=kwargs.get('temperature', 0.1)
                        # Note: LM Studio doesn't support response_format parameter
                    )
                    
                    result = response.choices[0].message.content
                    logger.debug(f"Model {model} response: {result[:200]}...")
                    return result
                    
                except Exception as e:
                    logger.warning(f"Dual model request failed: {e}, trying without JSON format")
                    # Try without JSON format enforcement
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=kwargs.get('max_tokens', 300),
                            temperature=kwargs.get('temperature', 0.1)
                        )
                        result = response.choices[0].message.content
                        
                        # Try to extract JSON if present
                        import json
                        try:
                            # Try to find JSON in the response
                            start = result.find('{')
                            end = result.rfind('}') + 1
                            if start >= 0 and end > start:
                                json_str = result[start:end]
                                json.loads(json_str)  # Validate
                                return json_str
                        except:
                            pass
                        
                        return result
                    except Exception as e2:
                        logger.error(f"Fallback request also failed: {e2}")
                        # Return empty response
                        if is_fact_extraction:
                            return '{"facts": []}'
                        else:
                            return '{"memory": []}'
            
            self.llm.generate_response = dual_model_generate
            logger.info("Successfully set up dual model generation")
    
    def _patch_llm_for_compatibility(self):
        """Fallback single-model patching (kept for compatibility)"""
        logger.debug("Using single model compatibility patching")
        # This would contain the original patching logic from custom_mem0_service.py
        pass


class DualModelMem0MemoryService(BaseMem0MemoryService):
    """
    Enhanced Mem0 memory service that uses two specialized models
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
        dual_model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize with dual model support. Same parameters as base class plus dual_model_config.
        
        Args:
            dual_model_config: Configuration for dual models, should contain:
                - fact_model: Model name for fact extraction
                - update_model: Model name for memory updates
                - base_url: LM Studio base URL
        """
        # Store dual model config
        self.dual_model_config = dual_model_config or {
            'fact_model': os.getenv('MEM0_FACT_MODEL', 'qwen/qwen3-4b'),
            'update_model': os.getenv('MEM0_UPDATE_MODEL', 'qwen3-4b-instruct-2507'),
            'base_url': os.getenv('MEM0_BASE_URL', 'http://127.0.0.1:1234/v1')
        }
        
        # Update local config to include dual model settings
        if local_config:
            local_config.update({
                'fact_model': self.dual_model_config['fact_model'],
                'update_model': self.dual_model_config['update_model'],
                'lm_studio_base_url': self.dual_model_config['base_url']
            })
        
        # Call parent init with standard parameters
        super().__init__(
            api_key=api_key,
            local_config=local_config,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            params=params,
            host=host
        )
        
        # Replace Memory instance with DualModelMemory if using local config
        if local_config:
            logger.info("Initializing DualModelMemory with two specialized models")
            # Create enhanced config with dual model settings
            enhanced_config = local_config.copy()
            enhanced_config['fact_model'] = self.dual_model_config['fact_model']
            enhanced_config['update_model'] = self.dual_model_config['update_model']
            enhanced_config['lm_studio_base_url'] = self.dual_model_config['base_url']
            
            # Create DualModelMemory instance and then set up dual models
            logger.debug(f"Enhanced config for DualModelMemory: fact_model={enhanced_config.get('fact_model')}, update_model={enhanced_config.get('update_model')}, lm_studio_base_url={enhanced_config.get('lm_studio_base_url')}")
            # Use from_config for proper parsing, then patch
            self.memory_client = DualModelMemory.from_config(local_config)
            # Now patch in the dual model configuration
            self.memory_client.fact_model = self.dual_model_config['fact_model']
            self.memory_client.update_model = self.dual_model_config['update_model']
            self.memory_client.lm_studio_base_url = self.dual_model_config['base_url']
            # Trigger dual model setup
            if hasattr(self.memory_client, '_setup_dual_models'):
                self.memory_client._setup_dual_models()
    
    # Inherit all other methods from base class
    # They will automatically use the dual model setup