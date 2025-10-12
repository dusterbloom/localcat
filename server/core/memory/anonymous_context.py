"""
Anonymous-aware context aggregator wrapper that can clear context and skip Context Guide in anonymous mode.
"""

from typing import Any, Optional, Dict
from loguru import logger
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


class AnonymousAwareContextAggregator:
    """
    Wrapper that makes OpenAILLMContext aware of anonymous mode.
    
    In anonymous mode:
    - Clears conversation history
    - Removes Context Guide system message
    - Sets anonymous mode flag for downstream processors
    """
    
    def __init__(self, context_aggregator: Any, context: OpenAILLMContext):
        """Initialize with a Pipecat context aggregator and its context."""
        self._aggregator = context_aggregator
        self._context = context
        self._original_messages = None
        self._anonymous_mode = False
        self._context_guide_added = False
        
        # Track if Context Guide was added
        messages = context.get_messages()
        for msg in messages:
            if msg.get("role") == "system" and "Context Guide:" in msg.get("content", ""):
                self._context_guide_added = True
                break
    
    def user(self) -> Any:
        """Get user aggregator (delegated to wrapped aggregator)."""
        return self._aggregator.user()
    
    def assistant(self) -> Any:
        """Get assistant aggregator (delegated to wrapped aggregator)."""
        return self._aggregator.assistant()
    
    @property
    def context(self) -> OpenAILLMContext:
        """Get the underlying context."""
        return self._context
    
    def set_anonymous_mode(self, enabled: bool) -> None:
        """
        Enable or disable anonymous mode.
        
        When enabled:
        - Clears all conversation history
        - Removes Context Guide system message if present
        - Sets flag for downstream processors
        """
        if self._anonymous_mode == enabled:
            return  # No change needed
        
        self._anonymous_mode = bool(enabled)
        
        if enabled:
            logger.info("[AnonymousContext] Entering anonymous mode - clearing context")
            
            # Store current messages (except system messages)
            self._original_messages = self._context.get_messages().copy()
            
            # Clear context to only essential system messages
            messages = self._context.get_messages()
            filtered_messages = []
            
            for msg in messages:
                content = msg.get("content", "")
                # Keep only essential system messages; drop Context Guide and Session Context
                if msg.get("role") == "system":
                    if ("Context Guide:" in content) or (isinstance(content, str) and content.startswith("[Session Context]")):
                        logger.debug("[AnonymousContext] Removing non-essential system message for anonymous mode")
                    else:
                        filtered_messages.append(msg)
                # Skip all user/assistant messages (conversation history)
                elif msg.get("role") in ["user", "assistant"]:
                    logger.debug(f"[AnonymousContext] Removing {msg.get('role')} message: {content[:50]}...")
                    continue
                else:
                    filtered_messages.append(msg)
            
            # Reset context with filtered messages
            self._context._messages = filtered_messages
            
            # Add anonymous marker as system message
            self._context.add_message({
                "role": "system", 
                "content": "Anonymous mode: No conversation history or memory context is available."
            })
            
        else:
            logger.info("[AnonymousContext] Exiting anonymous mode - restoring context")
            
            # Restore original messages if available
            if self._original_messages:
                self._context._messages = self._original_messages.copy()
                self._original_messages = None
            else:
                # If no original messages, at least ensure Context Guide is back if it was there
                if self._context_guide_added:
                    # Check if Context Guide is missing
                    messages = self._context.get_messages()
                    has_context_guide = any(
                        "Context Guide:" in msg.get("content", "") 
                        for msg in messages 
                        if msg.get("role") == "system"
                    )
                    
                    if not has_context_guide:
                        # Re-add Context Guide
                        guide_default = (
                            "Context Guide:\n"
                            "- When uncertain, ask one short clarifying question.\n"
                            "- Keep replies brief and helpful."
                        )
                        import os
                        guide_text = os.getenv("MEMORY_CONTEXT_GUIDE", guide_default)
                        self._context.add_message({"role": "system", "content": guide_text})
                        logger.debug("[AnonymousContext] Restored Context Guide system message")
    
    @property
    def anonymous_mode(self) -> bool:
        """Check if currently in anonymous mode."""
        return self._anonymous_mode
