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
    - Disables memory injection via ephemeral mode
    - Rebuilds system prompt without memory section
    - Sets anonymous mode flag for downstream processors
    """

    def __init__(
        self,
        context_aggregator: Any,
        context: OpenAILLMContext,
        memory_processor: Optional[Any] = None,
        factory: Optional[Any] = None,
        vision_injector: Optional[Any] = None
    ):
        """
        Initialize with context aggregator, memory processor, and factory.

        Args:
            context_aggregator: Pipecat context aggregator
            context: OpenAI LLM context
            memory_processor: Optional memory processor for controlling injection
            factory: Optional factory for rebuilding system prompt
            vision_injector: Optional vision context injector for camera state tracking
        """
        self._aggregator = context_aggregator
        self._context = context
        self._memory_processor = memory_processor
        self._factory = factory
        self._vision_injector = vision_injector
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
        # logger.debug("[AnonymousContext] Providing user context aggregator")
        return ContextAggregatorWrapper(self._aggregator.user(), "user", self._anonymous_mode)

    def assistant(self) -> Any:
        """Get assistant aggregator (delegated to wrapped aggregator)."""
        # logger.debug("[AnonymousContext] Providing assistant context aggregator")
        return ContextAggregatorWrapper(self._aggregator.assistant(), "assistant", self._anonymous_mode)
    
    @property
    def context(self) -> OpenAILLMContext:
        """Get the underlying context."""
        return self._context
    
    def set_anonymous_mode(self, enabled: bool) -> None:
        """
        Enable or disable anonymous mode.

        When enabled:
        - Disables memory injection via ephemeral mode
        - Rebuilds system prompt without memory section
        - Clears all conversation history
        - Removes Context Guide system message if present
        - Adds anonymous mode marker
        """
        if self._anonymous_mode == enabled:
            return  # No change needed

        self._anonymous_mode = bool(enabled)

        if enabled:
            logger.info("[AnonymousContext] Entering anonymous mode - clearing context and disabling memory")

            # 1. Disable memory injection by setting ephemeral mode
            if self._memory_processor:
                try:
                    self._memory_processor.set_ephemeral_mode(True)
                    logger.info("[AnonymousContext] Disabled memory injection for anonymous mode")
                except Exception as e:
                    logger.warning(f"[AnonymousContext] Failed to set ephemeral mode: {e}")

            # 2. Rebuild system prompt WITHOUT memory section
            if self._factory:
                try:
                    camera_active = self._get_camera_state()
                    new_prompt = self._factory.build_system_prompt(skip_memory=True, camera_active=camera_active)
                    self._update_system_prompt(new_prompt)
                    logger.debug(f"[AnonymousContext] Updated system prompt for anonymous mode (memory excluded, camera_active={camera_active})")
                except Exception as e:
                    logger.warning(f"[AnonymousContext] Failed to update system prompt: {e}")
            
            # Store current messages (except system messages)
            self._original_messages = self._context.get_messages().copy()
            
            # # Clear context to only essential system messages
            # messages = self._context.get_messages()
            # filtered_messages = []
            
            # for msg in messages:
            #     content = msg.get("content", "")
            #     # Keep only essential system messages; drop Context Guide and Session Context
            #     if msg.get("role") == "system":
            #         if ("Context Guide:" in content) or (isinstance(content, str) and content.startswith("[Session Context]")):
            #             logger.debug("[AnonymousContext] Removing non-essential system message for anonymous mode")
            #         else:
            #             filtered_messages.append(msg)
            #     # Skip all user/assistant messages (conversation history)
            #     elif msg.get("role") in ["user", "assistant"]:
            #         logger.debug(f"[AnonymousContext] Removing {msg.get('role')} message: {content[:50]}...")
            #         continue
            #     else:
            #         filtered_messages.append(msg)
            
            # Reset context with filtered messages
            self._context._messages = self._original_messages
            
            # Add anonymous marker as system message
            self._context.add_message({
                "role": "system",
                "content": "Anonymous session: No conversation history is stored."
            })

        else:
            logger.info("[AnonymousContext] Exiting anonymous mode - restoring context and memory")

            # 1. Re-enable memory injection
            if self._memory_processor:
                try:
                    self._memory_processor.set_ephemeral_mode(False)
                    logger.info("[AnonymousContext] Re-enabled memory injection")
                except Exception as e:
                    logger.warning(f"[AnonymousContext] Failed to restore ephemeral mode: {e}")

            # 2. Restore original messages (conversation history) if available
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

            # 3. Rebuild system prompt WITH memory section (AFTER restoring messages)
            if self._factory:
                try:
                    camera_active = self._get_camera_state()
                    new_prompt = self._factory.build_system_prompt(skip_memory=False, camera_active=camera_active)
                    self._update_system_prompt(new_prompt)
                    logger.debug(f"[AnonymousContext] Restored system prompt with memory section (camera_active={camera_active})")
                except Exception as e:
                    logger.warning(f"[AnonymousContext] Failed to restore system prompt: {e}")
    
    @property
    def anonymous_mode(self) -> bool:
        """Check if currently in anonymous mode."""
        return self._anonymous_mode

    def set_vision_injector(self, vision_injector: Any) -> None:
        """Link vision injector for camera state tracking (called after pipeline creation)."""
        self._vision_injector = vision_injector
        logger.debug("[AnonymousContext] Vision injector linked for camera state tracking")

    def _get_camera_state(self) -> bool:
        """Get current camera state from vision injector."""
        if self._vision_injector and hasattr(self._vision_injector, 'camera_active'):
            return self._vision_injector.camera_active
        return False

    def _update_system_prompt(self, new_prompt: str) -> None:
        """
        Replace the persona system prompt message with a new one.

        Finds the first system message that looks like the main persona prompt
        and replaces it. The persona prompt typically contains "Locat" and
        instructions about the assistant's behavior.

        Args:
            new_prompt: New system prompt content
        """
        messages = list(self._context.get_messages())

        # Find and replace the persona prompt
        # The persona prompt is the one that contains "Locat" and is the main instruction set
        for i, msg in enumerate(messages):
            if msg.get('role') == 'system':
                content = msg.get('content', '')
                if isinstance(content, str):
                    # Skip special injected messages (not the main persona)
                    if ('Anonymous session:' in content or
                        '[Session Context]' in content or
                        content.startswith('Memory context to be used') or
                        'Context Guide:' in content):
                        continue
                    # Check if this looks like the main persona prompt
                    if 'Locat' in content and 'voice assistant' in content:
                        # This is the persona prompt - replace it
                        messages[i] = {'role': 'system', 'content': new_prompt}
                        logger.debug("[AnonymousContext] Updated system prompt")
                        break

        self._context.set_messages(messages)


class ContextAggregatorWrapper:
    """
    Wrapper for context aggregator processors that adds logging for frame processing.

    This wraps the actual context aggregator processors returned by Pipecat
    to provide visibility into when TranscriptionFrames and TTSTextFrames
    are processed and added to the conversation context.
    """

    def __init__(self, aggregator_processor: Any, aggregator_type: str, anonymous_mode: bool = False):
        """
        Initialize the wrapper.

        Args:
            aggregator_processor: The actual context aggregator processor from Pipecat
            aggregator_type: Either "user" or "assistant" for logging identification
            anonymous_mode: Whether we're currently in anonymous mode
        """
        self._processor = aggregator_processor
        self._type = aggregator_type
        self._anonymous_mode = anonymous_mode

    async def process_frame(self, frame, direction):
        """
        Process frames with logging.

        This method is called by Pipecat pipeline when frames flow through
        the context aggregator. We log what type of frame is being processed
        and delegate to the actual processor.
        """
        from pipecat.frames.frames import TranscriptionFrame, TTSTextFrame

        # Log frame processing with details
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"[ContextAggregator-{self._type}] Processing TranscriptionFrame: '{frame.text[:100]}{'...' if len(frame.text) > 100 else ''}' (anonymous: {self._anonymous_mode})")
        elif isinstance(frame, TTSTextFrame):
            logger.info(f"[ContextAggregator-{self._type}] Processing TTSTextFrame: '{frame.text[:100]}{'...' if len(frame.text) > 100 else ''}' (anonymous: {self._anonymous_mode})")
        else:
            logger.debug(f"[ContextAggregator-{self._type}] Processing frame type: {type(frame).__name__}")

        # Delegate to actual processor
        try:
            result = await self._processor.process_frame(frame, direction)

            # Log successful context update
            if isinstance(frame, (TranscriptionFrame, TTSTextFrame)):
                logger.info(f"[ContextAggregator-{self._type}] Successfully added {type(frame).__name__} to context")

            return result
        except Exception as e:
            logger.error(f"[ContextAggregator-{self._type}] Error processing {type(frame).__name__}: {e}")
            raise

    def __getattr__(self, name):
        """Delegate any other attribute access to the wrapped processor."""
        return getattr(self._processor, name)
