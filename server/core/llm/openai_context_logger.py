"""
OpenAI Context Logger Service - Complete context visibility wrapper

This service wraps Pipecat's OpenAILLMService to provide complete context logging
that matches the format and detail level of DirectMLXLLMService.

Ensures 100% context logging coverage regardless of which LLM backend is selected.
"""

from loguru import logger
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.frames.frames import Frame


class OpenAIContextLoggerService(OpenAILLMService):
    """
    OpenAI service wrapper with complete context logging.

    Extends OpenAILLMService to add comprehensive context logging
    that matches DirectMLXLLMService format for consistency.
    """

    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        """
        Process LLM context with complete logging before delegating to base service.

        This method logs the complete context that will be sent to the OpenAI API,
        ensuring full visibility regardless of which LLM service is selected.
        """
        try:
            # Get messages from context using the appropriate method
            if isinstance(context, OpenAILLMContext):
                messages = context.get_messages_for_logging()
            else:
                # Universal LLMContext - use adapter to format messages
                adapter = self.get_llm_adapter()
                messages = adapter.get_messages_for_logging(context)

            logger.info("📄 COMPLETE CONTEXT SENT TO LLM:")

            # Log each message with role and content (matching DirectMLX format)
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")

                # Handle different content types (vision, text, etc.)
                if isinstance(content, list):
                    # Structured content (vision + text)
                    content_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_content = item.get("text", "")[:200]
                            content_parts.append(f"text: {text_content}{'...' if len(item.get('text', '')) > 200 else ''}")
                        elif item.get("type") == "image_url":
                            content_parts.append("image: [base64 image data]")
                    content_str = " | ".join(content_parts)
                elif isinstance(content, str):
                    # Simple text content
                    content_str = content[:200] + "..." if len(content) > 200 else content
                else:
                    content_str = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)

                logger.info(f"  {i+1}/{len(messages)} [{role}]: {content_str}")

            # Log context summary
            logger.info(f"📝 OpenAI LLM processing {len(messages)} messages")

        except Exception as e:
            logger.warning(f"OpenAI context logging failed: {e}")

        # Delegate to base OpenAILLMService for actual LLM processing
        async for frame in super()._process_context(context):
            yield frame