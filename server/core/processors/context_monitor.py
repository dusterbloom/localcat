"""
Context Monitor - Pipecat-native context visibility

This processor monitors OpenAILLMContextFrame objects to provide visibility
into when and how messages are added to the conversation context.
Works with Pipecat's architecture instead of fighting it.
"""

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame


class ContextMonitor(FrameProcessor):
    """
    Pipecat-native processor that monitors context changes.

    This processor sits in the pipeline and logs when OpenAILLMContextFrame
    objects pass through, providing visibility into context updates
    without interfering with normal operation.
    """

    def __init__(self, name: str = "ContextMonitor"):
        """Initialize the context monitor."""
        super().__init__()
        self._name = name
        self._last_message_count = 0
        logger.info(f"[{name}] Context monitor initialized")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process frames and log context changes.

        Args:
            frame: The frame to process
            direction: Frame direction in pipeline
        """
        # Monitor OpenAILLMContextFrame for context changes
        if isinstance(frame, OpenAILLMContextFrame):
            await self._handle_context_frame(frame)

        # Pass frame through unchanged
        await self.push_frame(frame, direction)

    async def _handle_context_frame(self, frame: OpenAILLMContextFrame):
        """Handle OpenAILLMContextFrame and log the complete context."""
        try:
            context = frame.context
            messages = context.get_messages()
            message_count = len(messages)

            # Log context summary
            logger.info(f"[{self._name}] ===== COMPLETE CONTEXT ({message_count} messages) =====")

            # Log message count change
            if message_count != self._last_message_count:
                if message_count > self._last_message_count:
                    logger.info(f"[{self._name}] Context grew: {message_count - self._last_message_count} new messages")
                else:
                    logger.info(f"[{self._name}] Context shrank: {self._last_message_count - message_count} messages removed")
                self._last_message_count = message_count

            # Log the ENTIRE context
            logger.info(f"[{self._name}] Full conversation context:")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")

                # Format the message for readability
                if role == "SYSTEM":
                    logger.info(f"[{self._name}] ┌─ SYSTEM: {content}")
                elif role == "USER":
                    logger.info(f"[{self._name}] ├─ USER {i+1}: {content}")
                elif role == "ASSISTANT":
                    logger.info(f"[{self._name}] ├─ ASSISTANT {i+1}: {content}")
                else:
                    logger.info(f"[{self._name}] ├─ {role}: {content}")

            logger.info(f"[{self._name}] ===== END CONTEXT ({message_count} messages) =====")

            # Log context window info if available
            if hasattr(context, 'max_tokens'):
                max_tokens = getattr(context, 'max_tokens', 'unknown')
                logger.info(f"[{self._name}] Context window: {max_tokens} max tokens")

            # Log any tool calls or special context
            for i, msg in enumerate(messages):
                if "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    logger.info(f"[{self._name}] Tool calls in message {i+1}: {len(tool_calls)} tool(s)")
                elif "name" in msg and "role" in msg and msg["role"] == "tool":
                    tool_name = msg.get("name", "unknown")
                    logger.info(f"[{self._name}] Tool response: {tool_name}")

        except Exception as e:
            logger.warning(f"[{self._name}] Error processing context frame: {e}")


def create_context_monitor_pipeline_stage(monitor_name: str = "ContextMonitor"):
    """
    Create a context monitor that can be added to pipeline stages.

    Returns:
        ContextMonitor instance ready for pipeline integration
    """
    return ContextMonitor(name=monitor_name)