"""
FunctionFilter to block tool call TextFrames from reaching TranscriptProcessor.

This prevents tool call JSON and XML from appearing in the UI transcript.
Only natural language responses should appear in transcripts.
"""

import re
from loguru import logger

from pipecat.frames.frames import Frame, TextFrame, LLMTextFrame
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection


async def filter_tool_call_text_frames(frame: Frame) -> bool:
    """
    Filter function to block TextFrames containing tool call syntax.

    Blocks frames containing:
    - XML tool call syntax: <function=...>...</function>
    - JSON tool call syntax: { "information": "..." }
    - Think tags: <think>...</think>
    - IM tokens: <|im_start|>...<|im_end|>

    Args:
        frame: Frame to evaluate

    Returns:
        True to allow frame through, False to block it
    """
    # Log every frame that comes through for debugging
    logger.debug(f"[ToolCallFilter] Processing frame type: {type(frame).__name__}")

    # Only filter TextFrame and LLMTextFrame types
    if not isinstance(frame, (TextFrame, LLMTextFrame)):
        logger.debug(f"[ToolCallFilter] Allowing non-text frame: {type(frame).__name__}")
        return True  # Allow all non-text frames through

    text = frame.text
    logger.debug(f"[ToolCallFilter] Checking text: '{text[:100]}...' (len={len(text)})")

    # Tool call patterns to detect and block
    tool_call_patterns = [
        r'<function=\w+>',  # Opening function tag
        r'<function\s*=',    # Opening function tag with whitespace
        r'</function>',      # Closing function tag
        r'<think>',          # Opening think tag
        r'</think>',         # Closing think tag
        r'<\|im_start\|>',   # IM start token
        r'<\|im_end\|>',     # IM end token
        # JSON patterns for tool calls (with flexible whitespace)
        r'\{\s*"information"\s*:',  # Memory add pattern
        r'\{\s*"query"\s*:',         # Memory search/edit/delete pattern
        r'\{\s*"new_information"\s*:', # Memory edit pattern
        r'"query"\s*:\s*"',          # JSON field
        r'"information"\s*:\s*"',    # JSON field
    ]

    # Check if text is ENTIRELY a JSON object (likely tool call argument)
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        # Check if it looks like a JSON object with quotes
        if '"' in stripped:
            logger.debug(f"[ToolCallFilter] ✅ BLOCKING JSON object TextFrame: '{text[:50]}'")
            return False

    # Check if text contains any tool call pattern
    for i, pattern in enumerate(tool_call_patterns):
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            logger.debug(f"[ToolCallFilter] ✅ BLOCKING tool call TextFrame (pattern {i}): '{text[:50]}'")
            return False  # Block this frame

    # Allow frame through
    logger.debug(f"[ToolCallFilter] ✓ Allowing text through: '{text[:50]}'")
    return True


def create_tool_call_filter() -> FunctionFilter:
    """
    Create a FunctionFilter configured to block tool call TextFrames.

    Returns:
        FunctionFilter instance ready to be added to pipeline
    """
    return FunctionFilter(
        filter=filter_tool_call_text_frames,
        direction=FrameDirection.DOWNSTREAM  # Filter frames going to transcript
    )
