"""
Enhanced Direct MLX-LM Service with Tool Calling Support.

This extends DirectMLXLLMService to add function calling capabilities
for Qwen3 and other tool-compatible models.
"""

import asyncio
import json
import re
import threading
import time
from typing import AsyncGenerator, List, Dict, Any, Optional

from loguru import logger

# Import global MLX lock for Metal operation coordination
from core.utils.mlx_lock import MLX_GLOBAL_LOCK

try:
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
except ImportError:
    mlx_lm = None
    logger.error("mlx_lm not available - install with: pip install mlx-lm")

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    LLMTextFrame,
    ErrorFrame,
)
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import (
    OpenAIContextAggregatorPair,
    OpenAIUserContextAggregator,
    OpenAIAssistantContextAggregator,
)
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)

from .direct_mlx_llm import DirectMLXLLMService


class DirectMLXLLMServiceWithTools(DirectMLXLLMService):
    """
    Enhanced Direct MLX-LM service with tool calling support.

    Extends DirectMLXLLMService to add:
    - Function calling capabilities for Qwen3 models
    - Tool detection and formatting
    - Function call parsing and execution
    - Streaming tool call results

    Usage:
        # In .env:
        LLM_USE_DIRECT_MLX=true
        LLM_MODEL=mlx-community/Qwen3-1.7B-8bit

        # In service factory:
        llm = DirectMLXLLMServiceWithTools(
            model="mlx-community/Qwen3-1.7B-8bit"
        )
    """

    def __init__(self, *args, **kwargs):
        """Initialize enhanced Direct MLX-LM service with tool support."""
        super().__init__(*args, **kwargs)

        # Check if model supports tool calling
        self._supports_tools = self._check_tool_support()
        logger.info(f"🔧 Tool support: {'✅ ENABLED' if self._supports_tools else '❌ DISABLED'}")

        # Tool call state tracking
        self._current_tool_call = None
        self._tool_call_buffer = ""
        self._in_tool_call_block = False  # Track if we're currently inside a tool call
        self._tool_syntax_buffer = ""  # Accumulate tokens to detect multi-token patterns
        self._angle_bracket_depth = 0  # Track nesting depth of angle brackets

        # TextFrame tracking for duplication detection
        self._llm_frame_counter = 0
        self._emitted_text_frames = {}  # text -> (frame_id, timestamp, count)

    def _check_tool_support(self) -> bool:
        """
        Check if the loaded model supports tool calling.

        Returns:
            True if model appears to support tool calling
        """
        try:
            # Check tokenizer config for tool calling indicators
            if hasattr(self._tokenizer, 'chat_template'):
                template = str(self._tokenizer.chat_template)
                tool_indicators = ['tool_calls', 'function_call', 'tools', '<tool']
                return any(indicator in template.lower() for indicator in tool_indicators)
        except Exception as e:
            logger.warning(f"Could not check tool support: {e}")

        # Check model name for known tool-capable models
        tool_models = ['qwen3', 'qwen2.5', 'llama-3.1', 'llama-3.2']
        model_lower = self._model_name.lower()
        return any(model in model_lower for model in tool_models)

    async def _process_context(
        self, context: OpenAILLMContext | LLMContext
    ) -> AsyncGenerator[Frame, None]:
        """
        Process LLM context with tool calling support.

        Enhanced version that detects tools, formats messages appropriately,
        and handles both text and function call responses.
        """
        logger.debug(f"🎯 _process_context called (Direct MLX-LM with Tools)")

        # Emit start frame to signal beginning of LLM response
        yield LLMFullResponseStartFrame()

        try:
            # Get messages and tools from context
            if isinstance(context, OpenAILLMContext):
                messages = context.get_messages()
                tools = getattr(context, 'tools', None)
            else:
                # Universal LLMContext - use adapter to format messages
                adapter = self.get_llm_adapter()
                messages = adapter.get_messages(context)
                tools = getattr(context, 'tools', None)

            logger.debug(f"📝 Processing {len(messages)} messages with {len(tools) if tools else 0} tools")

            # Log complete context for debugging
            logger.info("📄 COMPLETE CONTEXT SENT TO LLM:")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")

                # Handle different content types
                if isinstance(content, list):
                    content_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_content = item.get("text", "")[:200]
                            content_parts.append(f"text: {text_content}{'...' if len(item.get('text', '')) > 200 else ''}")
                        elif item.get("type") == "image_url":
                            content_parts.append("image: [base64 image data]")
                    content_str = " | ".join(content_parts)
                elif isinstance(content, str):
                    content_str = content[:200] + "..." if len(content) > 200 else content
                else:
                    content_str = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)

                # Log tool calls if present
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    content_str += f" [🔧 {len(tool_calls)} tool call(s)]"

                logger.info(f"  {i+1}/{len(messages)} [{role}]: {content_str}")

            # Format messages with tool support
            formatted_messages = self._format_messages_with_tools(messages, tools)

            # Apply chat template
            try:
                prompt = self._tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {e}")
                prompt = self._format_prompt_fallback(formatted_messages)

            logger.debug(f"🧠 LLM generating (Direct MLX with Tools, model={self._model_name})")

            # Start timing for TTFT measurement
            start_time = time.time()
            first_token_time = None

            # Bridge synchronous generator to async using queue
            token_queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _generate_to_queue():
                """
                Run synchronous mlx_lm.stream_generate() in thread.
                Enhanced to detect and handle tool calls in the response.
                """
                try:
                    # Create sampler with temperature
                    sampler = make_sampler(temp=self._settings["temperature"])

                    # CRITICAL: Use global MLX lock to prevent concurrent access with STT/TTS
                    with MLX_GLOBAL_LOCK:
                        response_text = ""

                        for chunk in mlx_lm.stream_generate(
                            self._model,
                            self._tokenizer,
                            prompt=prompt,
                            max_tokens=self._settings["max_tokens"],
                            sampler=sampler
                        ):
                            if chunk.text:
                                # Always accumulate for tool call detection
                                response_text += chunk.text

                                # Check for tool call patterns in accumulated text
                                if self._supports_tools and self._detect_tool_call_start(response_text):
                                    # Mark that we detected a tool call, but continue generation
                                    if not self._current_tool_call:
                                        self._handle_tool_call_detection(response_text)

                                # Only push non-tool-call tokens to prevent transcript/TTS pollution
                                # Tool call syntax is silently accumulated but not streamed to UI
                                if not self._is_tool_call_syntax(chunk.text):
                                    loop.call_soon_threadsafe(token_queue.put_nowait, chunk.text)
                                else:
                                    logger.debug(f"[DirectMLX] Skipping tool call syntax token: '{chunk.text[:20]}'")

                        # After generation complete, process tool calls using Pipecat's built-in pattern
                        if self._supports_tools and self._current_tool_call:
                            tool_calls_list = self._extract_tool_calls(response_text)

                            if tool_calls_list:
                                # Send tool call marker to queue (will be processed in main loop)
                                loop.call_soon_threadsafe(
                                    token_queue.put_nowait,
                                    ("TOOL_CALLS", tool_calls_list)
                                )

                except Exception as e:
                    logger.error(f"MLX generation error: {e}")
                    loop.call_soon_threadsafe(token_queue.put_nowait, ("ERROR", str(e)))
                finally:
                    # Signal completion with None sentinel
                    loop.call_soon_threadsafe(token_queue.put_nowait, None)

            # Start generation in background thread
            loop.run_in_executor(None, _generate_to_queue)

            # Stream tokens as they arrive from queue
            while True:
                token = await token_queue.get()

                # None signals completion
                if token is None:
                    break

                # Check for error tuple
                if isinstance(token, tuple) and token[0] == "ERROR":
                    raise Exception(token[1])

                # Check for tool calls tuple
                if isinstance(token, tuple) and token[0] == "TOOL_CALLS":
                    tool_calls_list = token[1]
                    logger.info(f"🔧 Processing {len(tool_calls_list)} tool calls using Pipecat pattern")

                    # Create FunctionCallFromLLM objects
                    function_calls = []
                    for tc in tool_calls_list:
                        function_call = FunctionCallFromLLM(
                            context=context,
                            tool_call_id=tc["id"],
                            function_name=tc["function"]["name"],
                            arguments=json.loads(tc["function"]["arguments"])
                        )
                        function_calls.append(function_call)

                    # Use Pipecat's built-in function execution
                    await self.run_function_calls(function_calls)
                    logger.debug(f"✅ Tool calls executed, results will flow back to LLM")
                    continue  # Don't yield this as a text frame

                # Measure TTFT on first token
                if first_token_time is None:
                    first_token_time = (time.time() - start_time) * 1000
                    logger.debug(f"⚡ TTFT: {first_token_time:.1f}ms (Direct MLX with Tools)")

                # Allow natural streaming behavior as Pipecat expects
                # Track LLM TextFrame emissions for duplication detection (only within same response)
                self._llm_frame_counter += 1
                frame_id = f"LLM_TF{self._llm_frame_counter}_{int(time.time() * 1000)}"

                # Check for duplicate text emission within same response
                if token in self._emitted_text_frames:
                    prev_id, prev_timestamp, count = self._emitted_text_frames[token]
                    time_diff = time.time() - prev_timestamp

                    # Only log if duplicate within same response (< 1s indicates true duplicate)
                    if time_diff < 1.0:
                        logger.warning(f"Duplicate token in same response: '{token[:20]}' ({time_diff:.1f}s ago)")

                    # Update counter
                    self._emitted_text_frames[token] = (frame_id, time.time(), count + 1)
                else:
                    # First time emitting this token
                    self._emitted_text_frames[token] = (frame_id, time.time(), 1)

                yield LLMTextFrame(text=token)

            # Log completion
            total_time = (time.time() - start_time) * 1000
            logger.debug(f"✅ LLM complete in {total_time:.1f}ms (Direct MLX with Tools)")

        except asyncio.CancelledError:
            logger.debug("🚫 LLM generation cancelled (user interruption)")
            raise

        except Exception as e:
            logger.error(f"❌ Direct MLX-LM with Tools error: {e}", exc_info=True)

            # Yield error frame (non-fatal by default)
            yield ErrorFrame(
                error=f"LLM error: {e}",
                fatal=False
            )
        finally:
            # Always emit end frame to signal completion of LLM response
            yield LLMFullResponseEndFrame()

            # Clear state for next response
            self._emitted_text_frames.clear()
            self._in_tool_call_block = False
            self._current_tool_call = None
            self._tool_syntax_buffer = ""
            self._angle_bracket_depth = 0
            logger.debug("[DirectMLX] Cleared state for next response")

    def _format_messages_with_tools(self, messages: List[Dict], tools: Optional[List] = None) -> List[Dict]:
        """
        Format messages for tool-capable models.

        Args:
            messages: Original messages from context
            tools: Tool definitions from context

        Returns:
            Formatted messages with tool context
        """
        formatted_messages = []

        # Add tools system message if tools are available
        if tools and self._supports_tools:
            tools_message = {
                "role": "system",
                "content": self._format_tools_for_system(tools)
            }
            formatted_messages.append(tools_message)

        # Add existing messages
        for msg in messages:
            formatted_msg = msg.copy()

            # Convert tool calls to the format expected by the model
            if "tool_calls" in msg:
                if self._supports_tools:
                    # Format tool calls for Qwen3-style models
                    formatted_msg["content"] = self._format_tool_calls_for_model(msg["tool_calls"])
                else:
                    # Remove tool calls if model doesn't support them
                    formatted_msg["content"] = "[Tool calls not supported by this model]"
                    del formatted_msg["tool_calls"]

            formatted_messages.append(formatted_msg)

        return formatted_messages

    def _format_tools_for_system(self, tools: List[Dict]) -> str:
        """
        Format tool definitions for system message.

        Args:
            tools: List of tool definitions

        Returns:
            Formatted tool descriptions
        """
        tool_descriptions = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                name = func["name"]
                desc = func["description"]
                params = func.get("parameters", {})

                tool_desc = f"- {name}: {desc}"
                if params.get("properties"):
                    required_params = params.get("required", [])
                    param_descs = []

                    for param_name, param_info in params["properties"].items():
                        param_type = param_info.get("type", "string")
                        param_desc = param_info.get("description", "")
                        req_marker = " (required)" if param_name in required_params else " (optional)"
                        param_descs.append(f"  {param_name} ({param_type}){req_marker}: {param_desc}")

                    if param_descs:
                        tool_desc += "\n  Parameters:\n" + "\n".join(param_descs)

                tool_descriptions.append(tool_desc)

        return f"""You have access to the following tools:

{chr(10).join(tool_descriptions)}

To use a tool, respond with a function call in this format:
<|im_start|>assistant
<function=tool_name>
{{
  "parameter_name": "parameter_value"
}}
</function><|im_end|>"""

    def _format_tool_calls_for_model(self, tool_calls: List[Dict]) -> str:
        """
        Format tool calls for model input.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Formatted tool call string
        """
        if not tool_calls:
            return ""

        formatted_calls = []
        for call in tool_calls:
            if call.get("type") == "function":
                func = call["function"]
                name = func["name"]
                args = func.get("arguments", "{}")

                # Try to parse arguments for better formatting
                try:
                    args_dict = json.loads(args) if isinstance(args, str) else args
                    formatted_args = json.dumps(args_dict, indent=2)
                except:
                    formatted_args = args

                formatted_call = f"<function={name}>\n{formatted_args}\n</function>"
                formatted_calls.append(formatted_call)

        return "\n".join(formatted_calls)

    def _detect_tool_call_start(self, text: str) -> bool:
        """
        Detect if the response text contains the start of a tool call.

        Args:
            text: Accumulated response text

        Returns:
            True if tool call pattern detected
        """
        # Common tool call patterns for Qwen3 and similar models
        tool_patterns = [
            r'<function=\w+>',
            r'<\|im_start\|>assistant\s*\n<function',
            r'```tool',
            r'',
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in tool_patterns)

    def _handle_tool_call_detection(self, text: str):
        """
        Handle the detection of a tool call in the response.

        Args:
            text: Response text containing tool call
        """
        logger.info("🔧 Tool call detected in LLM response")
        self._current_tool_call = text
        self._tool_call_buffer = text

        # Set flag to start blocking tokens (tool call content should not be spoken)
        if not self._in_tool_call_block:
            self._in_tool_call_block = True
            self._tool_syntax_buffer = text  # Start tracking for closing tag
            self._angle_bracket_depth = text.count('<') - text.count('>')
            logger.debug(f"[DirectMLX] Entered tool call block via detection (depth={self._angle_bracket_depth})")

    def _process_complete_tool_call(self, response_text: str, token_queue: asyncio.Queue, loop):
        """
        Process a complete tool call from the response.

        Args:
            response_text: Full response text containing tool call
            token_queue: Queue for sending tokens back to main thread
            loop: Event loop for thread-safe operations
        """
        try:
            # Extract tool call information
            tool_calls = self._extract_tool_calls(response_text)

            if tool_calls:
                logger.info(f"🔧 Extracted {len(tool_calls)} tool call(s)")

                # Send tool call frames to pipeline
                for tool_call in tool_calls:
                    loop.call_soon_threadsafe(
                        token_queue.put_nowait,
                        ("TOOL_CALL", tool_call)
                    )
            else:
                # No valid tool calls found, treat as regular text
                loop.call_soon_threadsafe(token_queue.put_nowait, response_text)

        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            # Fall back to regular text
            loop.call_soon_threadsafe(token_queue.put_nowait, response_text)

    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """
        Extract tool calls from response text.

        Args:
            text: Response text potentially containing tool calls

        Returns:
            List of extracted tool call dictionaries
        """
        tool_calls = []

        # Pattern for Qwen3-style function calls
        pattern = r'<function=(\w+)>\s*\n?(.*?)\n?</function>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for func_name, args_text in matches:
            try:
                # Parse arguments
                args_text = args_text.strip()
                if args_text:
                    try:
                        args_dict = json.loads(args_text)
                    except json.JSONDecodeError:
                        # Try to extract JSON from mixed content
                        json_match = re.search(r'\{.*\}', args_text, re.DOTALL)
                        if json_match:
                            args_dict = json.loads(json_match.group())
                        else:
                            args_dict = {"raw_input": args_text}
                else:
                    args_dict = {}

                tool_call = {
                    "id": f"call_{int(time.time() * 1000)}_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args_dict)
                    }
                }

                tool_calls.append(tool_call)

            except Exception as e:
                logger.warning(f"Failed to parse tool call {func_name}: {e}")

        return tool_calls

    def _is_tool_call_syntax(self, text: str) -> bool:
        """Check if text is part of tool call XML/JSON syntax that should not be spoken.

        Simplified approach: Only block tokens when we're inside a CONFIRMED tool call block.
        Detection happens in _detect_tool_call_start() using accumulated response text.

        Args:
            text: Token text to check

        Returns:
            True if text is part of tool call syntax
        """
        # If we're inside a confirmed tool call block, block all tokens
        if self._in_tool_call_block:
            # Accumulate in buffer to detect closing tag
            self._tool_syntax_buffer += text

            # Track angle bracket depth for nested tags
            if '<' in text:
                self._angle_bracket_depth += text.count('<')
            if '>' in text:
                self._angle_bracket_depth -= text.count('>')

                # Check if we're closing the tool call block
                if re.search(r'</\s*(function|think)>|<\|im_end\|>', self._tool_syntax_buffer, re.IGNORECASE):
                    if self._angle_bracket_depth <= 0:
                        self._in_tool_call_block = False
                        self._angle_bracket_depth = 0
                        self._tool_syntax_buffer = ""
                        logger.debug(f"[DirectMLX] Exited tool call block")

            return True  # Block all tokens inside tool call block

        # Not in a tool call block - allow token through
        # Tool call detection happens separately in _detect_tool_call_start()
        return False
