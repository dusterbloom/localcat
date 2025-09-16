"""
Smart token counter that uses tiktoken for accurate token counting.
"""
import os
import logging
from typing import List, Dict, Optional
from functools import lru_cache

from .exceptions import TokenCountingError, ConfigurationError

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to estimation")


class TokenCounter:
    """Smart token counter that uses tiktoken for OpenAI models or falls back to estimation"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize token counter for a specific model.

        Args:
            model_name: The model name to use for token counting. If None, tries to detect from environment.
        """
        self.model_name = model_name or self._detect_model_name()
        self._encoder = None
        self._init_encoder()

    def _detect_model_name(self) -> str:
        """Auto-detect model name from common environment variables"""
        # Common environment variable names used in different setups
        env_vars = [
            "OPENAI_MODEL",
            "LLM_MODEL",
            "MODEL_NAME",
            "AI_MODEL"
        ]

        for var in env_vars:
            model = os.getenv(var)
            if model:
                logger.debug(f"Detected model from {var}: {model}")
                return model

        # Default fallback - works for most OpenAI-compatible APIs
        default_model = "gpt-3.5-turbo"
        logger.debug(f"No model detected in environment, using default: {default_model}")
        return default_model

    @lru_cache(maxsize=1)
    def _init_encoder(self):
        """Initialize the appropriate tokenizer (cached)"""
        if not TIKTOKEN_AVAILABLE:
            logger.debug("tiktoken not available, using estimation fallback")
            return

        try:
            # Try to get encoding for the specific model
            if any(model_prefix in self.model_name.lower() for model_prefix in ["gpt-4", "gpt-3.5", "gpt-35"]):
                self._encoder = tiktoken.encoding_for_model(self.model_name)
                logger.debug(f"Initialized tiktoken encoder for model: {self.model_name}")
            else:
                # For local models or non-standard names, use cl100k_base (GPT-4 encoding)
                # This is a reasonable default that works well for most modern models
                self._encoder = tiktoken.get_encoding("cl100k_base")
                logger.debug(f"Using cl100k_base encoding for model: {self.model_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoder: {e}, falling back to estimation")
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string accurately.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens in the text

        Raises:
            TokenCountingError: If token counting fails critically
        """
        if text is None:
            raise TokenCountingError("Text cannot be None")

        if not text:
            return 0

        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                raise TokenCountingError(f"Cannot convert input to string: {e}")

        # Try tiktoken first (most accurate)
        if self._encoder:
            try:
                tokens = self._encoder.encode(text)
                return len(tokens)
            except Exception as e:
                logger.debug(f"tiktoken encoding failed: {e}, using estimation")
                # Don't raise here, fall back to estimation

        try:
            # Fallback to character-based estimation
            # This is the same heuristic used in the original code
            estimated_tokens = max(1, (len(text) + 3) // 4)
            return estimated_tokens
        except Exception as e:
            raise TokenCountingError(f"Token estimation failed: {e}")

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of chat messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Total token count including message structure overhead

        Raises:
            TokenCountingError: If messages format is invalid or counting fails
        """
        if messages is None:
            raise TokenCountingError("Messages cannot be None")

        if not messages:
            return 0

        if not isinstance(messages, list):
            raise TokenCountingError("Messages must be a list")

        total = 0

        try:
            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    logger.debug(f"Skipping non-dict message at index {i}: {type(message)}")
                    continue

                # Count tokens in message content
                content = message.get("content", "")
                if content is None:
                    content = ""

                content_tokens = self.count_tokens(str(content))

                # Add overhead for message structure
                # In OpenAI's format, each message has ~4 tokens overhead for role/structure
                # This is approximate but reasonable for planning purposes
                message_overhead = 4

                total += content_tokens + message_overhead

            # Add a small overhead for the entire messages array structure
            array_overhead = 3

            return total + array_overhead

        except Exception as e:
            raise TokenCountingError(f"Failed to count tokens in messages: {e}")

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current token counting setup"""
        return {
            "model_name": self.model_name,
            "encoder_type": "tiktoken" if self._encoder else "estimation",
            "tiktoken_available": str(TIKTOKEN_AVAILABLE),
            "encoding_name": getattr(self._encoder, 'name', 'none') if self._encoder else 'none'
        }


# Global instance for convenience (lazy-loaded)
_global_counter = None

def get_global_counter() -> TokenCounter:
    """Get a global TokenCounter instance (singleton pattern)"""
    global _global_counter
    if _global_counter is None:
        _global_counter = TokenCounter()
    return _global_counter


# Convenience functions that use the global counter
def count_tokens(text: str) -> int:
    """Count tokens in text using the global counter"""
    return get_global_counter().count_tokens(text)


def count_messages(messages: List[Dict[str, str]]) -> int:
    """Count tokens in messages using the global counter"""
    return get_global_counter().count_messages(messages)