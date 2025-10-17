"""Fast and accurate token estimation for context management."""

from typing import Optional, Dict, Any, List
from loguru import logger

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using simple heuristic")


class TokenEstimator:
    """Token estimation with tiktoken (accurate) or fallback heuristic."""

    _encoder = None
    _encoding_name = "cl100k_base"  # GPT-3.5/GPT-4 encoding, works for most models

    @classmethod
    def get_encoder(cls):
        """Lazy-load tiktoken encoder."""
        if not TIKTOKEN_AVAILABLE:
            return None

        if cls._encoder is None:
            try:
                cls._encoder = tiktoken.get_encoding(cls._encoding_name)
                logger.debug(f"[TokenEstimator] Loaded tiktoken encoder: {cls._encoding_name}")
            except Exception as e:
                logger.warning(f"[TokenEstimator] Failed to load tiktoken: {e}")
                return None

        return cls._encoder

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate tokens for text using tiktoken (accurate) or fallback heuristic.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        encoder = TokenEstimator.get_encoder()

        if encoder:
            # Accurate counting with tiktoken
            try:
                return len(encoder.encode(text, disallowed_special=()))
            except Exception as e:
                logger.debug(f"[TokenEstimator] tiktoken encoding failed: {e}, using fallback")

        # Fallback: Simple heuristic (4 chars ≈ 1 token)
        return max(1, len(text) // 4)

    @staticmethod
    def estimate_message_tokens(message: Dict[str, Any]) -> int:
        """
        Estimate tokens in a message dict.

        Accounts for message structure overhead (role, formatting, etc.)
        Based on OpenAI's token counting: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

        Args:
            message: Message dict with 'role' and 'content'

        Returns:
            Estimated token count including message overhead
        """
        if not isinstance(message, dict):
            return 0

        content = message.get("content", "")
        role = message.get("role", "")

        # Count tokens in content
        if isinstance(content, str):
            content_tokens = TokenEstimator.estimate_tokens(content)
        elif isinstance(content, list):
            # Handle multi-modal content (vision)
            content_tokens = sum(
                TokenEstimator.estimate_tokens(item.get("text", ""))
                if isinstance(item, dict) and item.get("type") == "text"
                else 0
                for item in content
            )
        else:
            content_tokens = 0

        # Add message overhead (role tokens + formatting)
        # Each message has ~4 tokens overhead for formatting
        overhead = 4 + TokenEstimator.estimate_tokens(role)

        return content_tokens + overhead

    @staticmethod
    def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
        """
        Estimate total tokens for a list of messages.

        Args:
            messages: List of message dicts

        Returns:
            Total token count
        """
        return sum(TokenEstimator.estimate_message_tokens(msg) for msg in messages)

    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        """Get token estimator metrics for debugging."""
        return {
            "tiktoken_available": TIKTOKEN_AVAILABLE,
            "encoding": TokenEstimator._encoding_name if TIKTOKEN_AVAILABLE else "fallback",
            "encoder_loaded": TokenEstimator._encoder is not None
        }
