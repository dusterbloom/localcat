"""Fast and accurate token estimation for context management."""

from typing import Optional, Dict, Any, List
from loguru import logger

# Try model-native tokenizer first (works with any HuggingFace model)
_tokenizer = None
_tokenizer_name: Optional[str] = None

# Fallback: tiktoken for OpenAI models
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TokenEstimator:
    """Token estimation with model-native tokenizer, tiktoken fallback, or heuristic."""

    _encoder = None
    _encoding_name = "cl100k_base"

    @classmethod
    def configure(cls, model_name: str) -> None:
        """
        Configure token estimator for a specific model.
        Tries AutoTokenizer first (works with any HF model including MLX),
        falls back to tiktoken, then heuristic.
        """
        global _tokenizer, _tokenizer_name
        if _tokenizer_name == model_name:
            return  # already configured

        # Try AutoTokenizer (works with mlx-lm models, llama, qwen, gemma, etc.)
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _tokenizer_name = model_name
            logger.info(f"[TokenEstimator] Using AutoTokenizer for model: {model_name}")
            return
        except Exception as e:
            logger.debug(f"[TokenEstimator] AutoTokenizer failed for {model_name}: {e}")

        # Try tiktoken encoding name match
        if TIKTOKEN_AVAILABLE:
            try:
                cls._encoder = tiktoken.encoding_for_model(model_name)
                _tokenizer_name = model_name
                logger.info(f"[TokenEstimator] Using tiktoken for model: {model_name}")
                return
            except Exception:
                pass

        logger.warning(f"[TokenEstimator] No tokenizer found for {model_name}, using heuristic")

    @classmethod
    def get_encoder(cls):
        """Lazy-load tiktoken encoder (fallback only)."""
        if _tokenizer is not None:
            return None  # prefer AutoTokenizer path

        if not TIKTOKEN_AVAILABLE:
            return None

        if cls._encoder is None:
            try:
                cls._encoder = tiktoken.get_encoding(cls._encoding_name)
            except Exception as e:
                logger.warning(f"[TokenEstimator] Failed to load tiktoken: {e}")
                return None

        return cls._encoder

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate tokens using best available tokenizer."""
        if not text:
            return 0

        # Path 1: Model-native tokenizer
        if _tokenizer is not None:
            try:
                return len(_tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass

        # Path 2: tiktoken
        encoder = TokenEstimator.get_encoder()
        if encoder:
            try:
                return len(encoder.encode(text, disallowed_special=()))
            except Exception:
                pass

        # Path 3: Heuristic (4 chars ≈ 1 token)
        return max(1, len(text) // 4)

    @staticmethod
    def estimate_message_tokens(message: Dict[str, Any]) -> int:
        """
        Estimate tokens in a message dict.
        Accounts for message structure overhead (role, formatting, etc.)
        """
        if not isinstance(message, dict):
            return 0

        content = message.get("content", "")
        role = message.get("role", "")

        if isinstance(content, str):
            content_tokens = TokenEstimator.estimate_tokens(content)
        elif isinstance(content, list):
            content_tokens = sum(
                TokenEstimator.estimate_tokens(item.get("text", ""))
                if isinstance(item, dict) and item.get("type") == "text"
                else 0
                for item in content
            )
        else:
            content_tokens = 0

        overhead = 4 + TokenEstimator.estimate_tokens(role)
        return content_tokens + overhead

    @staticmethod
    def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
        """Estimate total tokens for a list of messages."""
        return sum(TokenEstimator.estimate_message_tokens(msg) for msg in messages)

    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        """Get token estimator metrics for debugging."""
        return {
            "tokenizer_type": "auto" if _tokenizer else ("tiktoken" if TIKTOKEN_AVAILABLE else "heuristic"),
            "model": _tokenizer_name,
            "encoder_loaded": _tokenizer is not None or TokenEstimator._encoder is not None
        }
