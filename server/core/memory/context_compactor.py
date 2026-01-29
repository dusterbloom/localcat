"""
Context compactor — gradient-bang pattern for infinite context.

Monitors context size and compresses old messages using a dedicated small
MLX model (Qwen3-0.6B-4bit) when usage exceeds a configurable threshold.
Replaces old user/assistant messages with a summary while preserving system
messages and recent conversation.

Design:
- Runs as a FrameProcessor in the main pipeline
- Monitors OpenAILLMContextFrame after each assistant turn
- When token count > threshold, spawns background compaction
- Atomically replaces old messages with summary on completion
"""

import asyncio
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .token_estimator import TokenEstimator

# MLX compactor model config
COMPACTOR_MODEL = os.getenv("COMPACTOR_MODEL", "mlx-community/Qwen3-0.6B-4bit")
COMPACT_THRESHOLD = float(os.getenv("CONTEXT_COMPACT_THRESHOLD", "0.75"))
MIN_MESSAGES_AFTER_COMPACT = int(os.getenv("CONTEXT_COMPACT_COOLDOWN_MESSAGES", "5"))
COMPACT_TIMEOUT_S = float(os.getenv("CONTEXT_COMPACT_TIMEOUT", "30"))

# Lazy-loaded compactor model
_compactor_model = None
_compactor_tokenizer = None
_compactor_lock = threading.Lock()


def _load_compactor():
    """Load the compactor SLM (once, thread-safe)."""
    global _compactor_model, _compactor_tokenizer
    if _compactor_model is not None:
        return _compactor_model, _compactor_tokenizer

    with _compactor_lock:
        if _compactor_model is not None:
            return _compactor_model, _compactor_tokenizer

        try:
            import mlx_lm
            logger.info(f"[Compactor] Loading compactor model: {COMPACTOR_MODEL}")
            start = time.time()
            _compactor_model, _compactor_tokenizer = mlx_lm.load(COMPACTOR_MODEL)
            logger.info(f"[Compactor] Model loaded in {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"[Compactor] Failed to load model: {e}")
            raise

    return _compactor_model, _compactor_tokenizer


def _generate_summary(messages_text: str, max_tokens: int = 256) -> str:
    """Generate summary using the compactor SLM with MLX_GLOBAL_LOCK."""
    model, tokenizer = _load_compactor()

    prompt = (
        "<|im_start|>system\n"
        "You are a conversation summarizer. Condense the following conversation "
        "into a concise summary preserving all important facts, preferences, and "
        "context. Keep names, dates, numbers, and specific details. Be brief.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Summarize this conversation:\n\n{messages_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        from core.utils.mlx_lock import MLX_GLOBAL_LOCK
    except ImportError:
        MLX_GLOBAL_LOCK = threading.Lock()

    import mlx_lm

    with MLX_GLOBAL_LOCK:
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.3,
        )

    return response.strip()


class ContextCompactor:
    """
    Monitors context size and compresses when needed.

    Integrates into the memory system — called after each turn by the
    frame processor or context injector. Not a separate pipeline processor
    to keep integration simple.
    """

    def __init__(self, max_context_tokens: int = 4096):
        self._max_tokens = max_context_tokens
        self._threshold = COMPACT_THRESHOLD
        self._messages_since_compact = 0
        self._compacting = False
        self._total_compactions = 0

    async def check_and_compact(self, context) -> bool:
        """
        Check if context needs compaction and perform it if so.

        Args:
            context: OpenAI-style context object with get_messages()/set_messages()

        Returns:
            True if compaction was performed, False otherwise.
        """
        if self._compacting:
            return False

        self._messages_since_compact += 1
        if self._messages_since_compact < MIN_MESSAGES_AFTER_COMPACT:
            return False

        messages = context.get_messages()
        total_tokens = TokenEstimator.estimate_messages_tokens(messages)

        threshold_tokens = int(self._max_tokens * self._threshold)
        if total_tokens <= threshold_tokens:
            return False

        logger.info(
            f"[Compactor] Context at {total_tokens}/{self._max_tokens} tokens "
            f"({total_tokens/self._max_tokens*100:.0f}%) — compacting"
        )

        self._compacting = True
        try:
            new_messages = await asyncio.wait_for(
                asyncio.to_thread(self._compact_messages, messages),
                timeout=COMPACT_TIMEOUT_S,
            )
            context.set_messages(new_messages)
            self._messages_since_compact = 0
            self._total_compactions += 1

            new_tokens = TokenEstimator.estimate_messages_tokens(new_messages)
            logger.info(
                f"[Compactor] Compacted {total_tokens} → {new_tokens} tokens "
                f"({len(messages)} → {len(new_messages)} messages)"
            )
            return True

        except asyncio.TimeoutError:
            logger.warning(f"[Compactor] Compaction timed out after {COMPACT_TIMEOUT_S}s")
            return False
        except Exception as e:
            logger.error(f"[Compactor] Compaction failed: {e}")
            return False
        finally:
            self._compacting = False

    def _compact_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Synchronous compaction (runs in thread pool).
        Keeps system messages + recent messages, summarizes the rest.
        """
        # Separate system messages and conversation messages
        system_msgs = []
        conv_msgs = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                conv_msgs.append(msg)

        if len(conv_msgs) <= MIN_MESSAGES_AFTER_COMPACT:
            return messages  # nothing to compact

        # Keep the most recent messages, summarize the rest
        keep_count = max(MIN_MESSAGES_AFTER_COMPACT, len(conv_msgs) // 3)
        to_summarize = conv_msgs[:-keep_count]
        to_keep = conv_msgs[-keep_count:]

        if not to_summarize:
            return messages

        # Build text from messages to summarize
        text_parts = []
        for msg in to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                text_parts.append(f"{role}: {content}")

        messages_text = "\n".join(text_parts)

        # Generate summary
        summary = _generate_summary(messages_text)

        # Build new message list
        summary_msg = {
            "role": "system",
            "content": f"<conversation_summary>\n{summary}\n</conversation_summary>"
        }

        return system_msgs + [summary_msg] + to_keep

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_compactions": self._total_compactions,
            "messages_since_compact": self._messages_since_compact,
            "compacting": self._compacting,
            "threshold": self._threshold,
            "max_tokens": self._max_tokens,
        }
