"""
Background summarization for conversation turns.

Generates periodic summaries of conversation for long-term memory.
"""

import asyncio
import json
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any
from loguru import logger
from .config_manager import MemoryConfiguration


class BackgroundSummarizer:
    """
    Generate background summaries of conversations.

    Responsibilities:
    - Track turns since last summary
    - Trigger summarization at intervals
    - Generate summaries using LLM
    - Store summaries in memory
    - Support both time-based and turn-based modes
    """

    def __init__(
        self,
        hot_memory,
        config: MemoryConfiguration,
        store=None
    ):
        self.hot = hot_memory
        self.config = config
        self.store = store

        # Summary tracking
        self._turns_since_summary = 0
        self._last_summarized_turn = 0
        self._summary_task: Optional[asyncio.Task] = None

        # Validate configuration
        if self.config.summarization_enabled:
            if self.config.summary_window_mode not in ["delta", "turn_pairs"]:
                logger.warning(f"Invalid summary_window_mode: {self.config.summary_window_mode}, using 'turn_pairs'")
                self.config.summary_window_mode = "turn_pairs"

        logger.info(f"[BackgroundSummarizer] Initialized with mode: {self.config.summary_window_mode}")

    def should_summarize_turns(self, turn_id: int) -> bool:
        """
        Check if summarization should trigger based on turn count.
        
        Args:
            turn_id: Current turn ID
            
        Returns:
            True if summarization should trigger
        """
        if not self.config.summarization_enabled:
            return False

        if self.config.summary_window_mode != "turn_pairs":
            return False

        # Trigger at configured intervals
        if turn_id > 0 and turn_id % self.config.summary_turn_pairs == 0:
            return True

        return False

    async def summarize_turns(self, turn_id: int, session_id: str) -> bool:
        """
        Generate summary of recent turns based on turn count.

        Args:
            turn_id: Current turn ID
            session_id: Session identifier
            
        Returns:
            True if summary was generated and stored
        """
        if not self.config.summarization_enabled:
            return False

        try:
            logger.info(f"[BackgroundSummarizer] Generating turn-based summary at turn {turn_id}")

            # Get recent conversation turns
            messages_to_get = self.config.summary_turn_pairs * 2  # Each turn has user + assistant
            recent = self._get_conversation_chunks(session_id, limit=messages_to_get)
            
            if not recent:
                logger.debug("[BackgroundSummarizer] No recent messages to summarize")
                return False

            # Combine text (limit to 1200 chars)
            text = "; ".join(t for (t, _ts) in recent if t)[:1200]
            logger.info(f"[BackgroundSummarizer] Combined text for summary ({len(text)} chars): {text[:100]}...")
            
            if not text.strip():
                logger.info("[BackgroundSummarizer] No text content to summarize")
                return False

            # Call LLM to generate summary
            content = await self._call_summarizer_llm(text)
            if content:
                now_ms = int(asyncio.get_event_loop().time() * 1000)
                note = f"Summary: {content}"
                
                # Store summary in memory store
                if self.store:
                    self.store.enqueue_mention("summary", note, now_ms, session_id, turn_id)
                    self.store.flush_if_needed()
                
                # Also store in semantic sidecar if available
                try:
                    from .semantic_sidecar import ingest_summary
                    ingest_summary(content, session_id=session_id, ts=now_ms)
                except ImportError:
                    pass  # Semantic sidecar not available

                logger.debug(f"[BackgroundSummarizer] Stored turn-based summary at turn {turn_id}")
                self._last_summarized_turn = turn_id
                return True
            else:
                logger.warning("[BackgroundSummarizer] LLM returned empty summary")
                return False

        except Exception as e:
            logger.warning(f"[BackgroundSummarizer] Turn summary generation failed: {e}")
            return False

    async def start_background_task(self, session_id: str) -> bool:
        """
        Start background summarization task for delta mode.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if task was started successfully
        """
        if not self.config.summarization_enabled:
            return False

        if self.config.summary_window_mode != "delta":
            return False

        if self._summary_task is not None:
            return True  # Already running

        try:
            self._summary_task = asyncio.create_task(self._summary_loop(session_id))
            logger.debug("[BackgroundSummarizer] Background summarizer started (delta mode)")
            return True
        except Exception as e:
            logger.warning(f"[BackgroundSummarizer] Could not start summarizer: {e}")
            return False

    async def stop_background_task(self) -> None:
        """Stop background summarization task."""
        if self._summary_task is not None:
            try:
                self._summary_task.cancel()
                await self._summary_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[BackgroundSummarizer] Error stopping task: {e}")
            finally:
                self._summary_task = None
                logger.debug("[BackgroundSummarizer] Background summarizer stopped")

    async def _summary_loop(self, session_id: str):
        """Periodic background task for delta mode."""
        while True:
            try:
                await asyncio.sleep(self.config.summary_interval_secs)
                
                # Collect recent user utterances
                recent = self._get_conversation_chunks(session_id, limit=self.config.summary_max_messages)
                if not recent:
                    continue

                text = "; ".join(t for (t, _ts) in recent if t)[:1200]
                if not text.strip():
                    continue

                # Call LLM to generate summary
                content = await self._call_summarizer_llm(text)
                if content:
                    now_ms = int(asyncio.get_event_loop().time() * 1000)
                    note = f"Summary: {content}"
                    
                    # Store summary
                    if self.store:
                        self.store.enqueue_mention("summary", note, now_ms, session_id, 0)
                        self.store.flush_if_needed()
                    
                    # Also store in semantic sidecar if available
                    try:
                        from .semantic_sidecar import ingest_summary
                        ingest_summary(content, session_id=session_id, ts=now_ms)
                    except ImportError:
                        pass

                    logger.debug("[BackgroundSummarizer] Stored time-based summary (delta mode)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[BackgroundSummarizer] Summary loop error: {e}")

    async def _call_summarizer_llm(self, text: str) -> Optional[str]:
        """
        Call the summarizer LLM and return the summary content.

        Args:
            text: Text to summarize
            
        Returns:
            Summary content or None if failed
        """
        try:
            # LAYER 3 DEFENSE: Improved prompt to extract facts, not confusion
            sys_prompt = """You are a fact-focused summarizer. Extract only clear, actionable facts from the user's utterances.

RULES:
1. Extract ONLY positive facts (what they like, want, have, do)
2. IGNORE negations, confusion, and meta-commentary
3. Skip phrases like "I don't know", "confusing", "unclear"
4. Focus on names, preferences, experiences, and concrete details
5. Keep under 400 characters

EXAMPLES:
Input: "I'm not interested in this classic detail, it's confusing"
Output: [skip - only negation and confusion]

Input: "My name is John and I love hiking in Colorado"
Output: "User's name is John. Loves hiking in Colorado."

Input: "I have a dog named Max and I'm not sure about cats"
Output: "Has a dog named Max."

Provide ONLY the final summary or nothing if no clear facts exist."""

            # Build OpenAI-compatible chat request
            payload = {
                "model": self.config.summary_model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text},
                ],
                "max_tokens": self.config.summary_max_tokens,
                "temperature": 0.2,
                "stream": False,
            }

            url = f"{self.config.summary_base_url}/chat/completions"
            req = urllib.request.Request(url, method="POST")
            req.add_header("Content-Type", "application/json")
            if self.config.summary_api_key:
                req.add_header("Authorization", f"Bearer {self.config.summary_api_key}")

            data = json.dumps(payload).encode("utf-8")
            
            timeout = 5  # Use a short timeout for LLM calls
            with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
                resp_data = resp.read().decode("utf-8")
            
            j = json.loads(resp_data)
            content = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content if content else None
            
        except urllib.error.URLError as e:
            logger.warning(f"[BackgroundSummarizer] Summarizer LLM call failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"[BackgroundSummarizer] Summarizer LLM error: {e}")
            return None

    def _get_conversation_chunks(self, session_id: str, limit: int = 10) -> List[tuple]:
        """
        Get recent conversation chunks for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of chunks to return
            
        Returns:
            List of (text, timestamp) tuples
        """
        if not self.store:
            return []

        try:
            # Use the store's get_recent_chunks_by_session method
            return self.store.get_recent_chunks_by_session("conversation", session_id, limit)
        except Exception as e:
            logger.debug(f"[BackgroundSummarizer] Failed to get conversation chunks: {e}")
            return []

    def increment_turn_count(self) -> None:
        """Increment turn counter for delta mode tracking."""
        self._turns_since_summary += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get summarizer metrics."""
        metrics = {
            "enabled": self.config.summarization_enabled,
            "window_mode": self.config.summary_window_mode,
            "turns_since_summary": self._turns_since_summary,
            "last_summarized_turn": self._last_summarized_turn,
            "background_task_running": self._summary_task is not None
        }

        if self.config.summarization_enabled:
            metrics.update({
                "summary_interval_secs": self.config.summary_interval_secs,
                "summary_turn_pairs": self.config.summary_turn_pairs,
                "summary_max_tokens": self.config.summary_max_tokens,
                "summary_max_messages": self.config.summary_max_messages
            })

        return metrics

    async def generate_final_summary(self, session_id: str, turn_id: int) -> bool:
        """
        Generate final summary before session end.
        
        Args:
            session_id: Session identifier
            turn_id: Final turn ID
            
        Returns:
            True if final summary was generated
        """
        if not self.config.summarization_enabled:
            return True  # No-op if disabled

        if turn_id <= 1 or turn_id <= self._last_summarized_turn:
            return True  # Nothing to summarize

        logger.info(f"[BackgroundSummarizer] Generating final summary for session (turns {self._last_summarized_turn+1} to {turn_id})")
        
        try:
            # Use shorter timeout for final summary
            summary_task = asyncio.create_task(self.summarize_turns(turn_id, session_id))
            result = await asyncio.wait_for(summary_task, timeout=3.0)
            return result
        except asyncio.TimeoutError:
            logger.warning("[BackgroundSummarizer] Final summary generation timed out")
            return False
        except Exception as e:
            logger.warning(f"[BackgroundSummarizer] Final summary failed: {e}")
            return False
