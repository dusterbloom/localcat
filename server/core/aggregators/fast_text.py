"""Token-aware text aggregator optimized for Kokoro TTS."""

import asyncio
import re
import time
import uuid
from typing import Optional

from loguru import logger

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to character-based token estimation")


from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TextFrame,
    LLMTextFrame  # Import LLMTextFrame to filter it out
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FastTextAggregator(FrameProcessor):
    """Token-aware text aggregator optimized for Kokoro TTS.

    Releases text at natural phoneme boundaries for fluent speech,
    similar to LiveKit's Kokoro implementation.
    """

    def __init__(self, min_tokens: int = 10, max_tokens: int = 250, max_time: float = 0.5, sentence_delimiters: Optional[str] = None, min_words: int = 10):
        # Use direct mode so we don't depend on __process_queue creation in StartFrame.
        # This prevents early non-system frames from tripping internal queues at startup.
        super().__init__(enable_direct_mode=True)
        self._min_tokens = min_tokens  # TARGET_MIN_TOKENS equivalent
        self._max_tokens = max_tokens  # TARGET_MAX_TOKENS equivalent
        self._max_time = max_time  # Fallback timeout
        self._min_words = min_words  # Minimum words before releasing on clause boundaries
        self._aggregation = ""
        self._timer = None
        self._last_release_time = asyncio.get_event_loop().time()
        # Sentence ending patterns
        if sentence_delimiters:
            self._sentence_endings = set(sentence_delimiters)
        else:
            self._sentence_endings = {'.', '!', '?', '。', '！', '？'}
        self._clause_endings = {',', ';', ':', '，', '；', '：'}
        # Auxiliary verbs that should never be separated from their main verbs
        self._auxiliary_verbs = {
            'have', 'has', 'had', 'having',
            'will', 'would', 'shall', 'should', 'could', 'can', 'may', 'might', 'must',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'doing'
        }

        # Frame tracking for duplication detection
        self._seen_text_frames = {}  # text -> (frame_id, timestamp, count)
        self._frame_counter = 0

        # Response tracking to prevent multiple releases
        self._response_complete = False

        # CRITICAL FIX: Lock to prevent buffer corruption during concurrent access
        self._release_lock = asyncio.Lock()

        # Initialize tiktoken for accurate token counting
        self._encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo, text-embedding-ada-002)
                self._encoding = tiktoken.get_encoding("cl100k_base")
                logger.debug("[FastTextAggregator] Using tiktoken for accurate token counting")
            except Exception as e:
                logger.warning(f"[FastTextAggregator] Failed to load tiktoken encoding: {e}")

    async def _release_text(self):
        """Release accumulated text to TTS."""
        # CRITICAL FIX: Use lock to prevent race conditions during concurrent access
        async with self._release_lock:
            # CRITICAL FIX: Check if response is already complete to prevent duplicate releases
            if self._response_complete:
                logger.debug("[FastTextAggregator] _release_text: Response already complete, ignoring")
                return

            # Double-check that we still have content after acquiring lock
            if not self._aggregation.strip():
                return

            import traceback

            # Store text to release and clear buffer atomically
            text_to_release = self._aggregation
            self._aggregation = ""

            # Clean and format text for better TTS
            clean_text = self._clean_text_for_tts(text_to_release)
            if clean_text:
                logger.debug(f"[FastTextAggregator] Releasing: '{clean_text[:50]}...' ({len(clean_text)} chars)")
                await self.push_frame(TextFrame(clean_text))

            # Verify buffer was actually cleared
            if len(self._aggregation) > 0:
                logger.error(f"🚨 FastTextAggregator buffer not cleared! Still has {len(self._aggregation)} chars after release")

            logger.debug(f"[FastTextAggregator] Cleared buffer (was {len(text_to_release)} chars)")

            self._last_release_time = asyncio.get_event_loop().time()

            if self._timer and not self._timer.done():
                self._timer.cancel()
            self._timer = None

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean and format text for better TTS output."""
        from tools.text_formatter import sanitize_for_voice

        # Remove markdown formatting but preserve spacing
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold** but keep text
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *italic* but keep text
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Remove `code` but keep text

        # Remove tool call patterns (XML and JSON) that should not be spoken
        # These are function calls meant for internal processing, not TTS
        text = re.sub(r'<function=\w+>.*?</function>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
        # Remove standalone JSON that looks like tool calls
        text = re.sub(r'\{\s*"query"\s*:.*?"new_information"\s*:.*?\}', '', text, flags=re.DOTALL)

        # Remove emojis and problematic characters for TTS
        text = sanitize_for_voice(text)

        # Normalize ellipses and repeated dots to a single period
        text = re.sub(r'…+', '.', text)
        text = re.sub(r'\.{3,}', '.', text)

        # Remove leading meta hints in parentheses, e.g., (Calculating)
        text = re.sub(r'^\(([^)]+)\)\s*', '', text, flags=re.IGNORECASE)

        # Join digits separated by spaces (e.g., '2  75' -> '275')
        text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)

        # Clean up extra whitespace but preserve sentence structure
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()

        # Ensure proper spacing around punctuation for natural speech
        # Add space after colon if followed by word (but not if it's a time like "1:30")
        text = re.sub(r':(?=\w)', ': ', text)

        # Add space after semicolon if followed by word
        text = re.sub(r';(?=\w)', '; ', text)

        # Clean up any double spaces that might have been created
        text = re.sub(r'\s+', ' ', text)

        return text

    def _count_words(self, text: str) -> int:
        """Count words in text (simple whitespace split)."""
        return len(text.strip().split())

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken (or fallback to estimation).

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception as e:
                logger.warning(f"[FastTextAggregator] tiktoken encoding failed: {e}, falling back to estimation")

        # Fallback: rough approximation (1 token ≈ 4 chars for English)
        return len(text) // 4

    def _is_safe_break_point(self, text: str, position: int) -> bool:
        """Check if position is safe for breaking (not right after auxiliary verb).

        Returns True if it's safe to break at the given position.
        Returns False if breaking would separate auxiliary verb from main verb.
        """
        if position <= 0 or position >= len(text):
            return False

        # Get the word before the break position
        before_break = text[:position].strip()
        if not before_break:
            return False

        # Extract the last word before the break
        last_word = before_break.split()[-1].lower().rstrip(',.;:!?')

        # Check if it's an auxiliary verb that shouldn't be separated
        if last_word in self._auxiliary_verbs:
            return False

        return True

    async def _delayed_release(self):
        """Release text after timeout."""
        try:
            await asyncio.sleep(self._max_time)
            if self._aggregation.strip():
                await self._release_text()
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            return

        # Handle interruptions
        if isinstance(frame, (CancelFrame, InterruptionFrame, BotInterruptionFrame)):
            self._aggregation = ""
            if self._timer and not self._timer.done():
                self._timer.cancel()
                self._timer = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            # Defensive check: ensure frame.text is actually a string
            if not isinstance(frame.text, str):
                logger.error(f"🚨 FastTextAggregator received TextFrame with non-string text: {type(frame.text)} = {frame.text}")
                await self.push_frame(frame, direction)  # Pass through without processing
                return

            # CRITICAL FIX: Reset response_complete flag when new text arrives after end
            if self._response_complete:
                logger.debug("[FastTextAggregator] New TextFrame arrived after response complete, resetting flag")
                self._response_complete = False

            # FRAME TRACKING: Check for duplicates
            self._frame_counter += 1
            frame_id = f"TF{self._frame_counter}_{int(time.time() * 1000)}"

            text_content = frame.text.strip()
            if text_content in self._seen_text_frames:
                prev_id, prev_timestamp, count = self._seen_text_frames[text_content]
                time_diff = time.time() - prev_timestamp

                # Track for debugging only - don't block (common words appear in different responses)
                logger.debug(f"Token seen before: '{text_content[:30]}' ({time_diff:.1f}s ago, count: {count + 1})")

                # Update counter for tracking
                self._seen_text_frames[text_content] = (frame_id, time.time(), count + 1)

                # Continue processing - no blocking (removed overly aggressive duplicate prevention)
            else:
                # First time seeing this text in current session
                self._seen_text_frames[text_content] = (frame_id, time.time(), 1)
                logger.debug(f"🔍 New TextFrame: '{text_content[:30]}...' (len={len(text_content)})")

            # Clean asterisks from incoming text
            clean_text = frame.text.replace('*', '')

            # Simple text accumulation as Pipecat standard pattern
            self._aggregation += clean_text

            logger.debug(f"[FastTextAggregator] Accumulated: '{self._aggregation[:80]}...' (total_len={len(self._aggregation)})")

            # Count tokens accurately with tiktoken (or fallback to estimation)
            estimated_tokens = self._count_tokens(self._aggregation)

            # Check for natural boundaries
            should_release = False

            # Check if we hit a sentence ending - ALWAYS release complete sentences immediately
            # This is critical for natural voice conversation flow
            if self._aggregation.rstrip() and self._aggregation.rstrip()[-1] in self._sentence_endings:
                should_release = True
                logger.debug(f"[FastTextAggregator] Sentence boundary detected, releasing: '{self._aggregation[:60]}...'")

            # Check if we hit max token limit - but try to find a good break point
            elif estimated_tokens >= self._max_tokens:
                # Look for the last good break point (sentence or clause ending)
                text = self._aggregation.rstrip()
                last_sentence_idx = -1
                last_clause_idx = -1

                # Find last sentence boundary
                for i in range(len(text) - 1, -1, -1):
                    if text[i] in self._sentence_endings:
                        last_sentence_idx = i
                        break
                    elif text[i] in self._clause_endings:
                        if last_clause_idx == -1:
                            last_clause_idx = i

                # Prefer sentence boundary, then clause, then word boundary
                if last_sentence_idx > len(text) // 2:  # Found sentence boundary in second half
                    self._aggregation = text[:last_sentence_idx + 1]
                    should_release = True
                elif last_clause_idx > len(text) // 2:  # Found clause boundary in second half
                    self._aggregation = text[:last_clause_idx + 1]
                    should_release = True
                else:
                    # Force release at word boundary, but avoid breaking auxiliary verbs
                    # Start from the end and find the last SAFE space (not after auxiliary verb)
                    halfway_idx = len(text) // 2
                    last_safe_space = -1

                    # Search backward from end for safe break point
                    for i in range(len(text) - 1, halfway_idx, -1):
                        if text[i] == ' ' and self._is_safe_break_point(text, i):
                            last_safe_space = i
                            break

                    if last_safe_space > halfway_idx:
                        self._aggregation = text[:last_safe_space]
                        logger.debug(f"[FastTextAggregator] Safe word boundary break at position {last_safe_space}")
                    else:
                        # Fallback: just use last space if no safe point found
                        # (better to break at auxiliary verb than cut mid-word)
                        last_space = text.rfind(' ')
                        if last_space > halfway_idx:
                            self._aggregation = text[:last_space]
                            logger.warning(f"[FastTextAggregator] No safe break point found, using last space (may break auxiliary verb)")
                    should_release = True
            # Check if we have enough tokens and hit a clause boundary
            # BUT require minimum word count to avoid releasing tiny phrases like "Hello,"
            elif estimated_tokens >= self._min_tokens:
                if self._aggregation.rstrip() and self._aggregation.rstrip()[-1] in self._clause_endings:
                    word_count = self._count_words(self._aggregation)
                    if word_count >= self._min_words:
                        should_release = True
                        logger.debug(f"[FastTextAggregator] Clause boundary with {word_count} words, releasing")
                    else:
                        logger.debug(f"[FastTextAggregator] Clause boundary but only {word_count} words (min {self._min_words}), continuing")

            if should_release:
                # Release text immediately at natural boundaries
                await self._release_text()
                # CRITICAL FIX: Cancel any pending timer to prevent duplicate release
                if self._timer and not self._timer.done():
                    self._timer.cancel()
                    self._timer = None
            else:
                # Schedule release after timeout as fallback
                if self._timer and not self._timer.done():
                    self._timer.cancel()
                self._timer = asyncio.create_task(self._delayed_release())

        elif isinstance(frame, (EndFrame, LLMFullResponseEndFrame)):
            # CRITICAL FIX: Use lock to prevent race conditions with concurrent _release_text calls
            async with self._release_lock:
                # Cancel any pending timer
                if self._timer and not self._timer.done():
                    self._timer.cancel()
                    self._timer = None

                # CRITICAL FIX: Only release if we have accumulated text AND buffer is not empty
                # This prevents double-release of text that was already released by boundaries
                if self._aggregation.strip() and len(self._aggregation) > 0:
                    logger.debug(f"[FastTextAggregator] EndFrame releasing remaining text: '{self._aggregation[:50]}...' ({len(self._aggregation)} chars)")
                    # Release text without using the lock (we already hold it)
                    text_to_release = self._aggregation
                    self._aggregation = ""

                    # Clean and format text for better TTS
                    clean_text = self._clean_text_for_tts(text_to_release)
                    if clean_text:
                        logger.debug(f"[FastTextAggregator] EndFrame pushing final text: '{clean_text[:50]}...'")
                        await self.push_frame(TextFrame(clean_text))
                else:
                    logger.debug("[FastTextAggregator] EndFrame: No remaining text to release")

                # Mark response as complete to prevent further releases
                self._response_complete = True
                logger.debug("[FastTextAggregator] EndFrame marked response as complete")

                # Clear seen frames cache for next response (prevents cross-response interference)
                self._seen_text_frames.clear()
                logger.debug("[FastTextAggregator] Cleared frame tracking cache for next response")

            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._timer and not self._timer.done():
            self._timer.cancel()
        await super().cleanup()
