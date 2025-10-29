"""Token-aware text aggregator optimized for Kokoro TTS."""

import asyncio
import re
from typing import Optional

from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TextFrame
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

    async def _release_text(self):
        """Release accumulated text to TTS."""
        if self._aggregation.strip():
            # Clean and format text for better TTS
            clean_text = self._clean_text_for_tts(self._aggregation)
            if clean_text:
                from loguru import logger
                logger.debug(f"[FastTextAggregator] Releasing text: '{clean_text[:100]}...' (len={len(clean_text)}, aggregated={len(self._aggregation)})")
                await self.push_frame(TextFrame(clean_text))

        self._aggregation = ""
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
            # Clean asterisks from incoming text
            clean_text = frame.text.replace('*', '')
            self._aggregation += clean_text

            from loguru import logger
            logger.debug(f"[FastTextAggregator] Accumulated: '{self._aggregation[:80]}...' (total_len={len(self._aggregation)})")

            # Estimate token count (rough approximation: 1 token ≈ 4 chars for English)
            estimated_tokens = len(self._aggregation) // 4

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
                await self._release_text()
            else:
                # Schedule release after timeout as fallback
                if self._timer and not self._timer.done():
                    self._timer.cancel()
                self._timer = asyncio.create_task(self._delayed_release())

        elif isinstance(frame, (EndFrame, LLMFullResponseEndFrame)):
            if self._aggregation.strip():
                await self._release_text()
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._timer and not self._timer.done():
            self._timer.cancel()
        await super().cleanup()
