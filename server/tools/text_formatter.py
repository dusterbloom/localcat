"""Utilities for preparing text for voice output."""

import html
import re
from typing import List


def sanitize_for_voice(text: str, expand_contractions: bool = False) -> str:
    """Remove emojis, markup, and symbols that degrade TTS pronunciation.

    Args:
        text: Input text to sanitize
        expand_contractions: If True, expand contractions for clearer TTS
    """
    if not text:
        return ""

    cleaned = html.unescape(text)
    cleaned = re.sub(r"<[^>]*?>", "", cleaned)

    # Normalize apostrophes for better TTS handling
    # Convert various apostrophe types to standard apostrophe
    cleaned = re.sub(r"[''`´]", "'", cleaned)

    # Optionally expand contractions for better TTS pronunciation
    if expand_contractions:
        # Expand problematic contractions
        expansions = {
            r"\byou'd\b": "you would",
            r"\bwe'd\b": "we would",
            r"\bthey'd\b": "they would",
            r"\bhe'd\b": "he would",
            r"\bshe'd\b": "she would",
            r"\bit'd\b": "it would",
            r"\byou'll\b": "you will",
            r"\bwe'll\b": "we will",
            r"\bthey'll\b": "they will",
            r"\bhe'll\b": "he will",
            r"\bshe'll\b": "she will",
            r"\bit'll\b": "it will",
            r"\byou're\b": "you are",
            r"\bwe're\b": "we are",
            r"\bthey're\b": "they are",
            r"\byou've\b": "you have",
            r"\bwe've\b": "we have",
            r"\bthey've\b": "they have",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bwouldn't\b": "would not",
            r"\bcouldn't\b": "could not",
            r"\bshouldn't\b": "should not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
            r"\bhadn't\b": "had not",
        }
        for pattern, replacement in expansions.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Skip the contraction fixes below since we expanded them
    else:
        # Keep contractions but ensure they're properly formatted
        pass  # The apostrophe normalization above handles this


    # Remove broad emoji ranges and zero-width joiners
    emoji_ranges = [
        r"\U0001F600-\U0001F64F",
        r"\U0001F300-\U0001F5FF",
        r"\U0001F680-\U0001F6FF",
        r"\U0001F700-\U0001F77F",
        r"\U0001F780-\U0001F7FF",
        r"\U0001F800-\U0001F8FF",
        r"\U0001F900-\U0001F9FF",
        r"\U0001FA00-\U0001FA6F",
        r"\U0001FA70-\U0001FAFF",
        r"\U0001F1E0-\U0001F1FF",
        r"\U00002600-\U000026FF",
        r"\U00002700-\U000027BF",
        r"\U0000FE00-\U0000FE0F",
    ]
    cleaned = re.sub("[" + "".join(emoji_ranges) + "]", "", cleaned)
    cleaned = re.sub(r"[\u200C\u200D]", "", cleaned)

    # Strip markdown artifacts and standalone symbols (but preserve apostrophes)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"[~^¨]", "", cleaned)  # Removed backtick and ´ from here since we handle them above
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove URLs and markdown link targets
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://[^\s]+", "", cleaned)

    return cleaned.strip()


def sanitize_for_kokoro(text: str, max_sentence_length: int = 110) -> str:
    """Enhanced sanitization specifically for Kokoro TTS with sentence splitting.

    Addresses Kokoro's specific issues:
    - Problematic quotes and special characters
    - Long sentences that cause high latency
    - Complex punctuation patterns

    Args:
        text: Input text to sanitize
        max_sentence_length: Maximum character length per sentence

    Returns:
        Sanitized text optimized for Kokoro TTS
    """
    if not text:
        return ""

    # Basic sanitization first
    cleaned = sanitize_for_voice(text, expand_contractions=False)

    # Kokoro-specific fixes
    # 1. Replace problematic quotes with TTS-friendly alternatives
    cleaned = re.sub(r'"([^"]+)"', r'\1', cleaned)  # Remove quotes entirely
    cleaned = re.sub(r'"', '', cleaned)  # Remove any remaining quotes

    # 2. Fix specific problematic patterns for Kokoro
    cleaned = re.sub(r'\bi\.e\.\b', 'that is', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\be\.g\.\b', 'for example', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\betc\.\b', 'and so on', cleaned, flags=re.IGNORECASE)

    # 3. Normalize dashes and hyphens
    cleaned = re.sub(r'[—–−]', '-', cleaned)
    cleaned = re.sub(r'\s+-\s+', ' - ', cleaned)  # Ensure spaces around dashes

    # 4. Fix parenthetical expressions that confuse TTS
    cleaned = re.sub(r'\(([^)]{1,30})\)', r'- \1 -', cleaned)  # Short parentheticals
    cleaned = re.sub(r'\([^)]{31,}\)', '', cleaned)  # Remove long parentheticals

    return cleaned


def chunk_for_kokoro_ultra_low_latency(text: str, max_chars: int = 25, *, min_chars: int = 12) -> List[str]:
    """
    Optimal text chunking for Kokoro TTS - Ultra Low Latency
    Target: <800ms voice-to-voice latency

    Based on extensive benchmarking showing TTFT scales terribly with text length:
    - 25 chars: ~487ms TTFT (optimal)
    - 96 chars: ~1,486ms TTFT (3x slower)
    - 257 chars: ~3,556ms TTFT (7x slower)

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (25 recommended for <800ms latency)

    Returns:
        List of chunks optimized for streaming TTS
    """
    # Defensive bounds
    max_chars = max(8, int(max_chars))
    min_chars = max(1, min(int(min_chars), max_chars))

    chunks = []

    # Split on sentence boundaries first
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Split long sentences by clauses/commas first
            clauses = re.split(r'[,;:]+', sentence)
            for clause in clauses:
                clause = clause.strip()
                if clause and len(clause) <= max_chars:
                    chunks.append(clause)
                elif clause:
                    # Force split at word boundaries for very long clauses
                    words = clause.split()
                    current_chunk = ""
                    for word in words:
                        test_chunk = f"{current_chunk} {word}" if current_chunk else word
                        if len(test_chunk) <= max_chars:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word
                    if current_chunk:
                        chunks.append(current_chunk)

    chunks = [chunk for chunk in chunks if chunk.strip()]

    # Post-pass: coalesce micro-chunks to preserve natural prosody
    # Merge consecutive small chunks until they reach ~60% of max_chars
    if not chunks:
        return chunks

    merged: List[str] = []
    buffer = ""
    threshold = max(1, int(0.6 * max_chars))

    for c in chunks:
        if not buffer:
            buffer = c
            continue
        if len(buffer) < min_chars or len(buffer) < threshold:
            candidate = f"{buffer} {c}".strip()
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
        merged.append(buffer)
        buffer = c

    if buffer:
        merged.append(buffer)

    return merged


def split_text_for_kokoro_streaming(text: str, min_length: int = 50, max_length: int = 120) -> List[str]:
    """Split text into optimal chunks for Kokoro streaming.

    ⚠️ DEPRECATED: Use chunk_for_kokoro_ultra_low_latency() for <800ms latency
    This function uses 50-120 char chunks which are too slow for real-time voice agents.

    Kokoro works best with natural sentences of 50-120 characters.
    This function finds the natural sentence boundaries and groups them
    into optimal-sized chunks.

    Args:
        text: Input text to split
        min_length: Minimum length per chunk (avoid too-small chunks)
        max_length: Maximum length per chunk (Kokoro's sweet spot)

    Returns:
        List of text chunks optimized for Kokoro performance
    """
    if not text:
        return []

    # First sanitize the text but don't split yet
    cleaned = sanitize_for_kokoro(text, max_length * 2)  # Allow longer for natural grouping

    if not cleaned:
        return []

    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    # Post-process to fix obvious abbreviation breaks
    if sentences:
        merged = [sentences[0]]
        for sentence in sentences[1:]:
            # Check if previous sentence ends with common abbreviation
            prev_ends_with_abbrev = (
                merged[-1].endswith('Dr.') or
                merged[-1].endswith('Mr.') or
                merged[-1].endswith('Mrs.') or
                merged[-1].endswith('Ms.') or
                merged[-1].endswith('Prof.') or
                merged[-1].endswith('Sr.') or
                merged[-1].endswith('Jr.') or
                merged[-1].endswith('e.g.') or
                merged[-1].endswith('i.e.') or
                merged[-1].endswith('etc.') or
                merged[-1].endswith('vs.') or
                merged[-1].endswith('cf.') or
                merged[-1].endswith('al.') or
                (len(merged[-1]) > 2 and merged[-1][-1].isupper() and merged[-1][-2] == '.' and merged[-1][-3] == ' ')
            )

            # Merge if it looks like a broken abbreviation
            if prev_ends_with_abbrev:
                merged[-1] = merged[-1] + ' ' + sentence
            else:
                merged.append(sentence)

        sentences = merged

    # Filter and group sentences optimally
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If sentence alone is within optimal range, use it
        if min_length <= len(sentence) <= max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence)

        # If sentence is too long, split it intelligently
        elif len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split long sentence at natural breaks
            parts = smart_sentence_split(sentence, max_length)
            chunks.extend(parts)

        # If sentence is short, try to group with others
        else:
            test_chunk = (current_chunk + " " + sentence).strip()
            if len(test_chunk) <= max_length:
                current_chunk = test_chunk
            else:
                # Current chunk would be too long
                if current_chunk and len(current_chunk) >= min_length:
                    # Current chunk is good size, save it
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                elif current_chunk:
                    # Current chunk is too small, force combine anyway if not too big
                    if len(test_chunk) <= max_length + 20:  # Allow slight overflow for grouping
                        current_chunk = test_chunk
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                else:
                    current_chunk = sentence

    # Add any remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Post-process: merge chunks that are too short to avoid tiny utterances
    min_speakable_length = 5  # Minimum characters for a speakable chunk (conversational)
    merged_chunks = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Skip memory context that shouldn't be spoken
        if chunk.startswith('[graph]') or chunk.startswith('•') or 'ago)' in chunk:
            continue

        # If this chunk is too short and we have a previous chunk, merge them
        # Be more aggressive about merging for conversational responses
        if (len(chunk) < min_speakable_length and merged_chunks and
            len(merged_chunks[-1]) + len(chunk) + 1 <= max_length + 50):  # Allow more overflow
            merged_chunks[-1] = merged_chunks[-1] + ' ' + chunk
        else:
            merged_chunks.append(chunk)

    return merged_chunks


def smart_sentence_split(text: str, max_length: int = 110) -> List[str]:
    """Split text into TTS-friendly sentences with intelligent breaking.

    Args:
        text: Input text to split
        max_length: Maximum character length per sentence

    Returns:
        List of sentences optimized for TTS
    """
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Post-process to fix obvious abbreviation breaks
    if sentences:
        merged = [sentences[0]]
        for sentence in sentences[1:]:
            # Check if previous sentence ends with common abbreviation
            prev_ends_with_abbrev = (
                merged[-1].endswith('Dr.') or
                merged[-1].endswith('Mr.') or
                merged[-1].endswith('Mrs.') or
                merged[-1].endswith('Ms.') or
                merged[-1].endswith('Prof.') or
                merged[-1].endswith('Sr.') or
                merged[-1].endswith('Jr.') or
                merged[-1].endswith('e.g.') or
                merged[-1].endswith('i.e.') or
                merged[-1].endswith('etc.') or
                merged[-1].endswith('vs.') or
                merged[-1].endswith('cf.') or
                merged[-1].endswith('al.') or
                (len(merged[-1]) > 2 and merged[-1][-1].isupper() and merged[-1][-2] == '.' and merged[-1][-3] == ' ')
            )

            # Merge if it looks like a broken abbreviation
            if prev_ends_with_abbrev:
                merged[-1] = merged[-1] + ' ' + sentence
            else:
                merged.append(sentence)

        sentences = merged

    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If sentence is short enough, keep it
        if len(sentence) <= max_length:
            result.append(sentence)
            continue

        # Split long sentences at natural break points
        parts = split_long_sentence(sentence, max_length)
        result.extend(parts)

    return [s for s in result if s.strip()]


def split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """Split a single long sentence at natural break points.

    Args:
        sentence: The sentence to split
        max_length: Maximum length per part

    Returns:
        List of sentence parts
    """
    if len(sentence) <= max_length:
        return [sentence]

    # Try to split at natural break points in order of preference
    break_patterns = [
        r'(,\s+(?:but|and|or|yet|so|for)\s+)',  # Coordinating conjunctions
        r'(,\s+(?:however|therefore|moreover|furthermore|nevertheless)\s+)',  # Conjunctive adverbs
        r'(,\s+(?:which|that|who|where|when)\s+)',  # Relative clauses
        r'(,\s+)',  # Any comma
        r'(\s+(?:and|or|but)\s+)',  # Conjunctions without comma
        r'(\s+)',  # Any whitespace as last resort
    ]

    parts = []
    remaining = sentence

    while len(remaining) > max_length:
        best_split = None
        best_pos = -1

        # Find the best split point within max_length
        for pattern in break_patterns:
            matches = list(re.finditer(pattern, remaining[:max_length], re.IGNORECASE))
            if matches:
                # Use the last match (closest to max_length)
                match = matches[-1]
                split_pos = match.end()
                if split_pos > best_pos:
                    best_pos = split_pos
                    best_split = pattern
                break

        if best_pos > 0:
            # Split at the best position
            part = remaining[:best_pos].strip()
            # Ensure part ends with proper punctuation
            if part and not part[-1] in '.!?,' and not part.endswith('--'):
                part += ','
            parts.append(part)
            remaining = remaining[best_pos:].strip()
        else:
            # No good split found, force split at max_length
            part = remaining[:max_length].strip()
            if part and not part[-1] in '.!?,' and not part.endswith('--'):
                part += ','
            parts.append(part)
            remaining = remaining[max_length:].strip()

    # Add remaining text
    if remaining:
        # Ensure it ends with punctuation
        if remaining and not remaining[-1] in '.!?':
            remaining += '.'
        parts.append(remaining)

    return parts
