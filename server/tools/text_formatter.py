"""Utilities for preparing text for voice output."""

import html
import re


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
