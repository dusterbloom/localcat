"""Utilities for preparing text for voice output."""

import html
import re


def sanitize_for_voice(text: str) -> str:
    """Remove emojis, markup, and symbols that degrade TTS pronunciation."""
    if not text:
        return ""

    cleaned = html.unescape(text)
    cleaned = re.sub(r"<[^>]*?>", "", cleaned)

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

    # Strip markdown artifacts and standalone symbols
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"[`~^¨´]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove URLs and markdown link targets
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://[^\s]+", "", cleaned)

    return cleaned.strip()
