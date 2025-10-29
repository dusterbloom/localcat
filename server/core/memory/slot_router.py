"""
Slot/attribute detection for memory retrieval.

Provides lightweight, dependency-free (besides stdlib re) detection of
common attribute slots in user queries and conversation snippets.

Goals:
- Detect a small set of high-value slots (favorite_color, favorite_number, favorite_music)
- Handle UK/US variants (favourite/favorite, colour/color)
- Provide a simple API: detect_slot(text) -> slot_id|None, confidence
- Provide alignment check for candidate texts: is_slot_aligned(text, slot_id)

This is intentionally conservative and fast to keep the hot path lean.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


class SlotRouter:
    """Lightweight slot detection and alignment helpers."""

    # Precompiled regexes for slot detection in queries/snippets
    _FAV_COLOR_PATTERNS = [
        # US
        re.compile(r"\bfavorite\s+(?:\w+\s+){0,2}(?:color|shade)\b", re.IGNORECASE),
        re.compile(r"\b(?:color|shade)\s+(?:is|=)\s+\w+", re.IGNORECASE),
        # UK
        re.compile(r"\bfavourite\s+(?:\w+\s+){0,2}(?:colour|shade)\b", re.IGNORECASE),
        re.compile(r"\b(?:colour|shade)\s+(?:is|=)\s+\w+", re.IGNORECASE),
    ]

    _FAV_NUMBER_PATTERNS = [
        re.compile(r"\bfavorite\s+(?:\w+\s+){0,1}(?:number|digit)\b", re.IGNORECASE),
        re.compile(r"\bfavourite\s+(?:\w+\s+){0,1}(?:number|digit)\b", re.IGNORECASE),
        re.compile(r"\b(?:number|digit)\s+(?:is|=)\s+\w+", re.IGNORECASE),
    ]

    _FAV_MUSIC_PATTERNS = [
        re.compile(r"\bfavorite\s+(?:\w+\s+){0,2}(?:music|song|genre)\b", re.IGNORECASE),
        re.compile(r"\bfavourite\s+(?:\w+\s+){0,2}(?:music|song|genre)\b", re.IGNORECASE),
        re.compile(r"\b(?:music|song|genre)\s+(?:is|=)\s+\w+", re.IGNORECASE),
    ]

    @classmethod
    def detect_slot(cls, text: str) -> Tuple[Optional[str], float]:
        """
        Detect a slot (attribute) in the given text.

        Returns:
            (slot_id, confidence) where slot_id is one of:
                'favorite_color', 'favorite_number', 'favorite_music'
            or (None, 0.0) if no confident slot detected.
        """
        t = (text or "").strip()
        if not t:
            return None, 0.0

        # Order matters: more specific slots first
        for rx in cls._FAV_COLOR_PATTERNS:
            if rx.search(t):
                return "favorite_color", 0.9

        for rx in cls._FAV_NUMBER_PATTERNS:
            if rx.search(t):
                return "favorite_number", 0.85

        for rx in cls._FAV_MUSIC_PATTERNS:
            if rx.search(t):
                return "favorite_music", 0.8

        return None, 0.0

    @classmethod
    def is_slot_aligned(cls, text: str, slot_id: Optional[str]) -> bool:
        """
        Return True if the text expresses the given slot.

        Used to filter retrieval candidates so that, for example, a color
        query won't surface 'favorite number' or 'favorite music' snippets.
        """
        if not slot_id:
            return True  # No slot to align against
        t = (text or "").strip()
        if not t:
            return False

        if slot_id == "favorite_color":
            return any(rx.search(t) for rx in cls._FAV_COLOR_PATTERNS)
        if slot_id == "favorite_number":
            return any(rx.search(t) for rx in cls._FAV_NUMBER_PATTERNS)
        if slot_id == "favorite_music":
            return any(rx.search(t) for rx in cls._FAV_MUSIC_PATTERNS)
        return False

