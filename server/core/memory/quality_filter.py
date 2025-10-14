"""
Unified quality filtering for conversation text in memory systems.

Provides multi-layer defense against low-quality conversation fragments:
- Layer 2: Storage-time filtering (prevent junk from being stored)
- Layer 4: Retrieval-time filtering (prevent junk from being injected)

This module eliminates ~150 lines of duplicate code previously split between
hotpath_processor.py and retrieval.py.
"""

from typing import Set
import re
from loguru import logger


class QualityFilterConfig:
    """Configuration for quality filtering thresholds"""

    # Minimum word counts
    MIN_WORDS_FOR_STORAGE = 3
    MIN_WORDS_FOR_RETRIEVAL = 4

    # Pattern matching
    MAX_BRACKET_RATIO = 0.3  # Max 30% of text in brackets
    MAX_REPEATED_CHARS = 4  # "aaaaa" is suspicious


class QualityFilter:
    """
    Unified quality filtering for conversation text.

    Multi-layer defense strategy:
    - Layer 2 (Storage): Prevent junk from entering memory store
    - Layer 4 (Retrieval): Prevent junk from reaching context injection

    Each layer has different thresholds based on its position in pipeline.
    """

    # Shared pattern definitions (single source of truth)
    CONFUSION_PATTERNS = [
        "confus",  # confusing, confused
        "unclear",
        "don't understand",
        "doesn't make sense",
        "not sure what",
        "what do you mean",
        "what are you",
        "i don't know",
        "dunno",
        "no idea",
        "meta",
        "context unclear",
        "repetitive",
    ]

    SYSTEM_PATTERNS = [
        "[memory",
        "[session",
        "[context",
        "[system",
        "[debug",
        "[log",
        "[error",
        "[warning",
        "context]",
        "summary:",
    ]

    FILLER_PATTERNS = [
        "um",
        "uh",
        "er",
        "ah",
        "hmm",
        "mm",
        "like",
        "you know",
        "i mean",
        "kind of",
    ]

    TRANSCRIPTION_ARTIFACTS = [
        "[inaudible]",
        "[crosstalk]",
        "[silence]",
        "[background noise]",
        "[music]",
        "...",
        "---",
    ]

    EMPTY_RESPONSES = [
        "ok",
        "okay",
        "yes",
        "no",
        "yeah",
        "nope",
        "sure",
        "fine",
        "alright",
        "got it",
        "thanks",
    ]

    # Interjections (used in Layer 4 only for stricter filtering)
    INTERJECTIONS = [
        "oh",
        "wow",
        "lol",
        "yeah",
        "hmm",
        "uh",
        "ok",
        "okay",
        "right",
        "sure",
        "thanks",
        "omg",
        "whoa",
        "yay",
        "eww",
        "ugh",
        "huh",
        "meh",
        "nah",
        "yep",
        "yup",
        "nope",
        "awesome",
        "amazing",
        "cool",
        "nice",
        "great",
        "good",
        "bad",
        "terrible",
    ]

    # Content patterns (used in Layer 4 only for content detection)
    CONTENT_PATTERNS = [
        " work",
        " job",
        " live",
        " home",
        " family",
        " friend",
        " school",
        " college",
        " university",
        " company",
        " business",
        " project",
        " task",
        " meeting",
        " appointment",
        " travel",
        " trip",
        " vacation",
        " movie",
        " book",
        " music",
        " food",
        " restaurant",
        " cook",
        " eat",
        " drink",
        " buy",
        " purchase",
        " sell",
        " rent",
        " own",
        " have",
        " think",
        " believe",
        " feel",
        " want",
        " need",
        " prefer",
        " like",
        " love",
        " hate",
        " go",
        " come",
        " move",
        " drive",
        " fly",
        " walk",
        " run",
        " play",
        " watch",
        " make",
        " create",
        " build",
        " design",
        " write",
        " read",
        " learn",
        " teach",
    ]

    def __init__(self):
        """Initialize quality filter with compiled patterns"""
        # Pre-compile regex patterns for performance
        self._confusion_regex = re.compile(
            "|".join(re.escape(p) for p in self.CONFUSION_PATTERNS), re.IGNORECASE
        )
        self._system_regex = re.compile(
            "|".join(re.escape(p) for p in self.SYSTEM_PATTERNS), re.IGNORECASE
        )
        self._filler_regex = re.compile(
            "|".join(r"\b" + re.escape(p) + r"\b" for p in self.FILLER_PATTERNS),
            re.IGNORECASE,
        )

    def is_quality_for_storage(self, text: str) -> bool:
        """
        Layer 2 defense: Should this text be stored in memory?

        Replaces hotpath_processor.py:416-488 (_is_quality_conversation)

        Args:
            text: Conversation text to evaluate

        Returns:
            True if text meets storage quality standards
        """
        if not text or not text.strip():
            return False

        text = text.strip()

        # Check minimum word count
        words = text.split()
        if len(words) < QualityFilterConfig.MIN_WORDS_FOR_STORAGE:
            return False

        t = text.lower()

        # Check for system/debug artifacts
        if self._system_regex.search(text):
            logger.debug(f"[QualityFilter] Filtering system-like message: '{text[:50]}...'")
            return False

        # Check for transcription artifacts
        if any(artifact in t for artifact in self.TRANSCRIPTION_ARTIFACTS):
            return False

        # Check for confusion/misunderstanding
        if self._confusion_regex.search(text):
            logger.debug(
                f"[QualityFilter] Filtering confused/meta utterance: '{text[:50]}...'"
            )
            return False

        # Check for excessive brackets (metadata pollution)
        bracket_count = text.count("[") + text.count("]") + text.count("(") + text.count(")")
        token_count = len([tok for tok in text.split() if tok])
        bracket_token_count = len(
            [
                tok
                for tok in text.split()
                if tok.startswith(("[", "(")) and tok.endswith(("]", ")"))
            ]
        )
        if len(text) > 0 and bracket_count / len(text) > QualityFilterConfig.MAX_BRACKET_RATIO:
            return False
        if token_count and bracket_token_count / token_count > QualityFilterConfig.MAX_BRACKET_RATIO:
            return False

        # Check for repeated characters (transcription errors)
        if re.search(
            r"(.)\1{" + str(QualityFilterConfig.MAX_REPEATED_CHARS) + r",}", text
        ):
            return False

        # Check if text is mostly filler words
        filler_matches = len(self._filler_regex.findall(text))
        if filler_matches > len(words) * 0.5:  # More than 50% filler
            return False

        # Filter pure questions without assertions
        if t.endswith("?"):
            question_words = {
                "who",
                "what",
                "where",
                "when",
                "why",
                "how",
                "whom",
                "whose",
                "which",
                "do",
                "does",
                "did",
                "can",
                "could",
                "would",
                "should",
                "will",
            }
            leading = t.split()[0] if t.split() else ""
            if leading in question_words:
                logger.debug(
                    f"[QualityFilter] Filtering pure question without assertions: '{text[:50]}...'"
                )
                return False

        return True

    def is_quality_for_retrieval(self, text: str) -> bool:
        """
        Layer 4 defense: Should this text be injected into context?

        Replaces retrieval.py:1081-1154 (_is_quality_bullet)

        More strict than storage filtering - we want only the best
        context bullets to be injected.

        Args:
            text: Memory bullet text to evaluate

        Returns:
            True if text meets retrieval quality standards
        """
        if not text or not text.strip():
            return False

        text = text.strip()
        t = text.lower()

        # Check minimum word count (stricter than storage)
        words = t.split()
        if len(words) < QualityFilterConfig.MIN_WORDS_FOR_RETRIEVAL:
            return False

        # Filter very short utterances (< 15 chars for stricter filtering)
        if len(t) < 15:
            return False

        # Filter common interjections/fillers unless followed by substantive content
        # Check if text starts with or is mostly interjection
        if words and words[0] in self.INTERJECTIONS and len(t) < 30:
            return False

        # Filter pure interjections (very short after removing interjection words)
        filtered_text = " ".join([w for w in words if w not in self.INTERJECTIONS])
        if len(filtered_text.strip()) < 10:
            return False

        # Check for system/debug artifacts
        if self._system_regex.search(text):
            return False

        # Check for transcription artifacts
        if any(artifact in t for artifact in self.TRANSCRIPTION_ARTIFACTS):
            return False

        # Check for empty responses (too generic for context)
        if any(t == empty.lower() for empty in self.EMPTY_RESPONSES):
            return False

        # Check for confusion/misunderstanding
        if self._confusion_regex.search(text):
            return False

        # Check for excessive brackets
        bracket_count = text.count("[") + text.count("]") + text.count("(") + text.count(")")
        token_count = len([tok for tok in text.split() if tok])
        bracket_token_count = len(
            [
                tok
                for tok in text.split()
                if tok.startswith(("[", "(")) and tok.endswith(("]", ")"))
            ]
        )
        if len(text) > 0 and bracket_count / len(text) > QualityFilterConfig.MAX_BRACKET_RATIO:
            return False
        if token_count and bracket_token_count / token_count > QualityFilterConfig.MAX_BRACKET_RATIO:
            return False

        # Check for repeated characters
        if re.search(
            r"(.)\1{" + str(QualityFilterConfig.MAX_REPEATED_CHARS) + r",}", text
        ):
            return False

        # Stricter filler check for retrieval
        filler_matches = len(self._filler_regex.findall(text))
        if filler_matches > len(words) * 0.3:  # More than 30% filler
            return False

        # Filter pure questions without assertions
        if t.endswith("?"):
            question_words = {
                "who",
                "what",
                "where",
                "when",
                "why",
                "how",
                "whom",
                "whose",
                "which",
                "do",
                "does",
                "did",
                "can",
                "could",
                "would",
                "should",
                "will",
            }
            leading = t.split()[0] if t.split() else ""
            if leading in question_words:
                return False

        # Require at least one content token (heuristic: contains a content verb/noun pattern)
        has_content = any(pattern in t for pattern in self.CONTENT_PATTERNS)
        if not has_content and len(t) < 40:  # Allow longer utterances even without obvious content patterns
            return False

        return True

    def get_quality_score(self, text: str) -> float:
        """
        Calculate quality score (0.0-1.0) for text.

        Useful for ranking/sorting memory bullets by quality.

        Args:
            text: Text to score

        Returns:
            Quality score from 0.0 (lowest) to 1.0 (highest)
        """
        if not text or not text.strip():
            return 0.0

        score = 1.0
        text = text.strip()
        words = text.split()

        # Penalize short text
        if len(words) < 5:
            score -= 0.25

        # Penalize confusion patterns
        if self._confusion_regex.search(text):
            score -= 0.4

        # Penalize filler words
        filler_ratio = len(self._filler_regex.findall(text)) / max(len(words), 1)
        score -= filler_ratio * 0.6

        # Penalize brackets/metadata
        bracket_ratio = (text.count("[") + text.count("]")) / max(len(text), 1)
        score -= bracket_ratio * 0.5

        # Penalize system artifacts
        if self._system_regex.search(text):
            score -= 0.5

        return max(0.0, min(1.0, score))
