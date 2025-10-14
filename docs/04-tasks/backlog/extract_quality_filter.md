# Extract QualityFilter to Eliminate DRY Violation

**Priority**: Critical (DRY Violation)
**Effort**: 2 days
**Assigned To**: Memory Systems Specialist

## Problem Statement

Quality filtering logic is duplicated across 2 locations with ~150 lines of duplicate code. Pattern lists (confusion patterns, system patterns, etc.) are maintained in multiple places, causing:
- Pattern updates requiring changes in 2 files
- Inconsistent filtering between storage and retrieval
- Higher maintenance burden

**Duplicate Locations**:
1. `hotpath_processor.py:414-486` (_is_quality_conversation)
2. `retrieval.py:746-819` (_is_quality_bullet)

**Example Duplication**:
```python
# DUPLICATED in both files:
confusion_patterns = [
    "confus", "unclear", "don't understand", ...
]
system_patterns = [
    "[memory", "[session", ...
]
# ... same patterns repeated in both files
```

## Impact

- **Maintenance**: Pattern updates require changing 2 files
- **Consistency**: Risk of divergent filtering logic
- **Code Quality**: 150 lines of unnecessary duplication

## Success Metrics

- ✓ Single QualityFilter class replaces both duplications
- ✓ ~150 lines of duplicate code eliminated
- ✓ All existing memory tests pass
- ✓ Quality filtering consistent between storage and retrieval
- ✓ Performance maintained (<800ms voice latency)

## Implementation Approach

### Step 1: Create QualityFilter Class

```python
# server/core/memory/quality_filter.py (NEW FILE)
"""
Unified quality filtering for conversation text in memory systems.

Provides multi-layer defense against low-quality conversation fragments:
- Layer 2: Storage-time filtering (prevent junk from being stored)
- Layer 4: Retrieval-time filtering (prevent junk from being injected)
"""

from typing import Set
import re


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
        "confus", "unclear", "don't understand", "not sure",
        "what do you mean", "huh", "sorry", "pardon",
        "can you repeat", "didn't catch", "missed that"
    ]

    SYSTEM_PATTERNS = [
        "[memory", "[session", "[context", "[system",
        "[debug", "[log", "[error", "[warning"
    ]

    FILLER_PATTERNS = [
        "um", "uh", "er", "ah", "hmm", "mm",
        "like", "you know", "i mean", "kind of"
    ]

    TRANSCRIPTION_ARTIFACTS = [
        "[inaudible]", "[crosstalk]", "[silence]",
        "[background noise]", "[music]", "...", "---"
    ]

    EMPTY_RESPONSES = [
        "ok", "okay", "yes", "no", "yeah", "nope",
        "sure", "fine", "alright", "got it", "thanks"
    ]

    def __init__(self):
        """Initialize quality filter with compiled patterns"""
        # Pre-compile regex patterns for performance
        self._confusion_regex = re.compile(
            "|".join(re.escape(p) for p in self.CONFUSION_PATTERNS),
            re.IGNORECASE
        )
        self._system_regex = re.compile(
            "|".join(re.escape(p) for p in self.SYSTEM_PATTERNS),
            re.IGNORECASE
        )
        self._filler_regex = re.compile(
            "|".join(r"\b" + re.escape(p) + r"\b" for p in self.FILLER_PATTERNS),
            re.IGNORECASE
        )

    def is_quality_for_storage(self, text: str) -> bool:
        """
        Layer 2 defense: Should this text be stored in memory?

        Replaces hotpath_processor.py:414-486 (_is_quality_conversation)

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

        # Check for system/debug artifacts
        if self._system_regex.search(text):
            return False

        # Check for transcription artifacts
        if any(artifact in text.lower() for artifact in self.TRANSCRIPTION_ARTIFACTS):
            return False

        # Check for confusion/misunderstanding
        if self._confusion_regex.search(text):
            return False

        # Check for excessive brackets (metadata pollution)
        bracket_count = text.count('[') + text.count(']') + text.count('(') + text.count(')')
        if bracket_count / len(text) > QualityFilterConfig.MAX_BRACKET_RATIO:
            return False

        # Check for repeated characters (transcription errors)
        if re.search(r'(.)\1{' + str(QualityFilterConfig.MAX_REPEATED_CHARS) + r',}', text):
            return False

        # Check if text is mostly filler words
        filler_matches = len(self._filler_regex.findall(text))
        if filler_matches > len(words) * 0.5:  # More than 50% filler
            return False

        return True

    def is_quality_for_retrieval(self, text: str) -> bool:
        """
        Layer 4 defense: Should this text be injected into context?

        Replaces retrieval.py:746-819 (_is_quality_bullet)

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

        # Check minimum word count (stricter than storage)
        words = text.split()
        if len(words) < QualityFilterConfig.MIN_WORDS_FOR_RETRIEVAL:
            return False

        # Check for system/debug artifacts
        if self._system_regex.search(text):
            return False

        # Check for transcription artifacts
        if any(artifact in text.lower() for artifact in self.TRANSCRIPTION_ARTIFACTS):
            return False

        # Check for empty responses (too generic for context)
        text_lower = text.lower()
        if any(text_lower == empty.lower() for empty in self.EMPTY_RESPONSES):
            return False

        # Check for confusion/misunderstanding
        if self._confusion_regex.search(text):
            return False

        # Check for excessive brackets
        bracket_count = text.count('[') + text.count(']') + text.count('(') + text.count(')')
        if bracket_count / len(text) > QualityFilterConfig.MAX_BRACKET_RATIO:
            return False

        # Check for repeated characters
        if re.search(r'(.)\1{' + str(QualityFilterConfig.MAX_REPEATED_CHARS) + r',}', text):
            return False

        # Stricter filler check for retrieval
        filler_matches = len(self._filler_regex.findall(text))
        if filler_matches > len(words) * 0.3:  # More than 30% filler
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
            score -= 0.2

        # Penalize confusion patterns
        if self._confusion_regex.search(text):
            score -= 0.3

        # Penalize filler words
        filler_ratio = len(self._filler_regex.findall(text)) / max(len(words), 1)
        score -= filler_ratio * 0.4

        # Penalize brackets/metadata
        bracket_ratio = (text.count('[') + text.count(']')) / max(len(text), 1)
        score -= bracket_ratio * 0.5

        # Penalize system artifacts
        if self._system_regex.search(text):
            score -= 0.5

        return max(0.0, min(1.0, score))
```

### Step 2: Refactor hotpath_processor.py

```python
# server/core/memory/hotpath_processor.py
from .quality_filter import QualityFilter

class HotPathMemoryProcessor(BaseProcessor):
    def __init__(self, ...):
        # ... existing init code ...

        # NEW: Initialize quality filter
        self.quality_filter = QualityFilter()

    def _is_quality_conversation(self, text: str) -> bool:
        """REPLACE with QualityFilter method (Layer 2 defense)"""
        return self.quality_filter.is_quality_for_storage(text)
```

### Step 3: Refactor retrieval.py

```python
# server/core/memory/retrieval.py
from .quality_filter import QualityFilter

class MemRetrieval:
    def __init__(self, ...):
        # ... existing init code ...

        # NEW: Initialize quality filter
        self.quality_filter = QualityFilter()

    def _is_quality_bullet(self, text: str) -> bool:
        """REPLACE with QualityFilter method (Layer 4 defense)"""
        return self.quality_filter.is_quality_for_retrieval(text)
```

## Testing Requirements

### Unit Tests

```python
# server/tests/unit/memory_filter/test_quality_filter.py (NEW)
import pytest
from server.core.memory.quality_filter import QualityFilter

def test_quality_for_storage_minimum_words():
    """Reject text below minimum word count"""
    filter = QualityFilter()

    assert filter.is_quality_for_storage("Hi") == False  # Too short
    assert filter.is_quality_for_storage("Hello there") == False  # 2 words
    assert filter.is_quality_for_storage("Hello there friend") == True  # 3 words OK

def test_quality_for_storage_system_patterns():
    """Reject system/debug artifacts"""
    filter = QualityFilter()

    assert filter.is_quality_for_storage("[memory] storing fact") == False
    assert filter.is_quality_for_storage("[debug] test message") == False
    assert filter.is_quality_for_storage("Normal conversation text") == True

def test_quality_for_storage_confusion():
    """Reject confusion/misunderstanding"""
    filter = QualityFilter()

    assert filter.is_quality_for_storage("I'm confused about this") == False
    assert filter.is_quality_for_storage("What do you mean by that") == False
    assert filter.is_quality_for_storage("I understand this clearly") == True

def test_quality_for_storage_transcription_artifacts():
    """Reject transcription artifacts"""
    filter = QualityFilter()

    assert filter.is_quality_for_storage("[inaudible] some text") == False
    assert filter.is_quality_for_storage("Clear speech here") == True

def test_quality_for_storage_filler_words():
    """Reject text with excessive filler"""
    filter = QualityFilter()

    assert filter.is_quality_for_storage("um like uh you know") == False  # All filler
    assert filter.is_quality_for_storage("I like pizza") == True  # "like" as verb, OK

def test_quality_for_retrieval_stricter_than_storage():
    """Retrieval filter should be stricter"""
    filter = QualityFilter()

    # Text that passes storage but fails retrieval
    text = "okay got it"  # Only 3 words

    # Should pass storage (3 words minimum)
    assert filter.is_quality_for_storage(text) == True

    # Should fail retrieval (4 words minimum + empty response)
    assert filter.is_quality_for_retrieval(text) == False

def test_quality_for_retrieval_empty_responses():
    """Reject empty/generic responses"""
    filter = QualityFilter()

    assert filter.is_quality_for_retrieval("okay") == False
    assert filter.is_quality_for_retrieval("yes") == False
    assert filter.is_quality_for_retrieval("got it") == False
    assert filter.is_quality_for_retrieval("The user likes pizza") == True

def test_quality_score_calculation():
    """Quality scoring for ranking"""
    filter = QualityFilter()

    # High quality text
    high_score = filter.get_quality_score("The user lives in San Francisco")
    assert high_score > 0.8

    # Medium quality text (short)
    medium_score = filter.get_quality_score("User likes pizza")
    assert 0.5 < medium_score < 0.8

    # Low quality text (filler + confusion)
    low_score = filter.get_quality_score("um I'm confused about this")
    assert low_score < 0.5

def test_bracket_ratio_threshold():
    """Reject text with excessive brackets"""
    filter = QualityFilter()

    # Excessive brackets (metadata pollution)
    assert filter.is_quality_for_storage("[tag1] [tag2] [tag3] short text") == False

    # Normal brackets
    assert filter.is_quality_for_storage("The user (named John) likes pizza") == True

def test_repeated_characters():
    """Reject text with transcription errors"""
    filter = QualityFilter()

    assert filter.is_quality_for_storage("I liiiiiike this") == False  # Repeated i
    assert filter.is_quality_for_storage("I like this") == True
```

### Integration Tests

```python
# server/tests/integration/test_quality_filtering_consistency.py (NEW)
def test_storage_and_retrieval_filtering_consistency():
    """Verify consistent filtering across storage and retrieval"""

    # Create processor (uses storage filter)
    processor = HotPathMemoryProcessor(...)

    # Try to store low-quality text
    processor.process_turn("um okay", role="user")

    # Verify it was filtered out
    retrieval = MemRetrieval(...)
    bullets = retrieval.retrieve("what did I say?")

    # Should NOT contain the filtered text
    assert not any("um okay" in bullet for bullet in bullets)

def test_quality_filtering_layer_defense():
    """Test multi-layer defense (storage + retrieval)"""

    # Low quality text that somehow passes storage
    processor = HotPathMemoryProcessor(...)

    # Bypass storage filter (simulate edge case)
    processor.hot.store.store_conversation(
        text="[debug] test message",  # System artifact
        role="user"
    )

    # Retrieval filter should catch it (Layer 4 defense)
    retrieval = MemRetrieval(...)
    bullets = retrieval.retrieve("test")

    # Should NOT inject debug message into context
    assert not any("[debug]" in bullet for bullet in bullets)
```

### Regression Tests

```python
# server/tests/unit/test_memory_system.py (EXISTING - must still pass)
# All existing tests must pass after refactoring
pytest server/tests/unit/test_memory_system.py -v
```

## Performance Validation

```bash
# Benchmark memory operations before/after
pytest server/tests/performance/test_memory_retrieval.py -v

# Expected: No regression in retrieval time (<50ms)
# Expected: Possible improvement due to compiled regex
```

## Files to Modify

1. **server/core/memory/quality_filter.py** (NEW)
   - QualityFilter class with all methods
   - QualityFilterConfig for thresholds
   - ~150 lines (consolidates 150 duplicate lines)

2. **server/core/memory/hotpath_processor.py**
   - Remove _is_quality_conversation implementation (lines 414-486)
   - Delegate to QualityFilter.is_quality_for_storage()

3. **server/core/memory/retrieval.py**
   - Remove _is_quality_bullet implementation (lines 746-819)
   - Delegate to QualityFilter.is_quality_for_retrieval()

4. **server/tests/unit/memory_filter/** (NEW)
   - test_quality_filter.py (comprehensive unit tests)

5. **server/tests/integration/** (EXISTING)
   - test_quality_filtering_consistency.py (NEW - cross-module consistency)

## Definition of Done

- [ ] QualityFilter class implemented with all methods
- [ ] hotpath_processor.py refactored to use QualityFilter
- [ ] retrieval.py refactored to use QualityFilter
- [ ] ~150 lines of duplicate code removed
- [ ] All existing tests pass (regression check)
- [ ] New unit tests pass (filter behavior)
- [ ] Integration tests pass (multi-layer defense)
- [ ] Performance tests pass (<800ms latency maintained)
- [ ] Code review completed
- [ ] Documentation added to quality_filter.py

## Delegation Command

```bash
# Manager delegates to Memory Systems Specialist
droid exec memory-systems-specialist --auto medium -f tasks/extract_quality_filter.md
```

---

**Related Issues**: Part of technical debt cleanup (Phase 1, Critical Priority)
**Blocks**: None (standalone refactor)
**References**: Tech debt guardian report - Critical Issue #5 (Quality filter duplication)
