# Memory Deduplication Enhancement

## Problem

The memory retrieval system was returning duplicate information from different sources (graph, conversation history, summaries). For example:
- Graph: "you favorite_color yellow"
- Graph: "favorite color is yellow"

These are semantically the same information but were being returned as separate memory bullets, wasting context tokens and confusing the LLM.

## Root Causes

1. **Text-based deduplication too weak**: The original `_normalize_candidate_text()` only did basic text normalization (case, whitespace) but didn't handle:
   - Underscores in graph relations (e.g., `favorite_color` vs `favorite color`)
   - Punctuation variations
   - Semantic similarity (different wording, same meaning)

2. **Diversity penalty limited to same source**: The `_calculate_diversity_penalty()` method only compared candidates within the same source (`if other.source != candidate.source: continue`), so it couldn't detect duplicates across sources.

## Solution

### 1. Enhanced Text Normalization (`retrieval.py:624-641`)

```python
def _normalize_candidate_text(self, text: str) -> str:
    """
    Normalize candidate text for cross-source deduplication.

    Removes source tags, normalizes case, removes punctuation/underscores, and collapses whitespace.
    This creates a more aggressive normalization for better duplicate detection.
    """
    import re
    # Remove source tags like [graph], [convo], etc.
    normalized = re.sub(r'\[(graph|convo|summary|semantic)\]\s*', '', text, flags=re.IGNORECASE)
    # Normalize case
    normalized = normalized.lower()
    # Replace underscores with spaces (before other punctuation removal)
    normalized = normalized.replace('_', ' ')
    # Remove punctuation and normalize whitespace
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = ' '.join(normalized.split())
    return normalized
```

**Improvements**:
- Converts underscores to spaces
- Removes all punctuation
- More aggressive normalization catches more duplicates

### 2. Semantic Similarity Check (`retrieval.py:643-673`)

```python
def _are_semantically_similar(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
    """
    Check if two texts are semantically similar using Jaccard similarity.

    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (default 0.6 = 60% overlap)

    Returns:
        True if texts are similar enough to be considered duplicates
    """
    # Normalize both texts
    norm1 = self._normalize_candidate_text(text1)
    norm2 = self._normalize_candidate_text(text2)

    # Split into word sets
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    # Handle empty sets
    if not words1 or not words2:
        return norm1 == norm2

    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    if union == 0:
        return False

    similarity = intersection / union
    return similarity >= threshold
```

**How it works**:
- Uses Jaccard similarity: `|A ∩ B| / |A ∪ B|`
- Example: "you favorite_color yellow" vs "favorite color is yellow"
  - Normalized: "you favorite color yellow" vs "favorite color is yellow"
  - Words A: {you, favorite, color, yellow}
  - Words B: {favorite, color, is, yellow}
  - Intersection: {favorite, color, yellow} = 3 words
  - Union: {you, favorite, color, is, yellow} = 5 words
  - Similarity: 3/5 = 0.6 (60%) → **Detected as duplicate!**

### 3. Enhanced Deduplication Logic (`retrieval.py:552-579`)

```python
for score, candidate, components in scored_candidates:
    # Enhanced cross-source deduplication:
    # 1. Exact match on normalized text
    normalized_text = self._normalize_candidate_text(candidate.text)

    if normalized_text in seen_normalized_texts:
        logger.debug(f"[Retrieval] Skipping exact duplicate: '{candidate.text[:50]}...'")
        continue

    # 2. Semantic similarity check against all selected candidates
    try:
        similarity_threshold = float(os.getenv("MEMORY_DEDUP_THRESHOLD", "0.6"))
    except (ValueError, TypeError):
        similarity_threshold = 0.6

    is_duplicate = False
    for selected in selected_candidates:
        if self._are_semantically_similar(candidate.text, selected.text, similarity_threshold):
            logger.debug(
                f"[Retrieval] Skipping semantic duplicate: '{candidate.text[:50]}...' "
                f"(similar to '{selected.text[:50]}...')"
            )
            is_duplicate = True
            break

    if is_duplicate:
        continue
```

**Two-phase deduplication**:
1. **Exact match**: Fast hash-based lookup for identical normalized text
2. **Semantic similarity**: Jaccard similarity check against already-selected candidates

### 4. Cross-Source Diversity Penalty (`retrieval.py:1787-1791`)

```python
for other in other_candidates:
    if other is candidate:
        continue  # Skip self-comparison

    # REMOVED SOURCE RESTRICTION: Now compares across all sources
    # This enables cross-source diversity penalty to catch duplicates
```

**Change**: Removed `if other.source != candidate.source: continue` to enable cross-source comparison.

## Configuration

The deduplication threshold is configurable via environment variable:

```bash
# Default: 0.6 (60% word overlap required)
MEMORY_DEDUP_THRESHOLD=0.6

# More aggressive (50% overlap)
MEMORY_DEDUP_THRESHOLD=0.5

# Less aggressive (70% overlap)
MEMORY_DEDUP_THRESHOLD=0.7
```

## Testing

Comprehensive test suite in `tests/unit/test_memory_deduplication.py`:

- ✅ Exact duplicate detection
- ✅ Punctuation/underscore normalization
- ✅ Semantic similarity detection (60% threshold)
- ✅ Low overlap rejection
- ✅ Threshold configuration
- ✅ Cross-source deduplication
- ✅ Case-insensitive matching
- ✅ Empty text handling
- ✅ Configurable thresholds

All 9 tests pass.

## Performance Impact

**Minimal**:
- Normalization is O(n) where n = text length (fast)
- Semantic check is O(m) where m = number of already-selected candidates (typically ≤ 3)
- Only applied during final bullet selection, not during candidate collection

**Benefit**:
- Eliminates redundant memory bullets
- Reduces context token usage
- Improves LLM response quality

## Example

### Before Fix
```
• [graph] you favorite_color yellow [conf=0.75 rec=0.99]
• [graph] favorite color is yellow [conf=0.75 rec=0.99]
• [convo] I like yellow a lot [conf=0.60 rec=0.95]
```

### After Fix
```
• [graph] you favorite_color yellow [conf=0.75 rec=0.99]
• [convo] I like yellow a lot [conf=0.60 rec=0.95]
```

**Result**: 33% fewer bullets, no redundancy, same information coverage!

## Deployment

1. No database changes required
2. No configuration changes required (uses sensible defaults)
3. Can adjust threshold via `MEMORY_DEDUP_THRESHOLD` if needed
4. Fully backward compatible

## Future Enhancements

Potential improvements for even better deduplication:

1. **Embedding-based similarity**: Use sentence embeddings for deeper semantic matching
2. **Coreference resolution**: Detect "yellow" vs "that color" as referring to same entity
3. **Relation-aware deduplication**: Understand that "favorite_color" and "likes color" are related
4. **Token-based scoring**: Prefer shorter formulations when duplicates have same information
