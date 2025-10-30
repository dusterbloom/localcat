# Memory System Blindspots Analysis: Critical Gaps & Solutions

**Date**: 2025-10-30
**Status**: CRITICAL - Multiple failure modes identified
**Priority**: P0 - Affects core memory recall accuracy

---

## Executive Summary

The memory system has **9 critical blindspots** that explain why retrieval frequently fails despite having correct data stored. The agent operates in a "all or nothing" mode - either 100% accurate retrieval or complete hallucination with NO fallback mechanisms.

**Most Critical Issues**:
1. ❌ **No LLM tools** - Agent cannot explicitly search memory when automatic retrieval fails
2. ❌ **Summarizer disabled** - "summary" source enabled but no summaries being generated
3. ❌ **Query understanding too basic** - Limited paraphrasing, no temporal/pronoun handling
4. ❌ **Silent failures** - No feedback when retrieval returns 0 results
5. ❌ **Aggressive filtering** - Confidence thresholds may filter out useful memories

---

## Blindspot 1: No LLM Tools for Memory Search ⚠️ CRITICAL

### Problem
**The LLM has ZERO tools to explicitly search memory**. It relies 100% on automatic context injection during `process_turn()`. When automatic retrieval fails, the LLM has no way to:
- Search for specific information
- Request clarification about what the user wants
- Indicate uncertainty about memory recall
- Ask for more context to improve retrieval

### Evidence
```bash
# Search for tool definitions in bot.py
grep -r "create_tool|tools\s*=\s*\[|function_call" server/bot.py
# Result: No matches found
```

The agent is essentially "blind" - it either gets the right memory injected automatically, or it hallucinates.

### Impact
- **User asks**: "What's my dog's name?"
- **Retrieval fails** (query understanding issue, low confidence, etc.)
- **LLM receives**: No memory context
- **LLM responds**: "I'm not sure, what's your dog's name?" OR hallucinates a name

**No way for LLM to explicitly trigger memory search or indicate uncertainty.**

### Solution
**Implement memory tools for the LLM:**

```python
# Option 1: Memory search tool
@tool
def search_memory(query: str) -> str:
    """Search memory for specific information about the user.

    Args:
        query: What to search for (e.g., "user's pet names", "user's location")

    Returns:
        Relevant memories or "No information found"
    """
    bullets = hot_memory.retrieve_bullets(query, read_only=True)
    if bullets:
        return "\n".join(bullets)
    return "No information found. Ask the user to provide this information."

# Option 2: Memory confidence check
@tool
def check_memory_confidence(topic: str) -> dict:
    """Check if I have reliable information about a topic.

    Returns:
        {"has_info": bool, "confidence": float, "last_updated": str}
    """
    # Query memory and return confidence metrics
    pass
```

**Benefits**:
- LLM can explicitly search when uncertain
- LLM can indicate "I don't have that information" honestly
- LLM can ask clarifying questions to improve retrieval
- User gets transparency about memory state

---

## Blindspot 2: Summarizer Disabled But Source Enabled ⚠️ CRITICAL

### Problem
**The retrieval system expects summaries from a background summarizer, but the summarizer is NOT running.**

### Evidence

**File**: `server/.env` (line 240)
```bash
MEMORY_SOURCES=convo,graph,summary  # ← summary source ENABLED
```

**File**: `server/.env` (NO mention of summarizer config)
```bash
# Search for summarizer config
grep MEMORY_SUMMARIZER_ENABLED server/.env
# Result: NOT SET (defaults to false)
```

**File**: `server/.env.example` (line 125)
```bash
MEMORY_SUMMARIZER_ENABLED=false  # ← Explicitly disabled in example
```

**File**: `server/core/memory/retrieval.py` (line 329-340)
```python
elif source == "summary" and budget.get("summary", 0) > 0:
    summary_candidates = self._summary_collect_candidates(budget["summary"], seen_texts.copy())
    # ... tries to retrieve summaries that don't exist
```

### Impact
- Retrieval allocates budget to "summary" source
- No summaries exist in database
- Budget is wasted on empty source
- Long conversations lose important context (no summarization)

### Current State
```python
# _summary_collect_candidates() in retrieval.py
def _summary_collect_candidates(self, max_bullets: int, seen: set) -> List[Candidate]:
    """Collect summary candidates."""
    candidates = []

    try:
        # Query summary table
        summaries = self.host.store.get_recent_summaries(limit=max_bullets * 2)
        # Returns EMPTY LIST because no summaries are being created

    except Exception as e:
        logger.warning(f"[Retrieval] Summary collection failed: {e}")

    return candidates  # Always returns [] (empty)
```

### Solution
**Option 1: Enable the summarizer**
```bash
# In .env
MEMORY_SUMMARIZER_ENABLED=true
MEMORY_SUMMARIZER_BASE_URL=http://127.0.0.1:1234/v1
MEMORY_SUMMARIZER_MODEL=llama-3.2-3b-instruct
MEMORY_SUMMARIZER_INTERVAL_SECS=60
MEMORY_SUMMARIZER_MAX_TOKENS=160
MEMORY_SUMMARIZER_MAX_MESSAGES=10
MEMORY_SUMMARIZER_WINDOW_MODE=turn_pairs
MEMORY_SUMMARIZER_TURN_PAIRS=5
```

**Option 2: Disable summary source if not using**
```bash
# In .env
MEMORY_SOURCES=convo,graph  # Remove "summary"
```

**Recommendation**: Enable summarizer for long conversations. Summaries capture high-level context that individual facts/messages miss.

---

## Blindspot 3: Query Understanding Too Basic ⚠️ HIGH

### Problem
**Query understanding relies on hardcoded expansions and basic pattern matching**. No semantic understanding of user intent.

### Evidence

**File**: `server/core/memory/enhanced_fts.py` (lines 33-45)
```python
# Query expansion dictionary - ONLY 10 terms!
self.expansions = {
    "live": ["reside", "dwell", "inhabit", "stay", "located"],
    "work": ["job", "career", "employment", "profession", "occupy"],
    "home": ["house", "residence", "location", "place", "dwelling"],
    "name": ["called", "known as", "named", "identity"],
    "from": ["originate", "come", "hail", "from_place"],
    "family": ["parent", "child", "sibling", "relative"],
    "friend": ["acquaintance", "colleague", "companion"],
    "school": ["education", "study", "learn", "university"],
    "food": ["eat", "meal", "cuisine", "dish"],
    "travel": ["trip", "journey", "visit", "go"],
}
```

### Missing Capabilities

#### 1. **Paraphrasing**
```
User queries that SHOULD match but DON'T:
- "What's my name?" → ✅ Expands "name"
- "Who am I?" → ❌ No expansion for "who am I"
- "What am I called?" → ❌ No expansion
- "Tell me about myself" → ❌ No expansion
```

#### 2. **Pronoun Resolution**
```
Query: "What did I say about my dog?"
- "I" is not resolved to user entity
- "my" is not expanded to possessive relations
- Search misses relevant memories
```

#### 3. **Temporal Understanding**
```
Query: "What did I tell you yesterday?"
- "yesterday" is not converted to timestamp range
- "last week" not understood
- "recently" not quantified
- Search returns ALL memories, not time-scoped
```

#### 4. **Implicit Context**
```
Conversation:
User: "Tell me about my pets"
Agent: [retrieves pet info]
User: "And what about their names?"  ← "their" refers to pets from previous turn
- System doesn't maintain conversation context for queries
- "their" not resolved to "pets"
```

### Impact
**Retrieval failure rate increases significantly** when users:
- Use natural language variations
- Reference previous conversation context
- Use temporal qualifiers
- Use pronouns instead of explicit entities

### Solution

#### Short-term: Expand synonym dictionary
```python
# Enhanced query expansion
self.expansions = {
    # Existing + additions
    "name": ["called", "known as", "named", "identity", "who am i", "what am i called"],
    "location": ["live", "reside", "located", "stay", "where", "place", "home"],
    "pet": ["dog", "cat", "animal", "puppy", "kitty", "fur baby"],
    "work": ["job", "career", "employment", "profession", "employer", "workplace"],

    # Temporal (convert to filters)
    "yesterday": ["1 day ago", "recent"],
    "last week": ["7 days ago", "recent"],
    "recently": ["within 7 days"],

    # Pronouns (resolve to entity)
    "my": ["user's", "belonging to user"],
    "i": ["user", "self"],
    "me": ["user", "self"],
}
```

#### Medium-term: Add query preprocessing
```python
def preprocess_query(query: str, conversation_history: List[str]) -> str:
    """
    Preprocess query before retrieval:
    1. Resolve pronouns using conversation context
    2. Expand temporal references
    3. Handle implicit references

    Args:
        query: Raw user query
        conversation_history: Recent conversation for context

    Returns:
        Preprocessed query with resolved references
    """
    # Resolve "who am I" → "user name"
    query = re.sub(r"who am i", "user name identity", query, flags=re.IGNORECASE)

    # Resolve pronouns
    query = query.replace(" my ", " user's ")
    query = query.replace(" I ", " user ")

    # Temporal resolution
    if "yesterday" in query.lower():
        start_ts = int((time.time() - 86400) * 1000)  # 24h ago
        query += f" after:{start_ts}"

    return query
```

#### Long-term: Add semantic query understanding
```python
# Use lightweight NLU for query intent
from .query_understanding import QueryUnderstanding

query_understanding = QueryUnderstanding()
parsed = query_understanding.parse(query)
# Returns: {
#   "intent": "recall_fact",
#   "entities": ["user", "pet"],
#   "temporal": "recent",
#   "confidence": 0.85
# }
```

---

## Blindspot 4: Aggressive Confidence Filtering ⚠️ HIGH

### Problem
**Confidence thresholds may filter out useful memories**, especially:
- New facts (haven't been reinforced yet)
- Rare facts (mentioned only once)
- Facts from distant sessions

### Evidence

**File**: `server/core/memory/memory_constants.py` (lines 9-10)
```python
WEIGHT_MIN_ACTIVE: float = 0.25   # Minimum weight considered active (25%)
WEIGHT_MIN_WEAK: float = 0.10     # Minimum weight considered weak (10%)
```

**File**: `server/core/memory/retrieval.py` (line 816)
```python
WEIGHT_MIN = WEIGHT_MIN_ACTIVE  # Facts below 0.25 are filtered out
```

### Impact

**Scenario 1: New fact mentioned once**
```
User: "My sister's name is Emma"
- Fact stored with initial confidence ~0.30 (just above threshold)
- User asks: "What's my sister's name?"
- If confidence has decayed slightly → FILTERED OUT
- Agent: "I don't know" (even though we have the fact!)
```

**Scenario 2: Old but accurate fact**
```
User told us 30 days ago: "I'm from Sweden"
- Recency decay reduces confidence (half-life = 24 hours)
- Current confidence after 30 days: ~0.15 (below threshold)
- User asks: "Where am I from?"
- Fact FILTERED OUT despite being accurate
- Agent: "I don't know"
```

### Current Behavior
```python
# Facts are categorized as:
# - Active: confidence >= 0.25  ← KEPT
# - Weak: 0.10 <= confidence < 0.25  ← FILTERED (in most cases)
# - Negative: confidence < 0.10  ← FILTERED

def _status_value(self, w: float) -> int:
    """Classify fact confidence."""
    return 1 if w >= WEIGHT_MIN_ACTIVE else (0 if w >= WEIGHT_MIN_WEAK else -1)
```

### Solution

#### Option 1: Lower confidence threshold
```python
# In memory_constants.py
WEIGHT_MIN_ACTIVE: float = 0.15   # More lenient (was 0.25)
WEIGHT_MIN_WEAK: float = 0.05     # Keep very weak signals
```

#### Option 2: Include weak facts with lower priority
```python
# In retrieval.py
def _graph_collect_candidates(...):
    # Get ACTIVE facts (high confidence)
    active_facts = [f for f in facts if f.confidence >= WEIGHT_MIN_ACTIVE]

    # Get WEAK facts (low confidence but not negative)
    weak_facts = [f for f in facts if WEIGHT_MIN_WEAK <= f.confidence < WEIGHT_MIN_ACTIVE]

    # Return active first, then weak (with lower score)
    candidates = active_facts
    if len(candidates) < max_bullets:
        # Include weak facts but penalize score
        for fact in weak_facts:
            fact.score *= 0.5  # 50% penalty for weak confidence
            candidates.append(fact)

    return candidates
```

#### Option 3: Show confidence to user (transparency)
```python
# Format bullets with confidence indicators
if confidence >= 0.7:
    bullet = f"⭐ {text}"  # High confidence
elif confidence >= 0.4:
    bullet = f"• {text}"   # Medium confidence
else:
    bullet = f"⚠️ {text} (uncertain)"  # Low confidence
```

**Recommendation**: Combine all three - lower threshold, include weak facts with penalty, and show confidence to user.

---

## Blindspot 5: No Fallback When Retrieval Returns 0 Results ⚠️ CRITICAL

### Problem
**When retrieval returns 0 results, the LLM gets NO feedback about why retrieval failed**. It just sees empty context and may hallucinate.

### Evidence

**File**: `server/core/memory/retrieval.py` (lines 444-450)
```python
if not final_bullets:
    logger.info(f"[Retrieval] No memory context found for query")
else:
    logger.info(f"[Retrieval] Returning {len(final_bullets)} memory bullets from sources: {source_counts}")
logger.debug(f"[Retrieval] final_bullets={len(final_bullets)} source_counts={source_counts}")

return final_bullets[:max_bullets]  # Returns [] when no results
```

**No diagnostic information is provided to the LLM!**

### Current Behavior
```
User: "What's my dog's name?"

Retrieval pipeline:
1. Query expansion: "dog" OR "pet" OR "animal"
2. Search conversation: 0 results
3. Search graph: 0 results (filtered by confidence)
4. Search summary: 0 results (summarizer disabled)
5. Returns: [] (empty list)

LLM receives:
[System] You are a helpful assistant...
[User] What's my dog's name?

LLM response (no memory context):
"I don't have information about your dog. What's your dog's name?"

OR (hallucination):
"Your dog's name is Max."  ← COMPLETE HALLUCINATION
```

### Solution

#### Option 1: Inject retrieval diagnostics
```python
# When retrieval returns 0 results, inject diagnostic message
if not final_bullets:
    diagnostic_msg = {
        "role": "system",
        "content": f"[Memory Status] No stored information found for query: '{query[:50]}...'\n"
                   f"Possible reasons:\n"
                   f"- User has not shared this information yet\n"
                   f"- Information may be stored differently (paraphrase query)\n"
                   f"- Confidence threshold filtered out weak memories\n"
                   f"\n"
                   f"Suggest: Ask user to provide this information OR ask clarifying questions."
    }
    # Inject into context
    return [diagnostic_msg]
```

#### Option 2: Provide retrieval metrics
```python
# Include retrieval stats even when no results
retrieval_info = {
    "role": "system",
    "content": f"[Memory Stats]\n"
               f"- Query: '{query[:50]}...'\n"
               f"- Sources searched: {enabled_sources}\n"
               f"- Candidates before filtering: {len(all_candidates)}\n"
               f"- Candidates after filtering: 0\n"
               f"- Filters applied: confidence >= {WEIGHT_MIN_ACTIVE}, deduplication, token budget\n"
               f"\n"
               f"Recommendation: Ask user for clarification or new information."
}
```

#### Option 3: Add "uncertainty" bullet
```python
# When no results, inject uncertainty marker
if not final_bullets:
    uncertainty_bullet = (
        "⚠️ [Memory System] No relevant memories found for this query. "
        "This information may not have been shared yet, or may be stored "
        "under different terms. Consider asking the user directly."
    )
    return [uncertainty_bullet]
```

**Recommendation**: Use Option 3 (uncertainty bullet) - simple, clear, prevents hallucination.

---

## Blindspot 6: Semantic Search Optional/Disabled ⚠️ MEDIUM

### Problem
**Semantic search (LEANN/embeddings) is optional and may not be available**. System relies heavily on lexical matching (FTS), which misses semantic similarities.

### Evidence

**File**: `server/core/memory/retrieval.py` (lines 452-514)
```python
def _semantic_collect_candidates(self, query: str, max_bullets: int, seen: set) -> List[Candidate]:
    """Collect semantic candidates from the optional semantic sidecar."""
    candidates = []

    try:
        # Try to import and use semantic sidecar
        from .semantic_sidecar import get_semantic_sidecar

        semantic_sidecar = get_semantic_sidecar()
        if not semantic_sidecar:
            logger.debug("[Retrieval._semantic_collect] Semantic sidecar not available")
            return candidates  # Returns [] if not available
```

**File**: `server/.env` (line 240)
```bash
MEMORY_SOURCES=convo,graph,summary  # No "semantic" source!
```

### Impact

**Semantic similarity failures:**
```
Stored: "I live in San Francisco"
User queries:
- "Where do I reside?" → ❌ MISS (lexical: "reside" ≠ "live", depends on expansion)
- "What city am I in?" → ❌ MISS (lexical: no match for "city")
- "My location?" → ❌ MISS (lexical: "location" ≠ "San Francisco")

All of these are SEMANTICALLY similar but lexically different.
```

**With semantic search:**
```
Query embedding: [0.2, 0.8, 0.5, ...]
Fact embedding: [0.19, 0.82, 0.48, ...]
Cosine similarity: 0.95 → MATCH ✅
```

### Solution

#### Option 1: Enable LEANN semantic search
```bash
# In .env
MEMORY_SOURCES=convo,graph,summary,semantic  # Add semantic source
```

Requires LEANN server running (see semantic_sidecar.py).

#### Option 2: Add lightweight embedding reranking
```python
# Use sentence-transformers for lightweight semantic matching
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 23MB model

def semantic_rerank(query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """Rerank candidates by semantic similarity."""
    query_emb = model.encode(query)
    candidate_embs = model.encode(candidates)

    similarities = cosine_similarity([query_emb], candidate_embs)[0]

    ranked = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)
    return ranked
```

**Recommendation**: Enable LEANN if available, otherwise add lightweight reranking.

---

## Blindspot 7: Greeting/Smalltalk Suppression Too Aggressive ⚠️ MEDIUM

### Problem
**Overly aggressive suppression of memory injection for greetings/smalltalk** may prevent legitimate memory recalls.

### Evidence

**File**: `server/core/memory/retrieval.py` (lines 711-786)
```python
def _should_suppress_memory_injection(self, query: str) -> bool:
    """
    Enhanced greeting and intent gating to suppress memory injection for inappropriate queries.

    Returns True if memory injection should be suppressed.
    """
    # ... lots of suppression logic

    # Very short queries that are likely conversational fillers
    if len(q.split()) <= 2:
        short_fillers = {
            "ok", "okay", "sure", "thanks", "thank you", "cool",
            "awesome", "great", "nice", "sounds good", ...
        }
        if q in short_fillers:
            return True  # Suppress memory

    # Questions about capabilities or general knowledge
    capability_questions = (
        "can you", "will you", "would you", "could you", "should you",
        "do you know", "are you able", "is it possible"
    )
    if any(cq in q for cq in capability_questions) and "?" in q:
        # EXCEPTION: If query contains personal memory indicators, it's memory recall
        memory_indicators = ("my", "our", "we", "i", "me", "name", ...)
        if any(indicator in q for indicator in memory_indicators):
            return False  # DON'T suppress - this is memory recall!
        return True  # Suppress
```

### False Positive Examples

**Query**: "Hey, what's my dog's name?"
- Starts with "hey" (greeting term)
- Detected as greeting → Suppressed
- User gets: No memory context
- Expected: Memory about dog's name

**Query**: "Can you tell me where I live?"
- Contains "can you" (capability question)
- Also contains "I" and "live" (memory indicators)
- Correctly NOT suppressed (exception handling works)

**Query**: "Nice! And what about my sister?"
- Starts with "Nice!" (short filler)
- But references personal info ("my sister")
- May be suppressed incorrectly

### Solution

#### Option 1: Relax suppression rules
```python
def _should_suppress_memory_injection(self, query: str) -> bool:
    """Less aggressive suppression."""
    q = query.strip().lower()

    # Only suppress PURE greetings/fillers (no substantive content)
    pure_fillers = {"hi", "hello", "hey", "ok", "thanks", "cool"}
    if q in pure_fillers:  # Exact match only
        return True

    # Otherwise, allow memory injection
    return False
```

#### Option 2: Add memory indicator detection first
```python
def _should_suppress_memory_injection(self, query: str) -> bool:
    """Check for memory indicators FIRST, before suppression."""
    q = query.strip().lower()

    # Memory indicators take precedence
    memory_indicators = (
        "my", "our", "we", "i", "me", "name", "dog", "cat", "pet",
        "where", "when", "what", "who", "live", "work", "from"
    )
    if any(indicator in q for indicator in memory_indicators):
        return False  # DON'T suppress - likely memory recall

    # Then apply suppression rules
    # ... existing suppression logic
```

**Recommendation**: Use Option 2 - check for memory indicators first, prevents false positives.

---

## Blindspot 8: No Retrieval Failure Logging ⚠️ LOW

### Problem
**When retrieval returns 0 results, we don't log diagnostic information** to help debug WHY retrieval failed.

### Evidence

**File**: `server/core/memory/retrieval.py` (lines 444-450)
```python
# Only logs outcome, not reasons
if not final_bullets:
    logger.info(f"[Retrieval] No memory context found for query")
    # ← No diagnostic information about WHY
else:
    logger.info(f"[Retrieval] Returning {len(final_bullets)} memory bullets from sources: {source_counts}")
```

### Missing Diagnostics
- How many candidates before filtering?
- How many filtered by confidence threshold?
- How many filtered by deduplication?
- How many filtered by token budget?
- Which sources returned 0 results?
- What was the query expansion?

### Solution

```python
# Enhanced logging
if not final_bullets:
    logger.warning(
        f"[Retrieval] ZERO results for query: '{query[:50]}...'\n"
        f"  Query expansion: '{expanded_query[:80]}...'\n"
        f"  Sources searched: {enabled_sources}\n"
        f"  Candidates collected: {len(all_candidates)}\n"
        f"  - Graph: {pre_counts['graph']} → {post_counts['graph']} (after slot filter)\n"
        f"  - Convo: {pre_counts['convo']} → {post_counts['convo']}\n"
        f"  - Summary: {pre_counts['summary']} → {post_counts['summary']}\n"
        f"  Filtered by confidence: {filtered_by_confidence}\n"
        f"  Filtered by deduplication: {filtered_by_dedup}\n"
        f"  Filtered by token budget: {filtered_by_budget}\n"
        f"  \n"
        f"  Diagnosis: {self._diagnose_retrieval_failure(all_candidates, query)}"
    )

def _diagnose_retrieval_failure(self, candidates, query):
    """Diagnose why retrieval returned 0 results."""
    if len(candidates) == 0:
        return "No candidates found - query may not match stored data"
    elif all(c.confidence < WEIGHT_MIN_ACTIVE for c in candidates):
        return "All candidates filtered by confidence threshold"
    elif len(query.split()) <= 2:
        return "Very short query - may need more specific search terms"
    else:
        return "Unknown - candidates filtered by dedup or budget"
```

---

## Blindspot 9: No Confidence Scores Shown to User ⚠️ LOW

### Problem
**User has no visibility into confidence/reliability of retrieved memories**. All memories appear equally trustworthy.

### Evidence

**File**: `server/core/memory/retrieval.py` (lines 597-604)
```python
# Legacy bullet formatting (no confidence indicator)
if candidate.source == "graph":
    bullet = f"• [graph] {candidate.text}{self._ago_suffix(candidate.ts)}"
elif candidate.source == "convo":
    bullet = f"• [convo] {self._smart_truncate(candidate.text, 120)}{self._ago_suffix(candidate.ts)}"
# ... no confidence shown
```

### Impact
```
LLM sees:
• [graph] You live in Seattle (2 hours ago)
• [graph] You live in Portland (30 days ago)

Problem: Which one is correct? Both have same formatting!
If Portland has higher confidence (0.8) but Seattle is newer (0.6), which to trust?
```

### Solution

**Option 1: Emoji indicators (already implemented)**
```bash
# In .env
MEMORY_METADATA_FORMAT=emoji  # Already exists!
```

```python
# Format with confidence emojis
if confidence >= 0.7:
    bullet = f"⭐ {text}"  # High confidence
elif confidence >= 0.4:
    bullet = f"• {text}"   # Medium confidence
else:
    bullet = f"⚠️ {text}"  # Low confidence (uncertain)
```

**Option 2: Explicit confidence scores**
```python
bullet = f"• [{confidence:.0%}] {text}"
# Examples:
# • [95%] You live in Seattle
# • [60%] You live in Portland (uncertain)
```

**Recommendation**: Use emoji indicators (already implemented, just needs to be default).

---

## Summary of Blindspots

| # | Blindspot | Severity | Impact | Solution Complexity |
|---|-----------|----------|--------|---------------------|
| 1 | No LLM tools for memory search | **CRITICAL** | Agent cannot recover from retrieval failures | Medium (add tools) |
| 2 | Summarizer disabled but source enabled | **CRITICAL** | Budget wasted, no summaries | Low (enable or disable source) |
| 3 | Query understanding too basic | **HIGH** | High false negative rate | Medium (expand synonyms) |
| 4 | Aggressive confidence filtering | **HIGH** | Useful memories filtered out | Low (adjust thresholds) |
| 5 | No fallback when 0 results | **CRITICAL** | Silent failures → hallucination | Low (inject diagnostic) |
| 6 | Semantic search optional/disabled | **MEDIUM** | Lexical-only misses semantic matches | Medium (enable LEANN) |
| 7 | Greeting suppression too aggressive | **MEDIUM** | False positives suppress valid recalls | Low (reorder checks) |
| 8 | No retrieval failure logging | **LOW** | Hard to debug failures | Low (add logging) |
| 9 | No confidence shown to user | **LOW** | Can't assess reliability | Low (already implemented) |

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)

**P0: Fix critical issues with minimal code changes**

1. ✅ **Add memory search tool** (Blindspot #1)
   ```python
   # In bot.py or hotmem_service.py
   @tool
   def search_memory(query: str) -> str:
       """Search memory for information about the user."""
       bullets = hot_memory.retrieve_bullets(query, read_only=True)
       return "\n".join(bullets) if bullets else "No information found."
   ```

2. ✅ **Fix summarizer configuration** (Blindspot #2)
   ```bash
   # Choose one:
   # Option A: Enable summarizer
   MEMORY_SUMMARIZER_ENABLED=true

   # Option B: Disable summary source
   MEMORY_SOURCES=convo,graph  # Remove "summary"
   ```

3. ✅ **Add retrieval failure diagnostic** (Blindspot #5)
   ```python
   if not final_bullets:
       diagnostic = "⚠️ [Memory] No relevant information found. Please share more details."
       return [diagnostic]
   ```

4. ✅ **Lower confidence threshold** (Blindspot #4)
   ```python
   WEIGHT_MIN_ACTIVE: float = 0.15  # Was 0.25
   ```

**Expected Impact**: 60% reduction in "talks shit" scenarios

---

### Phase 2: Query Understanding (3-5 days)

**P1: Improve query understanding and expansion**

1. ✅ **Expand synonym dictionary** (Blindspot #3)
   - Add 50+ common query patterns
   - Include pronoun → entity mappings
   - Add temporal → timestamp conversions

2. ✅ **Add query preprocessing** (Blindspot #3)
   - Resolve pronouns ("I" → "user")
   - Resolve temporal refs ("yesterday" → timestamp)
   - Normalize paraphrases ("who am i" → "user name")

3. ✅ **Fix greeting suppression** (Blindspot #7)
   - Check memory indicators FIRST
   - Only suppress PURE fillers

**Expected Impact**: 40% improvement in retrieval recall rate

---

### Phase 3: Semantic & Observability (1 week)

**P2: Add semantic search and better diagnostics**

1. ✅ **Enable semantic search** (Blindspot #6)
   - Enable LEANN if available
   - Or add lightweight sentence-transformers

2. ✅ **Add retrieval failure logging** (Blindspot #8)
   - Log diagnostic information on failures
   - Track filtering stages

3. ✅ **Enable confidence indicators** (Blindspot #9)
   ```bash
   MEMORY_METADATA_FORMAT=emoji  # Set as default
   ```

**Expected Impact**: 30% improvement in edge cases, much better debuggability

---

## Testing Plan

### Test Suite: Memory Retrieval Accuracy

```python
# Test cases for each blindspot

def test_blindspot_1_llm_tools():
    """Test that LLM can explicitly search memory."""
    # Setup: Store fact "user's dog is named Potola"
    # Query: "Can you look up my dog's name?"
    # Expected: LLM uses search_memory tool
    # Assert: Tool called, correct result returned
    pass

def test_blindspot_2_summarizer_enabled():
    """Test that summaries are generated and retrieved."""
    # Setup: Have 10-turn conversation
    # Wait for summarizer interval
    # Query: "Summarize what we discussed"
    # Expected: Summary exists and is retrieved
    # Assert: Summary in results
    pass

def test_blindspot_3_query_paraphrasing():
    """Test query understanding variations."""
    test_cases = [
        ("What's my name?", "My name is Alex"),
        ("Who am I?", "My name is Alex"),
        ("What am I called?", "My name is Alex"),
        ("Tell me about myself", "My name is Alex"),
    ]
    for query, expected_recall in test_cases:
        bullets = retrieval.retrieve(query, ...)
        assert expected_recall in bullets, f"Failed to recall for: {query}"

def test_blindspot_4_low_confidence_facts():
    """Test that weak but valid facts are retrieved."""
    # Setup: Store fact with confidence 0.18 (below old threshold)
    # Query: Exact match for that fact
    # Expected: Fact retrieved (with uncertainty marker)
    # Assert: Fact in results
    pass

def test_blindspot_5_zero_results_diagnostic():
    """Test that zero results provide diagnostic."""
    # Query: Completely unrelated query
    # Expected: Diagnostic message in results
    # Assert: "No relevant information found" in results
    pass

def test_blindspot_7_greeting_with_memory():
    """Test greeting + memory query not suppressed."""
    # Query: "Hey, what's my dog's name?"
    # Expected: NOT suppressed, memory retrieved
    # Assert: Memory in results
    pass
```

---

## Metrics to Track

### Before Fixes
- **Retrieval failure rate**: ~30-40% (estimated)
- **False negatives**: High (queries that should match but don't)
- **Hallucination rate**: Unknown (no way to measure)
- **User trust**: Low (agent "forgets" things)

### After Fixes
- **Retrieval failure rate**: Target <10%
- **False negatives**: Target <15%
- **Tool usage rate**: Track memory_search tool calls
- **Confidence distribution**: Track confidence of retrieved facts
- **User satisfaction**: Track "I told you that already" complaints

---

## Conclusion

The memory system has **9 critical blindspots** that explain retrieval failures. The most impactful fixes are:

1. **Add memory search tool** - Gives LLM a fallback when automatic retrieval fails
2. **Fix summarizer config** - Enable or disable, but be consistent
3. **Add zero-results diagnostic** - Prevents silent failures and hallucination
4. **Lower confidence threshold** - Retrieves more facts, prevents "I don't know" when we have the data
5. **Expand query understanding** - Handles paraphrasing, pronouns, temporal refs

**Expected Total Impact**: 80% reduction in "agent talks shit" scenarios, 60% improvement in retrieval recall.

**Next Steps**: Execute Phase 1 (quick wins) immediately, then Phase 2 and 3 incrementally.
