# Duplicate Assistant Text Bug - Root Cause Analysis & Fix

## Problem Summary

Duplicate assistant messages were appearing in the UI with nearly identical text but different spacing:
- Message 1: "You can think of me as having a short-term memory, able to recall..."
- Message 2: "You can think of me as having a short-term memory,able to recall..." (missing space after comma)

## Root Cause Analysis

### 1. Text Splitting in FastTextAggregator

**File:** `server/core/aggregators/fast_text.py:227-231`

The FastTextAggregator splits text at clause boundaries (commas, semicolons) when:
- Token count >= `min_tokens` (default: 10)
- Word count >= `min_words` (default: 10)

```python
elif estimated_tokens >= self._min_tokens:
    if self._aggregation.rstrip() and self._aggregation.rstrip()[-1] in self._clause_endings:
        word_count = self._count_words(self._aggregation)
        if word_count >= self._min_words:
            should_release = True
```

**Evidence from logs (13:54:46.870):**
```
[FastTextAggregator] Releasing text: 'You can think of me as having a short-term memory,...'
```

The sentence was split into TWO TextFrames:
1. Fragment 1: " You can think of me as having a short-term memory," (10 words)
2. Fragment 2: " able to recall the latest 3+ turns of our chat."

### 2. Space Removal in Text Cleaning

**File:** `server/core/aggregators/fast_text.py:93`

The `_clean_text_for_tts` method strips leading/trailing spaces:

```python
text = text.strip()  # Removes leading space from " able"
```

This converts:
- Fragment 1: " You can think..." → "You can think..."
- Fragment 2: " able to recall..." → "able to recall..." (SPACE REMOVED!)

### 3. Space-less Concatenation in Context Aggregator

**File:** `server/.venv/.../pipecat/processors/aggregators/llm_response.py:1008`

The OpenAIAssistantContextAggregator concatenates without adding spaces:

```python
async def _handle_text(self, frame: TextFrame):
    if not self._started:
        return

    if self._params.expect_stripped_words:
        self._aggregation += f" {frame.text}" if self._aggregation else frame.text
    else:
        self._aggregation += frame.text  # <-- NO SPACE ADDED!
```

**Result:** "memory," + "able" = "memory,able" (missing space)

**Evidence from logs (13:54:59.074):**
```
📝 CONVERSATION [ASSISTANT]: I can remember details... You can think of me as having a short-term memory,able to recall...
```

### 4. Failed Client-Side Deduplication

**File:** `client/src/components/VoiceApp.tsx:217-219` (BEFORE FIX)

The original deduplication logic:

```typescript
if (msg.text === newText ||
    msg.text.includes(newText) ||
    newText.includes(msg.text)) {
```

This fails when comparing:
- Version A: "memory, able" (with space)
- Version B: "memory,able" (without space)

Both checks fail:
```javascript
"memory, able".includes("memory,able")  // false
"memory,able".includes("memory, able")  // false
```

## The Fix

### Solution: Improved Whitespace Normalization

**Files Modified:**
- `client/src/components/VoiceApp.tsx` (lines 173-179, 228-233)

**New normalization function:**

```typescript
const normalizeWhitespace = (text: string) =>
  text
    .replace(/\s+/g, ' ')  // Collapse multiple spaces to single space
    .replace(/\s*([,;:.!?])\s*/g, '$1')  // Remove spaces around punctuation
    .trim();
```

**How it works:**

1. Collapses multiple spaces: `"memory,  able"` → `"memory, able"`
2. Normalizes punctuation spacing: `"memory, able"` → `"memory,able"`
3. Trims leading/trailing: `" text "` → `"text"`

**Result:**
```typescript
normalizeWhitespace("memory, able")   // "memory,able"
normalizeWhitespace("memory,able")    // "memory,able"
// These are now identical!
```

### Updated Deduplication Logic

```typescript
const normalizedNewText = normalizeWhitespace(newText);

for (const msg of allAssistantMessages) {
  const normalizedMsgText = normalizeWhitespace(msg.text);

  // Exact match check (after normalization)
  if (normalizedMsgText === normalizedNewText) {
    console.log('🚫 Duplicate sentence found (exact match after normalization)');
    return prev;
  }

  // Substring match (one contains the other)
  if (normalizedMsgText.includes(normalizedNewText) ||
      normalizedNewText.includes(normalizedMsgText)) {
    console.log('🚫 Duplicate/partial sentence found (substring match)');
    return prev;
  }
}
```

## Testing

### Test Coverage

**File:** `client/src/utils/__tests__/deduplication.test.ts`

Tests cover:
1. Exact duplicate detection with whitespace variations
2. Multiple space normalization
3. Real-world bug scenario from server logs
4. Fragment vs full text matching
5. Edge cases (empty strings, different texts)

### Verification Script

**File:** `verify_dedup_fix.js`

Run with: `node verify_dedup_fix.js`

**Results:**
```
Test 1: Exact duplicate from logs ✅
Test 2: Fragment vs full text ✅
Test 3: Multiple spaces normalization ✅
Test 4: Different texts ✅

Results: 4 passed, 0 failed
```

## Evidence Trail

### Server Logs (2025-10-29 13:54)

```
13:54:46.870 | [FastTextAggregator] Releasing text: 'You can think... memory,...'
13:54:59.074 | CONVERSATION [ASSISTANT]: ...memory,able to recall...
```

### Screenshot Evidence

Two nearly identical messages in UI:
1. "You can think of me as having a short-term memory, able to recall..." (with space)
2. "You can think of me as having a short-term memory,able to recall..." (no space)

## Pipeline Flow

```
LLM Output (TextFrame with tokens)
    ↓
FastTextAggregator
    ├─ Accumulates tokens
    ├─ Detects 10 words + comma → SPLIT
    ├─ Fragment 1: "...memory,"
    └─ Fragment 2: " able to recall..."
    ↓
_clean_text_for_tts
    ├─ Strips leading/trailing spaces
    ├─ Fragment 1: "...memory,"
    └─ Fragment 2: "able to recall..." (space removed!)
    ↓
TTS (KokoroProfessional with push_text_frames=True)
    ├─ Emits TTSTextFrame for each fragment
    └─ Client receives via onBotTtsText
    ↓
OpenAIAssistantContextAggregator
    ├─ Concatenates WITHOUT spaces
    ├─ Result: "memory," + "able" = "memory,able"
    └─ Stored in conversation context
    ↓
Client Receives BOTH:
    ├─ TTSTextFrames: Multiple events with proper spacing
    └─ Context/Transcript: One event with missing space
    ↓
Old Deduplication: FAILS (different spacing)
New Deduplication: SUCCEEDS (normalized comparison)
```

## Impact

- **Before:** Users see duplicate messages with subtle spacing differences
- **After:** Duplicates are correctly detected and prevented, regardless of whitespace variations

## Files Changed

1. `client/src/components/VoiceApp.tsx`
   - Lines 173-195 (onBotTranscript deduplication)
   - Lines 228-247 (onBotTtsText deduplication)

2. `client/src/utils/__tests__/deduplication.test.ts` (new)
   - Comprehensive test suite

3. `verify_dedup_fix.js` (new)
   - Verification script

## Related Issues

- FastTextAggregator clause boundary splitting (server/core/aggregators/fast_text.py:227-231)
- Text cleaning strips leading spaces (server/core/aggregators/fast_text.py:93)
- Context aggregator concatenation without spaces (pipecat llm_response.py:1008)
- Client deduplication not handling whitespace variations

## Future Considerations

### Option 1: Fix Server-Side (Not Recommended)
- Modify FastTextAggregator to preserve leading spaces
- Update context aggregator to add spaces when concatenating
- **Risk:** May affect TTS voice naturalness and other downstream consumers

### Option 2: Client-Side Fix (IMPLEMENTED)
- Normalize text before comparison
- **Benefits:** No risk to TTS quality, handles ALL whitespace variations
- **Status:** ✅ Implemented and tested

### Option 3: Disable Clause Splitting (Not Recommended)
- Increase `min_words` threshold to prevent splitting at commas
- **Risk:** May create overly long TTS chunks, affecting voice quality

## Conclusion

The duplicate text bug was caused by a multi-stage pipeline issue:
1. Text aggregator splits at clause boundaries
2. Text cleaning removes leading spaces
3. Context aggregator concatenates without adding spaces
4. Client receives both "memory, able" and "memory,able"
5. Old deduplication fails to detect them as duplicates

**Fix:** Improved client-side deduplication with whitespace normalization successfully prevents all duplicate variations while preserving server-side TTS quality.
