# Memory System Architecture Map

## Overview

LocalCat's memory system provides ultra-low latency (<200ms) memory extraction, retrieval, and confidence scoring with TRUE audio-aware confidence based on how users actually speak.

**Key Innovation**: Prosody-aware confidence scoring using audio intelligence to measure user certainty from speech patterns (pitch, energy, speaking rate) rather than arbitrary rule-based confidence.

## System Components

### 1. Frame Processing Pipeline

```
Audio Input → AudioIntelligenceProcessor → MemoryFrameProcessor → MemoryStore
                     ↓                              ↓
            AudioIntelligenceFrame            Prosody Storage
              (prosody data)                   (turn_meta)
```

#### AudioIntelligenceProcessor
**Location**: `core/audio/audio_intelligence.py`

**Purpose**: Extracts prosody features from audio for TRUE confidence scoring

**Key Features**:
- **Dual-mode operation**: Speaker recognition and prosody analysis operate independently
- **Privacy-preserving**: Can disable speaker recognition while keeping prosody active
- **Session 3 enhancement**: Added prosody certainty calculation from audio features

**Prosody Extraction**:
```python
# Always extracts prosody features, regardless of speaker recognition state
prosody_features = self._prosody_analyzer.extract(audio_array)
prosody_certainty = prosody_features.certainty_modifier  # Range: 0.0-1.0
```

**Control Methods**:
- `set_enabled(bool)`: Enable/disable ALL audio processing
- `set_speaker_recognition_enabled(bool)`: Toggle speaker recognition while keeping prosody active

**Frame Emission**:
- Emits `AudioIntelligenceFrame` with prosody data on every utterance
- Even when speaker recognition is disabled, prosody frames are still emitted

#### MemoryFrameProcessor
**Location**: `core/memory/frame_processor.py`

**Purpose**: Routes frames through memory pipeline and captures prosody data

**Key Operations**:
1. **Prosody Capture** (lines 170-177):
   ```python
   if AUDIO_INTEL_AVAILABLE and isinstance(frame, AudioIntelligenceFrame):
       if hasattr(frame, 'prosody_certainty'):
           self.capture_prosody_certainty(frame.prosody_certainty)
           logger.debug(f"Captured prosody: {frame.prosody_certainty:.3f}")
   ```

2. **Prosody Storage** (lines 258-292):
   ```python
   self.hot_memory.store.set_turn_prosody(
       session_id, turn_id,
       self._last_prosody_certainty,
       meta
   )
   ```

3. **Memory Extraction**: Processes transcriptions through HotMemory for fact extraction

4. **Context Injection**: Retrieves and injects memory bullets into LLM context

### 2. Memory Storage Layer

```
SQLite (memory.db)
├── graph_triples: Fact triples with confidence
├── edge_sources: Source mentions for facts
├── turn_meta: Prosody certainty per turn (NEW)
└── session_meta: Session metadata

LMDB (graph.lmdb)
└── Entity graph with typed relationships
```

#### Prosody Storage
**Table**: `turn_meta`

**Schema**:
```sql
CREATE TABLE turn_meta (
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    metadata TEXT,
    PRIMARY KEY (session_id, turn_id, key)
)
```

**Storage Pattern**:
```python
# Store prosody certainty with metadata
store.set_turn_prosody(
    session_id="session_234",
    turn_id=5,
    certainty=0.85,  # High certainty - user spoke assertively
    meta={"source": "frame_processor", "captured_at": 1697123456789}
)

# Retrieve for confidence scoring
certainty, meta = store.get_turn_prosody(session_id, turn_id)
```

### 3. Confidence Scoring System

**Location**: `core/memory/confidence_strategy.py`

#### Strategy Pattern

Three confidence strategies available:

1. **RelationTypeConfidence** (Baseline)
   - Static rules based on relation type
   - 0.95 for names, 0.85 for verbs, 0.9 for others
   - No learning or adaptation

2. **UsageBasedConfidence** (Structural)
   - Learns from usage patterns
   - Factors: reinforcement, recency, source count
   - No prosody awareness

3. **ProsodyAwareConfidence** (TRUE Confidence) ✨
   - **Session 3 enhancement**: Uses actual audio features
   - Combines prosody + linguistic + usage signals
   - Replaces arbitrary confidence with measured certainty

#### ProsodyAwareConfidence Implementation

**Primary Scoring** (with inline prosody):
```python
def score(self, edge: Edge, context: Context) -> float:
    if self.fusion and context.prosody_features:
        # Calculate fusion confidence from audio + text
        fusion_conf = self.fusion.calculate(
            relation=edge.rel,
            text=context.text,
            prosody=context.prosody_features,  # Pitch, energy, rate
            emotion=context.emotion,
            arousal=context.arousal
        )

        # Apply usage-based adjustments
        reinforcement = self._reinforcement_multiplier(edge)
        recency = self._recency_multiplier(edge)
        source_count = self._source_count_multiplier(edge, context)

        return fusion_conf * reinforcement * recency * source_count
```

**Fallback Scoring** (with stored prosody - lines 252-347):
```python
def _score_with_stored_prosody(self, edge: Edge, context: Context) -> float:
    # Retrieve stored prosody from turn_meta
    certainty, meta = context.store.get_turn_prosody(
        context.session_id,
        context.turn_id
    )

    # Create synthetic prosody features from stored certainty
    certainty_modifier = max(-0.3, min(0.3, certainty - 0.5))
    synthetic_prosody = SyntheticProsodyFeatures(certainty_modifier)

    # Re-use fusion logic with synthetic prosody
    fusion_conf = self.fusion.calculate(
        relation=edge.rel,
        text=context.text,
        prosody=synthetic_prosody,
        emotion=context.emotion,
        arousal=context.arousal
    )

    return fusion_conf * reinforcement * recency * source_count
```

**Confidence Modifiers**:
- High certainty (> 0.7): Boosts confidence by up to 15%
- Neutral certainty (~0.5): No adjustment
- Low certainty (< 0.3): Reduces confidence by up to 15%

### 4. Retrieval & Reranking

**Location**: `core/memory/retrieval.py`

#### Composite Scoring

Memory bullets are ranked using weighted composite scores:

```python
composite_score = (
    MEMORY_WEIGHT_GRAPH * graph_score +      # 0.3: Graph importance
    MEMORY_WEIGHT_CONVO * convo_score +      # 0.5: Conversation relevance
    MEMORY_WEIGHT_SUMMARY * summary_score +  # 0.2: Summary importance
    MEMORY_WEIGHT_PROSODY * prosody_score    # 0.15: Prosody certainty
)
```

**Component Breakdown** (visible in logs):
```
[Retrieval] Top-1 candidate components:
{
  'wsrc': 0.3,      # Source weight
  'wconf': 0.308,   # Content confidence
  'wrec': 0.25,     # Recency
  'wuse': 0.112,    # Usage count
  'wsim': 0.0,      # Semantic similarity
  'wpro': 0.075,    # Prosody certainty (0.5 * 0.15)
  'wdiv': -0.035    # Diversity penalty
}
```

**Prosody Weight Calculation**:
```python
wpro = stored_prosody_certainty * MEMORY_WEIGHT_PROSODY
# Example: 0.85 (high certainty) * 0.15 (weight) = 0.1275
# Example: 0.50 (neutral)       * 0.15 (weight) = 0.0750
# Example: 0.20 (uncertain)     * 0.15 (weight) = 0.0300
```

### 5. Context Injection

**Location**: `core/memory/context_injector.py`

**Injection Format** (bullets mode):
```markdown
Use the following factual context if helpful.

MEMORY FORMAT GUIDE:
• [conf=X]: Content confidence (0.0-1.0) - reliability of this fact
• [pro=X]: Prosody certainty (0.0-1.0) - how confidently the user spoke
  * pro > 0.7 = User was very certain (assertive, clear)
  * pro ~0.5 = User was neutral/conversational
  * pro < 0.3 = User was uncertain (hedging, questioning)
• [rec=X]: Recency (1.0=just now, 0.0=very old)

Use prosody to match user's confidence level in your responses.

• [graph] you has dog (0s ago)
• [graph] you talk to dog (1h ago)
• [convo] Well PipeCat is pretty extensive... (46s ago)
```

**Configuration**:
- Injection role: `system` (via `MEMORY_INJECT_ROLE`)
- Max bullets: 5 (via `MEMORY_MAX_BULLETS`)
- Token budget: 600 tokens (via `MEMORY_TOKEN_BUDGET`)
- Injection mode: `bullets` or `headers` (via `MEMORY_INJECTION_MODE`)

## Data Flow: Complete Journey

### 1. Audio → Prosody Capture

```
User speaks "I love coffee" (assertive tone, strong pitch)
          ↓
AudioIntelligenceProcessor extracts:
  - pitch_std: 45.2 Hz (high variation = certainty)
  - energy_mean: 0.82 (strong = certainty)
  - speaking_rate: 3.8 words/sec (normal)
          ↓
ProsodyAnalyzer calculates certainty:
  certainty_modifier = 0.35 (boost due to assertive tone)
  prosody_certainty = 0.85 (0.5 + 0.35)
          ↓
AudioIntelligenceFrame emitted with prosody_certainty=0.85
```

### 2. Frame Processing → Storage

```
MemoryFrameProcessor receives AudioIntelligenceFrame
          ↓
capture_prosody_certainty(0.85)
  - Stores in self._last_prosody_certainty
          ↓
TranscriptionFrame arrives: "I love coffee"
          ↓
_store_prosody_for_turn() called
          ↓
SQLite: INSERT INTO turn_meta VALUES (
  'session_234', 5, 'prosody_certainty', '0.85',
  '{"source":"frame_processor","captured_at":1697123456789}'
)
```

### 3. Fact Extraction → Confidence

```
HotMemory.process_turn("I love coffee")
          ↓
Extract triple: (user, loves, coffee)
          ↓
Store with baseline confidence: 0.90 (verb relation)
          ↓
ProsodyAwareConfidence.score() called during retrieval
          ↓
Retrieve stored prosody: certainty=0.85
          ↓
Apply confidence boost: 0.90 * 1.15 = 1.035 → 1.0 (capped)
          ↓
Final confidence: 1.0 (high certainty fact)
```

### 4. Retrieval → Reranking

```
Query: "What do I like?"
          ↓
Candidate: (user, loves, coffee)
          ↓
Composite scoring:
  - wsrc:  0.30 (graph source)
  - wconf: 0.30 (baseline confidence 0.90 * 0.33)
  - wrec:  0.25 (just now)
  - wuse:  0.10 (first mention)
  - wpro:  0.13 (0.85 * 0.15) ← PROSODY BOOST
  - wdiv:  -0.03 (diversity)
  ────────────
  Total:   1.05 (high relevance)
```

### 5. Context Injection → LLM

```
Memory bullets prepared:
• [graph] user loves coffee (0s ago) [pro=0.85]
          ↓
Injected into LLM context as system message
          ↓
LLM sees user spoke with HIGH CERTAINTY (pro=0.85)
          ↓
LLM response matches confidence level:
"You definitely love coffee!" (assertive response)
```

## Session 3: Prosody Integration Fix

**Problem**: AudioIntelligence was completely disabled after speaker recognition, preventing prosody capture.

**Root Cause**: `EnrollmentCoordinator` called `set_enabled(False)` which disabled ALL audio processing including prosody.

**Solution** (October 2025):

### Code Changes

1. **AudioIntelligenceProcessor** (`audio_intelligence.py`):
   - Added `_speaker_recognition_enabled` flag separate from `_enabled`
   - Added `set_speaker_recognition_enabled()` method
   - Modified `_process_utterance()` to always extract prosody (lines 427-456)
   - Emits `AudioIntelligenceFrame` with prosody even when speaker recognition disabled

2. **EnrollmentCoordinator** (`enrollment_coordinator.py`):
   - Line 353-354: Changed `set_enabled(False)` → `set_speaker_recognition_enabled(False)`
   - Line 400-402: Same change after enrollment completion
   - Preserves prosody extraction while disabling speaker recognition for privacy

### Behavioral Changes

**Before**:
- Returning user recognized → ALL audio intelligence disabled
- No prosody extraction → stuck at neutral default (0.5)
- wpro values always 0.075 (0.5 * 0.15)
- Confidence scoring degraded to arbitrary baseline

**After**:
- Returning user recognized → Only speaker recognition disabled
- Prosody extraction continues → varies based on actual speech
- wpro values range 0.03-0.15 (0.2-1.0 certainty * 0.15 weight)
- TRUE confidence scoring based on how user actually speaks

### Verification

**Log Indicators** (successful prosody capture):
```
[AudioIntel] Speaker recognition DISABLED (prosody still active)
[AudioIntel] Prosody: ProsodyFeatures(pitch_std=45.2, energy=0.82, rate=3.8)
[AudioIntel] Emitted prosody-only frame (speaker recognition disabled)
[FrameProcessor] Captured prosody from AudioIntelligenceFrame: 0.850
[FrameProcessor] Stored prosody certainty 0.850 for session=234, turn=5
```

**Database Verification**:
```sql
-- Check prosody storage
SELECT session_id, turn_id, value as certainty, metadata
FROM turn_meta
WHERE key = 'prosody_certainty'
ORDER BY turn_id DESC
LIMIT 10;

-- Should show varied certainty values (not all 0.5)
-- Example output:
-- session_234, 5, 0.850, {"source":"frame_processor"}
-- session_234, 4, 0.450, {"source":"frame_processor"}
-- session_234, 3, 0.720, {"source":"frame_processor"}
```

**Retrieval Component Logs**:
```
[Retrieval] Top-1 candidate components:
  'wpro': 0.128  ← HIGH (0.85 * 0.15) - assertive speech

[Retrieval] Top-2 candidate components:
  'wpro': 0.068  ← LOW (0.45 * 0.15) - uncertain speech

[Retrieval] Top-3 candidate components:
  'wpro': 0.108  ← MEDIUM (0.72 * 0.15) - confident speech
```

## Configuration

### Environment Variables

**Memory System**:
```bash
# Core settings
MEMORY_ENABLED=true
MEMORY_HOTPATH_ENABLED=true
MEMORY_BULLETS_MAX=3
MEMORY_TOKEN_BUDGET=600

# Source weights for composite scoring
MEMORY_WEIGHT_GRAPH=0.3          # Graph fact importance
MEMORY_WEIGHT_CONVO=0.5          # Conversation history
MEMORY_WEIGHT_SUMMARY=0.2        # Summary importance
MEMORY_WEIGHT_PROSODY=0.15       # Prosody certainty (NEW)

# Confidence strategy
CONFIDENCE_STRATEGY=prosody_aware  # relation_type | usage_based | prosody_aware

# Storage paths
MEMORY_SQLITE_PATH=../data/memory.db
MEMORY_LMDB_PATH=../data/graph.lmdb

# Injection mode
MEMORY_INJECTION_MODE=bullets      # bullets | headers
MEMORY_INJECT_ROLE=system
```

**Audio Intelligence**:
```bash
# Audio processing
AUDIO_INTELLIGENCE_ENABLED=true
AUDIO_INTEL_USE_MPS=true           # Use Apple Silicon GPU
AUDIO_INTEL_ENABLE_EMOTION=false   # Emotion detection (optional)
AUDIO_INTEL_ENABLE_PROSODY=true    # Prosody analysis (REQUIRED)

# Speaker recognition
SPEAKER_PROFILE_DIR=data/speaker_profiles
SPEAKER_SIMILARITY_THRESHOLD=0.55
SPEAKER_AUTO_ENROLL_UTTERANCES=3
```

## Performance Characteristics

### Latency Breakdown

```
Total Memory Pipeline: <200ms target

1. Audio → Prosody:        ~10-20ms  (real-time analysis)
2. Prosody Storage:        ~1-2ms    (SQLite write)
3. Fact Extraction:        ~50-100ms (HotPath extraction)
4. Retrieval + Rerank:     ~20-50ms  (composite scoring)
5. Context Injection:      ~5-10ms   (bullet formatting)
                          ──────────
Total:                     ~86-182ms (within target)
```

### Storage Overhead

```
Per Turn:
- Prosody metadata:     ~100 bytes (turn_meta row)
- Fact triples:         ~200-500 bytes (graph_triples)
- Conversation text:    ~100-1000 bytes (FTS index)
                       ─────────────
Total per turn:        ~400-1600 bytes

Per Session (20 turns): ~8-32 KB
Per Month (600 turns):  ~240-960 KB
```

## Architecture Principles

### 1. Separation of Concerns

- **AudioIntelligenceProcessor**: Audio feature extraction only
- **MemoryFrameProcessor**: Frame routing and prosody capture
- **MemoryStore**: Data persistence
- **ConfidenceStrategy**: Scoring logic
- **Retrieval**: Query and ranking

### 2. Strategy Pattern

Confidence scoring uses strategy pattern for flexibility:
```python
# Pluggable strategies
strategy = create_confidence_strategy("prosody_aware")
confidence = strategy.score(edge, context)
```

### 3. Privacy by Design

- Speaker recognition can be disabled independently
- Prosody features don't identify individuals
- All data stored locally (no external APIs)
- Ephemeral mode available (no storage)

### 4. Performance First

- Prosody extraction: Real-time during speech
- Storage: Write-ahead for minimal latency
- Retrieval: Indexed queries with composite scoring
- Caching: Turn-level prosody cache

## Testing

### Unit Tests

```bash
# Test prosody storage
pytest tests/unit/test_prosody_confidence.py

# Test confidence strategies
pytest tests/unit/test_confidence_strategy.py

# Test retrieval with prosody weighting
pytest tests/unit/test_prosody_rerank.py
```

### Integration Tests

```bash
# Test end-to-end prosody capture
pytest tests/integration/test_prosody_capture.py

# Test frame processing pipeline
pytest tests/integration/test_frame_fix.py
```

### Manual Verification

```python
# 1. Check prosody capture in logs
grep "Captured prosody" data/logs.log

# 2. Query stored prosody
sqlite3 data/memory.db "SELECT * FROM turn_meta WHERE key='prosody_certainty'"

# 3. Verify component weights
grep "wpro" data/logs.log | tail -20
```

## Troubleshooting

### Prosody Not Capturing

**Symptoms**:
- No "Captured prosody" messages in logs
- wpro values stuck at 0.075 (neutral default)
- All pro values show 0.50

**Checks**:
```bash
# 1. Verify audio intelligence is enabled
grep "AUDIO_INTEL_ENABLE_PROSODY" .env
# Should show: AUDIO_INTEL_ENABLE_PROSODY=true

# 2. Check if speaker recognition disabled prosody
grep "set_enabled\|set_speaker_recognition_enabled" data/logs.log
# Should show: "Speaker recognition DISABLED (prosody still active)"

# 3. Verify prosody analyzer initialized
grep "Prosody analyzer initialized" data/logs.log
```

**Solutions**:
1. Enable prosody in `.env`: `AUDIO_INTEL_ENABLE_PROSODY=true`
2. Update `EnrollmentCoordinator` to use `set_speaker_recognition_enabled(False)`
3. Check ProsodyAnalyzer dependencies: `pip install numpy librosa`

### Low Prosody Variation

**Symptoms**:
- Prosody values all similar (0.45-0.55)
- Little impact on confidence scores

**Causes**:
- Monotone speech input
- Poor audio quality (background noise)
- Incorrect sample rate (should be 16kHz)

**Solutions**:
1. Test with varied speech (whisper vs. assertive)
2. Improve microphone quality
3. Verify audio sample rate in config

### Storage Failures

**Symptoms**:
- "Failed to store prosody" warnings in logs
- Empty turn_meta table

**Checks**:
```bash
# 1. Verify database file writable
ls -la data/memory.db

# 2. Check database schema
sqlite3 data/memory.db ".schema turn_meta"

# 3. Test write permissions
sqlite3 data/memory.db "INSERT INTO turn_meta VALUES ('test', 1, 'test', '0.5', '{}')"
```

## Future Enhancements

### Planned Improvements

1. **Multi-modal Prosody**:
   - Video: Facial expressions + gestures
   - Text: Punctuation + capitalization patterns
   - Fusion: Combined audio + video + text certainty

2. **Adaptive Weighting**:
   - Learn optimal wpro weight per user
   - Adjust based on speaker characteristics
   - A/B test different weight configurations

3. **Prosody Patterns**:
   - Detect user-specific certainty patterns
   - Learn "hedge words" per individual
   - Personalized confidence thresholds

4. **Real-time Feedback**:
   - Show prosody certainty to user during speech
   - Visual indicator (green=certain, yellow=neutral, red=uncertain)
   - Help users calibrate their expression

## References

### Code Locations

- **Audio Intelligence**: `core/audio/audio_intelligence.py`
- **Frame Processor**: `core/memory/frame_processor.py`
- **Memory Store**: `core/memory/memory_store.py`
- **Confidence Strategy**: `core/memory/confidence_strategy.py`
- **Retrieval**: `core/memory/retrieval.py`
- **Context Injector**: `core/memory/context_injector.py`
- **Enrollment Coordinator**: `core/audio/enrollment_coordinator.py`

### Related Documentation

- **Memory Usage Guide**: `docs/07-guides/memory-usage.md`
- **Coreference Guide**: `docs/07-guides/coreference.md`
- **Session Tracking Fix**: `docs/09-reports/investigations/session-tracking-fix-summary.md`

### Configuration Files

- **Environment**: `.env` (server root)
- **Example Config**: `env.example` (reference)

---

**Last Updated**: October 2025
**Status**: ✅ Prosody integration complete and tested
**Version**: Session 3 (TRUE Confidence)
