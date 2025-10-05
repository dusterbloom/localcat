# Multi-Signal Confidence System

**Purpose**: Detailed specification of the confidence fusion algorithm that combines prosody, linguistic, contextual, and usage signals to produce TRUE confidence scores for memory facts.

**Goal**: Replace arbitrary confidence (0.85-0.95) with evidence-based confidence from multiple voice and text signals.

---

## System Overview

### Current Problem

```python
# memory_hotpath.py:164-169 (WRONG)
if r == "name":
    conf = 0.95  # Arbitrary
elif r.startswith("v:"):
    conf = 0.85  # Made up
else:
    conf = 0.9   # Guessing
```

**Both get 0.9**:
- ✅ "My NAME is ALICE!" (emphatic, certain)
- ❌ "I... think... maybe Alice?" (hesitant, uncertain)

### Proposed Solution

```python
# Enhanced confidence system (RIGHT)
confidence = ConfidenceFusion.calculate(
    relation="name",
    text="My name is Alice",
    prosody=prosody_features,      # Voice signals
    linguistic=linguistic_features, # Text certainty markers
    contextual=context_info,        # Session, topic context
    usage=usage_stats               # Historical retrieval success
)

# Result: TRUE confidence based on EVIDENCE
# "My NAME is ALICE!" → 0.95 (high certainty from prosody + linguistic)
# "I think maybe Alice?" → 0.35 (low certainty from hedge + uncertain tone)
```

---

## Signal Taxonomy

### 1. Base Confidence (Fact Type)

**Weight**: 25% of final confidence

**Rationale**: Different fact types have different permanence and certainty

**Mapping**:

| Fact Type | Base Confidence | Reasoning |
|-----------|----------------|-----------|
| **Name** | 0.90 | Names are usually permanent, rarely change |
| **Identity** (age, birthday) | 0.85 | Permanent but may be withheld initially |
| **Location** (lives_in, works_at) | 0.70 | Can change frequently (move, job change) |
| **Relationship** (has, friend_of) | 0.75 | Relatively stable but can change |
| **Preference** (likes, favorite) | 0.60 | Changes over time, subjective |
| **Temporary state** (feels, wants) | 0.40 | Highly volatile |
| **Action/verb** (v:*) | 0.70 | Depends on tense, context |
| **Default** | 0.65 | Unknown fact types |

**Implementation**:

```python
def get_base_confidence(relation: str, text_span: str) -> float:
    """Get base confidence from fact type."""

    # Explicit name relations
    if relation == "name":
        return 0.90

    # Identity facts
    if relation in {"age", "birthday", "born_in", "born_on"}:
        return 0.85

    # Location facts
    if relation in {"lives_in", "works_at", "from", "located_in"}:
        return 0.70

    # Relationships
    if relation in {"has", "owns", "friend_of", "parent_of", "child_of"}:
        return 0.75

    # Preferences
    if relation in {"likes", "dislikes", "prefers", "favorite", "hates"}:
        return 0.60

    # Temporary states
    if relation in {"feels", "wants", "needs", "thinking"}:
        return 0.40

    # Actions/verbs
    if relation.startswith("v:"):
        # Check for temporal markers
        if any(word in text_span.lower() for word in ["always", "never", "usually"]):
            return 0.75  # Habitual
        elif any(word in text_span.lower() for word in ["now", "currently", "today"]):
            return 0.50  # Temporary
        else:
            return 0.70  # General action

    # Default
    return 0.65
```

---

### 2. Prosody Confidence (Voice Signals)

**Weight**: 35% of final confidence

**Rationale**: HOW user says something reveals certainty, emphasis, emotion

**Components**:

#### 2.1 Pitch Contour (25% of prosody weight)

```python
def calculate_pitch_confidence(prosody: ProsodyFeatures) -> float:
    """Confidence from pitch patterns."""
    confidence = 0.5  # Neutral baseline

    # Pitch slope analysis
    pitch_slope = prosody.mean_pitch_slope

    if pitch_slope < -15:  # Falling (statement)
        confidence += 0.20
    elif pitch_slope > 15:  # Rising (question)
        confidence -= 0.20
    elif -5 < pitch_slope < 5:  # Flat (bored/uncertain)
        confidence -= 0.05

    # Pitch variability (engagement)
    pitch_range = prosody.pitch_max - prosody.pitch_min
    speaker_baseline_range = prosody.speaker_pitch_range

    if pitch_range > speaker_baseline_range * 1.5:  # Animated
        confidence += 0.10
    elif pitch_range < speaker_baseline_range * 0.5:  # Monotone
        confidence -= 0.10

    return max(0.0, min(1.0, confidence))
```

#### 2.2 Energy/Emphasis (30% of prosody weight)

```python
def calculate_energy_confidence(prosody: ProsodyFeatures) -> float:
    """Confidence from energy patterns."""
    confidence = 0.5

    # Peak energy on keywords
    peak_energy = prosody.max_energy
    mean_energy = prosody.mean_energy

    if peak_energy > mean_energy * 1.5:  # Strong emphasis
        confidence += 0.15
    elif peak_energy < mean_energy * 0.8:  # Subdued
        confidence -= 0.10

    # Overall energy level (relative to speaker baseline)
    if prosody.overall_energy > prosody.speaker_baseline_energy * 1.2:
        confidence += 0.10  # Louder = more confident
    elif prosody.overall_energy < prosody.speaker_baseline_energy * 0.7:
        confidence -= 0.15  # Quieter = less confident

    return max(0.0, min(1.0, confidence))
```

#### 2.3 Speaking Rate/Fluency (25% of prosody weight)

```python
def calculate_fluency_confidence(prosody: ProsodyFeatures) -> float:
    """Confidence from speaking rate and fluency."""
    confidence = 0.5

    # Speaking rate analysis
    rate = prosody.syllables_per_second

    if rate > 5.0:  # Fast, fluent
        confidence += 0.20
    elif rate < 2.5:  # Slow, hesitant
        confidence -= 0.20

    # Pause analysis
    pause_count = len(prosody.pauses)
    total_pause_duration = sum(p.duration_ms for p in prosody.pauses)

    if pause_count == 0 and total_pause_duration == 0:
        confidence += 0.15  # No pauses = fluent
    elif pause_count > 3 or total_pause_duration > 1000:
        confidence -= 0.25  # Many/long pauses = uncertain

    # Filled pauses ("um", "uh")
    filler_count = prosody.filler_count
    if filler_count > 2:
        confidence -= 0.15 * filler_count  # Heavy penalty for fillers

    return max(0.0, min(1.0, confidence))
```

#### 2.4 Voice Quality (20% of prosody weight)

```python
def calculate_voice_quality_confidence(prosody: ProsodyFeatures) -> float:
    """Confidence from voice quality features."""
    confidence = 0.5

    # Harmonics-to-Noise Ratio (HNR)
    # Clear voice = confident, breathy/creaky = uncertain
    hnr = prosody.hnr

    if hnr > 20:  # Very clear voice
        confidence += 0.10
    elif hnr < 10:  # Breathy or noisy
        confidence -= 0.10

    # Jitter/shimmer (pitch/amplitude irregularity)
    # Low = stable, high = tense/uncertain
    jitter = prosody.jitter

    if jitter < 0.01:  # Stable pitch
        confidence += 0.05
    elif jitter > 0.03:  # Irregular pitch
        confidence -= 0.10

    return max(0.0, min(1.0, confidence))
```

#### Combined Prosody Confidence

```python
def calculate_prosody_confidence(prosody: ProsodyFeatures) -> float:
    """Weighted combination of prosody features."""

    pitch_conf = calculate_pitch_confidence(prosody)
    energy_conf = calculate_energy_confidence(prosody)
    fluency_conf = calculate_fluency_confidence(prosody)
    quality_conf = calculate_voice_quality_confidence(prosody)

    # Weighted average
    prosody_confidence = (
        pitch_conf * 0.25 +
        energy_conf * 0.30 +
        fluency_conf * 0.25 +
        quality_conf * 0.20
    )

    return prosody_confidence
```

---

### 3. Linguistic Confidence (Text Signals)

**Weight**: 25% of final confidence

**Rationale**: Text contains explicit certainty markers

**Components**:

#### 3.1 Hedge Words (40% of linguistic weight)

```python
HEDGE_WORDS = {
    "maybe": -0.25,
    "perhaps": -0.20,
    "possibly": -0.20,
    "probably": -0.15,
    "might": -0.20,
    "could": -0.15,
    "I think": -0.20,
    "I believe": -0.15,
    "I guess": -0.25,
    "I suppose": -0.20,
    "sort of": -0.15,
    "kind of": -0.15,
    "like": -0.10,  # As hedge, not as verb
}

def calculate_hedge_confidence(text: str) -> float:
    """Confidence reduction from hedge words."""
    confidence = 0.0
    text_lower = text.lower()

    for hedge, penalty in HEDGE_WORDS.items():
        if hedge in text_lower:
            confidence += penalty

    return confidence  # Returns negative adjustment
```

#### 3.2 Certainty Markers (40% of linguistic weight)

```python
CERTAINTY_MARKERS = {
    "definitely": +0.20,
    "certainly": +0.20,
    "absolutely": +0.25,
    "always": +0.15,
    "never": +0.15,
    "clearly": +0.15,
    "obviously": +0.15,
    "undoubtedly": +0.20,
    "sure": +0.15,
    "positive": +0.15,
    "know for sure": +0.25,
    "100%": +0.25,
}

def calculate_certainty_confidence(text: str) -> float:
    """Confidence boost from certainty markers."""
    confidence = 0.0
    text_lower = text.lower()

    for marker, boost in CERTAINTY_MARKERS.items():
        if marker in text_lower:
            confidence += boost

    return confidence  # Returns positive adjustment
```

#### 3.3 Question Detection (20% of linguistic weight)

```python
def calculate_question_confidence(text: str) -> float:
    """Confidence reduction if phrased as question."""
    confidence = 0.0

    # Ends with question mark
    if text.strip().endswith("?"):
        confidence -= 0.25

    # Starts with wh-word
    wh_words = ["who", "what", "when", "where", "why", "how", "which"]
    first_word = text.strip().split()[0].lower()
    if first_word in wh_words:
        confidence -= 0.20

    return confidence
```

#### Combined Linguistic Confidence

```python
def calculate_linguistic_confidence(text: str) -> float:
    """Combined linguistic certainty score."""

    hedge_conf = calculate_hedge_confidence(text)
    certainty_conf = calculate_certainty_confidence(text)
    question_conf = calculate_question_confidence(text)

    # Start at neutral (0.5)
    linguistic_confidence = 0.5

    # Apply adjustments (weighted)
    linguistic_confidence += hedge_conf * 0.40
    linguistic_confidence += certainty_conf * 0.40
    linguistic_confidence += question_conf * 0.20

    return max(0.0, min(1.0, linguistic_confidence))
```

---

### 4. Contextual Confidence (Context Signals)

**Weight**: 10% of final confidence

**Rationale**: Context provides additional certainty signals

**Components**:

#### 4.1 Temporal Context

```python
def calculate_temporal_confidence(text: str) -> float:
    """Confidence from temporal markers."""
    confidence = 0.5
    text_lower = text.lower()

    # Permanent markers
    permanent_markers = ["always", "forever", "never", "since birth"]
    if any(m in text_lower for m in permanent_markers):
        confidence += 0.20

    # Temporary markers
    temporary_markers = ["now", "currently", "today", "right now", "at the moment"]
    if any(m in text_lower for m in temporary_markers):
        confidence -= 0.15

    # Past markers (less certain)
    past_markers = ["used to", "was", "were", "back then"]
    if any(m in text_lower for m in past_markers):
        confidence -= 0.10

    return max(0.0, min(1.0, confidence))
```

#### 4.2 Correction Context

```python
def calculate_correction_confidence(text: str, prosody: ProsodyFeatures,
                                   history: ConversationHistory) -> float:
    """Detect if this is a correction (high confidence replacement)."""

    # Linguistic correction markers
    correction_markers = ["no", "not", "wrong", "actually", "I meant",
                         "correction", "sorry", "mistake", "oops"]

    has_correction_marker = any(m in text.lower() for m in correction_markers)

    # Prosodic correction signals
    has_emphatic_prosody = (
        prosody.max_energy > prosody.mean_energy * 1.8 or
        prosody.pitch_delta > 40 or
        prosody.arousal > 0.7
    )

    # Check if contradicts recent fact
    contradicts_recent = history.contradicts_recent_fact(text)

    if has_correction_marker and has_emphatic_prosody:
        return 0.98  # STRONG CORRECTION
    elif has_correction_marker and contradicts_recent:
        return 0.90  # CORRECTION
    else:
        return 0.5  # Not a correction
```

#### 4.3 Response Latency

```python
def calculate_latency_confidence(latency_ms: float) -> float:
    """Confidence from response speed."""
    confidence = 0.5

    if latency_ms < 500:  # Immediate response
        confidence += 0.15
    elif 500 <= latency_ms < 1000:  # Normal
        confidence += 0.0
    elif 1000 <= latency_ms < 2000:  # Thinking
        confidence -= 0.10
    else:  # Long delay (> 2s)
        confidence -= 0.20

    return max(0.0, min(1.0, confidence))
```

#### Combined Contextual Confidence

```python
def calculate_contextual_confidence(text: str, prosody: ProsodyFeatures,
                                   context: ConversationContext) -> float:
    """Combined contextual signals."""

    temporal_conf = calculate_temporal_confidence(text)
    correction_conf = calculate_correction_confidence(text, prosody, context.history)
    latency_conf = calculate_latency_confidence(context.response_latency_ms)

    # Weighted average
    contextual_confidence = (
        temporal_conf * 0.30 +
        correction_conf * 0.50 +  # Corrections are strong signal
        latency_conf * 0.20
    )

    return contextual_confidence
```

---

### 5. Usage Confidence (Historical Feedback)

**Weight**: 5% of final confidence

**Rationale**: Facts that are useful when retrieved have higher true confidence

**Components**:

#### 5.1 Retrieval Success Rate

```python
def calculate_usage_confidence(fact: Tuple[str, str, str],
                               usage_stats: UsageStats) -> float:
    """Confidence from historical usage."""

    # Get usage statistics for this fact
    times_injected = usage_stats.get_injection_count(fact)
    times_used = usage_stats.get_usage_count(fact)  # Based on LLM behavior analysis

    if times_injected == 0:
        return 0.5  # No usage data yet

    # Calculate success rate
    success_rate = times_used / times_injected

    # Map to confidence adjustment
    if success_rate > 0.8:  # Frequently used
        return 0.9
    elif success_rate > 0.5:  # Sometimes used
        return 0.7
    elif success_rate > 0.2:  # Rarely used
        return 0.4
    else:  # Almost never used
        return 0.2
```

#### 5.2 Reinforcement/Contradiction History

```python
def calculate_reinforcement_confidence(fact: Tuple[str, str, str],
                                      store: MemoryStore) -> float:
    """Confidence from reinforcement history."""

    # Get edge statistics
    edge = store.get_edge(*fact)

    pos_count = edge.positive_count
    neg_count = edge.negative_count

    # Calculate reinforcement ratio
    total = pos_count + neg_count
    if total == 0:
        return 0.5

    reinforcement_ratio = pos_count / total

    # Map to confidence
    confidence = 0.3 + (reinforcement_ratio * 0.7)  # Range: 0.3 - 1.0

    return confidence
```

#### Combined Usage Confidence

```python
def calculate_usage_confidence_combined(fact: Tuple[str, str, str],
                                       usage_stats: UsageStats,
                                       store: MemoryStore) -> float:
    """Combined usage-based confidence."""

    usage_conf = calculate_usage_confidence(fact, usage_stats)
    reinforcement_conf = calculate_reinforcement_confidence(fact, store)

    # Weighted average
    return usage_conf * 0.6 + reinforcement_conf * 0.4
```

---

## Complete Fusion Algorithm

### Master Confidence Calculation

```python
class ConfidenceFusion:
    """Multi-signal confidence fusion system."""

    def __init__(self, store: MemoryStore, usage_stats: UsageStats):
        self.store = store
        self.usage_stats = usage_stats

    def calculate(self,
                 relation: str,
                 text_span: str,
                 prosody: ProsodyFeatures,
                 context: ConversationContext,
                 fact: Tuple[str, str, str]) -> float:
        """
        Calculate TRUE confidence from all available signals.

        Args:
            relation: Relation type (e.g., "name", "lives_in")
            text_span: Text containing the fact
            prosody: Prosody features from audio
            context: Conversation context (history, latency, etc.)
            fact: The (subject, relation, object) triple

        Returns:
            Confidence score in [0.0, 1.0]
        """

        # 1. Base confidence from fact type (25%)
        base_conf = get_base_confidence(relation, text_span)

        # 2. Prosody confidence from voice (35%)
        prosody_conf = calculate_prosody_confidence(prosody)

        # 3. Linguistic confidence from text (25%)
        linguistic_conf = calculate_linguistic_confidence(text_span)

        # 4. Contextual confidence (10%)
        contextual_conf = calculate_contextual_confidence(
            text_span, prosody, context
        )

        # 5. Usage confidence from history (5%)
        usage_conf = calculate_usage_confidence_combined(
            fact, self.usage_stats, self.store
        )

        # Weighted fusion
        final_confidence = (
            base_conf * 0.25 +
            prosody_conf * 0.35 +
            linguistic_conf * 0.25 +
            contextual_conf * 0.10 +
            usage_conf * 0.05
        )

        # Clamp to valid range
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Log for debugging/GEPA learning
        self._log_confidence_calculation({
            'fact': fact,
            'text': text_span,
            'base': base_conf,
            'prosody': prosody_conf,
            'linguistic': linguistic_conf,
            'contextual': contextual_conf,
            'usage': usage_conf,
            'final': final_confidence
        })

        return final_confidence

    def _log_confidence_calculation(self, details: dict):
        """Log confidence calculation for analysis."""
        # Store for GEPA to learn from
        logger.debug(f"Confidence: {details['final']:.3f} = "
                    f"base({details['base']:.3f}) * 0.25 + "
                    f"prosody({details['prosody']:.3f}) * 0.35 + "
                    f"ling({details['linguistic']:.3f}) * 0.25 + "
                    f"ctx({details['contextual']:.3f}) * 0.10 + "
                    f"usage({details['usage']:.3f}) * 0.05")
```

---

## Example Calculations

### Example 1: High Confidence Statement

**Input**:
```
Audio: "My NAME is ALICE."
  Prosody:
    - Pitch slope: -22 Hz/s (falling)
    - Peak energy: 0.89 (emphatic)
    - Speaking rate: 5.1 syll/s (fast)
    - Pauses: 0
    - HNR: 23 (clear voice)

  Text: "My name is Alice"
  Relation: "name"
  Context:
    - Response latency: 380ms (quick)
    - No correction markers
    - First time stating this fact
```

**Calculation**:
```python
base_conf = 0.90      # Name relation
prosody_conf = 0.88   # Falling pitch, emphatic, fluent, clear
linguistic_conf = 0.50 # Neutral (no hedge or certainty markers)
contextual_conf = 0.65 # Quick response, temporal neutral
usage_conf = 0.50     # No history yet

final = (0.90 * 0.25) + (0.88 * 0.35) + (0.50 * 0.25) + (0.65 * 0.10) + (0.50 * 0.05)
      = 0.225 + 0.308 + 0.125 + 0.065 + 0.025
      = 0.748
```

**Result**: **0.75 confidence** ✅ (High, appropriate for emphatic statement)

---

### Example 2: Low Confidence Speculation

**Input**:
```
Audio: "I... think... maybe... I prefer blue?"
  Prosody:
    - Pitch slope: +18 Hz/s (rising, questioning)
    - Peak energy: 0.42 (quiet)
    - Speaking rate: 2.3 syll/s (slow, hesitant)
    - Pauses: 3 (total 850ms)
    - Fillers: 2 ("think", hesitation)
    - HNR: 14 (moderate)

  Text: "I think maybe I prefer blue"
  Relation: "prefers"
  Context:
    - Response latency: 1800ms (slow)
    - No correction
    - Question phrasing
```

**Calculation**:
```python
base_conf = 0.60      # Preference relation (changeable)
prosody_conf = 0.25   # Rising pitch, quiet, slow, many pauses
linguistic_conf = 0.15 # Hedge words ("think", "maybe"), question
contextual_conf = 0.35 # Slow response, uncertain temporal
usage_conf = 0.50     # No history

final = (0.60 * 0.25) + (0.25 * 0.35) + (0.15 * 0.25) + (0.35 * 0.10) + (0.50 * 0.05)
      = 0.150 + 0.088 + 0.038 + 0.035 + 0.025
      = 0.336
```

**Result**: **0.34 confidence** ✅ (Low, appropriate for uncertain speculation)

---

### Example 3: Strong Correction

**Input**:
```
Audio: "NO! I said ALICE, not Alicia!"
  Prosody:
    - Pitch spike: +45 Hz on "NO", "ALICE"
    - Peak energy: 0.95 (very loud, emphatic)
    - Speaking rate: 5.8 syll/s (fast, assertive)
    - Pauses: 0
    - Arousal: 0.85 (frustrated)

  Text: "No I said Alice not Alicia"
  Relation: "name"
  Context:
    - Response latency: 250ms (immediate)
    - Correction markers: "no", "not"
    - Contradicts recent fact ("Alicia")
```

**Calculation**:
```python
base_conf = 0.90      # Name relation
prosody_conf = 0.95   # Emphatic, fast, loud, high arousal
linguistic_conf = 0.50 # Neutral base (correction markers don't add certainty markers)
contextual_conf = 0.98 # STRONG CORRECTION DETECTED!
usage_conf = 0.50     # No history

final = (0.90 * 0.25) + (0.95 * 0.35) + (0.50 * 0.25) + (0.98 * 0.10) + (0.50 * 0.05)
      = 0.225 + 0.333 + 0.125 + 0.098 + 0.025
      = 0.806
```

**Result**: **0.81 confidence** ✅ (Very high due to correction context)

**Action**: Also triggers correction handling:
- negate_edge(you, name, "Alicia", conf=0.0)
- observe_edge(you, name, "Alice", conf=0.81)

---

## Integration with Memory System

### Modified Extraction Pipeline

```python
# memory_hotpath.py - Enhanced process_turn()

def process_turn(self, text: str, audio: np.ndarray,
                session_id: str, turn_id: int,
                word_timestamps: List[Tuple[str, float, float]]):
    """Process turn with multi-signal confidence."""

    # Existing extraction
    entities, triples, neg_count, doc = self.extractor.extract(text, lang)

    # NEW: Extract prosody features
    prosody_features = self.prosody_analyzer.extract(
        audio=audio,
        text=text,
        word_timestamps=word_timestamps
    )

    # NEW: Extract linguistic features
    linguistic_features = self.linguistic_analyzer.analyze(text)

    # NEW: Get conversation context
    context = self.get_conversation_context(session_id, turn_id)

    now_ts = int(time.time() * 1000)

    for s, r, d in triples:
        # OLD WAY (WRONG):
        # conf = 0.95 if r == "name" else 0.9

        # NEW WAY (RIGHT): Multi-signal confidence
        fact = (s, r, d)
        text_span = self.extract_text_span(text, s, r, d)

        # Get prosody for this specific fact/span
        span_prosody = prosody_features.get_span_features(text_span)

        # Calculate TRUE confidence
        conf = self.confidence_fusion.calculate(
            relation=r,
            text_span=text_span,
            prosody=span_prosody,
            context=context,
            fact=fact
        )

        # Store with TRUE confidence
        if not self._is_question(text):
            self.store.observe_edge(s, r, d, conf=conf, now_ts=now_ts)
```

### Modified Retrieval Pipeline

```python
# retrieval.py - Use confidence in ranking

def _graph_retrieve(self, query: str, entities: List[str],
                   turn_id: int, max_bullets: int, seen: set):
    """Retrieve facts ranked by confidence × priority × recency."""

    for entity in query_entities:
        if entity in self.host.entity_index:
            candidates = list(self.host.entity_index[entity])
            scored: List[Tuple[float, str, str, str]] = []

            for s, r, d in candidates:
                # Get edge from store
                edge = self.host.store.get_edge(s, r, d)

                # NEW: Use TRUE confidence (not ignored!)
                confidence = edge.weight

                # Priority from relation type
                priority = pred_pri.get(r, 50)

                # Recency factor
                age_seconds = (now_ts - edge.updated_at) / 1000
                recency_factor = math.exp(-age_seconds / (30 * 24 * 3600))  # 30-day decay

                # Combined score: confidence × priority × recency
                score = confidence * (priority / 100) * recency_factor

                scored.append((score, s, r, d))

            # Sort by combined score (highest first)
            scored.sort(key=lambda x: x[0], reverse=True)

            for score, s, r, d in scored:
                # Only inject high-confidence facts
                if confidence < 0.70:
                    continue  # Filter low-confidence

                fact = f"{s} {r} {d}"
                if fact not in seen:
                    human = self._humanize_fact(s, r, d)
                    if human:
                        out.append(f"• [graph] {human}")
                        seen.add(fact)

                        # Track injection for usage stats
                        self.usage_stats.record_injection(s, r, d)

                        if len(out) >= max_bullets:
                            return out
```

---

## GEPA Integration

### What GEPA Optimizes

With TRUE confidence, GEPA can now optimize:

1. **Fusion Weights**:
   - Current: 25% base, 35% prosody, 25% linguistic, 10% contextual, 5% usage
   - GEPA learns: Optimal weights per user, per fact type

2. **Prosody Thresholds**:
   - Current: pitch_slope < -15 → +0.20 confidence
   - GEPA learns: User-specific thresholds (some speak differently)

3. **Confidence Filtering**:
   - Current: Only inject facts > 0.70 confidence
   - GEPA learns: Optimal threshold from retrieval success

4. **Signal Importance**:
   - GEPA discovers which signals matter most
   - Example: For this user, energy matters more than pitch

### Trajectory Format for GEPA

```python
trajectory = {
    # Input
    "text": "My name is Alice",
    "audio": audio_data,
    "word_timestamps": [...],

    # Extracted signals
    "prosody": {
        "pitch_slope": -22,
        "peak_energy": 0.89,
        "rate": 5.1,
        "pauses": 0
    },
    "linguistic": {
        "hedge_count": 0,
        "certainty_count": 0,
        "question": False
    },

    # Confidence calculation
    "confidence_breakdown": {
        "base": 0.90,
        "prosody": 0.88,
        "linguistic": 0.50,
        "contextual": 0.65,
        "usage": 0.50,
        "final": 0.75,
        "weights": [0.25, 0.35, 0.25, 0.10, 0.05]
    },

    # Retrieval outcome
    "was_injected": True,
    "was_used_by_llm": True,

    # Quality feedback
    "user_corrected": False,
    "user_confirmed": False,
    "retrieval_success": True
}
```

**GEPA can then**:
- Analyze which signals best predict `was_used_by_llm`
- Adjust weights to maximize useful injections
- Learn user-specific patterns
- Evolve thresholds over time

---

## Adaptive Learning Loop

```
┌─────────────────────────────────────────────────────────────┐
│ Real-Time: Multi-Signal Confidence                          │
│ ─────────────────────────────────────────────────────────── │
│ Extract prosody + linguistic + contextual → TRUE confidence │
│ Store with confidence → Retrieve by confidence              │
│ Log: (signals, confidence, outcome)                         │
└─────────────────────────────────────────────────────────────┘
  ↓ (Accumulate trajectories)
┌─────────────────────────────────────────────────────────────┐
│ Offline: GEPA Optimization                                   │
│ ─────────────────────────────────────────────────────────── │
│ Analyze trajectories:                                        │
│   - Which signal combinations predict success?              │
│   - What confidence threshold optimizes precision/recall?   │
│   - Are fusion weights optimal?                             │
│                                                              │
│ Evolve:                                                      │
│   - Adjust prosody thresholds                               │
│   - Update fusion weights                                   │
│   - Tune filtering threshold                                │
└─────────────────────────────────────────────────────────────┘
  ↓ (Deploy improved config)
┌─────────────────────────────────────────────────────────────┐
│ Next Session: Better Confidence System                      │
│ ─────────────────────────────────────────────────────────── │
│ Uses evolved weights → Better confidence → Better retrieval │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Considerations

### Latency Budget

**Total overhead**: ~40-60ms

| Component | Time | Budget |
|-----------|------|--------|
| Prosody extraction | ~15ms | 35% prosody signal weight |
| Linguistic analysis | ~5ms | 25% linguistic signal weight |
| Context lookup | ~5ms | 10% contextual signal weight |
| Usage stats lookup | ~2ms | 5% usage signal weight |
| Confidence calculation | <1ms | Negligible |
| **Total** | **~30ms** | Within 800ms target |

### Optimization Strategies

1. **Parallel Processing**:
   ```python
   # Run prosody + linguistic in parallel
   with ThreadPoolExecutor() as executor:
       prosody_future = executor.submit(extract_prosody, audio)
       ling_future = executor.submit(analyze_linguistic, text)

       prosody = prosody_future.result()
       linguistic = ling_future.result()
   ```

2. **Caching**:
   - Cache prosody features per utterance
   - Cache linguistic analysis per text span
   - Cache usage stats in RAM

3. **Early Exit**:
   ```python
   # If audio quality too low, skip prosody
   if audio_snr < threshold:
       prosody_conf = 0.5  # Neutral
       # Reweight: give more to linguistic
       weights = [0.30, 0.0, 0.50, 0.15, 0.05]
   ```

---

## Success Metrics

### Before (Arbitrary Confidence)

- All name relations: 0.95
- All actions: 0.85
- Cannot distinguish certain vs uncertain
- Retrieval precision: ~60%

### After (TRUE Confidence)

- Confidence correlates with human labels: r > 0.80
- High-conf facts (>0.8): 90%+ usage when retrieved
- Low-conf facts (<0.5): Filtered out
- Retrieval precision: >85%

### Measurable KPIs

1. **Confidence Validity**:
   - Pearson correlation with human-labeled confidence: r > 0.80
   - Spearman rank correlation: ρ > 0.75

2. **Retrieval Quality**:
   - Precision @ 3 bullets: 60% → 85%+
   - F1 score: 0.65 → 0.88+

3. **Signal Importance**:
   - Which signals matter most? (GEPA discovers)
   - User-specific patterns learned

4. **Attention Budget Efficiency**:
   - Token waste reduction: 66% → <10%
   - Signal-to-noise ratio: 1:2 → 9:1

---

## Conclusion

The multi-signal confidence system transforms arbitrary confidence scores into evidence-based TRUE confidence by fusing:
- **Voice signals** (prosody, emotion, timing)
- **Text signals** (hedge words, certainty markers)
- **Context signals** (temporal, corrections, latency)
- **Usage signals** (historical retrieval success)

This enables:
1. ✅ **Accurate confidence** (correlates with reality)
2. ✅ **Signal-aware retrieval** (filter by TRUE confidence)
3. ✅ **GEPA optimization** (learn from real signals)
4. ✅ **Self-improving system** (adaptive weights)

**Next Steps**:
1. Implement Phase 1 (prosody extraction)
2. Implement Phase 2 (confidence fusion)
3. Test on real conversations
4. Measure improvement vs baseline
5. Integrate with GEPA for continuous improvement