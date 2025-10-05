# Prosody Integration Roadmap

**Purpose**: Extract voice prosody features to establish TRUE confidence scores for memory facts, enabling signal-aware retrieval and GEPA optimization.

**Problem**: Current confidence is arbitrary (0.85-0.95) based only on relation type. We're a voice agent ignoring voice signals.

**Solution**: Integrate prosody analysis to measure user certainty, emphasis, corrections, and emotional importance.

---

## Why Prosody Matters

### Current Confidence System (Broken)

```python
# memory_hotpath.py:164-169
if r == "name":
    conf = 0.95  # WHY? Arbitrary!
elif r.startswith("v:"):
    conf = 0.85  # Made up!
else:
    conf = 0.9   # Guessing!
```

**Both get 0.9 confidence**:
1. ✅ "My NAME is ALICE!" (emphatic, certain, important)
2. ❌ "I... think... maybe I prefer blue?" (hesitant, uncertain, speculative)

We can't distinguish because we only use text patterns.

### What Prosody Reveals

**Certainty**:
```
"My name is Alice"
  Audio: Falling pitch, clear articulation, no hesitation
  → TRUE confidence: 0.95 ✅

"I think my name might be Alice?"
  Audio: Rising pitch, hesitation, questioning tone
  → TRUE confidence: 0.40 ✅
```

**Emphasis** (importance):
```
"My DOG is named FLUFFY"
  Audio: Strong stress on "DOG" and "FLUFFY"
  → These entities are important to user
  → Retrieval priority: HIGH

"My dog is named fluffy"
  Audio: Flat, unstressed
  → Less emotionally significant
  → Retrieval priority: NORMAL
```

**Corrections**:
```
"NO, I said ALICE, not Alicia!"
  Audio: Sharp onset, emphatic stress, frustrated tone
  → Demote "Alicia" (confidence → 0.0)
  → Reinforce "Alice" (confidence → 0.98)
```

---

## Prosody Features Taxonomy

### 1. Pitch/Intonation

**What it reveals**: Certainty, question vs statement, emotional state

**Extractable features**:
- **F0 contour** (fundamental frequency over time)
  - Falling pitch: Statement/certainty
  - Rising pitch: Question/uncertainty
  - Flat pitch: Low engagement
  - High variance: Emotional/emphatic

**Mapping to confidence**:
```python
if pitch_slope < -20:  # Falling
    certainty_bonus = +0.15
elif pitch_slope > 20:  # Rising
    certainty_penalty = -0.20
else:
    certainty_modifier = 0.0
```

**Example**:
```
"I live in Paris."  (falling) → certainty_bonus = +0.15
"I live in Paris?"  (rising)  → certainty_penalty = -0.20
```

### 2. Emphasis/Stress

**What it reveals**: Important vs incidental information

**Extractable features**:
- **Loudness peaks** (energy/amplitude on syllables)
- **Duration** (lengthened syllables = emphasis)
- **Pitch accent** (pitch jumps on stressed words)

**Mapping to importance**:
```python
def calculate_entity_importance(entity_tokens, audio_segment):
    stress_scores = []
    for token in entity_tokens:
        loudness = get_rms_energy(token.audio)
        duration = token.duration
        pitch_accent = has_pitch_jump(token.audio)

        stress_score = (
            loudness * 0.4 +
            duration * 0.3 +
            pitch_accent * 0.3
        )
        stress_scores.append(stress_score)

    avg_stress = mean(stress_scores)

    # Map to importance weight
    if avg_stress > 0.8:
        importance = "HIGH"      # Strongly emphasized
    elif avg_stress > 0.5:
        importance = "NORMAL"    # Normal stress
    else:
        importance = "LOW"       # De-emphasized

    return importance
```

**Example**:
```
"My DOG is named FLUFFY"
  "dog": stress=0.9 → HIGH importance
  "fluffy": stress=0.85 → HIGH importance
  "named": stress=0.3 → LOW importance

  → Store: (you, has, dog, importance=HIGH)
  → Store: (dog, name, fluffy, importance=HIGH)
```

### 3. Speech Rate/Fluency

**What it reveals**: Confidence, cognitive load, uncertainty

**Extractable features**:
- **Speaking rate** (syllables per second)
- **Pause frequency** (hesitations)
- **Pause duration** (thinking time)
- **Filled pauses** ("um", "uh")

**Mapping to confidence**:
```python
def calculate_fluency_confidence(utterance_audio):
    speaking_rate = syllables_per_second(utterance_audio)
    pause_count = count_pauses(utterance_audio)
    filler_count = count_fillers(utterance_audio, ["um", "uh", "like"])

    # Fast, fluent = confident
    if speaking_rate > 4.5 and pause_count < 2 and filler_count == 0:
        fluency_bonus = +0.20
    # Slow, hesitant = uncertain
    elif speaking_rate < 2.5 or pause_count > 4 or filler_count > 2:
        fluency_penalty = -0.25
    else:
        fluency_modifier = 0.0

    return fluency_modifier
```

**Example**:
```
"My name is Alice"
  Rate: 5.2 syll/s, pauses: 0, fillers: 0
  → fluency_bonus = +0.20

"I... um... think... my name is... uh... Alice?"
  Rate: 2.1 syll/s, pauses: 4, fillers: 2
  → fluency_penalty = -0.25
```

### 4. Emotional Valence/Arousal

**What it reveals**: Personal importance, engagement

**Extractable features**:
- **Valence** (positive/negative emotion)
- **Arousal** (excited/calm)
- **Voice quality** (breathy, tense, relaxed)

**Mapping to importance**:
```python
def calculate_emotional_importance(audio_segment):
    valence = get_valence(audio_segment)  # -1 to +1
    arousal = get_arousal(audio_segment)  # 0 to 1

    # High arousal + positive valence = personally important
    if arousal > 0.7 and valence > 0.5:
        emotional_importance = "HIGH"
    # High arousal + negative valence = frustration/correction
    elif arousal > 0.7 and valence < -0.5:
        emotional_importance = "CORRECTION"
    else:
        emotional_importance = "NEUTRAL"

    return emotional_importance
```

**Example**:
```
"I'm SO EXCITED about my new dog!" (happy, aroused)
  → emotional_importance = HIGH
  → Retrieval priority boost for "dog" facts

"My dog died last year" (sad, flat)
  → emotional_importance = HIGH (different reason)
  → Sensitive topic, handle carefully
```

### 5. Correction Prosody

**What it reveals**: Fact updates, error corrections

**Extractable features**:
- **Prosodic shift** (sudden change from baseline)
- **Emphatic stress** on corrected word
- **Negation markers** + emphatic tone
- **Frustrated/impatient tone**

**Mapping to graph updates**:
```python
def detect_correction(current_audio, previous_audio):
    # Prosodic contrast detection
    baseline_pitch = mean_pitch(previous_audio)
    current_pitch = mean_pitch(current_audio)
    pitch_delta = abs(current_pitch - baseline_pitch)

    # Negation + emphasis
    has_negation = contains_words(current_audio, ["no", "not", "wrong"])
    has_emphasis = peak_energy(current_audio) > 0.8

    # Frustration/impatience markers
    arousal_spike = arousal(current_audio) > arousal(previous_audio) + 0.3

    if pitch_delta > 30 and has_negation and has_emphasis:
        return "STRONG_CORRECTION"
    elif arousal_spike and has_emphasis:
        return "CORRECTION"
    else:
        return "NORMAL"
```

**Example**:
```
User: "My name is Alicia"
  → Store: (you, name, Alicia, conf=0.9)

User: "NO! I said ALICE, not Alicia!"
  Prosody: pitch_delta=45, emphatic stress, frustrated
  → Detection: STRONG_CORRECTION
  → Update: negate_edge(you, name, Alicia, conf=0.0)
  → Update: observe_edge(you, name, Alice, conf=0.98)
```

---

## Implementation Architecture

### Phase 1: Feature Extraction Pipeline

```
Audio Stream (from Parakeet STT)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Audio Segmentation                                          │
│ ─────────────────────────────────────────────────────────── │
│ Input: Continuous audio + transcript with word timestamps   │
│ Output: Audio segments aligned to words/utterances          │
│                                                              │
│ Component: TimeAlignedAudioSegmenter                        │
│   - Uses Parakeet word timestamps                           │
│   - Segments audio by utterance boundaries                  │
│   - Maintains audio-text alignment                          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Prosody Feature Extraction                                  │
│ ─────────────────────────────────────────────────────────── │
│ Library: parselmouth (Praat Python binding)                 │
│                                                              │
│ Features extracted:                                          │
│   • Pitch (F0) contour                                      │
│   • Energy/loudness (RMS)                                   │
│   • Duration per phoneme/syllable                           │
│   • Spectral features (formants)                            │
│   • Voice quality (jitter, shimmer)                         │
│                                                              │
│ Output: ProsodyFeatures(                                    │
│   pitch_contour=[...],                                      │
│   energy=[...],                                             │
│   rate=4.2,                                                 │
│   pauses=[...],                                             │
│   ...                                                        │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Linguistic Feature Extraction                               │
│ ─────────────────────────────────────────────────────────── │
│ Input: Transcript text                                      │
│                                                              │
│ Features extracted:                                          │
│   • Hedge words: "I think", "maybe", "probably"            │
│   • Certainty markers: "definitely", "always", "never"     │
│   • Negations: "no", "not", "don't"                        │
│   • Temporal markers: "used to", "now", "forever"          │
│   • Correction phrases: "I meant", "actually", "sorry"     │
│                                                              │
│ Component: LinguisticCertaintyAnalyzer                      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Multi-Signal Confidence Fusion                              │
│ ─────────────────────────────────────────────────────────── │
│ Inputs:                                                      │
│   - Prosody features (pitch, stress, fluency)              │
│   - Linguistic certainty (hedge words, markers)            │
│   - Fact type (name=permanent, preference=changeable)      │
│                                                              │
│ Output: TRUE confidence score [0.0-1.0]                     │
│                                                              │
│ Component: ConfidenceFusion (see multi_signal_confidence)   │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Integration with Memory Pipeline

```
Current Pipeline:
  Transcript → Extract → Store(conf=0.9) → Retrieve

Enhanced Pipeline:
  Audio + Transcript → Extract → Prosody → TRUE conf → Store → Retrieve
```

**Integration point**: Before storage in `memory_hotpath.py:process_turn()`

```python
# memory_hotpath.py - Enhanced extraction
def process_turn(self, text: str, audio: np.ndarray, session_id: str, turn_id: int):
    # Existing extraction
    entities, triples, neg_count, doc = self.extractor.extract(text, lang)

    # NEW: Prosody analysis
    prosody_features = self.prosody_analyzer.extract(audio, text, word_timestamps)

    # NEW: Multi-signal confidence
    for s, r, d in triples:
        # Old way (WRONG):
        # conf = 0.95 if r == "name" else 0.9

        # New way (RIGHT):
        conf = self.confidence_fusion.calculate(
            relation=r,
            text_span=get_text_span(s, r, d, text),
            prosody=prosody_features.get_span(text_span),
            linguistic=self.linguistic_analyzer.analyze(text_span)
        )

        # Store with TRUE confidence
        self.store.observe_edge(s, r, d, conf=conf, now_ts=now_ts)
```

---

## Technical Implementation

### Required Libraries

```python
# requirements.txt additions
parselmouth>=0.4.3        # Prosody feature extraction (Praat Python)
librosa>=0.10.0           # Audio processing utilities
pydub>=0.25.1             # Audio segmentation
numpy>=1.24.0             # Numerical operations
scipy>=1.11.0             # Signal processing
```

### Core Components

#### 1. ProsodyAnalyzer

```python
# core/prosody/analyzer.py
import parselmouth
from parselmouth.praat import call
import numpy as np

class ProsodyAnalyzer:
    """Extract prosody features from audio aligned to text."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract(self, audio: np.ndarray, text: str,
                word_timestamps: List[Tuple[str, float, float]]) -> ProsodyFeatures:
        """
        Extract prosody features from audio.

        Args:
            audio: Audio samples (16kHz, mono)
            text: Transcript text
            word_timestamps: [(word, start_time, end_time), ...]

        Returns:
            ProsodyFeatures with pitch, energy, rate, pauses, etc.
        """
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)

        # Extract pitch (F0)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()

        # Extract intensity (loudness)
        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        intensity_times = intensity.xs()

        # Calculate speaking rate
        speaking_rate = self._calculate_speaking_rate(text, word_timestamps)

        # Detect pauses
        pauses = self._detect_pauses(word_timestamps)

        # Align features to words
        word_features = []
        for word, start, end in word_timestamps:
            word_pitch = self._get_pitch_in_range(pitch_values, pitch_times, start, end)
            word_energy = self._get_energy_in_range(intensity_values, intensity_times, start, end)

            word_features.append(WordProsody(
                word=word,
                start=start,
                end=end,
                pitch_mean=np.mean(word_pitch),
                pitch_range=np.max(word_pitch) - np.min(word_pitch),
                pitch_slope=self._calculate_slope(word_pitch),
                energy_mean=np.mean(word_energy),
                energy_peak=np.max(word_energy),
                duration=end - start
            ))

        return ProsodyFeatures(
            pitch_contour=pitch_values,
            pitch_times=pitch_times,
            intensity_contour=intensity_values,
            intensity_times=intensity_times,
            speaking_rate=speaking_rate,
            pauses=pauses,
            word_features=word_features
        )

    def _calculate_speaking_rate(self, text: str,
                                  word_timestamps: List[Tuple[str, float, float]]) -> float:
        """Calculate syllables per second."""
        if not word_timestamps:
            return 0.0

        syllable_count = sum(self._count_syllables(word) for word, _, _ in word_timestamps)
        duration = word_timestamps[-1][2] - word_timestamps[0][1]

        return syllable_count / duration if duration > 0 else 0.0

    def _detect_pauses(self, word_timestamps: List[Tuple[str, float, float]],
                       threshold_ms: float = 200) -> List[Tuple[float, float]]:
        """Detect pauses between words."""
        pauses = []
        for i in range(len(word_timestamps) - 1):
            current_end = word_timestamps[i][2]
            next_start = word_timestamps[i + 1][1]
            gap = (next_start - current_end) * 1000  # Convert to ms

            if gap > threshold_ms:
                pauses.append((current_end, next_start))

        return pauses
```

#### 2. LinguisticCertaintyAnalyzer

```python
# core/prosody/linguistic_certainty.py
import spacy
from typing import Dict

class LinguisticCertaintyAnalyzer:
    """Analyze text for certainty/uncertainty markers."""

    # Hedge words reduce certainty
    HEDGE_WORDS = {
        "maybe", "perhaps", "possibly", "probably", "might", "could",
        "I think", "I believe", "I guess", "I suppose", "sort of", "kind of"
    }

    # Certainty markers increase certainty
    CERTAINTY_MARKERS = {
        "definitely", "certainly", "absolutely", "always", "never",
        "clearly", "obviously", "undoubtedly", "sure", "positive"
    }

    # Correction markers indicate update
    CORRECTION_MARKERS = {
        "no", "not", "wrong", "actually", "I meant", "correction",
        "sorry", "mistake", "oops"
    }

    def analyze(self, text: str) -> LinguisticFeatures:
        """Extract linguistic certainty features."""
        text_lower = text.lower()

        # Count markers
        hedge_count = sum(1 for h in self.HEDGE_WORDS if h in text_lower)
        certainty_count = sum(1 for c in self.CERTAINTY_MARKERS if c in text_lower)
        correction_count = sum(1 for c in self.CORRECTION_MARKERS if c in text_lower)

        # Detect question marks
        is_question = text.strip().endswith('?')

        # Calculate certainty score
        certainty_score = 0.5  # Neutral baseline

        # Adjust based on markers
        certainty_score += certainty_count * 0.15
        certainty_score -= hedge_count * 0.20
        certainty_score -= 0.25 if is_question else 0.0

        # Clamp to [0, 1]
        certainty_score = max(0.0, min(1.0, certainty_score))

        return LinguisticFeatures(
            hedge_count=hedge_count,
            certainty_count=certainty_count,
            correction_count=correction_count,
            is_question=is_question,
            certainty_score=certainty_score
        )
```

#### 3. ConfidenceFusion

```python
# core/prosody/confidence_fusion.py

class ConfidenceFusion:
    """Fuse multiple signals into TRUE confidence score."""

    def calculate(self, relation: str, text_span: str,
                  prosody: WordProsodySpan,
                  linguistic: LinguisticFeatures) -> float:
        """
        Calculate TRUE confidence from multiple signals.

        Weights:
          - Base confidence (fact type): 40%
          - Prosody features: 35%
          - Linguistic certainty: 25%
        """
        # Base confidence from fact type
        base_conf = self._get_base_confidence(relation)

        # Prosody confidence
        prosody_conf = self._calculate_prosody_confidence(prosody)

        # Linguistic confidence
        linguistic_conf = linguistic.certainty_score

        # Weighted fusion
        final_conf = (
            base_conf * 0.40 +
            prosody_conf * 0.35 +
            linguistic_conf * 0.25
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, final_conf))

    def _get_base_confidence(self, relation: str) -> float:
        """Base confidence from relation type (same as before, but lower weight)."""
        if relation == "name":
            return 0.90  # Names are usually permanent
        elif relation.startswith("v:"):
            return 0.70  # Verbs/actions less certain
        elif relation in {"lives_in", "works_at"}:
            return 0.80  # Location/work can change
        else:
            return 0.75  # Default

    def _calculate_prosody_confidence(self, prosody: WordProsodySpan) -> float:
        """Calculate confidence from prosody features."""
        confidence = 0.5  # Neutral baseline

        # Pitch slope (falling = certain)
        if prosody.mean_pitch_slope < -15:
            confidence += 0.20
        elif prosody.mean_pitch_slope > 15:
            confidence -= 0.20

        # Speaking rate (fast = confident)
        if prosody.speaking_rate > 4.5:
            confidence += 0.15
        elif prosody.speaking_rate < 2.5:
            confidence -= 0.15

        # Pauses (few = confident)
        if prosody.pause_count == 0:
            confidence += 0.10
        elif prosody.pause_count > 3:
            confidence -= 0.15

        # Energy/emphasis (high = important/certain)
        if prosody.mean_energy > 0.75:
            confidence += 0.10

        return max(0.0, min(1.0, confidence))
```

---

## Integration Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Extract basic prosody features without breaking existing system

**Tasks**:
1. Install parselmouth, librosa
2. Create `core/prosody/` module structure
3. Implement `ProsodyAnalyzer.extract()` (pitch, energy, rate)
4. Unit tests for feature extraction

**Deliverable**: Working prosody extraction on sample audio

### Phase 2: Audio Pipeline Integration (Week 3)

**Goal**: Get audio + word timestamps from Parakeet STT

**Tasks**:
1. Modify Parakeet integration to capture audio stream
2. Extract word-level timestamps from Parakeet
3. Create `AudioSegmenter` for word-aligned segments
4. Test: Can we align prosody features to words?

**Deliverable**: Audio segments aligned to transcript words

### Phase 3: Linguistic Analysis (Week 4)

**Goal**: Detect certainty/uncertainty from text

**Tasks**:
1. Implement `LinguisticCertaintyAnalyzer`
2. Build hedge word / certainty marker dictionaries
3. Test on real conversation samples
4. Tune thresholds

**Deliverable**: Linguistic certainty scores for utterances

### Phase 4: Confidence Fusion (Week 5)

**Goal**: Combine prosody + linguistic → TRUE confidence

**Tasks**:
1. Implement `ConfidenceFusion.calculate()`
2. Tune fusion weights (40% base, 35% prosody, 25% linguistic)
3. Test on labeled dataset (manual confidence labels)
4. Compare: TRUE conf vs arbitrary conf

**Deliverable**: Multi-signal confidence system

### Phase 5: Memory Integration (Week 6)

**Goal**: Replace arbitrary confidence with TRUE confidence

**Tasks**:
1. Modify `memory_hotpath.py:process_turn()` to use prosody
2. Pass audio + timestamps to extraction pipeline
3. Store edges with TRUE confidence
4. Test: Does retrieval improve?

**Deliverable**: End-to-end prosody-aware memory system

### Phase 6: GEPA Optimization (Week 7+)

**Goal**: GEPA learns optimal prosody weights

**Tasks**:
1. Log trajectories with prosody features
2. GEPA optimizes fusion weights
3. GEPA evolves prosody thresholds
4. Measure: Quality improvement over time

**Deliverable**: Self-improving confidence system

---

## Success Metrics

### Baseline (Current System)

- Confidence: Arbitrary (0.85-0.95)
- Retrieval precision: ~60%
- Can't distinguish certain vs uncertain facts

### Target (With Prosody)

- Confidence: TRUE (measured from voice)
- Retrieval precision: >80% (filter by conf > 0.7)
- Detects corrections automatically
- Captures user emphasis
- Adapts to certainty markers

### Measurable KPIs

1. **Confidence Validity**:
   - Correlation: TRUE conf vs human-labeled conf > 0.80

2. **Retrieval Quality**:
   - Precision @ 3 bullets: 60% → 85%+
   - High-conf facts (>0.8): used 90%+ of time
   - Low-conf facts (<0.5): filtered, not injected

3. **Correction Detection**:
   - Detect user corrections: >90% recall
   - Auto-update contradicted facts

4. **Emphasis Tracking**:
   - Important entities (stressed): prioritized in retrieval
   - De-emphasized facts: lower priority

---

## Technical Challenges & Solutions

### Challenge 1: Real-Time Performance

**Problem**: Prosody extraction must stay <20ms (within 800ms latency budget)

**Solution**:
- Use efficient libraries (parselmouth, librosa)
- Parallel processing (audio + text extraction)
- Cache prosody features per utterance
- Profile: Target 10-15ms for prosody extraction

### Challenge 2: Audio Quality

**Problem**: Background noise, poor microphone quality

**Solution**:
- Noise reduction preprocessing (spectral subtraction)
- Robust feature extraction (pitch tracking with confidence)
- Fallback to linguistic-only confidence if audio quality low
- Quality detection: Skip prosody if SNR < threshold

### Challenge 3: Word-Level Alignment

**Problem**: Parakeet may not provide word timestamps

**Solution**:
- Force-align using Montreal Forced Aligner (MFA)
- Or: Use attention weights from STT model
- Or: Fall back to utterance-level features
- Test: Does word-level alignment improve accuracy?

### Challenge 4: Speaker Variability

**Problem**: Different speakers have different prosodic baselines

**Solution**:
- Speaker-normalized features (z-score per speaker)
- Adaptive thresholds (learn per-user baselines)
- Relative measures (pitch delta vs absolute pitch)
- GEPA learns user-specific prosody patterns

---

## Future Enhancements

### Post-MVP Features

1. **Emotion Recognition**:
   - Detect frustration → likely correction
   - Detect excitement → personally important
   - Detect sadness → sensitive topic

2. **Turn-Taking Cues**:
   - Detect when user wants to interrupt
   - Adjust VAD based on prosody
   - Better conversation flow

3. **Personality Adaptation**:
   - Learn user's prosodic style
   - Some users speak fast naturally (not always confident)
   - Some users rarely emphasize (calibrate differently)

4. **Multimodal Fusion**:
   - Combine prosody + visual cues (if video available)
   - Gesture + prosody = even stronger signal
   - Facial expression + tone = emotion detection

---

## Appendix: Example Scenarios

### Scenario 1: High Confidence Statement

```
Audio: "My NAME is ALICE."
  Prosody:
    - Pitch: Falling contour (-22 Hz/s)
    - Energy: Peak on "NAME" (0.89), "ALICE" (0.92)
    - Rate: 5.1 syllables/sec (fluent)
    - Pauses: 0

  Linguistic:
    - Hedge words: 0
    - Certainty markers: 0
    - Question: No

  Confidence Calculation:
    Base (name): 0.90 * 0.40 = 0.36
    Prosody: 0.90 * 0.35 = 0.315  (falling pitch, emphatic, fluent)
    Linguistic: 0.50 * 0.25 = 0.125  (neutral)

    Final: 0.36 + 0.315 + 0.125 = 0.80

  ✅ High confidence (0.80) → Store and inject
```

### Scenario 2: Low Confidence Speculation

```
Audio: "I... think... maybe... I prefer blue?"
  Prosody:
    - Pitch: Rising contour (+18 Hz/s)
    - Energy: Low, flat (0.45 avg)
    - Rate: 2.3 syllables/sec (slow, hesitant)
    - Pauses: 3 long pauses

  Linguistic:
    - Hedge words: 2 ("think", "maybe")
    - Certainty markers: 0
    - Question: Yes

  Confidence Calculation:
    Base (preference): 0.75 * 0.40 = 0.30
    Prosody: 0.20 * 0.35 = 0.07  (rising pitch, hesitant, slow)
    Linguistic: 0.10 * 0.25 = 0.025  (hedge + question)

    Final: 0.30 + 0.07 + 0.025 = 0.395

  ❌ Low confidence (0.39) → Store but don't inject (below 0.70 threshold)
```

### Scenario 3: Emphatic Correction

```
Audio: "NO! I said ALICE, not Alicia!"
  Prosody:
    - Pitch: Spike (+45 Hz) on "NO", "ALICE"
    - Energy: Peak 0.95 on "ALICE"
    - Rate: 5.8 syllables/sec (fast, emphatic)
    - Pauses: 0

  Linguistic:
    - Correction markers: 2 ("NO", "not")
    - Certainty markers: 0
    - Question: No

  Detection: STRONG_CORRECTION

  Actions:
    1. negate_edge(you, name, "Alicia", conf=0.0)
    2. observe_edge(you, name, "Alice", conf=0.98)

  ✅ Correction detected and applied
```

---

## Conclusion

Prosody integration is ESSENTIAL for TRUE confidence. Without it, we're optimizing retrieval based on arbitrary scores that don't reflect reality. This roadmap provides a concrete path from current state (arbitrary conf) to future state (voice-aware conf) that enables all downstream optimizations including GEPA.

**Next Steps**:
1. Review this roadmap
2. Create `voice_intelligence_plan.md` (full voice taxonomy)
3. Create `multi_signal_confidence_system.md` (fusion details)
4. Begin Phase 1 implementation