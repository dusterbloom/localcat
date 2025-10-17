# Voice Intelligence Plan

**Purpose**: Comprehensive taxonomy of voice signals and how to extract actionable intelligence for memory confidence, retrieval, and user intent understanding.

**Vision**: Move beyond text-only AI to become truly voice-native—using prosody, emotion, timing, and acoustic features to understand not just WHAT users say, but HOW they say it.

---

## Core Principle

> **"In voice conversation, HOW something is said carries as much information as WHAT is said."**

A text-only system misses:
- 😊 **Emotion** (excitement, frustration, sadness)
- 💪 **Emphasis** (what's important vs incidental)
- 🤔 **Certainty** (confident vs unsure)
- 🔄 **Corrections** (updates vs new info)
- ⏱️ **Timing** (urgency, hesitation, thinking)

---

## Voice Signal Taxonomy

### 1. Prosody Signals

#### 1.1 Pitch/Intonation (F0)

**What it conveys**: Certainty, emotion, question vs statement, emphasis

**Acoustic features**:
- Fundamental frequency (F0) in Hz
- Pitch contour (shape over time)
- Pitch range (variability)
- Pitch slope (rising vs falling)

**Semantic mappings**:

| Pitch Pattern | Meaning | Example | Confidence Modifier |
|--------------|---------|---------|-------------------|
| **Falling** (statement) | Certainty, finality | "My name is Alice." ↘ | +0.15 |
| **Rising** (question) | Uncertainty, seeking confirmation | "My name is Alice?" ↗ | -0.20 |
| **High rising** (disbelief) | Surprise, doubt | "Really??" ↗↗ | -0.25 |
| **Flat** (bored/neutral) | Low engagement, routine | "My name is alice" → | -0.05 |
| **Contour variation** | Emphasis on words | "My NAME is ALICE" ∧∧ | +0.10 per peak |
| **Compressed range** (monotone) | Sad, depressed, uncertain | Flat delivery | -0.10 |
| **Expanded range** (animated) | Excited, engaged, certain | Lots of variation | +0.10 |

**Implementation notes**:
- Extract F0 using CREPE, pYIN, or Parselmouth
- Normalize by speaker baseline (some speak higher/lower naturally)
- Focus on relative changes, not absolute values

#### 1.2 Energy/Loudness (RMS)

**What it conveys**: Emphasis, importance, emotion intensity

**Acoustic features**:
- RMS energy per frame
- Peak amplitude
- Dynamic range
- Energy distribution across utterance

**Semantic mappings**:

| Energy Pattern | Meaning | Example | Action |
|---------------|---------|---------|--------|
| **High peak on word** | Emphatic stress | "My DOG is named Fluffy" | Mark "dog" as HIGH importance |
| **Sustained high** | Excited, emphatic overall | Loud throughout | High arousal state |
| **Low overall** | Subdued, sad, uncertain | Quiet delivery | Lower confidence |
| **Sharp attack** | Assertive, correcting | "NO!" | Correction marker |
| **Gradual decay** | Fading interest | Trailing off | Lower priority |

**Implementation notes**:
- Calculate RMS energy in 25ms windows
- Compare to speaker baseline (some speak louder/quieter)
- Look for peaks relative to utterance mean (>1.5x = emphasis)

#### 1.3 Speaking Rate/Tempo

**What it conveys**: Confidence, cognitive load, emotion

**Acoustic features**:
- Syllables per second
- Articulation rate (excluding pauses)
- Speaking vs pause time ratio

**Semantic mappings**:

| Rate Pattern | Meaning | Range (syllables/sec) | Confidence Modifier |
|-------------|---------|---------------------|-------------------|
| **Fast** | Confident, fluent, excited | >5.0 | +0.15 |
| **Normal** | Neutral, conversational | 3.5-5.0 | 0.0 |
| **Slow** | Uncertain, careful, sad | 2.0-3.5 | -0.10 |
| **Very slow** | Hesitant, confused, lying | <2.0 | -0.25 |
| **Accelerating** | Building confidence, getting excited | Increasing | +0.05 per trend |
| **Decelerating** | Losing confidence, uncertain | Decreasing | -0.05 per trend |

**Implementation notes**:
- Count syllables using syllable counter or phoneme segmentation
- Exclude pauses from rate calculation (articulation rate)
- Track rate changes within utterance (acceleration/deceleration)

#### 1.4 Pauses/Hesitations

**What it conveys**: Uncertainty, cognitive load, planning, emphasis

**Acoustic features**:
- Pause duration
- Pause frequency
- Pause location (mid-word, mid-phrase, end of utterance)
- Filled pauses ("um", "uh", "like")

**Semantic mappings**:

| Pause Pattern | Meaning | Example | Confidence Modifier |
|--------------|---------|---------|-------------------|
| **No pauses** | Fluent, confident | "My name is Alice" | +0.15 |
| **Short pauses at phrase boundaries** | Normal planning | "My name is Alice, and I live in Paris" | 0.0 |
| **Long pauses mid-utterance** | Uncertainty, thinking | "My name is... Alice?" | -0.20 |
| **Multiple filled pauses** | High uncertainty | "I... um... think... uh..." | -0.30 |
| **Strategic pause before emphasis** | Deliberate emphasis | "My name is... ALICE." | +0.10 |

**Implementation notes**:
- Detect silence >150ms as pause
- Classify filled pauses using ASR
- Location matters: mid-word pauses more significant than boundary pauses

#### 1.5 Voice Quality

**What it conveys**: Emotion, physical state, attitude

**Acoustic features**:
- Jitter (pitch irregularity)
- Shimmer (amplitude irregularity)
- Harmonics-to-Noise Ratio (HNR)
- Spectral tilt
- Breathiness

**Semantic mappings**:

| Voice Quality | Meaning | Acoustic Signature | Use Case |
|--------------|---------|-------------------|----------|
| **Breathy** | Intimacy, sadness, fatigue | Low HNR, high noise | Sensitive topics |
| **Tense/pressed** | Anger, stress, emphasis | High jitter, constricted | Corrections, complaints |
| **Creaky** | Disengagement, boredom | Low F0, irregular | Low priority info |
| **Clear/modal** | Neutral, engaged | High HNR, stable F0 | Normal confidence |

**Implementation notes**:
- Extract using Parselmouth (Praat)
- Useful for emotion detection, not direct confidence
- Combine with other features for full picture

---

### 2. Temporal Signals

#### 2.1 Turn-Taking Cues

**What it conveys**: User intent to speak, interrupt, or yield

**Acoustic features**:
- Inbreath detection (sharp intake before speaking)
- Phrase-final lengthening (slowing down at end)
- Pitch drop at turn end
- Volume decrease

**Semantic mappings**:

| Temporal Pattern | User Intent | Action |
|-----------------|-------------|--------|
| **Sharp inbreath during bot speech** | Wants to interrupt | Stop speaking, listen |
| **No phrase-final lengthening** | More to say, not done | Don't interrupt |
| **Pitch drop + volume decrease** | Turn complete, yielding | Bot can respond |
| **Quick response (<200ms)** | Engaged, confident | High-quality interaction |
| **Slow response (>2s)** | Uncertain, thinking | Lower confidence on response |

**Implementation notes**:
- Integrate with VAD and turn detection
- Detect inbreaths using energy spike + spectral features
- Measure inter-turn latency for engagement

#### 2.2 Response Latency

**What it conveys**: Cognitive load, certainty, engagement

**Measurement**:
- Time from bot utterance end to user speech start
- Time from user question to bot response (feedback for bot quality)

**Semantic mappings**:

| Latency | Meaning | Confidence Impact |
|---------|---------|------------------|
| **<500ms** | Immediate, confident | +0.10 |
| **500ms-1s** | Normal thinking | 0.0 |
| **1-2s** | Uncertain, considering | -0.10 |
| **>2s** | Confused, unsure | -0.20 |

**Implementation notes**:
- Track using turn timestamps
- Exclude cases where user was interrupted or background noise

#### 2.3 Overlap/Interruption Patterns

**What it conveys**: Engagement, corrections, urgency

**Types**:
- **Cooperative overlap**: Adding to bot's point ("Yes, and...")
- **Competitive overlap**: Interrupting to correct ("No, wait...")
- **Backchannel**: Minimal feedback ("uh-huh", "yeah")

**Semantic mappings**:

| Overlap Type | Meaning | Action |
|-------------|---------|--------|
| **Competitive + loud** | Strong correction | Negate previous fact, high conf on new |
| **Cooperative** | Confirmation, agreement | Reinforce fact |
| **Backchannel** | Listening, engaged | Continue, user is following |
| **Frustrated interruption** | Bot error, misunderstanding | Stop, acknowledge, clarify |

---

### 3. Emotional Signals

#### 3.1 Valence (Positive/Negative)

**What it conveys**: User satisfaction, topic importance

**Acoustic correlates**:
- Positive: Higher pitch, faster rate, more pitch variation
- Negative: Lower pitch, slower rate, flat contour

**Semantic mappings**:

| Valence | Meaning | Memory Action |
|---------|---------|--------------|
| **Highly positive** | Excited about topic | Mark as HIGH importance |
| **Neutral** | Factual information | Normal priority |
| **Negative** | Unpleasant topic | Sensitive, handle carefully |

**Use cases**:
- "I'm SO EXCITED about my new dog!" → High importance, retrieve often
- "My dog passed away last year" → Sensitive, retrieve cautiously

#### 3.2 Arousal (Activated/Calm)

**What it conveys**: Intensity of emotion, importance

**Acoustic correlates**:
- High arousal: Louder, faster, more pitch variation
- Low arousal: Quieter, slower, less variation

**Semantic mappings**:

| Arousal | Valence | Emotion | Memory Action |
|---------|---------|---------|--------------|
| **High** | Positive | Excitement, joy | HIGH importance |
| **High** | Negative | Anger, frustration | Correction likely |
| **Low** | Positive | Contentment, calm | Normal priority |
| **Low** | Negative | Sadness, depression | Sensitive topic |

#### 3.3 Emotion Categories

**Discrete emotions** detectable from voice:

| Emotion | Acoustic Signature | Memory Interpretation |
|---------|-------------------|---------------------|
| **Joy** | High pitch, fast rate, laughter | Positive, important topic |
| **Anger** | Loud, tense voice, sharp | Correction, complaint |
| **Sadness** | Low pitch, slow, breathy | Sensitive, low confidence |
| **Fear** | High pitch, fast, trembling | Uncertain, stressed |
| **Surprise** | High pitch spike, fast | Unexpected, update memory |
| **Disgust** | Creaky voice, harsh | Negative topic |

**Implementation**:
- Use pre-trained emotion recognition models (wav2vec2-emotion, etc.)
- Or: Extract features → train classifier on labeled data
- Or: Rule-based from prosody features (simpler, faster)

---

### 4. Linguistic-Acoustic Integration

#### 4.1 Certainty Markers + Prosody

**Principle**: Combine what they say with how they say it

**Examples**:

| Text | Prosody | Combined Interpretation | Confidence |
|------|---------|------------------------|------------|
| "I think..." | Uncertain tone, hesitant | Low certainty | 0.30 |
| "I think..." | Confident tone, emphatic | Idiomatic usage, actually certain | 0.75 |
| "Definitely!" | Emphatic, loud | Very certain | 0.95 |
| "Definitely" | Flat, quiet | Sarcasm or low confidence | 0.40 |
| "Maybe" | High pitch, unsure | Very uncertain | 0.20 |
| "Maybe" | Confident, certain | Hedging but actually knows | 0.60 |

**Lesson**: Text alone is insufficient. Prosody disambiguates.

#### 4.2 Correction Detection

**Linguistic markers**: "No", "actually", "I meant", "sorry", "mistake"
**Prosodic markers**: Sharp onset, emphatic stress, frustrated tone

**Detection logic**:
```python
is_correction = (
    ("no" in text_lower or "wrong" in text_lower)
    AND
    (peak_energy > 0.85 OR pitch_delta > 30 OR arousal > 0.7)
)
```

**Example**:
- Text: "No, I said Alice!" → Linguistic marker present
- Prosody: Loud (0.91), pitch spike (+42 Hz), fast rate → Emphatic
- Combined: **STRONG_CORRECTION** detected ✅

#### 4.3 Emphasis Detection

**Principle**: Stressed words are important to user

**Acoustic cues**:
- Higher pitch on word
- Louder energy on word
- Lengthened duration
- Pitch accent (F0 jump)

**Example**:
```
"My DOG is named FLUFFY"

  Prosody analysis:
    "my": pitch=180Hz, energy=0.45, duration=120ms
    "dog": pitch=220Hz ⬆️, energy=0.89 ⬆️, duration=180ms ⬆️
    "is": pitch=175Hz, energy=0.42, duration=100ms
    "named": pitch=170Hz, energy=0.40, duration=110ms
    "fluffy": pitch=215Hz ⬆️, energy=0.87 ⬆️, duration=200ms ⬆️

  Detected emphasis: "dog" (stress=0.92), "fluffy" (stress=0.88)

  Action: Mark these entities as HIGH importance in memory
```

#### 4.4 Sarcasm/Irony Detection

**Challenge**: Text says one thing, prosody says opposite

**Acoustic signatures**:
- Exaggerated pitch contour (unnatural)
- Lengthened vowels (drawn out)
- Flat affect on positive words
- Mismatch between text and prosody

**Example**:
- Text: "Oh great, that's just wonderful" (positive words)
- Prosody: Flat pitch, low energy, slow rate (negative affect)
- Detection: **SARCASM** → Text meaning reversed, confidence LOW

**Use case**: Don't extract "that's wonderful" as positive fact

---

### 5. Multimodal Integration (Future)

#### 5.1 Visual Cues (if video available)

**Signals**:
- Facial expressions (smile, frown, confusion)
- Eye gaze (engagement, thinking)
- Gestures (emphasis, correction)
- Head movements (nod=yes, shake=no)

**Integration with voice**:
- Smile + positive prosody → Genuine positive emotion
- Frown + negative prosody → Genuine negative emotion
- Smile + negative prosody → Sarcasm or masking

#### 5.2 Context Signals

**Environmental**:
- Background noise level → Stress indicator
- Time of day → Energy level interpretation
- Location (if known) → Context for facts

**Conversational**:
- Topic continuity → Related facts
- Topic shifts → New memory domain
- Repetition → Important, reinforce

---

## Voice Intelligence Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│ Audio Input (from microphone)                               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Voice Activity Detection (VAD)                              │
│ - Detect speech vs silence                                  │
│ - Segment into utterances                                   │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Speech-to-Text (STT) + Word Timestamps                      │
│ - Parakeet STT                                              │
│ - Extract word-level timestamps                             │
│ - Output: Transcript + timing                               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────┬────────────────────────┐
│ Prosody Analysis                   │ Linguistic Analysis    │
│ ────────────────────────────────── │ ────────────────────── │
│ • Pitch (F0) extraction           │ • Hedge word detection │
│ • Energy (RMS) per word           │ • Certainty markers    │
│ • Speaking rate calculation       │ • Correction phrases   │
│ • Pause detection                 │ • Temporal markers     │
│ • Voice quality features          │ • Negation detection   │
│                                    │                        │
│ Output: ProsodyFeatures           │ Output: LingFeatures   │
└────────────────────────────────────┴────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Voice Intelligence Fusion                                    │
│ ─────────────────────────────────────────────────────────── │
│ Combine:                                                     │
│   • Prosody signals (pitch, energy, rate, pauses)          │
│   • Linguistic markers (hedge words, certainty)            │
│   • Emotional state (valence, arousal)                     │
│   • Temporal patterns (latency, turn-taking)               │
│   • Contextual signals (topic, history)                    │
│                                                              │
│ Generate:                                                    │
│   • TRUE confidence score [0.0-1.0]                        │
│   • Importance weight [0.0-1.0]                            │
│   • Correction flag [bool]                                 │
│   • Emotion label [joy/anger/sad/neutral/...]             │
│   • Certainty level [certain/uncertain/very_uncertain]     │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Memory System Integration                                    │
│ ─────────────────────────────────────────────────────────── │
│ Use voice intelligence for:                                  │
│   • Fact confidence (prosody + linguistic certainty)        │
│   • Entity importance (emphasis detection)                  │
│   • Correction handling (prosodic + linguistic markers)     │
│   • Retrieval priority (importance + emotion)               │
│   • Sensitive topic detection (sad/negative emotion)        │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Adaptive Learning (GEPA)                                     │
│ ─────────────────────────────────────────────────────────── │
│ Learn:                                                       │
│   • Optimal prosody thresholds per user                     │
│   • Fusion weights (prosody vs linguistic)                  │
│   • User-specific prosodic patterns                         │
│   • Correction detection sensitivity                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Foundation (Immediate)

**Goal**: Extract basic prosody features

**Components**:
1. Prosody analyzer (pitch, energy, rate, pauses)
2. Word-level alignment (audio segments ↔ transcript words)
3. Linguistic certainty analyzer (hedge words, markers)

**Deliverable**: Working prosody extraction

### Phase 2: Confidence System (Week 2-3)

**Goal**: Generate TRUE confidence from voice

**Components**:
1. Multi-signal confidence fusion
2. Emphasis detection (stressed entities)
3. Basic emotion detection (valence, arousal)

**Deliverable**: Voice-aware confidence scores

### Phase 3: Correction Detection (Week 4)

**Goal**: Automatically detect and apply corrections

**Components**:
1. Prosodic correction detection
2. Linguistic correction markers
3. Automatic graph updates

**Deliverable**: Self-correcting memory

### Phase 4: Advanced Intelligence (Week 5+)

**Goal**: Full voice intelligence integration

**Components**:
1. Emotion recognition (discrete emotions)
2. Turn-taking prediction
3. Sarcasm detection
4. Importance scoring

**Deliverable**: Comprehensive voice understanding

### Phase 5: Adaptive Learning (Week 6+)

**Goal**: GEPA learns optimal voice patterns

**Components**:
1. User-specific prosody baselines
2. Adaptive thresholds
3. Fusion weight optimization
4. Pattern learning

**Deliverable**: Self-improving voice intelligence

---

## Success Metrics

### Baseline (Text-Only)

- Confidence: Arbitrary (0.85-0.95), no voice awareness
- Corrections: Not detected automatically
- Emphasis: Ignored
- Emotion: Not used
- User engagement: Not measured

### Target (Voice-Intelligent)

- **Confidence validity**: TRUE conf correlates with human labels (r > 0.80)
- **Correction detection**: >90% recall, <10% false positives
- **Emphasis detection**: Stressed entities prioritized (80%+ of emphasized entities rank higher)
- **Emotion awareness**: Sensitive topics handled appropriately
- **Engagement tracking**: Response latency, arousal, engagement measured

### Measurable KPIs

1. **Confidence Accuracy**:
   - Human-labeled confidence vs system confidence: r > 0.80
   - High-conf facts (>0.8): Used >90% when retrieved
   - Low-conf facts (<0.5): Filtered out, not injected

2. **Correction Detection**:
   - True positives: >90% (detect actual corrections)
   - False positives: <10% (don't flag normal statements)
   - Correction latency: <50ms (real-time detection)

3. **Emphasis Tracking**:
   - Stressed entities: Ranked higher in retrieval 80%+ of time
   - Important facts: Retrieved more often (usage correlation)

4. **Emotion Integration**:
   - Positive topics: Higher retrieval priority
   - Negative topics: Careful handling, sensitivity flags
   - Frustration: Trigger apology/clarification

---

## Technical Challenges

### Challenge 1: Real-Time Processing

**Constraint**: All processing must complete within 800ms total latency budget

**Breakdown**:
- VAD: ~5ms
- STT: ~50-100ms (Parakeet)
- Prosody extraction: ~10-20ms (target)
- Linguistic analysis: ~5ms
- Confidence fusion: <5ms
- Memory operations: ~10ms

**Total voice intelligence overhead**: ~30-50ms (acceptable!)

**Solutions**:
- Optimize prosody extraction (use efficient libraries)
- Parallel processing (prosody + linguistic in parallel)
- Cache results per utterance

### Challenge 2: Speaker Variability

**Problem**: Different speakers have different baselines

**Solutions**:
- Speaker normalization (z-score features)
- Adaptive baselines (learn per user)
- Relative features (pitch delta, not absolute)
- GEPA learns user-specific patterns

### Challenge 3: Audio Quality

**Problem**: Noise, poor mic, compression artifacts

**Solutions**:
- Noise reduction preprocessing
- Quality detection (skip prosody if quality too low)
- Robust features (F0 tracking with confidence scores)
- Fallback to linguistic-only confidence

### Challenge 4: Multimodal Fusion

**Problem**: How to weight prosody vs linguistic vs contextual?

**Solutions**:
- Initial weights from literature (40% base, 35% prosody, 25% linguistic)
- GEPA learns optimal weights from data
- User-specific adaptation
- Confidence in confidence (meta-confidence based on feature quality)

---

## Research References

### Prosody & Emotion

- **Banse & Scherer (1996)**: Acoustic profiles in vocal emotion expression
- **Juslin & Scherer (2005)**: Vocal expression of affect
- **Bänziger & Scherer (2005)**: The role of intonation in emotional expressions

### Certainty & Confidence

- **Liscombe et al. (2005)**: Detecting certainty in spoken tutorial dialogues
- **Pon-Barry (2008)**: Prosodic manifestations of confidence and uncertainty
- **Swerts & Krahmer (2005)**: Audiovisual prosody and feeling of knowing

### Turn-Taking

- **Gravano & Hirschberg (2011)**: Turn-taking cues in task-oriented dialogue
- **Ward & Tsukahara (2000)**: Prosodic features which cue back-channel responses

### Speech Rate & Fluency

- **Goldman-Eisler (1968)**: Psycholinguistics: Experiments in spontaneous speech
- **Bortfeld et al. (2001)**: Disfluency rates in conversation

---

## Appendix: Feature Extraction Code Templates

### Pitch Extraction (Parselmouth)

```python
import parselmouth

def extract_pitch(audio, sample_rate=16000):
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    pitch = sound.to_pitch()

    pitch_values = pitch.selected_array['frequency']
    pitch_times = pitch.xs()

    return pitch_values, pitch_times
```

### Energy Extraction (Parselmouth)

```python
def extract_energy(audio, sample_rate=16000):
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    intensity = sound.to_intensity()

    energy_values = intensity.values[0]
    energy_times = intensity.xs()

    return energy_values, energy_times
```

### Speaking Rate Calculation

```python
def calculate_speaking_rate(text, word_timestamps):
    syllable_count = sum(count_syllables(word) for word, _, _ in word_timestamps)
    duration = word_timestamps[-1][2] - word_timestamps[0][1]

    return syllable_count / duration if duration > 0 else 0.0

def count_syllables(word):
    # Simple vowel-based syllable counter
    vowels = "aeiouy"
    word = word.lower()
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    return max(1, count)  # At least 1 syllable per word
```

### Pause Detection

```python
def detect_pauses(word_timestamps, threshold_ms=200):
    pauses = []
    for i in range(len(word_timestamps) - 1):
        current_end = word_timestamps[i][2]
        next_start = word_timestamps[i + 1][1]
        gap_ms = (next_start - current_end) * 1000

        if gap_ms > threshold_ms:
            pauses.append({
                'start': current_end,
                'end': next_start,
                'duration_ms': gap_ms
            })

    return pauses
```

---

## Conclusion

Voice intelligence transforms memory from text-only to truly voice-native. By extracting prosody, emotion, timing, and acoustic features, we can:

1. **Establish TRUE confidence** (not arbitrary)
2. **Detect corrections automatically** (user says "no", we update)
3. **Track importance** (what user emphasizes)
4. **Handle emotion** (sensitive topics, excitement)
5. **Measure engagement** (response latency, arousal)

This enables GEPA to optimize retrieval based on REAL signals, not guesses.

**Next Step**: Create `multi_signal_confidence_system.md` detailing the fusion algorithm.