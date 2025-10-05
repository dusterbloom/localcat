# Prosody Integration for LocalCat Graph Intelligence
*Tracking Emotional Context in Real-Time Knowledge Graphs*

## Executive Summary

This document outlines the integration of prosodic analysis into LocalCat's graph extraction pipeline, enabling emotion-aware knowledge graphs that capture not just *what* was said, but *how* it was said. By analyzing pitch, rhythm, and vocal qualities in parallel with UD/SRL extraction, we can achieve <50ms additional latency while adding crucial emotional dimensions to the graph.

## 🎯 Strategic Value

### Why Prosody Matters
- **38% of communication** is conveyed through tone (Mehrabian's 7-38-55 rule)
- Current NLP systems are "emotionally blind" - they miss sarcasm, uncertainty, emphasis
- Prosody changes meaning: "Sure" (agreement) vs "Sure?" (skepticism)

### Unique Advantages for LocalCat
- **Temporal emotional maps**: Query by emotional state ("What did John say when excited?")
- **Speaker characterization**: Build prosodic profiles for better diarization
- **Confidence detection**: Uncertainty in voice → lower extraction confidence
- **Emphasis extraction**: Stressed words → higher graph importance

## 🏗️ Technical Architecture

### Current Pipeline
```
Audio → Whisper → Text → UD/SRL → Graph → LMDB
         (MLX)           (ONNX)
```

### Enhanced Pipeline with Prosody
```
Audio → Whisper → Text → UD/SRL → Graph → LMDB
  ↓      (MLX)           (ONNX)      ↑
  └─→ Prosody Extractor ─────────────┘
      (Parselmouth/ONNX)
```

## 📊 Prosody Feature Extraction

### Core Features to Extract

#### 1. **Fundamental Frequency (F0)**
- **Pitch contour**: Rising/falling/flat patterns
- **Pitch range**: Speaker's emotional activation
- **Pitch variance**: Monotone vs expressive
- **Implementation**: 10ms windows, 5ms hop

#### 2. **Intensity/Energy**
- **Volume patterns**: Emphasis detection
- **Energy distribution**: Word-level stress
- **Silent pauses**: Hesitation markers

#### 3. **Temporal Features**
- **Speaking rate**: Words per minute
- **Articulation rate**: Syllables per second
- **Pause patterns**: Cognitive load indicators

#### 4. **Voice Quality**
- **Harmonic-to-Noise Ratio (HNR)**: Vocal strain
- **Jitter/Shimmer**: Voice stability
- **Formants**: Vowel quality changes

### Emotion Mapping Rules

```python
EMOTION_PATTERNS = {
    'excited': {
        'pitch_mean': (200, 400),  # Hz
        'pitch_variance': 'high',
        'energy': 'high',
        'rate': 'fast'
    },
    'uncertain': {
        'pitch_variance': 'high',
        'pitch_contour': 'rising',
        'pauses': 'frequent',
        'fillers': ['um', 'uh']
    },
    'confident': {
        'pitch_variance': 'low',
        'pitch_contour': 'falling',
        'rate': 'steady',
        'pauses': 'minimal'
    },
    'angry': {
        'pitch_mean': 'high',
        'energy': 'high',
        'hnr': 'low',
        'rate': 'variable'
    }
}
```

## 🛠️ Implementation Plan

### Phase 1: Core Prosody Extraction (Week 1)

#### Tool Selection
**Primary: Parselmouth** (Python wrapper for Praat)
```python
import parselmouth
from parselmouth.praat import call

def extract_prosody(audio_chunk):
    sound = parselmouth.Sound(audio_chunk)
    
    # Pitch extraction
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0_values = pitch.selected_array['frequency']
    
    # Intensity
    intensity = call(sound, "To Intensity", 75, 0.0)
    
    # Voice quality
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    
    return {
        'f0_mean': np.nanmean(f0_values),
        'f0_std': np.nanstd(f0_values),
        'f0_range': np.ptp(f0_values[~np.isnan(f0_values)]),
        'intensity_mean': intensity.get_mean(),
        'hnr': harmonicity.get_mean()
    }
```

**Fallback: MyProsody** for comprehensive analysis
```python
import myprosody as mysp

def analyze_speech_mood(audio_file):
    # Returns emotional state and confidence
    return mysp.myspmood(audio_file, "speaker")
```

### Phase 2: Graph Integration (Week 2)

#### Extended Graph Schema
```python
# Current relation
(subject, predicate, object, confidence)

# Enhanced with prosody
(subject, predicate, object, confidence, prosody_context)

prosody_context = {
    'emotion': 'excited',
    'pitch': 250.5,
    'energy': 0.8,
    'timestamp': 1234567890.123,
    'confidence': 0.85
}
```

#### LMDB Schema Extension
```python
# New indexes for prosody queries
PROSODY_INDEXES = {
    'emotion_index': 'emotion → [relation_ids]',
    'pitch_index': 'pitch_range → [relation_ids]',
    'temporal_emotion': 'timestamp+emotion → [relation_ids]'
}
```

### Phase 3: Real-time Processing Pipeline (Week 3)

#### Streaming Buffer Architecture
```python
class ProsodyStreamBuffer:
    def __init__(self, window_ms=1000, hop_ms=100):
        self.window = window_ms
        self.hop = hop_ms
        self.buffer = RingBuffer(window_ms * 48)  # 48kHz
        
    async def process_chunk(self, audio_chunk):
        self.buffer.add(audio_chunk)
        
        if self.buffer.ready():
            # Extract prosody features
            features = await self.extract_async()
            
            # Align with text from Whisper
            text_alignment = self.align_with_transcript()
            
            return {
                'features': features,
                'alignment': text_alignment,
                'latency_ms': self.last_latency
            }
```

#### Parallel Processing Strategy
```python
async def process_audio_stream(audio):
    # Parallel paths
    text_task = whisper_transcribe(audio)
    prosody_task = extract_prosody(audio)
    
    # Wait for both
    text, prosody = await asyncio.gather(text_task, prosody_task)
    
    # Merge for graph extraction
    enhanced_context = merge_text_prosody(text, prosody)
    
    # Extract with emotional context
    graph = extract_graph_with_emotion(enhanced_context)
    
    return graph
```

### Phase 4: Query Interface (Week 4)

#### Emotion-Aware Queries
```python
class EmotionalGraphQuery:
    def query_by_emotion(self, emotion, speaker=None):
        """Find all statements made with specific emotion"""
        return self.graph.filter(
            prosody__emotion=emotion,
            speaker=speaker
        )
    
    def temporal_emotion_map(self, time_range):
        """Get emotional trajectory over time"""
        return self.graph.aggregate(
            time_range=time_range,
            group_by='emotion',
            metrics=['count', 'avg_confidence']
        )
    
    def emphasis_detection(self):
        """Find emphasized/important statements"""
        return self.graph.filter(
            prosody__energy__gt=0.8,
            prosody__pitch_variance__gt='high'
        )
```

## 📈 Performance Targets

| Metric | Target | Current | Method |
|--------|--------|---------|--------|
| Prosody extraction | <20ms | - | Parselmouth with caching |
| Alignment accuracy | >95% | - | Fuzzy timestamp matching |
| Emotion detection | >85% | - | Rule-based + ML fallback |
| Memory overhead | <50MB | - | Ring buffer + pruning |
| Total pipeline | <130ms | 80ms | Parallel processing |

## 🔬 Evaluation Metrics

### Accuracy Metrics
- **Emotion classification F1**: Target >0.85
- **Emphasis detection precision**: Target >0.90
- **Speaker mood consistency**: Track over conversation

### Performance Metrics
- **Real-time factor**: Must stay <1.0 (faster than real-time)
- **Latency percentiles**: P99 <150ms
- **Memory usage**: <200MB total

### Quality Metrics
- **User feedback**: A/B test emotion-aware vs baseline
- **Query relevance**: Measure retrieval precision with emotion filters
- **Temporal consistency**: Emotional transitions should be smooth

## 🚨 Risk Mitigation

### Technical Risks
1. **Latency spikes**: Use pre-computed prosody for common patterns
2. **Alignment issues**: Implement fuzzy matching with confidence scores
3. **Speaker overlap**: Fallback to speaker-independent features

### Quality Risks
1. **Cultural differences**: Build speaker-specific baselines
2. **Noise robustness**: Use robust pitch tracking (CREPE/PYIN)
3. **Emotion ambiguity**: Output probability distributions, not hard labels

## 🔄 Integration with GEPA

Prosody patterns can be optimized by GEPA:
```python
# GEPA can learn:
# - Speaker-specific emotion thresholds
# - Context-dependent prosody interpretation
# - Optimal feature combinations for emotions

gepa_feedback = {
    'execution': prosody_extraction_trace,
    'ground_truth': user_corrected_emotions,
    'metric': emotion_classification_accuracy
}
```

## 📚 References & Resources

### Libraries
- **Parselmouth**: https://github.com/YannickJadoul/Parselmouth
- **MyProsody**: https://github.com/Shahabks/myprosody
- **DisVoice**: https://github.com/jcvasquezc/DisVoice
- **pyAudioAnalysis**: https://github.com/tyiannak/pyAudioAnalysis

### Research Papers
- Jadoul et al. (2018). "Introducing Parselmouth: A Python interface to Praat"
- Schuller et al. (2013). "The INTERSPEECH 2013 computational paralinguistics challenge"
- Eyben et al. (2016). "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)"

### Datasets for Testing
- **RAVDESS**: Emotional speech with prosody labels
- **IEMOCAP**: Interactive emotional dyadic motion capture
- **CMU-MOSEI**: Multimodal sentiment/emotion dataset

## 🎯 Success Criteria

- [ ] Prosody extraction adds <50ms latency
- [ ] Emotion detection accuracy >85%
- [ ] Graph queries can filter by emotion
- [ ] Users report improved context understanding
- [ ] System learns speaker-specific patterns

## 📅 Timeline

**Week 1**: Basic prosody extraction with Parselmouth
**Week 2**: Graph schema updates and LMDB integration  
**Week 3**: Real-time pipeline with alignment
**Week 4**: Query interface and evaluation
**Month 2**: GEPA optimization of emotion patterns
**Month 3**: Production deployment with monitoring

---

*This prosody integration will transform LocalCat from a "what was said" system to a "how it was said" system, adding crucial emotional intelligence to your knowledge graphs.*
