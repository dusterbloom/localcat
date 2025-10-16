# LocalCat Server Backlog

## Completed (2025-10-16)

### ✅ Vision Processing Optimizations
**Goal:** Reduce LLM vision processing latency by 50-75%
- Image preprocessing (resize, compress, aspect ratio preservation)
- Context pruning (limit images to prevent bloat)
- Frame deduplication (skip identical frames)
- Keyword filtering (inject only on vision-related queries)
- Comprehensive test suite (14 tests)

### ✅ TTS Ultra-Low Latency (<100ms TTFB)
**Goal:** Achieve <800ms voice-to-voice latency
- Interruption handling (barge-in support)
- Text chunking (25-char optimal chunks)
- Buffer optimization (40-80ms TTFB)
- Comprehensive test suite (10 tests)

### ✅ STT Hallucination Detection
**Goal:** Reduce false positive transcriptions by 80%
- Pattern-based detection (replaces confidence heuristics)
- 15+ known hallucination patterns
- Short noise filtering
- Comprehensive test suite (9 tests)

## In Progress

### 🚧 Session Persistence Improvements
- Goal: Enable conversation resume across restarts
- Status: Researching storage strategies

### 🚧 Multi-turn Context Management
- Goal: Improve long conversation handling
- Status: Designing pruning strategies

## Backlog (Prioritized)

### P0 - Critical

#### Performance
- [ ] Memory retrieval latency optimization (<50ms p95)
- [ ] LLM streaming latency reduction
- [ ] Audio pipeline optimization

#### Reliability
- [ ] Error recovery and graceful degradation
- [ ] Connection stability improvements
- [ ] Resource leak detection and prevention

### P1 - High Priority

#### Features
- [ ] Multi-language support (Spanish, French, German)
- [ ] Voice cloning integration
- [ ] Emotion detection and prosody analysis
- [ ] Custom wake word support

#### Developer Experience
- [ ] Hot reload for development
- [ ] Better debugging tools
- [ ] Performance profiling dashboard

### P2 - Medium Priority

#### Features
- [ ] Audio recording and playback
- [ ] Transcript export (PDF, JSON)
- [ ] Voice command system
- [ ] Plugin architecture

#### Testing
- [ ] Integration test suite expansion
- [ ] Load testing framework
- [ ] Fuzz testing for audio pipeline

### P3 - Low Priority

#### Polish
- [ ] UI/UX improvements
- [ ] Documentation expansion
- [ ] Tutorial videos
- [ ] Example applications

## Research & Exploration

### Under Investigation
- Neural codec for audio compression
- On-device model fine-tuning
- Federated learning for privacy-preserving improvements
- Advanced prompt engineering techniques

## Completed Milestones

### Q4 2024
- ✅ HotMem ultra-fast memory system (<200ms p95)
- ✅ Parakeet-MLX streaming STT (<100ms latency)
- ✅ Kokoro TTS optimization (375ms TTFB)
- ✅ Intent-aware multi-source retrieval
- ✅ SOLID/DRY coreference resolution
- ✅ Turn-based summarization system

### Q3 2024
- ✅ Initial LocalCat voice agent implementation
- ✅ WebRTC transport for real-time audio
- ✅ Two-model architecture (conversation + memory)
- ✅ Custom mem0 memory service integration
