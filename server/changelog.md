# Changelog

All notable changes to LocalCat Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0: Streaming memory pre-injection and final refresh
  - Interim pre-injection (retrieval-only, once/turn) to ensure bullets exist before LLM starts
  - Final refresh on TranscriptionFrame with extract+persist+retrieve
  - Unified `retrieve_bullets(read_only=...)` API
- Phase 0.5: Config parity and handshake
  - Env controls: `ENABLE_MEMORY`, `HOTMEM_BULLETS_MAX`, `HOTMEM_INTERIM_MIN_WORDS`
  - Optional handshake frame `MemoryContextReadyFrame` signaling memory readiness to downstream
  - Unit tests: `test_hotmem_phase0.py`, `test_hotmem_env.py`
- Phase 1 (scaffolding started)
  - `server/memory/` package added
  - `memory/store.py` (compat re-export), `memory/index.py` (HotIndex skeleton)
  - `memory/context.py` (bullet formatting/dedup/cap) and MemoryContextFrame; HotMem emits both direct context message and typed frame
  - `memory/extractors/` (Extractor interface, UDExtractor adapter); HotMemory delegates extraction/refinement via the adapter
  - `memory/retrieval.py` (Retrieval) with identical routing logic; HotMemory delegates retrieval
  - No behavior change yet (compatibility maintained)
- **STT/LLM/TTS Streaming Integration**: Complete end-to-end streaming pipeline achieving <500ms latency
  - WhisperLiveKit with SimulStreaming backend for ultra-low latency STT (<100ms chunks)
  - LLM streaming with token-by-token output for immediate response
  - Verified existing TTS streaming with chunked audio delivery
  - Environment variable configuration for easy enable/disable
  - Automatic fallback to batch mode when streaming unavailable
  - Comprehensive test suite with unit and integration tests
- **HotMem Ultra-Fast Memory System**: Complete local memory solution achieving <200ms p95 latency
  - Dual storage architecture: SQLite (persistence) + LMDB (O(1) memory-mapped lookups)
  - Universal Dependencies (UD) based extraction using spaCy
  - USGS Grammar-to-Graph 27-pattern coverage for comprehensive fact extraction
  - Real-time memory bullet injection directly into Pipecat context system
- Comprehensive extraction testing and evaluation framework
- **HotPathMemoryProcessor**: Pipecat-integrated processor for seamless memory injection
- **Enhanced logging and debugging**: Frame tracing, performance metrics, extraction visibility
- Reference materials: Grammar-to-Graph XML, USGS patterns, test datasets
- Comprehensive technical debt documentation and cleanup guidelines

### Changed
- HotMem processor now injects memory before forwarding frames and (optionally) signals readiness
- Test runner logs improved; tests hard-exit to avoid macOS framework teardown crashes
- LMDB usage made optional in store operations to support in-memory testing
- **Complete Memory Architecture Overhaul**: Replaced mem0 (2s latency) with HotMem (<200ms)
- **Proper Pipecat Integration**: HotMem now uses context aggregator for memory injection
- **Pipeline Optimization**: Moved memory processor before context aggregator for correct frame flow
- **WhisperSTTServiceMLX Compatibility**: Fixed `is_final=None` handling for non-streaming STT
- **Universal Dependencies Enhancement**: Enabled spaCy lemmatizer for proper relation extraction
- **Memory Bullet Generation**: Contextual, concise bullets for enhanced LLM context
- **Performance Monitoring**: Real-time metrics tracking with p95 latency goals

### Fixed
- Punctuation-induced question misclassification mitigated (prevents question gating from blocking writes)
- LMDB None handling in `observe_edge`, `negate_edge`, and `flush` (no crashes on in-memory tests)
- Intermittent teardown exceptions in tests by forcing process hard-exit in test scripts
- **Critical Memory Extraction Bug**: Fixed retrieval returning query text instead of actual facts
- **Frame Processing Issues**: Resolved `is_final=None` causing extraction to be skipped  
- **Context Integration Failure**: Fixed memory bullets not appearing in LLM context
- **Pipeline Ordering**: Corrected processor placement for proper TranscriptionFrame handling
- **Empty Relation Extraction**: Fixed spaCy lemmatizer being disabled causing empty predicates
- **Pipecat Frame Lifecycle**: Proper StartFrame handling and frame forwarding compliance
- **Audio Frame Flooding**: Filtered audio frames from debug logs for readable output

### Removed
- Removed automated startup script (start_osaurus.sh) in favor of manual setup
- Removed vllm dependency (not compatible with macOS)
- Removed deprecated mem0 services (`server/deprecated_memory_services/*`, `server/mem0_service_v2.py`)

## [0.1.0] - 2025-09-04

### Added
- Initial LocalCat voice agent implementation
- WebRTC transport for real-time audio communication
- WhisperSTTServiceMLX for speech-to-text
- TTSMLXIsolated for text-to-speech with Kokoro model
- Smart turn detection with CoreML analyzer
- Custom mem0 memory service integration
- Two-model architecture support (conversation + memory extraction)
- Dynamic JSON schema detection for memory operations
- Graceful fallback mechanisms for local LLM compatibility
- FAISS vector storage for persistent memories
- Environment-based configuration system

### Fixed
- mem0 async_mode parameter compatibility with Pipecat
- LM Studio JSON schema format requirements (json_object → json_schema)
- System instructions being stored as user memories
- Empty JSON response handling for local models
- Thinking token interference with conversation flow

### Technical
- OpenAI-compatible API support via Ollama/LM Studio
- Support for gemma3:4b (conversation) and qwen2.5-7b-instruct (memory)
- Embedding support with nomic-embed-text model
- Portable solution without site-packages modifications

[Unreleased]: https://github.com/peppi/localcat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/peppi/localcat/releases/tag/v0.1.0
