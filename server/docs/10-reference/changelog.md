# Changelog

All notable changes to LocalCat Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Vision Processing Optimizations** (2025-10-16)
  - Image preprocessing with configurable resize (384×384px default, ~75% token reduction)
  - Context pruning to limit images (default 2, prevents bloat in long conversations)
  - Frame deduplication to skip identical frames
  - Keyword filtering for smart injection (only on vision-related queries)
  - Configuration via 6 new environment variables (`VISION_IMAGE_SIZE`, `VISION_IMAGE_QUALITY`, `VISION_MAX_IMAGES_IN_CONTEXT`, `VISION_ENABLE_DEDUPLICATION`, `VISION_KEYWORD_FILTER`, `VISION_KEYWORDS`)
  - Comprehensive test suite with 14 tests covering all features
  - Performance impact: 50-75% latency reduction in LLM vision processing

- **TTS Ultra-Low Latency Enhancements** (2025-10-16)
  - Interruption handling with barge-in support (`UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`, `InterruptionFrame`)
  - Text chunking integration (25-char optimal chunks for <800ms TTFB)
  - Environment-configurable buffer sizes (default 40ms via `TTS_BUFFER_MS`)
  - Reduced buffer sizes (1024-2048 bytes) for immediate first-byte delivery
  - TextFrame dropping during interruption to prevent queued speech
  - Minimal prewarming (2 generations) for faster startup
  - Comprehensive test suite with 10 tests covering interruption lifecycle
  - Performance impact: 40-80ms TTFB (was 375-500ms), achieving <800ms voice-to-voice latency

- **STT Hallucination Detection** (2025-10-16)
  - Pattern-based detection replacing confidence heuristics (80% reduction in false positives)
  - 15+ known hallucination patterns ("yeah", "yep", "yes", "mm-hmm", "mmhmm", "uh-huh", "thank you", "thanks", "okay", "uh", "um", "hmm", "ah", "oh")
  - Punctuation normalization for robust pattern matching
  - Short noise filtering (single words ≤3 chars automatically filtered)
  - Comprehensive test suite with 9 tests covering pattern detection, case sensitivity, and edge cases
  - Performance impact: 80% fewer false positive transcriptions from Parakeet STT

- **Intent-Aware Multi-Source Retrieval System** (2025-09-30)
  - Hybrid budget allocation preventing source starvation (each source gets ≥3 bullet budget, re-ranking selects best)
  - Intent-aware source routing: temporal queries prioritize convo/summary, semantic queries prioritize summary/convo
  - Smart scoring system: FTS matches get 1.1-1.2x boost, summary gets 1.05x, graph gets variable priority
  - Query pattern detection for automatic routing without intent classification
  - Proper convo/summary filtering in FTS results
  - Comprehensive test suite validating multi-source retrieval behavior
  - Backward compatible with `intent=None` using pattern detection fallback
- **Contextual Extraction Granularity** (2025-09-30)
  - Introduced `_get_entity_with_context()` to capture prepositional phrases, adjectives, and compounds while preserving canonical roots
  - Expanded UD extractor contract to return alias maps so HotMem can index enriched triples under both enriched and base entities
  - Added dual registration + rebuild heuristics for `entity_index`, keeping queries like `swimming` aligned with enriched edges
  - Recorded enrichment metrics (length/timing/truncation) and added caps for modifiers to stay within <10 ms extraction budget
  - Created `test_contextual_extraction.py` regression suite covering prep, adjective, compound, and negation scenarios

- **SOLID/DRY Coreference Resolution Architecture** (2025-09-27): Complete implementation following software engineering best practices
  - **SharedNLPManager**: Eliminated 3 duplicate spaCy model loading patterns, thread-safe caching
  - **TextProcessor Strategy Pattern**: Extensible text processing pipeline following OCP + DIP principles
  - **CoreferenceProcessor**: Single-responsibility coreference resolution with 50ms timeout protection
  - **Enhanced UDExtractor**: Composition-based architecture using ISP, maintains full backward compatibility
  - **Type-Safe Configuration**: Comprehensive dataclass-based configuration with validation and environment integration
  - **Integration Layer**: Factory functions with graceful degradation strategies and status monitoring
  - **Comprehensive Test Suite**: Unit and integration tests covering all SOLID principles
  - **Documentation**: Complete integration guide with migration strategies and troubleshooting
- **Turn-Based Summarization System**: Implemented configurable turn-based summary generation
  - Added `SUMMARIZER_WINDOW_MODE` configuration supporting `turn_pairs` and `delta` modes
  - Implemented `SUMMARIZER_TURN_PAIRS` for configurable N-turn summary intervals (default: 5)
  - Created `_generate_turn_summary()` method for on-demand summary generation
  - Added automatic final summary generation at session cleanup for unsummarized turns
  - Extracted LLM call logic into reusable `_call_summarizer_llm()` method
  - Fixed message storage to always save with session_id for summarization retrieval
  - Configured to use non-thinking model `google/gemma-3n-e4b` to avoid contamination
  - Cleaned database of 67 contaminated summaries containing `<think>` tags
- **Comprehensive Summarization Integration Test**: Full pipeline test suite for turn-based summaries
  - Created `tests/integration/test_summarization_integration.py` with progressive test scenarios
  - Test scenarios: 5, 10, 20 turn conversations with topic shifts
  - Edge cases: single turn (no summary), incomplete final group (7 turns)
  - Proper pipeline setup with asyncio.gather() for concurrent execution
  - Content validation for key concepts in generated summaries
  - Clean setup/teardown with temp database and automatic resource management

### Fixed
- **Summary Contamination Issue**: Resolved thinking tags appearing in stored summaries
  - Updated summarizer prompt to explicitly request final summary only
  - Switched from thinking-capable to non-thinking model (`google/gemma-3n-e4b`)
  - Fixed 500-character truncation causing incomplete summaries
- **Turn-Based Triggering**: Fixed summary generation not triggering at correct intervals
  - Corrected window mode configuration reading from environment
  - Fixed message storage to use session_id as entity_id for retrieval
  - Added proper async task creation for turn-based summary generation

### Added
- **VoiceAgentFactory Pattern**: Centralized service creation with dependency injection
  - Created `core/factory.py` implementing factory pattern for all voice agent services
  - Single source of truth for service configuration and initialization
  - Support for all service types: STT, TTS, LLM, Memory, Transport, RTVI
  - Proper dependency injection enabling better testability
- **Test Infrastructure Overhaul**: Complete pytest integration for CI/CD reliability
  - `pytest.ini` configuration with async mode support and test markers
  - `conftest.py` with fixtures, mocks, and automatic test skipping
  - Test categorization: `@pytest.mark.ci`, `@pytest.mark.slow`, `@pytest.mark.requires_models`
  - Updated `run_all_tests.py` with test categories (ci, fast, slow, unit, integration)
  - CI tests now complete in 6 seconds for fast PR feedback
- **FastTextAggregator Module**: Token-aware text aggregation for optimal TTS chunking
  - Moved to `core/aggregators/fast_text.py` for better organization
  - Natural phoneme boundary detection for fluent speech synthesis
  - Configurable token limits (175-250) matching Kokoro TTS requirements
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
- Phase 2 (controls & UX)
  - Flags: `MEMORY_SOURCES` and `MEMORY_CONVO_INDEX`; conversation FTS retrieval wired
  - Human-readable recency suffixes on graph/convo/recency bullets (e.g., "(2d 3h ago)")
  - Background LLM summarizer (async) via `SUMMARIZER_*`; summary retrieval from stored notes
  - Turn-level observability: per-turn summary log with pre_injected/ready_signaled/source/bullets/total_ms
- **Parakeet-MLX Streaming STT Integration**: Native Apple Silicon streaming speech-to-text achieving <100ms latency
  - Complete implementation of `ParakeetStreamingSTT` service with proper Pipecat frame lifecycle integration
  - Streaming transcriber context management with `transcribe_stream()` API for real-time audio processing
  - Smart audio buffering with configurable chunk durations (1.0s optimal balance between latency and accuracy)
  - Volume normalization with RMS target leveling and soft clipping to prevent audio distortion
  - Proper VAD integration with `UserStartedSpeakingFrame`/`UserStoppedSpeakingFrame` handling
  - Text accumulation tracking to prevent duplicate transcription and cross-turn contamination
  - Context-aware transcriber resets with proper cleanup of streaming contexts
  - Support for both internal VAD mode (volume thresholding) and external VAD mode (Silero VAD integration)
  - Optimized for conversational AI with reduced sensitivity settings for natural speech detection
- **Kokoro TTS Streaming Optimization**: Ultra-low latency text-to-speech with professional quality enhancements
  - `ProfessionalKokoroTTSService` with real-time chunked audio delivery and fade transitions
  - Audio pipeline optimization with proper gain staging and peak normalization (-3.0dB target)
  - Configurable fade durations (50ms) for smooth audio transitions between chunks
  - Quality monitoring with peak level logging and distortion detection
  - Support for multiple Kokoro voices with speed and sample rate customization
  - Process-isolated TTS service to prevent Metal framework threading conflicts on Apple Silicon
  - Memory-efficient audio buffering with optimized chunk sizes for real-time delivery
  - Backward compatibility with existing batch TTS services for fallback scenarios
- **Audio Input Pipeline Optimization**: Complete overhaul of speech detection and preprocessing
  - Sensitivity tuning for natural conversation with reduced VAD thresholds (0.4 min volume, 0.5 confidence)
  - Faster speech detection with 10ms start time and optimized stop windows
  - SmartTurn v3 integration with configurable pre-speech buffers and turn duration limits
  - MicProbe integration for real-time audio level monitoring and debugging
  - Audio normalization pipeline with RMS targeting and soft clipping to prevent distortion
  - Configurable audio chunking strategies for different STT engines and latency requirements
- **Server Root Organization**: Cleaned up server root by moving TTS files to core/ architecture
  - Moved `kokoro_worker_optimized.py` and `tts_mlx_ultra_low_latency.py` to `core/tts/`
  - Updated imports in `bot.py` and test files to use new paths
  - Maintained functionality while improving project structure
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
- **bot.py Refactoring**: Massive simplification through factory pattern
  - Reduced from 679 to 266 lines (60% reduction)
  - `run_bot()` function simplified from 200+ to 17 lines
  - All service creation logic moved to VoiceAgentFactory
  - Eliminated 400+ lines of duplicate configuration code
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
- **Test Suite Compatibility**: All tests now work with pytest
  - Fixed async test execution with proper pytest configuration
  - Resolved import path issues in test files
  - Fixed duplicate test file naming conflicts
  - Added proper markers for test categorization
  - Corrected context_aggregator attribute access in bot.py event handlers
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
