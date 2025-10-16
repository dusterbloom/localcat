# LocalCat Current Status (October 2025)

**Last Updated:** October 16, 2025
**Version:** 2.0 (Factory Pattern Refactor)

## Executive Summary

LocalCat is a production-ready, fully-local voice agent for macOS featuring:
- **Sub-800ms latency** end-to-end response time
- **Token-aware context management** preventing LLM degradation
- **HotMem memory service** with prosody-enhanced multi-source retrieval
- **Vision processing** with intelligent keyword filtering
- **Audio intelligence** with speaker recognition and enrollment
- **Factory pattern architecture** for clean service composition

---

## Current Architecture (v2.0)

### Core System Components

```
┌─ VoiceAgentFactory (NEW) ────────────────────────┐
│  Centralized service creation                    │
│  ├─ STT Service (Parakeet)                       │
│  ├─ TTS Service (Kokoro MLX)                     │
│  ├─ LLM Service (MiniCPM-V/Gemma/Llama)          │
│  ├─ Memory Service (HotMem)                      │
│  ├─ Vision Service (Context Injector)            │
│  ├─ Audio Intelligence (Speaker Recognition)     │
│  └─ Transport (SmallWebRTC)                      │
└──────────────────────────────────────────────────┘

┌─ Configuration System (NEW) ─────────────────────┐
│  VoiceAgentConfig (Unified)                      │
│  ├─ Base Config (server/config/base_config.py)   │
│  ├─ Settings (server/config/settings.py)         │
│  ├─ Parsers (server/config/parsers.py)           │
│  └─ Environment-driven with validation           │
└──────────────────────────────────────────────────┘

┌─ Memory System (HotMem Service) ─────────────────┐
│  Modular Architecture (66% code reduction)       │
│  ├─ Background Summarizer                        │
│  ├─ Context Injector (Token-aware)               │
│  ├─ Frame Processor                              │
│  ├─ Session Manager                              │
│  ├─ Quality Filter                               │
│  └─ Entity Resolver                              │
│                                                   │
│  Multi-Source Retrieval:                         │
│  ├─ Conversation History (50% weight)            │
│  ├─ Graph Facts (30% weight)                     │
│  ├─ Summaries (20% weight)                       │
│  └─ Semantic Search (optional LEANN)             │
└──────────────────────────────────────────────────┘
```

---

## Current Model Pipeline

Based on `server/bot.py` and `server/.env`:

```
User Speech
    ↓
[Silero VAD] → Voice Activity Detection
    ↓
[Smart Turn v2] → Turn Management (1.5s timeout)
    ↓
[Parakeet STT] → Batch processing with hallucination filtering
    ↓
[Token Estimator] → Count tokens, prune if > 70% of 3000 token limit
    ↓
[Vision Injector] → Add images only for vision keywords
    ↓
[Memory Injector] → Inject 5 bullets (600 token budget)
    ↓
[LLM - MiniCPM-V 4.5] → Generate response (supports vision)
    ↓
[Kokoro TTS MLX] → Ultra-low latency speech (40-80ms TTFB)
    ↓
User Hears Response
```

**Latency Budget:**
- STT: ~100-200ms (Parakeet batch)
- Memory Retrieval: <150ms
- Token Management: <10ms
- LLM: ~200-400ms (varies by model/prompt)
- TTS First Token: 40-80ms
- **Total Target:** <800ms end-to-end

---

## Recent Major Changes (Last 30 Commits)

### October 16, 2025 - Config Unification & Factory Decomposition (853889e)

**What Changed:**
- Created unified `VoiceAgentConfig` with factory pattern
- Centralized all service creation in `VoiceAgentFactory`
- Broke down configuration into modular components
- Added comprehensive configuration validation

**Impact:**
- Reduced bot.py from complex initialization to clean factory calls
- Improved testability with dependency injection
- Single source of truth for configuration
- Easier to swap service implementations

**Files Added/Modified:**
- `server/config/base_config.py` - Base configuration classes
- `server/config/parsers.py` - Configuration parsing utilities
- `server/core/factories/service_factory.py` - Service factory
- `server/tests/unit/test_base_config.py` - Configuration tests

### October 16, 2025 - Dynamic System Prompts (196705f)

**What Changed:**
- System prompts now built dynamically based on enabled features
- Anonymous mode enhancement with proper context handling
- Factory builds system prompt from configuration

**Impact:**
- LLM receives only relevant instructions
- Reduces token usage for unused features
- Improves coherence of system instructions

### October 16, 2025 - Token-Aware Context Management (1a76c27)

**What Changed:**
- Added token counting with tiktoken library
- Intelligent context pruning at 70% threshold (3000 token limit)
- Maintains minimum conversation coherence (4 turns)
- Memory system uses sliding window (4 turn pairs max)

**Impact:**
- **CRITICAL**: Prevents LLM degradation in long conversations
- Keeps context within model limits
- Maintains conversation quality
- Avoids context overflow errors

**Configuration:**
```bash
LLM_CONTEXT_MAX_TOKENS=3000
LLM_CONTEXT_PRUNE_THRESHOLD=0.70
LLM_CONTEXT_MIN_TURNS=4
MEMORY_CONTEXT_SLIDING_WINDOW=true
MEMORY_CONTEXT_MAX_TURN_PAIRS=4
```

### October 15-16, 2025 - Vision/TTS/STT Performance Optimizations (5a66b9b)

**What Changed:**
- **Vision**: Keyword-filtered injection (only for vision-related queries)
- **Vision**: Image deduplication to reduce redundant frames
- **TTS**: Ultra-low latency improvements (40-80ms TTFB target)
- **STT**: Parakeet hallucination filtering with blacklist

**Impact:**
- Vision processing saves significant tokens (only inject when needed)
- Prevents duplicate images in context
- Improved TTS responsiveness
- Cleaner STT output (filters common hallucinations)

**Vision Keywords:**
`see,look,show,what,describe,image,picture,video,color,object,room,view,watch,observe`

### October 15, 2025 - Prosody-Aware Retrieval (c01631b)

**What Changed:**
- Audio intelligence integration with memory system
- Prosody confidence scoring in retrieval
- Video frame processing with throttling
- Enhanced multi-signal memory scoring

**Impact:**
- Memory retrieval weighs prosody confidence (15%)
- Better retrieval quality with audio cues
- Vision frames properly throttled (0.5 FPS)

**Scoring Weights:**
- Conversation: 50%
- Graph: 30%
- Summary: 20%
- Prosody: 15%

### October 14, 2025 - Complete HotMem Modularization (f8b406a)

**What Changed:**
- Broke up 1,100-line HotPathMemoryProcessor God-object
- Created focused components:
  - `background_summarizer.py` (358 lines)
  - `config_manager.py` (424 lines)
  - `context_injector.py` (331 lines)
  - `frame_processor.py` (407 lines)
  - `quality_filter.py` (466 lines)
  - `entity_resolver.py` (272 lines)
  - `semantic_sidecar.py` (604 lines)
  - `session_manager.py` (343 lines)

**Impact:**
- 66% code reduction (1,100 → 373 lines in processor)
- SOLID compliance across all components
- Improved testability
- Easier to understand and maintain

---

## Current Feature Status

### ✅ Production Ready

| Feature | Status | Configuration | Notes |
|---------|--------|---------------|-------|
| **Speech-to-Text** | ✅ Production | `STT_ENGINE=parakeet` | Batch mode with hallucination filtering |
| **Text-to-Speech** | ✅ Production | `VOICE_AGENT_TTS_ENGINE=kokoro_mlx` | Ultra-low latency (40-80ms TTFB) |
| **LLM Integration** | ✅ Production | `LLM_BASE_URL`, `LLM_MODEL` | Supports LM Studio, Ollama |
| **Memory System** | ✅ Production | `MEMORY_HOTPATH_ENABLED=true` | Token-aware, multi-source |
| **Token Management** | ✅ Production | `LLM_CONTEXT_MAX_TOKENS=3000` | Prevents degradation |
| **Vision Processing** | ✅ Production | `VISION_KEYWORD_FILTER=true` | Keyword-filtered, optimized |
| **Speaker Recognition** | ✅ Production | `AUDIO_INTELLIGENCE_ENABLED=true` | Auto-enrollment (3 utterances) |
| **Session Management** | ✅ Production | `SESSION_USE_DATABASE=true` | Persistent sessions |
| **Factory Pattern** | ✅ Production | Built-in | Clean service composition |

### ⚠️ Known Issues

| Issue | Status | Workaround | Priority |
|-------|--------|------------|----------|
| **Emotion Detection** | 🔴 Disabled | `AUDIO_INTEL_ENABLE_EMOTION=false` | Medium |
| **Prosody** | ✅ Working | Integrated with memory | - |

**Emotion Detection Issue:**
- SpeechBrain model API incompatibility
- Error: `'ModuleDict' object has no attribute 'compute_features'`
- Impact: Missing emotion context in memory
- Status: Temporarily disabled, needs investigation

---

## Environment Configuration Summary

### Core Settings (Required)

```bash
# User and Agent Identity
USER_ID=peppi
AGENT_ID=locat

# LLM Configuration
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=minicpm-v-4_5  # or llama3.2:1b, gemma3n:4b

# Token-Aware Context Management (CRITICAL)
LLM_CONTEXT_MAX_TOKENS=3000
LLM_CONTEXT_PRUNE_THRESHOLD=0.70
LLM_CONTEXT_MIN_TURNS=4

# Memory System
MEMORY_ENABLED=true
MEMORY_HOTPATH_ENABLED=true
MEMORY_MAX_BULLETS=5
MEMORY_TOKEN_BUDGET=600
MEMORY_SOURCES=convo,summary,graph,semantic

# STT/TTS
STT_ENGINE=parakeet
VOICE_AGENT_STT_ENGINE=parakeet_batch
VOICE_AGENT_TTS_ENGINE=kokoro_mlx
```

### Optional Features

```bash
# Vision Processing (Saves tokens with keyword filtering)
VIDEO_INPUT_ENABLED=true
VISION_KEYWORD_FILTER=true
VISION_KEYWORDS=see,look,show,what,describe,image,picture,video

# Audio Intelligence
AUDIO_INTELLIGENCE_ENABLED=true
AUDIO_INTEL_INTRO_PIPELINE=true
SPEAKER_AUTO_ENROLL_UTTERANCES=3

# Performance
HF_HUB_OFFLINE=1  # After first run
TTS_PREWARM=true
TARGET_LATENCY_MS=800
```

---

## Testing Status

### Configuration Tests (NEW)

```bash
# Run configuration tests
cd server
pytest tests/unit/test_base_config.py -v
pytest tests/unit/test_config_parsers.py -v
pytest tests/unit/test_voice_agent_config.py -v
```

**Results:** All configuration tests passing

### Memory System Tests

```bash
# HotMem component tests
pytest tests/unit/test_config_manager.py -v
pytest tests/unit/test_context_injector.py -v
pytest tests/unit/test_frame_processor.py -v
pytest tests/unit/test_session_manager.py -v
pytest tests/unit/test_background_summarizer.py -v

# Integration test
pytest tests/integration/test_hotpath_processor_refactor.py -v
```

**Results:** 8/8 unit tests + 1/1 integration test passing

### Vision Processing Tests

```bash
pytest tests/unit/test_vision_context_injector.py -v
```

**Results:** All vision tests passing

---

## Performance Metrics (Apple Silicon M2)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| End-to-End Latency | <800ms | 400-600ms | ✅ Excellent |
| Memory Processing | <200ms | 150-170ms | ✅ Good |
| TTS First Token | <80ms | 40-80ms | ✅ Excellent |
| Token Pruning | <20ms | ~10ms | ✅ Excellent |
| Vision Injection | N/A | <50ms | ✅ Good |

**Resource Usage:**
- Memory: ~500MB baseline
- CPU: 15-25% during conversation
- Startup: 2-5s (with cache), 10-30s (first run)

---

## Next Priorities (From Backlog)

### Phase 2: Retrieval Quality

- [ ] Optional BM25 re-ranking (SQLite FTS5)
- [ ] Optional vector re-ranking (LEANN)
- [ ] Strict budget enforcement

### Phase 3: Observability

- [ ] Per-turn diagnostic logs
- [ ] Metrics export for dashboards
- [ ] Performance monitoring tools

### Phase 4: DX & Config

- [ ] .env presets (minimal/default/advanced)
- [ ] Configuration validation CLI
- [ ] Better documentation

---

## Migration Notes

If upgrading from older versions:

1. **Configuration Variables Changed:**
   - `OPENAI_*` → `LLM_*`
   - `HOTMEM_*` → `MEMORY_*`
   - `VOICE_AGENT_*` → Component-specific prefixes

2. **New Required Variables:**
   - `LLM_CONTEXT_MAX_TOKENS` (default: 3000)
   - `LLM_CONTEXT_PRUNE_THRESHOLD` (default: 0.70)
   - `MEMORY_TOKEN_BUDGET` (default: 600)

3. **Factory Pattern:**
   - Services now created via `VoiceAgentFactory`
   - Configuration via `VoiceAgentConfig.from_env()`

4. **Memory System:**
   - HotMem is now modular
   - Multi-source retrieval by default
   - Token-aware context injection

---

## Quick Start Checklist

- [ ] Copy `server/.env.example` to `server/.env`
- [ ] Set `LLM_BASE_URL` and `LLM_MODEL`
- [ ] Set `USER_ID` and `AGENT_ID`
- [ ] Configure token management (`LLM_CONTEXT_MAX_TOKENS`)
- [ ] Enable desired features (vision, audio intelligence)
- [ ] Run `uv run bot.py` or `python bot.py`
- [ ] First run will download models (~10-30s)
- [ ] Subsequent runs use cache (~2-5s startup)

---

## Documentation Map

- **Getting Started**: `/docs/01-getting-started/`
- **Architecture**: `/docs/02-architecture/`
- **Configuration**: `/docs/01-getting-started/configuration.md`
- **Backlog**: `/docs/08-roadmap/backlog.md`
- **Server Architecture**: `/docs/02-architecture/server-architecture.md`
- **Memory System**: `/docs/02-architecture/memory-system-map.md`

---

**For Support:** See `README.md` or open an issue on GitHub.
