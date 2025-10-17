# 🐱 LocalCat Server

> **Ultra-Fast Local Voice AI with Persistent Memory**

A production-ready voice assistant server built for Apple Silicon, featuring sub-800ms end-to-end latency with advanced memory capabilities and SOLID architecture principles.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Built with Pipecat](https://img.shields.io/badge/built%20with-Pipecat-purple)](https://github.com/pipecat-ai/pipecat)
[![Apple Silicon Optimized](https://img.shields.io/badge/Apple%20Silicon-Optimized-lightgrey.svg)]()
[![SOLID Principles](https://img.shields.io/badge/SOLID-Compliant-green.svg)]()

## ✨ Key Features

- 🎤 **Ultra-Low Latency Voice**: <800ms end-to-end response time with WebRTC transport
- 🧠 **Intelligent Memory System**: SOLID-compliant architecture with coreference resolution (85-95% accuracy)
- 🏠 **Fully Local**: Works with Ollama/LM Studio, zero cloud dependencies
- 🎛️ **Professional Audio**: Artifact-free TTS with professional audio processing
- 🔧 **Enterprise Architecture**: SOLID principles, comprehensive testing, type-safe configuration

## 🏗️ Architecture Overview

```
┌─ Voice Pipeline ────────────────────────────────────┐
│  Silero VAD → Smart Turn → Parakeet → Ollama LLM    │
│                                             ↓       │
│  Kokoro TTS ← Memory injection / query    ←─┘       │
└─────────────────────────────────────────────────────┘

┌─ Memory System (SOLID Architecture) ────────────────┐
│  SharedNLPManager → CoreferenceProcessor → UDExtractor │
│            ↓                    ↓              ↓     │
│  Type-Safe Config → Strategy Pattern → LMDB Storage │
└─────────────────────────────────────────────────────┘
```

### Core Components

- **Voice Processing**: Pipecat-based pipeline with Parakeet STT and kokoro TTS
- **Memory System**: SOLID-compliant architecture with coreference resolution
- **Configuration**: Type-safe, environment-driven configuration management

### Model Pipeline

1. **Voice Activity Detection**: Silero VAD for precise speech detection
2. **Speech-to-Text**: MLX Parakeet streaming (Apple Silicon optimized)
3. **Memory Processing**: SharedNLPManager → Coreference → UD extraction
4. **Language Model**: Any local llm via OpenAI-compatible server
5. **Text-to-Speech**: Kokoro TTS with artifact-free processing

## 🚀 Quick Start

### Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.12+**
- **Ollama** for LLM hosting
- **LM Studio** (optional, for memory extraction models)

### Server Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd localcat/server

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Install required models:**
```bash
# Core conversation model
ollama pull gemma3n:4b

# Speech recognition (downloads automatically on first run)
# TTS models (cached automatically)
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your preferences
```

4. **Run the server:**
```bash
# Quick start (models download automatically)
python bot.py

# Or with offline mode (after first run)
HF_HUB_OFFLINE=1 python bot.py
```

5. **Connect client:**
   - Open the Next.js client (in `../client/`)
   - Or use any WebRTC-compatible voice client
   - Connect to `http://localhost:7860`

## ⚙️ Configuration

### Essential Environment Variables

```bash
# === Core Settings ===
OPENAI_BASE_URL=http://127.0.0.1:11434/v1  # Ollama endpoint
OPENAI_MODEL=gemma3n:4b                     # Main conversation model
AGENT_ID=localcat                           # Agent identifier
USER_ID=your-user-id                        # User identifiers

# === Memory System ===
MEMORY_ENABLED=true                         # Enable/disable memory
MEMORY_BULLETS_MAX=3                        # Max memory bullets per turn
MEMORY_COREFERENCE_ENABLED=true             # Enable coreference resolution
MEMORY_COREFERENCE_TIMEOUT_MS=50            # Coreference timeout protection



# === Performance Tuning ===
TTS_ULTRA_LOW_LATENCY=true                 # Enable ultra-low latency TTS
VAD_STOP_SECS=0.8                          # Voice activity timeout
PREWARM_MODELS=true                        # Cache models on startup
```

### Advanced Configuration

```bash
# LocalCat Voice Agent Configuration
# Single source of truth - consolidated from dual .env files
# Generated: 2025-09-26

#########################
# Core Agent Settings
#########################
USER_ID=your-name
AGENT_ID=agent-name

#########################
# Speech-to-Text (STT)
#########################
STT_ENGINE=parakeet
STT_CONFIDENCE_THRESHOLD=0.1
STT_CHUNK_DURATION=1.0
STT_ENABLE_VAD=false


#########################
# Text-to-Speech (TTS)
#########################
TTS_ENGINE=kokoro_mlx
TTS_VOICE=af_heart
TTS_SPEED=1.0

# Ultra-low latency settings (40-80ms TTFB)
TTS_PREWARM=true
TTS_BUFFER_MS=50
TTS_MIN_TOKENS=175
TTS_MAX_TOKENS=250
TTS_MODEL=mlx-community/Kokoro-82M-bf16

# Audio quality
TTS_FADE_DURATION_MS=50.0
TTS_TARGET_PEAK_DB=-3.0
TTS_ENABLE_QUALITY_LOGGING=true

#########################
# Language Model (LLM)
#########################
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=not-needed
LLM_MODEL=llama3.2:1b
LLM_TEMPERATURE=0.7
LLM_EMBEDDING_MODEL=nomic-embed-text:latest

# Turn management
LLM_AGGREGATION_TIMEOUT=0.2
LLM_TURN_EMULATED_VAD_TIMEOUT=0.5
LLM_ENABLE_EMULATED_VAD_INTERRUPTION=true

#########################
# Voice Activity Detection (VAD)
#########################
VAD_CONFIDENCE=0.5
VAD_START_SECS=0.1
VAD_STOP_SECS=0.8
VAD_MIN_VOLUME=0.4

# Smart turn detection
VAD_SMART_TURN_MODEL_PATH=pipecat-ai/smart-turn-v2
VAD_SMART_TURN_STOP_SECS=1.5
VAD_SMART_TURN_PRE_SPEECH_MS=300.0
VAD_SMART_TURN_MAX_DURATION_SECS=16.0

#########################
# Memory System
#########################
MEMORY_ENABLED=true
MEMORY_HOTPATH_ENABLED=true
# Memory backend: 'hotpath' (current processor) or 'hotmem' (Pipecat-compatible service)
MEMORY_BACKEND=hotpath
MEMORY_BULLETS_MAX=3
MEMORY_INTERIM_MIN_WORDS=6
MEMORY_ENABLE_HANDSHAKE=true
MEMORY_SOURCES=graph,convo,summary
MEMORY_CONVO_INDEX=true

# Coreference
MEMORY_COREFERENCE_ENABLED=true
MEMORY_COREFERENCE_TIMEOUT_MS=50
MEMORY_COREFERENCE_MIN_LENGTH=10

# Storage paths (relative to server/)
MEMORY_SQLITE_PATH=../data/memory.db
MEMORY_LMDB_PATH=../data/graph.lmdb

# Semantic retrieval (LEANN)
MEMORY_USE_LEANN=true
MEMORY_LEANN_INDEX_PATH=../data/memory_vectors.leann
MEMORY_LEANN_BACKEND=hnsw
MEMORY_LEANN_COMPLEXITY=16
MEMORY_REBUILD_LEANN_ON_SESSION_END=true

# Extraction settings
MEMORY_DECOMPOSE_CLAUSES=false
MEMORY_EXTRA_CONFIDENCE=false
MEMORY_CONFIDENCE_THRESHOLD=0.3
MEMORY_BYPASS_CONFIDENCE_FOR_BASIC=true
MEMORY_CONFIDENCE_FLOOR_BASIC=0.6

# Injection formatting
MEMORY_INJECT_ROLE=system
MEMORY_INJECT_HEADER=Use the following factual context if helpful.

# Logging
MEMORY_LOG_FILE=.logs/hotmem.log
MEMORY_CONSOLE_DEBUG=true
MEMORY_LOG_LEVEL=WARNING
MEMORY_TRACE_FRAMES=false

#########################
# Session Management
#########################
SESSION_USE_DATABASE=true
SESSION_DB_PATH=data/sessions.db
SESSION_PERSISTENCE=true

#########################
# Summarization (DISABLED)
#########################
MEMORY_SUMMARIZER_ENABLED=false

# Settings if re-enabling summarizer
# MEMORY_SUMMARIZER_MODEL=google/gemma-3n-e4b
# MEMORY_SUMMARIZER_BASE_URL=http://127.0.0.1:1234/v1
# MEMORY_SUMMARIZER_API_KEY=
# MEMORY_SUMMARIZER_MAX_TOKENS=120
# MEMORY_SUMMARIZER_WINDOW_MODE=turn_pairs
# MEMORY_SUMMARIZER_TURN_PAIRS=10
# MEMORY_SUMMARIZER_INTERVAL_SECS=300

#########################
# Performance & Debug
#########################
TARGET_LATENCY_MS=800
DEBUG_MODE=false
LOG_LEVEL=WARNING
ENABLE_PERFORMANCE_LOGGING=false

#########################
# Parakeet STT Settings
#########################
PARAKEET_CONFIDENCE_THRESHOLD=0.1
PARAKEET_TEMPERATURE=0.0
PARAKEET_SENTENCE_PAUSE_THRESHOLD=1.2
PARAKEET_MAX_CHUNK_DURATION=4.0
PARAKEET_CONTEXT_SIZE=256,256
PARAKEET_DEPTH=3
PARAKEET_VOLUME_THRESHOLD=0.001

# Legacy/Deprecated (kept for reference)
# VOICE_AGENT_* prefixes deprecated - use domain-specific prefixes above
# HOTMEM_* prefixes deprecated - use MEMORY_* prefix
# KOKORO_* individual settings deprecated - use TTS_* prefix
# SUMMARIZER_ENABLED deprecated - use MEMORY_SUMMARIZER_ENABLED

```

## 🧠 Memory System Features

LocalCat features a sophisticated memory system built with SOLID principles:

### SOLID Architecture Compliance

- ✅ **Single Responsibility**: Each component has one focused purpose
- ✅ **Open/Closed**: Extensible via strategy pattern without modification
- ✅ **Liskov Substitution**: All implementations respect interface contracts
- ✅ **Interface Segregation**: No forced dependencies on unused interfaces
- ✅ **Dependency Inversion**: Depends on abstractions, not concretions

### Key Components

```
core/memory/
├── nlp_manager.py              # Consolidated model management (DRY)
├── config.py                   # Type-safe configuration
├── processors/
│   ├── base.py                # TextProcessor strategy interface
│   └── coreference.py         # Coreference resolution with timeout
├── extractors/
│   └── ud.py                  # Enhanced dependency parsing
└── coreference_integration.py # Factory functions & monitoring
```

### Coreference Resolution

**Before:**
```
"John went to the store. He bought milk."
→ Misses connection between "He" and "John"
```

**After:**
```
"John went to the store. He bought milk."
→ Resolves "He" → "John"
→ Extracts: [("john", "went_to", "store"), ("john", "bought", "milk")]
```

### Performance

- **Accuracy**: 70-85% → 85-95% (15% improvement with coreference)
- **Latency**: <200ms p95 (including 50ms timeout protection)
- **Memory Usage**: Shared model caching reduces resource consumption
- **Error Handling**: Graceful fallbacks, never crashes on failures

## 🎯 Intent Classification

Smart routing system that optimizes performance based on conversation intent:

```python
# Greeting Detection → Skip Memory Processing
"Hello!" → Casual Intent → 150ms saved per turn

# Memory Queries → Full Processing
"What's my name?" → Recall Intent → Full memory pipeline

# Corrections → Enhanced Processing
"Actually, I meant..." → Correction Intent → Deletion-focused processing
```

### Performance Impact

- **Average Classification**: 17.5ms (well under 20ms budget)
- **Casual Conversation**: 75% performance improvement
- **Success Rate**: 100% with 0% fallback rate in testing
- **Memory Savings**: Skip processing for greetings, confirmations, casual chat

## 🔧 Development & Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest -m "ci"          # Fast CI tests
python -m pytest -m "integration" # Integration tests
python -m pytest -m "slow"        # Comprehensive tests

# Run with coverage
python -m pytest --cov=core
```

### Memory System Testing

```bash
# Test coreference integration
python -m pytest tests/unit/test_coreference_integration.py -v

# Test intent classification
python -m pytest tests/unit/test_intent_integration.py -v

# Test memory pipeline
python -m pytest tests/unit/test_hotmem_comprehensive.py -v
```

### Performance Benchmarking

```bash
# Benchmark memory system
python scripts/benchmark_memory.py

# Test end-to-end latency
python scripts/test_latency.py

# Monitor real-time performance
python scripts/monitor_performance.py
```

## 📊 Performance Metrics

### Target Performance (Apple Silicon M2)

| Component | Target | Achieved |
|-----------|--------|----------|
| End-to-End Latency | <800ms | ~400-600ms |
| Memory Processing | <200ms | ~150-170ms |
| TTS First Token | <80ms | ~40-80ms |
| Memory Accuracy | 85%+ | 85-95% |

### Resource Usage

- **Memory**: ~500MB baseline (including models)
- **CPU**: ~15-25% during conversation (M2)
- **Startup Time**: ~10-30s (first run with downloads)
- **Startup Time**: ~2-5s (subsequent runs with cache)

## 🏗️ Architecture Principles

### SOLID Compliance

Every major component follows SOLID principles:

- **Memory System**: Strategy pattern, dependency injection, single responsibilities
- **Configuration**: Type-safe, validated, hierarchical structure
- **Audio Processing**: Composition-based, extensible pipeline

### Error Handling Philosophy

- **Fail-Safe**: Systems never crash, always provide fallbacks
- **Timeout Protection**: Hard limits prevent latency spikes
- **Graceful Degradation**: Reduced functionality rather than failures
- **Comprehensive Logging**: Detailed observability for debugging

### Performance Philosophy

- **Latency First**: Sub-800ms response time is non-negotiable
- **Resource Efficient**: Shared caching, model reuse
- **Scalable Architecture**: Clean separation of concerns
- **Apple Silicon Optimized**: MLX models, Metal framework usage

## 📚 Documentation

- **[Integration Guide](docs/coreference_integration_guide.md)**: Complete coreference setup
- **[Development Backlog](backlog.md)**: Detailed progress tracking
- **[Technical Debt](techdebt.md)**: Architecture improvements made
- **[Changelog](changelog.md)**: Version history and updates

## 🔍 Troubleshooting

### Common Issues

**"Model not found" errors:**
```bash
# Check model availability
ollama list

# Re-download if missing
ollama pull gemma3n:4b
```

**High latency (>800ms):**
```bash
# Check configuration
grep -E "TIMEOUT|LATENCY" .env

# Monitor performance
python scripts/monitor_performance.py
```

**Memory processing disabled:**
```bash
# Check memory configuration
python -c "
from core.memory.config import get_memory_config
config = get_memory_config()
print(f'Memory enabled: {config.enabled}')
print(f'Coreference enabled: {config.coreference.enabled}')
"
```

**Audio artifacts:**
```bash
# Enable professional audio processing
echo "TTS_ULTRA_LOW_LATENCY=true" >> .env
```

### Debug Mode

```bash
# Enable detailed logging
export HOTMEM_LOG_LEVEL=DEBUG
export MEMORY_PROCESSOR_METRICS=true

# Run with debug output
python bot.py
```

## 🤝 Contributing

We welcome contributions! Please see our development guidelines:

1. **Follow SOLID Principles**: All new code should adhere to SOLID design principles
2. **Maintain Performance**: Keep latency under budgets (<200ms memory, <800ms end-to-end)
3. **Add Tests**: Comprehensive test coverage for new features
4. **Document Changes**: Update relevant documentation and changelog

### Architecture Guidelines

- **Single Responsibility**: Each class/module has one clear purpose
- **Strategy Pattern**: Use for extensible behavior (see TextProcessor)
- **Dependency Injection**: Avoid tight coupling
- **Type Safety**: Use dataclasses and proper typing
- **Error Handling**: Fail-safe with graceful fallbacks

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🎉 What's New

### Latest Features (September 2025)

- ✅ **SOLID/DRY Coreference Architecture**: Complete rewrite following software engineering best practices
- ✅ **Professional Audio**: Artifact-free STT/TTS with ultra-low latency
- ✅ **Type-Safe Configuration**: Comprehensive environment-driven configuration
- ✅ **Comprehensive Testing**: Full test suite covering SOLID principles

### Coming Next

- 🔄 **Retrieval Quality Improvements**: BM25 and vector re-ranking
- 📊 **Advanced Observability**: Comprehensive metrics and monitoring
- 🎯 **Configuration Presets**: One-click setup for different use cases

---

**Built with ❤️ for the local AI community**

Shout outs:
pipecat.ai , nvidia parakeet, kokoro, ollama, sqlite, faiss, mem0, small language models that kick ass (all the qwen3s, llama3.2:1b, mistral)

*LocalCat demonstrates that production-quality voice AI can run entirely locally on consumer hardware while maintaining enterprise-grade architecture standards.*
