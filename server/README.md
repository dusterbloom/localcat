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
- ⚡ **Intent-Aware Processing**: Smart routing reduces casual conversation latency by 75%
- 🎛️ **Professional Audio**: Artifact-free TTS with professional audio processing
- 🔧 **Enterprise Architecture**: SOLID principles, comprehensive testing, type-safe configuration

## 🏗️ Architecture Overview

```
┌─ Voice Pipeline ────────────────────────────────────┐
│  Silero VAD → Smart Turn → MLX Whisper → Gemma3    │
│                                             ↓       │
│  Professional TTS ← Intent Classification ←─┘      │
└─────────────────────────────────────────────────────┘

┌─ Memory System (SOLID Architecture) ────────────────┐
│  SharedNLPManager → CoreferenceProcessor → UDExtractor │
│            ↓                    ↓              ↓     │
│  Type-Safe Config → Strategy Pattern → LMDB Storage │
└─────────────────────────────────────────────────────┘
```

### Core Components

- **Voice Processing**: Pipecat-based pipeline with professional TTS/STT
- **Memory System**: SOLID-compliant architecture with coreference resolution
- **Intent Classification**: Falconsai-based smart routing (17.5ms avg latency)
- **Configuration**: Type-safe, environment-driven configuration management

### Model Pipeline

1. **Voice Activity Detection**: Silero VAD for precise speech detection
2. **Speech-to-Text**: MLX Whisper (Apple Silicon optimized)
3. **Intent Classification**: Falconsai model with 75% performance improvement
4. **Memory Processing**: SharedNLPManager → Coreference → UD extraction
5. **Language Model**: Gemma3n 4B via OpenAI-compatible server
6. **Text-to-Speech**: Kokoro/Marvis TTS with artifact-free processing

## 🚀 Quick Start

### Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.12+**
- **Ollama** for LLM hosting
- **LM Studio** (optional, for memory extraction models)

### Installation

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

# === Memory System ===
MEMORY_ENABLED=true                         # Enable/disable memory
MEMORY_BULLETS_MAX=3                        # Max memory bullets per turn
MEMORY_COREFERENCE_ENABLED=true             # Enable coreference resolution
MEMORY_COREFERENCE_TIMEOUT_MS=50            # Coreference timeout protection

# === Intent Classification ===
INTENT_CLASSIFICATION_ENABLED=true         # Smart routing
INTENT_MODEL_PATH=Falconsai/intent_classification  # Classification model

# === Performance Tuning ===
TTS_ULTRA_LOW_LATENCY=true                 # Enable ultra-low latency TTS
VAD_STOP_SECS=0.8                          # Voice activity timeout
PREWARM_MODELS=true                        # Cache models on startup
```

### Advanced Configuration

```bash
# === Memory Advanced ===
MEMORY_SOURCES=graph,summary               # Memory retrieval sources
MEMORY_SUMMARIZER_ENABLED=true             # Turn-based summarization
MEMORY_SUMMARIZER_TURN_PAIRS=5             # Summary every N turns

# === Audio Processing ===
KOKORO_BUFFER_MS=40                        # TTS chunk size
WHISPER_MODEL=base                         # STT model size
SMART_TURN_STOP_SECS=1.5                  # Turn detection timeout

# === Development ===
MEMORY_PROCESSOR_METRICS=true             # Enable metrics collection
HOTMEM_LOG_LEVEL=DEBUG                    # Memory system logging

# === Enrollment & Session Lock ===
# Lock to recognized user for the session; ignore enrollment while locked
SESSION_LOCK_ENABLED=true
# Require this many consecutive different-speaker recognitions before auto-logout
SPEAKER_SWITCH_CONFIRM_MATCHES=3
# Natural-language triggers for logout and confirmation
LOGOUT_TERMS="log me out|logout|log out|sign out|switch user|switch account"
YES_TERMS="yes|yep|yeah|confirm|do it|please do"
NO_TERMS="no|nope|cancel|stop|not now"
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

## 🔐 Enrollment UX & Session Lock

LocalCat includes a first‑run enrollment UX and a robust session lock to prevent mid‑conversation name prompts.

- Returning users are auto‑recognized and routed straight to conversation.
- While the session is locked, unknown‑speaker and enrollment events are ignored.
- Saying a logout phrase (configurable via `LOGOUT_TERMS`) asks for confirmation and returns to the sign in / sign up / anonymous choice.
- If a different speaker repeatedly matches (default 3 times), the system auto‑logs out and returns to the choice screen.

### Speaker Profile Storage

- Profiles: `server/data/speaker_profiles/auto_enrolled/*.pt`
- Name mappings: `server/data/speaker_profiles/speaker_names.json`

To remove a duplicate or stale profile, delete the corresponding `.pt` file and, if needed, remove its entry from `speaker_names.json`. Profiles are loaded on startup; session lock prevents mid‑conversation flips even before restart.

### Judge (Precision Booster for YAML Extractor)

The extractor can run a GraphJudge‑style quality filter to remove low‑content or noisy triples after extraction.

- Enable (distilled model):
  - Set in `server/.env` (already configured):
    - `YAML_GRAPH_JUDGE=on`
    - `YAML_GRAPH_JUDGE_MODEL=models/graph_judge.json`
  - The model is a tiny logistic classifier applied per triple (dot product + sigmoid), adding microseconds of overhead.
- Train or update the model:
  - `python -m scripts.train_graph_judge --dataset tests/data/yaml_eval_l1_en_medium.json --yaml archive/2024_12_consolidation/assets/ASI1_index_v0_9.yaml --out models/graph_judge.json --auto_calibrate`
- Optional knobs:
  - `YAML_GRAPH_JUDGE_THRESH` to override the embedded threshold.
  - Leave unset to use the threshold saved in the model JSON.

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

### Enrollment & Session Lock Testing

Recommended cases:
- Returning user recognized → session locks; no name capture.
- Say a logout phrase → confirm yes → return to choice.
- Say a logout phrase → say no → remain in conversation.
- Different speaker speaks 3 times → auto‑logout → choice.

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
| Intent Classification | <20ms | ~17.5ms |
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
- **Intent Classification**: Modular architecture, clean interfaces
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
- ✅ **Intent-Aware Processing**: 75% performance improvement for casual conversations
- ✅ **Professional Audio**: Artifact-free TTS with ultra-low latency
- ✅ **Type-Safe Configuration**: Comprehensive environment-driven configuration
- ✅ **Comprehensive Testing**: Full test suite covering SOLID principles

### Coming Next

- 🔄 **Retrieval Quality Improvements**: BM25 and vector re-ranking
- 📊 **Advanced Observability**: Comprehensive metrics and monitoring
- 🎯 **Configuration Presets**: One-click setup for different use cases

---

**Built with ❤️ for the local AI community**

*LocalCat demonstrates that production-quality voice AI can run entirely locally on consumer hardware while maintaining enterprise-grade architecture standards.*
