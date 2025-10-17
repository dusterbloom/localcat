# Configuration Guide

Complete reference for configuring LocalCat's behavior and performance.

## Configuration Overview

LocalCat uses environment variables for configuration, loaded from `server/.env`. The system follows a hierarchical configuration pattern with sensible defaults that can be overridden.

## Configuration Files

### Primary Configuration
- **Location**: `server/.env`
- **Template**: `server/.env.example`
- **Loader**: `server/config/settings.py`

### Configuration Components
- **Memory System**: `server/core/memory/config_manager.py`
- **Audio Settings**: Inline in `server/bot.py`
- **Model Settings**: Various service configurations

## Core Settings

### LLM Configuration

```bash
# OpenAI-compatible endpoint (LM Studio or Ollama)
LLM_BASE_URL=http://127.0.0.1:1234/v1

# Model to use for conversation
LLM_MODEL=minicpm-v-4_5  # Supports vision (MiniCPM-V) or text-only (llama3.2:1b, gemma3n:4b)

# Optional API key (not needed for local servers)
LLM_API_KEY=not-needed

# Embedding model for semantic memory
LLM_EMBEDDING_MODEL=nomic-embed-text:latest

# Token limits and temperature
LLM_TEMPERATURE=0.7

# Turn management
LLM_AGGREGATION_TIMEOUT=0.12
LLM_TURN_EMULATED_VAD_TIMEOUT=0.3
LLM_ENABLE_EMULATED_VAD_INTERRUPTION=true
```

### Token-Aware Context Management

**NEW: Prevents LLM degradation in long conversations**

```bash
# Maximum tokens for LLM context (reserve room for response)
LLM_CONTEXT_MAX_TOKENS=3000

# Prune at this threshold (0.70 = prune at 70% capacity)
LLM_CONTEXT_PRUNE_THRESHOLD=0.70

# Always keep at least this many recent turns for coherence
LLM_CONTEXT_MIN_TURNS=4

# Memory system context window (turn pairs to keep)
MEMORY_CONTEXT_SLIDING_WINDOW=true
MEMORY_CONTEXT_MAX_TURN_PAIRS=4
```

### Voice Pipeline Settings

```bash
# Speech-to-Text (Parakeet is current default)
STT_ENGINE=parakeet
VOICE_AGENT_STT_ENGINE=parakeet_batch  # or parakeet_streaming
STT_CHUNK_DURATION=1.0
STT_ENABLE_VAD=false

# Parakeet STT Settings (with hallucination filtering)
PARAKEET_CONFIDENCE_THRESHOLD=0.2
PARAKEET_TEMPERATURE=0.0
PARAKEET_SENTENCE_PAUSE_THRESHOLD=1.2
PARAKEET_MAX_CHUNK_DURATION=4.0
PARAKEET_BATCH_CONFIDENCE_THRESHOLD=0.2
# NOTE: Confidence filtering removed - using hallucination blacklist instead

# Text-to-Speech (Kokoro MLX ultra-low latency)
VOICE_AGENT_TTS_ENGINE=kokoro_mlx
VOICE_AGENT_TTS_VOICE=af_heart
VOICE_AGENT_TTS_SPEED=1.0
VOICE_AGENT_TTS_SAMPLE_RATE=24000
VOICE_AGENT_TTS_FADE_DURATION_MS=50.0

# Ultra-low latency settings (40-80ms TTFB)
TTS_PREWARM=true
TTS_BUFFER_MS=40
TTS_MIN_TOKENS=150
TTS_MAX_TOKENS=200
TTS_MODEL=mlx-community/Kokoro-82M-bf16

# Audio quality
TTS_FADE_DURATION_MS=40.0
TTS_TARGET_PEAK_DB=-3.0
TTS_ENABLE_QUALITY_LOGGING=false

# Voice Activity Detection
VAD_CONFIDENCE=0.3
VAD_START_SECS=0.001
VAD_STOP_SECS=0.8
VAD_MIN_VOLUME=0.6

# Smart turn detection
VAD_SMART_TURN_MODEL_PATH=pipecat-ai/smart-turn-v2
VAD_SMART_TURN_STOP_SECS=1.5
VAD_SMART_TURN_PRE_SPEECH_MS=300.0
VAD_SMART_TURN_MAX_DURATION_SECS=16.0
```

## Memory System Configuration

### HotMem Service Settings

**NEW: Modular memory architecture with multi-source retrieval**

```bash
# Enable/disable memory system
MEMORY_ENABLED=true
MEMORY_HOTPATH_ENABLED=true

# Retrieval budget settings (IMPROVED for better context)
MEMORY_MAX_BULLETS=5                    # Increased from 2 → 5 bullets
MEMORY_TOKEN_BUDGET=600                 # Increased from 300 → 600 tokens

# Memory sources to use
MEMORY_SOURCES=convo,summary,graph,semantic

# Source weights for composite scoring
MEMORY_WEIGHT_GRAPH=0.3                 # Graph fact importance
MEMORY_WEIGHT_CONVO=0.5                 # Conversation history importance (BOOSTED)
MEMORY_WEIGHT_SUMMARY=0.2               # Summary importance
MEMORY_WEIGHT_PROSODY=0.15              # Prosody confidence

# Injection mode: "bullets" (legacy) or "headers"
MEMORY_INJECTION_MODE=bullets           # bullets provides more context

# Score threshold for header auto-expand (0.0-1.0)
MEMORY_HEADER_EXPAND_THRESHOLD=0.65

# Session handling
SESSION_USE_DATABASE=true
SESSION_DB_PATH=data/sessions.db
SESSION_PERSISTENCE=true
```

### Advanced Memory Features

```bash
# Coreference Resolution
MEMORY_COREFERENCE_ENABLED=true
MEMORY_COREFERENCE_TIMEOUT_MS=50
MEMORY_COREFERENCE_MODEL_PATH=facebook/bart-large-cnnm

# Intent Classification
INTENT_CLASSIFICATION_ENABLED=true
INTENT_MODEL_PATH=Falconsai/intent_classification
INTENT_MIN_CONFIDENCE=0.75

# Summarization
MEMORY_SUMMARIZER_ENABLED=true
MEMORY_SUMMARIZER_TURN_PAIRS=5
MEMORY_SUMMARIZER_MODEL=qwen2.5:0.5b

# Quality Filtering
MEMORY_QUALITY_MIN_WORDS=3
MEMORY_QUALITY_MAX_WORDS=200
```

### Memory Performance Tuning

```bash
# Processing timeouts
MEMORY_PROCESSING_TIMEOUT_MS=200
MEMORY_RETRIEVAL_TIMEOUT_MS=150

# Cache settings
MEMORY_CACHE_SIZE=1000
MEMORY_CACHE_TTL_SECONDS=3600

# Batch processing
MEMORY_BATCH_SIZE=10
MEMORY_PARALLEL_PROCESSING=true
```

## Audio Intelligence

### Speaker Recognition and Enrollment

```bash
# Enable audio intelligence features
AUDIO_INTELLIGENCE_ENABLED=true
AUDIO_INTEL_USE_MPS=true  # Use Apple Silicon GPU

# Speaker profile settings
SPEAKER_PROFILE_DIR=data/speaker_profiles
SPEAKER_SIMILARITY_THRESHOLD=0.55  # Recognition threshold (lower = more forgiving)
SPEAKER_MIN_UTTERANCE_SEC=1.0
SPEAKER_AUTO_ENROLL_UTTERANCES=3
SPEAKER_CONSISTENCY_THRESHOLD=0.50  # Consistency check

# Enrollment UX (Intro Pipeline)
AUDIO_INTEL_INTRO_PIPELINE=true           # Enable guided enrollment
AUDIO_INTEL_SKIP_FOR_RETURNING=false      # Re-enroll or skip for known speakers
ENABLE_EPHEMERAL_CHOICE=true              # Allow anonymous mode choice
ENROLLMENT_FIXED_PHRASE="A quick brown fox jumped over a lazy dog."
ENROLLMENT_REQUIRE_FIXED_PHRASE=false
BEEP_ON_ENROLL_COMPLETE=true

# Sign-up/Sign-in/Anonymous detection
SIGN_ME_UP_TERMS="sign me up|register me|enroll me|get started|create profile"
SIGN_IN_TERMS="sign in|log in|i'm back|it's me|recognize me"
ANONYMOUS_TERMS="anonymous|private|don't store|do not store|skip|no"
```

### Prosody and Emotion

```bash
# Prosody-aware confidence (integrated with memory)
AUDIO_INTEL_ENABLE_PROSODY=true
CONFIDENCE_STRATEGY=prosody_aware

# Emotion Detection (TEMPORARILY DISABLED - API bug)
AUDIO_INTEL_ENABLE_EMOTION=false

# Summarizer prosody bias
SUMMARY_PROSODY_ENABLED=true
```

## Vision Processing

**NEW: Context-aware video frame processing**

```bash
# Enable video input and vision processing
VIDEO_INPUT_ENABLED=true
VIDEO_TARGET_FPS=0.5  # Capture rate
VIDEO_OUT_ENABLED=false
VISION_MODEL_ENABLED=true

# Vision optimization - only inject images for vision-related queries
VISION_KEYWORD_FILTER=true  # Saves significant tokens!
VISION_KEYWORDS=see,look,show,what,describe,image,picture,video,color,object,room,view,watch,observe

# Image quality and size settings
VISION_IMAGE_SIZE=384  # Resize to this dimension
VISION_IMAGE_QUALITY=85  # JPEG quality (0-100)
VISION_MAX_IMAGES_IN_CONTEXT=2  # Limit images to save tokens
VISION_ENABLE_DEDUPLICATION=true  # Prevent duplicate frames
```

## Performance Optimization

### Latency Optimization

```bash
# Target latency
TARGET_LATENCY_MS=800  # End-to-end target

# Ultra-low latency mode
TTS_ULTRA_LOW_LATENCY=true

# Model prewarming
TTS_PREWARM=true

# Offline mode (after first run)
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCH_HOME=/Users/yourusername/.cache/torch
```

### Resource Management

```bash
# Memory limits
MAX_MEMORY_MB=2048
MEMORY_WARNING_THRESHOLD=0.8

# Thread pools
THREAD_POOL_SIZE=4
ASYNC_WORKERS=2

# Model caching
MODEL_CACHE_DIR=~/.cache/localcat
HF_HUB_OFFLINE=false  # Set true after first run
```

## Development & Debugging

### Logging Configuration

```bash
# Log levels: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
HOTMEM_LOG_LEVEL=WARNING

# Component-specific logging
MEMORY_PROCESSOR_METRICS=false
PIPELINE_DEBUG_MODE=false
AUDIO_DEBUG_LOGGING=false

# Performance metrics
ENABLE_METRICS=true
METRICS_INTERVAL_SECONDS=60
```

### Development Tools

```bash
# Hot reload
DEV_HOT_RELOAD=false

# Mock services
USE_MOCK_LLM=false
USE_MOCK_STT=false
USE_MOCK_TTS=false

# Testing
TEST_MODE=false
TEST_BYPASS_AUTH=false
```

## Configuration Profiles

### Minimal Latency Profile

```bash
# Optimize for speed over quality
WHISPER_MODEL=tiny
TTS_ULTRA_LOW_LATENCY=true
MEMORY_ENABLED=false
INTENT_CLASSIFICATION_ENABLED=false
LLM_MAX_TOKENS=256
VAD_STOP_SECS=0.5
```

### High Quality Profile

```bash
# Optimize for quality over speed
WHISPER_MODEL=medium
TTS_ULTRA_LOW_LATENCY=false
MEMORY_ENABLED=true
MEMORY_BULLETS_MAX=5
INTENT_CLASSIFICATION_ENABLED=true
LLM_MAX_TOKENS=1024
```

### Privacy-First Profile

```bash
# Maximum privacy, no persistence
MEMORY_EPHEMERAL_MODE=true
AUDIO_INTELLIGENCE_ENABLED=false
METRICS_ENABLED=false
LOG_LEVEL=ERROR
HF_HUB_OFFLINE=true
```

### Developer Profile

```bash
# Maximum observability
LOG_LEVEL=DEBUG
HOTMEM_LOG_LEVEL=DEBUG
MEMORY_PROCESSOR_METRICS=true
PIPELINE_DEBUG_MODE=true
ENABLE_METRICS=true
DEV_HOT_RELOAD=true
```

## Environment Variable Precedence

1. **Command line**: `MEMORY_ENABLED=false python bot.py`
2. **Shell environment**: `export MEMORY_ENABLED=false`
3. **.env file**: `MEMORY_ENABLED=true`
4. **Defaults**: Built into code

## Configuration Validation

LocalCat validates configuration on startup:

```python
# Check configuration
python -c "
from server.config.settings import Settings
settings = Settings()
print(settings.model_dump_json(indent=2))
"

# Validate memory configuration
python -c "
from server.core.memory.config_manager import MemoryConfiguration
config = MemoryConfiguration.from_env()
warnings = config.validate_flags()
for warning in warnings:
    print(f'Warning: {warning}')
"
```

## Dynamic Configuration

Some settings can be changed at runtime:

```python
# Via API endpoints
POST /api/config/memory
{
  "enabled": true,
  "bullets_max": 5
}

# Via WebSocket commands
{
  "type": "config_update",
  "config": {
    "ephemeral_mode": true
  }
}
```

## Configuration Best Practices

### 1. Start Simple
Begin with defaults and adjust based on performance needs.

### 2. Profile First
Measure latency before optimizing:
```bash
python scripts/benchmark_latency.py
```

### 3. Version Control
Don't commit `.env` files. Use `.env.example` as template.

### 4. Document Changes
Comment your `.env` file:
```bash
# Reduced for Raspberry Pi deployment
WHISPER_MODEL=tiny  # Was: base
```

### 5. Monitor Impact
Watch metrics when changing configuration:
```bash
MEMORY_PROCESSOR_METRICS=true python bot.py
```

## Troubleshooting Configuration

### Configuration Not Loading
```bash
# Check file exists
ls -la server/.env

# Check syntax
python -c "from dotenv import load_dotenv; load_dotenv('server/.env', verbose=True)"
```

### Conflicting Settings
Some settings override others:
- `MEMORY_EPHEMERAL_MODE=true` overrides all persistence settings
- `TTS_ULTRA_LOW_LATENCY=true` overrides quality settings

### Performance Issues
If experiencing high latency:
1. Check `MEMORY_PROCESSOR_METRICS=true` output
2. Reduce `MEMORY_BULLETS_MAX`
3. Disable `INTENT_CLASSIFICATION_ENABLED`
4. Use smaller models

## Next Steps

- 🚀 Try different [Configuration Profiles](#configuration-profiles)
- 📊 Monitor with [Performance Metrics](../09-reports/performance/)
- 🔧 Set up [Development Environment](../03-development/setup.md)
- 📖 Understand [Memory System](../02-architecture/memory-system.md)

---

**Questions?** See [Environment Variables Reference](../10-reference/environment-vars.md) for complete list.