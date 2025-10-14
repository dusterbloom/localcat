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
# OpenAI-compatible endpoint (Ollama default)
OPENAI_BASE_URL=http://127.0.0.1:11434/v1

# Model to use for conversation
OPENAI_MODEL=gemma3n:4b

# Optional API key (not needed for local servers)
OPENAI_API_KEY=

# Enable streaming for lower latency
LLM_USE_STREAMING=true

# Token limits
LLM_MAX_TOKENS=512
LLM_TEMPERATURE=0.7
```

### Voice Pipeline Settings

```bash
# Speech-to-Text
VOICE_AGENT_STT_ENGINE=mlx-whisper
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
WHISPER_LANGUAGE=en

# Text-to-Speech
VOICE_AGENT_TTS_ENGINE=kokoro
TTS_MODEL_PATH=mlx-community/Kokoro-82M-bf16
TTS_ULTRA_LOW_LATENCY=true
KOKORO_BUFFER_MS=40  # Chunk size for streaming

# Voice Activity Detection
VAD_STOP_SECS=0.8  # Silence threshold
VAD_MIN_SPEECH_SECS=0.1
VAD_ACTIVATION_THRESHOLD=0.5
```

## Memory System Configuration

### Basic Memory Settings

```bash
# Enable/disable memory system
MEMORY_ENABLED=true

# Memory bullets per response
MEMORY_BULLETS_MAX=3
MEMORY_BULLETS_MIN=1

# Memory sources to use
MEMORY_SOURCES=graph,summary  # Options: graph, summary, vector

# Session handling
MEMORY_SESSION_HEADER=true
MEMORY_EPHEMERAL_MODE=false  # Privacy mode - no persistence
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

### Speaker Enrollment

```bash
# Enable audio intelligence features
AUDIO_INTELLIGENCE_ENABLED=true

# Enrollment settings
AUDIO_INTEL_ENROLL_ON_STARTUP=false
AUDIO_INTEL_INTRO_PIPELINE=true  # Guided enrollment
AUDIO_INTEL_MIN_ENROLLMENT_SAMPLES=3

# Speaker verification
AUDIO_INTEL_VERIFICATION_THRESHOLD=0.85
AUDIO_INTEL_VERIFICATION_ENABLED=true
```

### Voice Features

```bash
# Prosody analysis
AUDIO_INTEL_PROSODY_ENABLED=false
AUDIO_INTEL_EMOTION_DETECTION=false

# Audio processing
AUDIO_INTEL_NOISE_REDUCTION=true
AUDIO_INTEL_ECHO_CANCELLATION=true
```

## Performance Optimization

### Latency Optimization

```bash
# Ultra-low latency mode
TTS_ULTRA_LOW_LATENCY=true
STT_ULTRA_LOW_LATENCY=true

# Model prewarming
PREWARM_MODELS=true
PREWARM_ON_STARTUP=true

# Pipeline optimization
PIPELINE_PARALLEL_PROCESSING=true
PIPELINE_BUFFER_SIZE=4096
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