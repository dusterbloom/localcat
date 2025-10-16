# LocalCat - Local Voice Agent on macOS with Pipecat

![screenshot](assets/debug-console-screenshot.png)

Pipecat is an open-source, vendor-neutral framework for building real-time voice (and video) AI applications.

This repository contains a production-ready voice agent running entirely with local models on macOS. On an M-series Mac, you can achieve voice-to-voice latency of <800ms with relatively strong models, featuring persistent memory, vision processing, and speaker recognition.

## Current Model Pipeline

The [server/bot.py](server/bot.py) file uses:

  - **VAD**: Silero (voice activity detection)
  - **Turn Management**: Smart-turn v2
  - **STT**: Parakeet (batch processing with hallucination filtering)
  - **LLM**: Configurable via local OpenAI-compatible server (Gemma3n 4B, MiniCPM-V 4.5, etc.)
  - **TTS**: Kokoro MLX (ultra-low latency 40-80ms TTFB)
  - **Memory**: HotMem service (token-aware, prosody-enhanced)
  - **Vision**: Optional video processing with keyword filtering

You can swap any of these out for other models or completely reconfigure the pipeline. The system supports tool calling, MCP server integrations, parallel pipelines for async inference, custom processing steps, and flexible interrupt handling.

The bot and web client here communicate using a low-latency, local, serverless WebRTC connection. For more information on serverless WebRTC, see the Pipecat [SmallWebRTCTransport docs](https://docs.pipecat.ai/server/services/transport/small-webrtc) and this [article](https://www.daily.co/blog/you-dont-need-a-webrtc-server-for-your-voice-agents/). You could switch over to a different Pipecat transport (for example, a WebSocket-based transport), but WebRTC is the best choice for realtime audio.

For a deep dive into voice AI, including network transport, optimizing for latency, and notes on designing tool calling and complex workflows, see the [Voice AI & Voice Agents Illustrated Guide](https://voiceaiandvoiceagents.com/).

# Getting Started

## Quick Start

1) **Configure environment**

```bash
cp server/.env.example server/.env
# Edit server/.env to configure your setup
```

Key configuration variables:
```bash
# LLM Configuration (LM Studio or Ollama)
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=minicpm-v-4_5  # or llama3.2:1b, gemma3n:4b, etc.

# Core Agent Settings
USER_ID=your_username
AGENT_ID=locat

# Memory System
MEMORY_ENABLED=true
MEMORY_HOTPATH_ENABLED=true

# Audio Intelligence (Speaker Recognition)
AUDIO_INTELLIGENCE_ENABLED=true
AUDIO_INTEL_INTRO_PIPELINE=true

# Vision Processing
VIDEO_INPUT_ENABLED=true
VISION_KEYWORD_FILTER=true
```

2) **Start the server**

```bash
cd server
# Using uv (recommended)
uv run bot.py

# Or using pip
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py
```

3) **Start the web client**

```bash
cd client
npm i
npm run dev
```

4) **Optional: Speed up startup after first run**

```bash
cd server
HF_HUB_OFFLINE=1 uv run bot.py
```

# Key Features

## Advanced Capabilities

- **🧠 HotMem Memory Service**: Token-aware persistent memory with prosody-enhanced retrieval
  - Prevents LLM degradation with intelligent context pruning (3000 token limit, 70% threshold)
  - Multi-source retrieval: conversation history, graph facts, summaries, semantic search
  - Session management with speaker-specific memory

- **🎤 Audio Intelligence**: Speaker recognition and enrollment
  - Automatic speaker enrollment (3 utterances)
  - Privacy-first with ephemeral/anonymous modes
  - Prosody-aware confidence scoring

- **👁️ Vision Processing**: Context-aware video frame processing
  - Keyword-filtered vision injection (only for vision-related queries)
  - Image deduplication to reduce token usage
  - Configurable image quality and size

- **⚡ Ultra-Low Latency**: <800ms end-to-end response time
  - Parakeet STT with hallucination filtering
  - Kokoro TTS with 40-80ms TTFB
  - Token-based streaming and chunking

## Environment Setup

Copy the example env file and adjust values for your setup:
```bash
cp server/.env.example server/.env
```

**Core Configuration Variables:**
- `LLM_BASE_URL`, `LLM_MODEL` - Your local LLM server (LM Studio or Ollama)
- `STT_ENGINE` - Speech-to-text engine (parakeet, parakeet_batch, parakeet_streaming)
- `VOICE_AGENT_TTS_ENGINE` - Text-to-speech (kokoro_mlx recommended)
- `MEMORY_ENABLED`, `MEMORY_HOTPATH_ENABLED` - Memory system toggles
- `AUDIO_INTELLIGENCE_ENABLED` - Speaker recognition
- `VIDEO_INPUT_ENABLED`, `VISION_MODEL_ENABLED` - Vision processing

**Run a local OpenAI-compatible LLM server:**
- **LM Studio**: Start server from Developer tab, supports vision models (MiniCPM-V, etc.)
- **Ollama**: Supports text-only models with OpenAI-compatible endpoint

# Models and dependencies

Silero VAD and MLX Whisper run inside the Pipecat process. When the agent code starts, it will need to download model weights that aren't already cached, so first startup can take some time.

The LLM service in this bot uses the OpenAI-compatible chat completion HTTP API. So you will need to run a local OpenAI-compatible LLM server. 

One easy, high-performance, way to run a local LLM server on macOS is [LM Studio](https://lmstudio.ai/). From inside the LM Studio graphical interface, go to the "Developer" tab on the far left to start an HTTP server.

# Run the voice agent

The core voice agent code lives in a single file: [server/bot.py](server/bot.py). There's one custom service here that's not included in Pipecat core: we implemented a local MLX-Audio frame processor on top of the excellent [mlx-audio library](https://github.com/Blaizzy/mlx-audio).

Note that the first time you start the bot it will take some time to initialize the three models. It can be 30 seconds or more before the bot is fully ready to go. Subsequent startups will be much faster.

It's not a bad idea to run a quick `mlx-audio.generate` process from the command line before you run the bot the first time, so you're not waiting for a relatively bug HuggingFace model download for the voice model.

```shell
mlx-audio.generate --model "Marvis-AI/marvis-tts-250m-v0.1" --text "Hello, I'm Pipecat!" --output "output.wav"
# or
mlx-audio.generate --model "mlx-community/Kokoro-82M-bf16" --text "Hello, I'm Pipecat!" --output "output.wav"
```

```shell
cd server/
```

If you're using uv

```
uv run bot.py
```

If you're using pip

```
python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python bot.py
```

After you run the first time and have all the models cached, you can set the HF_HUB_OFFLINE environment variable to prevent the Hugging Face libraries from going to the network and checking for model updates. This makes the initial bot startup and first conversation turn a lot faster.

```
HF_HUB_OFFLINE=1 uv run bot.py
```

# Start the web client

The web client is a React app. You can connect to your local macOS agent using any client that can negotiate a serverless WebRTC connection. The client in this repo is based on [voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit) and just uses that library's standard debug console template.

```shell
cd client/

npm i

npm run dev

# Navigate to URL shown in terminal in your web browser
```

# Configuration Quick Reference

Server configuration is environment-driven and loaded from `server/.env`:
- **Unified Config**: `server/config/settings.py` - VoiceAgentConfig with factory pattern
- **Memory Config**: `server/core/memory/config_manager.py` - MemoryConfiguration
- **Service Factory**: `server/core/factories/service_factory.py` - Centralized service creation

## Common Configuration Tweaks

**Performance Optimization:**
```bash
# Token-aware context management (prevents LLM degradation)
LLM_CONTEXT_MAX_TOKENS=3000           # Maximum context size
LLM_CONTEXT_PRUNE_THRESHOLD=0.70      # Prune at 70% capacity
LLM_CONTEXT_MIN_TURNS=4               # Minimum conversation history to keep

# Ultra-low latency TTS
TTS_BUFFER_MS=40                      # 40-80ms TTFB target
TTS_MIN_TOKENS=150
TTS_MAX_TOKENS=200
```

**Memory System:**
```bash
MEMORY_HOTPATH_ENABLED=true           # Enable HotMem service
MEMORY_TOKEN_BUDGET=600               # Token budget for memory context
MEMORY_MAX_BULLETS=5                  # Maximum memory bullets
MEMORY_INJECTION_MODE=bullets         # bullets or headers
MEMORY_SOURCES=convo,summary,graph,semantic  # Retrieval sources
```

**Vision Processing:**
```bash
VISION_KEYWORD_FILTER=true            # Only inject for vision queries
VISION_KEYWORDS=see,look,show,what,describe...
VISION_MAX_IMAGES_IN_CONTEXT=2        # Limit images to save tokens
VISION_ENABLE_DEDUPLICATION=true      # Prevent duplicate frames
```

**Audio Intelligence:**
```bash
AUDIO_INTEL_INTRO_PIPELINE=true       # Guided speaker enrollment
AUDIO_INTEL_SKIP_FOR_RETURNING=false  # Re-enroll or skip for known speakers
SPEAKER_AUTO_ENROLL_UTTERANCES=3      # Utterances needed for enrollment
```

# Pre-commit hooks

This repo includes pre-commit hooks for Python and the Next.js client.

- Install hooks: `pip install pre-commit && pre-commit install`
- Run on all files: `pre-commit run --all-files`

What runs:
- General hygiene: whitespace, EOF, YAML/TOML/JSON validity, conflict markers, secret keys, large files
- Python (server/, scripts/): Black formatter + Flake8 linter
- Frontend (client/): Prettier formatting + ESLint via `next lint`

Notes:
- The ESLint hook requires dependencies installed in `client/` (`npm i`).
- Very large artifacts (e.g., `docs/locomo10.json`, `docs/mem0_github_repo.txt`, `server/uv.lock`) are excluded from Prettier.
