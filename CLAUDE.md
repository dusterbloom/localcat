# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local voice agent application built with Pipecat framework, designed to run entirely on macOS with local models. The project demonstrates voice-to-voice AI with minimal latency (<800ms) using Apple Silicon hardware, featuring an advanced memory system and audio intelligence.

## Architecture

### Core Components
- **Server (`server/`)**: FastAPI-based Python server containing the main voice agent logic
  - `bot.py`: Main entry point with FastAPI + Pipecat pipeline
  - `config/`: Centralized configuration system (settings.py, base_config.py)
  - `core/`: Voice agent components organized by domain:
    - `core/tts/`: Text-to-speech (kokoro_mlx.py, tts_mlx_ultra_low_latency.py)
    - `core/stt/`: Speech-to-text (parakeet_streaming.py, whisper_mlx.py)
    - `core/memory/`: HotMem memory system (hotmem_service.py, memory_orchestrator.py, nlp_manager.py)
    - `core/audio/`: Audio intelligence (speaker recognition, prosody analysis)
    - `core/video/`: Vision processing (frame_throttler.py, vision_context_injector.py)
    - `core/factories/`: Service factory pattern (service_factory.py)

- **Client (`client/`)**: Next.js React application using Pipecat voice UI components
  - Uses `@pipecat-ai/voice-ui-kit` for WebRTC-based voice interface
  - Connects to local server via serverless WebRTC transport

### Model Pipeline
The voice pipeline uses these models in sequence:
1. Silero VAD (voice activity detection)
2. Smart-turn v2 (conversation turn management)
3. Parakeet STT (speech-to-text, Apple Silicon optimized)
4. Local LLM via OpenAI-compatible server (Ollama/LM Studio)
5. HotMem memory injection (context retrieval)
6. Kokoro TTS (text-to-speech, MLX optimized)

### Key Dependencies
- **Pipecat AI**: Core framework for voice agent pipelines
- **MLX**: Apple Silicon optimized ML inference (mlx-lm, mlx-audio)
- **FastAPI/Uvicorn**: Server framework
- **WebRTC**: Low-latency audio transport
- **LMDB**: Fast memory storage backend
- **spaCy**: NLP for coreference resolution and entity extraction
- **Next.js**: React-based client framework

## Development Commands

### Server Development
```bash
cd server/

# Using uv (preferred)
uv run bot.py

# Using pip
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py

# For faster startup after models are cached
HF_HUB_OFFLINE=1 uv run bot.py
```

### Client Development
```bash
cd client/

npm i
npm run dev        # Start development server
npm run build      # Build for production
npm run start      # Start production server
npm run lint       # Run ESLint
```

### Model Preparation
Before first run, cache TTS models to avoid delays:
```bash
# Cache Kokoro TTS model
mlx-audio.generate --model "mlx-community/Kokoro-82M-bf16" --text "Hello, I'm Pipecat!" --output "output.wav"
```

### Running Tests
```bash
cd server/

# Run all tests
python -m pytest

# Run specific test categories
python -m pytest -m "ci"           # Fast CI tests
python -m pytest -m "integration"  # Integration tests
```

## Prerequisites

### Required External Services
- **Local LLM Server**: Run an OpenAI-compatible server (Ollama recommended, LM Studio also works)
  - Ollama: `ollama pull gemma3n:4b` or any compatible model
  - LM Studio: Configure in "Developer" tab

### System Requirements
- macOS with Apple Silicon (M-series chips)
- Python 3.12+
- Node.js for client development

## Important Notes

- **Configuration**: All settings are in `server/config/`. Copy `.env.example` to `.env` and customize.
- **Memory System**: HotMem provides persistent memory with coreference resolution. Configure via `MEMORY_*` env vars.
- **First Startup**: Initial startup can take 30+ seconds while downloading/caching models
- **WebRTC Transport**: Uses serverless WebRTC for low-latency audio - no external WebRTC server needed
- **Model Caching**: Set `HF_HUB_OFFLINE=1` after first run to prevent network model checks and improve startup time