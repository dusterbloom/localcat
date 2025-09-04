# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local voice agent application built with Pipecat framework, designed to run entirely on macOS with local models. The project demonstrates voice-to-voice AI with minimal latency (<800ms) using Apple Silicon hardware.

## Architecture

### Core Components
- **Server (`server/`)**: FastAPI-based Python server containing the main voice agent logic
  - `bot.py`: Main entry point with Pipecat pipeline configuration
  - `tts_mlx_isolated.py`: Process-isolated TTS service to avoid Metal threading conflicts
  - `*_worker.py`: Isolated worker processes for different TTS models (Kokoro, Marvis)
  
- **Client (`client/`)**: Next.js React application using Pipecat voice UI components
  - Uses `@pipecat-ai/voice-ui-kit` for WebRTC-based voice interface
  - Connects to local server via serverless WebRTC transport

### Model Pipeline
The voice pipeline uses these models in sequence:
1. Silero VAD (voice activity detection)
2. Smart-turn v2 (conversation turn management)
3. MLX Whisper (speech-to-text)
4. Gemma3n 4B (language model via local OpenAI-compatible server)
5. Kokoro TTS or Marvis TTS (text-to-speech)

### Key Dependencies
- **Pipecat AI**: Core framework for voice agent pipelines
- **MLX**: Apple Silicon optimized ML inference (mlx-lm, mlx-audio)
- **FastAPI/Uvicorn**: Server framework
- **WebRTC**: Low-latency audio transport
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
# Cache Marvis TTS model
mlx-audio.generate --model "Marvis-AI/marvis-tts-250m-v0.1" --text "Hello, I'm Pipecat!" --output "output.wav"

# Or cache Kokoro TTS model
mlx-audio.generate --model "mlx-community/Kokoro-82M-bf16" --text "Hello, I'm Pipecat!" --output "output.wav"
```

## Prerequisites

### Required External Services
- **Local LLM Server**: Run an OpenAI-compatible server (e.g., LM Studio) for the language model
  - Configure in LM Studio's "Developer" tab
  - Must serve Gemma3n 4B or compatible model

### System Requirements
- macOS with Apple Silicon (M-series chips)
- Python 3.12+
- Node.js for client development

## Important Notes

- **Process Isolation**: TTS services use separate processes to avoid Metal framework threading conflicts on Apple Silicon
- **First Startup**: Initial startup can take 30+ seconds while downloading/caching models
- **WebRTC Transport**: Uses serverless WebRTC for low-latency audio - no external WebRTC server needed
- **Model Caching**: Set `HF_HUB_OFFLINE=1` after first run to prevent network model checks and improve startup time