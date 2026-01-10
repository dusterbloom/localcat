# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocalCat is a production-ready local voice assistant built with Pipecat framework, designed for macOS Apple Silicon. Key features:
- **MLX-LM as primary LLM** (DirectMLXLLMService, ~544ms TTFT)
- **Voice enrollment** (ECAPA-TDNN speaker recognition, 3-utterance auto-enroll)
- **Tauri desktop app** (code signed, DMG bundled, offline model support)
- **Advanced memory** (HotMem/HotPath graph-based memory with RAGAS evaluation)

## Architecture

```
LocalCat (Tauri Desktop App)
├── Frontend: Next.js + @pipecat-ai/voice-ui-kit
├── Tauri Backend (Rust):
│   ├── Daemon Manager (Python subprocess lifecycle)
│   └── Sidecar (macOS native STT - optional)
├── FastAPI Server (Python):
│   ├── Voice Pipeline: Silero VAD → MLX Whisper → Memory → LLM → TTS
│   ├── Memory: HotMem/HotPath + SQLite + LMDB + FTS
│   ├── Audio Intelligence: Speaker recognition + enrollment
│   └── ServiceFactory (builder pattern for services)
└── Bundled Models: MLX Whisper, Kokoro TTS, optional Parakeet
```

### Core Components

- **Server (`server/`)**: FastAPI-based Python server
  - `bot.py`: Main entry point with Pipecat pipeline configuration
  - `core/llm/`: LLM services (DirectMLXLLMService primary, OpenAI-compat fallback)
  - `core/memory/`: HotMem/HotPath memory system
  - `core/audio/`: Audio intelligence, speaker enrollment
  - `tts_mlx_isolated.py`: Process-isolated TTS (avoids Metal threading conflicts)

- **Client (`client/`)**: Next.js React application
  - Uses `@pipecat-ai/voice-ui-kit` for WebRTC-based voice interface
  - Connects to local server via serverless WebRTC transport

- **App (`app/`)**: Tauri desktop application
  - Rust backend manages Python daemon lifecycle
  - Code signing and notarization for macOS distribution
  - Build profiles: light (3.2GB), full (5.8GB)

### Model Pipeline

1. Silero VAD (voice activity detection)
2. Smart-turn v2 (conversation turn management)
3. MLX Whisper (speech-to-text)
4. HotMem/HotPath (memory retrieval + context injection)
5. DirectMLXLLMService (primary) or OpenAI-compat (fallback)
6. Kokoro TTS MLX (text-to-speech)

### Voice Enrollment Flow

1. User chooses: "sign me up" / "sign in" / "anonymous"
2. For enrollment: 3 voice samples → ECAPA-TDNN embedding → auto-enrolled
3. Name capture with validation
4. Returning users recognized automatically via speaker embeddings

## Development Commands

### Server Development
```bash
cd server/

# Using uv (preferred) - MLX-LM mode
LLM_USE_DIRECT_MLX=true uv run bot.py

# OpenAI-compatible fallback (requires LM Studio or similar)
uv run bot.py

# For faster startup after models are cached
HF_HUB_OFFLINE=1 LLM_USE_DIRECT_MLX=true uv run bot.py
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

### Tauri App Development
```bash
cd app/

cargo tauri dev                    # Development mode
BUILD_PROFILE=light ./build-production.sh   # Light build (3.2GB)
BUILD_PROFILE=full ./build-production.sh    # Full build (5.8GB)
```

### Testing
```bash
cd server/

# Run CI-safe tests (Linux compatible)
pytest tests/ -m "ci" -v

# Run all tests (macOS only, requires models)
pytest tests/ -v --run-slow

# Run specific test category
python tests/run_all_tests.py --category unit
```

### Model Preparation
```bash
# Cache Kokoro TTS model
mlx-audio.generate --model "mlx-community/Kokoro-82M-bf16" --text "Hello!" --output "output.wav"

# Cache Marvis TTS model (alternative)
mlx-audio.generate --model "Marvis-AI/marvis-tts-250m-v0.1" --text "Hello!" --output "output.wav"
```

## Configuration

Primary environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_USE_DIRECT_MLX` | Use MLX-LM directly (recommended) | `false` |
| `VOICE_AGENT_TTS_ENGINE` | TTS engine: `kokoro_mlx`, `supertonic`, `siri` | `kokoro_mlx` |
| `ENABLE_INTRO_PIPELINE` | Enable voice enrollment flow | `true` |
| `ENABLE_MEMORY` | Enable HotMem memory system | `true` |
| `AUDIO_INTEL_ENABLE_EMOTION` | Enable emotion detection | `false` |

## System Requirements

- macOS with Apple Silicon (M-series chips)
- Python 3.12+
- Node.js 20+
- Rust (for Tauri development)

## Key Dependencies

- **Pipecat AI**: Core framework for voice agent pipelines
- **MLX**: Apple Silicon optimized ML inference (mlx-lm, mlx-audio)
- **SpeechBrain**: Speaker recognition (ECAPA-TDNN) and emotion detection
- **FastAPI/Uvicorn**: Server framework
- **Tauri**: Desktop app framework
- **Next.js**: React-based client framework

## Important Notes

- **Process Isolation**: TTS services use separate processes to avoid Metal framework threading conflicts
- **First Startup**: Initial startup can take 30+ seconds while downloading/caching models
- **WebRTC Transport**: Uses serverless WebRTC for low-latency audio - no external WebRTC server needed
- **Model Caching**: Set `HF_HUB_OFFLINE=1` after first run to prevent network model checks
- **Memory System**: HotMem uses SQLite + LMDB for persistence, with in-memory graph for fast retrieval
