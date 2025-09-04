# LocalCat Server

A local voice assistant with persistent memory powered by local LLMs.

## Features

- 🎤 **Real-time Voice Chat** - WebRTC-based audio communication
- 🧠 **Persistent Memory** - Remembers conversations using mem0 + FAISS
- 🏠 **Fully Local** - Works with Ollama/LM Studio, no cloud required
- ⚡ **Fast Response** - Optimized for local inference
- 🔄 **Smart Turn Detection** - Natural conversation flow

## Architecture

- **Main Model**: Gemma3:4b (conversation) via Ollama
- **Memory Model**: Qwen3-1.7B-4bit (fact extraction) via Osaurus
- **STT**: Whisper MLX (local speech recognition)
- **TTS**: Kokoro MLX (local text-to-speech)
- **Memory**: mem0 + FAISS vector storage
- **Hybrid Setup**: Ollama + Osaurus for optimal Apple Silicon performance

## Quick Start

### Prerequisites
- Python 3.12+
- Ollama installed and running
- Osaurus app installed (download from [GitHub releases](https://github.com/dinoki-ai/osaurus/releases))

### Installation

1. Clone and setup:
```bash
cd server
python -m venv .venv
source .venv/bin/activate  # or .venv\Scriptsctivate on Windows
pip install -r requirements.txt
```

2. Install models:
```bash
# Main conversation model (Ollama)
ollama pull gemma3:4b
ollama pull nomic-embed-text:latest
```

3. Setup Osaurus:
```bash
# Open Osaurus app
open /Applications/osaurus.app

# In Osaurus:
# 1. Set port to 8000
# 2. Download model: mlx-community/Qwen3-1.7B-4bit
# 3. Start the server
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Run:
```bash
python bot.py
```

## Configuration

Key environment variables in `.env`:

```env
# Main conversation model (Ollama)
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_MODEL=gemma3:4b

# Memory extraction model (Osaurus)
MEM0_BASE_URL=http://127.0.0.1:8000/v1
MEM0_MODEL=mlx-community/Qwen3-1.7B-4bit

# Embeddings (Ollama)
EMBEDDING_MODEL=nomic-embed-text:latest

# Memory storage
MEMORY_FAISS_PATH=/path/to/memory/data
USER_ID=your_user_id
AGENT_ID=locat
```

## Memory System

The assistant uses a dual-model approach:
- **Conversation Model**: Handles real-time chat responses
- **Memory Model**: Extracts and processes facts for long-term storage

Memory is automatically:
- Extracted from conversations
- Stored in FAISS vector database
- Retrieved for contextual responses

## Troubleshooting

### Common Issues

**mem0 JSON Errors**: 
- Solution implemented: Custom mem0 service with fallback to `infer=False`
- Alternative: Use Ollama for both models instead of LM Studio

**Model Not Found**:
- Ensure models are pulled: `ollama list`
- Check model names match .env configuration

**Port Conflicts**:
- Default: Ollama (11434), Osaurus (8000), Bot (7860)
- Check for SurrealDB blocking port 8000: `lsof -ti:8000`
- Kill conflicting processes: `kill $(lsof -ti:8000)`

## Development

See `backlog.md` for development notes and `changelog.md` for version history.

## License

MIT
