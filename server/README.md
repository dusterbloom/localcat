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
- **Memory Model**: Qwen2.5-7B-Instruct (fact extraction) via LM Studio/Ollama
- **STT**: Whisper MLX (local speech recognition)
- **TTS**: Kokoro MLX (local text-to-speech)
- **Memory**: mem0 + FAISS vector storage

## Quick Start

### Prerequisites
- Python 3.12+
- Ollama installed and running
- LM Studio (or use Ollama for both models)

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

# Memory extraction model (Ollama - recommended)
ollama pull qwen2.5:7b-instruct
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run:
```bash
python bot.py
```

## Configuration

Key environment variables in `.env`:

```env
# Main conversation model (Ollama)
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_MODEL=gemma3:4b

# Memory extraction model 
MEM0_BASE_URL=http://127.0.0.1:11434/v1  # Use Ollama for both
MEM0_MODEL=qwen2.5:7b-instruct

# Embeddings
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
- Default: Ollama (11434), LM Studio (1234), Bot (7860)
- Modify ports in .env if needed

## Development

See `backlog.md` for development notes and `changelog.md` for version history.

## License

MIT
