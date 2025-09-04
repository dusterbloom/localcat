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
- **Memory Models**: Dual-model approach via LM Studio
  - **Fact Extraction**: Qwen3-4B (extracts facts from conversations)
  - **Memory Updates**: Qwen3-4B-Instruct-2507 (handles ADD/UPDATE/DELETE operations)
- **STT**: Whisper MLX (local speech recognition)
- **TTS**: Kokoro MLX (local text-to-speech)
- **Memory**: mem0 + FAISS vector storage
- **Hybrid Setup**: Ollama + LM Studio for optimal performance and JSON compliance

## Quick Start

### Prerequisites
- Python 3.12+
- Ollama installed and running
- LM Studio installed (download from [lmstudio.ai](https://lmstudio.ai))

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

3. Setup LM Studio:
```bash
# Open LM Studio app
open /Applications/LM\ Studio.app

# In LM Studio:
# 1. Download models:
#    - qwen/qwen3-4b (for fact extraction)
#    - qwen3-4b-instruct-2507 (for memory updates)
# 2. Go to Server tab
# 3. Load model and start server on port 1234
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

# Memory extraction models (LM Studio - dual model approach)
MEM0_BASE_URL=http://127.0.0.1:1234/v1
MEM0_FACT_MODEL=qwen/qwen3-4b           # For fact extraction
MEM0_UPDATE_MODEL=qwen3-4b-instruct-2507  # For memory updates

# Embeddings (Ollama)
EMBEDDING_MODEL=nomic-embed-text:latest

# Memory storage
MEMORY_FAISS_PATH=/path/to/memory/data
USER_ID=your_user_id
AGENT_ID=locat
```

## Memory System

The assistant uses a three-model approach for optimal performance:
- **Conversation Model** (Gemma3 4B via Ollama): Handles real-time chat responses
- **Fact Extraction Model** (Qwen3 4B via LM Studio): Extracts facts from conversations with JSON schema compliance
- **Memory Update Model** (Qwen3 4B Instruct via LM Studio): Manages ADD/UPDATE/DELETE operations with structured output

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
