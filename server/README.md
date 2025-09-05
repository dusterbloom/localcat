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

### Critical FrameProcessor Implementation Rules

NEVER ignore these patterns when creating custom processors:

1. Mandatory Parent Method Calls:
```python
class CustomProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # REQUIRED - sets up _FrameProcessor__input_queue
        # Your initialization here
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)  # REQUIRED - handles initialization state
        # Your frame processing logic here
```

2. StartFrame Handling Pattern:
```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    await super().process_frame(frame, direction)  # ALWAYS call parent first
    
    if isinstance(frame, StartFrame):
        # Push StartFrame downstream IMMEDIATELY
        await self.push_frame(frame, direction)
        # Then do processor-specific initialization
        self._your_initialization_logic()
        return
    
    # Handle other frames
    if isinstance(frame, YourFrameType):
        # Process your frames
        pass
    
    # ALWAYS forward frames to prevent pipeline blocking
    await self.push_frame(frame, direction)
```

3. Frame Forwarding Rule:
- MUST forward ALL frames with `await self.push_frame(frame, direction)`
- Failing to forward frames WILL block the entire pipeline
- This is the #1 cause of "start frameprocessor push frame" errors

### Common Initialization Errors

**Error: `RTVIProcessor#0 Trying to process SpeechControlParamsFrame#0 but StartFrame not received yet`**
- **Cause**: Processor receives frames before StartFrame due to pipeline timing
- **Solution**: Implement proper StartFrame checking in process_frame
- **Prevention**: Use the exact patterns above

**Error: `AttributeError: '_FrameProcessor__input_queue'`**
- **Cause**: Missing `super().__init__()` call in custom processor
- **Solution**: Always call parent init with proper kwargs
- **Prevention**: Follow mandatory parent method calls pattern

**Error: Pipeline hangs or "start frameprocessor push frame" issues**
- **Cause**: Processor not forwarding frames, blocking pipeline flow
- **Solution**: Ensure every process_frame method calls `await self.push_frame(frame, direction)`
- **Prevention**: Follow frame forwarding rule religiously

### StartFrame Initialization Lifecycle

1. **Pipeline Creation**: Pipeline creates all processors
2. **StartFrame Propagation**: StartFrame flows through pipeline in order
3. **Processor Initialization**: Each processor receives StartFrame and initializes
4. **Frame Processing Begins**: Normal frame processing starts
5. **Frame Flow**: All frames must be forwarded to maintain pipeline flow

### RTVIProcessor Specific Issues

RTVIProcessor requires proper initialization state before processing frames. If using RTVIProcessor in your pipeline:

- Ensure StartFrame reaches RTVIProcessor before any other frames
- RTVIProcessor is automatically added when metrics are enabled in transport
- Use proper initialization checking in custom processors that interact with RTVIProcessor

### Debugging Frame Issues

1. **Enable Pipeline Logging**: Set debug level to see frame flow
2. **Check Frame Forwarding**: Verify every processor calls `push_frame`
3. **Verify Parent Calls**: Ensure `super().__init__()` and `super().process_frame()` are called
4. **Monitor StartFrame Propagation**: Trace StartFrame through pipeline
5. **Use Pipeline Observers**: Implement observers to monitor frame lifecycle

### When Adding New Processors

**CHECKLIST - Use this for EVERY new processor:**
- [ ] Inherits from FrameProcessor
- [ ] Calls `super().__init__(**kwargs)` in __init__
- [ ] Calls `await super().process_frame(frame, direction)` in process_frame
- [ ] Handles StartFrame by pushing it downstream immediately
- [ ] Forwards ALL frames with `await self.push_frame(frame, direction)`
- [ ] Does not block frame flow under any circumstances
- [ ] Tested in isolation and in full pipeline
- [ ] Handles frame processing errors gracefully

## License

MIT
