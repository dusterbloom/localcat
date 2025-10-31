# LocalCat Memory System Guide

## Overview

LocalCat uses a unified memory system called **HotPath** that provides:

- **🚀 Automatic context injection** with emoji formatting for visual clarity
- **🛠️ Explicit tool access** (remember, recall, forget, search)
- **⚡ DirectMLX performance** (5-6x faster than HTTP-based solutions)
- **🎯 Superior retrieval accuracy** and relevance scoring
- **💾 Local storage** with no external API dependencies

## Quick Start

### Basic Setup

Set your memory backend to HotPath (recommended):

```bash
# In your .env file
MEMORY_BACKEND=hotpath
VOICE_AGENT_MEMORY_ENABLED=true
```

### Factory Integration

HotPath integrates automatically through the VoiceAgentFactory:

```python
from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory

# Create factory - HotPath is used by default
config = VoiceAgentConfig()
factory = VoiceAgentFactory(config)

# Memory processor with HotPath backend
session_tracker = factory.create_session_tracker()
memory_processor = factory.create_memory_processor(None, session_tracker)

# HotPath tools are automatically registered with compatible LLMs
voice_agent = factory.create_voice_agent(
    room_url="your-room-url",
    transport=your_transport,
    stt=your_stt,
    llm=your_llm,  # DirectMLX with tools recommended
    tts=your_tts
)
```

## Key Features

### 1. Automatic Context Injection

HotPath automatically injects relevant memories into the conversation context:

```
🧠 Memory Context:
👤 User mentioned: "I work as a software engineer"
🏠 User lives in: "San Francisco"
💼 User prefers: "Python and TypeScript"
🎯 User goal: "Building a voice assistant"
```

### 2. Memory Tools Available

The LLM can use these tools explicitly:

- `hotmem_remember`: Store specific information
- `hotmem_recall`: Retrieve specific information
- `hotmem_forget`: Remove information
- `hotmem_search`: Search with different strategies (conversation, graph, context, related, entity, temporal, semantic)

### 3. High-Performance Storage

HotPath uses a dual storage system:

- **SQLite**: Structured data with FTS5 full-text search
- **LMDB**: Graph data for relationship traversal
- **Processing**: 27-pattern extraction system
- **Performance**: <10ms total for storage + retrieval

## Configuration Options

### Environment Variables

```bash
# Core memory settings
MEMORY_BACKEND=hotpath                    # Use HotPath (recommended)
VOICE_AGENT_MEMORY_ENABLED=true          # Enable memory system
VOICE_AGENT_HOTPATH_ENABLED=true         # Enable HotPath processor
VOICE_AGENT_SESSION_PERSISTENCE=true     # Enable session persistence

# Storage paths (optional - uses defaults if not set)
MEMORY_SQLITE_PATH=./data/memory.db      # SQLite database path
MEMORY_LMDB_PATH=./data/graph.lmdb       # LMDB directory path

# Behavior settings
MEMORY_BULLETS_MAX=3                     # Max memory bullets to inject
MEMORY_INTERIM_MIN_WORDS=6               # Min words for interim processing
MEMORY_INJECT_ROLE=user                  # Role for memory messages
MEMORY_INJECT_HEADER=[Memory context]    # Header for memory messages
MEMORY_SOURCES=graph,convo,summary       # Memory sources to use

# Context management
CONTEXT_SLIDING_WINDOW=true              # Enable sliding window
CONTEXT_MAX_TURN_PAIRS=4                 # Max turn pairs in context
LLM_CONTEXT_MAX_TOKENS=3000              # Max tokens for LLM context
LLM_CONTEXT_PRUNE_THRESHOLD=0.70         # Prune at 70% capacity
LLM_CONTEXT_MIN_TURNS=3                  # Min turns to keep
```

### LLM Configuration for Tools

For the best experience, use DirectMLX with tool calling support:

```bash
# Model selection (tool-capable models recommended)
LLM_MODEL=mlx-community/Qwen3-1.7B-8bit  # Tool-capable model
LLM_USE_DIRECT_MLX=true                  # Enable DirectMLX
LLM_MAX_TOKENS=256                       # Response tokens
LLM_TEMPERATURE=0.7                      # Creativity level
```

## Memory Tool Usage Examples

### Automatic Memory

Users don't need to do anything - memories are stored automatically:

```
User: "My name is Sarah and I'm a graphic designer from New York."
Assistant: "Nice to meet you, Sarah! How can I help you today?"
# [HotPath automatically stores this information]
```

### Explicit Memory Requests

Users can explicitly ask the agent to use memory tools:

```
User: "Remember that I prefer short, direct responses."
Assistant: "I'll remember that preference for future conversations."
# [Agent uses hotmem_remember tool]

User: "What do you know about my work?"
Assistant: "Let me check what I remember about your work."
# [Agent uses hotmem_recall tool]
```

### Memory Search

```
User: "Search for information about my project deadlines."
Assistant: "I'll search through my memory for project deadline information."
# [Agent uses hotmem_search with different search types]
```

## Performance Comparison

| Feature | HotPath | HTTP-based Solutions |
|---------|---------|---------------------|
| Storage | ~3-5ms | ~50-200ms |
| Retrieval | ~2-4ms | ~100-300ms |
| Tool Calling | Native DirectMLX | HTTP overhead |
| Context Injection | Automatic with emojis | Manual or basic |
| Dependencies | Local only | External APIs |
| Performance | **5-6x faster** | Baseline |

## Advanced Usage

### Custom Storage Paths

```python
from core.memory.database_path import setup_database_paths

# Setup custom paths
setup_database_paths(
    sqlite_path="/custom/location/memory.db",
    lmdb_dir="/custom/location/graph.lmdb"
)
```

### Session Tracking

```python
# Session tracking is automatic with HotPath
# Sessions persist across restarts if VOICE_AGENT_SESSION_PERSISTENCE=true
```

### Memory Statistics

```python
# Access memory processor for advanced usage
memory_processor = factory.create_memory_processor(None, session_tracker)

# Get performance metrics
if hasattr(memory_processor, 'hot'):
    stats = memory_processor.hot.get_performance_stats()
    print(f"Storage time: {stats.get('avg_storage_time_ms', 0):.2f}ms")
    print(f"Retrieval time: {stats.get('avg_retrieval_time_ms', 0):.2f}ms")
```

## Migration from Other Memory Systems

### From HotMem (Legacy)

HotMem is now deprecated. Simply change your configuration:

```bash
# Before (deprecated)
MEMORY_BACKEND=hotmem

# After (recommended)
MEMORY_BACKEND=hotpath
```

All your existing memories will be accessible through HotPath with improved performance and features.

### From External Memory Services

```bash
# Before (external service)
MEMORY_BACKEND=external
API_KEY=your-api-key

# After (local HotPath)
MEMORY_BACKEND=hotpath
# No API key needed - uses local storage
```

## Testing and Validation

### Basic Functionality Test

```bash
# Test HotPath tools integration
python test_hotpath_tools_simple.py

# Test comprehensive HotPath functionality
python test_hotpath_tools_integration.py
```

### Performance Testing

```bash
# Test memory backend performance
python test_memory_backends_comparison.py
```

## Troubleshooting

### Common Issues

1. **Tools not available**: Ensure you're using a tool-capable model with `LLM_USE_DIRECT_MLX=true`
2. **Slow performance**: Check that DirectMLX is being used (not HTTP-based LLM)
3. **No context injection**: Verify `VOICE_AGENT_MEMORY_ENABLED=true` and `MEMORY_BACKEND=hotpath`

### Debug Logging

```python
import os
os.environ["HOTMEM_LOG_LEVEL"] = "DEBUG"  # Shows HotPath operations
```

### Storage Issues

Check file permissions for your data directories:
- SQLite database: `./data/memory.db` (or custom path)
- LMDB graph: `./data/graph.lmdb/` (or custom path)

## Architecture Notes

HotPath combines:

1. **Automatic Processing**: Conversation messages are processed automatically
2. **Tool Integration**: Memory tools available for explicit LLM control
3. **High-Performance Storage**: SQLite + LMDB dual storage system
4. **Smart Retrieval**: Context-aware memory injection with emojis
5. **DirectMLX Support**: Native tool calling without HTTP overhead

This unified approach provides:
- ✅ **Zero configuration** - Works out of the box
- ✅ **Maximum performance** - DirectMLX + local storage
- ✅ **Rich context** - Emoji-formatted memory injection
- ✅ **Explicit control** - Tool-based memory operations
- ✅ **No external dependencies** - Completely local

## Summary

HotPath is the recommended memory solution for LocalCat that provides:

🚀 **Best performance** - 5-6x faster than HTTP-based solutions
🧠 **Automatic context** - Smart memory injection with visual formatting
🛠️ **Tool support** - Explicit memory operations when needed
💾 **Local storage** - No external API dependencies
📈 **Superior retrieval** - Advanced search and relevance scoring

Simply set `MEMORY_BACKEND=hotpath` and enjoy the unified memory experience!