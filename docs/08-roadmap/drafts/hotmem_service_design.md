# HotMem Service Design Document

## Executive Summary

This document outlines the design for `HotMemService`, a Pipecat-compatible memory service that combines:
- **HotPath's ultra-fast performance** (<200ms)
- **Tool-based explicit interface** (no intent classification)
- **Drop-in Mem0 compatibility** (for existing Pipecat users)

This document presents the current state, requirements, and design questions **without assumptions**, allowing for informed implementation decisions.

---

## Part 1: Current State Analysis

### 1.1 Existing HotPath Implementation

Located in: `server/core/memory/hotpath_processor.py`

**Current Capabilities:**
- Ultra-fast memory storage and retrieval (<200ms target)
- Graph-based entity/relationship storage
- LMDB/SQLite dual persistence
- Extraction pipeline with 27 patterns
- Intent classification integration (currently causing 42.9% accuracy issues)

**Current Limitations:**
- Automatic processing (no user control)
- Tied to intent classification pipeline
- Not Pipecat-compatible interface

### 1.2 Pipecat Mem0 Interface Analysis

Located in: `.venv/lib/python3.12/site-packages/pipecat/services/memory.py`

**Key Interface Requirements:**
```python
class Mem0MemoryService(FrameProcessor):
    def __init__(self, api_key, user_id, agent_id, run_id, params, host):
        # Must call super().__init__()
        # Initialize memory client

    def _store_messages(self, messages):
        # Store conversation messages

    def _retrieve_memories(self, query):
        # Retrieve relevant memories

    def _enhance_context_with_memories(self, context, query):
        # Add memories to LLM context

    async def process_frame(self, frame, direction):
        # Process pipeline frames
```

**Current Mem0 Behavior:**
- **Automatic processing**: Stores/retrieves on every user message
- **Context enhancement**: Automatically adds memories to LLM context
- **No user control**: Implicit memory operations
- **API-dependent**: Requires Mem0 cloud or local server

### 1.3 Current Pipeline Architecture

```
User Speech → STT → Intent Classification → [IF memory needed] → HotPath Memory → LLM
                                          → [IF casual] → Skip Memory → LLM
```

**Issues with Current Architecture:**
- Intent classification accuracy: 42.9%
- Technical discussions misclassified as "general_chat"
- No user control over memory operations
- Classification overhead on every turn

---

## Part 2: Requirements

### 2.1 Must-Have Requirements

1. **Pipecat Compatibility**: Drop-in replacement for `Mem0MemoryService`
2. **HotPath Performance**: Leverage existing <200ms memory system
3. **Tool-Based Interface**: Explicit user control over memory operations
4. **No Intent Classification**: Remove guessing/classification overhead
5. **Backward Compatibility**: Existing Pipecat code should work unchanged

### 2.2 Performance Requirements

- **Memory operations**: <200ms (matching current HotPath)
- **Pipeline overhead**: Minimal for non-memory turns
- **Total latency**: Maintain <800ms voice-to-voice target

### 2.3 User Experience Requirements

- **Explicit control**: User decides when to use memory
- **Transparent operations**: Clear when memory tools are used
- **Predictable behavior**: No false positives/negatives
- **Graceful degradation**: Continue working if memory fails

---

## Part 3: Design Decisions Made ✅

### 3.1 Core Tool Set - DECIDED ✅

**Final Decision**: Clean 4-tool interface with extended search capabilities

```python
# Core HotMem Tools
tools = [
    "hotmem_remember",   # Store information explicitly
    "hotmem_recall",     # Retrieve specific information
    "hotmem_forget",     # Remove information explicitly
    "hotmem_search"      # Unified search with search_types parameter
]

# hotmem_search supports search_types:
search_types = [
    "conversation",  # Search dialog history
    "graph",        # Navigate entity/relationship graph
    "context",      # Get relevant context for topic
    "related"       # Find related information
]
```

**Rationale**:
- Simple enough for small models to handle
- `search_types` parameter provides flexibility
- If models struggle with search_types, fallback to simple search works
- Extensible design allows future enhancements

### 3.2 Interface Design - DECIDED ✅

**Final Decision**: Hybrid approach preserving current automatic storage

```python
class HotMemService(Mem0MemoryService):
    """
    Hybrid approach:
    - KEEP automatic storage of messages, sessions, metadata, summaries
    - MAKE retrieval/search tool-based and explicit
    - PRESERVE existing context building patterns
    - REUSE db_session_tracker.add_message() and context injection logic
    """

    def _store_messages(self, messages):
        # KEEP: Automatic storage (minimal latency impact)
        # Use existing db_session_tracker.add_message()

    def _enhance_context_with_memories(self, context, query):
        # CHANGE: Add tool definitions instead of automatic retrieval
        # Let LLM choose when to use memory tools

    async def process_frame(self, frame, direction):
        # HANDLE tool calls when LLM invokes memory tools
        # PRESERVE automatic message storage flow
```

**Rationale**:
- Leverages existing efficient storage patterns
- Eliminates intent classification by making retrieval explicit
- Maintains full Pipecat compatibility
- Reduces code duplication by reusing current `_inject_memory_context()` patterns

### 3.3 Memory Storage - DECIDED ✅

**Final Decision**: Use existing HotPath system as-is

**Implementation**:
- Direct integration with current HotPath storage
- Reuse existing graph-based entities/relationships
- Leverage LMDB/SQLite dual persistence
- Maintain 27-pattern extraction pipeline
- Continue using current `_inject_memory_context()` patterns

**Rationale**: Proven <200ms performance, no need to fix what's working

### 3.4 Search Capabilities - DECIDED ✅

**Final Decision**: Provide all search types, let users/agents decide

**Available Search Types**:
```python
SEARCH_TYPES = {
    "conversation": "Search dialog history and past conversations",
    "graph": "Navigate entity/relationship connections",
    "context": "Get relevant contextual information",
    "related": "Find semantically related information",
    "entity": "Search for specific entity information",
    "temporal": "Time-based memory queries",
    "semantic": "Conceptual similarity search"
}
```

**Rationale**: Comprehensive capabilities with user/agent choice, no forced limitations

### 3.5 Backward Compatibility - DECIDED ✅

**Final Decision**: Full Mem0 compatibility

**Requirements**:
- Support all `Mem0MemoryService` methods
- Existing Pipecat code works unchanged
- Same initialization interface
- Same frame processing pipeline
- Drop-in replacement capability

**Rationale**: Serves Pipecat community by providing upgrade path without breaking changes

---

## Part 4: Implementation Phases

### Phase 1: Basic Drop-in Replacement

**Goal**: Create minimal viable HotMemService

**Tasks:**
1. Extend `Mem0MemoryService` base class
2. Override storage methods to use HotPath
3. Remove automatic memory processing
4. Basic compatibility testing

**Success Criteria:**
- Existing Pipecat code doesn't break
- HotPath storage/retrieval working
- No automatic memory overhead

### Phase 2: Tool Interface Implementation

**Goal**: Add explicit memory tools

**Tasks:**
1. Define core tool set (based on decisions from Part 3.1)
2. Implement tool processing logic
3. Add LLM context enhancement for tools
4. Test tool activation/response

**Success Criteria:**
- Tools available to LLM
- Explicit memory operations working
- User control over memory usage

### Phase 3: Enhanced Capabilities

**Goal**: Add advanced memory features

**Tasks:**
1. Implement search capabilities (based on decisions from Part 3.4)
2. Add performance optimizations
3. Create comprehensive documentation
4. Package for Pipecat community

**Success Criteria:**
- Rich memory operations available
- Performance targets met
- Ready for open-source contribution

---

## Part 5: Implementation Specifications

### 5.1 Tool Interface Definitions

```python
class HotMemService(Mem0MemoryService):
    """Drop-in replacement for Pipecat's Mem0MemoryService with tool-based interface"""

    TOOL_DEFINITIONS = [
        {
            "name": "hotmem_remember",
            "description": "Store information in memory for future recall",
            "parameters": {
                "information": {"type": "string", "description": "Information to remember"}
            }
        },
        {
            "name": "hotmem_recall",
            "description": "Retrieve specific information from memory",
            "parameters": {
                "query": {"type": "string", "description": "What to recall"}
            }
        },
        {
            "name": "hotmem_forget",
            "description": "Remove information from memory",
            "parameters": {
                "query": {"type": "string", "description": "What to forget"}
            }
        },
        {
            "name": "hotmem_search",
            "description": "Search memory with optional search type",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "search_type": {
                    "type": "string",
                    "enum": ["conversation", "graph", "context", "related"],
                    "description": "Type of search to perform"
                }
            }
        }
    ]
```

### 5.2 Integration with Existing Systems

**Context Building - REUSE EXISTING**:
- Continue using `context.get_messages()` and `context.set_messages()`
- Reuse `_inject_memory_context()` pattern from hotpath_processor.py:463
- Maintain `db_session_tracker.add_message()` for automatic storage

**Storage Layer - DIRECT INTEGRATION**:
- Use existing HotPath storage without abstraction
- Leverage current LMDB/SQLite dual persistence
- Maintain 27-pattern extraction pipeline performance

**Pipeline Integration - HYBRID APPROACH**:
- Preserve automatic message storage (minimal latency)
- Add tool definitions to context for LLM usage
- Handle tool calls explicitly when invoked

### 5.3 Performance Specifications

**Target Latencies**:
- Tool calls: <200ms (matching current HotPath)
- Storage operations: <50ms (automatic, background)
- Search operations: <300ms (acceptable for explicit operations)

**Resource Usage**:
- Memory: Reuse existing HotPath memory footprint
- CPU: Background operations during tool calls only
- Storage: Leverage existing LMDB/SQLite efficiency

### 5.4 Error Handling Strategy

**Tool Call Failures**:
- Log error details for debugging
- Return graceful error message to user
- Continue conversation without breaking pipeline

**Storage Failures**:
- Continue automatic storage attempts
- Log failures without interrupting conversation
- Maintain conversation flow as primary priority

---

## Part 6: Next Steps

### Immediate Actions Needed

1. **Review this document** and provide decisions on open questions
2. **Define core tool set** based on actual agent needs
3. **Choose integration pattern** that fits your LLM setup
4. **Specify performance requirements** for different operations

### Implementation Sequence

1. **Decision gathering**: Resolve open questions in Parts 3 and 5
2. **Phase 1 implementation**: Basic drop-in replacement
3. **Testing and iteration**: Verify approach with real usage
4. **Phase 2/3 implementation**: Based on Phase 1 learnings

### Success Metrics

- **Performance**: Memory operations maintain <200ms target
- **Usability**: Clear improvement over intent classification approach
- **Compatibility**: Existing Pipecat users can adopt easily
- **Maintainability**: No ongoing model tuning required

---

## Conclusion

This document provides a complete specification for HotMemService implementation with all design decisions made. The system will:

1. **Drop-in replace** Pipecat's Mem0MemoryService with full compatibility
2. **Eliminate intent classification** issues by using explicit tool interface
3. **Leverage existing HotPath** performance and storage systems
4. **Provide comprehensive search** capabilities while maintaining simplicity
5. **Reduce technical debt** by reusing proven context building patterns

**Implementation is ready to begin** with clear requirements, performance targets, and integration specifications.