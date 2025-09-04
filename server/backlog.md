# LocalCat Server Development Backlog

## 🚧 DRAFT: Context Management Improvements (2025-09-04)

### Problem Analysis
- **Critical Issue**: Context window bloat causing degraded performance
- **Token Growth**: From 1624 → 2705+ tokens in just a few turns
- **Memory Duplication**: Same memories repeated 10+ times (e.g., "Hello! It's great to see you again." appears 10x)
- **No Deduplication**: Every conversation stored without filtering
- **Risk**: Can exceed model's context window, causing failures

### Root Causes
1. **Pipecat's Mem0MemoryService**: 
   - Stores every conversation turn without deduplication
   - Returns top 10 memories by default (even if duplicates)
   - No semantic similarity filtering

2. **Memory Retrieval Pattern**:
   - Each user message triggers memory retrieval
   - Retrieved memories added as system messages
   - No sliding window or context pruning

3. **Context Accumulation**:
   - Full conversation history kept in context
   - Memories prepended to every LLM call
   - No token counting or limits

### Research-Based Solution: Hybrid Approach

#### Option 1: Minimal Changes (SELECTED - Quick Win) ✅
Based on research of DSPy, MemGPT/Letta, and LangChain best practices:

**Phase 1A: Memory Deduplication (`custom_mem0_service.py`)**
```python
from collections import deque
import hashlib

class CustomMem0MemoryService:
    def __init__(self, ...):
        # Add sliding window for recent turns
        self.recent_turns = deque(maxlen=10)  # Keep last 10 turns
        
    def _deduplicate_memories(self, memories):
        """Simple hash-based deduplication - fixes 90% of duplicate issue"""
        seen = set()
        unique = []
        
        for mem in memories.get('results', []):
            text = mem.get('memory', '').strip()
            
            # Skip empty or very short memories
            if len(text) < 10:
                continue
                
            # Simple hash dedup (fast)
            text_hash = hashlib.md5(text.lower().encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(mem)
                
        return unique[:5]  # Max 5 memories
    
    def _enhance_context_with_memories(self, context_messages, memories):
        """Enhanced with deduplication and sliding window"""
        # Deduplicate memories first
        unique_memories = self._deduplicate_memories(memories)
        
        # Build memory text from unique memories only
        if unique_memories:
            memory_text = self.system_prompt
            for i, memory in enumerate(unique_memories, 1):
                memory_text += f"{i}. {memory.get('memory', '')}\n\n"
            
            # Add as system message
            context_messages.insert(self.position, {
                "role": "system", 
                "content": memory_text
            })
        
        # Manage conversation window (keep recent turns)
        self._manage_conversation_window(context_messages)
        
        return context_messages
        
    def _manage_conversation_window(self, context_messages):
        """Keep only recent conversation turns to prevent bloat"""
        # Separate system messages from conversation
        system_msgs = [msg for msg in context_messages if msg.get('role') == 'system']
        convo_msgs = [msg for msg in context_messages if msg.get('role') != 'system']
        
        # Keep only recent conversation turns (last 10 user/assistant pairs = 20 messages)
        if len(convo_msgs) > 20:
            convo_msgs = convo_msgs[-20:]
            
        # Rebuild context: system messages + recent conversation
        context_messages.clear()
        context_messages.extend(system_msgs)
        context_messages.extend(convo_msgs)
```

#### Future Enhancement: DSPy Integration (Option 3)
Since Qwen3 works well with DSPy, plan for phase 2:
- Replace memory extraction prompts with DSPy modules
- Auto-optimize memory relevance scoring
- Use DSPy.BootstrapRS for few-shot memory examples
- Keep current deduplication logic but enhance memory quality

**Why This Approach:**
1. **Immediate impact** - Fixes duplicate memory issue now
2. **LangChain-inspired** - Uses proven sliding window pattern  
3. **MemGPT-influenced** - Tiered memory management concept
4. **DSPy-ready** - Easy to enhance memory extraction later

### Implementation Priority
1. **High Priority** (Immediate):
   - Memory deduplication in `_enhance_context_with_memories()`
   - Limit memories to top 5
   - Add duplicate detection

2. **Medium Priority** (This week):
   - Implement sliding window for conversation turns
   - Add token counting and monitoring
   - Create context manager class

3. **Low Priority** (Future):
   - Memory importance scoring
   - Semantic memory consolidation
   - Long-term memory archival

### Success Metrics
- [ ] Context stays under 50% of model limit (4k tokens)
- [ ] No duplicate memories in context
- [ ] Consistent response times even after long conversations
- [ ] Memory retrieval returns only unique, relevant entries
- [ ] Token usage logged and monitored

### Testing Plan
1. Simulate long conversation (20+ turns)
2. Verify context doesn't exceed limits
3. Check memory deduplication working
4. Measure response latency over time
5. Validate memory quality filtering

### Files to Modify/Create
- `server/custom_mem0_service.py` - Add deduplication, filtering
- `server/context_manager.py` (NEW) - Sliding window manager
- `server/bot.py` - Integrate context manager
- `.env` - Add context size limits configuration

### Notes
- Current system works but inefficient at scale
- Priority is preventing context overflow failures
- Keep changes backward compatible with existing Pipecat API
- Consider caching embeddings for duplicate detection

---

## ✅ Completed: Simplified Memory Service (Mem0ServiceV2) - Works but Slow (2025-09-05)

### Final Solution: Let Mem0 Work As Designed
After extensive research into mem0's GitHub documentation, discovered the correct approach:

**Key Insight**: Mem0 is designed to be simple - just call `memory.add(messages, user_id="peppi")` and it handles everything internally:
- Fact extraction
- Memory deduplication  
- ADD/UPDATE/DELETE operations using semantic similarity
- No dual models or schema enforcement needed

### Implementation: Mem0ServiceV2
Created `mem0_service_v2.py` based on GitHub examples:
```python
class Mem0ServiceV2(BaseMem0MemoryService):
    def _store_messages(self, messages):
        # Only filter meaningful messages, let mem0 handle the rest
        meaningful_messages = messages[-3:]  # Keep recent context
        super()._store_messages(meaningful_messages)  # Let mem0 do everything
```

### Current Status: ✅ Working but Issues Remain
- ✅ **Memory storage**: Works correctly, no more JSON errors
- ✅ **Memory retrieval**: Finds stored information properly
- ✅ **No context object errors**: Fixed 'list' has no 'add_message' issues
- ⚠️ **Performance**: Super slow due to LM Studio context accumulation
- ⚠️ **Context refresh**: LM Studio doesn't reset context per call as needed

### Outstanding Performance Issues
1. **LM Studio Context Accumulation**: 
   - Each mem0 call adds to LM Studio's context
   - Eventually hits context limit and becomes very slow
   - Need context reset mechanism between calls

2. **No Session Management**: 
   - LM Studio maintains conversation state across mem0 operations
   - Memory operations should be isolated/stateless
   - Need session_id rotation or context clearing

### Files Created/Modified:
- ✅ `server/mem0_service_v2.py` (Renamed from simple_mem0_service_v2.py)
- ✅ `server/bot.py` (Updated import to use Mem0ServiceV2)
- ✅ Removed complex dual-model approaches that fought against mem0's design

### Next Steps: Fix Performance
1. **Implement context reset for LM Studio** between mem0 operations
2. **Add session_id rotation** to prevent context accumulation  
3. **Consider batching** memory operations vs per-turn processing

---

## ✅ Completed: Dual-Model Memory Service with LM Studio (2025-09-04) [DEPRECATED]

### Problem Evolution
1. **Initial**: mem0 + Osaurus compatibility errors:
   - JSON responses truncated at 81-93 characters
   - Model returning conversational text instead of JSON
   - Token limit issues with small models

2. **Discovery**: Osaurus has hardcoded max_new_tokens=100 default
   - Causes systematic truncation regardless of max_tokens setting
   - Llama 3.2 1B not instruction-following properly

### Solution: Dual-Model Architecture with LM Studio
- Created dual_model_mem0_service.py
- Three-model architecture:
  - **Conversation**: Gemma3 4B (Ollama) 
  - **Fact Extraction**: Qwen3 4B (LM Studio)
  - **Memory Updates**: Qwen3 4B Instruct (LM Studio)
- JSON schema enforcement via LM Studio
- Automatic model selection based on task type
- No more JSON truncation or parsing issues

### Files Created/Modified:
- server/dual_model_mem0_service.py (NEW)
- server/memory_schemas_and_prompts.json (NEW)
- server/test_with_lm_studio.py (NEW)
- server/flexible_memory_extractor.py (NEW)
- server/bot.py (updated import)
- .env (MEM0_FACT_MODEL, MEM0_UPDATE_MODEL variables)

### Result: Reliable memory persistence with proper JSON output

---

## ✅ Completed: Technical Debt Cleanup (2025-09-04)

### Target Issues - ALL RESOLVED! 
**Status**: 🎉 **COMPLETE** - All actionable technical debt eliminated

#### ✅ Immediate Fixes Completed (< 1 hour total)
1. **Pipecat Transport Import Deprecations** ✅ FIXED
   - Updated `bot.py` imports from deprecated modules
   - `pipecat.transports.network.small_webrtc` → `pipecat.transports.smallwebrtc.transport`
   - **Result**: No more Pipecat deprecation warnings

2. **Dependency Version Fixes** ✅ FIXED
   - Downgraded scikit-learn to compatible version (1.5.1) 
   - Downgraded PyTorch to tested version (2.5.0)
   - **Result**: No more version compatibility warnings

3. **Requirements.txt Cleanup** ✅ FIXED
   - Pinned all dependency versions to prevent breaking changes
   - Removed unused `vllm` dependency (macOS incompatible)
   - **Result**: Stable, reproducible builds

#### ℹ️ WebSockets Issue - External Dependency
4. **WebSockets Legacy API** ⚠️ NOT FIXABLE
   - Issue is in uvicorn's internal code, not ours
   - Warning is harmless and will be fixed in future uvicorn updates
   - **Status**: External dependency, not actionable

### ✅ Outcomes Achieved
- ✅ **Clean startup**: Eliminated all our deprecation warnings
- ✅ **Stable dependency versions**: All versions pinned and compatible
- ✅ **Future-proof imports**: Using current Pipecat APIs
- ✅ **Reduced maintenance burden**: No more version conflicts

### Files Modified
- ✅ `bot.py`: Updated deprecated imports 
- ✅ `requirements.txt`: Pinned compatible versions
- ✅ `techdebt.md`: Updated with resolved issues

### Success Criteria - MET! 
```bash
python bot.py  # Now starts with only 2 harmless external warnings (vs. 4+ critical before)
```

**Next Priority**: Focus on core system improvements (memory inference, integration tests)

---

## ✅ Completed: TTS and Greeting Fixes (2025-09-04)

### Issues Fixed
1. **Emoji Removal from TTS Output** ✅ FIXED
   - **Problem**: TTS was attempting to speak emoji characters (😊, etc.), causing garbled audio
   - **Solution**: Added comprehensive `remove_emojis()` function in `tts_mlx_isolated.py`
   - **Coverage**: All major Unicode emoji ranges (flags, symbols, emoticons, etc.)
   - **Result**: Clean TTS output, emojis silently filtered out

2. **First Sentence Duplication** ✅ FIXED
   - **Problem**: Initial greeting was spoken twice at startup
   - **Root Cause**: Deprecated `get_context_frame()` triggered LLM response + TextFrame duplication
   - **Solution**: Removed deprecated context frame trigger, send greeting directly to TTS
   - **Result**: Single greeting spoken at startup, no LLM response until user speaks

### Files Modified
- ✅ `server/bot.py`: Fixed greeting duplication in `on_client_ready` handler
- ✅ `server/tts_mlx_isolated.py`: Added emoji filtering with comprehensive Unicode ranges
- ✅ Import cleanup: Added `re` module for regex pattern matching

### Technical Details
- **Emoji Pattern**: Covers 15+ Unicode ranges including flags, symbols, pictographs
- **Empty Text Handling**: Skips TTS entirely if text becomes empty after emoji removal
- **Logging**: Added debug logs for skipped emoji-only text segments
- **Performance**: Minimal overhead, regex compiled once at module level

### Success Criteria - MET!
```bash
# Before: "Hello! 😊 What can I do for you today? What can I do for you today?"
# After: "Hello! It's great to see you again."
```

**Next Priority**: Continue with core system improvements and user experience enhancements
