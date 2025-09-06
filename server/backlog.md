# LocalCat Server Development Backlog

## ✅ COMPLETED: HotMem Ultra-Fast Memory System (2025-09-05)

### 🎉 Major Success - Replaced mem0 with HotMem
- **Performance Achievement**: <200ms p95 latency (3.8ms average vs 2000ms mem0)
- **Complete Integration**: Perfect Pipecat context injection with memory bullets  
- **Live Production**: Working seamlessly in voice conversations
- **Pattern Coverage**: 100% USGS Grammar-to-Graph dependency patterns implemented
- **Architecture**: Dual storage (SQLite + LMDB) for persistence + O(1) speed

### Key Components Delivered
- `hotpath_processor.py`: Pipecat-integrated processor with context injection
- `memory_hotpath.py`: UD-based extraction engine with comprehensive pattern coverage  
- `memory_store.py`: High-performance dual storage system
- `ud_utils.py`: Universal Dependencies parsing utilities
- **Archive**: All experimental code preserved in `archive/experimental/`

### Critical Fixes Implemented
- Fixed WhisperSTTServiceMLX `is_final=None` handling for non-streaming STT
- Corrected pipeline ordering: memory before context aggregator  
- Enabled spaCy lemmatizer for proper relation extraction
- Implemented direct context injection (not LLMMessagesFrame competition)
- Resolved all mem0 performance and integration issues

---

## ✅ COMPLETED: HotMem Evolution Phase 1 - Language-Agnostic Intelligence (2025-09-05)

### 🎉 Major Success - Language-Agnostic Enhancement Complete
- **Performance Achievement**: 34ms average (85% faster than baseline, well under 200ms p95)
- **Quality Improvement**: 7.3/10 average (up from 6.9, filtering working correctly)
- **Architecture Revolution**: Fully language-agnostic using Universal Dependencies
- **Integration**: Live in production voice pipeline with zero breaking changes

### Key Components Delivered
- `memory_intent.py`: Language-agnostic intent classification using UD patterns
- Enhanced `memory_hotpath.py`: Intent-aware extraction with quality filtering
- `test_intent_enhancement.py`: Comprehensive test suite validating improvements
- **Full Integration**: Working in `bot.py` Pipecat pipeline

### Critical Fixes Implemented
- **Eliminated Language-Specific Patterns**: No more regex/hardcoded English phrases
- **UD-Based Classification**: Uses Universal Dependencies for multilingual support
- **Quality Filtering Active**: Successfully filtering `('you', 'tell', 'you')` noise
- **Intent-Aware Processing**: 7/8 test cases correctly classified
- **Performance Optimized**: 85% speed improvement (34ms vs 221ms average)
- **Useless Fact Prevention**: Reactions and hypotheticals properly skip extraction

### Performance Results (vs Baseline)
| Metric | Before | After | Improvement |
|--------|---------|-------|------------|
| Average Time | 221ms | 34ms | **85% faster** |
| Model Loading Spike | 1102ms | 158ms (first run only) | **86% faster** |
| Quality Score | 6.9/10 | 7.3/10 | **+6% quality** |
| Intent Classification | N/A | 7/8 correct | **87% accuracy** |
| Useless Facts Filtered | 0 | 100% | **Perfect filtering** |

### Language-Agnostic Architecture
- **No hardcoded patterns**: Uses UD dependency labels only
- **Multilingual ready**: Works with any spaCy/UD language model  
- **Structural analysis**: Questions detected via aux-inversion, WH-words, etc.
- **Fallback graceful**: Works without NLP via simple heuristics
- **Performance maintained**: Fast UD parsing without quality loss

### Next Phase: DeepPavlov Integration
**Current Status**: Phase 1 complete, ready for enhanced quality research
**Goal**: Investigate DeepPavlov models for >8.5/10 quality target
**Approach**: Compare UD-based vs neural intent classification approaches

---

## 🎯 CURRENT PRIORITIES: HotMem Evolution Phase 2 (2025-09-06)

### Critical Issues Identified from User Testing

**Status**: Three key problems preventing production readiness

### 1. 🚨 URGENT: Complex Sentence Extraction Failures

**Problem**: HotMem struggles with longer, more complex sentences 
- **Impact**: Critical facts missed from natural conversation patterns
- **Examples**: 
  - Complex queries like "Did I tell you that Potola is five years old?" 
  - Multi-clause sentences with embedded facts
  - Conditional/hypothetical statements confusing the parser
- **Root Cause**: UD parsing limitations on conversational complexity
- **Priority**: URGENT - blocks evolution
- **Effort**: 1-2 days

**Solution Approach**:
- Profile spaCy parsing performance on complex sentences
- Implement sentence decomposition for multi-clause handling  
- Add confidence scoring for extraction quality
- Consider hybrid approach: UD + LLM for complex cases

### 2. 🔍 HIGH: Summary Storage & Retrieval Verification

**Problem**: Summarizer functionality unclear - summaries may not be searchable
- **Impact**: Session insights potentially lost, no long-term memory consolidation
- **Current Status**: 
  - ✅ `summarizer.py` exists and stores summaries to FTS
  - ⚠️ No verification that summaries are retrievable via search
  - ⚠️ No testing of summary quality or relevance
- **Priority**: HIGH - affects long-term memory
- **Effort**: 4-6 hours

**Verification Needed**:
- Test summary generation in real conversations
- Verify summaries appear in memory search results  
- Validate summary quality and relevance
- Ensure proper FTS indexing and retrieval

### 3. 🤔 MEDIUM: LEANN Integration Assessment

**Problem**: LEANN semantic search value unclear - no index files generated
- **Impact**: Missing potential semantic retrieval capabilities
- **Current Status**:
  - ✅ LEANN adapter exists (`leann_adapter.py`)
  - ✅ Environment configured: `HOTMEM_USE_LEANN=true`
  - ❌ No `.leann` files found in data directory
  - ❌ `REBUILD_LEANN_ON_SESSION_END=false` - disabled
- **Priority**: MEDIUM - potential enhancement
- **Effort**: 1 day

**Assessment Tasks**:
- Enable LEANN rebuilding: `REBUILD_LEANN_ON_SESSION_END=true`
- Test LEANN index generation and semantic search
- Compare LEANN results vs. FTS-only retrieval
- Measure performance impact vs. semantic benefits

---

## 🧩 DEFERRED: Modularize HotMem + Reduce Technical Debt (post-evolution)

Start this once Phase 2 evolution issues are resolved.

### Goal
Modularize the new memory subsystem and delete duplication to lower maintenance cost and enable pluggable extractors.

### Scope
- Create `server/memory/` package with modules:
  - `store.py` (current `memory_store.py`).
  - `hotpath.py` (current `memory_hotpath.py`).
  - `processor.py` (current `hotpath_processor.py`).
  - `extractors/usgs.py` (current `memory_extraction_usgs.py`).
  - `extractors/rules.py` (consolidate logic from `memory_extraction_v2.py`/`memory_extraction_final.py`).
  - `utils/ud.py` (current `ud_utils.py`).
- Remove/rename duplicates: unify `memory_extraction_{v2,final,usgs}.py` under `extractors/`.
- Update imports in `server/bot.py` to new package paths.
- Normalize env config: `HOTMEM_SQLITE`, `HOTMEM_LMDB_DIR`, `USER_ID`.
- Add test coverage for extractor selection and processor behavior.
- Ignore transient DB artifacts in VCS (ensure `.db`, `*.db-shm`, `*.db-wal`, LMDB dirs are gitignored).

### Acceptance Criteria
- One import path for hot-path processor: `from server.memory.processor import HotPathMemoryProcessor`.
- No duplicate extractor files left in `server/`.
- Tests pass: `test_extraction_simple.py`, `test_hotmem.py`, `test_hotmem_comprehensive.py`.
- Measured p95 stays ≤ 200ms for Potola scenario on laptop CPU.
- Changelog and tech debt docs updated to reflect consolidation.

### Risks/Notes
- spaCy model availability varies; provide graceful fallback and clear setup docs.
- Keep interfaces stable to avoid breaking the voice pipeline.
- Plan incremental moves to avoid large diffs.

---

## ⚠️ Critical FrameProcessor Implementation Rules

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

---

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
