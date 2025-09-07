# Technical Debt

This document tracks existing technical debt in the LocalCat Server project.

## Definition
Technical debt refers to the cost of additional rework caused by choosing an easy solution now instead of a better approach that would take longer.

---

## Current Technical Debt

### 🎯 HotMem Evolution Phase 2/3 Issues (Updated - 2025-09-06)

**Complex Sentence Extraction Failures**  
- **Issue**: UD parsing struggles with multi-clause sentences and embedded facts
- **Impact**: Missing critical information from natural conversation patterns
- **Examples**: 
  - Complex queries: "Did I tell you that Potola is five years old?"
  - Multi-clause with embedded facts
  - Conditional/hypothetical statements confusing parser
- **Root Cause**: spaCy UD limitations on conversational complexity
- **Solution**: Sentence decomposition, confidence scoring, hybrid UD+LLM approach
- **Priority**: URGENT - blocks production use
- **Effort**: 1-2 days

**Summary Storage & Retrieval Hardening**
- **Issue**: Summarizer stores to FTS but retrieval testing incomplete
- **Impact**: Long-term memory consolidation potentially broken
- **Current Status**: 
  - ✅ `summarizer.py` stores summaries every 30 seconds
  - ✅ Fusion path can include summary snippets; orchestrator has `CONTEXT_INCLUDE_SUMMARY`
  - ❌ Strict formatting missing; snippets can be noisy
- **Solution**: Strict prompt (“Key Facts / Topics / Open Loops”), TTL/pruning, simple quality gate before write
- **Priority**: HIGH - affects session continuity
- **Effort**: 6–10 hours

**LEANN Integration Value Unknown**
- **Issue**: LEANN semantic search configured but not generating indices
- **Impact**: Missing potential semantic retrieval improvements
- **Current Status**:
  - ✅ `leann_adapter.py` exists, LEANN installed
  - ✅ `HOTMEM_USE_LEANN=true` configured
  - ❌ `REBUILD_LEANN_ON_SESSION_END=false` (disabled)
  - ❌ No `.leann` files found in data directory
- **Solution**: Enable LEANN rebuilding, benchmark semantic vs FTS performance
- **Priority**: MEDIUM - enhancement opportunity
- **Effort**: 1 day

**Per-turn Context Packing on All Turns**
- **Issue**: Orchestrator currently runs packing on turns with memory bullets; dialogue-only turns can grow
- **Impact**: Risk of exceeding soft token budgets on long chit-chat
- **Solution**: Always pack per final transcription; skip memory/summary sections if empty
- **Priority**: MEDIUM
- **Effort**: 2–4 hours

**Deterministic Utilities**
- **Issue**: Trivial questions (e.g., count letters) still go to LLM
- **Impact**: Occasional accuracy drift
- **Solution**: Add pre-LLM deterministic guards for counts/dates/simple math when high confidence
- **Priority**: MEDIUM
- **Effort**: 2–4 hours

### ⚠️ Remaining Startup Warnings

**WebSockets Legacy API Deprecation**
- **Issue**: uvicorn internally uses deprecated websockets.legacy module
- **Warning**: `websockets.legacy is deprecated; see upgrade instructions`
- **Impact**: Harmless - this is uvicorn's internal code, not ours
- **Solution**: Wait for uvicorn to update their websockets usage
- **Priority**: Low (not actionable by us)
- **Status**: External dependency issue

**AudioOp Deprecation** 
- **Issue**: pipecat internally uses deprecated audioop module
- **Warning**: `'audioop' is deprecated and slated for removal in Python 3.13`
- **Impact**: Harmless - this is pipecat's internal code
- **Solution**: Wait for pipecat to update their audio processing
- **Priority**: Low (not actionable by us) 
- **Status**: External dependency issue

### 🔧 Infrastructure & Configuration

**Manual Osaurus Setup**
- **Issue**: Osaurus configuration is manual (port, model download, server start)
- **Impact**: Setup friction for new users, potential configuration errors
- **Solution**: Automated setup script or configuration validation
- **Priority**: Medium
- **Effort**: 2-4 hours

**Environment Configuration Complexity**
- **Issue**: Multiple services with different port configurations (Ollama 11434, Osaurus 8000)
- **Impact**: Port conflicts (SurrealDB blocking 8000), configuration errors
- **Solution**: Unified configuration management, port availability checks
- **Priority**: Medium
- **Effort**: 3-5 hours

### 🧠 Memory System [FULLY RESOLVED - 2025-09-05]

**🎉 HotMem Implementation Complete**
- **Status**: ✅ **MAJOR SUCCESS** - Achieved <200ms p95 memory system replacing 2s mem0
- **Results**: 
  - Ultra-fast extraction: 3.8ms average processing time 
  - Perfect context integration: Memory bullets properly injected
  - Live production deployment: Working seamlessly in voice conversations
  - 100% USGS pattern coverage: All 27 dependency patterns implemented
  - Real-time memory bullets: Contextual facts injected before each LLM call

**LM Studio Context Accumulation** [RESOLVED - 2025-09-05]
- **Issue**: ~~LM Studio accumulates context across mem0 operations~~
- **Resolution**: **Completely replaced mem0 with HotMem local system**
- **Impact**: Eliminated all context accumulation issues and 2s latency
- **Status**: ✅ **Obsolete** - no longer using LM Studio for memory operations

**All Legacy Memory Issues** [RESOLVED - 2025-09-05]  
- **Custom mem0 Service Complexity**: ✅ Eliminated by removing mem0 dependency
- **Dual Schema Handling**: ✅ No longer needed with local UD extraction
- **JSON Parsing Issues**: ✅ Fixed with local spaCy-based processing
- **Simplified Service Slowness**: ✅ Replaced with ultra-fast HotMem
- **Silent Memory Failures**: ✅ Fixed with comprehensive logging and error handling

### 🔄 Error Handling

**Silent Fallbacks** [PARTIALLY RESOLVED - 2025-09-06]
- **Issue**: ~~Memory operations fail silently and continue with empty responses~~ 
- **Status**: ✅ HotMem has comprehensive logging and error handling
- **Remaining**: Need health checks with exponential backoff for external services (Ollama, LM Studio); degrade banners
- **Solution**: Service health monitoring, automatic restart mechanisms
- **Priority**: Medium
- **Effort**: 1 week

**Process Management**
- **Issue**: No automated process monitoring/restart for Ollama/Osaurus
- **Impact**: Manual intervention required when services crash
- **Solution**: Health check endpoints, automatic restart mechanisms
- **Priority**: Low
- **Effort**: 1 week

### 📦 Dependencies

**Mixed Dependency Sources**
- **Issue**: Using both Ollama and Osaurus for different functions
- **Impact**: Multiple moving parts, different update cycles, compatibility issues
- **Solution**: Standardize on single LLM serving solution (evaluate vLLM on macOS)
- **Priority**: Low
- **Effort**: 1-2 weeks

**Version Pinning**
- **Issue**: Some dependencies not version-pinned in requirements.txt
- **Impact**: Potential breaking changes on updates
- **Solution**: Pin all versions, establish update testing process
- **Priority**: Medium
- **Effort**: 2-3 hours

### 🏗️ Architecture

**Hardcoded Configurations**
- **Issue**: Model names, ports, paths hardcoded in multiple places
- **Impact**: Configuration drift, difficult to modify setup
- **Solution**: Centralized configuration management
- **Priority**: Medium
- **Effort**: 1 week

**No Integration Tests**
- **Issue**: No end-to-end testing of voice pipeline + memory system
- **Impact**: Regressions not caught, deployment risks
- **Solution**: Automated integration test suite
- **Priority**: Medium
- **Effort**: 1-2 weeks

---

## Addressed Technical Debt

### ✅ Resolved

**Pipecat Transport Import Deprecations** (Resolved: 2025-09-04)
- **Issue**: Using deprecated `pipecat.transports.network.small_webrtc` modules
- **Solution**: Updated to new import paths in `bot.py`
- **Impact**: Eliminated startup deprecation warnings

**Dependency Version Incompatibilities** (Resolved: 2025-09-04)
- **Issue**: scikit-learn 1.7.1 incompatible with coremltools, PyTorch 2.8.0 untested
- **Solution**: Downgraded scikit-learn to 1.5.1, PyTorch to 2.5.0, pinned versions in requirements.txt
- **Impact**: Eliminated version compatibility warnings

**mem0 Parameter Compatibility** (Resolved: 2025-09-04)
- **Issue**: async_mode parameter errors with Pipecat
- **Solution**: Custom service wrapper removing incompatible parameters

**JSON Schema Format Incompatibility** (Resolved: 2025-09-04)
- **Issue**: LM Studio requiring json_schema vs mem0's json_object format
- **Solution**: Dynamic conversion in custom service

**System Instructions as Memory** (Resolved: 2025-09-04)
- **Issue**: System prompts being stored as user memories
- **Solution**: Changed role from "user" to "system" in context

**Requirements.txt Cleanup** (Resolved: 2025-09-04)
- **Issue**: Unpinned versions, unused vllm dependency
- **Solution**: Pinned all versions for stability, removed vllm (macOS incompatible)

---

## Prioritization Matrix

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| **HotMem Evolution Phase 2 (2025-09-06)** | | | |
| Complex Sentence Extraction Failures | High | Medium | URGENT |
| Summary Storage & Retrieval Unverified | Medium | Low | HIGH |
| LEANN Integration Value Unknown | Low | Medium | MEDIUM |
| **Remaining Warnings (External)** | | | |
| WebSockets Legacy API (uvicorn) | Low | N/A | Low |
| AudioOp Deprecation (pipecat) | Low | N/A | Low |
| **Infrastructure** | | | |
| Manual Osaurus Setup | Medium | Medium | Medium |
| Environment Config Complexity | Medium | Medium | Medium |
| Hardcoded Configurations | Medium | Medium | Medium |
| No Integration Tests | Medium | High | Medium |
| Process Management | Low | Medium | Low |
| Mixed Dependencies | Low | High | Low |

---

## Contributing

When addressing technical debt:
1. Update this document with progress
2. Add tests for the fixed functionality
3. Document the solution in changelog.md
4. Consider if the fix creates new technical debt

Last updated: 2025-09-06

## 🎉 Recent Success: Major Cleanup Complete!

**2025-09-04**: Successfully eliminated all actionable startup warnings and technical debt:
- ✅ **Clean startup**: No more deprecation warnings from our code
- ✅ **Stable versions**: All dependencies pinned and compatible  
- ✅ **Future-proof imports**: Updated to current Pipecat API
- ✅ **Streamlined dependencies**: Removed incompatible packages

**Impact**: Startup now shows only 2 minor external warnings (uvicorn + pipecat internals) vs. previous 4+ critical warnings.

---

## 📚 Lessons Learned: HotMem Implementation (2025-09-05)

### Key Technical Insights

**1. Framework Integration Patterns Are Critical**
- **Issue**: Initially tried to inject `LLMMessagesFrame` objects competing with context aggregator
- **Solution**: Direct context injection via `context.add_message()` follows Pipecat's design
- **Lesson**: Always understand framework lifecycle and integration patterns before implementing

**2. Frame Processing Order Matters**
- **Issue**: Placed HotMem after context aggregator, missing TranscriptionFrames  
- **Solution**: Pipeline order: `stt → memory → context_aggregator → llm`
- **Lesson**: Understand data flow through pipeline processors before placement decisions

**3. Edge Cases in Non-Streaming Services**
- **Issue**: WhisperSTTServiceMLX sets `is_final=None`, not `True/False`
- **Solution**: Handle `None` as `True` for non-streaming STT services
- **Lesson**: Test with actual frame attributes, not just documentation assumptions

**4. Performance vs. Accuracy Trade-offs**
- **Results**: UD extraction (48ms, 100% patterns) vs mREBEL (17s, 0% success)
- **Insight**: Conversational text needs different extraction approaches than formal text
- **Lesson**: Validate ML model performance on actual use-case data, not benchmarks

**5. Debugging Complex Systems**
- **Breakthrough**: Enhanced frame tracing revealed extraction working but injection failing
- **Tools**: Audio frame filtering, detailed logging, performance metrics
- **Lesson**: Invest in observability early - it pays dividends during debugging

### Implementation Wisdom

**Start Simple, Scale Smart**
- Built working extraction first, then optimized retrieval and injection
- Avoided over-engineering until core functionality was proven
- Modular design allowed incremental improvements

**Documentation Drives Understanding**  
- Reading Pipecat source code and docs was crucial for proper integration
- Understanding context management patterns enabled correct implementation
- Framework documentation often contains critical integration details

**Performance Budgets Work**
- Set 200ms p95 target from start and measured continuously  
- Achieved 3.8ms average - well under budget with room for complexity
- Early performance constraints guide architectural decisions

**Test Real Scenarios Early**
- Live voice conversation testing caught integration issues unit tests missed
- User scenarios (Potola, age, name) provided realistic test cases  
- End-to-end testing reveals system interactions that isolated tests miss

### Future Technical Debt Prevention

1. **Document Integration Patterns**: Capture framework integration learnings
2. **Performance Guardrails**: Automated testing for latency regressions  
3. **Observability First**: Build debugging tools alongside features
4. **Real Data Testing**: Use actual conversation patterns, not synthetic data
5. **Incremental Complexity**: Start with working simple, evolve to optimal complex
