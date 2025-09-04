# Technical Debt

This document tracks existing technical debt in the LocalCat Server project.

## Definition
Technical debt refers to the cost of additional rework caused by choosing an easy solution now instead of a better approach that would take longer.

---

## Current Technical Debt

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

### 🧠 Memory System

**Custom mem0 Service Complexity**
- **Issue**: Custom wrapper needed for LM Studio/Ollama compatibility
- **Impact**: Maintenance burden, potential breaking changes with mem0 updates
- **Solution**: Contribute fixes upstream to mem0 project, or migrate to native solution
- **Priority**: Low
- **Effort**: 1-2 weeks

**Dual Schema Handling**
- **Issue**: Dynamic JSON schema detection based on prompt keywords
- **Impact**: Brittle, may fail if prompts change, hard to debug
- **Solution**: Explicit API calls instead of prompt analysis
- **Priority**: Medium
- **Effort**: 1 week

**Fallback to infer=False**
- **Issue**: Memory extraction disabled for local models due to JSON parsing issues
- **Impact**: Reduced memory intelligence, stores raw conversations without fact extraction
- **Solution**: Fix JSON parsing or implement alternative fact extraction
- **Priority**: High
- **Effort**: 1-2 weeks

### 🔄 Error Handling

**Silent Fallbacks**
- **Issue**: Memory operations fail silently and continue with empty responses
- **Impact**: User unaware of memory system failures, debugging difficulties
- **Solution**: Proper error reporting, health checks, monitoring
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
| **Remaining Warnings (External)** | | | |
| WebSockets Legacy API (uvicorn) | Low | N/A | Low |
| AudioOp Deprecation (pipecat) | Low | N/A | Low |
| **Core System Issues** | | | |
| Fallback to infer=False | High | High | High |
| Dual Schema Handling | Medium | Medium | Medium |
| Silent Fallbacks | Medium | Medium | Medium |
| **Infrastructure** | | | |
| Manual Osaurus Setup | Medium | Medium | Medium |
| Environment Config Complexity | Medium | Medium | Medium |
| Hardcoded Configurations | Medium | Medium | Medium |
| No Integration Tests | Medium | High | Medium |
| Process Management | Low | Medium | Low |
| Mixed Dependencies | Low | High | Low |
| Custom mem0 Service | Low | High | Low |

---

## Contributing

When addressing technical debt:
1. Update this document with progress
2. Add tests for the fixed functionality
3. Document the solution in changelog.md
4. Consider if the fix creates new technical debt

Last updated: 2025-09-04

## 🎉 Recent Success: Major Cleanup Complete!

**2025-09-04**: Successfully eliminated all actionable startup warnings and technical debt:
- ✅ **Clean startup**: No more deprecation warnings from our code
- ✅ **Stable versions**: All dependencies pinned and compatible  
- ✅ **Future-proof imports**: Updated to current Pipecat API
- ✅ **Streamlined dependencies**: Removed incompatible packages

**Impact**: Startup now shows only 2 minor external warnings (uvicorn + pipecat internals) vs. previous 4+ critical warnings.
