# Technical Debt

This document tracks existing technical debt in the LocalCat Server project.

## Definition
Technical debt refers to the cost of additional rework caused by choosing an easy solution now instead of a better approach that would take longer.

---

## Current Technical Debt

### ⚠️ Startup Warnings & Deprecations

**Pipecat Transport Import Deprecations**
- **Issue**: Using deprecated `pipecat.transports.network.small_webrtc` modules
- **Warning**: `Module 'pipecat.transports.network.small_webrtc' is deprecated, use 'pipecat.transports.smallwebrtc.transport' instead`
- **Impact**: May break in future Pipecat versions
- **Solution**: Update imports to new module paths
- **Priority**: High
- **Effort**: 30 minutes

**WebSockets Legacy API Deprecation**
- **Issue**: Using deprecated websockets.legacy module
- **Warning**: `websockets.legacy is deprecated; see upgrade instructions`
- **Impact**: May break in future websockets versions
- **Solution**: Follow websockets upgrade guide
- **Priority**: Medium  
- **Effort**: 1-2 hours

**Scikit-learn Version Incompatibility**
- **Issue**: scikit-learn 1.7.1 not supported (max 1.5.1)
- **Warning**: `scikit-learn version 1.7.1 is not supported. Maximum required version: 1.5.1`
- **Impact**: ML conversion API disabled
- **Solution**: Downgrade scikit-learn or update coremltools
- **Priority**: Medium
- **Effort**: 30 minutes

**PyTorch Version Warning**
- **Issue**: PyTorch 2.8.0 not tested with coremltools
- **Warning**: `Torch version 2.8.0 has not been tested with coremltools`
- **Impact**: Potential unexpected errors with MLX models
- **Solution**: Use tested PyTorch version (2.5.0) or update coremltools
- **Priority**: Medium
- **Effort**: 30 minutes

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

**mem0 Parameter Compatibility** (Resolved: 2025-09-04)
- **Issue**: async_mode parameter errors with Pipecat
- **Solution**: Custom service wrapper removing incompatible parameters

**JSON Schema Format Incompatibility** (Resolved: 2025-09-04)
- **Issue**: LM Studio requiring json_schema vs mem0's json_object format
- **Solution**: Dynamic conversion in custom service

**System Instructions as Memory** (Resolved: 2025-09-04)
- **Issue**: System prompts being stored as user memories
- **Solution**: Changed role from "user" to "system" in context

---

## Prioritization Matrix

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| **Startup Warnings & Deprecations** | | | |
| Pipecat Transport Deprecations | High | Low | High |
| Scikit-learn Version | Medium | Low | Medium |
| PyTorch Version Warning | Medium | Low | Medium |
| WebSockets Legacy API | Medium | Medium | Medium |
| **Core System Issues** | | | |
| Fallback to infer=False | High | High | High |
| Dual Schema Handling | Medium | Medium | Medium |
| Silent Fallbacks | Medium | Medium | Medium |
| **Infrastructure** | | | |
| Manual Osaurus Setup | Medium | Medium | Medium |
| Environment Config Complexity | Medium | Medium | Medium |
| Version Pinning | Medium | Low | Medium |
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
