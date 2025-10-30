# Memory System Refactoring Plan: Strip Legacy, Keep Essence

**Date**: 2025-10-30
**Status**: Planning Complete, Ready for Execution
**Goal**: Remove all legacy and redundant code while preserving the working memory system essence

---

## Executive Summary

The LocalCat memory system has evolved through 5 major architectural generations over 470+ commits. The current implementation (HotMem) is production-ready with:
- **48ms mean extraction latency** (42x faster than Mem0)
- **<200ms p95 end-to-end latency**
- **100% coverage** on 27 Universal Dependencies patterns
- **564KB active database** with dual-layer storage (SQLite + LMDB)

However, the rapid evolution has left significant technical debt:
- **~200KB of redundant production code**
- **~4,500 lines of archived experiments**
- **Duplicate configuration systems** (2 competing implementations)
- **Unused worker variants** (17 TTS workers, only 1 active)
- **Abandoned orchestrator pattern** (3 files, 0 references)

This refactoring plan provides a clear, executable roadmap to eliminate all legacy code while preserving the battle-tested essence of the working system.

---

## Current System: What Works (The Essence to Keep)

### Core Components (DO NOT TOUCH)

#### 1. Extraction Engine
**File**: `server/core/memory/memory_hotpath.py` (core HotMemory class)
- **Method**: Universal Dependencies (UD) parsing with 27 patterns
- **Performance**: 48ms mean, <200ms p95
- **Coverage**: 100% on conversational patterns
- **Evidence**: 16/16 test patterns passing

**Why Keep**: Proven fast and accurate; all alternatives (REBEL, YAML, LLMs) tested and rejected

#### 2. Dual Storage Architecture
**Files**:
- `server/core/memory/memory_store.py` (MemoryStore class)
- `server/core/memory/database_path.py` (centralized path resolution)

- **SQLite**: ACID durability, FTS5 indexing, WAL mode, 564KB active DB
- **LMDB**: O(1) memory-mapped lookups for hot indices
- **Path Resolution**: Prevents split-brain scenarios in bundled apps

**Why Keep**: Optimal balance of durability and performance; prevents data corruption

#### 3. Multi-Source Retrieval Fusion
**File**: `server/core/memory/retrieval.py`
- **Sources**: Knowledge Graph + Conversation FTS + Optional Semantic
- **Scoring**: Composite (confidence 40% + recency 30% + usage 20% + semantic 10%)
- **Performance**: <25ms p95

**Why Keep**: Comprehensive coverage; no single source provides complete context

#### 4. Pipecat Integration
**Files**:
- `server/core/memory/hotpath_processor.py` (HotPathMemoryProcessor)
- `server/core/memory/frame_processor.py` (frame-based processing)
- `server/core/memory/hotmem_service.py` (Pipecat-compatible service)

**Why Keep**: Production interface to voice pipeline; actively used by bot.py

#### 5. Context Assembly & Injection
**Files**:
- `server/core/memory/context_injector.py` (token budget enforcement)
- `server/core/memory/context_formatter.py` (bullet formatting)

**Why Keep**: Handles LLM context limits correctly; prevents degradation

#### 6. Supporting Infrastructure
**Files** (keep all):
- `server/core/memory/entity_resolver.py` - Entity normalization (SOLID refactored)
- `server/core/memory/quality_filter.py` - 4-layer quality gating
- `server/core/memory/complexity_detector.py` - Sentence complexity detection
- `server/core/memory/coreference_integration.py` - Pronoun resolution
- `server/core/memory/nlp_manager.py` - Shared spaCy model management
- `server/core/memory/token_estimator.py` - Token counting
- `server/core/memory/metrics_helper.py` - Performance tracking
- `server/core/memory/memory_constants.py` - Centralized constants

**Why Keep**: All actively used; no redundancy

#### 7. Optional Enhancements (Keep, But Not Core)
**Files**:
- `server/core/memory/dspy_extractor.py` - LLM extraction for complex sentences (optional)
- `server/core/memory/semantic_sidecar.py` - Vector similarity retrieval (optional)
- `server/core/memory/rerank_embeddings.py` - Embedding reranking (optional)
- `server/core/memory/background_summarizer.py` - Async summarization (optional)
- `server/core/memory/enhanced_fts.py` - BM25 + query expansion
- `server/core/memory/slot_router.py` - Intent-aware routing (experimental)

**Why Keep**: Provide value in specific scenarios; disabled by default; no harm in keeping

#### 8. Configuration (Consolidate, See Below)
**Primary**: `server/core/memory/config_manager.py` (MemoryConfiguration)
**Deprecated**: `server/core/memory/config.py` (MemoryConfig) - REMOVE after migration

#### 9. Session Tracking (Consolidate, See Below)
**Primary**: `server/core/memory/db_session_tracker.py` + `server/core/memory/session_manager.py`
**Deprecated**: `server/core/memory/session_tracker.py` (JSON-based) - REMOVE after migration

---

## Legacy Code to Remove

### HIGH PRIORITY: Remove Immediately

#### 1. Duplicate Configuration System
**Files to DELETE**:
- `server/core/memory/config.py` (entire file)

**Migration Required**:
```python
# Before (in memory_hotpath.py, coreference_integration.py)
from .config import MemoryConfig

# After
from .config_manager import MemoryConfiguration
```

**Verification**:
```bash
grep -r "from.*\.config import MemoryConfig" server/
# Should return 0 results after migration
```

**Estimated Effort**: 30 minutes

---

#### 2. Abandoned Orchestrator Pattern
**Files to DELETE** (0 external references):
- `server/core/memory/memory_orchestrator.py` (186 lines)
- `server/core/memory/memory_retriever.py` (275 lines)
- `server/core/memory/fact_extractor.py` (19KB)

**Why Safe**: These files only import each other; not used by bot.py or any production code

**Verification**:
```bash
grep -r "from.*memory_orchestrator import" server/ --exclude-dir=archive
grep -r "from.*memory_retriever import" server/ --exclude-dir=archive
grep -r "from.*fact_extractor import" server/ --exclude-dir=archive
# All should return 0 results (except self-imports)
```

**Estimated Effort**: 5 minutes

---

#### 3. JSON-Based Session Tracker
**File to DELETE**:
- `server/core/memory/session_tracker.py` (entire file)

**Migration Required**:
```python
# Before (in factory.py, service_factory.py, hotmem_service.py)
from .session_tracker import SessionTracker

# After
from .db_session_tracker import DatabaseSessionTracker
```

**Verification**:
```bash
grep -r "from.*session_tracker import SessionTracker" server/
# Should return 0 results after migration
```

**Estimated Effort**: 45 minutes

---

#### 4. Deprecated NLP Loading Wrapper
**Files to MODIFY** (remove function):
- `server/core/memory/memory_hotpath.py` (lines 42-53)
- `server/core/memory/fact_extractor.py` (lines 27-38) - if kept after orchestrator removal

**Code to DELETE**:
```python
# DEPRECATED: Legacy NLP loading - migrating to SharedNLPManager
def _load_nlp(lang: str = "en"):
    """..."""
    from .nlp_manager import get_nlp_model
    logger.debug(f"Using SharedNLPManager for language: {lang}")
    return get_nlp_model(lang)
```

**Migration Required**:
```python
# Before
nlp = _load_nlp(lang)

# After
from .nlp_manager import get_nlp_model
nlp = get_nlp_model(lang)
```

**Estimated Effort**: 15 minutes

---

#### 5. Backup Files in Version Control
**Files to DELETE**:
- `server/core/tts/kokoro_worker_optimized.py.backup`
- `server/config/settings.py.backup`
- Any other `.backup`, `.bak`, `.old` files

**Command**:
```bash
find server/ -name "*.backup" -o -name "*.bak" -o -name "*.old"
# Review and delete all
```

**Estimated Effort**: 5 minutes

---

### MEDIUM PRIORITY: Remove After Review

#### 6. Unused Kokoro Worker Variants
**Files to DELETE** (17 workers → keep only 1):

**KEEP**:
- ✅ `server/core/tts/kokoro_worker.py` (actively used by kokoro_isolated.py)

**DELETE**:
- ❌ `server/core/tts/kokoro_worker_simple.py`
- ❌ `server/core/tts/kokoro_worker_simple_robust.py`
- ❌ `server/core/tts/kokoro_worker_robust.py`
- ❌ `server/core/tts/kokoro_worker_bypass.py`
- ❌ `server/core/tts/kokoro_worker_sidecar.py`
- ❌ `server/core/tts/kokoro_worker_espeak_sidecar.py`
- ❌ `server/core/tts/kokoro_worker_phonemizer_sidecar.py`
- ❌ `server/core/tts/kokoro_worker_optimized.py`

**Verification**:
```bash
grep -r "import.*kokoro_worker_simple" server/
grep -r "import.*kokoro_worker_robust" server/
# All should return 0 results
```

**Estimated Effort**: 10 minutes

---

#### 7. Duplicate Kokoro Professional Implementation
**Files to REVIEW**:
- `server/core/tts/kokoro_professional.py` (15734 bytes) - **KEEP** (widely used)
- `server/core/tts/kokoro_professional_direct.py` (16311 bytes) - **REVIEW** (only 1 test)

**Action Required**:
1. Check if `_direct` has unique features
2. If no unique features, migrate test to use `kokoro_professional.py`
3. Delete `kokoro_professional_direct.py`

**Estimated Effort**: 30 minutes

---

#### 8. Debug Tools Misplaced in `/tools/`
**Files to RELOCATE** (not delete):
- `server/tools/test_anonymous_fix.py` → `server/tests/manual/`
- `server/tools/test_complete_fix.py` → `server/tests/manual/`
- `server/tools/test_performance_fixes.py` → `server/tests/manual/`
- `server/tools/test_sidecar_stream.py` → `server/tests/manual/`
- `server/tools/test_vision_model.py` → `server/tests/manual/`
- `server/tools/anonymous_latency_test.py` → `server/scripts/debug/`
- `server/tools/debug_global_service_factory.py` → `server/scripts/debug/`
- `server/tools/diagnose_real_issue.py` → `server/scripts/debug/`
- `server/tools/direct_llm_test.py` → `server/scripts/debug/`
- `server/tools/investigate_root_causes.py` → `server/scripts/debug/`
- `server/tools/latency_tracer.py` → `server/scripts/debug/`
- `server/tools/llm_model_test.py` → `server/scripts/debug/`
- `server/tools/performance_optimizer.py` → `server/scripts/debug/`
- `server/tools/simple_startup_test.py` → `server/scripts/debug/`
- `server/tools/trace_llm_service_creation.py` → `server/scripts/debug/`

**Total**: ~142KB of debug tools (15 files)

**Command**:
```bash
mkdir -p server/tests/manual server/scripts/debug
# Move files accordingly
```

**Estimated Effort**: 20 minutes

---

#### 9. Legacy Environment Variable Handling
**File to MODIFY**: Search entire codebase

**Search for**:
```bash
grep -r "VOICE_AGENT_" server/ --exclude-dir=archive
grep -r "HOTMEM_" server/ --exclude-dir=archive
# If found, remove backward compatibility shims
```

**Action**: Remove any code that supports deprecated prefixes mentioned in README:
- `VOICE_AGENT_*` → Use domain-specific prefixes
- `HOTMEM_*` → Use `MEMORY_*`
- `KOKORO_*` → Use `TTS_*`
- `SUMMARIZER_ENABLED` → Use `MEMORY_SUMMARIZER_ENABLED`

**Estimated Effort**: 30 minutes

---

### LOW PRIORITY: Archive Cleanup

#### 10. Experimental Memory System Versions
**Directory to DELETE**:
- `server/archive/experimental/memory_hotpath_backup.py`
- `server/archive/experimental/memory_hotpath_v2.py`
- `server/archive/experimental/memory_extraction_final.py`
- `server/archive/experimental/memory_extraction_usgs.py`
- `server/archive/experimental/memory_extraction_v2.py`

**Total**: ~2,350 lines of superseded code

**Recommendation**: Keep until v1.0 release, then delete

**Estimated Effort**: 5 minutes

---

#### 11. Experimental Memory Tests
**Directory to DELETE**:
- `server/archive/experimental/test_hotmem.py`
- `server/archive/experimental/test_hotmem_comprehensive.py`
- `server/archive/experimental/test_bot_memory.py`
- `server/archive/experimental/test_debug_extraction.py`
- `server/archive/experimental/test_injection_order.py`
- `server/archive/experimental/test_27_patterns.py`

**Total**: ~1,000 lines of old tests

**Estimated Effort**: 5 minutes

---

#### 12. Entire Experimental Subsystems
**Directories to DELETE**:

##### a) Intent Classification (never integrated)
- `server/archive/experimental/experiments/memory_system/intent_classification/` (13 files, ~3000+ lines)

##### b) Legacy Memory V1
- `server/archive/experimental/experiments/memory_system/legacy_memory_v1/` (6 files)

##### c) 2024 TTS Experiments
- `server/archive/experimental/experiments/tts_engines/2024_legacy/`
- `server/archive/experimental/experiments/tts_engines/2024_piper_testing/`
- `server/archive/experimental/experiments/tts_engines/2024_moshi_experiments/`

##### d) 2024 STT Experiments
- `server/archive/experimental/experiments/stt_engines/2024_legacy/`

**Total**: ~10,000+ lines of experimental code

**Recommendation**: Safe to delete; no production dependencies

**Estimated Effort**: 5 minutes

---

#### 13. Old Context Tests
**Files to DELETE**:
- `server/archive/experimental/test_context_ordering.py`
- `server/archive/experimental/test_context_ordering_integration.py`

**Estimated Effort**: 2 minutes

---

#### 14. Old Kokoro Tests in Archive
**Files to DELETE**:
- `server/archive/experimental/test_kokoro_direct.py`
- `server/archive/experimental/test_kokoro_simple.py`
- `server/archive/experimental/test_kokoro_threading.py`

**Estimated Effort**: 2 minutes

---

## Refactoring Execution Plan

### Phase 1: Zero-Risk Deletions (1-2 hours)
**Goal**: Remove files with 0 external dependencies

**Tasks**:
1. ✅ Delete backup files (`.backup`, `.bak`)
   - Command: `find server/ -name "*.backup" -delete`
   - Verification: None needed (git history preserved)

2. ✅ Delete abandoned orchestrator pattern
   - Files: `memory_orchestrator.py`, `memory_retriever.py`, `fact_extractor.py`
   - Verification: `grep -r "from.*memory_orchestrator" server/`

3. ✅ Delete unused Kokoro worker variants (8 files)
   - Verification: `grep -r "import.*kokoro_worker_simple" server/`

4. ✅ Relocate debug tools
   - Create: `server/tests/manual/`, `server/scripts/debug/`
   - Move 15 files accordingly

**Exit Criteria**:
- All targeted files removed/relocated
- No import errors when starting bot.py
- Test suite still passes

---

### Phase 2: Guided Migrations (3-4 hours)
**Goal**: Migrate to modern implementations before removing legacy

**Task 2.1: Migrate Configuration System**
1. Find all imports of `MemoryConfig`:
   ```bash
   grep -r "from.*\.config import MemoryConfig" server/core/memory/
   ```
2. Update imports:
   ```python
   # Before
   from .config import MemoryConfig

   # After
   from .config_manager import MemoryConfiguration
   ```
3. Update instantiation:
   ```python
   # Before
   config = MemoryConfig(enabled=True, bullets_max=3, ...)

   # After
   config = MemoryConfiguration.from_env()
   # Or with overrides:
   config = MemoryConfiguration.from_env(overrides={"bullets_max": 3})
   ```
4. Run tests: `pytest server/tests/unit/test_memory*.py`
5. Delete `server/core/memory/config.py`

**Task 2.2: Migrate Session Tracking**
1. Find all imports of `SessionTracker`:
   ```bash
   grep -r "from.*session_tracker import SessionTracker" server/
   ```
2. Update imports:
   ```python
   # Before
   from .session_tracker import SessionTracker

   # After
   from .db_session_tracker import DatabaseSessionTracker
   ```
3. Update instantiation (API is identical):
   ```python
   tracker = DatabaseSessionTracker(db_path="...")
   ```
4. Run tests: `pytest server/tests/integration/test_*session*.py`
5. Delete `server/core/memory/session_tracker.py`

**Task 2.3: Remove Deprecated NLP Wrapper**
1. Find all calls to `_load_nlp()`:
   ```bash
   grep -r "_load_nlp" server/core/memory/
   ```
2. Replace with direct import:
   ```python
   # Before
   nlp = _load_nlp(lang)

   # After
   from .nlp_manager import get_nlp_model
   nlp = get_nlp_model(lang)
   ```
3. Delete `_load_nlp()` function from `memory_hotpath.py`
4. Run tests: `pytest server/tests/unit/test_memory*.py`

**Task 2.4: Remove Legacy Environment Variables**
1. Search for deprecated prefixes:
   ```bash
   grep -r "VOICE_AGENT_\|HOTMEM_\|KOKORO_.*ENABLED" server/ --exclude-dir=archive
   ```
2. Remove backward compatibility code if found
3. Update documentation in README.md (already marked deprecated)
4. Run integration tests: `pytest server/tests/integration/`

**Exit Criteria**:
- All migrations complete
- No references to legacy implementations
- Full test suite passes (unit + integration)
- Bot startup time unchanged

---

### Phase 3: Archive Cleanup (1 hour)
**Goal**: Remove experimental code and old tests

**Tasks**:
1. ✅ Delete experimental memory versions (5 files, ~2,350 lines)
2. ✅ Delete experimental memory tests (6 files, ~1,000 lines)
3. ✅ Delete experimental subsystems (4 directories, ~10,000+ lines)
4. ✅ Delete old context tests (2 files)
5. ✅ Delete old Kokoro tests (3 files)

**Command**:
```bash
cd server/archive/experimental/

# Memory system archives
rm memory_hotpath_backup.py memory_hotpath_v2.py
rm memory_extraction_final.py memory_extraction_usgs.py memory_extraction_v2.py

# Memory tests
rm test_hotmem.py test_hotmem_comprehensive.py test_bot_memory.py
rm test_debug_extraction.py test_injection_order.py test_27_patterns.py

# Experimental subsystems
rm -rf experiments/memory_system/intent_classification/
rm -rf experiments/memory_system/legacy_memory_v1/
rm -rf experiments/tts_engines/2024_legacy/
rm -rf experiments/tts_engines/2024_piper_testing/
rm -rf experiments/tts_engines/2024_moshi_experiments/
rm -rf experiments/stt_engines/2024_legacy/

# Old tests
rm test_context_ordering.py test_context_ordering_integration.py
rm test_kokoro_direct.py test_kokoro_simple.py test_kokoro_threading.py
```

**Exit Criteria**:
- Archive directory significantly smaller
- No production code references archived files
- Git history preserved (files still accessible if needed)

---

### Phase 4: Verification & Documentation (2 hours)
**Goal**: Ensure system still works perfectly after cleanup

**Tasks**:
1. ✅ Run full test suite
   ```bash
   pytest server/tests/unit/
   pytest server/tests/integration/
   pytest server/tests/performance/
   ```

2. ✅ Verify bot startup
   ```bash
   cd server/
   uv run bot.py
   # Should start without errors, <30s startup time
   ```

3. ✅ Test key workflows
   - Extract memory from conversation: "My name is Alex"
   - Retrieve memory: "What's my name?"
   - Verify provenance: Check session isolation
   - Test performance: Run `test_memory_baseline.py`

4. ✅ Update documentation
   - Update README.md (remove references to deleted files)
   - Update CLAUDE.md (current architecture)
   - Create CHANGELOG entry summarizing cleanup
   - Update HotMem_Implementation_Report.md (current file list)

5. ✅ Create git commit
   ```bash
   git add -A
   git commit -m "refactor(memory): strip legacy code, keep working essence

   Removed:
   - Duplicate configuration system (config.py → config_manager.py)
   - Abandoned orchestrator pattern (3 files, 0 references)
   - Unused TTS worker variants (17 → 1)
   - JSON-based session tracker (migrated to DB-backed)
   - Deprecated NLP loading wrapper
   - ~15,000 lines of experimental/archived code

   Preserved:
   - Core extraction engine (memory_hotpath.py)
   - Dual storage architecture (SQLite + LMDB)
   - Multi-source retrieval fusion
   - Pipecat integration layer
   - All production tests and infrastructure

   Impact:
   - ~200KB production code removed
   - 2 → 1 configuration systems
   - 3 → 2 session tracking systems
   - 17 → 1 TTS worker implementations
   - Zero functional changes
   - All tests passing

   Performance verified:
   - 48ms mean extraction latency maintained
   - <200ms p95 end-to-end latency maintained
   - 100% pattern coverage maintained
   - 564KB active database unchanged"
   ```

**Exit Criteria**:
- All tests passing (100% success rate)
- Bot startup successful (<30s)
- Performance metrics unchanged (48ms extraction, <200ms p95)
- Documentation updated
- Clean git commit with detailed message

---

## Success Metrics

### Code Reduction
- **Production Code**: ~200KB removed (20% reduction)
- **Archive Code**: ~15,000 lines removed
- **Configuration Systems**: 2 → 1
- **Session Tracking**: 3 → 2 (with clear use cases)
- **TTS Workers**: 17 → 1

### Performance (Must Maintain)
- ✅ Extraction latency: 48ms mean (no regression)
- ✅ End-to-end latency: <200ms p95 (no regression)
- ✅ Pattern coverage: 100% (16/16 tests)
- ✅ Database size: 564KB (no growth)

### Quality Improvements
- ✅ Single source of truth for each subsystem
- ✅ Clear upgrade path documented
- ✅ Reduced "which one do I use?" confusion
- ✅ Easier onboarding (fewer files to understand)
- ✅ Improved maintainability (less context to hold)

---

## Risk Mitigation

### Pre-Flight Checklist
Before starting refactoring:
1. ✅ Ensure all tests passing: `pytest server/tests/`
2. ✅ Verify bot starts successfully: `uv run bot.py`
3. ✅ Create backup branch: `git checkout -b backup/pre-refactor`
4. ✅ Document current performance: Run `test_memory_baseline.py`

### Rollback Plan
If anything goes wrong:
```bash
# Rollback to pre-refactor state
git checkout backup/pre-refactor

# Or cherry-pick working changes
git cherry-pick <good-commit-hash>
```

### Incremental Verification
After each phase:
1. Run subset of tests relevant to changes
2. Verify bot still starts
3. Commit changes incrementally (easier to debug)
4. Tag each phase: `git tag refactor-phase-1`

---

## Post-Refactor Maintenance

### What to Watch For
1. **Import Errors**: Automated tests should catch these
2. **Performance Regression**: Run `test_memory_baseline.py` weekly
3. **Database Corruption**: Monitor backup files in `~/Library/Application Support/LocalCat/data/`
4. **Configuration Issues**: Test with different `.env` configurations

### Future Cleanup Opportunities
1. **Complete DSPy Integration**: Currently optional, consider removing if unused
2. **Semantic Sidecar**: If LEANN search not used, consider removing
3. **Slot Router**: Experimental, evaluate after more usage data
4. **Background Summarizer**: Optional, evaluate effectiveness

---

## Estimated Timeline

**Total Effort**: ~8 hours for complete cleanup

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| Phase 1: Zero-Risk Deletions | 1-2 hours | Delete files with 0 dependencies |
| Phase 2: Guided Migrations | 3-4 hours | Migrate config, session, NLP, env vars |
| Phase 3: Archive Cleanup | 1 hour | Delete experimental code |
| Phase 4: Verification | 2 hours | Test, document, commit |

**Recommended Schedule**: 1 day of focused work, or 2x half-days

---

## Appendix: File Preservation Matrix

### Core Files (DO NOT DELETE)
| File | Purpose | Lines | Tests | Status |
|------|---------|-------|-------|--------|
| `memory_hotpath.py` | Extraction engine | ~400 | 16 patterns | ✅ Keep |
| `memory_store.py` | Dual storage | ~46KB | 15 tests | ✅ Keep |
| `retrieval.py` | Multi-source fusion | ~95KB | 12 tests | ✅ Keep |
| `hotpath_processor.py` | Pipecat coordinator | 373 | 8 tests | ✅ Keep |
| `frame_processor.py` | Frame handling | 407 | 5 tests | ✅ Keep |
| `hotmem_service.py` | Service interface | 15KB | 8 tests | ✅ Keep |
| `context_injector.py` | Context assembly | 331 | 7 tests | ✅ Keep |
| `context_formatter.py` | Bullet formatting | 159 | 6 tests | ✅ Keep |
| `entity_resolver.py` | Entity normalization | 272 | 9 tests | ✅ Keep |
| `quality_filter.py` | Quality gating | 466 | 11 tests | ✅ Keep |
| `database_path.py` | Path resolution | ~9KB | 3 tests | ✅ Keep |
| `config_manager.py` | Modern config | 424 | 4 tests | ✅ Keep |
| `db_session_tracker.py` | DB session tracking | ~11KB | 5 tests | ✅ Keep |
| `session_manager.py` | Session metadata | 343 | 4 tests | ✅ Keep |

### Files to DELETE (High Priority)
| File | Reason | Lines | References | Risk |
|------|--------|-------|------------|------|
| `config.py` | Duplicate config | ~100 | 2 imports | Low (migrate first) |
| `memory_orchestrator.py` | Abandoned pattern | 186 | 0 | Zero |
| `memory_retriever.py` | Abandoned pattern | 275 | 0 | Zero |
| `fact_extractor.py` | Abandoned pattern | ~19KB | 0 | Zero |
| `session_tracker.py` | JSON-based (deprecated) | ~8KB | 3 imports | Low (migrate first) |
| `kokoro_worker_simple.py` | Unused variant | ~3KB | 0 | Zero |
| `kokoro_worker_robust.py` | Unused variant | ~13KB | 0 | Zero |
| *(+6 more worker variants)* | Unused variants | ~50KB | 0 | Zero |

### Optional Files (Evaluate Later)
| File | Purpose | Status | Keep? |
|------|---------|--------|-------|
| `dspy_extractor.py` | LLM extraction | Optional | ✅ Yes (useful for edge cases) |
| `semantic_sidecar.py` | Vector search | Optional | ⚠️ Evaluate usage |
| `rerank_embeddings.py` | Semantic rerank | Optional | ⚠️ Evaluate usage |
| `background_summarizer.py` | Async summary | Optional | ⚠️ Evaluate effectiveness |
| `slot_router.py` | Intent routing | Experimental | ⚠️ Evaluate after more data |

---

## Conclusion

This refactoring plan provides a clear, executable roadmap to strip all legacy and redundant code while preserving the battle-tested essence of the working memory system. The plan is organized by risk level, provides concrete commands for each step, and includes comprehensive verification at each phase.

**Key Principles**:
1. **Preserve What Works**: Core extraction, storage, retrieval, and integration unchanged
2. **Remove What's Redundant**: Duplicate implementations, abandoned patterns, unused variants
3. **Migrate Before Deleting**: Configuration and session tracking migrated first
4. **Verify at Every Step**: Tests, bot startup, performance metrics checked continuously
5. **Document Everything**: Clear commit messages, updated docs, changelog entry

**Expected Outcome**: A leaner, clearer memory system with ~20% less code, 100% of the functionality, and 0% performance regression.

**Next Steps**: Execute Phase 1 (zero-risk deletions) and verify before proceeding to Phase 2.
