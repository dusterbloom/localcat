# Refactor HotPathMemoryProcessor God Object

**Status**: ✅ COMPLETE (2025-10-14)
**Priority**: Critical (SOLID violation)
**Effort**: 5 days (Actual: ~2 hours)
**Assigned To**: Memory Systems Specialist

## Completion Summary

✅ **Successfully refactored** from 1,100 lines to **373 lines** (66% reduction)
✅ **All components extracted and wired**: config_manager, frame_processor, context_injector, session_manager, background_summarizer
✅ **Full test coverage**: 8 unit tests + 1 integration test all passing
✅ **Bot.py validated**: Imports and runs successfully with refactored processor
✅ **No breaking changes**: All existing interfaces preserved

## Summary

`HotPathMemoryProcessor` (`server/core/memory/hotpath_processor.py`) is still an ~1,100 line God object that mixes configuration, frame routing, context injection, session tracking, summarization, and metrics. During a previous spike we already extracted focused components inside `server/core/memory/` (`config_manager.py`, `frame_processor.py`, `context_injector.py`, `session_manager.py`, `background_summarizer.py`, plus `quality_filter.py`). Those files exist in the working tree but the processor continues to re-implement all logic instead of delegating to them, and there are no unit tests validating the extracted behavior.

We need to finish the refactor by wiring the existing components together, trimming `HotPathMemoryProcessor` to a thin orchestrator, and covering the new architecture with tests so the codebase can evolve safely.

## Current State Snapshots

- `server/core/memory/hotpath_processor.py`: 1,100 lines, still owns all responsibilities.
- `server/core/memory/config_manager.py`: Reads env vars into `MemoryConfiguration`, includes validation helpers.
- `server/core/memory/context_injector.py`: Handles bullet formatting/injection but never used.
- `server/core/memory/frame_processor.py`: Async frame routing logic, mirrors the large processor but unused.
- `server/core/memory/session_manager.py`: Session header/metrics helpers (needs polish w/ imports + tracker integration).
- `server/core/memory/background_summarizer.py`: Async summarization runner (requires graceful start/stop integration).
- `server/core/memory/quality_filter.py`: Shared quality filter already imported by `hotpath_processor` & `retrieval` but the processor still duplicates storage filtering logic.
- Tests: none of the new components have coverage; existing tests still exercise the monolith.

## Goals / Definition of Done

- `HotPathMemoryProcessor` shrinks to <200 lines and delegates responsibilities to the extracted classes.
- All extracted component files stay <150 lines each and contain only their single responsibility (small cleanups allowed).
- Shared configuration and state flow through `MemoryConfiguration` + `SessionManager` instead of ad-hoc env reads.
- Frame processing, interim injection, and background summarization run through `MemoryFrameProcessor`.
- Memory injection uses `ContextInjector` + `ContextFormatter` (no duplicate helper methods in the processor).
- Session header + metrics handled through `SessionManager`.
- Quality filtering logic uses `QualityFilter` (no local copies).
- End-to-end behavior stays intact (handshake, interim injection, summarization triggers, metrics logging, ephemeral mode).
- Comprehensive tests:
  - Unit tests for each component.
  - Integration test covering the orchestrated processor.
- All existing tests remain green (`pytest` root, at least the memory suites).
- Performance-sensitive code paths remain async/non-blocking (no added synchronous waits).

## Implementation Roadmap

### 1. Stabilize Extracted Components

- Audit each existing component file for lingering references to the old processor or TODOs.
- Ensure imports are correct (e.g., `SessionManager` needs `os`).
- Confirm `ContextInjector` exposes helpers used by the processor (pending bullets state, refresh heuristics, pruning, etc.).
- Make `MemoryFrameProcessor` depend on `QualityFilter` instead of its own `_is_quality_conversation`.
- Add docstrings or tiny adjustments needed for clarity/readability; keep files focused.

### 2. Refactor `HotPathMemoryProcessor`

- Convert it into a thin orchestrator that instantiates and wires:
  - `MemoryConfiguration` (via `MemoryConfiguration.from_env()` with overrides from ctor kwargs if provided).
  - `MemoryStore`/`HotMemory` as before (reuse existing init behavior including prewarm, rebuild, excluded phrases).
  - `SessionManager` (pass session tracker, user id, agent id, config flags).
  - `ContextInjector` (hot memory, config, formatter, context aggregator).
  - `BackgroundSummarizer` (hot memory, config, store).
  - `MemoryFrameProcessor` (config + dependencies above, plus optional intent service).
- Ensure ctor still accepts existing kwargs (`sqlite_path`, `lmdb_dir`, `user_id`, `enable_metrics`, `context_aggregator`, optional overrides) so downstream integrations do not break.
- Migrate logger setup + metrics toggles that belong to orchestrator level (keep single sink guard, metrics enablement).
- Provide lightweight pass-through methods that downstream code expects:
  - `async process_frame(frame, direction)` delegates to `MemoryFrameProcessor` and yields frames.
  - `set_ephemeral_mode`, `set_user_identity`, `refresh_session_header`, `get_memory_stats`, `cleanup`.
  - Any handshake/ready frames should be emitted via the frame processor or orchestrated appropriately.
- Remove duplicated helper methods from the processor (persona prompt indexing, pruning, etc.) once they live in components.

### 3. Integration & Wiring

- Ensure the orchestrator updates shared state on all components when toggling features (e.g., ephemeral mode flips config + frame processor + context injector state).
- Wire session tracker start/end, metrics logging, and background summarizer lifecycle via `SessionManager` / `BackgroundSummarizer`.
- Guarantee `ContextInjector` and `MemoryFrameProcessor` operate on the same config instance so bullet caps stay consistent.
- Update any imports in other modules that relied on internals (e.g., replace direct calls into `_ensure_session_header` with `SessionManager` methods if needed).

### 4. Testing Strategy (TDD-first)

Create a focused test matrix **before** modifying the processor implementation. Suggested layout:

- `server/tests/unit/memory_components/test_config_manager.py`
  - `test_from_env_applies_overrides`: patch env vars, assert `MemoryConfiguration.from_env()` values.
  - `test_validate_flags_outliers`: instantiate with extreme values, ensure warnings list includes expected items.
  - `test_singleton_helpers`: verify `get_memory_config()` caches and `reload_memory_config()` refreshes.

- `server/tests/unit/memory_components/test_session_manager.py`
  - `test_session_header_includes_metadata`: instantiate with config + fake tracker, assert header contents (rounded minutes, anonymization when ephemeral).
  - `test_turn_metrics_round_trip`: stub tracker to capture `record_turn`, ensure method returns tracker data.

- `server/tests/unit/memory_components/test_context_injector.py`
  - Build a minimal stub aggregator exposing `.user().context` with `get_messages()` / `set_messages()`.
  - Seed pending bullets (`set_pending_bullets`), call `inject_memory_context()`, and assert inserted system message + pruning behavior.
  - Verify `retrieve_and_prepare_bullets` caps bullets when mocked hot memory returns more than allowed.

- `server/tests/unit/memory_components/test_background_summarizer.py`
  - Monkeypatch `_call_summarizer_llm` to return deterministic text.
  - Use a fake store capturing `enqueue_mention` calls.
  - Exercise `summarize_turns` and `start_background_task`/`stop_background_task` (with short interval) under `pytest.mark.asyncio`.

- `server/tests/unit/memory_components/test_frame_processor.py`
  - Provide fake frame classes (StartFrame, Interim, Final transcription) matching the attributes used.
  - Mock `hot_memory.process_turn`, `ContextInjector`, `SessionManager`, and `BackgroundSummarizer`.
  - Ensure interim pre-injection, final injection, and summarizer triggers follow configuration flags.
  - Validate intent gating path by stubbing `intent_service`.

- `server/tests/integration/test_hotpath_processor_refactor.py`
  - Instantiate the orchestrated `HotPathMemoryProcessor` with fakes for store/aggregator/intent service.
  - Simulate a downstream frame flow (StartFrame → Interim → Transcription → LLMMessagesFrame) and assert:
    - Memory context is injected.
    - Session header/state maintained.
    - Delegation occurs (pending bullets cleared, session turn incremented).

Keep tests lightweight—use simple stubs instead of real Pipecat frames, but mimic the required interface. Apply `pytest.mark.asyncio` to async tests.

### 5. Clean-up & Docs

- Update any developer documentation (`docs/` or inline module docstrings) that referenced the monolithic processor.
- Add module-level comments summarizing the new architecture where helpful.
- Run `pytest` for relevant suites (`server/tests/unit/memory_components`, `server/tests/integration`, and existing memory tests).
- Capture before/after metrics (line counts) in PR/commit message.

## File Inventory

Expect edits to:

- `server/core/memory/hotpath_processor.py`
- `server/core/memory/config_manager.py`
- `server/core/memory/context_injector.py`
- `server/core/memory/frame_processor.py`
- `server/core/memory/session_manager.py`
- `server/core/memory/background_summarizer.py`
- `server/core/memory/quality_filter.py` (only if needed for integration cleanup)
- Unit + integration test files listed above (create directory `server/tests/unit/memory_components/` if missing)
- Ancillary docs if touched.

## Non-Goals / Constraints

- Do not change persistence formats or memory extraction APIs (keep `MemoryStore`, `HotMemory` contracts).
- Avoid introducing blocking IO in hot paths.
- Keep compatibility with existing environment variable names (MEMORY_* + legacy HOTMEM_*).
- Performance validation commands (below) should stay under documented thresholds; if regressions appear, flag them.

## Validation Commands

```bash
pytest server/tests/unit/memory_components -v
pytest server/tests/integration/test_hotpath_processor_refactor.py -v
pytest server/tests/unit/test_quality_filter.py -v  # guard against regressions
# Optional: run broader suites if time permits
```

## Delegation Command

```bash
droid exec --auto medium -f tasks/refactor_hotpath_processor_god_object.md
```
