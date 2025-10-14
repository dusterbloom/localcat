# Spec: Optionalize Heavy ML Dependencies and Lazy Imports

Context
- Recent changes trimmed heavy ML deps (e.g., `torch`, `transformers`) from `server/requirements.txt`.
- Several modules import these libraries unconditionally (e.g., `core/audio/audio_intelligence.py`, `core/[fast_|simple_]intent_classifier.py`, `core/intent_classifier.py`).
- A fresh install without these deps risks `ImportError` when those modules are imported, even if features aren’t used.

Goal
- Keep the base server install minimal and bootable without heavy ML deps.
- Make ML-powered features (audio intelligence, intent classifiers) fully optional and lazily imported.
- Provide a clean opt-in path to enable ML features via an extra requirements file or extras in `pyproject.toml`.

Non-Goals
- Do not re-introduce heavy ML deps into the base `server/requirements.txt`.
- Do not change functional defaults unless necessary for stability; continue to rely on env flags where present.

Acceptance Criteria
- Base server boots and can handle core flows without `torch` or `transformers` installed.
- Importing `core.audio` package does not immediately import `torch`/SpeechBrain.
- Importing classifier modules does not fail at import time if `transformers` is absent; attempting to instantiate a classifier without deps yields a clear, actionable error.
- Tests skip gracefully when ML deps are unavailable.
- Documented opt-in install path for ML features (single command).

Deliverables
1) Lazy import for `core.audio` package
   - Implement module-level `__getattr__` in `server/core/audio/__init__.py` to lazily import symbols from `.audio_intelligence` only when accessed.
   - Keep `__all__` intact.

2) Guard heavy imports in classifiers
   - `server/core/simple_intent_classifier.py`
   - `server/core/fast_intent_classifier.py`
   - `server/core/intent_classifier.py`
   Changes:
   - Move `from transformers import ...` and `import torch` out of module top-level and into `initialize()/__init__` or call sites.
   - Wrap with informative `ImportError` that explains how to enable: e.g., "Install ML extras: pip install -r server/requirements-ml.txt".

3) Ensure factory gating remains safe
   - In `server/core/factory.py`, the `AUDIO_INTELLIGENCE_ENABLED` gate already prevents import when disabled. Retain this order and add a short comment if needed.

4) Testing (TDD)
   Add/adjust tests under `server/tests/`:
   - New: `tests/unit/test_optional_imports.py`
     - `test_import_core_audio_package_without_ml_deps`: Import `core.audio` without touching attributes; ensure no ImportError.
     - `test_factory_returns_none_when_audio_disabled`: With `AUDIO_INTELLIGENCE_ENABLED=false`, `VoiceAgentFactory.create_audio_intelligence_processor()` returns `None` without ImportError even if ML deps missing (monkeypatch env).
     - `test_import_classifier_modules_without_transformers`: `import core.simple_intent_classifier` and `import core.fast_intent_classifier` succeed; creating a classifier raises an informative `ImportError` if transformers missing.
   - Modify: `tests/integration/test_audio_device_debug.py`
     - Add `pytest.importorskip("torch")` and a skip marker if `AUDIO_INTELLIGENCE_ENABLED` is false.

5) Packaging: ML opt-in
   - Add `server/requirements-ml.txt` containing:
     - `torch` (no strict pin; allow platform default)
     - `transformers`
     - `sentencepiece`
     - `accelerate`
     - `speechbrain`
     - `praat-parselmouth`
   - Alternatively (nice-to-have): add `[project.optional-dependencies] ml = [...]` to `server/pyproject.toml` so users can `pip install .[ml]` from `server/`.

6) Docs
   - Update `server/README.md` with a short section:
     - Base install: `pip install -r server/requirements.txt`
     - Enable audio intelligence/intent classifiers: `pip install -r server/requirements-ml.txt`
     - Toggle via env var: `AUDIO_INTELLIGENCE_ENABLED=true|false`.

Implementation Notes
- Use module-level `__getattr__` in `core/audio/__init__.py` to defer import of heavy modules until attributes are actually used.
- For classifier modules, avoid any heavy import at top-level to keep `import core.simple_intent_classifier` safe.
- Keep current defaults; rely on factory gating + try/except for graceful degradation when ML deps are missing.

TDD Plan
1) Write tests (see 4 above) to fail on current branch:
   - Import of classifier modules fails without transformers → should pass after change.
   - Integration test should self-skip when torch missing.
2) Implement lazy imports and guarded imports.
3) Add `requirements-ml.txt` and minimal docs.
4) Run tests locally: `pytest server/`.

Rollback/Backout
- Changes are additive and guarded. If issues arise, revert only edits to `core/audio/__init__.py` and classifier modules; tests will revert with them. The base requirements remain unchanged.

Risks
- Platform-specific torch wheel resolution. Mitigated by keeping torch in optional ML requirements only and documenting install.
- Slight behavioral change: importing `core.audio` no longer exposes symbols until accessed; covered by the lazy `__getattr__` behavior.

Owner
- Python Voice Specialist (via Droid Exec)

Commands (to be run by Droid Exec)
```bash
pytest server/tests -q
```

