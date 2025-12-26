# Supertonic TTS Integration Plan

## Overview

Replace Kokoro TTS implementations with Supertonic for improved performance, text handling, and bundle compatibility.

**Goal**: Reduce TTS complexity from 10+ files to 2 (Supertonic primary, Siri fallback)

---

## Phase 1: Research & Validation (Current)

### 1.1 Verify Supertonic Requirements

- [ ] Install `supertonic` package and dependencies
- [ ] Verify ONNX model download and caching
- [ ] Test basic synthesis on Apple Silicon
- [ ] Measure actual latency (RTF) on target hardware
- [ ] Confirm 44.1kHz output compatibility with Pipecat pipeline

### 1.2 Bundle Compatibility Check

- [ ] Identify ONNX model file locations
- [ ] Test model loading from custom path (for bundling)
- [ ] Verify no Metal/GPU dependencies that could cause hangs
- [ ] Check license compatibility (MIT - should be fine)

### 1.3 Audio Pipeline Compatibility

Current pipeline expects:
- Sample rate: Configurable (currently 24kHz for Kokoro)
- Format: Raw PCM frames
- Streaming: Chunk-based yielding

Supertonic provides:
- Sample rate: 44.1kHz
- Format: 16-bit WAV
- Streaming: Batch synthesis (causal decoder suggests streaming possible)

**Action needed**: Resample 44.1kHz → 24kHz if pipeline requires, or update pipeline to 44.1kHz

---

## Phase 2: Implementation

### 2.1 Create SupertonicTTSService

**File**: `server/core/tts/supertonic_service.py`

```python
"""
SupertonicTTSService - Lightning-fast on-device TTS

Features:
- 167x faster than real-time on M4 Pro
- Automatic handling of numbers, dates, currency
- 44.1kHz high-quality audio output
- ONNX-based (bundle compatible)
"""

from typing import AsyncGenerator
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts import TTSService

class SupertonicTTSService(TTSService):
    """Supertonic TTS with Pipecat integration."""

    VOICES = {
        "M1", "M2", "M3", "M4", "M5",  # Male voices
        "F1", "F2", "F3", "F4", "F5",  # Female voices
    }

    def __init__(
        self,
        voice: str = "F1",
        model_path: str | None = None,  # For bundled models
        inference_steps: int = 2,  # 2 = fast, 5 = higher quality
        target_sample_rate: int = 24000,  # Resample for pipeline compat
        **kwargs
    ):
        super().__init__(sample_rate=target_sample_rate, **kwargs)
        self._voice = voice
        self._model_path = model_path
        self._inference_steps = inference_steps
        self._target_sample_rate = target_sample_rate
        self._native_sample_rate = 44100
        self._tts = None

    async def start(self, frame: Frame):
        await super().start(frame)
        await self._ensure_loaded()

    async def _ensure_loaded(self):
        if self._tts is None:
            from supertonic import Supertonic
            logger.info(f"Loading Supertonic TTS (voice={self._voice})")

            # Allow custom model path for bundling
            if self._model_path:
                self._tts = Supertonic(model_path=self._model_path)
            else:
                self._tts = Supertonic()

            logger.info("Supertonic TTS loaded")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        await self._ensure_loaded()

        yield TTSStartedFrame()

        try:
            # Supertonic handles numbers, dates, currency automatically
            audio = self._tts.synthesize(
                text,
                voice=self._voice,
                steps=self._inference_steps
            )

            # Resample if needed
            if self._target_sample_rate != self._native_sample_rate:
                audio = self._resample(audio)

            # Yield in chunks for streaming behavior
            chunk_samples = 4096
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                yield TTSAudioRawFrame(
                    audio=chunk.tobytes(),
                    sample_rate=self._target_sample_rate,
                    num_channels=1
                )

        except Exception as e:
            logger.error(f"Supertonic TTS error: {e}")
            raise
        finally:
            yield TTSStoppedFrame()

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample from 44.1kHz to target sample rate."""
        import scipy.signal as signal

        ratio = self._target_sample_rate / self._native_sample_rate
        new_length = int(len(audio) * ratio)
        return signal.resample(audio, new_length).astype(np.int16)
```

### 2.2 Add Configuration

**File**: `server/config/base_config.py` (additions)

```python
@dataclass
class TTSConfiguration:
    # Existing fields...

    # Supertonic settings
    supertonic_voice: str = "F1"
    supertonic_model_path: str | None = None  # For bundled models
    supertonic_inference_steps: int = 2  # 2=fast, 5=quality
```

### 2.3 Integrate with ServiceFactory

**File**: `server/core/factories/service_factory.py` (modifications)

```python
def create_tts_service(self, config: TTSConfiguration) -> TTSService:
    engine = config.engine

    if engine == "supertonic":
        from core.tts.supertonic_service import SupertonicTTSService
        return SupertonicTTSService(
            voice=config.supertonic_voice,
            model_path=config.supertonic_model_path,
            inference_steps=config.supertonic_inference_steps,
        )
    elif engine == "siri_streaming":
        # Keep as fallback
        from core.tts.siri_streaming import SiriStreamingTTSService
        return SiriStreamingTTSService(...)
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")
```

---

## Phase 3: Testing

### 3.1 Unit Tests

**File**: `server/tests/unit/test_supertonic_tts.py`

```python
import pytest
from core.tts.supertonic_service import SupertonicTTSService

@pytest.mark.asyncio
async def test_supertonic_basic_synthesis():
    """Test basic text synthesis."""
    tts = SupertonicTTSService(voice="F1")

    frames = []
    async for frame in tts.run_tts("Hello, world!"):
        frames.append(frame)

    assert len(frames) > 2  # Start, audio chunks, stop

@pytest.mark.asyncio
async def test_supertonic_complex_text():
    """Test automatic handling of numbers, dates, currency."""
    tts = SupertonicTTSService(voice="F1")

    # These should work without preprocessing
    test_cases = [
        "The price is $99.99",
        "Available from 2025-12-26",
        "Dr. Smith arrived at 3:30pm",
        "Revenue increased 15.7% to $1.2M",
    ]

    for text in test_cases:
        frames = []
        async for frame in tts.run_tts(text):
            frames.append(frame)
        assert len(frames) > 2, f"Failed for: {text}"

@pytest.mark.asyncio
async def test_supertonic_all_voices():
    """Test all available voices."""
    voices = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

    for voice in voices:
        tts = SupertonicTTSService(voice=voice)
        frames = []
        async for frame in tts.run_tts("Test"):
            frames.append(frame)
        assert len(frames) > 2, f"Failed for voice: {voice}"
```

### 3.2 Integration Test

```python
@pytest.mark.asyncio
async def test_supertonic_in_pipeline():
    """Test Supertonic in full voice pipeline."""
    # Create minimal pipeline with Supertonic
    # Verify audio frames flow correctly
    pass
```

### 3.3 Latency Benchmark

```python
import time

def benchmark_supertonic():
    """Measure actual latency on current hardware."""
    from supertonic import Supertonic

    tts = Supertonic()

    test_texts = [
        "Hello",  # Short
        "The quick brown fox jumps over the lazy dog.",  # Medium
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",  # Long
    ]

    for text in test_texts:
        start = time.perf_counter()
        audio = tts.synthesize(text, voice="F1", steps=2)
        elapsed = time.perf_counter() - start

        audio_duration = len(audio) / 44100
        rtf = elapsed / audio_duration

        print(f"Text: {len(text)} chars")
        print(f"  Synthesis: {elapsed*1000:.1f}ms")
        print(f"  Audio: {audio_duration:.2f}s")
        print(f"  RTF: {rtf:.4f} ({1/rtf:.0f}x real-time)")
```

---

## Phase 4: Bundle Integration

### 4.1 Model Bundling Strategy

Supertonic ONNX models need to be bundled with the app:

```bash
# Download models to bundle location
# Models are on Hugging Face: Supertone/supertonic
app/
├── src-tauri/
│   └── resources/
│       └── models/
│           └── supertonic/
│               ├── model.onnx
│               ├── config.json
│               └── voices/
```

### 4.2 Build Script Modifications

**File**: `app/build-production.sh` (additions)

```bash
# Phase: Bundle Supertonic models
echo "📦 Bundling Supertonic TTS models..."

SUPERTONIC_CACHE="${HF_HUB_CACHE}/models--Supertone--supertonic"
SUPERTONIC_DEST="${RESOURCES_DIR}/models/supertonic"

if [ -d "$SUPERTONIC_CACHE" ]; then
    mkdir -p "$SUPERTONIC_DEST"
    cp -r "$SUPERTONIC_CACHE/snapshots/"*/* "$SUPERTONIC_DEST/"
    echo "✅ Supertonic models bundled"
else
    echo "⚠️ Supertonic models not cached, will download on first run"
fi
```

### 4.3 Daemon Manager Updates

**File**: `app/src-tauri/src/daemon_manager.rs` (modifications)

```rust
// Set Supertonic model path for bundled models
if is_production_bundle {
    let supertonic_path = resource_dir.join("models/supertonic");
    cmd.env("SUPERTONIC_MODEL_PATH", supertonic_path);
}
```

### 4.4 Environment Configuration

```bash
# .env additions
VOICE_AGENT_TTS_ENGINE=supertonic
SUPERTONIC_VOICE=F1
SUPERTONIC_INFERENCE_STEPS=2
# SUPERTONIC_MODEL_PATH=  # Set by daemon_manager in bundle
```

---

## Phase 5: Cleanup (After Validation)

### 5.1 Files to Delete

```bash
# TTS implementations (replaced by Supertonic)
server/core/tts/kokoro_mlx.py
server/core/tts/kokoro_professional.py
server/core/tts/kokoro_isolated.py
server/core/tts/kokoro_sidecar.py
server/core/tts/kokoro_pytorch.py
server/core/tts/tts_mlx_ultra_low_latency.py

# TTS workers (no longer needed)
server/core/tts/kokoro_worker.py
server/core/tts/kokoro_worker_bypass.py
server/core/tts/kokoro_worker_espeak_sidecar.py
server/core/tts/kokoro_worker_phonemizer_sidecar.py
server/core/tts/kokoro_worker_robust.py
server/core/tts/kokoro_worker_sidecar.py
server/core/tts/kokoro_worker_simple.py
server/core/tts/kokoro_worker_simple_robust.py
server/core/tts/kokoro_onnx_worker.py

# Sidecars (replaced)
server/sidecars/tts_sidecar_mlx.py
server/sidecars/tts_sidecar_onnx.py
server/sidecars/tts_sidecar_onnx_hardened.py

# Config (consolidate)
server/core/tts/kokoro_config.py

# Tests for removed implementations
server/core/tts/test_mlx_worker.py
server/core/tts/test_onnx_worker.py
```

### 5.2 Files to Keep

```bash
# New primary TTS
server/core/tts/supertonic_service.py  # NEW

# Fallback TTS
server/core/tts/siri_streaming.py

# Shared utilities (review if still needed)
server/tools/text_formatter.py  # May not be needed with Supertonic
```

### 5.3 Factory Updates

Remove all Kokoro-related engine options, keep only:
- `supertonic` (primary)
- `siri_streaming` (fallback)

---

## Phase 6: Documentation

### 6.1 Update CLAUDE.md

```markdown
### TTS Pipeline
- **Primary**: Supertonic (66M params, 167x real-time, ONNX)
- **Fallback**: Siri Native (macOS only)
```

### 6.2 Update README

Document new TTS options and voice selection.

### 6.3 Migration Guide

For users upgrading from Kokoro-based setup.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Supertonic quality worse than Kokoro | Low | High | Benchmark before committing |
| Bundle size increase | Medium | Low | Models are ~100MB similar to Kokoro |
| API changes in Supertonic | Low | Medium | Pin version in requirements.txt |
| Streaming not working well | Medium | Medium | Keep Siri as fallback |
| 44.1kHz causes pipeline issues | Low | Medium | Resample to 24kHz if needed |

---

## Success Criteria

- [ ] Supertonic synthesizes text correctly
- [ ] Latency ≤ Kokoro (ideally much faster)
- [ ] Complex text (numbers, dates, $) works without preprocessing
- [ ] All 10 voices work
- [ ] Bundle builds successfully
- [ ] Bundle runs without hangs
- [ ] Audio quality acceptable (subjective test)

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Research | 1-2 hours |
| Phase 2: Implementation | 2-3 hours |
| Phase 3: Testing | 1-2 hours |
| Phase 4: Bundle | 2-3 hours |
| Phase 5: Cleanup | 1 hour |
| Phase 6: Documentation | 1 hour |
| **Total** | **8-12 hours** |

---

*Created: December 2025*
