# Parakeet Batch vs Streaming: Architecture Analysis

**Purpose**: Document the key differences between `parakeet_batch.py` and `parakeet_streaming.py` to guide fixes for the broken batch version.

**Status**: Batch version is currently broken; streaming version works correctly.

**Last Updated**: 2025-09-30

---

## Executive Summary

The **streaming version** (parakeet_streaming.py) uses a stateful streaming transcriber context that processes audio incrementally and manages VAD state properly. The **batch version** (parakeet_batch.py) attempts to transcribe complete utterances but has critical issues with VAD integration and state management.

### Key Issue in Batch Version
The batch version was recently modified (see system reminder) to add VAD integration state (`_vad_active`, `_buffered_audio`, etc.) but the implementation is **incomplete**:
- ❌ Missing `process_frame()` method implementation
- ❌ `run_stt()` method doesn't properly integrate with VAD lifecycle
- ❌ No proper finalization when `UserStoppedSpeakingFrame` arrives

---

## Architectural Differences

### 1. Model Initialization

#### Streaming (CORRECT) ✅
```python
# Creates a STREAMING TRANSCRIBER CONTEXT
self._transcriber_context = self._model.transcribe_stream(
    context_size=self.context_size,
    depth=self.depth,
    keep_original_attention=False  # Local attention for streaming
)
self._transcriber = self._transcriber_context.__enter__()
```

**Key Points**:
- Uses `transcribe_stream()` API
- Enters context manager to get stateful transcriber
- Maintains state across multiple audio chunks
- Can be reset between conversation turns

#### Batch (NEEDS FIX) ❌
```python
# Just loads the model - no transcriber context
if PARAKEET_OLD_FORMAT:
    self._model = load_model(self._model_path)
else:
    self._model = from_pretrained(self._model_path)
```

**Issues**:
- Only loads base model, no transcriber context
- Uses `transcribe(audio_path)` which expects complete audio files
- Cannot process streaming audio chunks
- No state management

---

### 2. Audio Processing Pipeline

#### Streaming (CORRECT) ✅
```python
async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
    # 1. Convert and normalize audio
    audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    audio_array = self._normalize_audio(audio_array)

    # 2. Check VAD state
    should_process = self._vad_active  # External VAD from SmartTurn

    # 3. Buffer audio if active
    if should_process:
        self.audio_buffer.append(audio_array)
        self.buffer_duration += len(audio_array) / 16000.0

    # 4. Process when buffer reaches chunk_duration
    if self.buffer_duration >= self.chunk_duration:
        full_audio = np.concatenate(self.audio_buffer)
        audio_mlx = mx.array(full_audio)

        # 5. Feed to streaming transcriber
        self._transcriber.add_audio(audio_mlx)

        # 6. Get incremental result
        result = self._transcriber.result

        # 7. Yield interim frames with only NEW text
        if hasattr(result, 'text'):
            full_text = result.text.strip()
            if len(full_text) > self._last_sent_length:
                new_text = full_text[self._last_sent_length:]
                yield InterimTranscriptionFrame(text=new_text, ...)
                self._last_sent_length = len(full_text)
```

**Key Features**:
- ✅ Processes audio incrementally
- ✅ Yields `InterimTranscriptionFrame` during speech
- ✅ Tracks sent text to avoid duplicates
- ✅ Uses streaming transcriber API

#### Batch (BROKEN) ❌
```python
async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
    try:
        if self._vad_active:
            self._append_audio(audio)
        # Batch mode yields no interims
    except Exception as e:
        logger.error(f"Error buffering audio: {e}")
        yield ErrorFrame(...)
```

**Issues**:
- ❌ Only buffers audio, never transcribes it in `run_stt()`
- ❌ No frames yielded until VAD ends
- ❌ Transcription happens in `process_frame()` but that method is INCOMPLETE

---

### 3. VAD Integration

#### Streaming (CORRECT) ✅
```python
async def process_frame(self, frame: Frame, direction=None):
    await super().process_frame(frame, direction)

    if isinstance(frame, UserStartedSpeakingFrame):
        self._vad_active = True
        self.reset_transcriber()  # Clear state for new turn

    elif isinstance(frame, UserStoppedSpeakingFrame):
        now = time.time()
        since_last = now - self._last_finalized_time
        if since_last < 0.15:  # Debounce
            return

        self._vad_active = False
        await self._finalize_pending_transcription()

async def _finalize_pending_transcription(self):
    # 1. Process remaining buffered audio
    if self.audio_buffer:
        full_audio = np.concatenate(self.audio_buffer)
        self._transcriber.add_audio(mx.array(full_audio))

    # 2. Get final result
    result = self._transcriber.result
    full_text = result.text.strip()

    # 3. Yield TranscriptionFrame with complete text
    if full_text:
        frame = TranscriptionFrame(text=full_text, ...)
        await self.push_frame(frame)

    # 4. Reset state for next turn
    self._last_sent_length = 0
    self.audio_buffer = []
    self._current_turn_text = ""
```

**Key Features**:
- ✅ Properly handles VAD start/stop lifecycle
- ✅ Resets transcriber on new turn
- ✅ Finalizes with complete text on stop
- ✅ Debounces to avoid duplicate stops

#### Batch (INCOMPLETE) ❌
```python
async def process_frame(self, frame: Frame, direction=None):
    await super().process_frame(frame, direction)

    if isinstance(frame, UserStartedSpeakingFrame):
        self._vad_active = True
        self._reset_buffer()

    elif isinstance(frame, UserStoppedSpeakingFrame):
        self._vad_active = False
        try:
            audio_bytes = self._get_buffer_bytes()
            self._reset_buffer()
            if not audio_bytes:
                return

            # TRANSCRIBE synchronously (BLOCKING!)
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._process_audio_batch, audio_bytes
            )

            if text:
                await self.push_frame(
                    TranscriptionFrame(text=text, ...)
                )
        except Exception as e:
            logger.error(f"Batch finalize error: {e}")
```

**Issues**:
- ⚠️ Runs `_process_audio_batch()` in executor (blocks thread pool)
- ❌ `_process_audio_batch()` expects WAV file path, gets bytes
- ❌ No debouncing for duplicate stop frames
- ❌ Error handling incomplete

---

### 4. Audio Buffer Management

#### Streaming (CORRECT) ✅
```python
# Simple list of numpy arrays
self.audio_buffer = []  # List[np.ndarray] in float32 [-1, 1]
self.buffer_duration = 0.0

# Append
self.audio_buffer.append(audio_array)
self.buffer_duration += len(audio_array) / 16000.0

# Process
full_audio = np.concatenate(self.audio_buffer)
audio_mlx = mx.array(full_audio)
self._transcriber.add_audio(audio_mlx)

# Clear
self.audio_buffer = []
self.buffer_duration = 0.0
```

#### Batch (CURRENT - CORRECT) ✅
```python
# Same approach - list of float32 arrays
self._buffered_audio: list[np.ndarray] = []  # float32 [-1,1] @ 16kHz
self._buffer_duration: float = 0.0

def _append_audio(self, audio_bytes: bytes) -> None:
    arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    arr = self._normalize_audio(arr)
    self._buffered_audio.append(arr)
    self._buffer_duration += len(arr) / float(self._sample_rate)

def _get_buffer_bytes(self) -> bytes:
    full = np.concatenate(self._buffered_audio) if len(self._buffered_audio) > 1 else self._buffered_audio[0]
    audio_int16 = np.clip(full * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return audio_int16.tobytes()
```

**Note**: Buffer management is actually correct in current batch version!

---

## Critical Bugs in Batch Version

### Bug #1: `_process_audio_batch()` expects WAV file path, gets bytes

**Current Code** (BROKEN):
```python
def _process_audio_batch(self, audio_bytes: bytes) -> str:
    # Expects bytes, converts to WAV
    audio_path = self._audio_bytes_to_wav(audio_bytes)

    # Passes path to transcribe()
    result = self._model.transcribe(audio_path)
```

**Called from** (MISMATCHED):
```python
# process_frame() calls with bytes
text = await asyncio.get_event_loop().run_in_executor(
    None, self._process_audio_batch, audio_bytes  # ← bytes
)
```

**The Fix**:
The signature is actually correct! The issue is that `_process_audio_batch()` creates a WAV file and expects the model to transcribe it. This should work.

### Bug #2: Using batch transcribe API instead of streaming

**Current Approach** (INEFFICIENT):
```python
# Creates temp WAV file every time
audio_path = self._audio_bytes_to_wav(audio_bytes)
result = self._model.transcribe(audio_path)  # Batch API
os.unlink(audio_path)
```

**Better Approach** (from streaming):
```python
# Use streaming API with single utterance
audio_mlx = mx.array(audio_float32)
self._transcriber.add_audio(audio_mlx)
result = self._transcriber.result
self._transcriber.reset()  # Reset for next utterance
```

### Bug #3: No debouncing in `process_frame()`

Streaming version has:
```python
since_last = now - self._last_finalized_time
if since_last < 0.15:  # Debounce
    return
```

Batch version has none! This causes duplicate transcriptions.

---

## Recommended Fixes for Batch Version

### Option A: Minimal Fix (Keep Batch API)
```python
async def process_frame(self, frame: Frame, direction=None):
    await super().process_frame(frame, direction)

    if isinstance(frame, UserStartedSpeakingFrame):
        self._vad_active = True
        self._reset_buffer()

    elif isinstance(frame, UserStoppedSpeakingFrame):
        # Add debouncing
        now = time.time()
        if hasattr(self, '_last_finalized_time'):
            since_last = now - self._last_finalized_time
            if since_last < 0.15:
                return

        self._vad_active = False
        self._last_finalized_time = now

        try:
            audio_bytes = self._get_buffer_bytes()
            self._reset_buffer()

            if not audio_bytes or len(audio_bytes) < 3200:  # <0.1s @ 16kHz
                return

            # Run batch transcription in executor
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._process_audio_batch, audio_bytes
            )

            if text:
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id=self._user_id or "user",
                        timestamp=str(time.time())
                    )
                )
        except Exception as e:
            logger.error(f"Batch finalize error: {e}")
```

### Option B: Use Streaming API (Recommended)
```python
def _init_parakeet_model(self):
    """Initialize with streaming transcriber like parakeet_streaming.py"""
    if PARAKEET_OLD_FORMAT:
        raise ImportError("Legacy format not supported")

    # Load model
    result = from_pretrained(self._model_path)
    self._model = result[0] if isinstance(result, tuple) else result

    # Create streaming transcriber context
    self._transcriber_context = self._model.transcribe_stream(
        context_size=(256, 256),
        depth=self.depth if hasattr(self, 'depth') else 3,
        keep_original_attention=False
    )
    self._transcriber = self._transcriber_context.__enter__()

def _process_audio_batch(self, audio_bytes: bytes) -> str:
    """Process using streaming API (more efficient)"""
    # Convert to float32 [-1, 1]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_np = self._normalize_audio(audio_np)

    # Feed to streaming transcriber
    audio_mlx = mx.array(audio_np)
    self._transcriber.add_audio(audio_mlx)

    # Get result
    result = self._transcriber.result
    text = result.text.strip() if hasattr(result, 'text') else ""

    # Reset for next utterance
    if hasattr(self._transcriber, 'reset'):
        self._transcriber.reset()

    return text
```

---

## Configuration Differences

| Parameter | Streaming | Batch | Notes |
|-----------|-----------|-------|-------|
| `enable_vad` | `False` | N/A | Streaming relies on external VAD (SmartTurn) |
| `chunk_duration` | `1.0s` | N/A | Streaming processes every 1 second |
| `confidence_threshold` | `0.4` | `0.3` | Streaming is stricter |
| `temperature` | `0.1` | `0.0` | Streaming uses slight randomness |
| `depth` | `3` | N/A | Streaming transcriber depth |
| `context_size` | `(256, 256)` | N/A | Streaming context window |

---

## Test Files to Check

1. **scripts/test_parakeet_harvard.py** - Harvard sentences test
2. **scripts/test_parakeet_direct_only.py** - Direct API test
3. **scripts/test_real_parakeet_streaming.py** - Real streaming test
4. **scripts/test_parakeet_depth_comparison.py** - Depth parameter test

---

## Key Learnings from Git History

### Commit `37b6ece` - Initial Streaming Implementation
- Introduced streaming transcriber context
- Added VAD integration with `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame`
- Implemented incremental transcription with `InterimTranscriptionFrame`

### Commit `4501951` - VAD Alignment Fix
- Set `enable_vad=False` to rely on SmartTurn external VAD
- Adjusted thresholds for better quality
- Added debouncing for duplicate stop frames

### Commit `e47f22c` - Batch Version Added
- Initial batch implementation (now broken after modifications)
- Used temp WAV files instead of streaming API
- Missing proper VAD integration

### Recent Changes (system reminder)
- Added VAD state tracking to batch version
- Added audio buffering methods
- **BUT**: `process_frame()` implementation is incomplete
- **MISSING**: Proper initialization with streaming API

---

## Summary: What Needs to be Fixed

1. ✅ **Audio buffer management** - Already correct
2. ❌ **Add debouncing** - Prevent duplicate transcriptions
3. ❌ **Fix `process_frame()`** - Add minimum audio length check
4. ❌ **Optional: Use streaming API** - More efficient than WAV files
5. ❌ **Add `_last_finalized_time` tracking** - For debouncing
6. ❌ **Improve error handling** - Catch edge cases

---

## Codex Instructions

To fix `parakeet_batch.py`:

1. Add `_last_finalized_time: float = 0.0` to `__init__`
2. In `process_frame()` for `UserStoppedSpeakingFrame`:
   - Add debouncing check (0.15s minimum between finalizations)
   - Add minimum audio length check (>0.1s)
   - Update `_last_finalized_time` after processing
3. Test with `scripts/test_parakeet_harvard.py`
4. Verify VAD integration with SmartTurn in `bot.py`

**Reference**: Use `parakeet_streaming.py` as the gold standard for VAD integration patterns.