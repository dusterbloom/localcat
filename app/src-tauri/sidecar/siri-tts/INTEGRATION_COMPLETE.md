# Siri TTS → Pipecat Integration - Implementation Guide

## ✅ COMPLETED (Phases 1-2)

### Phase 1: Swift Streaming Sidecar ✅
**File:** `main-streaming.swift` (400 lines)
**Binary:** `siri-tts` (115KB, compiled and tested)

**Features Implemented:**
- ✅ Dual-mode operation (file WAV + PCM streaming)
- ✅ Enhanced CLI: `--stream-pcm`, `--text`, `--lang`, `--voice-id`, `--rate`, `--pitch`, `--target-rate`
- ✅ Resampling: 24kHz → 16kHz (WebRTC compatible)
- ✅ Robust error handling with timeouts
- ✅ Clean process lifecycle

**Usage:**
```bash
# File mode (backward compatible)
./siri-tts "Hello" output.wav

# Streaming mode for Pipecat
./siri-tts --stream-pcm --text "Hello" --lang en-US --target-rate 16000 > output.pcm
```

### Phase 2: Python Pipecat Service ✅
**File:** `server/core/tts/siri_streaming.py` (280 lines)

**Features Implemented:**
- ✅ Full Pipecat `TTSService` implementation
- ✅ Async subprocess management
- ✅ PCM streaming via stdout
- ✅ 25+ language voice mappings
- ✅ Binary path resolution (dev + production)
- ✅ Proper frame yielding (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)

---

## 🔨 REMAINING WORK (Phases 3-7)

### Phase 3: ServiceFactory Integration (30 min)

**File to Edit:** `server/core/factories/service_factory.py`

**Changes Required:**

1. Add import at top:
```python
from core.tts.siri_streaming import SiriStreamingTTSService, resolve_siri_binary
```

2. Modify `create_tts_service()` method (around line 195), add new case:
```python
def create_tts_service(self, use_boundaries: bool = True) -> Any:
    # ... existing code ...

    elif self.config.tts_engine == "siri_streaming":
        logger.debug("Using Siri Streaming TTS (native macOS voices)")

        try:
            # Resolve binary path
            from core.tts.siri_streaming import resolve_siri_binary
            siri_binary = resolve_siri_binary()

            tts = SiriStreamingTTSService(
                binary_path=str(siri_binary),
                language=tts_config.get("language", "en-US"),
                voice_id=tts_config.get("voice_id"),  # Optional override
                rate=tts_config.get("speed", 0.52),
                pitch=tts_config.get("pitch", 1.0),
                sample_rate=16000,  # WebRTC standard
            )
            logger.info("✅ Siri Streaming TTS ready (native macOS)")
        except Exception as e:
            logger.error(f"❌ Siri TTS initialization failed: {e}")
            logger.warning("Falling back to Professional Kokoro")
            # Fallback to existing TTS
            tts = ProfessionalKokoroTTSService(
                voice=tts_config["voice"],
                speed=tts_config["speed"],
                sample_rate=tts_config["sample_rate"],
            )

    else:
        # ... existing fallback code ...
```

---

### Phase 4: Configuration Support (20 min)

**File to Edit:** `server/config/settings.py`

**Changes Required:**

1. Add to TTS engine enum/validation:
```python
VALID_TTS_ENGINES = [
    "kokoro_professional",
    "kokoro_mlx",
    "siri_streaming",  # NEW
]
```

2. Add Siri-specific TTS configuration section:
```python
# In TTS configuration section
SIRI_TTS_LANGUAGE = os.getenv("SIRI_TTS_LANGUAGE", "en-US")
SIRI_TTS_RATE = float(os.getenv("SIRI_TTS_RATE", "0.52"))
SIRI_TTS_PITCH = float(os.getenv("SIRI_TTS_PITCH", "1.0"))
SIRI_TTS_VOICE_ID = os.getenv("SIRI_TTS_VOICE_ID")  # Optional override
```

**Environment Variable Documentation:**
```bash
# Add to server/.env or set via Tauri
VOICE_AGENT_TTS_ENGINE=siri_streaming  # Use Siri TTS
SIRI_TTS_LANGUAGE=en-US                 # Language (or it-IT, es-ES, etc.)
SIRI_TTS_RATE=0.52                      # Speech rate 0.0-1.0
SIRI_TTS_PITCH=1.0                      # Pitch multiplier
SIRI_TTS_VOICE_ID=...                   # Optional: explicit voice override
```

---

### Phase 5: Tauri Bundle Configuration (45 min)

**File to Edit:** `app/src-tauri/tauri.conf.json`

**Changes Required:**

1. Add to `bundle.externalBin` array:
```json
{
  "bundle": {
    "externalBin": [
      "sidecar/siri-tts/siri-tts"
    ]
  }
}
```

2. Ensure resources include sidecar directory (should already be present):
```json
{
  "bundle": {
    "resources": [
      "../../server/**/*",
      "sidecar/**/*"  // Includes siri-tts binary
    ]
  }
}
```

**Code Signing (macOS):**

Update `build.sh` to include signing:
```bash
#!/bin/bash
set -e

SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)

echo "Building siri-tts with streaming support..."
swiftc -O \
  -sdk "$SDK_PATH" \
  -target arm64-apple-macos12.0 \
  -o siri-tts \
  main-streaming.swift \
  2>&1 | grep -v "using sysroot"

if [ -f siri-tts ]; then
  echo "✅ Build successful: $(ls -lh siri-tts | awk '{print $5}')"

  # Code sign if Developer ID available
  if security find-identity -v -p codesigning | grep -q "Developer ID Application"; then
    echo "🔏 Code signing binary..."
    codesign --force --sign "Developer ID Application" siri-tts
    echo "✅ Binary signed"
  else
    echo "⚠️  No Developer ID certificate found, skipping code signing"
  fi

  # Test both modes
  echo "Testing file mode..."
  ./siri-tts --text "Test" /tmp/test-file.wav && echo "✅ File mode OK"

  echo "Testing streaming mode..."
  ./siri-tts --stream-pcm --text "Test" --target-rate 16000 2>/dev/null | \
    head -c 1000 > /dev/null && echo "✅ Streaming mode OK"
else
  echo "❌ Build failed"
  exit 1
fi
```

---

### Phase 6: Testing (60 min)

#### 6.1 Unit Tests

**Create:** `server/tests/unit/test_siri_streaming.py`

```python
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from core.tts.siri_streaming import SiriStreamingTTSService, SIRI_VOICE_MAP


@pytest.mark.asyncio
async def test_voice_mapping():
    """Test language → voice ID mapping."""
    assert "en-US" in SIRI_VOICE_MAP
    assert "it-IT" in SIRI_VOICE_MAP
    assert SIRI_VOICE_MAP["en-US"].startswith("com.apple.voice")


@pytest.mark.asyncio
async def test_service_initialization():
    """Test service initialization with valid binary path."""
    with patch.object(Path, 'exists', return_value=True):
        service = SiriStreamingTTSService(
            binary_path="/mock/siri-tts",
            language="en-US",
        )
        assert service._language == "en-US"
        assert service._sample_rate == 16000


@pytest.mark.asyncio
async def test_streaming_output():
    """Test that run_tts yields correct frame types."""
    with patch.object(Path, 'exists', return_value=True):
        service = SiriStreamingTTSService(
            binary_path="/mock/siri-tts",
            language="en-US",
        )

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.stdout.read = AsyncMock(side_effect=[
            b'\x00\x01' * 1000,  # Some PCM data
            b'',  # End of stream
        ])
        mock_process.wait = AsyncMock(return_value=None)
        mock_process.returncode = 0
        mock_process.stderr.read = AsyncMock(return_value=b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            frames = [frame async for frame in service.run_tts("test")]

            assert len(frames) >= 3  # Started + Audio + Stopped
            assert frames[0].__class__.__name__ == "TTSStartedFrame"
            assert frames[-1].__class__.__name__ == "TTSStoppedFrame"
            # Middle frames should be TTSAudioRawFrame
            for frame in frames[1:-1]:
                assert frame.__class__.__name__ == "TTSAudioRawFrame"
```

#### 6.2 Integration Test

**Manual Test Script:**
```bash
cd /Users/peppi/Dev/localcat/server
source .venv/bin/activate

# Test Python service directly
python3 << 'EOF'
import asyncio
from core.tts.siri_streaming import SiriStreamingTTSService, resolve_siri_binary

async def test():
    binary = resolve_siri_binary()
    print(f"Found binary: {binary}")

    service = SiriStreamingTTSService(
        binary_path=str(binary),
        language="en-US",
        rate=0.52,
    )

    print("Generating TTS...")
    frame_count = 0
    byte_count = 0

    async for frame in service.run_tts("Hello from Siri streaming TTS"):
        frame_type = frame.__class__.__name__
        print(f"Frame {frame_count}: {frame_type}")
        if hasattr(frame, 'audio'):
            byte_count += len(frame.audio)
        frame_count += 1

    print(f"✅ Complete: {frame_count} frames, {byte_count} bytes")

asyncio.run(test())
EOF
```

#### 6.3 End-to-End Test

```bash
# 1. Set environment
export VOICE_AGENT_TTS_ENGINE=siri_streaming
export SIRI_TTS_LANGUAGE=en-US

# 2. Start server
cd /Users/peppi/Dev/localcat/server
source .venv/bin/activate
python bot.py --host 127.0.0.1 --port 7860

# 3. Connect client (in new terminal)
cd /Users/peppi/Dev/localcat/client
npm run dev

# 4. Test conversation
# Open http://localhost:3000 and speak
# Listen for Siri voice in responses
```

---

### Phase 7: Documentation (30 min)

**Update:** `README.md`

```markdown
## TTS Engine Options

LocalCat supports multiple TTS engines:

1. **Siri Streaming** (macOS only, recommended)
   - Native Siri voices
   - Zero model loading time
   - Excellent quality
   - 25+ languages supported
   - Bundle size: ~60KB

2. **Kokoro Professional** (default)
   - High quality neural TTS
   - Works on all platforms
   - ~160MB model files

3. **Kokoro MLX** (Apple Silicon)
   - In-process MLX acceleration
   - Fast on M-series chips

### Using Siri TTS

```bash
# Server environment
export VOICE_AGENT_TTS_ENGINE=siri_streaming
export SIRI_TTS_LANGUAGE=en-US  # or it-IT, es-ES, etc.
export SIRI_TTS_RATE=0.52       # 0.0-1.0
export SIRI_TTS_PITCH=1.0       # pitch multiplier

# Start server
cd server && python bot.py
```

**Supported Languages:**
- English: en-US, en-GB, en-AU, en-IN
- European: fr-FR, de-DE, es-ES, it-IT, pt-PT, nl-NL, pl-PL, ru-RU
- Asian: ja-JP, ko-KR, zh-CN, zh-HK, zh-TW
- And 10+ more (see `SIRI_VOICE_MAP` in `server/core/tts/siri_streaming.py`)
```

---

## Testing Checklist

- [ ] Swift binary compiles without warnings
- [ ] File mode produces valid WAV files
- [ ] Streaming mode outputs PCM to stdout
- [ ] Python service initializes without errors
- [ ] Binary path resolution works (dev mode)
- [ ] Binary path resolution works (production bundle)
- [ ] ServiceFactory creates Siri TTS service
- [ ] Environment variables are respected
- [ ] Multi-language synthesis works (test 3+ languages)
- [ ] Rate/pitch customization works
- [ ] Error handling triggers ONNX fallback
- [ ] Integration with Pipecat pipeline streams to WebRTC
- [ ] TTFB is < 200ms
- [ ] No memory leaks (test 100+ utterances)
- [ ] Process cleanup on cancellation
- [ ] Bundle includes binary with correct permissions

---

## Performance Benchmarks

**Expected Results:**
- Startup time: 0ms (no model loading)
- TTFB: <200ms
- Total latency: <500ms voice-to-voice
- Memory overhead: ~50MB (subprocess)
- Bundle size reduction: -160MB (no TTS models needed)

**Measurement:**
```python
import time
import asyncio
from core.tts.siri_streaming import SiriStreamingTTSService, resolve_siri_binary

async def benchmark():
    service = SiriStreamingTTSService(
        binary_path=str(resolve_siri_binary()),
        language="en-US",
    )

    start = time.time()
    first_audio = None
    total_bytes = 0

    async for frame in service.run_tts("Hello, this is a test"):
        if hasattr(frame, 'audio') and first_audio is None:
            first_audio = time.time()
        if hasattr(frame, 'audio'):
            total_bytes += len(frame.audio)

    end = time.time()

    print(f"TTFB: {(first_audio - start) * 1000:.1f}ms")
    print(f"Total: {(end - start) * 1000:.1f}ms")
    print(f"Audio: {total_bytes} bytes ({total_bytes/16000/2:.2f}s @ 16kHz)")

asyncio.run(benchmark())
```

---

## Troubleshooting

### Binary Not Found
```bash
# Verify binary exists
ls -lh /Users/peppi/Dev/localcat/app/src-tauri/sidecar/siri-tts/siri-tts

# Rebuild if needed
cd /Users/peppi/Dev/localcat/app/src-tauri/sidecar/siri-tts
./build.sh
```

### No Audio Output
```bash
# Test streaming mode manually
./siri-tts --stream-pcm --text "Test" --target-rate 16000 > /tmp/test.pcm
ls -lh /tmp/test.pcm  # Should have data

# Play the PCM file
ffplay -f s16le -ar 16000 -ac 1 /tmp/test.pcm
```

### Process Hangs
- Check stderr output from Swift process
- Verify RunLoop is processing (should work automatically in streaming mode)
- Increase timeout if synthesizing very long text

### Voice Not Available
- List available voices on your system:
  ```swift
  swift -e 'import AVFoundation; AVSpeechSynthesisVoice.speechVoices().forEach { print($0.identifier, $0.name) }'
  ```
- Update `SIRI_VOICE_MAP` with available voices

---

## Next Steps

1. Complete Phase 3: ServiceFactory integration (30 min)
2. Complete Phase 4: Configuration (20 min)
3. Complete Phase 5: Tauri bundle (45 min)
4. Complete Phase 6: Testing (60 min)
5. Complete Phase 7: Documentation (30 min)

**Total remaining time: ~3 hours**

---

## Status Summary

**✅ Complete:**
- Phase 1: Swift Streaming Sidecar (60 min)
- Phase 2: Python Pipecat Service (90 min)

**🔨 Remaining:**
- Phase 3: ServiceFactory Integration (30 min)
- Phase 4: Configuration (20 min)
- Phase 5: Tauri Bundle (45 min)
- Phase 6: Testing (60 min)
- Phase 7: Documentation (30 min)

**Total Progress: 2.5/5.5 hours (~45%)**
