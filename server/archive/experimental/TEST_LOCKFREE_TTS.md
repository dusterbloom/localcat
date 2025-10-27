# Lock-Free TTS Test Guide

Testing lock-free TTS generation to match offline-voice-ai architecture.

## Architecture Comparison

### offline-voice-ai (Reference)
```
MLX Lock:
  ├─ STT (Whisper MLX) ─────┐
  └─ LLM (MLX-LM)           ├─ LOCKED
                            ┘
TTS (Kokoro PyTorch) ──────── LOCK-FREE ⚡
```

### localcat Standard (kokoro_pytorch)
```
MLX Lock (initialization only):
  ├─ STT (Parakeet MLX) ────┐
  └─ LLM (OpenAI API)       ├─ LOCKED on init
                            ┘
TTS threading.Lock:
  └─ TTS Generation ────────── LOCKED (threading.Lock)
```

### localcat Lock-Free Test (kokoro_pytorch_lockfree)
```
MLX Lock (initialization only):
  ├─ STT (Parakeet MLX) ────┐
  └─ LLM (OpenAI API)       ├─ LOCKED on init
                            ┘
TTS (Kokoro PyTorch) ──────── LOCK-FREE ⚡ (like offline-voice-ai!)
```

## Test Setup

### 1. Standard Mode (Current)
Edit `.env`:
```bash
VOICE_AGENT_TTS_ENGINE=kokoro_pytorch
# Uses threading.Lock during generation
# max_workers=2
```

### 2. Lock-Free Mode (Test)
Edit `.env`:
```bash
VOICE_AGENT_TTS_ENGINE=kokoro_pytorch_lockfree
TTS_MAX_WORKERS=4  # Optional: increase parallelism (default: 4)
```

## Expected Results

### Standard Mode
- ✅ Safe: Lock prevents concurrent access issues
- ⚠️  Serialized: Only one TTS generation at a time
- ⚠️  Limited workers: 2 threads
- 📊 Latency: Moderate

### Lock-Free Mode
- ⚡ Faster: Multiple TTS chunks can generate in parallel
- ⚡ Better throughput: 4 workers by default
- ⚡ Lower TTFB: No lock contention
- ⚠️  Risk: PyTorch must handle concurrent access safely

## Performance Metrics to Watch

Look for these in logs:

### Standard Mode
```
[DEBUG] Kokoro PyTorch generated 50 chars in 0.234s
🚀 Kokoro PyTorch TTFB: 245.2ms
```

### Lock-Free Mode
```
[DEBUG] 🎯 LOCK-FREE generated 50 chars in 0.189s
🚀 Kokoro PyTorch LOCK-FREE TTFB: 198.5ms
```

## Testing Procedure

1. **Baseline Test (Standard)**
   ```bash
   # In .env
   VOICE_AGENT_TTS_ENGINE=kokoro_pytorch

   # Restart server
   cd /Users/peppi/Dev/localcat/server
   source .venv/bin/activate
   python bot.py

   # Test with client and note:
   # - TTFB (Time To First Byte)
   # - Total response time
   # - CPU usage
   ```

2. **Lock-Free Test**
   ```bash
   # In .env
   VOICE_AGENT_TTS_ENGINE=kokoro_pytorch_lockfree
   TTS_MAX_WORKERS=4

   # Restart server
   python bot.py

   # Test with same queries and compare metrics
   ```

3. **Stress Test (Optional)**
   ```bash
   # Lock-Free with more workers
   TTS_MAX_WORKERS=8

   # Test with longer responses to see parallel benefit
   ```

## Safety Considerations

### Why Lock-Free Might Be Safe
1. **PyTorch is Thread-Safe**: Modern PyTorch handles concurrent inference well
2. **No MLX Conflict**: PyTorch and MLX use separate Metal heaps
3. **offline-voice-ai Proof**: Works reliably in minirepo

### Why Lock-Free Might Fail
1. **macOS Sequoia Bug**: Concurrent Metal access sometimes triggers kills
2. **Memory Pressure**: Multiple TTS workers increase memory usage
3. **Model Not Thread-Safe**: Kokoro pipeline might have internal state

## Rollback

If lock-free mode causes crashes:
```bash
# Revert to standard mode
VOICE_AGENT_TTS_ENGINE=kokoro_pytorch
```

## Results

**Document your findings here:**

### Test 1: Standard Mode
- TTFB: ___ ms
- Response Time: ___ s
- CPU Usage: ___%
- Stability: ___

### Test 2: Lock-Free Mode (4 workers)
- TTFB: ___ ms
- Response Time: ___ s
- CPU Usage: ___%
- Stability: ___

### Test 3: Lock-Free Mode (8 workers)
- TTFB: ___ ms
- Response Time: ___ s
- CPU Usage: ___%
- Stability: ___

### Conclusion
- [ ] Lock-free is faster
- [ ] Lock-free is stable
- [ ] Recommend lock-free as default
- [ ] Keep standard mode as default (safer)

## Implementation Notes

Key code differences:

**Standard (`kokoro_pytorch.py`)**:
```python
def __init__(self):
    self._generation_lock = threading.Lock()  # ← Lock exists

def _generate_audio_sync(self, text: str):
    with self._generation_lock:  # ← Locked
        audio = self._pipeline(text)
```

**Lock-Free (`kokoro_pytorch_lockfree.py`)**:
```python
def __init__(self):
    # NO _generation_lock  # ← No lock!

def _generate_audio_sync(self, text: str):
    # NO lock here  # ← Lock-free
    audio = self._pipeline(text)
```
