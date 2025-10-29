# Pipeline Flow - Quick Reference Guide

## Complete Data Flow Paths

### Path 1: LLM Token Output
```
mlx_lm.stream_generate() [direct_mlx_llm.py:264-270]
  ↓ (async queue)
yield LLMTextFrame(text=token) [direct_mlx_llm.py:301]
```

### Path 2: Token Aggregation
```
LLMTextFrame(token) [fast_text.py input]
  ↓ (accumulate in self._aggregation)
[sentence boundary detected]
  ↓
_release_text() [fast_text.py:52-67]
  ↓
push_frame(TextFrame(sentence)) [fast_text.py:60]
```

### Path 3: TTS Processing
```
TextFrame(sentence) [kokoro_professional.py input]
  ↓ (with push_text_frames=True)
TTSTextFrame(text=sentence) [Pipecat emits]
TTSAudioRawFrame(audio_bytes) [Pipecat emits]
  ↓ (parallel paths)
  ├─→ RTVIObserver → bot-tts-text (RTVI message)
  └─→ transport.output() → PCM audio (WebRTC)
```

### Path 4: Client Reception
```
onBotTtsText callback [VoiceApp.tsx:196-244]
  ├─ Input: {text: "sentence..."}
  ├─ formatBotText() [VoiceApp.tsx:199]
  ├─ setCurrentAssistantTranscript() [VoiceApp.tsx:201]
  ├─ Detect punctuation /[.!?]$/ [VoiceApp.tsx:207]
  └─ If complete:
      ├─ Check duplicates [VoiceApp.tsx:211-223]
      ├─ Add to conversationHistory [VoiceApp.tsx:226-231]
      └─ Reset accumulator [VoiceApp.tsx:234]
```

---

## Critical Files by Stage

| Stage | File | Key Line(s) | Key Method |
|-------|------|-----------|-----------|
| **LLM** | `core/llm/direct_mlx_llm.py` | 301 | `_process_context()` yields LLMTextFrame |
| **Aggregation** | `core/aggregators/fast_text.py` | 60 | `_release_text()` pushes TextFrame |
| **TTS** | `core/tts/kokoro_professional.py` | 70 | Constructor sets `push_text_frames=True` |
| **Pipeline** | `core/factory.py` | 291, 525 | Pipeline assembly includes text_aggregator |
| **RTVI** | `core/factory.py` | 550-551 | RTVIObserver added to PipelineTask |
| **Client** | `client/src/components/VoiceApp.tsx` | 196-244 | `onBotTtsText` callback handler |
| **UI** | `client/src/components/VoiceApp.tsx` | 866-905 | Transcript panel rendering |

---

## Frame Type Emission Points

```
LLM Output:
  direct_mlx_llm.py:370-371 ─→ LLMFullResponseStartFrame
  direct_mlx_llm.py:301      ─→ LLMTextFrame(text=token)
  [Pipecat]                  ─→ LLMFullResponseEndFrame

Text Aggregation:
  fast_text.py:60            ─→ TextFrame(text=sentence)

TTS Processing:
  kokoro_professional.py:70  ─→ TTSStartedFrame
                             ─→ TTSTextFrame(text=sentence)
                             ─→ TTSAudioRawFrame(data=audio)
                             ─→ TTSStoppedFrame

Transcript:
  factory.py:428             ─→ TranscriptFrame
```

---

## Configuration Checklist

### Server-Side Must-Haves
- [ ] `push_text_frames=True` in TTS initialization (kokoro_professional.py:70)
- [ ] `aggregate_sentences=True` in TTS initialization (kokoro_professional.py:69)
- [ ] FastTextAggregator enabled in pipeline (factory.py:291, 525)
- [ ] RTVI processor added after transport.input() (factory.py:332-334, 451-453)
- [ ] RTVIObserver added to PipelineTask (factory.py:550-551)

### Client-Side Must-Haves
- [ ] `onBotTtsText` callback implemented (VoiceApp.tsx:196-244)
- [ ] Punctuation detection regex `/[.!?]$/` (VoiceApp.tsx:207)
- [ ] Duplicate prevention logic (VoiceApp.tsx:211-223)
- [ ] Conversation history state management (VoiceApp.tsx:80)
- [ ] Transcript panel rendering (VoiceApp.tsx:866-905)

---

## Key Performance Metrics

| Stage | Metric | Value | Dependency |
|-------|--------|-------|-----------|
| LLM | TTFT | 500-600ms | Model size, hardware |
| Aggregation | Latency | <50ms | Token throughput |
| TTS | Synthesis | 200-400ms | Sentence length |
| WebRTC | Transport | <100ms | Network |
| **Total** | **E2E** | **~800ms** | All stages |

---

## Debugging Checklist

### Server-Side Issues

**No TTSTextFrame in pipeline:**
- [ ] Verify `push_text_frames=True` in TTS init (kokoro_professional.py:70)
- [ ] Check RTVI processor is added (factory.py:332-334 or 451-453)
- [ ] Verify RTVIObserver is in observers list (factory.py:550-551)
- [ ] Log: Look for "📡 RTVI processor added" message

**Text not accumulating:**
- [ ] Verify FastTextAggregator in pipeline stages (factory.py:291 or 525)
- [ ] Check LLMTextFrame is being emitted (look for tokens in logs)
- [ ] Verify sentence boundary detection (debug fast_text.py:56)
- [ ] Log: Look for "[FastTextAggregator] Releasing text:" message

**TTS not emitting audio:**
- [ ] Check `push_text_frames=True` is set (kokoro_professional.py:70)
- [ ] Verify TextFrame input is reaching TTS (add debug logging)
- [ ] Check TTS service is in pipeline (factory.py:291/525)

### Client-Side Issues

**No text in transcript:**
- [ ] Check `onBotTtsText` callback is registered (VoiceApp.tsx:196)
- [ ] Verify ttsData.text exists (VoiceApp.tsx:198)
- [ ] Check formatBotText() is working (VoiceApp.tsx:199)
- [ ] Verify conversationHistory state update (VoiceApp.tsx:227-231)
- [ ] Log: Look for "🎵 onBotTtsText:" and "💾 Saving new sentence:" messages

**Text duplication in transcript:**
- [ ] Check duplicate prevention logic (VoiceApp.tsx:211-223)
- [ ] Verify regex is detecting punctuation correctly (VoiceApp.tsx:207)
- [ ] Check onBotTranscript fallback is not duplicating (VoiceApp.tsx:158-194)
- [ ] Log: Look for "🚫 Duplicate" messages

**Text not displaying:**
- [ ] Check conversationHistory rendering (VoiceApp.tsx:866-905)
- [ ] Verify unique message ID generation (VoiceApp.tsx:226)
- [ ] Check CSS/styling of transcript panel (VoiceApp.tsx:786-979)
- [ ] Verify auto-scroll is working (VoiceApp.tsx:289-294)

---

## Environment Variables & Config

### Python/Server
```bash
# Direct MLX LLM
LLM_USE_DIRECT_MLX=true
LLM_MODEL=mlx-community/Qwen3-VL-4B-Instruct-4bit

# TTS
TTS_ENGINE=kokoro
KOKORO_VOICE=af_heart

# Memory
ENABLE_MEMORY=true

# Logging
LOG_LEVEL=debug
```

### Client/.env
```bash
NEXT_PUBLIC_SERVER_URL=http://127.0.0.1:7860
```

---

## Testing Commands

### Verify LLM Output
```bash
# Check direct MLX is generating tokens
grep -r "LLMTextFrame" /Users/peppi/Dev/localcat/server/core --include="*.py"
grep "⚡ TTFT:" server_logs.txt  # Find TTFT measurement
```

### Verify Text Aggregation
```bash
# Check FastTextAggregator is releasing sentences
grep "\[FastTextAggregator\]" server_logs.txt
grep "Releasing text:" server_logs.txt
```

### Verify TTS Text Mirroring
```bash
# Check TTSTextFrame is being emitted and observed
grep "TTSTextFrame" server_logs.txt
grep "🔍 \[TRANSCRIPT.ASSISTANT INPUT\]" server_logs.txt
```

### Verify Client Reception
```bash
# Check browser console for receipt
grep "🎵 onBotTtsText:" browser_console.log
grep "🎯 Complete sentence detected:" browser_console.log
grep "💾 Saving new sentence:" browser_console.log
```

---

## Architecture Decisions

### Why Text Mirroring via TTSTextFrame?
- **Problem:** TTS audio is synthesized, but client has no text to display
- **Solution:** Pipecat's `push_text_frames=True` emits TTSTextFrame in parallel
- **Benefit:** Text sent via same pipeline, synchronized with audio
- **Alternative:** Would require separate text→client mechanism

### Why Sentence Aggregation?
- **Problem:** TTS more efficient with full sentences (better prosody)
- **Solution:** FastTextAggregator groups tokens at sentence boundaries
- **Benefit:** Natural speech output, fewer TTS calls
- **Challenge:** Must detect boundaries correctly (auxiliary verbs, abbreviations)

### Why Duplicate Prevention?
- **Problem:** Multiple event channels (onBotTtsText, onBotTranscript) can emit same text
- **Solution:** Check new text against ALL previous assistant messages
- **Methods:** Exact match, substring inclusion, reverse inclusion
- **Benefit:** Clean transcript without repetition

### Why RTVI Observer?
- **Problem:** TTSTextFrame is internal pipeline frame, client can't access directly
- **Solution:** RTVIObserver watches pipeline, converts to RTVI protocol
- **Benefit:** Uses WebRTC messaging infrastructure already in place
- **Alternative:** Would need custom message protocol

---

## Common Issues & Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| No text in transcript | `push_text_frames=True` missing | Check TTS init (kokoro_professional.py:70) |
| Text duplication | Multiple event sources | Add duplicate detection (VoiceApp.tsx:211-223) |
| Text lag | Slow aggregation | Check FastTextAggregator boundaries |
| Audio without text | RTVI not configured | Verify RTVIObserver in PipelineTask |
| Missing sentences | onBotTtsStopped not saving | Check fallback handler (VoiceApp.tsx:250-277) |

---

## References

- **Full Documentation:** `/Users/peppi/Dev/localcat/LLM_TO_CLIENT_FLOW_MAP.md`
- **LLM Service:** `/Users/peppi/Dev/localcat/server/core/llm/direct_mlx_llm.py`
- **Text Aggregation:** `/Users/peppi/Dev/localcat/server/core/aggregators/fast_text.py`
- **TTS Service:** `/Users/peppi/Dev/localcat/server/core/tts/kokoro_professional.py`
- **Pipeline Factory:** `/Users/peppi/Dev/localcat/server/core/factory.py`
- **Client UI:** `/Users/peppi/Dev/localcat/client/src/components/VoiceApp.tsx`
