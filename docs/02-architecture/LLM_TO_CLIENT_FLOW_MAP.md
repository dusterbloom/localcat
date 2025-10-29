# Complete LLM to Client UI Pipeline Flow Map

## Executive Summary

The pipeline implements a **token-to-speech-to-text-to-display** architecture:

1. **LLM** streams tokens continuously via `mlx_lm.stream_generate()`
2. **FastTextAggregator** groups tokens into natural sentences (sentence boundary detection)
3. **TTS Service** (Kokoro) synthesizes audio AND emits `TTSTextFrame` for text mirroring
4. **RTVIObserver** captures TTSTextFrame and sends as RTVI `bot-tts-text` message
5. **WebRTC Transport** delivers both audio and text to browser
6. **Client** receives text via `onBotTtsText` callback, accumulates until punctuation, saves to history
7. **UI** renders conversation history in transcript panel with duplicate prevention

**Total Latency:** ~800ms end-to-end (LLM TTFT + TTS synthesis + WebRTC transport)

---

## 1. LLM OUTPUT STAGE - Token Generation

### Location
**File:** `/Users/peppi/Dev/localcat/server/core/llm/direct_mlx_llm.py`

### Flow: Lines 172-320 (`_process_context()` async generator)

#### Step 1.1: Async Token Generation
```python
# Lines 264-270: Synchronous mlx_lm.stream_generate() in background thread
for chunk in mlx_lm.stream_generate(
    self._model,
    self._tokenizer,
    prompt=prompt,
    max_tokens=self._settings["max_tokens"],
    sampler=sampler
):
    if chunk.text:
        # Line 273: Push token to async queue (thread-safe)
        loop.call_soon_threadsafe(token_queue.put_nowait, chunk.text)
```

**Key Points:**
- Synchronous generator runs in background thread (Line 282: `loop.run_in_executor()`)
- Tokens arrive via `call_soon_threadsafe()` to maintain async-safety
- Line 273: Tokens pushed immediately without waiting for full response

#### Step 1.2: Frame Emission - LLMTextFrame
```python
# Lines 284-301: Stream tokens as they arrive
while True:
    token = await token_queue.get()
    if token is None:
        break
    if isinstance(token, tuple) and token[0] == "ERROR":
        raise Exception(token[1])
    
    # Line 298-299: Measure TTFT (Time To First Token)
    if first_token_time is None:
        first_token_time = (time.time() - start_time) * 1000
        logger.debug(f"⚡ TTFT: {first_token_time:.1f}ms (Direct MLX)")
    
    # Line 301: CRITICAL - Emit frame for each token
    yield LLMTextFrame(text=token)
```

**Frames Emitted:**
- `LLMFullResponseStartFrame` (Line 371 in `process_frame()`)
- `LLMTextFrame(text=token)` - Line 301 (streaming tokens)
- `LLMFullResponseEndFrame` (framework-emitted)

**Performance:**
- TTFT: ~500-600ms
- Tokens: Continuous streaming (no batching)

---

## 2. TEXT AGGREGATION STAGE - Sentence Assembly

### Location
**File:** `/Users/peppi/Dev/localcat/server/core/aggregators/fast_text.py`

### Flow: Token Accumulation → Sentence Release

#### Step 2.1: Token Buffering
```python
# Lines 143-191: process_frame() - Main processing loop
async def process_frame(self, frame: Frame, direction: FrameDirection):
    await super().process_frame(frame, direction)
    
    if isinstance(frame, InterimTranscriptionFrame):
        return
    
    # Handle interruptions
    if isinstance(frame, (CancelFrame, InterruptionFrame, BotInterruptionFrame)):
        # Reset on user interruption
        self._aggregation = ""
        ...
    
    # For LLMTextFrame: append token to accumulator
    if isinstance(frame, LLMTextFrame):
        self._aggregation += frame.text
        # Check if we've hit a sentence boundary...
```

**Accumulation Logic:**
- Tokens stored in `self._aggregation` (instance variable)
- Continues until sentence boundary detected

#### Step 2.2: Sentence Boundary Detection
```python
# Lines 27-50: Sentence delimiters and auxiliary verbs
self._sentence_endings = {'.', '!', '?', '。', '！', '？'}
self._clause_endings = {',', ';', ':', '，', '；', '：'}
self._auxiliary_verbs = {
    'have', 'has', 'had', 'having',
    'will', 'would', 'shall', 'should', ...
}
```

**Detection Algorithm:**
1. Check for sentence-ending punctuation (`.!?`)
2. Validate safe break point (not after auxiliary verb)
3. Ensure minimum word count before release

#### Step 2.3: Text Release
```python
# Lines 52-67: _release_text() - Release aggregated sentence
async def _release_text(self):
    if self._aggregation.strip():
        # Line 56: Clean text
        clean_text = self._clean_text_for_tts(self._aggregation)
        if clean_text:
            logger.debug(f"[FastTextAggregator] Releasing text: '{clean_text[:100]}...'")
            
            # Line 60: CRITICAL - Push TextFrame downstream
            await self.push_frame(TextFrame(clean_text))
    
    # Line 62: Reset for next sentence
    self._aggregation = ""
    self._last_release_time = asyncio.get_event_loop().time()
```

**Text Cleaning (Lines 69-105):**
- Remove markdown: `**bold**`, `*italic*`, `\`code\``
- Remove emojis and problematic characters
- Normalize ellipses to single period
- Remove leading meta hints (e.g., "(Calculating)")
- Join digits separated by spaces

**Frame Output:**
- `TextFrame(text=cleaned_sentence)` - Ready for TTS

### Pipeline Integration
**File:** `/Users/peppi/Dev/localcat/server/core/factory.py`
- **Line 291** (Intro pipeline): `services['text_aggregator']`
- **Line 525** (Standard pipeline): `services['text_aggregator']`

---

## 3. TTS PROCESSING STAGE - Audio + Text Generation

### Location
**File:** `/Users/peppi/Dev/localcat/server/core/tts/kokoro_professional.py`

### Initialization (Lines 51-98)

```python
class ProfessionalKokoroTTSService(TTSService):
    def __init__(self, ..., aggregate_sentences: bool = True, ...):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=aggregate_sentences,  # Line 69 - Enable sentence aggregation
            push_text_frames=True,  # Line 70 - CRITICAL for text mirroring!
            **kwargs
        )
        
        logger.info(f"🔧 [ProfessionalKokoroTTS] Initializing with "
                   f"aggregate_sentences={aggregate_sentences}, "
                   f"push_text_frames=True")
```

**Critical Configuration:**
- Line 69: `aggregate_sentences=True` - Groups text internally
- Line 70: **`push_text_frames=True`** - Tells Pipecat to emit TTSTextFrame in addition to audio
- Comment (Lines 70-71): "Let Pipecat emit TTSTextFrame for RTVI bot-tts-text messages"

### Frame Emission

When `push_text_frames=True`, Kokoro TTS emits:

1. **TTSStartedFrame** - TTS processing begins
2. **TTSTextFrame(text=sentence)** - Text being synthesized (handled by Pipecat internally)
3. **TTSAudioRawFrame(data=audio_bytes)** - PCM audio chunks (24kHz, 16-bit)
4. **TTSStoppedFrame** - TTS processing complete

### Alternative TTS Implementations (All Emit TTSTextFrame)

| File | Lines | Implementation |
|------|-------|-----------------|
| `kokoro_mlx.py` | 290-291 | `yield TTSTextFrame(text=sentence)` |
| `kokoro_pytorch.py` | 322-323 | `yield TTSTextFrame(text=sentence)` |
| `siri_streaming.py` | 179 | `yield TTSTextFrame(text=text)` |
| `tts_mlx_ultra_low_latency.py` | 249-250 | `yield TTSTextFrame(text=chunk_text)` |

**All emit TTSTextFrame when `push_text_frames=True`**

---

## 4. PIPELINE ROUTING & RTVI OBSERVER

### Location
**File:** `/Users/peppi/Dev/localcat/server/core/factory.py`

### Pipeline Stage Ordering

#### Intro-Aware Pipeline (Lines 176-435)
```python
# Line 328: Build main pipeline stages
stages = [transport.input()]

# Lines 332-334: RTVI processor added early
if services.get('rtvi'):
    stages.append(services['rtvi'])
    logger.debug("📡 RTVI processor added after transport.input() [intro-aware]")
```

#### Standard Pipeline (Lines 437-535)
```python
# Line 447: Start with transport input
stages = [transport.input()]

# Lines 451-453: RTVI processor right after input
if services.get('rtvi'):
    stages.append(services['rtvi'])
    logger.debug("📡 RTVI processor added after transport.input()")
```

**Comment (Lines 449-450):**
> "Add RTVI processor right after transport input (per Pipecat docs)"
> "This allows OutputTransportMessageUrgentFrame to flow downstream to transport.output()"

### RTVIObserver Integration (Lines 537-563)

```python
# Lines 550-551: Create observer
observers = []
if rtvi_processor:
    observers.append(RTVIObserver(rtvi_processor))

# Lines 557-561: Create task with observers
task = PipelineTask(
    pipeline,
    params=params,
    observers=observers
)
```

**What RTVIObserver Does:**
1. Listens to ALL frames flowing through pipeline
2. Detects TTSTextFrame frames
3. Converts to RTVI `bot-tts-text` message
4. Sends via WebRTC to client in real-time

### Audio Output (Lines 414 & 528)

```python
# Intro-aware pipeline (Line 414)
stages.append(transport.output())

# Standard pipeline (Line 528)
stages.append(transport.output())
```

**What transport.output() does:**
- Extracts TTSAudioRawFrame
- Streams PCM audio chunks via WebRTC
- Parallel to text via RTVI message

---

## 5. CONTEXT & TRANSCRIPT AGGREGATION

### Location
**File:** `/Users/peppi/Dev/localcat/server/core/factory.py`

### Assistant Context Aggregation

#### Intro-Aware Pipeline (Lines 418-428)
```python
# Line 418: Assistant aggregator AFTER transport output
stages.append(services['context_aggregator'].assistant())

# Lines 420-428: Debug filter for TTSTextFrame
async def _debug_transcript_assistant(frame) -> bool:
    from pipecat.frames.frames import TTSTextFrame
    if isinstance(frame, TTSTextFrame):
        logger.info(f"🔍 [TRANSCRIPT.ASSISTANT INPUT] TTSTextFrame: '{frame.text[:100]}...'")
    return True

stages.append(_FF(_debug_transcript_assistant))
stages.append(transcript.assistant())
```

#### Standard Pipeline (Lines 529-531)
```python
stages.extend([
    transport.output(),
    transcript.assistant(),
    services['context_aggregator'].assistant()
])
```

**What Happens:**
1. TTSTextFrame passes through transport.output() first (audio extraction)
2. OpenAIAssistantContextAggregator receives TTSTextFrame
3. Extracts `.text` property
4. Adds to context: `{"role": "assistant", "content": text}`
5. TranscriptProcessor logs to conversation history

**Frame Flow:**
```
TTSTextFrame → transport.output() → context_aggregator.assistant()
                    ↓                           ↓
            (PCM audio sent)         (text added to context)
```

### Transcript Processor Setup

```python
# Lines 376-390 (Intro-aware) / 475-489 (Standard)
transcript = TranscriptProcessor()

@transcript.event_handler("on_transcript_update")
async def log_conversation(processor, frame):
    logger.info(f"[Pipeline] Transcript update: {len(frame.messages)} messages")
    for i, message in enumerate(frame.messages):
        role = message.role.upper()
        content = message.content
        display_content = content[:150] + "..." if len(content) > 150 else content
        logger.info(f"📝 CONVERSATION [{role}]: {display_content}")
```

---

## 6. CLIENT UI RECEPTION & DISPLAY

### Location
**File:** `/Users/peppi/Dev/localcat/client/src/components/VoiceApp.tsx`

### PipecatClient Initialization (Lines 96-286)

```typescript
// Lines 97-283: Initialize PipecatClient
const initClient = async () => {
    const transport = new SmallWebRTCTransport();
    const pcClient = new PipecatClient({
        enableCam: false,
        enableMic: true,
        transport: transport,
        callbacks: {
            // ... callback handlers defined here (see 6.2)
        }
    });
    
    await pcClient.initDevices();
    setClient(pcClient);
};
```

### 6.1 Text Formatting Helper (Lines 52-67)

```typescript
const formatBotText = (text: string): string => {
    if (!text) return text;
    
    // Remove **bold** formatting
    let formatted = text.replace(/\*\*(.*?)\*\*/g, '$1');
    
    // Remove *italic* formatting
    formatted = formatted.replace(/\*(.*?)\*/g, '$1');
    
    // Remove `code` formatting
    formatted = formatted.replace(/`([^`]+)`/g, '$1');
    
    // Convert bullet points
    formatted = formatted.replace(/^\s*[-*+]\s+/gm, '• ');
    
    return formatted;
};
```

**Used by:** Both `onBotTtsText` and `onBotTranscript` handlers

### 6.2 Primary Handler: onBotTtsText (Lines 196-244)

```typescript
onBotTtsText: (ttsData) => {
    // Line 197: Log received TTS text
    console.log('🎵 onBotTtsText:', ttsData);
    
    // Line 198: Validation
    if (ttsData && ttsData.text && ttsData.text.trim().length > 0) {
        // Line 199: Clean formatting
        const newText = formatBotText(ttsData.text);
        
        // Line 201: Accumulate text
        setCurrentAssistantTranscript(prevText => {
            if (!prevText || newText.length > prevText.length) {
                console.log('📈 TTS text growing:', prevText?.length || 0, '→', newText.length);
                
                // Line 207: Check for sentence completion
                const endsWithPunctuation = /[.!?]$/.test(newText.trim());
                
                if (endsWithPunctuation) {
                    console.log('🎯 Complete sentence detected:', newText);
                    
                    // Lines 211-223: Duplicate prevention
                    setConversationHistory(prev => {
                        const allAssistantMessages = prev.filter(msg => msg.role === 'assistant');
                        
                        // Check ALL previous assistant messages for duplicates
                        for (const msg of allAssistantMessages) {
                            if (msg.text === newText ||
                                msg.text.includes(newText) ||
                                newText.includes(msg.text)) {
                                console.log('🚫 Duplicate/partial sentence found, skipping:', newText);
                                return prev;  // Don't add duplicate
                            }
                        }
                        
                        // Line 226: Create unique ID
                        const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                        
                        // Line 227-231: Add unique sentence to history
                        console.log('💾 Saving new sentence:', newText);
                        return [
                            ...prev,
                            { role: 'assistant', text: newText, id: messageId }
                        ];
                    });
                    
                    // Line 234: Return empty to start fresh
                    return '';
                }
                
                // Line 237: Accumulate if no punctuation yet
                return newText;
            }
            
            // Line 241: Keep longer existing text (shrinking case)
            console.log('🚫 TTS text shrinking, keeping existing:', prevText.length, 'vs', newText.length);
            return prevText;
        });
    }
},
```

**Key Logic:**
1. **Line 199:** Format text (remove markdown)
2. **Line 201:** Accumulate via `setCurrentAssistantTranscript()`
3. **Line 207:** Detect completion with regex `/[.!?]$/`
4. **Lines 211-223:** Check for duplicates (exact match, substring, reverse)
5. **Lines 226-231:** If unique, add to conversation history
6. **Line 234:** Reset accumulator for next sentence

### 6.3 Fallback Handler: onBotTranscript (Lines 158-194)

```typescript
onBotTranscript: (transcript) => {
    // Line 159: Comment about fallback
    // Fallback: some TTS services emit assistant text via transcript events
    
    try {
        const raw = transcript?.text || '';
        if (!raw.trim()) return;
        
        // Line 164: Same formatting
        const newText = formatBotText(raw);
        
        // Line 166-190: Identical state update logic
        setCurrentAssistantTranscript(prevText => {
            if (!prevText || newText.length > prevText.length) {
                const endsWithPunctuation = /[.!?]$/.test(newText.trim());
                if (endsWithPunctuation) {
                    setConversationHistory(prev => {
                        // ... duplicate check and add to history
                    });
                    return '';
                }
                return newText;
            }
            return prevText;
        });
    } catch (e) {
        console.warn('onBotTranscript fallback handling failed:', e);
    }
},
```

**Purpose:** Handles services that emit text via transcript channel instead of TTS text channel

### 6.4 TTS Lifecycle Handlers

#### onBotTtsStarted (Lines 245-249)
```typescript
onBotTtsStarted: () => {
    console.log('Bot TTS started - keep accumulating text');
    // Don't clear! Let text accumulate across TTS sessions
    setShowTranscript(true);  // Auto-open transcript when bot speaks
},
```

#### onBotTtsStopped (Lines 250-277)
```typescript
onBotTtsStopped: () => {
    console.log('Bot TTS stopped - check for any remaining text');
    
    // Fallback: save any remaining text without punctuation
    setCurrentAssistantTranscript(prevText => {
        if (prevText.trim()) {
            console.log('💾 Checking to save remaining text without punctuation:', prevText);
            
            setConversationHistory(prev => {
                const allAssistantMessages = prev.filter(msg => msg.role === 'assistant');
                
                for (const msg of allAssistantMessages) {
                    if (msg.text === prevText ||
                        msg.text.includes(prevText) ||
                        prevText.includes(msg.text)) {
                        console.log('🚫 Duplicate remaining text found, skipping:', prevText);
                        return prev;
                    }
                }
                
                // Save remaining text
                const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                console.log('✅ Saving remaining text:', prevText);
                return [...prev, { role: 'assistant', text: prevText, id: messageId }];
            });
        }
        return '';  // Clear for next response
    });
},
```

**Purpose:** Catches text that doesn't end with punctuation (prevents loss)

### 6.5 Transcript Panel Rendering (Lines 786-979)

#### State Definitions (Lines 80-82)
```typescript
const [conversationHistory, setConversationHistory] = useState<Array<{
    role: 'user' | 'assistant',
    text: string,
    isStreaming?: boolean,
    id: string
}>>([]);

const [currentAssistantTranscript, setCurrentAssistantTranscript] = useState('');
```

#### Rendering Loop (Lines 866-905)
```typescript
{conversationHistory.length > 0 ? (
    <div className="space-y-3">
        {conversationHistory.map((message, idx) => (
            <div key={message.id || idx} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-lg p-3 ${
                    message.role === 'user'
                        ? 'bg-black text-white'
                        : (isDarkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-200 text-gray-900')
                }`}>
                    {message.role === 'assistant' && message.isStreaming ? (
                        <StreamingText text={message.text} speed={30} />
                    ) : (
                        <div dangerouslySetInnerHTML={{...}} />
                    )}
                </div>
            </div>
        ))}
        
        {/* Streaming assistant text (in-progress) */}
        {currentAssistantTranscript && (
            <div className="flex justify-start mb-3">
                <StreamingText text={currentAssistantTranscript} speed={30} />
            </div>
        )}
    </div>
) : null}
```

**Display Logic:**
1. Map through `conversationHistory` (completed messages)
2. Render `currentAssistantTranscript` as in-progress (below history)
3. Auto-scroll to bottom (Lines 289-294)

---

## Complete End-to-End Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. LLM OUTPUT                          (server/core/llm/direct_mlx_llm.py)
├────────────────────────────────────────────────────────────────────────┤
│ mlx_lm.stream_generate() runs in background thread                     │
│ Line 273: Each token pushed to async queue                             │
│ Line 301: yield LLMTextFrame(text=token)                              │
│ Output: LLMFullResponseStartFrame, LLMTextFrame(s), LLMFullResponseEnd │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│ 2. TEXT AGGREGATION                    (server/core/aggregators/fast_text.py)
├────────────────────────────────────────────────────────────────────────┤
│ Line 143: process_frame() accumulates tokens in self._aggregation      │
│ Line 56: _clean_text_for_tts() cleans text                           │
│ Line 60: await self.push_frame(TextFrame(clean_text))                │
│ Output: TextFrame (complete sentences)                                 │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│ 3. TTS SYNTHESIS                       (server/core/tts/kokoro_professional.py)
├────────────────────────────────────────────────────────────────────────┤
│ Line 70: push_text_frames=True enables TTSTextFrame emission           │
│ Emits: TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame(s), TTSStoppedFrame
│ Output: Both TTSTextFrame (for client text) and TTSAudioRawFrame (audio)
└────────────────────────────────────────────────────────────────────────┘
                    ↓                                      ↓
        ┌──────────────────┐                    ┌──────────────────┐
        │ RTVI OBSERVER    │                    │ AUDIO OUTPUT     │
        │ (factory.py:550) │                    │ (factory.py:414) │
        │ TTSTextFrame →   │                    │ TTSAudioRawFrame │
        │ bot-tts-text msg │                    │ → PCM chunks     │
        └──────────────────┘                    └──────────────────┘
                    ↓                                      ↓
        ┌──────────────────┐                    ┌──────────────────┐
        │ RTVI MESSAGE     │                    │ WEBRTC TRANSPORT │
        │ (JSON over WS)   │                    │ (Audio stream)   │
        │ bot-tts-text     │                    │ PCM 24kHz        │
        └──────────────────┘                    └──────────────────┘
                    ↓                                      ↓
        ┌──────────────────────────────────────────────────────────┐
        │           BROWSER (PipecatClient)                        │
        └──────────────────────────────────────────────────────────┘
                    ↓
        ┌──────────────────────────────────────────────────────────┐
        │ 4. CLIENT TEXT RECEPTION (VoiceApp.tsx:196-244)          │
        ├──────────────────────────────────────────────────────────┤
        │ onBotTtsText(ttsData) callback                           │
        │ Line 199: formatBotText() → remove markdown              │
        │ Line 201: setCurrentAssistantTranscript() → accumulate   │
        │ Line 207: /[.!?]$/ → detect completion                   │
        │ Lines 211-223: Check duplicates in conversationHistory   │
        │ Line 226: Create unique ID                              │
        │ Lines 227-231: Add to conversationHistory if unique      │
        │ Output: conversationHistory updated                       │
        └──────────────────────────────────────────────────────────┘
                    ↓
        ┌──────────────────────────────────────────────────────────┐
        │ 5. UI RENDERING (VoiceApp.tsx:866-979)                   │
        ├──────────────────────────────────────────────────────────┤
        │ Line 868: Map through conversationHistory array           │
        │ Line 869: Render each message with unique key={message.id}
        │ Line 933: Display currentAssistantTranscript (streaming)  │
        │ Line 289: Auto-scroll to bottom                           │
        │ Output: Transcript panel with live conversation           │
        └──────────────────────────────────────────────────────────┘
```

---

## Frame Type Reference

| Frame Type | Source | Purpose | Lines |
|-----------|--------|---------|-------|
| `LLMFullResponseStartFrame` | LLM | Mark response start | direct_mlx_llm.py:371 |
| `LLMTextFrame` | LLM | Individual tokens | direct_mlx_llm.py:301 |
| `LLMFullResponseEndFrame` | Pipecat | Mark response end | (framework) |
| `TextFrame` | FastTextAggregator | Aggregated sentences | fast_text.py:60 |
| `TTSStartedFrame` | TTS | TTS begins | (Pipecat base) |
| `TTSTextFrame` | TTS (push_text_frames=True) | Text mirror for client | kokoro_professional.py:70 |
| `TTSAudioRawFrame` | TTS | PCM audio | (Pipecat base) |
| `TTSStoppedFrame` | TTS | TTS ends | (Pipecat base) |
| `TranscriptFrame` | TranscriptProcessor | Conversation logging | factory.py:428 |

---

## Key Configuration Parameters

### Server-Side

| Parameter | File | Line | Value | Purpose |
|-----------|------|------|-------|---------|
| `aggregate_sentences` | kokoro_professional.py | 69 | `True` | Enable internal sentence aggregation |
| `push_text_frames` | kokoro_professional.py | 70 | `True` | **CRITICAL** - Emit TTSTextFrame for text mirroring |
| `enable_direct_mode` | fast_text.py | 30 | `True` | Direct frame processing without queue |
| `min_tokens` | fast_text.py | 27 | `10` | Minimum tokens before release |
| `max_tokens` | fast_text.py | 27 | `250` | Maximum tokens per sentence |

### Client-Side

| State | File | Line | Type | Purpose |
|-------|------|------|------|---------|
| `conversationHistory` | VoiceApp.tsx | 80 | Array | Completed messages |
| `currentAssistantTranscript` | VoiceApp.tsx | 82 | String | In-progress streaming text |
| `showTranscript` | VoiceApp.tsx | 77 | Boolean | Transcript panel visibility |

---

## Error Handling & Edge Cases

### Duplicate Prevention (VoiceApp.tsx:211-223)
```typescript
// Check ALL previous assistant messages for duplicates
const allAssistantMessages = prev.filter(msg => msg.role === 'assistant');
for (const msg of allAssistantMessages) {
    if (msg.text === newText ||           // Exact match
        msg.text.includes(newText) ||     // Substring of previous
        newText.includes(msg.text)) {     // Previous is substring
        return prev;  // Skip duplicate
    }
}
```

**Scenarios Handled:**
1. Exact duplicate: Same text sent twice
2. Partial: "Hello world" sent, then "Hello world." 
3. Expansion: "Hello" sent, then "Hello world"

### Text Without Punctuation (VoiceApp.tsx:250-277)
```typescript
// onBotTtsStopped callback
// Saves remaining text that didn't end with .!?
// Prevents loss of final utterances
if (prevText.trim()) {
    setConversationHistory(prev => {
        // Check for duplicates...
        // Add remaining text
    });
}
```

### Text Shrinking Detection (VoiceApp.tsx:239-241)
```typescript
// If new text is shorter than previous, keep previous
if (!prevText || newText.length > prevText.length) {
    // Update to new text
} else {
    // Keep existing (longer) text
    console.log('🚫 TTS text shrinking, keeping existing');
    return prevText;
}
```

---

## Performance Characteristics

| Metric | Value | Source |
|--------|-------|--------|
| LLM TTFT | 500-600ms | direct_mlx_llm.py:299 |
| Text Aggregation | <50ms | fast_text.py processing |
| TTS Synthesis | 200-400ms/sentence | TTS service dependent |
| WebRTC Transport | <100ms | Network dependent |
| **End-to-End** | **~800ms** | TTFT + TTS + transport |

---

## Debugging & Logging

### Console Logs (Browser)
```javascript
// Line 106: Bot started speaking
'🎙️ Bot started speaking'

// Line 197: TTS text received
'🎵 onBotTtsText:', ttsData

// Line 204: Text accumulation
'📈 TTS text growing:', prevText?.length || 0, '→', newText.length

// Line 209: Complete sentence
'🎯 Complete sentence detected:', newText

// Line 220: Duplicate found
'🚫 Duplicate/partial sentence found, skipping:', newText

// Line 227: Save new sentence
'💾 Saving new sentence:', newText

// Line 241: Text shrinking
'🚫 TTS text shrinking, keeping existing:', prevText.length, 'vs', newText.length
```

### Server Logs
```python
# direct_mlx_llm.py:240
'🧠 LLM generating (Direct MLX, model=...)'

# direct_mlx_llm.py:299
'⚡ TTFT: XXXms (Direct MLX)'

# fast_text.py:59
'[FastTextAggregator] Releasing text: ...'

# kokoro_professional.py:65
'🔧 [ProfessionalKokoroTTS] Initializing with aggregate_sentences=True, push_text_frames=True'

# factory.py:332
'📡 RTVI processor added after transport.input()'

# factory.py:424
'🔍 [TRANSCRIPT.ASSISTANT INPUT] TTSTextFrame: ...'
```

---

## Summary

The complete pipeline implements a sophisticated **streaming text-to-speech-to-display** system:

1. **Server streams tokens** from LLM continuously
2. **Aggregates into sentences** at natural boundaries
3. **TTS synthesizes audio** while mirroring text via TTSTextFrame
4. **RTVI observer** captures TTSTextFrame as bot-tts-text message
5. **Client receives** both audio (WebRTC PCM) and text (RTVI JSON)
6. **UI accumulates** streaming text until punctuation
7. **Deduplicates** against full conversation history
8. **Displays** real-time transcript with fallback for incomplete sentences

**Key Innovation:** Using Pipecat's `push_text_frames=True` to achieve synchronized text mirroring without separate message passing, reducing latency and complexity.
