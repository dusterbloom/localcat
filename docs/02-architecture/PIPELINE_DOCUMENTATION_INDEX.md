# Pipeline Flow Documentation - Complete Index

## Overview
This documentation maps the complete data flow from LLM output through TTS processing to client UI display, including all frame types, processors, and intermediate transformations.

---

## Documentation Files

### 1. **LLM_TO_CLIENT_FLOW_MAP.md** (Primary - 32KB)
**Purpose:** Complete technical reference with line-by-line code analysis

**Sections:**
- Executive Summary (7-step pipeline overview)
- Stage 1: LLM Output (Lines 19-57)
  - Lines 172-320 of direct_mlx_llm.py
  - Token generation, frame emission, TTFT measurement
- Stage 2: Text Aggregation (Lines 59-91)
  - Lines 20-191 of fast_text.py
  - Token accumulation, sentence boundary detection, text release
- Stage 3: TTS Processing (Lines 93-128)
  - Lines 51-98 of kokoro_professional.py
  - Critical push_text_frames=True configuration
  - All TTS implementations that emit TTSTextFrame
- Stage 4: RTVI Observer (Lines 130-168)
  - Lines 537-563 of factory.py
  - Pipeline routing, RTVIObserver integration
  - Frame conversion to bot-tts-text RTVI message
- Stage 5: Context & Transcript (Lines 170-211)
  - Lines 418-428 of factory.py
  - Assistant context aggregation, transcript processor setup
- Stage 6: Client UI (Lines 213-347)
  - Lines 96-979 of VoiceApp.tsx
  - Text formatting, callback handlers, transcript rendering
- Complete Data Flow Diagram (ASCII)
- Frame Type Summary Table
- Configuration Keys
- Error Handling & Edge Cases
- Performance Characteristics
- Debugging & Logging Guide
- Full Summary

**Best For:** Understanding the complete architecture, debugging complex issues, code review

**File:** `/Users/peppi/Dev/localcat/LLM_TO_CLIENT_FLOW_MAP.md`

---

### 2. **PIPELINE_QUICK_REFERENCE.md** (Quick Lookup - 9KB)
**Purpose:** Fast reference for developers and debugging

**Sections:**
- Complete Data Flow Paths (4 paths visualized)
- Critical Files by Stage (table)
- Frame Type Emission Points
- Configuration Checklist (server-side, client-side)
- Key Performance Metrics
- Debugging Checklist (3 sections)
- Environment Variables & Config
- Testing Commands
- Architecture Decisions (4 explained)
- Common Issues & Solutions (table)
- References

**Best For:** Quick lookup during development, debugging reference, architecture review

**File:** `/Users/peppi/Dev/localcat/PIPELINE_QUICK_REFERENCE.md`

---

### 3. **PIPELINE_FLOW_SUMMARY.txt** (Text Format - 17KB)
**Purpose:** Comprehensive text-only reference with line number references

**Sections:**
- Document Location (3 files listed)
- Stage 1: LLM Output (Process, frames, performance)
- Stage 2: Text Aggregation (Methods, key points, integration)
- Stage 3: TTS Processing (Config, frames, alternatives)
- Stage 4: Pipeline Routing (RTVI placement, TTSTextFrame→message)
- Stage 5: Context & Transcript (Aggregation, frame flow)
- Stage 6: Client UI (Initialization, formatting, handlers)
- Complete End-to-End Flow (11-step process)
- Frame Type Summary Table
- Critical Configuration Parameters
- Performance Metrics Table
- Key Design Decisions (5 explained)
- Debugging & Logging (Console and server logs)
- Common Issues & Solutions
- Files Referenced (organized by stack)

**Best For:** Offline reference, text-only viewing, printing, copy-paste reference

**File:** `/Users/peppi/Dev/localcat/PIPELINE_FLOW_SUMMARY.txt`

---

### 4. **PIPELINE_DOCUMENTATION_INDEX.md** (This File)
**Purpose:** Navigation and guidance for using all documentation

**Best For:** Finding the right document for your needs

**File:** `/Users/peppi/Dev/localcat/PIPELINE_DOCUMENTATION_INDEX.md`

---

## Quick Navigation

### I need to...

#### Understand the complete architecture
→ Read **LLM_TO_CLIENT_FLOW_MAP.md** (Sections 1-6)

#### Debug missing text in transcript
→ Use **PIPELINE_QUICK_REFERENCE.md** (Debugging Checklist > Client-Side Issues)
→ Cross-reference **PIPELINE_FLOW_SUMMARY.txt** (Common Issues & Solutions)

#### Find a specific file location
→ Check **PIPELINE_QUICK_REFERENCE.md** (Critical Files by Stage table)

#### Verify configuration
→ Use **PIPELINE_QUICK_REFERENCE.md** (Configuration Checklist)

#### Trace a frame type through the pipeline
→ Check **PIPELINE_FLOW_SUMMARY.txt** (Frame Type Summary) or
→ **LLM_TO_CLIENT_FLOW_MAP.md** (Frame Type Reference)

#### Find line numbers in source code
→ Use **LLM_TO_CLIENT_FLOW_MAP.md** (all sections have line references)
→ Or **PIPELINE_FLOW_SUMMARY.txt** (comprehensive line references)

#### Test/verify the pipeline is working
→ Use **PIPELINE_QUICK_REFERENCE.md** (Testing Commands section)

#### Understand architecture decisions
→ Read **PIPELINE_QUICK_REFERENCE.md** (Architecture Decisions)
→ Or **PIPELINE_FLOW_SUMMARY.txt** (Key Design Decisions)

#### Understand performance bottlenecks
→ Check **PIPELINE_FLOW_SUMMARY.txt** (Performance Metrics)
→ Or **LLM_TO_CLIENT_FLOW_MAP.md** (Performance Optimizations)

---

## Key Concepts

### The Core Pipeline
```
LLM Output → Text Aggregation → TTS Processing → RTVI Observer → Client UI
   ↓              ↓                  ↓                ↓              ↓
 Tokens      Sentences          Audio+Text      bot-tts-text    Transcript
   LLMTextFrame TextFrame      TTSTextFrame    RTVI Message    conversationHistory
```

### Critical Configuration
The SINGLE most critical configuration is:
```python
# kokoro_professional.py, Line 70
push_text_frames=True  # Enables text mirroring to client
```

Without this, TTSTextFrame is never emitted, and no text reaches the client.

### Text Mirroring Mechanism
1. TTS synthesizes audio (TTSAudioRawFrame)
2. TTS emits text (TTSTextFrame) when `push_text_frames=True`
3. RTVIObserver catches TTSTextFrame
4. Converts to RTVI `bot-tts-text` message
5. Sends via WebRTC to client
6. Client receives via `onBotTtsText` callback

### Duplicate Prevention
The client prevents duplicate text via multi-level checking:
- Exact match: `msg.text === newText`
- Substring: `msg.text.includes(newText)`
- Reverse substring: `newText.includes(msg.text)`

This handles cases where text is sent multiple times from different sources.

### Sentence Boundary Detection
FastTextAggregator releases text on:
1. Sentence endings: `.!?`
2. Clause boundaries: `,;:` (with word count check)
3. Timeout: No text for >0.5 seconds
4. Interruption: User starts speaking

---

## File-to-Stage Mapping

| Stage | Server File | Key Lines | Client File | Key Lines |
|-------|------------|-----------|------------|-----------|
| **LLM Output** | `core/llm/direct_mlx_llm.py` | 301 | - | - |
| **Text Agg** | `core/aggregators/fast_text.py` | 60 | - | - |
| **TTS** | `core/tts/kokoro_professional.py` | 70 | - | - |
| **RTVI** | `core/factory.py` | 550-551 | - | - |
| **Context** | `core/factory.py` | 418 | - | - |
| **UI** | - | - | `components/VoiceApp.tsx` | 196-244 |
| **Display** | - | - | `components/VoiceApp.tsx` | 866-905 |

---

## Testing the Pipeline

### Minimal Test (Does text reach client?)
1. Server: Look for "🔍 [TRANSCRIPT.ASSISTANT INPUT] TTSTextFrame"
2. Client: Look for "🎵 onBotTtsText:" in browser console
3. UI: Check if conversationHistory updates

### Full Test (Does complete flow work?)
1. Send a message to the voice agent
2. Verify server logs show:
   - "⚡ TTFT:" (LLM token time)
   - "[FastTextAggregator] Releasing text:" (sentence detected)
   - "🔍 [TRANSCRIPT.ASSISTANT INPUT] TTSTextFrame:" (text in pipeline)
3. Verify client shows:
   - "🎵 onBotTtsText:" (text received)
   - "🎯 Complete sentence detected:" (punctuation detected)
   - "💾 Saving new sentence:" (added to history)
4. Verify UI shows:
   - Text appears in transcript panel
   - No duplicates
   - Proper formatting (no markdown)

---

## Debugging Strategy

### Step 1: Identify Where Text is Lost
1. Is LLM generating tokens? (Check "⚡ TTFT:" log)
2. Is text aggregating? (Check "[FastTextAggregator] Releasing text:")
3. Is TTSTextFrame in pipeline? (Check "🔍 [TRANSCRIPT.ASSISTANT INPUT]")
4. Is client receiving? (Check "🎵 onBotTtsText:" in console)
5. Is it in conversation history? (Check "💾 Saving new sentence:")
6. Is it rendering? (Visual check of transcript panel)

### Step 2: Check Configuration
- Is `push_text_frames=True` set? (kokoro_professional.py:70)
- Is RTVI processor added? (factory.py:332-334 or 451-453)
- Is RTVIObserver in observers? (factory.py:550-551)

### Step 3: Check State Management
- Is `conversationHistory` state updating? (VoiceApp.tsx:80)
- Is `currentAssistantTranscript` accumulating? (VoiceApp.tsx:82)
- Is transcript panel rendering? (VoiceApp.tsx:786-979)

---

## Performance Tuning

### Reduce Latency
1. **LLM:** Use smaller model (faster TTFT)
2. **Aggregation:** Increase min_tokens (fewer releases)
3. **TTS:** Disable intermediate processing
4. **Network:** Reduce RTT (local deployment)

### Improve Text Quality
1. **Aggregation:** Lower min_tokens for faster display
2. **Duplicate Prevention:** Add more comprehensive checks
3. **Formatting:** Enhance markdown removal

### Increase Reliability
1. **Fallback Handlers:** Ensure onBotTranscript is implemented
2. **Error Handling:** Catch exceptions in callbacks
3. **State Management:** Persist conversation history

---

## Common Gotchas

1. **Missing `push_text_frames=True`**
   - Text will never reach client
   - Solution: Set in TTS init (kokoro_professional.py:70)

2. **RTVI Observer not configured**
   - TTSTextFrame won't be converted to RTVI message
   - Solution: Add to PipelineTask observers (factory.py:550-551)

3. **Duplicate text from multiple sources**
   - onBotTtsText and onBotTranscript both emit same text
   - Solution: Implement duplicate prevention (VoiceApp.tsx:211-223)

4. **Text without punctuation lost**
   - onBotTtsStopped fallback not implemented
   - Solution: Save remaining text in callback (VoiceApp.tsx:250-277)

5. **RTVIObserver not in pipeline order**
   - TTSTextFrame might pass by observer
   - Solution: Add RTVI early, after transport.input() (factory.py:332-334)

---

## Reference Tables

### Frame Journey Through Pipeline
```
Frame Type              Source              Destination         Purpose
────────────────────────────────────────────────────────────────────────
LLMTextFrame            direct_mlx_llm      FastTextAggregator  Token streaming
TextFrame               FastTextAggregator  TTS Service         Sentence input
TTSTextFrame            TTS (service)       RTVIObserver        Text mirroring
TTSAudioRawFrame        TTS (service)       transport.output()  Audio streaming
```

### Performance Breakdown
```
Stage                   Typical Time        Bottleneck          Tuning
──────────────────────────────────────────────────────────────────────
LLM TTFT                500-600ms           Model size          Smaller model
Text Aggregation        <50ms               Overhead            N/A
TTS Synthesis           200-400ms/sent      Model speed         Faster GPU
WebRTC Transport        <100ms              Network             Latency
────────────────────────────────────────────────────────────────────────
Total End-to-End        ~800ms              LLM TTFT            Biggest factor
```

---

## Document Statistics

| Document | Size | Lines | Focus |
|----------|------|-------|-------|
| LLM_TO_CLIENT_FLOW_MAP.md | 32KB | 829 | Complete reference |
| PIPELINE_QUICK_REFERENCE.md | 9KB | 300+ | Quick lookup |
| PIPELINE_FLOW_SUMMARY.txt | 17KB | 400+ | Text format |
| **Total** | **58KB** | **1500+** | All aspects |

---

## Getting Help

### For questions about...

**LLM output:**
- File: `LLM_TO_CLIENT_FLOW_MAP.md` (Section 1)
- File: `PIPELINE_QUICK_REFERENCE.md` (Table: Critical Files)

**Text aggregation:**
- File: `LLM_TO_CLIENT_FLOW_MAP.md` (Section 2)
- File: `PIPELINE_FLOW_SUMMARY.txt` (STAGE 2)

**TTS configuration:**
- File: `PIPELINE_QUICK_REFERENCE.md` (Configuration Checklist)
- File: `PIPELINE_FLOW_SUMMARY.txt` (CRITICAL CONFIGURATION PARAMETERS)

**Client UI:**
- File: `LLM_TO_CLIENT_FLOW_MAP.md` (Section 6)
- File: `PIPELINE_FLOW_SUMMARY.txt` (STAGE 6)

**Debugging:**
- File: `PIPELINE_QUICK_REFERENCE.md` (Debugging Checklist)
- File: `PIPELINE_FLOW_SUMMARY.txt` (COMMON ISSUES & SOLUTIONS)

**Performance:**
- File: `PIPELINE_FLOW_SUMMARY.txt` (PERFORMANCE METRICS)
- File: `LLM_TO_CLIENT_FLOW_MAP.md` (Performance Optimizations)

---

## How to Use These Docs

### First Time Understanding?
1. Start: `PIPELINE_QUICK_REFERENCE.md` (2 min)
2. Then: `LLM_TO_CLIENT_FLOW_MAP.md` Section 1-3 (15 min)
3. Finally: Full `LLM_TO_CLIENT_FLOW_MAP.md` (30 min)

### Quick Debugging?
1. Check: `PIPELINE_QUICK_REFERENCE.md` (Debugging Checklist)
2. Search: `PIPELINE_FLOW_SUMMARY.txt` (Common Issues)

### Code Review?
1. Reference: `LLM_TO_CLIENT_FLOW_MAP.md` (with line numbers)
2. Check: `PIPELINE_QUICK_REFERENCE.md` (Tables)

### Integration Work?
1. Study: `PIPELINE_FLOW_SUMMARY.txt` (Complete flow)
2. Reference: `PIPELINE_QUICK_REFERENCE.md` (Critical files, config)

---

## Contributing to This Documentation

If you make changes to the pipeline, please update:

1. **Modified files:** Update line numbers in all docs
2. **New stages:** Add section to `LLM_TO_CLIENT_FLOW_MAP.md`
3. **Config changes:** Update `PIPELINE_QUICK_REFERENCE.md` tables
4. **Issue patterns:** Add to `PIPELINE_FLOW_SUMMARY.txt` (Common Issues)

---

## Last Updated

Generated: 2025-10-29
Scope: Complete LLM to Client UI Pipeline
Coverage: All stages, frame types, processors, client UI
Version: 1.0

---

**Start reading:** Choose your document above based on your needs.
**Questions?** Refer to "Getting Help" section above.
