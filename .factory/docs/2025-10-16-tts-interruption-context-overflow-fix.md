# TTS Interruption & Restart Issue Analysis and Solution

## Problem Identified

The agent is experiencing **context window overflow** where previous conversation responses are being repeated instead of responding to new user input. Here's what's happening:

1. **User interrupts mid-TTS response** (normal barge-in behavior)
2. **LLM context window grows too large** with repeated/malformed responses 
3. **New user input gets lost** in the massive context history
4. **Agent restarts old sentences** instead of processing new input

## Root Cause Evidence

From the logs:
- TTS interruptions working correctly: `User started speaking` → `Bot stopped speaking`
- **Critical warning**: `User stopped speaking but no new aggregation received` (twice)
- **Massive LLM context**: Shows extremely long repeated responses about "quick brown fox" 
- **Context growing**: Each turn adds more content without proper pruning

## Technical Root Causes

### 1. **LLM Context Aggregator Issues**
- `LLMUserAggregatorParams` with insufficient pruning
- Context window growing beyond token limits
- Old responses not being properly summarized/discarded

### 2. **Memory System Not Properly Integrated**
- Memory bullets are being added but not reducing conversation history
- Context accumulation instead of context replacement

### 3. **Barge-in Timing Issues**
- User speech starts during TTS but LLM processing may not have completed
- Race condition between interruption and response finalization

## Solution Plan

### Phase 1: Immediate Fixes (High Priority)
1. **Fix LLM Context Aggregation**
   - Reduce `CONTEXT_MAX_TURN_PAIRS` from 4 to 2
   - Implement aggressive context pruning on each turn
   - Add token budget enforcement

2. **Fix Barge-in Handling**
   - Ensure `LLM_ENABLE_EMULATED_VAD_INTERRUPTION` is working
   - Add proper context clearing on interruption
   - Fix "no new aggregation received" warnings

3. **TTS Interruption Robustness**
   - Improve `request_cancel()` reliability in TTS service
   - Add interruption state tracking

### Phase 2: Memory Integration (Medium Priority)
1. **Memory-First Context Strategy**
   - Use memory bullets to replace conversation history
   - Implement semantic compression of old turns
   - Add context quality scoring

2. **Context Window Optimization**
   - Dynamic turn count based on token usage
   - Smart summarization of long responses
   - Prevent response loops/repetition

### Phase 3: Performance & Monitoring (Low Priority)
1. **Enhanced Logging**
   - Add context size monitoring
   - Track interruption success rate
   - Monitor response quality metrics

## Implementation Order

1. **Fix context aggregation** (prevents immediate recurrence)
2. **Improve barge-in handling** (better user experience)  
3. **Integrate memory properly** (long-term stability)
4. **Add monitoring** (prevent future regressions)

## Files to Modify

- `server/core/factory.py` - Context aggregator parameters
- `server/core/memory/hotpath_processor.py` - Context pruning logic
- `server/core/tts/tts_mlx_ultra_low_latency.py` - Interruption handling
- `server/config/settings.py` - Default configuration values
- `server/bot.py` - Pipeline event handlers

This solution addresses both the immediate symptom (restarting old sentences) and the underlying cause (context window overflow).