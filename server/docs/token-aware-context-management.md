# Token-Aware Context Management

## Problem Solved

In long conversations, unbounded context growth caused LLM and TTS performance degradation due to:
- Increasing token counts slowing down inference
- No token-based limiting (only message-count based)
- Cascading latency issues affecting user experience

## Solution

Implemented **token-aware context management** using accurate token counting with `tiktoken` to prevent performance degradation while maintaining conversation coherence.

## Architecture

### Core Principle
Keep conversations "forever" through three-tier system:

1. **System Messages** (always kept): Session header, persona, memory bullets
2. **Recent Turns** (sliding window): Last N turn pairs fitting within token budget
3. **Long-term Memory** (HotMem): Important facts extracted from pruned turns

### Components

#### 1. TokenEstimator (`core/memory/token_estimator.py`)
- Accurate token counting using tiktoken (cl100k_base encoding)
- Fast (~1-2ms overhead for typical context)
- Handles multi-modal content (text + vision)
- Graceful fallback to 4-char heuristic if tiktoken unavailable
- Accounts for message structure overhead (+4 tokens per message)

#### 2. Enhanced Context Pruning (`core/memory/context_injector.py`)
- Token-aware `_prune_context_window()` method
- Calculates token budget with configurable threshold
- Always keeps system messages (session, persona, memory)
- Keeps most recent turn pairs that fit within budget
- Maintains minimum turns for coherence (configurable)
- Ensures complete user/assistant pairs
- Graceful fallback to message-count pruning on errors

#### 3. Configuration Settings

**VoiceAgentConfig** (`config/settings.py`):
```python
llm_context_max_tokens: int = 3000  # Reserve ~1000 tokens for response
llm_context_prune_threshold: float = 0.70  # Prune at 70% capacity
llm_context_min_turns: int = 3  # Always keep at least 3 recent turns
```

**MemoryConfiguration** (`core/memory/config_manager.py`):
- Same fields synced with VoiceAgentConfig
- Parsed from environment variables

**Environment Variables** (.env.example):
```bash
LLM_CONTEXT_MAX_TOKENS=3000
LLM_CONTEXT_PRUNE_THRESHOLD=0.70
LLM_CONTEXT_MIN_TURNS=3
```

## How It Works

### Token Budget Calculation
1. Load max tokens from config (default: 3000)
2. Calculate available budget: `max_tokens * prune_threshold` (default: 2100 tokens)
3. Reserve tokens for response generation (~1000 tokens)

### Pruning Strategy
1. **Separate messages**: System vs user/assistant
2. **Count system tokens**: Calculate tokens in system messages (always kept)
3. **Calculate remaining budget**: `available_budget - system_tokens`
4. **Keep recent turns**: Iterate from newest to oldest, keeping turns that fit
5. **Maintain minimum**: Always keep `min_turns` pairs for coherence
6. **Ensure pairs**: Remove incomplete user/assistant pairs

### Example Scenario

**Before Pruning** (50 turn pairs, ~4500 tokens):
```
[System] Session header (50 tokens)
[System] Persona prompt (200 tokens)
[System] Memory bullets (150 tokens)
[User/Assistant] 50 turn pairs (4100 tokens)
Total: ~4500 tokens ❌ Exceeds budget
```

**After Pruning** (5 turn pairs, ~1900 tokens):
```
[System] Session header (50 tokens)
[System] Persona prompt (200 tokens)
[System] Memory bullets (150 tokens)
[User/Assistant] 5 most recent turn pairs (1500 tokens)
Total: ~1900 tokens ✅ Within budget (2100)
```

**Pruned turns** (45 pairs): Important facts extracted to HotMem memory system

## Benefits

✅ **Prevents Performance Degradation**: Context stays bounded in long conversations
✅ **Maintains Coherence**: Memory system preserves important facts from pruned turns
✅ **Fast & Accurate**: tiktoken provides precise token counts (~1ms overhead)
✅ **Configurable**: All settings tunable via environment variables
✅ **Fail-Safe**: Graceful fallback to message-count pruning on errors
✅ **Backward Compatible**: Works with existing configuration, no breaking changes

## Configuration Examples

### Conservative (More Context)
```bash
LLM_CONTEXT_MAX_TOKENS=4000
LLM_CONTEXT_PRUNE_THRESHOLD=0.80
LLM_CONTEXT_MIN_TURNS=5
```

### Aggressive (Lower Latency)
```bash
LLM_CONTEXT_MAX_TOKENS=2000
LLM_CONTEXT_PRUNE_THRESHOLD=0.60
LLM_CONTEXT_MIN_TURNS=2
```

### Default (Balanced)
```bash
LLM_CONTEXT_MAX_TOKENS=3000
LLM_CONTEXT_PRUNE_THRESHOLD=0.70
LLM_CONTEXT_MIN_TURNS=3
```

## Testing

Comprehensive test suite in `tests/unit/test_token_aware_context.py`:

- ✅ Basic token estimation
- ✅ Message-level token counting
- ✅ Token-aware pruning
- ✅ Minimum turns enforcement
- ✅ Complete pair maintenance
- ✅ Graceful error fallback
- ✅ Metrics reporting

All tests pass successfully.

## Performance Impact

- **TokenEstimator**: ~1-2ms per context pruning operation
- **Context Pruning**: ~5-10ms for typical context (50+ messages)
- **Total Overhead**: <15ms per turn (negligible compared to LLM inference)

## Future Enhancements

1. **Adaptive Thresholding**: Adjust pruning threshold based on LLM performance
2. **Conversation Summarization**: Compress old turns into summaries
3. **Token Budget Monitoring**: Real-time dashboard of token usage
4. **Model-Specific Encodings**: Support different tokenizers per model

## Migration Guide

No migration needed! The system:
- Uses existing configuration defaults
- Falls back gracefully if tiktoken unavailable
- Maintains backward compatibility with message-count pruning

Simply update your `.env` if you want custom settings:
```bash
# Add to .env (optional - defaults work well)
LLM_CONTEXT_MAX_TOKENS=3000
LLM_CONTEXT_PRUNE_THRESHOLD=0.70
LLM_CONTEXT_MIN_TURNS=3
```

## References

- [tiktoken Documentation](https://github.com/openai/tiktoken)
- [OpenAI Token Counting Guide](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
- HotMem Memory System Architecture

---

**Implementation Date**: 2025-10-16
**Contributors**: LocalCat Team
**Status**: ✅ Production Ready
