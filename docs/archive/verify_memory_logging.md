# Memory Logging Verification

## Changes Made

The following INFO-level logs have been added to make memory processes visible in the server logs:

### 1. In `frame_processor.py`
- When processing transcription: `"[FrameProcessor] Processing transcription: 'text'"`
- When memory processing completes: `"[FrameProcessor] Memory processing complete: extracted X facts, prepared Y bullets"`
- When memory context is injected: `"[FrameProcessor] Memory context injected: X bullets added to conversation"`

### 2. In `memory_hotpath.py`
- When starting processing: `"[HotMem] Processing text: 'text...'"`
- When storing conversation turn: `"[HotMem] Storing conversation turn X in session Y"`
- When question detected: `"[HotMem] Question detected - not storing as memory: 'text...'"`
- When turn completes: `"[HotMem] Memory turn complete: X bullets generated, Y facts stored, Zms elapsed"`

### 3. In `retrieval.py`
- When searching memory: `"[Retrieval] Searching memory sources=[X] for query='text...'"`
- When greeting detected: `"[Retrieval] Greeting/smalltalk detected - no memory context needed for: 'text'"`
- When no memory found: `"[Retrieval] No memory context found for query"`
- When returning memories: `"[Retrieval] Returning X memory bullets from sources: {graph:X, convo:Y, summary:Z}"`

## Expected Log Output

After these changes, you should see INFO-level logs like:

```
2025-10-29 10:06:54.159 | INFO | core.memory.frame_processor:_handle_transcription_frame:290 | [FrameProcessor] Processing transcription: 'What's my favorite number?'
2025-10-29 10:06:54.173 | INFO | core.memory.memory_hotpath:process_turn:228 | [HotMem] Processing text: 'What's my favorite number?...'
2025-10-29 10:06:54.280 | INFO | core.memory.memory_hotpath:process_turn:282 | [HotMem] Question detected - not storing as memory: 'What's my favorite number?...'
2025-10-29 10:06:54.175 | INFO | core.memory.retrieval:retrieve:250 | [Retrieval] Searching memory sources=['graph'] for query='What's my favorite number?...'
2025-10-29 10:06:54.361 | INFO | core.memory.retrieval:retrieve:361 | [Retrieval] No memory context found for query
2025-10-29 10:06:54.175 | INFO | core.memory.frame_processor:_process_transcription:425 | [FrameProcessor] Memory processing complete: extracted 1 facts, prepared 0 bullets
2025-10-29 10:06:54.375 | INFO | core.memory.memory_hotpath:process_turn:375 | [HotMem] Memory turn complete: 0 bullets generated, 0 facts stored, 201.8ms elapsed
```

## Testing the Changes

To verify the logging is working:

1. Start the LocalCat server
2. Check the server log at `/Users/peppi/Library/Logs/LocalCat/server.log`
3. Speak to the voice agent with statements and questions
4. Look for the INFO-level memory logs listed above

## Filtering Memory Logs

To view only memory-related logs:

```bash
grep -E "INFO.*\[(HotMem|FrameProcessor|Retrieval)\]" /Users/peppi/Library/Logs/LocalCat/server.log
```

Or to see real-time memory logs:

```bash
tail -f /Users/peppi/Library/Logs/LocalCat/server.log | grep -E "INFO.*\[(HotMem|FrameProcessor|Retrieval)\]"
```