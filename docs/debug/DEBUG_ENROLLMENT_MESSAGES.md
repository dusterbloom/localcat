# Debug: Enrollment Message Investigation

## What User Hears vs What System Sends

### Observation:
User hears:
1. "Hello"
2. "I'm LocalCat"

### System Logs Show:
```
TTS input: Hello!...           ← RTVI initial greeting
TTS input: Hi!...              ← Intro message (truncated in log)
TTS input: I'm LocalCat....    ← Progress message?
```

## Analysis

The TTS logs truncate long messages with "...". We need to see the FULL text being sent.

### Added Debug Logging

In `enrollment_coordinator.py`:

```python
# Before sending intro
logger.info(f"[EnrollmentCoordinator] Sending intro: '{intro_text[:80]}...'")

# Before sending progress
logger.info(f"[EnrollmentCoordinator] Sending progress: '{progress_text}'")
```

This will show us exactly what text is being sent to TTS.

## Hypothesis

**Likely Issue:** The intro message IS being sent correctly, but:

1. RTVI sends "Hello!" first (client-ready response)
2. Our intro message comes ~0.5s later but user doesn't hear it (interrupted?)
3. Only the first TextFrame gets spoken

**Alternative:** The intro message text might be empty or wrong somehow

## Next Test

After restart, look for these new log lines:
```
[EnrollmentCoordinator] Sending intro: 'Hi! I'm LocalCat. I can learn to recognize...'
[EnrollmentCoordinator] Sending progress: 'Learning your voice... 1 of 3'
```

This will tell us if the problem is:
- ❌ Messages not being created correctly
- ❌ Messages being interrupted
- ❌ RTVI greeting overriding our messages
- ✅ Or something else entirely

## Expected Flow

1. Connection establishes
2. RTVI sends "Hello!" (config-driven)
3. User speaks first utterance
4. AudioIntel emits EnrollmentProgressFrame(1/3)
5. Coordinator sends intro: "Hi! I'm LocalCat..."
6. Coordinator sends progress: "Learning your voice... 1 of 3"
7. Both messages go to INTRO pipeline → TTS

## If Messages Are Correct But Not Heard

Possible causes:
1. **Messages being interrupted** - User speaking interrupts TTS
2. **RTVI config** - Initial greeting might be configured somewhere
3. **Pipeline routing** - Messages going to wrong branch
4. **TTS queuing** - Multiple messages colliding

## RTVI Greeting Config

The "Hello!" is likely from RTVI's config response. Check `bot.py` for RTVIProcessor configuration.

Might need to disable or customize the initial greeting.
