# Environment Variable Configuration Fix

## ❌ WRONG (What User Tried)
```bash
AUDIO_INTEL_FORCE_INTRO=true   AUDIO_INTEL_INTRO_PIPELINE=true &&     AUDIO_INTEL_SKIP_FOR_RETURNING=true && python bot.py
```

### Why This Failed
The `&&` operator **chains commands**, NOT environment variables!

This actually runs:
1. Set `AUDIO_INTEL_FORCE_INTRO=true` and `AUDIO_INTEL_INTRO_PIPELINE=true` → do nothing
2. Set `AUDIO_INTEL_SKIP_FOR_RETURNING=true` → do nothing  
3. Run `python bot.py` → **with NO environment variables!**

## ✅ CORRECT Ways to Set Environment Variables

### Option 1: One-liner (space-separated)
```bash
cd /Users/peppi/Dev/localcat/server
AUDIO_INTEL_FORCE_INTRO=true AUDIO_INTEL_INTRO_PIPELINE=true AUDIO_INTEL_SKIP_FOR_RETURNING=false python bot.py
```

### Option 2: Use the test script
```bash
./test_intro.sh
```

### Option 3: Export then run
```bash
export AUDIO_INTEL_FORCE_INTRO=true
export AUDIO_INTEL_INTRO_PIPELINE=true
export AUDIO_INTEL_SKIP_FOR_RETURNING=false
python bot.py
```

### Option 4: Update .env file
```bash
# Edit /Users/peppi/Dev/localcat/server/.env
AUDIO_INTEL_FORCE_INTRO=true
AUDIO_INTEL_INTRO_PIPELINE=true
AUDIO_INTEL_SKIP_FOR_RETURNING=false

# Then just run:
python bot.py
```

## Environment Variables Explained

| Variable | Value | Purpose |
|----------|-------|---------|
| `AUDIO_INTEL_FORCE_INTRO` | `true` | Force intro flow even for returning users (testing) |
| `AUDIO_INTEL_INTRO_PIPELINE` | `true` | Enable intro-aware pipeline (vs. standard pipeline) |
| `AUDIO_INTEL_SKIP_FOR_RETURNING` | `false` | Don't skip intro for users with existing profiles |

## Expected Behavior With Correct Config

### In Logs:
```
[Factory] Creating intro-aware pipeline (initial_state=intro, has_profiles=True)
```

### In Conversation:
- Agent says: "Welcome! I'm learning to recognize your voice. Please say a few words..."
- Progress updates: "Great! 1 of 3 samples collected..."
- Completion: "Perfect! I've enrolled your voice. Let's chat!"

## Verification

After starting with correct variables, check logs for:
1. ✅ `initial_state=intro` (not `conversation`)
2. ✅ Enrollment messages from EnrollmentCoordinator
3. ✅ No `EnrollmentProgressFrame` errors
4. ✅ Different TTS instance IDs (TTSMLXUltraLowLatency#0 and #1)
