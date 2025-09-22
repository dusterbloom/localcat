# Moshi/Kyutai STT Setup Guide

## Overview
This documents how to get the Kyutai/Moshi streaming STT working in the localcat-streaming project. Kyutai provides ultra-low latency (<80ms) streaming speech recognition using delayed streams modeling.

## Prerequisites

### 1. Install moshi_mlx package
```bash
cd server/
source .venv/bin/activate
pip install moshi_mlx
```

### 2. Verify Installation
```bash
python -c "from moshi_mlx import models, utils; print('moshi_mlx installed successfully')"
```

## Key Issues and Solutions

### Issue 1: Zero Transcripts Despite Audio
**Symptom**: The Kyutai STT initializes successfully but produces no transcripts, only PAD tokens.

**Cause**: Audio levels too low or microphone muted.

**Solution**:
- Check browser microphone permissions
- Ensure microphone is not muted in browser/system
- Verify audio amplitude > 0.01 for good detection

**Debugging**:
```python
# Added logging in kyutai_streaming_stt.py to track audio levels:
logger.info(f"🎤 Audio block: max={audio_max:.4f}, mean={audio_mean:.4f}")
if audio_max < 0.001:
    logger.warning("⚠️ Audio is silent or nearly silent!")
```

### Issue 2: Sentence Fragment Carryover
**Symptom**: Parts of previous sentences appear in new transcriptions:
- "This is wonderful" → "wonderful How fast is this?"

**Cause**: Text buffer not cleared between utterances.

**Solution**: Clear buffer when VAD detects user started speaking:
```python
async def process_frame(self, frame: Frame, direction=None):
    if isinstance(frame, UserStartedSpeakingFrame):
        logger.info("🎤 VAD detected user started speaking - clearing text buffer")
        self._text_buffer = []
        self._consecutive_eos_count = 0
        self._consecutive_pad_count = 0
```

### Issue 3: Aggressive Punctuation
**Symptom**: Question marks inserted inappropriately ("This is? wonderful")

**Cause**: Fast punctuation restoration model being overly aggressive.

**Solution**: Keep punctuation enabled but improve EOS token handling:
```python
# Only finalize if we have accumulated text and see an EOS
if self._text_buffer and self._consecutive_eos_count >= 1:
    final_text = "".join(self._text_buffer).strip()
    if final_text:
        punctuated_text = self._add_punctuation(final_text)
        yield TranscriptionFrame(text=punctuated_text)
    self._text_buffer = []
    self._consecutive_eos_count = 0
```

## Configuration

### Environment Variables
```bash
# Enable Kyutai streaming STT (default: true)
export USE_STREAMING_STT=true

# Kyutai model repository (MLX or Candle variant)
export KYUTAI_STT_REPO="kyutai/stt-1b-en_fr-mlx"  # MLX variant (recommended for Mac)
# export KYUTAI_STT_REPO="kyutai/stt-1b-en_fr-candle"  # Candle variant

# Optional: Disable punctuation if too aggressive
export ENABLE_PUNCTUATION=false  # Default: true
```

### Model Variants
- **MLX variant** (`kyutai/stt-1b-en_fr-mlx`): Optimized for Apple Silicon, uses moshi_mlx Mimi tokenizer
- **Candle variant** (`kyutai/stt-1b-en_fr-candle`): Uses RustyMimi tokenizer

## Technical Details

### Audio Processing Pipeline
1. **Input**: 16kHz PCM audio from WebRTC
2. **Resampling**: Upsampled to 24kHz (Kyutai's native rate)
3. **Blocking**: 80ms chunks (1920 samples at 24kHz)
4. **Tokenization**: Mimi audio tokenizer → Moshi LM → text tokens
5. **Decoding**: SentencePiece tokenizer → text
6. **Punctuation**: Optional restoration via FastPunctuationRestorer

### Token Types
- **PAD (ID: 3)**: Silence or no speech
- **EOS (ID: 2, 32000)**: End of sentence
- **BOS (ID: 1)**: Beginning of sentence
- **UNK (ID: 0)**: Unknown token

### Key Files
- `server/kyutai_streaming_stt.py`: Main STT implementation
- `server/bot.py`: Integration point (lines 39-45, 103-116)
- `server/fast_punctuation.py`: Punctuation restoration

## Testing

### Diagnostic Script
```bash
cd server/
source .venv/bin/activate
python tests/super_stt_diagnose.py \
    --hf-repo kyutai/stt-1b-en_fr-mlx \
    --text "hello how are you testing streaming" \
    --vad
```

### Live Testing
1. Start the server:
   ```bash
   cd server/
   source .venv/bin/activate
   python bot.py
   ```

2. Open http://localhost:7860 in browser

3. Allow microphone permissions and click mic button

4. Monitor logs for:
   - `✅ Successfully initialized Kyutai streaming STT`
   - `🎤 Audio block` messages showing audio levels
   - `🔤 Token` messages showing generated tokens
   - `Kyutai STT:` messages showing transcribed text

## Troubleshooting

### No Audio Reaching STT
Check for: `WARNING: Timeout: No audio frame received`
- Verify WebRTC connection established
- Check browser microphone permissions
- Ensure microphone not muted

### Only PAD Tokens Generated
Check audio levels in logs:
- Should see `max > 0.01` for speech
- If `max < 0.001`: microphone audio too quiet

### Import Errors
```bash
# Reinstall moshi_mlx
pip uninstall moshi_mlx
pip install moshi_mlx

# Verify all dependencies
pip install rustymimi sentencepiece mlx huggingface-hub
```

## Performance Notes

- **Latency**: ~80ms from audio to text token
- **Model warmup**: ~5 seconds on first load
- **Memory usage**: ~2GB for model weights
- **CPU usage**: Moderate (MLX optimized for Apple Silicon)

## Future Improvements

1. **Adaptive punctuation**: Fine-tune aggressiveness based on pause duration
2. **Better VAD integration**: Use Kyutai's built-in VAD confidence scores
3. **Model quantization**: Use Q4/Q8 variants for lower memory usage
4. **Streaming optimization**: Reduce block size for even lower latency