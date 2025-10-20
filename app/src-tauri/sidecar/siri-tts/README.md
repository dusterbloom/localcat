# Siri TTS Sidecar

A lightweight Swift sidecar that generates speech audio using Apple's native AVSpeechSynthesizer (Siri voices).

## Features

- **Zero bundle size** - Uses built-in macOS voices
- **Instant startup** - No model loading required
- **Native quality** - True Siri voice output
- **Offline by default** - No network dependencies
- **Tiny binary** - Only ~60KB compiled size

## Building

```bash
./build.sh
```

Or manually:
```bash
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
swiftc -O -sdk "$SDK_PATH" -target arm64-apple-macos12.0 -o siri-tts main.swift
```

## Usage

```bash
# Basic usage with default Ava voice
./siri-tts "Hello from LocalCat" output.wav

# With specific voice ID
./siri-tts "Hello" output.wav com.apple.voice.enhanced.en-US.Ava
```

## Available Voices

List all available voices programmatically in Swift:
```swift
for voice in AVSpeechSynthesisVoice.speechVoices() {
    print("\(voice.identifier) - \(voice.name) (\(voice.language))")
}
```

Common voice IDs:
- `com.apple.voice.enhanced.en-US.Ava` - Female US English (default)
- `com.apple.voice.compact.en-US.Samantha` - Female US English (compact)
- `com.apple.voice.compact.en-US.Alex` - Male US English

## Output Format

- **Format**: WAV (PCM 16-bit)
- **Sample Rate**: 24kHz (determined by AVSpeechSynthesizer)
- **Channels**: Mono
- **Compatible with**: Pipecat, FFmpeg, and standard audio tools

## Current Status

### What Works
- ✅ Swift code compiles to ~60KB binary
- ✅ Uses native Siri voices
- ✅ Proper WAV header generation
- ✅ Float32 to Int16 PCM conversion

### Known Issue
⚠️ **The `AVSpeechSynthesizer.write()` callback completion is not triggering properly**, causing the process to hang after synthesis.

This is likely due to:
1. The callback running on a background thread without a run loop
2. macOS 15 (Sequoia) API behavior changes
3. Missing RunLoop processing in CLI context

### Potential Solutions

1. **Add RunLoop processing** (recommended):
   ```swift
   // After synth.write() call
   while !isComplete {
       RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1))
   }
   ```

2. **Use `say` command wrapper** (simplest):
   ```bash
   say -o output.aiff --data-format=LEI16@24000 "text"
   afconvert -f WAVE -d LEI16 output.aiff output.wav
   ```

3. **Use delegate pattern with speak()** instead of write():
   - Speak to an audio file URL
   - Convert from AIFF/CAF to WAV

## Integration with LocalCat

Once working, integrate as a Tauri sidecar:

1. Add to `tauri.conf.json`:
```json
{
  "bundle": {
    "externalBin": ["sidecar/siri-tts/siri-tts"]
  }
}
```

2. Update server TTS config to use siri-tts binary

## Why This Approach?

Compared to Kokoro/MLX TTS:
- **160MB saved** - No model files to bundle
- **10-30s faster startup** - No model loading
- **Better quality** - Native Siri voices
- **Lower latency** - Hardware-accelerated synthesis

This makes LocalCat truly "zero-install" - just the app bundle, no models needed!
