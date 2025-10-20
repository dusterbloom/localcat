# LocalCat - Local Voice Agent

A privacy-first voice AI assistant that runs entirely on your Mac with Apple Silicon.

## What is LocalCat?

LocalCat is a voice-activated AI assistant that:
- Runs 100% locally on your Mac (no cloud, no internet required for core features)
- Uses Siri's voice for natural speech
- Responds to your voice commands with <800ms latency
- Remembers conversations and learns about you over time
- Supports vision capabilities (can see and discuss what's on your screen)

## System Requirements

- **macOS**: Monterey (12.0) or later
- **Processor**: Apple Silicon (M1/M2/M3/M4)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: ~7 GB for the app + models
- **Additional**: LM Studio running with a compatible model (see below)

## Installation

### Step 1: Extract the App

1. Download `LocalCat.app.zip`
2. Double-click to extract `LocalCat.app`
3. Drag `LocalCat.app` to your Applications folder
4. **Important**: Right-click → Open (first time only to bypass Gatekeeper)

### Step 2: Install LM Studio

LocalCat requires a local language model running via LM Studio:

1. Download LM Studio: https://lmstudio.ai
2. Install and open LM Studio
3. Download one of these models (in LM Studio):
   - **For voice-only**: `llama-3.2-1b-instruct` (faster, smaller)
   - **For voice + vision**: `minicpm-v-4_5` (can see your screen)
4. Load the model in LM Studio's "Local Server" tab
5. Start the server on port **1234** (default)

### Step 3: First Launch

1. Open LocalCat from Applications
2. Grant microphone permission when prompted
3. Wait ~30 seconds for initial model loading
4. The voice interface will appear when ready

## Usage

### Basic Interaction

1. **Talk**: Click the microphone button or just start talking (auto-mic mode)
2. **Listen**: LocalCat responds with Siri's voice
3. **See**: If using vision model, LocalCat can see and discuss your screen

### Features

- **Memory**: LocalCat remembers your conversations and preferences
- **Speaker Recognition**: Learns to recognize your voice
- **Vision** (with minicpm-v-4_5): Can see images and screen content
- **Offline**: Works without internet (except for initial model downloads)

### Tips

- Speak naturally - no wake word needed
- Clear, conversational sentences work best
- For vision queries, ask "what do you see?" or "describe this image"
- Memory builds over time - reference past conversations

## Troubleshooting

### App won't open
- Make sure you right-clicked → Open the first time
- Check that you have Apple Silicon (not Intel)
- Try: System Settings → Privacy & Security → Allow LocalCat

### No voice response
- Verify LM Studio server is running on port 1234
- Check microphone permissions in System Settings
- Restart LocalCat

### Slow performance
- Close other heavy applications
- Ensure LM Studio has GPU acceleration enabled
- Try the smaller llama-3.2-1b model for faster responses

### "Models downloading"
- First run downloads required AI models
- This is normal and only happens once
- Requires internet for initial setup

## Technical Details

### What's Inside?

- **STT**: Parakeet (NVIDIA) - speech recognition
- **TTS**: Siri Streaming - native macOS voices
- **LLM**: Your choice via LM Studio
- **Memory**: Local SQLite database
- **Vision**: MiniCPM-v-4.5 (optional)

### Privacy

- Everything runs on your Mac
- No data sent to cloud services
- No telemetry or tracking
- Memory stored locally in `~/Library/Application Support/LocalCat/`

## Support

For issues, questions, or contributions:
- GitHub: [Your repo URL]
- Developed by: [Your name/team]

## License

[Your license here]

---

**Enjoy your local AI assistant!** 🐱
