# LocalCat - Installation Guide

Welcome to LocalCat! This is a voice-to-voice AI assistant that runs entirely on your Mac with no cloud required.

## System Requirements

- **macOS**: Big Sur (11.0) or later
- **Mac with Apple Silicon**: M1, M2, M3, or M4 chip
- **Storage**: ~2GB free space (for models and app)
- **RAM**: 8GB minimum, 16GB recommended

## Quick Install (5 Steps)

### 1. Download the App

Download the latest `LocalCat.dmg` from the releases page.

### 2. Install

1. Open the downloaded `LocalCat.dmg` file
2. Drag the `LocalCat` app to your `Applications` folder
3. Eject the DMG

### 3. First Launch

1. Open your `Applications` folder
2. Double-click `LocalCat`
3. If you see a security warning, right-click the app → "Open" → "Open" again
   - This is normal for apps downloaded from the internet

### 4. Allow Microphone Access

When prompted, click "OK" to allow microphone access.
- This is required for voice input to work
- Your voice data stays on your Mac - nothing is sent to the cloud

### 5. Connect and Talk

1. Wait for the app to initialize (30-60 seconds on first run)
   - The app is downloading AI models in the background
   - Subsequent launches will be much faster (5-15 seconds)

2. Click the "Connect" button when it appears

3. Start talking! The AI will respond with voice

## What to Expect

### First Launch
- **Time**: 30-60 seconds (downloading ~2GB of AI models)
- **What's happening**:
  - Downloading Parakeet speech recognition model
  - Downloading Kokoro voice synthesis model
  - Setting up local AI services
- **This only happens once!**

### Regular Launches
- **Time**: 5-15 seconds
- **What's happening**: Loading models from disk (no download needed)

### Voice Interaction
- **Latency**: Under 1 second from speaking to hearing AI response
- **Quality**: Natural-sounding voice with Siri TTS
- **Privacy**: Everything runs on your Mac - no internet required after initial setup

## Troubleshooting

### "LocalCat.app can't be opened because Apple cannot check it for malicious software"

**Solution**:
1. Right-click (or Control-click) on LocalCat.app
2. Select "Open" from the menu
3. Click "Open" in the dialog that appears
4. The app will now run normally

### App Doesn't Start

**Check**:
1. Make sure you're running macOS 11 or later
2. Make sure you have an M1/M2/M3/M4 Mac (not Intel)
3. Try quitting and reopening the app

**Reset**:
```bash
# Quit the app first, then run:
killall LocalCat
# Then reopen the app
```

### "Initializing..." Takes Too Long

**If it's been more than 2 minutes**:
1. Check your internet connection (needed for first-time model download)
2. Check free disk space (need ~2GB free)
3. Try quitting and reopening the app

### Microphone Not Working

**Check**:
1. System Settings → Privacy & Security → Microphone
2. Make sure "LocalCat" is enabled
3. If not listed, try speaking in the app to trigger the permission request

### Voice Quality Issues

The app uses **Siri TTS** by default for the best quality and fastest response time.

If you experience issues:
1. Check that your Mac's system volume is not muted
2. Try adjusting the volume slider in System Settings → Sound

## FAQ

### Does this require an internet connection?

- **First launch**: Yes (to download AI models, ~2GB)
- **After that**: No! Everything runs locally on your Mac

### Where are my conversations stored?

Conversations are stored in memory only. When you quit the app, conversation history is cleared.

###Is my data sent to the cloud?

**No.** Everything happens on your Mac:
- Speech recognition: Local (Parakeet model)
- AI reasoning: Local (Gemma3n model via local server)
- Voice synthesis: Local (Siri TTS)

### Can I use this offline?

Yes! After the first launch (which downloads models), you can use LocalCat completely offline.

### Why is the app so large?

The app includes:
- Speech recognition AI models (~1.1GB)
- Voice synthesis models (~160MB)
- Python runtime and dependencies (~300MB)
- All libraries needed to run completely offline

### How do I uninstall?

1. Quit LocalCat
2. Move `LocalCat.app` from Applications to Trash
3. (Optional) Remove cached models:
   ```bash
   rm -rf ~/Library/Application\ Support/io.localcat.app
   ```

### Can I change the voice?

The current version uses Siri's default voice. Custom voice selection will be added in a future update.

### How do I update?

1. Download the new version
2. Replace the old LocalCat.app in Applications with the new one
3. Your settings and preferences will be preserved

## Getting Help

- **Issues**: Report bugs at [GitHub Issues](https://github.com/yourusername/localcat/issues)
- **Documentation**: Full docs at [docs/](./docs/)
- **Community**: Join discussions in GitHub Discussions

## What's Next?

Now that you have LocalCat running, try:

- **Ask questions**: "What's the weather like?" (it will tell you it runs offline!)
- **Get help**: "What can you help me with?"
- **Be creative**: Have a conversation, ask for jokes, get advice

Remember: This is a local AI, so it's:
- ✅ Private (runs on your Mac only)
- ✅ Fast (no network latency)
- ✅ Offline (after initial setup)
- ⚠️ Limited to its training data (no real-time internet info)

Enjoy your personal AI assistant!
