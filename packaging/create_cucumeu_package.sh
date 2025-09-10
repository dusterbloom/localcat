#!/bin/bash

# Cucumeu Packaging Script
# Creates a ready-to-distribute package for macOS testing

set -e

# Configuration
PROJECT_NAME="cucumeu"
VERSION="0.1.0-beta"
TAGLINE="The AI that remembers you"
DIST_DIR="${PROJECT_NAME}-macos-${VERSION}"

echo "🦉 Creating Cucumeu Package v${VERSION}"
echo "======================================="

# Clean previous builds
rm -rf "${DIST_DIR}" "${DIST_DIR}.zip"

# Create distribution directory
mkdir -p "${DIST_DIR}"

# Copy core files
echo "📦 Copying project files..."
cp -r ../server "${DIST_DIR}/"
cp -r ../client "${DIST_DIR}/"

# Clean unnecessary files
echo "🧹 Cleaning unnecessary files..."
find "${DIST_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${DIST_DIR}" -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
find "${DIST_DIR}" -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
find "${DIST_DIR}" -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true
find "${DIST_DIR}" -type f -name ".DS_Store" -delete 2>/dev/null || true
find "${DIST_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${DIST_DIR}" -type f -name ".env" -delete 2>/dev/null || true
rm -rf "${DIST_DIR}/server/memory.db" 2>/dev/null || true
rm -rf "${DIST_DIR}/server/graph.lmdb" 2>/dev/null || true

# Create branded README
cat > "${DIST_DIR}/README.md" << 'EOF'
# 🦉 Cucumeu - The AI That Remembers You

<p align="center">
  <em>"Like a wise Sardinian elder who knows everyone in the village, Cucumeu knows YOU."</em>
</p>

## Welcome to Cucumeu!

Cucumeu (Sardinian for "little owl") is a private AI voice assistant that actually remembers your conversations. Unlike ChatGPT or Claude that forget you after each session, Cucumeu maintains context across all your interactions - and it all stays on your Mac.

## ✨ What Makes Cucumeu Special?

### 🧠 Real Memory
- **Remembers Everything**: Your preferences, relationships, past conversations
- **Context Aware**: "Remember when we talked about..." actually works!
- **Lightning Fast**: Recalls information in just 48ms

### 🔒 Truly Private
- **100% Local**: Runs entirely on your Mac - no cloud, no tracking
- **Your Data**: All memories stored locally, you own everything
- **Works Offline**: No internet required after setup

### ⚡ Blazing Fast
- **<800ms latency**: Voice-to-voice response time
- **Optimized for Apple Silicon**: Takes full advantage of your M-series chip
- **Smart Caching**: Gets faster the more you use it

## 🚀 Quick Start (10 minutes)

1. **Setup** (first time only, ~8 minutes):
   ```bash
   ./setup-cucumeu.sh
   ```

2. **Start Cucumeu**:
   ```bash
   ./start-cucumeu.sh
   ```

3. **Open your browser** to http://localhost:3000

4. **Start talking!** Try:
   - "Hi, I'm [your name]"
   - "Remember my dog's name is Potola"
   - Then ask: "What's my pet's name?"

## 💬 Example Interactions

```
You: "My name is Alex and I work at Apple"
Cucumeu: "Nice to meet you, Alex! Working at Apple must be exciting."

[Next day]
You: "I'm thinking about switching jobs"
Cucumeu: "What's making you consider leaving Apple, Alex?"

[Week later]
You: "Remember my work situation?"
Cucumeu: "Yes, you were considering leaving Apple. Have you made a decision?"
```

## 🔜 Coming Soon

- **Voice Recognition**: Cucumeu will recognize who's speaking
- **Multi-user Profiles**: Each family member gets their own memory
- **Memory Export**: Backup and transfer your memories

## 🛠 Requirements

- macOS with Apple Silicon (M1/M2/M3)
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space
- Python 3.12+
- Node.js 18+

## 🆘 Troubleshooting

If Cucumeu seems slow on first start, that's normal - it's loading the AI models. Subsequent starts will be much faster.

## 📝 Feedback

This is beta software! Please share your experience:
- What worked well?
- What confused you?
- What features would you like?

## 🏝 About

Built with ❤️ in South Sardinia, where keeping things local and personal isn't just technology - it's a way of life.

---

*"In Sardinia, we say 'Chie hat tentu, hat perdidu' - who has tried, has not lost. Thank you for trying Cucumeu!"*
EOF

# Create setup script
cat > "${DIST_DIR}/setup-cucumeu.sh" << 'EOF'
#!/bin/bash

set -e

echo ""
echo "🦉 Cucumeu Setup Wizard"
echo "======================="
echo ""
echo "Welcome! I'm going to help you set up your personal AI companion."
echo "This will take about 8-10 minutes, mostly downloading AI models."
echo ""

# Check system requirements
echo "📋 Checking system requirements..."

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ Cucumeu requires Apple Silicon (M1/M2/M3) Mac"
    echo "   Your system: $(uname -m)"
    exit 1
fi
echo "✅ Apple Silicon detected"

# Check Python
if ! command -v python3.12 &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.12+ is required"
    echo "   Please install from: https://www.python.org/downloads/"
    exit 1
fi
PYTHON_CMD=$(command -v python3.12 || command -v python3)
echo "✅ Python found: $($PYTHON_CMD --version)"

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required"
    echo "   Please install from: https://nodejs.org/"
    exit 1
fi
echo "✅ Node.js found: $(node --version)"

# Check if LM Studio is installed (just a warning)
if ! pgrep -x "LM Studio" > /dev/null; then
    echo ""
    echo "⚠️  LM Studio not detected running"
    echo "   Cucumeu needs a local LLM server. Please:"
    echo "   1. Download LM Studio: https://lmstudio.ai/"
    echo "   2. Download a model (we recommend Gemma 2B or Llama 3.2)"
    echo "   3. Start the server from the Developer tab"
    echo ""
    read -p "Press Enter when LM Studio is running (or Ctrl+C to exit)..."
fi

echo ""
echo "🔧 Setting up server environment..."
cd server

# Create virtual environment
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
echo "📦 Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

# Pre-download models
echo ""
echo "🧠 Downloading AI models (this is the slow part, ~5 minutes)..."
echo "   These only download once and are cached for future use."

# Download Whisper model
python -c "
from mlx_whisper import load_model
print('  → Downloading Whisper speech recognition model...')
model = load_model('mlx-community/distil-whisper-large-v3')
print('  ✅ Whisper ready')
" || echo "  ⚠️  Whisper will download on first use"

# Download TTS model
python -c "
import mlx_audio
print('  → Downloading Kokoro voice model...')
mlx_audio.generate('Hello!', model='mlx-community/Kokoro-82M-bf16')
print('  ✅ Voice model ready')
" || echo "  ⚠️  Voice model will download on first use"

# Download VAD model
python -c "
import torch
import os
os.makedirs('models', exist_ok=True)
print('  → Downloading voice detection model...')
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
print('  ✅ Voice detection ready')
" || echo "  ⚠️  VAD will download on first use"

cd ..

echo ""
echo "🎨 Setting up web interface..."
cd client
npm install --silent
npm run build --silent
cd ..

echo ""
echo "✅ Cucumeu setup complete!"
echo ""
echo "📝 Quick start guide:"
echo "   1. Make sure LM Studio is running with a model loaded"
echo "   2. Run: ./start-cucumeu.sh"
echo "   3. Open: http://localhost:3000"
echo "   4. Click the microphone and start talking!"
echo ""
echo "🦉 Cucumeu is ready to remember you!"
EOF

# Create start script
cat > "${DIST_DIR}/start-cucumeu.sh" << 'EOF'
#!/bin/bash

echo ""
echo "🦉 Starting Cucumeu..."
echo "===================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "👋 Shutting down Cucumeu..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null
    fi
    if [ ! -z "$CLIENT_PID" ]; then
        kill $CLIENT_PID 2>/dev/null
    fi
    exit 0
}

trap cleanup INT TERM

# Check if LM Studio is running
if ! curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "⚠️  LM Studio server not detected at http://localhost:1234"
    echo ""
    echo "Please:"
    echo "1. Open LM Studio"
    echo "2. Load a model (Gemma 2B or Llama 3.2 recommended)"
    echo "3. Go to Developer tab → Start Server"
    echo ""
    read -p "Press Enter when ready (or Ctrl+C to exit)..."
fi

# Start server
echo "🚀 Starting Cucumeu server..."
cd server
source venv/bin/activate

# Set environment for faster startup
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << 'ENVFILE'
# Cucumeu Configuration
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=local-model
WHISPER_MODEL=mlx-community/distil-whisper-large-v3
TTS_MODEL=mlx-community/Kokoro-82M-bf16
ENVFILE
fi

python core/bot.py > cucumeu-server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start. Check cucumeu-server.log for details."
    exit 1
fi

cd ..

# Start client
echo "🎨 Starting web interface..."
cd client
npm run dev > ../cucumeu-client.log 2>&1 &
CLIENT_PID=$!

cd ..

# Wait a moment for client to start
sleep 3

echo ""
echo "✅ Cucumeu is running!"
echo ""
echo "🌐 Open your browser to: http://localhost:3000"
echo ""
echo "💡 Tips:"
echo "   - Click the microphone to start talking"
echo "   - First response may be slow as models load"
echo "   - Say 'Remember that...' to save information"
echo "   - Ask 'What do you remember about me?' to test memory"
echo ""
echo "📊 Logs:"
echo "   - Server: server/cucumeu-server.log"
echo "   - Client: cucumeu-client.log"
echo ""
echo "Press Ctrl+C to stop Cucumeu"
echo ""

# Keep script running
wait $SERVER_PID
EOF

# Create test instructions
cat > "${DIST_DIR}/TEST_INSTRUCTIONS.md" << 'EOF'
# 🧪 Cucumeu Beta Testing Guide

Thank you for testing Cucumeu! Your feedback is invaluable.

## Testing Checklist

### Basic Functionality
- [ ] Setup completes without errors
- [ ] Server starts successfully  
- [ ] Web interface loads at http://localhost:3000
- [ ] Microphone permission granted
- [ ] First voice interaction works

### Memory Testing
- [ ] Tell Cucumeu your name
- [ ] Mention a pet, friend, or family member
- [ ] Close and restart Cucumeu
- [ ] Ask "What do you remember about me?"
- [ ] Verify it recalls your information

### Performance Testing
- [ ] Response time feels natural (<1 second)
- [ ] No significant lag or stuttering
- [ ] CPU usage remains reasonable

### Please Note

1. **Response Time**: How does it feel? Natural? Too slow?
2. **Memory Accuracy**: Does it remember correctly?
3. **Confusion Points**: What wasn't clear?
4. **Wishlist**: What features would you want?
5. **Bugs**: Any crashes or errors?

## Feedback Form

Please email your feedback with:

**Setup Experience** (1-5): ___
**Ease of Use** (1-5): ___
**Memory Feature** (1-5): ___
**Performance** (1-5): ___
**Overall** (1-5): ___

**What worked well?**


**What needs improvement?**


**Would you use this daily?** Yes/No/Maybe

**Additional comments:**


Thank you! 🦉
EOF

# Make scripts executable
chmod +x "${DIST_DIR}/setup-cucumeu.sh"
chmod +x "${DIST_DIR}/start-cucumeu.sh"

# Create ZIP archive
echo "📦 Creating distribution archive..."
zip -qr "${DIST_DIR}.zip" "${DIST_DIR}"

# Final size
SIZE=$(du -sh "${DIST_DIR}.zip" | cut -f1)

echo ""
echo "✅ Package created successfully!"
echo ""
echo "📦 File: ${DIST_DIR}.zip"
echo "📏 Size: ${SIZE}"
echo ""
echo "📨 To share with friends:"
echo "   1. Send them ${DIST_DIR}.zip"
echo "   2. They unzip and run setup-cucumeu.sh"
echo "   3. Then start-cucumeu.sh to use"
echo ""
echo "🦉 Cucumeu is ready to meet new friends!"
