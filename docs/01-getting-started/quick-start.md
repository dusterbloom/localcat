# Quick Start Guide

Get LocalCat running in under 5 minutes with this streamlined setup guide.

## Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.12+**
- **Node.js 18+** (for the web client)
- **Ollama or LM Studio** for running local LLMs

## 1. Clone and Setup

```bash
git clone <repository-url>
cd localcat
```

## 2. Install Ollama (if not already installed)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required model
ollama pull gemma3n:4b

# Start Ollama server (if not already running)
ollama serve
```

## 3. Configure Environment

```bash
cd server
cp .env.example .env
# The defaults should work with Ollama running locally
```

## 4. Start the Server

```bash
# Using uv (recommended - fastest)
uv run bot.py

# Or using traditional pip
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py
```

> ⏱️ **First Run**: The initial startup will download model weights (30-60 seconds). Subsequent runs are much faster (2-5 seconds).

## 5. Start the Web Client

Open a new terminal:

```bash
cd client
npm install
npm run dev
```

## 6. Connect and Talk!

1. Open your browser to `http://localhost:3000`
2. Click the microphone button to start talking
3. Enjoy sub-800ms voice-to-voice latency!

## 🚀 Pro Tips

### Speed Up Subsequent Runs

After the first run, prevent model update checks:

```bash
HF_HUB_OFFLINE=1 uv run bot.py
```

### Pre-cache TTS Models

Run this once to avoid delays on first voice response:

```bash
mlx-audio.generate --model "mlx-community/Kokoro-82M-bf16" --text "Hello!" --output "test.wav"
```

### Using LM Studio Instead of Ollama

1. Open LM Studio
2. Go to the "Developer" tab
3. Start the local server
4. Update `.env`:
   ```bash
   OPENAI_BASE_URL=http://localhost:1234/v1  # LM Studio default
   ```

## Common Issues

### "Model not found" Error
- Ensure Ollama is running: `ollama serve`
- Check model is installed: `ollama list`

### High Latency (>800ms)
- Ensure you're using Apple Silicon Mac
- Check that TTS models are cached
- Try setting `TTS_ULTRA_LOW_LATENCY=true` in `.env`

### Client Can't Connect
- Ensure server is running on port 7860
- Check no firewall blocking local connections
- Try refreshing the browser

## Next Steps

- 📖 Read the [Server Architecture](../02-architecture/server-architecture.md)
- ⚙️ Check [Configuration Guide](./configuration.md) for customization
- 🧠 Learn about the [Memory System](../02-architecture/memory-system-map.md)
- 🔧 See [Development Guide](../03-development/using-localcat-team.md) for contributing

---

**Need help?** Open an issue on GitHub or check the main [README](../../README.md) for troubleshooting.