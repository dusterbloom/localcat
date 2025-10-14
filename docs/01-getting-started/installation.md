# Installation Guide

Comprehensive setup instructions for LocalCat on macOS.

## System Requirements

### Hardware
- **Apple Silicon Mac** (M1, M1 Pro/Max, M2, M2 Pro/Max, M3, M3 Pro/Max, M4 series)
- **Minimum RAM**: 8GB (16GB recommended)
- **Available Storage**: 5GB for models and dependencies

### Software
- **macOS**: 13.0 (Ventura) or later
- **Python**: 3.12 or later
- **Node.js**: 18.0 or later
- **Git**: For cloning the repository

## Step 1: Install Prerequisites

### Python 3.12

```bash
# Check current Python version
python3 --version

# If you need Python 3.12, install via Homebrew
brew install python@3.12

# Or download from python.org
# https://www.python.org/downloads/
```

### Node.js

```bash
# Install via Homebrew
brew install node

# Or download from nodejs.org
# https://nodejs.org/
```

### UV (Recommended Python Package Manager)

UV is significantly faster than pip and handles dependencies better:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add to your shell config)
export PATH="$HOME/.cargo/bin:$PATH"
```

## Step 2: Install LLM Server

Choose one of the following options:

### Option A: Ollama (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# In a new terminal, pull the required model
ollama pull gemma3n:4b
```

### Option B: LM Studio

1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install the application
3. Download a compatible model (e.g., Gemma 3N 4B)
4. Start the server from the Developer tab

## Step 3: Clone and Configure LocalCat

```bash
# Clone the repository
git clone https://github.com/your-org/localcat.git
cd localcat

# Setup server environment
cd server
cp .env.example .env

# Edit .env if using non-default settings
# nano .env  # or use your preferred editor
```

## Step 4: Install Dependencies

### Server Dependencies

```bash
cd server

# Using UV (fastest, recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Client Dependencies

```bash
cd ../client
npm install
```

## Step 5: Pre-cache Models (Optional but Recommended)

This prevents delays on first run:

```bash
cd ../server
source .venv/bin/activate  # or venv/bin/activate if using pip

# Cache TTS model
python -c "
from mlx_audio import generate_speech
generate_speech('Hello!', model='mlx-community/Kokoro-82M-bf16')
"

# Cache Whisper model
python -c "
import mlx_whisper
mlx_whisper.load_models.load_model('base')
"
```

## Step 6: Verify Installation

### Test Server

```bash
cd server
source .venv/bin/activate
python bot.py

# You should see:
# "Starting LocalCat server..."
# "Models loaded successfully"
# "Server ready at http://localhost:7860"
```

### Test Client

In a new terminal:

```bash
cd client
npm run dev

# You should see:
# "ready - started server on http://localhost:3000"
```

## Environment Configuration

### Essential Settings

Edit `server/.env` for your setup:

```bash
# LLM Configuration (for Ollama)
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_MODEL=gemma3n:4b

# For LM Studio, use:
# OPENAI_BASE_URL=http://localhost:1234/v1

# Performance Tuning
TTS_ULTRA_LOW_LATENCY=true
VAD_STOP_SECS=0.8
PREWARM_MODELS=true

# Memory System
MEMORY_ENABLED=true
MEMORY_BULLETS_MAX=3
```

### Advanced Options

```bash
# Enable metrics for debugging
MEMORY_PROCESSOR_METRICS=true
HOTMEM_LOG_LEVEL=DEBUG

# Audio settings
KOKORO_BUFFER_MS=40
WHISPER_MODEL=base  # or 'small' for better accuracy
```

## Platform-Specific Notes

### macOS Sonoma (14.x)
- Ensure microphone permissions are granted
- System Settings → Privacy & Security → Microphone

### macOS Ventura (13.x)
- May require Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Apple Silicon Optimization
- LocalCat is optimized for Apple Silicon
- Uses MLX framework for best performance
- Metal Performance Shaders utilized automatically

## Troubleshooting Installation

### "Module not found" Errors
```bash
# Ensure virtual environment is activated
source server/.venv/bin/activate  # or venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### "Port already in use"
```bash
# Find and kill process using port 7860
lsof -i :7860
kill -9 <PID>
```

### Slow First Startup
This is normal - models are being downloaded and cached. Subsequent starts will be much faster.

### Memory Issues
If you get memory errors with 8GB RAM:
- Close other applications
- Use smaller models (whisper 'tiny' instead of 'base')
- Reduce MEMORY_BULLETS_MAX in .env

## Uninstallation

To completely remove LocalCat:

```bash
# Remove repository
rm -rf /path/to/localcat

# Remove Python virtual environment (if created elsewhere)
rm -rf ~/.virtualenvs/localcat  # or wherever you created it

# Remove cached models
rm -rf ~/Library/Caches/huggingface
rm -rf ~/.cache/whisper
rm -rf ~/.ollama/models/gemma3n  # if using Ollama
```

## Next Steps

- ✅ Run through the [Quick Start Guide](./quick-start.md)
- 📖 Read about [Configuration Options](./configuration.md)
- 🏗️ Understand the [Architecture](../02-architecture/system-overview.md)
- 🔧 Set up for [Development](../03-development/setup.md)

---

**Need help?** Check the [Troubleshooting Guide](../10-reference/troubleshooting.md) or open an issue on GitHub.