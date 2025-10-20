#!/usr/bin/env bash
set -euo pipefail

# LocalCat Dev Bootstrap
# - Detect OS
# - Ensure Python venv + Node deps
# - Optionally warm (download) models
# - Start server and client
# - Open http://localhost:3000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SERVER_DIR="$ROOT_DIR/server"
CLIENT_DIR="$ROOT_DIR/client"

OS="unknown"
case "${OSTYPE:-}" in
  darwin*) OS="macOS" ;;
  linux*)  OS="Linux" ;;
  msys*|cygwin*|win32*) OS="Windows" ;;
  *) OS="unknown" ;;
esac

echo "🔎 Detected OS: $OS"
if [[ "$OS" == "Windows" ]]; then
  echo "⚠️  For Windows, please run: scripts/dev.ps1" >&2
  exit 1
fi

have_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_python() {
  if ! have_cmd python3; then
    echo "❌ python3 not found. Please install Python 3.12+ and retry." >&2
    exit 1
  fi
  local pyv
  pyv=$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')
  case "$pyv" in
    3.12|3.13) ;; # ok
    *) echo "⚠️  Python $pyv detected. Recommended: 3.12+" ;;
  esac
}

ensure_node() {
  if ! have_cmd node || ! have_cmd npm; then
    echo "❌ Node.js and npm are required. Install Node 18+ (or 20+) and retry." >&2
    exit 1
  fi
}

create_venv_and_install() {
  echo "🔧 Setting up Python venv + deps…"
  cd "$SERVER_DIR"
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
  fi
  if [[ -f requirements-ml.txt ]]; then
    pip install -r requirements-ml.txt || true
  fi

  # On macOS, prefer lightweight STT for dev unless user opts in to Parakeet (requires parakeet-mlx)
  if [[ "$OS" == "macOS" ]]; then
    if ! python -c 'import importlib; import sys; sys.exit(0 if importlib.util.find_spec("parakeet_mlx") else 1)'; then
      echo "ℹ️  parakeet-mlx not found; dev will use Whisper-MLX STT (faster setup)."
      export VOICE_AGENT_STT_ENGINE="whisper_mlx"
    fi
  fi
}

install_node_deps() {
  echo "📦 Installing client dependencies…"
  cd "$CLIENT_DIR"
  if [[ -d node_modules ]]; then
    echo "  ↳ node_modules present; running npm ci --prefer-offline"
    npm ci --prefer-offline || npm ci || npm install
  else
    npm ci || npm install
  fi
}

warm_models() {
  # Best‑effort: prefetch Kokoro (TTS) and STT models to avoid first‑run delays
  echo "🌐 Warming models (best‑effort)…"
  cd "$SERVER_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python - <<'PY'
import os
import sys

def try_import(msg, fn):
    try:
        fn()
        print(f"[warm] ✅ {msg}")
    except Exception as e:
        print(f"[warm] ⚠️  {msg} skipped: {e}")

# Warm Kokoro MLX TTS (downloads mlx-community/Kokoro-82M-bf16)
def warm_kokoro():
    from mlx_audio.tts.utils import load_model
    load_model(os.getenv("TTS_MODEL", "mlx-community/Kokoro-82M-bf16"))

try_import("Kokoro TTS model", warm_kokoro)

# Warm STT: Parakeet if available, else Whisper-MLX via Pipecat service
def warm_parakeet():
    from parakeet_mlx import from_pretrained
    from_pretrained(os.getenv("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3"))

def warm_whisper_mlx():
    # Pipecat service initialization typically pulls model/cache on first import
    from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
    WhisperSTTServiceMLX(model=MLXModel.MEDIUM)

try:
    import importlib
    if importlib.util.find_spec("parakeet_mlx"):
        try_import("Parakeet STT model", warm_parakeet)
    else:
        try_import("Whisper-MLX STT model", warm_whisper_mlx)
except Exception as e:
    print(f"[warm] ⚠️  STT warmup skipped: {e}")

PY
}

ensure_siri_sidecar() {
  if [[ "$OS" == "macOS" ]]; then
    local sidecar="$ROOT_DIR/app/src-tauri/sidecar/siri-tts/siri-tts"
    local builder="$ROOT_DIR/app/src-tauri/sidecar/siri-tts/build.sh"
    if [[ ! -x "$sidecar" ]]; then
      if [[ -x "$builder" ]]; then
        echo "🔨 Building Siri TTS sidecar…"
        (cd "$ROOT_DIR/app/src-tauri/sidecar/siri-tts" && ./build.sh) || echo "⚠️  Failed to build Siri sidecar; will use Kokoro TTS."
      fi
    fi
  fi
}

open_browser() {
  local url="http://localhost:3000"
  case "$OS" in
    macOS) open "$url" >/dev/null 2>&1 || true ;;
    Linux) { command -v xdg-open >/dev/null && xdg-open "$url" >/dev/null 2>&1; } || true ;;
  esac
}

start_server() {
  echo "🚀 Starting server…"
  cd "$SERVER_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate

  # Ensure .env exists
  [[ -f .env ]] || { [[ -f .env.example ]] && cp .env.example .env; }

  # Prefer Siri TTS on mac if sidecar present; otherwise Kokoro
  if [[ "$OS" == "macOS" ]]; then
    if [[ -x "$ROOT_DIR/app/src-tauri/sidecar/siri-tts/siri-tts" ]]; then
      export VOICE_AGENT_TTS_ENGINE="siri_streaming"
      echo "  ↳ macOS Siri TTS enabled"
    fi
  fi

  # Start FastAPI dev server (port 7860)
  python bot.py --host 127.0.0.1 --port 7860 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$ROOT_DIR/.server.pid"
}

start_client() {
  echo "🧩 Starting Next.js client (port 3000)…"
  cd "$CLIENT_DIR"
  npm run dev &
  CLIENT_PID=$!
  echo "$CLIENT_PID" > "$ROOT_DIR/.client.pid"
}

cleanup() {
  echo "\n🧹 Cleaning up…"
  [[ -f "$ROOT_DIR/.client.pid" ]] && kill "$(cat "$ROOT_DIR/.client.pid" 2>/dev/null)" 2>/dev/null || true
  [[ -f "$ROOT_DIR/.server.pid" ]] && kill "$(cat "$ROOT_DIR/.server.pid" 2>/dev/null)" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

main() {
  ensure_python
  ensure_node
  create_venv_and_install
  install_node_deps
  ensure_siri_sidecar

  # Warm models unless user opts out
  if [[ "${SKIP_WARM_MODELS:-}" != "1" ]]; then
    warm_models || true
  fi

  start_server
  start_client
  sleep 2
  open_browser

  echo "✅ Dev environment running. Open http://localhost:3000"
  echo "   Press Ctrl+C to stop."
  # Wait on client
  wait "$CLIENT_PID"
}

main "$@"
