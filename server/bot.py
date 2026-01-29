import argparse
import asyncio
import os
import re
import signal
import sys
import time
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Any

# Prevent tokenizers parallelism warning when forking processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress PyTorch deprecation warnings (RNN dropout, weight_norm)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Import centralized logging configuration
from core.logging_config import setup_logging_for_bot

# Load environment first
# override=False allows system env vars (e.g., from Tauri bundle) to take precedence over .env file
load_dotenv(override=False)  # Load from local server/.env, but don't override existing env vars

# Initialize centralized logging configuration BEFORE importing logger
setup_logging_for_bot()

# Import logger and modules that use logger AFTER configuration is complete
from loguru import logger

# Import factory AFTER logger is configured (factory.py imports logger at module level)
from core.factory import VoiceAgentFactory
from config import VoiceAgentConfig

import numpy as np

# ============================================================================
# MODEL PRELOADING INFRASTRUCTURE
# ============================================================================

@dataclass
class PreloadedModels:
    """Container for preloaded ML models that can be safely shared across connections.

    These models are stateless and can be reused. Services will create fresh instances
    around these models, as services have pipeline state and must be per-connection.
    """
    whisper_module: Any = None
    whisper_model_path: str = None
    kokoro_onnx_model: Any = None
    mlx_llm_model: Any = None
    mlx_llm_tokenizer: Any = None
    speechbrain_model: Any = None
    emotion_model: Any = None
    compactor_model: Any = None
    compactor_tokenizer: Any = None
    preload_time: float = 0.0

    @staticmethod
    async def preload(config: VoiceAgentConfig) -> 'PreloadedModels':
        """Preload all heavy ML models at server startup.

        This runs once at FastAPI startup and caches models in memory.
        Subsequent connections create lightweight service wrappers around these models.

        Args:
            config: Voice agent configuration to determine which models to load

        Returns:
            PreloadedModels instance with all models loaded
        """
        logger.info("=" * 70)
        logger.info("🚀 PRELOADING MODELS - This happens ONCE at server startup")
        logger.info("=" * 70)
        start_time = time.time()

        models = PreloadedModels()

        # ========== WHISPER STT (~1.3s) ==========
        try:
            stt_config = config.get_component_config("stt")
            if stt_config.get("type") == "whisper_mlx_direct":
                logger.info("  📝 Loading Whisper STT...")
                whisper_start = time.time()

                import mlx_whisper
                models.whisper_module = mlx_whisper
                models.whisper_model_path = stt_config.get("model", "mlx-community/whisper-small.en-mlx-q4")

                # Warmup: Forces model download and compilation
                dummy_audio = np.zeros(16000, dtype=np.float32)
                mlx_whisper.transcribe(
                    dummy_audio,
                    path_or_hf_repo=models.whisper_model_path,
                    verbose=False,
                    language="en",
                    temperature=0.0,
                    fp16=False
                )

                whisper_time = time.time() - whisper_start
                logger.info(f"  ✅ Whisper ready ({whisper_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Whisper preload failed: {e}")

        # ========== KOKORO TTS (~3s) ==========
        try:
            tts_config = config.get_component_config("tts")
            if tts_config.get("type") == "kokoro_professional":
                logger.info("  🎤 Loading Kokoro TTS...")
                kokoro_start = time.time()

                from kokoro_onnx import Kokoro
                voice = tts_config.get("voice", "af_heart")
                models.kokoro_onnx_model = Kokoro(voice_name=voice, lang="en-us")

                # Warmup: Test synthesis
                _ = models.kokoro_onnx_model("Hello", speed=1.0)

                kokoro_time = time.time() - kokoro_start
                logger.info(f"  ✅ Kokoro TTS ready ({kokoro_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Kokoro preload failed: {e}")

        # ========== MLX-LM (~700-850ms) ==========
        try:
            llm_config = config.get_component_config("llm")
            if os.getenv("LLM_USE_DIRECT_MLX", "false").lower() in ("true", "1", "yes"):
                logger.info("  🧠 Loading Direct MLX-LM...")
                mlx_start = time.time()

                import mlx_lm
                # Read model directly from env vars (VOICE_AGENT_LLM_MODEL or LLM_MODEL)
                model_id = os.getenv("VOICE_AGENT_LLM_MODEL") or os.getenv("LLM_MODEL") or llm_config.get("model")

                # CRITICAL FIX: mlx_lm.load() tries to call snapshot_download() even with HF_HUB_OFFLINE=1
                # To avoid this, check if model exists in local HF cache and pass absolute path
                hf_home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
                model_cache_name = model_id.replace("/", "--")
                cache_dir = os.path.join(hf_home, "hub", f"models--{model_cache_name}")

                # Try to find snapshot directory (either hash or "main")
                snapshot_path = None
                if os.path.exists(cache_dir):
                    snapshots_dir = os.path.join(cache_dir, "snapshots")
                    if os.path.exists(snapshots_dir):
                        # Get first snapshot directory (usually there's only one, or use "main")
                        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                        if snapshots:
                            # Prefer "main" if it exists, otherwise use first snapshot
                            snapshot = "main" if "main" in snapshots else snapshots[0]
                            snapshot_path = os.path.join(snapshots_dir, snapshot)

                # Use snapshot path if found, otherwise use model_id (will trigger download)
                model_path = snapshot_path if snapshot_path and os.path.exists(snapshot_path) else model_id
                logger.debug(f"Loading LLM from: {model_path}")

                models.mlx_llm_model, models.mlx_llm_tokenizer = mlx_lm.load(model_path)

                mlx_time = time.time() - mlx_start
                logger.info(f"  ✅ MLX-LM ready ({mlx_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ MLX-LM preload failed: {e}")

        # ========== AUDIO INTELLIGENCE (~2s) ==========
        try:
            if hasattr(config, 'audio_intelligence_enabled') and config.audio_intelligence_enabled:
                logger.info("  👂 Loading Audio Intelligence models...")
                audio_start = time.time()

                # Note: Full implementation would load SpeechBrain + emotion models
                # For now, we let the service handle it (it's already reasonably fast)
                # TODO: Preload these if they become a bottleneck

                audio_time = time.time() - audio_start
                logger.info(f"  ✅ Audio Intelligence ready ({audio_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Audio Intelligence preload failed: {e}")

        # ========== COMPACTOR SLM (~500ms) ==========
        try:
            compactor_model_id = os.getenv("COMPACTOR_MODEL", "mlx-community/Qwen3-0.6B-4bit")
            if os.getenv("ENABLE_CONTEXT_COMPACTOR", "true").lower() in ("true", "1", "yes"):
                logger.info(f"  🗜️ Loading Compactor SLM: {compactor_model_id}")
                compactor_start = time.time()

                import mlx_lm as _mlx_lm_compactor
                models.compactor_model, models.compactor_tokenizer = _mlx_lm_compactor.load(compactor_model_id)

                compactor_time = time.time() - compactor_start
                logger.info(f"  ✅ Compactor SLM ready ({compactor_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Compactor SLM preload failed: {e}")

        models.preload_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"✅ ALL MODELS PRELOADED in {models.preload_time:.2f}s")
        logger.info("   New connections will now be INSTANT (~0.5s)")
        logger.info("=" * 70)

        return models


# Global singleton for preloaded models
_preloaded_models: Optional[PreloadedModels] = None


def get_preloaded_models() -> Optional[PreloadedModels]:
    """Get the global preloaded models instance (if available).

    Returns:
        PreloadedModels instance if preloading succeeded, None otherwise
    """
    return _preloaded_models


# ============================================================================
# END MODEL PRELOADING INFRASTRUCTURE
# ============================================================================

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TextFrame
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from fastapi import WebSocket
from core.audio.enrollment_state import EnrollmentState


async def get_initial_greeting() -> str:
    """Simple greeting for now - HotMem will provide memory context."""
    return "Hello! How can I help you today?"

# IMPORTANT: Let Uvicorn own signal handling. We'll use FastAPI lifespan
# for startup/shutdown cleanup and avoid double SIGINT logs.

# Track active connection objects and their bot tasks
pcs_map: Dict[str, SmallWebRTCConnection] = {}
bot_tasks: Dict[str, asyncio.Task] = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
async def api_status():
    try:
        cfg = VoiceAgentConfig.from_env()
        tts_engine = getattr(cfg, "tts_engine", None)
    except Exception:
        tts_engine = None
    # Probe ONNX daemon briefly
    onnx_engine = None
    try:
        async with httpx.AsyncClient(timeout=0.2) as client:
            r = await client.get("http://127.0.0.1:8770/version")
            if r.status_code == 200:
                onnx_engine = r.json().get("engine")
    except Exception:
        pass
    return {"status": "ok", "tts_engine": tts_engine, "onnx_daemon": onnx_engine}

# Connection monitoring
# If CONNECTION_INACTIVITY_TIMEOUT_SECS is 0 or unset, inactivity cleanup is disabled.
CONNECTION_TIMEOUT = 0  # seconds (only for "connecting" state watchdog)
INACTIVITY_TIMEOUT = float(os.getenv("CONNECTION_INACTIVITY_TIMEOUT_SECS", "0"))
connection_monitor_task = None

async def monitor_connections():
    """Monitor active connections and clean up stuck ones."""
    while True:
        try:
            await asyncio.sleep(5)  # Check every 5 seconds
            current_time = time.time()

            for pc_id, connection in list(pcs_map.items()):
                try:
                    # Check if connection is stuck in connecting state
                    if hasattr(connection, 'connection_state') and connection.connection_state == "connecting":
                        if hasattr(connection, '_connection_start_time'):
                            if current_time - connection._connection_start_time > CONNECTION_TIMEOUT:
                                logger.warning(f"Connection {pc_id} stuck in connecting state for {CONNECTION_TIMEOUT}s, cleaning up")
                                try:
                                    # SmallWebRTCConnection exposes `disconnect()`
                                    await connection.disconnect()
                                except:
                                    pass
                                pcs_map.pop(pc_id, None)

                    # Optional: Check if connection appears dead (no recent activity)
                    # Disabled by default; enable by setting CONNECTION_INACTIVITY_TIMEOUT_SECS > 0
                    if INACTIVITY_TIMEOUT > 0 and hasattr(connection, '_last_activity'):
                        if current_time - connection._last_activity > INACTIVITY_TIMEOUT:
                            logger.warning(f"Connection {pc_id} appears inactive for {INACTIVITY_TIMEOUT:.0f}s, cleaning up")
                            try:
                                await connection.disconnect()
                            except Exception:
                                pass
                            pcs_map.pop(pc_id, None)

                except Exception as e:
                    logger.error(f"Error monitoring connection {pc_id}: {e}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in connection monitor: {e}")

# Offline/local default: host-only ICE (no public STUN)
ice_servers = []


# LocalSmartTurnAnalyzerV3 includes model weights bundled with Pipecat


async def run_bot(transport_or_connection):
    """Run bot with either WebSocket transport or WebRTC connection.

    Args:
        transport_or_connection: Either FastAPIWebsocketTransport or SmallWebRTCConnection
    """
    # Load centralized configuration
    config = VoiceAgentConfig.from_env()
    logger.info(f"Configuration loaded:\n{config.summary()}")

    # Create factory with configuration and preloaded models
    factory = VoiceAgentFactory(config, get_preloaded_models())

    # Build dynamic system prompt based on configuration
    system_instruction = factory.build_system_prompt()
    logger.debug(f"Generated system prompt:\n{system_instruction}")

    # Create all services using factory
    services = factory.create_voice_agent(transport_or_connection, system_instruction)

    # Extract services for event handlers
    transport = services['transport']
    rtvi = services['rtvi']
    memory = services['memory']
    context = services['context']
    task = services['task']
    router = services.get('enrollment_router')
    coordinator = services.get('enrollment_coordinator')

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

        # Initial prompts are handled in on_pipeline_started to ensure StartFrame ordering

        try:
            memory.refresh_session_header()
        except Exception:
            pass

    # IMPORTANT: Initial prompts sent via on_pipeline_started hook (Pipecat-compliant)
    # This ensures frames go to the ACTIVE task, not a dying/cancelled one
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(_, frame):
        try:
            # Check if enrollment coordinator wants to send initial prompt
            if coordinator and coordinator.should_send_initial_prompt():
                # Set router state BEFORE queueing frames
                if router:
                    await router.update_state(EnrollmentState.CHOICE)
                # Use task.queue_frames() to explicitly target THIS pipeline
                prompts = coordinator.get_initial_prompts()
                logger.info(f"[bot] Sending {len(prompts)} initial prompts via task.queue_frames()")
                await task.queue_frames(prompts)
            elif not factory.config.enable_intro_pipeline:
                # Fallback greeting path (no enrollment UX)
                greeting = await get_initial_greeting()
                context.add_message({"role": "assistant", "content": greeting})
                await task.queue_frames([TextFrame(greeting)])
        except Exception as e:
            logger.warning(f"Failed to enqueue initial prompt: {e}")

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.debug("Pipeline runner task cancelled; shutting down bot cleanly")
        try:
            await task.cancel()
        except Exception:
            pass
        raise
    finally:
        # Ensure transport/connection is disconnected on exit
        try:
            # WebRTC connection has disconnect()
            if hasattr(transport_or_connection, 'disconnect'):
                await transport_or_connection.disconnect()
        except Exception:
            pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice agent connection."""
    await websocket.accept()
    logger.info("🔌 WebSocket connection accepted")

    try:
        # Create WebSocket transport with protobuf serialization and VAD
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_out_enabled=True,
                add_wav_header=False,  # Not needed for WebSocket
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),  # Enable voice activity detection for STT
                vad_audio_passthrough=True,
                serializer=ProtobufFrameSerializer()
            )
        )

        # Run bot with WebSocket transport
        await run_bot(transport)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    # Log all /api/offer calls to detect duplicate connection attempts
    logger.info(f"📞 /api/offer called with pc_id={pc_id}. Active connections: {len(pcs_map)}")
    if pcs_map:
        logger.debug(f"   Existing connections: {list(pcs_map.keys())}")

    # CRITICAL: Always clean up old sessions to prevent state bleeding
    if pc_id and pc_id in pcs_map:
        old_connection = pcs_map.pop(pc_id)
        logger.warning(f"⚠️  Duplicate connection detected! Cleaning up old session for pc_id: {pc_id}")
        try:
            await old_connection.disconnect()
        except Exception as e:
            logger.warning(f"Error closing old connection: {e}")

    # Always create fresh connection and pipeline for clean sessions
    pipecat_connection = SmallWebRTCConnection(ice_servers)
    await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

    # Track connection start time for monitoring
    pipecat_connection._connection_start_time = time.time()
    pipecat_connection._last_activity = time.time()

    @pipecat_connection.event_handler("closed")
    async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
        logger.info(f"Connection closed for pc_id: {webrtc_connection.pc_id}")
        pcs_map.pop(webrtc_connection.pc_id, None)
        t = bot_tasks.pop(webrtc_connection.pc_id, None)
        if t and not t.done():
            t.cancel()

    # Run fresh bot instance for this session and track it so we can
    # cancel/await during shutdown.
    async def _start_session():
        try:
            await run_bot(pipecat_connection)
        finally:
            # Remove from maps on exit
            pcs_map.pop(pipecat_connection.pc_id, None)
            bot_tasks.pop(pipecat_connection.pc_id, None)

    task = asyncio.create_task(_start_session(), name=f"bot_session_{pc_id or 'new'}")

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection
    bot_tasks[answer["pc_id"]] = task

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection_monitor_task, _preloaded_models

    # Start connection monitoring
    logger.debug("Starting connection monitor...")
    connection_monitor_task = asyncio.create_task(monitor_connections())

    # ========== PRELOAD MODELS AT STARTUP ==========
    try:
        config = VoiceAgentConfig.from_env()
        _preloaded_models = await PreloadedModels.preload(config)
        logger.info("✅ Server ready with preloaded models - connections will be FAST!")
    except Exception as e:
        logger.error(f"❌ Model preloading failed: {e}")
        logger.warning("⚠️  Server will start anyway, but connections will be slower")
        _preloaded_models = None

    yield  # Run app

    # Cleanup
    logger.debug("Shutting down connection monitor...")
    if connection_monitor_task:
        connection_monitor_task.cancel()
        try:
            await connection_monitor_task
        except asyncio.CancelledError:
            pass

    # Cleanup on shutdown
    # 1) Cancel all running bot tasks
    for t in list(bot_tasks.values()):
        t.cancel()
    if bot_tasks:
        await asyncio.gather(*bot_tasks.values(), return_exceptions=True)
    bot_tasks.clear()

    # 2) Disconnect any leftover peer connections
    if pcs_map:
        await asyncio.gather(*(pc.disconnect() for pc in pcs_map.values()), return_exceptions=True)
        pcs_map.clear()


# Attach explicit lifespan manager so startup/shutdown hooks run
app.router.lifespan_context = lifespan


@app.post("/api/shutdown")
async def api_shutdown():
    """Programmatically request graceful shutdown (used by bundled app)."""
    logger.info("🛑 Shutdown requested via /api/shutdown endpoint")

    async def _delayed_kill():
        await asyncio.sleep(0.3)
        # Use SIGINT so Uvicorn performs graceful shutdown
        os.kill(os.getpid(), signal.SIGINT)

    asyncio.create_task(_delayed_kill())
    return {"status": "shutting_down"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host for HTTP server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    # Let Uvicorn own signal handling and lifespan. Avoid reload in prod/bundle.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False,
        loop="asyncio",
        http="h11",
    )
