import argparse
import asyncio
import os
import re
import signal
import sys
import time
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional

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

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TextFrame
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer

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


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    # Load centralized configuration
    config = VoiceAgentConfig.from_env()
    logger.info(f"Configuration loaded:\n{config.summary()}")

    # Create factory with configuration
    factory = VoiceAgentFactory(config)

    # Build dynamic system prompt based on configuration
    system_instruction = factory.build_system_prompt()
    logger.debug(f"Generated system prompt:\n{system_instruction}")

    # Create all services using factory
    services = factory.create_voice_agent(webrtc_connection, system_instruction)

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
        # Ensure transport is disconnected on exit
        try:
            await webrtc_connection.disconnect()
        except Exception:
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
    global connection_monitor_task

    # Start connection monitoring
    logger.debug("Starting connection monitor...")
    connection_monitor_task = asyncio.create_task(monitor_connections())

    # Pre-warm models on startup
    try:
        from experiments.development_tools.model_manager import initialize_models
        logger.debug("Pre-warming ML models for ultra-low latency...")
        await initialize_models()
        logger.debug("Model pre-warming complete")
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")

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
