import argparse
import asyncio
import os
import re
import sys
import time
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional


# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger




from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from config import VoiceAgentConfig
from core.factory import VoiceAgentFactory
from pipecat.frames.frames import TextFrame
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)



async def get_initial_greeting() -> str:
    """Simple greeting for now - HotMem will provide memory context."""
    return "Hello! How can I help you today?"

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

# Connection monitoring
CONNECTION_TIMEOUT = 30  # seconds
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
                                    await connection.close()
                                except:
                                    pass
                                pcs_map.pop(pc_id, None)

                    # Check if connection appears dead (no recent activity)
                    if hasattr(connection, '_last_activity'):
                        if current_time - connection._last_activity > CONNECTION_TIMEOUT * 2:
                            logger.warning(f"Connection {pc_id} appears inactive, cleaning up")
                            try:
                                await connection.close()
                            except:
                                pass
                            pcs_map.pop(pc_id, None)

                except Exception as e:
                    logger.error(f"Error monitoring connection {pc_id}: {e}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in connection monitor: {e}")

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


# LocalSmartTurnAnalyzerV3 includes model weights bundled with Pipecat


SYSTEM_INSTRUCTION =  """You are Locat, a personal assistant. You can remember things about the person you are talking to.
                        Some Guidelines:
                        - Make sure your responses are friendly yet short and concise.
                        - If the user asks you to remember something, make sure to remember it.
                        - Greet the user by their name if you know about it. 
                    """


async def run_bot(webrtc_connection):
    # Load centralized configuration
    config = VoiceAgentConfig.from_env()
    logger.info(f"Configuration loaded:\n{config.summary()}")

    # Create factory with configuration
    factory = VoiceAgentFactory(config)

    # Create all services using factory
    services = factory.create_voice_agent(webrtc_connection, SYSTEM_INSTRUCTION)

    # Extract services for event handlers
    transport = services['transport']
    rtvi = services['rtvi']
    memory = services['memory']
    context = services['context']
    task = services['task']

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

        # Get greeting
        greeting = await get_initial_greeting()

        # Add the greeting as an assistant message to start the conversation
        context.add_message({"role": "assistant", "content": greeting})

        # Send greeting directly to TTS without triggering LLM
        await task.queue_frames([TextFrame(greeting)])

        try:
            memory.refresh_session_header()
        except Exception:
            pass

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.debug(f"Reusing existing connection for pc_id: {pc_id}")

        # Only renegotiate if explicitly requested AND connection is actually broken
        # The voice-ui-kit sends restart_pc=true by default, so we ignore it for healthy connections
        restart_requested = request.get("restart_pc", False)

        # Check if connection appears healthy before deciding to renegotiate
        connection_healthy = (
            hasattr(pipecat_connection, 'connection_state') and
            pipecat_connection.connection_state == "connected" and
            hasattr(pipecat_connection, '_last_activity') and
            time.time() - pipecat_connection._last_activity < 60  # Active in last minute
        )

        if restart_requested and not connection_healthy:
            logger.debug(f"Connection appears broken, renegotiating pc_id: {pc_id}")
            await pipecat_connection.renegotiate(
                sdp=request["sdp"],
                type=request["type"],
                restart_pc=True,
            )
        else:
            # For normal conversation flow, just update the connection without renegotiation
            logger.debug(f"Updating SDP for existing connection pc_id: {pc_id} (restart_pc={restart_requested}, healthy={connection_healthy})")
            try:
                # Try to update the connection without full renegotiation
                await pipecat_connection.set_remote_description(request["sdp"], request["type"])
                # Update activity timestamp
                pipecat_connection._last_activity = time.time()
                logger.debug(f"SDP updated successfully for pc_id: {pc_id}")
            except Exception as e:
                logger.warning(f"SDP update failed, falling back to renegotiation: {e}")
                await pipecat_connection.renegotiate(
                    sdp=request["sdp"],
                    type=request["type"],
                    restart_pc=False,
                )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        # Track connection start time for monitoring
        pipecat_connection._connection_start_time = time.time()
        pipecat_connection._last_activity = time.time()

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.debug(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

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
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
