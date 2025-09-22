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




from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel

from pipecat.frames.frames import LLMRunFrame

# Import HotMem processor
from core.memory.hotpath_processor import HotPathMemoryProcessor
from core.memory.session_tracker import SessionTracker
from config import VoiceAgentConfig

# Import streaming STT service
try:
    from core.stt.parakeet_streaming import ParakeetStreamingSTT
    from mic_probe import MicProbe
    PARAKEET_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import ParakeetStreamingSTT: {e}")
    PARAKEET_AVAILABLE = False

from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
# Simple text aggregator for faster TTS response
from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TextFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FastTextAggregator(FrameProcessor):
    """Token-aware text aggregator optimized for Kokoro TTS.

    Releases text at natural phoneme boundaries for fluent speech,
    similar to LiveKit's Kokoro implementation.
    """

    def __init__(self, min_tokens: int = 175, max_tokens: int = 250, max_time: float = 0.5):
        super().__init__()
        self._min_tokens = min_tokens  # TARGET_MIN_TOKENS equivalent
        self._max_tokens = max_tokens  # TARGET_MAX_TOKENS equivalent
        self._max_time = max_time  # Fallback timeout
        self._aggregation = ""
        self._timer = None
        self._last_release_time = asyncio.get_event_loop().time()
        # Sentence ending patterns
        self._sentence_endings = {'.', '!', '?', '。', '！', '？'}
        self._clause_endings = {',', ';', ':', '，', '；', '：'}
    async def _release_text(self):
        """Release accumulated text to TTS."""
        if self._aggregation.strip():
            # Clean and format text for better TTS
            clean_text = self._clean_text_for_tts(self._aggregation)
            if clean_text:
                await self.push_frame(TextFrame(clean_text))

        self._aggregation = ""
        self._last_release_time = asyncio.get_event_loop().time()

        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean and format text for better TTS output."""
        import re
        from tools.text_formatter import sanitize_for_voice

        # Remove markdown formatting but preserve spacing
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold** but keep text
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *italic* but keep text
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Remove `code` but keep text

        # Remove emojis and problematic characters for TTS
        text = sanitize_for_voice(text)

        # Clean up extra whitespace but preserve sentence structure
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()

        # Ensure proper spacing around punctuation for natural speech
        # Add space after colon if followed by word (but not if it's a time like "1:30")
        text = re.sub(r':(?=\w)', ': ', text)

        # Add space after semicolon if followed by word
        text = re.sub(r';(?=\w)', '; ', text)

        # Clean up any double spaces that might have been created
        text = re.sub(r'\s+', ' ', text)

        return text

    async def _delayed_release(self):
        """Release text after timeout."""
        try:
            await asyncio.sleep(self._max_time)
            if self._aggregation.strip():
                await self._release_text()
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            return

        # Handle interruptions
        if isinstance(frame, (CancelFrame, InterruptionFrame, BotInterruptionFrame)):
            self._aggregation = ""
            if self._timer and not self._timer.done():
                self._timer.cancel()
                self._timer = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            # Clean asterisks from incoming text
            clean_text = frame.text.replace('*', '')
            self._aggregation += clean_text

            # Estimate token count (rough approximation: 1 token ≈ 4 chars for English)
            estimated_tokens = len(self._aggregation) // 4

            # Check for natural boundaries
            should_release = False

            # Check if we hit a sentence ending
            if self._aggregation.rstrip() and self._aggregation.rstrip()[-1] in self._sentence_endings:
                # Always release at sentence boundaries if we have minimum content
                if estimated_tokens >= self._min_tokens // 2:  # Half minimum for sentence ends
                    should_release = True
            # Check if we hit max token limit - but try to find a good break point
            elif estimated_tokens >= self._max_tokens:
                # Look for the last good break point (sentence or clause ending)
                text = self._aggregation.rstrip()
                last_sentence_idx = -1
                last_clause_idx = -1

                # Find last sentence boundary
                for i in range(len(text) - 1, -1, -1):
                    if text[i] in self._sentence_endings:
                        last_sentence_idx = i
                        break
                    elif text[i] in self._clause_endings:
                        if last_clause_idx == -1:
                            last_clause_idx = i

                # Prefer sentence boundary, then clause, then word boundary
                if last_sentence_idx > len(text) // 2:  # Found sentence boundary in second half
                    self._aggregation = text[:last_sentence_idx + 1]
                    should_release = True
                elif last_clause_idx > len(text) // 2:  # Found clause boundary in second half
                    self._aggregation = text[:last_clause_idx + 1]
                    should_release = True
                else:
                    # Force release at word boundary to avoid cutting mid-word
                    last_space = text.rfind(' ')
                    if last_space > len(text) // 2:
                        self._aggregation = text[:last_space]
                    should_release = True
            # Check if we have enough tokens and hit a clause boundary
            elif estimated_tokens >= self._min_tokens:
                if self._aggregation.rstrip() and self._aggregation.rstrip()[-1] in self._clause_endings:
                    should_release = True

            if should_release:
                await self._release_text()
            else:
                # Schedule release after timeout as fallback
                if self._timer and not self._timer.done():
                    self._timer.cancel()
                self._timer = asyncio.create_task(self._delayed_release())

        elif isinstance(frame, EndFrame):
            if self._aggregation.strip():
                await self._release_text()
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._timer and not self._timer.done():
            self._timer.cancel()
        await super().cleanup()
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer

from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

# TTS services imported from core directory
from core.tts.kokoro_professional import ProfessionalKokoroTTSService
from core.tts.kokoro_mlx import MLXKokoroTTSService

# Import legacy TTS services for backward compatibility
try:
    from fastapi_streaming_tts import FastAPIStreamingTTS
    FASTAPI_TTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"FastAPI TTS not available: {e}")
    FASTAPI_TTS_AVAILABLE = False


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

    # VAD configuration with backward compatibility
    vad_confidence = float(os.getenv("VAD_CONFIDENCE", "0.5"))
    vad_start_secs = float(os.getenv("VAD_START_SECS", "0.1"))
    # Use a more forgiving default stop window so brief pauses do not end the turn
    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "4.0"))
    vad_min_volume = float(os.getenv("VAD_MIN_VOLUME", "0.4"))

    vad_params = VADParams(
        confidence=vad_confidence,
        start_secs=vad_start_secs,
        stop_secs=max(vad_stop_secs, 0.8),
        min_volume=vad_min_volume,
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
            turn_analyzer=LocalSmartTurnAnalyzerV3(
                params=SmartTurnParams(
                    stop_secs=float(os.getenv("SMART_TURN_STOP_SECS", "4.0")),
                    pre_speech_ms=float(os.getenv("SMART_TURN_PRE_SPEECH_MS", "300")),
                    max_duration_secs=float(os.getenv("SMART_TURN_MAX_DURATION_SECS", "16.0")),
                )
            ),
        ),
    )

    # STT: Use centralized configuration with Parakeet streaming as default
    stt_config = config.get_component_config("stt")

    logger.debug(f"STT Engine: {config.stt_engine}, PARAKEET_AVAILABLE: {PARAKEET_AVAILABLE}")

    if config.stt_engine == "parakeet_streaming" and PARAKEET_AVAILABLE:
        try:
            # Use config defaults for Parakeet-specific settings
            model_path = stt_config.get("model", "mlx-community/parakeet-tdt-0.6b-v3")
            language = stt_config.get("language", "en")
            chunk_duration = float(os.getenv("PARAKEET_CHUNK_DURATION", "1.0"))
            enable_vad = os.getenv("PARAKEET_ENABLE_VAD", "false").lower() in ("1", "true", "yes")

            logger.debug(f"Initializing Parakeet streaming STT with model: {model_path}")
            stt = ParakeetStreamingSTT(
                model_path=model_path,
                language=language,
                chunk_duration=chunk_duration,
                enable_vad=enable_vad
            )
            logger.info("✅ Parakeet streaming STT ready")
        except Exception as e:
            logger.error(f"❌ Parakeet STT failed: {e}", exc_info=True)
            logger.warning("Falling back to Whisper MLX batch mode")
            stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
    elif config.stt_engine == "whisper_mlx":
        logger.debug("Using Whisper MLX batch mode (backup)")
        stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
    else:
        if not PARAKEET_AVAILABLE:
            logger.error("Parakeet STT not available, using Whisper MLX fallback")
        else:
            logger.warning(f"Unknown STT engine: {config.stt_engine}, using Whisper MLX fallback")
        stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)

    # TTS: Use centralized configuration with professional Kokoro as default
    tts_config = config.get_component_config("tts")

    if config.tts_engine == "kokoro_professional":
        logger.debug("Using Professional Kokoro TTS (default optimized)")
        tts = ProfessionalKokoroTTSService(
            voice=tts_config["voice"],
            speed=tts_config["speed"],
            sample_rate=tts_config["sample_rate"],
            fade_duration_ms=tts_config["fade_duration_ms"],
            target_peak_db=tts_config["target_peak_db"],
            enable_quality_logging=tts_config["enable_quality_logging"]
        )
        logger.info("✅ Professional Kokoro TTS ready")
    elif config.tts_engine == "kokoro_mlx":
        logger.debug("Using Ultra-Low Latency MLX Kokoro TTS")
        from tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency
        tts = TTSMLXUltraLowLatency(
            model="mlx-community/Kokoro-82M-bf16",
            voice=tts_config["voice"],
            speed=tts_config["speed"],
            sample_rate=tts_config["sample_rate"],
            buffer_ms=50  # 50ms buffer for optimal latency
        )
        logger.info("✅ Ultra-Low Latency MLX Kokoro TTS ready")
    elif config.tts_engine == "fastapi_streaming" and FASTAPI_TTS_AVAILABLE:
        logger.debug("Using FastAPI Streaming TTS (legacy)")
        tts = FastAPIStreamingTTS(
            voice=tts_config["voice"],
            speed=tts_config["speed"],
            sample_rate=tts_config["sample_rate"],
            socket_path="/tmp/fastapi-tts.sock"
        )
        logger.info("✅ FastAPI Streaming TTS ready")
    else:
        logger.warning(f"Unknown TTS engine: {config.tts_engine}, falling back to MLX Kokoro")
        tts = MLXKokoroTTSService(
            voice=tts_config["voice"],
            speed=tts_config["speed"],
            sample_rate=tts_config["sample_rate"]
        )
        logger.info("✅ MLX Kokoro TTS ready (fallback)")



    # LLM: Use centralized configuration
    llm_config = config.get_component_config("llm")
    use_llm_streaming = os.getenv("USE_LLM_STREAMING", "true").lower() == "true"

    llm = OpenAILLMService(
        api_key=llm_config["api_key"],
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        max_tokens=llm_config["max_tokens"],
        stream=use_llm_streaming,  # Enable streaming for faster response
        extra_body={
            "think": False,  # Disable thinking for main conversation model
            "stream": use_llm_streaming,  # Ensure streaming at API level
            "options": {  # Ollama-specific optimizations tuned for latency
                "num_predict": 768,
                "temperature": llm_config["temperature"],
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,
                "num_batch": 64,
                "use_mlock": True,
                "f16_kv": True,
                "keep_alive": "15m"
            }
        },
    )

    if use_llm_streaming:
        logger.debug("LLM streaming enabled for lower latency")
    else:
        logger.debug("LLM streaming disabled, using batch mode")

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            }

        ]
    )

    # Determine if we're using streaming STT for timeout optimization
    use_streaming_stt = config.stt_engine in ["parakeet_streaming"]
    default_timeout = "0.12" if use_streaming_stt else "0.2"
    agg_timeout = float(os.getenv("LLM_AGGREGATION_TIMEOUT", default_timeout))
    turn_timeout = float(os.getenv("LLM_TURN_EMULATED_VAD_TIMEOUT", "0.4"))
    agg_interruptions = os.getenv("LLM_ENABLE_EMULATED_VAD_INTERRUPTION", "true").lower() in ("1", "true", "yes")

    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(
            aggregation_timeout=agg_timeout,
            turn_emulated_vad_timeout=turn_timeout,
            enable_emulated_vad_interruptions=agg_interruptions,
        ),
    )

    session_tracker = SessionTracker()

    # Initialize HotMem ultra-fast memory processor with context aggregator
    memory = HotPathMemoryProcessor(
        sqlite_path=os.getenv("HOTMEM_SQLITE", ":memory:"),  # Use in-memory database for now
        lmdb_dir=os.getenv("HOTMEM_LMDB_DIR", None),  # Disable LMDB temporarily
        user_id=os.getenv("USER_ID", "default-user"),
        enable_metrics=True,  # Log performance metrics
        context_aggregator=context_aggregator,  # Pass context aggregator for injection
        session_tracker=session_tracker,
        agent_id=os.getenv("AGENT_ID", "locat"),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    stages = [transport.input()]

    # Optional mic probe
    if os.getenv("ENABLE_MIC_PROBE", "false").lower() in ("1", "true", "yes"):
        logger.debug("MicProbe enabled: logging mic input levels")
        stages.append(MicProbe())

 
    stages += [
        stt,
        rtvi,
        memory,  # Move HotMem BEFORE context_aggregator so it sees TranscriptionFrames
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]

    pipeline = Pipeline(stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        
        # Get greeting
        greeting = await get_initial_greeting()

        # Add the greeting as an assistant message to start the conversation
        context.add_message({"role": "assistant", "content": greeting})
        
        # Send greeting directly to TTS without triggering LLM
        from pipecat.frames.frames import TextFrame
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
