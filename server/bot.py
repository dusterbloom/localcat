import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict


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
from hotpath_processor import HotPathMemoryProcessor
from session_tracker import SessionTracker

# Import streaming STT service
try:
    from kyutai_streaming_stt import KyutaiStreamingSTT
    from mic_probe import MicProbe
    KYUTAI_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import KyutaiStreamingSTT: {e}")
    KYUTAI_AVAILABLE = False

from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer

from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

from tts_mlx_ultra_low_latency import TTSMLXUltraLowLatency


load_dotenv(override=True)



async def get_initial_greeting() -> str:
    """Simple greeting for now - HotMem will provide memory context."""
    return "Hello! How can I help you today?"

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

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
    vad_confidence = float(os.getenv("VAD_CONFIDENCE", "0.7"))
    vad_start_secs = float(os.getenv("VAD_START_SECS", "0.2"))
    # Use a more forgiving default stop window so brief pauses do not end the turn
    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "1.6"))
    vad_min_volume = float(os.getenv("VAD_MIN_VOLUME", "0.6"))

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

    # STT: Kyutai streaming by default, Whisper MLX as fallback
    use_streaming_stt = os.getenv("USE_STREAMING_STT", "true").lower() == "true"

    if use_streaming_stt and KYUTAI_AVAILABLE:
        try:
            hf_repo = os.getenv("KYUTAI_STT_REPO", "kyutai/stt-1b-en_fr-mlx")
            enable_vad = os.getenv("KYUTAI_ENABLE_VAD", "false").lower() in ("1", "true", "yes")
            max_steps = int(os.getenv("KYUTAI_MAX_STEPS", "4096"))

            logger.info(f"Initializing Kyutai streaming STT with repo: {hf_repo}")
            stt = KyutaiStreamingSTT(
                hf_repo=hf_repo,
                enable_vad=enable_vad,
                max_steps=max_steps
            )
            logger.info(f"✅ Kyutai streaming STT initialized ({'MLX' if hf_repo.endswith('-mlx') else 'Candle'})")
        except Exception as e:
            logger.error(f"❌ Kyutai STT failed: {e}", exc_info=True)
            logger.warning("Falling back to Whisper MLX batch mode")
            stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)
    else:
        if not KYUTAI_AVAILABLE:
            logger.warning("Kyutai STT not available")
        else:
            logger.info("Streaming STT disabled via USE_STREAMING_STT=false")
        logger.info("Using Whisper MLX batch mode (multilingual support)")
        stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)

    # Ultra-low latency TTS with optimized streaming
    logger.info("Using ultra-low latency TTS mode (40-80ms TTFB)")

    # Get buffer size from environment (default 80ms for stability)
    buffer_ms = int(os.getenv("KOKORO_BUFFER_MS", "80"))

    tts = TTSMLXUltraLowLatency(
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_heart",
        sample_rate=24000,
        speed=1.0,
        buffer_ms=buffer_ms,
        use_boundaries=True
    )

    try:
        await tts._initialize_if_needed()
    except Exception as e:
        logger.warning(f"TTS prewarm failed: {e}")



    # Enable LLM streaming for lower perceived latency
    use_llm_streaming = os.getenv("USE_LLM_STREAMING", "true").lower() == "true"

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL"),  # Small model. Uses ~4GB of RAM.
        # model="google/gemma-3-12b",  # Medium-sized model. Uses ~8.5GB of RAM.
        # model="mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ", # Large model. Uses ~110GB of RAM!
        base_url=os.getenv("OPENAI_BASE_URL"),
        max_tokens=4096,
        stream=use_llm_streaming,  # Enable streaming for faster response
        extra_body={
            "think": False,  # Disable thinking for main conversation model
            "stream": use_llm_streaming,  # Ensure streaming at API level
            "options": {  # Ollama-specific optimizations
                "num_predict": 4096,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 4096,
                "num_batch": 512,
                "use_mlock": True,
                "f16_kv": True
            }
        },
    )

    if use_llm_streaming:
        logger.info("LLM streaming enabled for lower latency")
    else:
        logger.info("LLM streaming disabled, using batch mode")

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            }
        ]

    )
    default_timeout = "0.3" if use_streaming_stt else "0.25"
    agg_timeout = float(os.getenv("LLM_AGGREGATION_TIMEOUT", default_timeout))
    turn_timeout = float(os.getenv("LLM_TURN_EMULATED_VAD_TIMEOUT", "0.7"))
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
        logger.info("MicProbe enabled: logging mic input levels")
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
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm models on startup
    try:
        from model_manager import initialize_models
        logger.info("Pre-warming ML models for ultra-low latency...")
        await initialize_models()
        logger.info("Model pre-warming complete")
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")

    yield  # Run app

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
