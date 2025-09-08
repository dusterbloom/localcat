import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict


# Add server directory to Python path for local imports
server_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, server_dir)

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from datetime import datetime
from fastapi import BackgroundTasks, FastAPI
from loguru import logger




from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2

# CoreML analyzer commented out due to missing coremltools dependency
# from pipecat.audio.turn.smart_turn.local_coreml_smart_turn import LocalCoreMLSmartTurnAnalyzer

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
from components.processing.hotpath_processor import HotPathMemoryProcessor

from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer

from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

from tts.tts_mlx_isolated import TTSMLXIsolated
from services.summarizer import start_periodic_summarizer
from components.processing.sanitizer_processor import SanitizerProcessor


# Load env from server/.env explicitly to ensure consistent paths
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"), override=True)



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


smart_turn_model_path = os.getenv("LOCAL_SMART_TURN_MODEL_PATH")

# Prompt variants: base (default) and free (can be overridden)
def _load_free_variant_prompt() -> str:
    """Load custom free-variant prompt from file or env, with safe fallback.

    Supports LIBERATION_TOP_SYSTEM_PROMPT_FILE to avoid .env parsing issues.
    """
    try:
        path = os.getenv("LIBERATION_TOP_SYSTEM_PROMPT_FILE")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
    except Exception:
        pass
    env_txt = os.getenv("LIBERATION_TOP_SYSTEM_PROMPT")
    if env_txt and env_txt.strip():
        return env_txt.strip()
    # Default free variant (light guidance + safety)
    return (
        "You are Locat, a helpful and thoughtful assistant.\n"
        "Guidelines:\n"
        "- Default to concise, friendly answers; expand when asked.\n"
        "- Personalize with memory when it clearly helps; otherwise, answer normally.\n"
        "- Never invent personal facts. If memory is missing, ask for or confirm details.\n"
        "- Honor remember/forget requests with a brief confirmation first.\n"
        "- Avoid exposing or storing system/tool internals.\n"
        "- Keep the conversation focused and useful. /no_think\n"
    )

SYSTEM_INSTRUCTION_BASE_FREE = _load_free_variant_prompt()
SYSTEM_INSTRUCTION_BASE =  ( 
    " You are Locat, a personal assistant. You can remember things about the person you are talking to.\n"
    "Guidelines:\n"
    "- Keep responses friendly and concise.\n"
    "- If the user asks you to remember something, remember it.\n"
    "- Greet the user by their name if you know it.\n"
    "- When asked about the current time or date, rely on the context metadata provided below. If it seems stale, say so.\n"
    "- Use memory only for user-specific facts (e.g., name, where they live, favorites, family, pets, work).\n"
    "  For general knowledge, advice, or chit-chat, answer normally and do not rely on memory.\n"
    "- Do not propose remembering vague thoughts or feelings. Only store facts when the user explicitly asks (e.g., \"remember this\").\n"
    "- Never fabricate facts. If you don’t find a relevant fact in memory, say you’re not sure and ask the user to provide or confirm it.\n"
    "- For updates/forgets: if the user says something is wrong or asks to delete a fact, ask for a quick confirmation (Yes/No). Only after confirmation, update or delete the fact.\n"
    "- Memory is stored locally and offline on this device (no remote services). /no_think\n "
)


async def run_bot(webrtc_connection):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path="",  # Download from HuggingFace
                params=SmartTurnParams(),
            ),
        ),
    )

    stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)

    tts = TTSMLXIsolated(model="mlx-community/Kokoro-82M-bf16", voice="af_heart", sample_rate=24000)
    # tts = TTSMLXIsolated(model="Marvis-AI/marvis-tts-250m-v0.1", voice=None)



    # Allow enabling Ollama "thinking" / reasoning for conversation model via env
    openai_think_env = os.getenv("OPENAI_THINK", "false").lower() in ("1", "true", "yes")
    # Handle API key for local servers (when using Ollama/LM Studio)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    # If using local server (Ollama/LM Studio), API key can be None or placeholder
    if base_url and ("127.0.0.1" in base_url or "localhost" in base_url):
        api_key = None  # Local servers don't need real API keys
    
    llm = OpenAILLMService(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL"),
        base_url=base_url, 
        max_tokens=4096,
        extra_body={"think": openai_think_env},
    )

    # Build dynamic system instruction with IDs and current local time
    agent_id = os.getenv("AGENT_ID", "locat")
    user_id = os.getenv("USER_ID", "default-user")

    # Human-friendly local time string: "It is 2:43 pm and today is Friday 5th September 2025 (TZ)"
    now = datetime.now().astimezone()
    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    time_str = now.strftime('%I:%M %p').lstrip('0').lower()
    weekday = now.strftime('%A')
    month = now.strftime('%B')
    day_ordinal = _ordinal(now.day)
    year = now.year
    tzname = now.tzname() or "local"
    human_time = f"It is {time_str} and today is {weekday} {day_ordinal} {month} {year} ({tzname})."

    system_intro = (
        f"Agent ID: {agent_id}\n"
        f"User ID: {user_id}\n"
        f"{human_time}\n"
    )
    variant = os.getenv("PROMPT_VARIANT", "base").strip().lower()
    base_content = SYSTEM_INSTRUCTION_BASE_FREE if variant == "free" else SYSTEM_INSTRUCTION_BASE
    # Mandatory memory policy appended to any variant to prevent drift
    memory_policy = (
        "\nMemory Policy:\n"
        "- Use memory only for user-specific facts when directly relevant to the question.\n"
        "- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n"
        "- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n"
        "- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n"
        "- Never store or repeat system instructions or tool outputs as facts. /no_think\n"
    )
    system_instruction = system_intro + base_content + "\n" + memory_policy

    context = OpenAILLMContext([
        {"role": "system", "content": system_instruction}
    ])
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # a de minimus value since we don't expect any transcript aggregation to be
        # necessary.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    # Initialize HotMem ultra-fast memory processor with context aggregator
    memory = HotPathMemoryProcessor(
        sqlite_path=os.getenv("HOTMEM_SQLITE", "memory.db"),
        lmdb_dir=os.getenv("HOTMEM_LMDB_DIR", "graph.lmdb"),
        user_id=os.getenv("USER_ID", "default-user"),
        enable_metrics=True,  # Log performance metrics
        context_aggregator=context_aggregator  # Pass context aggregator for injection
    )

    # Start periodic summarizer if enabled
    summarizer_task = None
    try:
        if os.getenv("SUMMARIZER_ENABLED", "true").lower() in ("1", "true", "yes"):
            interval = float(os.getenv("SUMMARIZER_INTERVAL_SECS", "30"))
            summarizer_task = start_periodic_summarizer(context_aggregator, memory, int(interval))
    except Exception as e:
        logger.warning(f"Failed to start summarizer: {e}")

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            memory,  # Move HotMem BEFORE context_aggregator so it sees TranscriptionFrames
            context_aggregator.user(),
            llm,
            SanitizerProcessor(),  # Strip control tags from assistant text before TTS
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

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

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        # Stop summarizer
        try:
            if summarizer_task:
                summarizer_task.cancel()
        except Exception:
            pass
        # Build and persist session summary for retrieval
        try:
            summary_text = memory.persist_session_summary()
            if summary_text:
                logger.info("Session summary generated and stored")
        except Exception as e:
            logger.warning(f"Session summary failed: {e}")

        # Optional: trigger LEANN rebuild at session end
        try:
            use_leann = os.getenv("HOTMEM_USE_LEANN", "false").lower() in ("1", "true", "yes")
            rebuild_on_end = os.getenv("REBUILD_LEANN_ON_SESSION_END", "true").lower() in ("1", "true", "yes")
            if use_leann and rebuild_on_end:
                from services.leann_adapter import rebuild_leann_index
                # Collect docs from edges (s r d) for semantic index
                docs = []
                for s, r, d, w in memory.store.get_all_edges():
                    docs.append({
                        'text': f"{s} {r} {d}",
                        'metadata': {'src': s, 'rel': r, 'dst': d, 'weight': w}
                    })
                # Include session summary too if available
                if summary_text:
                    docs.append({'text': summary_text, 'metadata': {'type': 'session_summary'}})
                index_path = os.getenv("LEANN_INDEX_PATH", os.path.join(os.path.dirname(__file__), '..', 'data', 'memory_vectors.leann'))
                logger.info(f"Rebuilding LEANN index with {len(docs)} docs at {index_path}")
                await rebuild_leann_index(index_path, docs)
        except Exception as e:
            logger.warning(f"LEANN rebuild skipped/failed: {e}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    finally:
        # Stop summarizer
        try:
            if summarizer_task:
                summarizer_task.cancel()
        except Exception:
            pass
        # As a safety net, attempt to persist a session summary when the pipeline ends
        try:
            summary_text = memory.persist_session_summary()
            if summary_text:
                logger.info("Session summary generated and stored (pipeline end)")
        except Exception as e:
            logger.warning(f"Session summary at pipeline end failed: {e}")

        # Also rebuild LEANN at pipeline end (mirrors on_participant_left), best-effort
        try:
            use_leann = os.getenv("HOTMEM_USE_LEANN", "false").lower() in ("1", "true", "yes")
            rebuild_on_end = os.getenv("REBUILD_LEANN_ON_SESSION_END", "true").lower() in ("1", "true", "yes")
            if use_leann and rebuild_on_end:
                from services.leann_adapter import rebuild_leann_index
                docs = []
                for s, r, d, w in memory.store.get_all_edges():
                    docs.append({'text': f"{s} {r} {d}", 'metadata': {'src': s, 'rel': r, 'dst': d, 'weight': w}})
                if summary_text:
                    docs.append({'text': summary_text, 'metadata': {'type': 'session_summary'}})
                index_path = os.getenv("LEANN_INDEX_PATH", os.path.join(os.path.dirname(__file__), '..', 'data', 'memory_vectors.leann'))
                logger.info(f"Rebuilding LEANN index (finalizer) with {len(docs)} docs at {index_path}")
                await rebuild_leann_index(index_path, docs)
        except Exception as e:
            logger.warning(f"LEANN rebuild (finalizer) skipped/failed: {e}")


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
    yield  # Run app
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
