import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Union


# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger




from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
#from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.turn.smart_turn.local_coreml_smart_turn import LocalCoreMLSmartTurnAnalyzer

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel

from pipecat.frames.frames import LLMRunFrame

# Use Mem0ServiceV2 - lets mem0 work as designed
from mem0_service_v2 import Mem0ServiceV2 as Mem0MemoryService

from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer

from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

from tts_mlx_isolated import TTSMLXIsolated


# Setup Mem0 memory service local configuration



load_dotenv(override=True)

try:
    from mem0 import Memory, MemoryClient  # noqa: F401
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Mem0, you need to `pip install mem0ai`. Also, set the environment variable MEM0_API_KEY."
    )
    raise Exception(f"Missing module: {e}")



async def get_initial_greeting(
    memory_client: Union[MemoryClient, Memory], user_id: str, agent_id: str, run_id: str
) -> str:
    """Fetch all memories for the user and create a personalized greeting.

    Returns:
        A personalized greeting based on user memories
    """
    try:
        if isinstance(memory_client, Memory):
            filters = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}
            filters = {k: v for k, v in filters.items() if v is not None}
            memories = memory_client.get_all(**filters)
        else:
            # Create filters based on available IDs
            id_pairs = [("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)]
            clauses = [{name: value} for name, value in id_pairs if value is not None]
            filters = {"AND": clauses} if clauses else {}

            # Get all memories for this user
            memories = memory_client.get_all(filters=filters, version="v2", output_format="v1.1")

        if not memories or len(memories) == 0:
            logger.debug(f"!!! No memories found for this user. {memories}")
            return "Hello! It's nice to meet you. How can I help you today?"

        # Create a personalized greeting based on memories
        greeting = "Hello! It's great to see you again. "

        # Add some personalization based on memories (limit to 3 memories for brevity)
        # if len(memories) > 0:
        #     greeting += "Based on our previous conversations, I remember: "
        #     for i, memory in enumerate(memories["results"][:3], 1):
        #         memory_content = memory.get("memory", "")
        #         # Keep memory references brief
        #         if len(memory_content) > 100:
        #             memory_content = memory_content[:97] + "..."
        #         greeting += f"{memory_content} "

        #     greeting += "How can I help you today?"

        logger.debug(f"Created personalized greeting from {len(memories)} memories")
        return greeting

    except Exception as e:
        logger.error(f"Error retrieving initial memories from Mem0: {e}")
        return "Hello! How can I help you today?"

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


smart_turn_model_path = os.getenv("LOCAL_SMART_TURN_MODEL_PATH")


SYSTEM_INSTRUCTION =  """You are Locat, a personal assistant. You can remember things about the person you are talking to.
                        Some Guidelines:
                        - Make sure your responses are friendly yet short and concise.
                        - If the user asks you to remember something, make sure to remember it.
                        - Greet the user by their name if you know about it. 
                    """


async def run_bot(webrtc_connection):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalCoreMLSmartTurnAnalyzer(
                smart_turn_model_path=smart_turn_model_path,  # Download from HuggingFace
                params=SmartTurnParams(
                stop_secs=2.0,  # Shorter stop time when using Smart Turn
                pre_speech_ms=0.0,
                max_duration_secs=20.0
            )
            ),
        ),
    )

    stt = WhisperSTTServiceMLX(model=MLXModel.MEDIUM)

    tts = TTSMLXIsolated(model="mlx-community/Kokoro-82M-bf16", voice="af_heart", sample_rate=24000)
    # tts = TTSMLXIsolated(model="Marvis-AI/marvis-tts-250m-v0.1", voice=None)

    # Add random context length variation to prevent LM Studio context caching issues
    import random
    context_length_variation = random.randint(-200, 200)  # ±200 tokens for more variation
    base_max_tokens = 3000  # Much higher to utilize 8k context window
    varied_max_tokens = max(500, base_max_tokens + context_length_variation)  # Ensure minimum 500
    
    local_config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MEM0_MODEL"),  # Single LM Studio model for all mem0 operations
                "api_key": "lm-studio",  # LM Studio doesn't need real API key
                "openai_base_url": os.getenv("MEM0_BASE_URL"),  # LM Studio endpoint
                "max_tokens": varied_max_tokens,  # Varied tokens to prevent context caching
                "temperature": 0.1  # Low temperature for consistent JSON output
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("EMBEDDING_MODEL"),
                "api_key": "not-needed",  # Ollama doesn't need API key
                "openai_base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")  # Ollama endpoint for embeddings
            }
        },
        "vector_store": {
            "provider": "faiss",
                "config": {
                    "collection_name": "memory",
                    "embedding_model_dims":768, # hard coded to match the embedding dim of the `nomic-embed-text` model

                    "path": os.getenv("MEMORY_FAISS_PATH"),
                    "distance_strategy": "cosine"  # Cosine similarity works better for embeddings than euclidean
                }   
        }
    }

    # Initialize Qwen3-optimized Mem0 memory service with proper frame processing
    memory = Mem0MemoryService(
        local_config=local_config,  # Use local LLM for memory processing
        user_id=os.getenv("USER_ID"),            # Unique identifier for the user
        agent_id=os.getenv("AGENT_ID"),     # Optional identifier for the agent
        # run_id="session1",        # Optional identifier for the run
        params=Mem0MemoryService.InputParams(
            search_limit=5,  # Limit memory retrieval for speed
            search_threshold=0.2,  # Threshold for relevant memories
            api_version="v2",
            system_prompt="Based on previous conversations, I recall: \n\n",
            add_as_system_message=True,
            position=1,
        )
    )


    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL"),  # Small model. Uses ~4GB of RAM.
        # model="google/gemma-3-12b",  # Medium-sized model. Uses ~8.5GB of RAM.
        # model="mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ", # Large model. Uses ~110GB of RAM!
        base_url=os.getenv("OPENAI_BASE_URL"), 
        max_tokens=4096,
        extra_body={"think": False},  # Disable thinking for main conversation model
    )

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            }
        ]

    )
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # a de minimus value since we don't expect any transcript aggregation to be
        # necessary.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            memory,
            llm,
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
        
        # Get personalized greeting based on user memories. Can pass agent_id and run_id as per requirement of the application to manage short term memory or agent specific memory.
        greeting = await get_initial_greeting(
            memory_client=memory.memory_client, user_id=os.getenv("USER_ID"), agent_id=os.getenv("AGENT_ID"), run_id=None
        )

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
