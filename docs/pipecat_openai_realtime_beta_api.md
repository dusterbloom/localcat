Speech-to-Speech
OpenAI Realtime Beta

Copy page

Real-time speech-to-speech service implementation using OpenAI’s Realtime Beta API

OpenAIRealtimeBetaLLMService provides real-time, multimodal conversation capabilities using OpenAI’s Realtime Beta API. It supports speech-to-speech interactions with integrated LLM processing, function calling, and advanced conversation management.
Real-time Interaction
Stream audio in real-time with minimal latency response times
Speech Processing
Built-in speech-to-text and text-to-speech capabilities with voice options
Advanced Turn Detection
Multiple voice activity detection options including semantic turn detection
Powerful Function Calling
Seamless support for calling external functions and APIs
​
Installation
To use OpenAIRealtimeBetaLLMService, install the required dependencies:

Copy

Ask AI
pip install "pipecat-ai[openai]"
You’ll also need to set up your OpenAI API key as an environment variable: OPENAI_API_KEY.
​
Configuration
​
Constructor Parameters
​
api_key
strrequired
Your OpenAI API key
​
model
strdefault:"gpt-4o-realtime-preview-2025-06-03"
The speech-to-speech model used for processing
​
base_url
strdefault:"wss://api.openai.com/v1/realtime"
WebSocket endpoint URL
​
session_properties
SessionProperties
Configuration for the realtime session
​
start_audio_paused
booldefault:"False"
Whether to start with audio input paused
​
send_transcription_frames
booldefault:"True"
Whether to emit transcription frames
​
Session Properties
The SessionProperties object configures the behavior of the realtime session:
​
modalities
List[Literal['text', 'audio']]
The modalities to enable (default includes both text and audio)
​
instructions
str
System instructions that guide the model’s behavior

Copy

Ask AI
service = OpenAIRealtimeBetaLLMService(
    api_key=os.getenv("OPENAI_API_KEY"),
    session_properties=SessionProperties(
        instructions="You are a helpful assistant. Be concise and friendly."
    )
)
​
voice
str
Voice ID for text-to-speech (options: alloy, echo, fable, onyx, nova, shimmer)
​
input_audio_format
Literal['pcm16', 'g711_ulaw', 'g711_alaw']
Format of the input audio
​
output_audio_format
Literal['pcm16', 'g711_ulaw', 'g711_alaw']
Format of the output audio
​
input_audio_transcription
InputAudioTranscription
Configuration for audio transcription

Copy

Ask AI
from pipecat.services.openai_realtime_beta.events import InputAudioTranscription

service = OpenAIRealtimeBetaLLMService(
    api_key=os.getenv("OPENAI_API_KEY"),
    session_properties=SessionProperties(
        input_audio_transcription=InputAudioTranscription(
            model="gpt-4o-transcribe",
            language="en",
            prompt="This is a technical conversation about programming"
        )
    )
)
​
input_audio_noise_reduction
InputAudioNoiseReduction
Configuration for audio noise reduction
​
turn_detection
Union[TurnDetection, SemanticTurnDetection, bool]
Configuration for turn detection (set to False to disable)
​
tools
List[Dict]
List of function definitions for tool/function calling
​
tool_choice
Literal['auto', 'none', 'required']
Controls when the model calls functions
​
temperature
float
Controls randomness in responses (0.0 to 2.0)
​
max_response_output_tokens
Union[int, Literal['inf']]
Maximum number of tokens to generate
​
Input Frames
​
Audio Input
​
InputAudioRawFrame
Frame
Raw audio data for speech input
​
Control Input
​
StartInterruptionFrame
Frame
Signals start of user interruption
​
UserStartedSpeakingFrame
Frame
Signals user started speaking
​
UserStoppedSpeakingFrame
Frame
Signals user stopped speaking
​
Context Input
​
OpenAILLMContextFrame
Frame
Contains conversation context
​
LLMMessagesAppendFrame
Frame
Appends messages to conversation
​
Output Frames
​
Audio Output
​
TTSAudioRawFrame
Frame
Generated speech audio
​
Control Output
​
TTSStartedFrame
Frame
Signals start of speech synthesis
​
TTSStoppedFrame
Frame
Signals end of speech synthesis
​
Text Output
​
TextFrame
Frame
Generated text responses
​
TranscriptionFrame
Frame
Speech transcriptions
​
Events
​
on_conversation_item_created
event
Emitted when a conversation item on the server is created. Handler receives:
item_id: str
item: ConversationItem
​
on_conversation_item_updated
event
Emitted when a conversation item on the server is updated. Handler receives:
item_id: str
item: Optional[ConversationItem] (may not exist for some updates)
​
Methods
​
retrieve_conversation_item
method
Retrieves a conversation item’s details from the server.

Copy

Ask AI
async def retrieve_conversation_item(self, item_id: str) -> ConversationItem
​
Usage Example

Copy

Ask AI
from pipecat.services.openai_realtime_beta import OpenAIRealtimeBetaLLMService
from pipecat.services.openai_realtime_beta.events import SessionProperties, TurnDetection

# Configure service
service = OpenAIRealtimeBetaLLMService(
    api_key="your-api-key",
    session_properties=SessionProperties(
        modalities=["audio", "text"],
        voice="alloy",
        turn_detection=TurnDetection(
            threshold=0.5,
            silence_duration_ms=800
        ),
        temperature=0.7
    )
)

# Use in pipeline
pipeline = Pipeline([
    audio_input,       # Produces InputAudioRawFrame
    service,           # Processes speech/generates responses
    audio_output       # Handles TTSAudioRawFrame
])
​
Function Calling
The service supports function calling with automatic response handling:

Copy

Ask AI
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai_realtime_beta import SessionProperties

# Define weather function using standardized schema
weather_function = FunctionSchema(
    name="get_weather",
    description="Get weather information",
    properties={
        "location": {"type": "string"}
    },
    required=["location"]
)

# Create tools schema
tools = ToolsSchema(standard_tools=[weather_function])

# Configure service with tools
llm = OpenAIRealtimeBetaLLMService(
    api_key="your-api-key",
    session_properties=SessionProperties(
        tools=tools,
        tool_choice="auto"
    )
)

llm.register_function("get_weather", fetch_weather_from_api)
See the Function Calling guide for:
Detailed implementation instructions
Provider-specific function definitions
Handler registration examples
Control over function call behavior
Complete usage examples
​
Frame Flow
InputAudioRawFrame

OpenAIRealtimeBetaLLMService

TranscriptionFrame

TTSStartedFrame

TTSAudioRawFrame

TTSStoppedFrame

ErrorFrame

LLMFullResponseStartFrame

Function Calls

LLMFullResponseEndFrame

Function Results

​
Metrics Support
The service collects comprehensive metrics:
Token usage (prompt and completion)
Processing duration
Time to First Byte (TTFB)
Audio processing metrics
Function call metrics
​
Advanced Features
​
Turn Detection

Copy

Ask AI
# Server-side basic VAD
turn_detection = TurnDetection(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=800
)

# Server-side semantic VAD
turn_detection = SemanticTurnDetection(
  type="semantic_vad",
  eagerness="auto", # default. could also be "low" | "medium" | "high"
  create_response=True # default
  interrupt_response=True # default
)

# Disable turn detection
turn_detection = False
​
Context Management

Copy

Ask AI
# Create context
context = OpenAIRealtimeLLMContext(
    messages=[],
    tools=[],
    system="You are a helpful assistant"
)

# Create aggregators
aggregators = service.create_context_aggregator(context)
​
Foundational Examples
OpenAI Realtime Beta Example
Basic implementation showing core realtime features including audio streaming, turn detection, and function calling.
​
Notes
Supports real-time speech-to-speech conversation
Handles interruptions and turn-taking
Manages WebSocket connection lifecycle
Provides function calling capabilities
Supports conversation context management
Includes comprehensive error handling
Manages audio streaming and processing
Handles both text and audio modalities
