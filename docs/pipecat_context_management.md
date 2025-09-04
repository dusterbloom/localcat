Context Management

Copy page

A guide to working with Pipecat’s Context and Context Aggregators

​
What is Context in Pipecat?
In Pipecat, context refers to the conversation history that the LLM uses to generate responses. The context consists of a list of alternating user/assistant messages that represents the collective history of the entire conversation.

Copy

Ask AI
# Example context structure
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help you?"},
    # Context aggregators automatically add new messages here
]
Since Pipecat is a real-time voice AI framework, context management happens automatically as the conversation flows, but you can also control it manually when needed.
​
How Context Updates During Conversations
Context updates happen automatically as frames flow through your pipeline:
User Messages:
User speaks → InputAudioRawFrame → STT Service → TranscriptionFrame
context_aggregator.user() receives TranscriptionFrame and adds user message to context
Assistant Messages:
LLM generates response → LLMTextFrame → TTS Service → TTSTextFrame
context_aggregator.assistant() receives TTSTextFrame and adds assistant message to context
Frame types that update context:
TranscriptionFrame: Contains user speech converted to text by STT service
LLMTextFrame: Contains LLM-generated responses
TTSTextFrame: Contains bot responses converted to text by TTS service (represents what was actually spoken)
The TTS service processes LLMTextFrames but outputs TTSTextFrames, which represent the actual spoken text returned by the TTS provider. This ensures context matches what users actually hear.
​
Setting Up Context Management
Pipecat includes a context aggregator that creates and manages context for both user and assistant messages:
​
1. Create the Context and Context Aggregator

Copy

Ask AI
# Create LLM service
llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

# Create context with initial messages
messages = [
    {"role": "system", "content": "You are a helpful voice assistant."}
]

# Create context (messages only)
context = OpenAILLMContext(messages)

# Create context aggregator instance
context_aggregator = llm.create_context_aggregator(context)
​
2. Context with Function Calling
Context can also include tools (function definitions) that the LLM can call during conversations:

Copy

Ask AI
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# Define available functions
weather_function = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use.",
        },
    },
    required=["location", "format"],
)

# Create tools schema
tools = ToolsSchema(standard_tools=[weather_function])

# Create context with both messages and tools
context = OpenAILLMContext(messages, tools)
context_aggregator = llm.create_context_aggregator(context)
Function call results are also automatically stored in the context, maintaining a complete conversation history including tool interactions.
We’ll cover function calling in detail in an upcoming section. The context aggregator handles function call storage automatically.
​
3. Add Context Aggregators to Your Pipeline

Copy

Ask AI
pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),      # User context aggregator
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(), # Assistant context aggregator
])
​
Context Aggregator Placement
The placement of context aggregator instances in your pipeline is crucial for proper operation:
​
User Context Aggregator
Place the user context aggregator downstream from the STT service. Since the user’s speech results in TranscriptionFrame objects pushed by the STT service, the user aggregator needs to be positioned to collect these frames.
​
Assistant Context Aggregator
Place the assistant context aggregator after transport.output(). This positioning is important because:
The TTS service outputs TTSTextFrames in addition to audio
The assistant aggregator must be downstream to collect those frames
It ensures context updates happen word-by-word for specific services (e.g. Cartesia, ElevenLabs, and Rime)
Your context stays updated at the word level in case an interruption occurs
Always place the assistant context aggregator after transport.output() to ensure proper word-level context updates during interruptions.
​
Manual Context Control
You can programmatically add new messages to the context by pushing or queueing specific frames:
​
Adding Messages
LLMMessagesAppendFrame: Appends a new message to the existing context
LLMMessagesUpdateFrame: Completely replaces the existing context with new messages

Copy

Ask AI
# Add a new system message to context
new_message = {"role": "system", "content": "You are now in expert mode."}
await task.queue_frames([
    LLMMessagesAppendFrame([new_message], run_llm=True), # Optionally trigger bot response, too
])
​
Retrieving Current Context
The context aggregator provides a context property for getting the current context:

Copy

Ask AI
context = context_aggregator.user().context
​
Triggering Bot Responses
You may want to manually trigger the bot to speak in two scenarios:
Starting a pipeline where the bot should speak first
After editing the context using LLMMessagesAppendFrame or LLMMessagesUpdateFrame

Copy

Ask AI
# Example: Bot speaks first when pipeline starts
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    # Trigger a response
    await task.queue_frames([LLMRunFrame()])

Copy

Ask AI
# Example: Bot speaks after context is edited
new_message = {"role": "user", "content": "Tell me a fun fact."}
await task.queue_frames([
    LLMMessagesAppendFrame([new_message], run_llm=True), # Trigger bot response
])
This gives you fine-grained control over when and how the bot responds during the conversation flow.
​
Key Takeaways
Context is conversation history - automatically maintained as users and bots exchange messages
Frame types matter - TranscriptionFrame for users, TTSTextFrame for assistants
Placement matters - user aggregator after STT, assistant aggregator after transport output
Tools are included - function definitions and results are stored in context
Manual control available - use frames to append messages or trigger responses when needed
Word-level precision - proper placement ensures context accuracy during interruptions
​
