Memory
Mem0

Copy page

Long-term conversation memory service powered by Mem0

​
Overview
Mem0MemoryService provides long-term memory capabilities for conversational agents by integrating with Mem0’s API. It automatically stores conversation history and retrieves relevant past context based on the current conversation, enhancing LLM responses with persistent memory across sessions.
​
Installation
To use the Mem0 memory service, install the required dependencies:

Copy

Ask AI
pip install "pipecat-ai[mem0]"
You’ll also need to set up your Mem0 API key as an environment variable: MEM0_API_KEY.
You can obtain a Mem0 API key by signing up at mem0.ai.
​
Mem0MemoryService
​
Constructor Parameters
​
api_key
strrequired
Mem0 API key for accessing the service
​
host
strrequired
Mem0 host for accessing the service
​
user_id
str
Unique identifier for the end user to associate with memories
​
agent_id
str
Identifier for the agent using the memory service
​
run_id
str
Identifier for the specific conversation session
​
params
InputParams
Configuration parameters for memory retrieval (see below)
​
local_config
dict
Configuration for using local LLMs and embedders instead of Mem0’s cloud API (see Local Configuration section)
At least one of user_id, agent_id, or run_id must be provided to organize memories.
​
Input Parameters
The params object accepts the following configuration settings:
​
search_limit
intdefault:"10"
Maximum number of relevant memories to retrieve per query
​
search_threshold
floatdefault:"0.1"
Relevance threshold for memory retrieval (0.0 to 1.0)
​
api_version
strdefault:"v2"
Mem0 API version to use
​
system_prompt
str
Prefix text to add before retrieved memories
​
add_as_system_message
booldefault:"True"
Whether to add memories as a system message (True) or user message (False)
​
position
intdefault:"1"
Position in the context where memories should be inserted
​
Input Frames
The service processes the following input frames:
​
OpenAILLMContextFrame
Frame
Contains OpenAI-specific conversation context
​
LLMMessagesFrame
Frame
Contains conversation messages in standard format
​
Output Frames
The service may produce the following output frames:
​
LLMMessagesFrame
Frame
Enhanced messages with relevant memories included
​
OpenAILLMContextFrame
Frame
Enhanced OpenAI context with memories included
​
ErrorFrame
Frame
Contains error information if memory operations fail
​
Memory Operations
The service performs two main operations automatically:
​
Message Storage
All conversation messages are stored in Mem0 for future reference. The service:
Captures full message history from context frames
Associates messages with the specified user/agent/run IDs
Stores metadata to enable efficient retrieval
​
Memory Retrieval
When a new user message is detected, the service:
Uses the message as a search query
Retrieves relevant past memories from Mem0
Formats memories with the configured system prompt
Adds the formatted memories to the conversation context
Passes the enhanced context downstream in the pipeline
​
Pipeline Positioning
The memory service should be positioned after the user context aggregator but before the LLM service:

Copy

Ask AI
context_aggregator.user() → memory_service → llm
This ensures that:
The user’s latest message is included in the context
The memory service can enhance the context before the LLM processes it
The LLM receives the enhanced context with relevant memories
​
Usage Examples
​
Basic Integration

Copy

Ask AI
from pipecat.services.mem0.memory import Mem0MemoryService
from pipecat.pipeline.pipeline import Pipeline

# Create the memory service
memory = Mem0MemoryService(
    api_key=os.getenv("MEM0_API_KEY"),
    user_id="user123",  # Unique user identifier
)

# Position the memory service between context aggregator and LLM
pipeline = Pipeline([
    transport.input(),
    context_aggregator.user(),
    memory,           # <-- Memory service enhances context here
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant()
])
​
Using Local Configuration
The local_config parameter allows you to use your own LLM and embedding providers instead of Mem0’s cloud API. This is useful for self-hosted deployments or when you want more control over the memory processing.

Copy

Ask AI
local_config = {
    "llm": {
        "provider": str,  # LLM provider name (e.g., "anthropic", "openai")
        "config": {
            # Provider-specific configuration
            "model": str,  # Model name
            "api_key": str,  # API key for the provider
            # Other provider-specific parameters
        }
    },
    "embedder": {
        "provider": str,  # Embedding provider name (e.g., "openai")
        "config": {
            # Provider-specific configuration
            "model": str,  # Model name
            # Other provider-specific parameters
        }
    }
}

# Initialize Mem0 memory service with local configuration
memory = Mem0MemoryService(
    local_config=local_config,  # Use local LLM for memory processing
    user_id="user123",          # Unique identifier for the user
)
When using local_config do not provide the api_key parameter.
​
Frame Flow
Query Mem0

Relevant Memories

Enhanced Context

Store Response

Context Aggregator

LLMMessagesFrame

Mem0MemoryService

Mem0 API

LLM Service

TTS

Output to User

Mem0 API

​
Error Handling
The service includes basic error handling to ensure conversation flow continues even when memory operations fail:
Exceptions during memory storage and retrieval are caught and logged
If an error occurs during frame processing, an ErrorFrame is emitted with error details
The original frame is still passed downstream to prevent the pipeline from stalling
Connection and authentication errors from the Mem0 API will be logged but won’t interrupt the conversation
