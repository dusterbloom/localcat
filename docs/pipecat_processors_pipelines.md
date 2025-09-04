Advanced Frame Processors
Producer & Consumer Processors

Copy page

Route frames between different parts of a pipeline, allowing selective frame sharing across parallel branches or within complex pipelines

​
Overview
The Producer and Consumer processors work as a pair to route frames between different parts of a pipeline, particularly useful when working with ParallelPipeline. They allow you to selectively capture frames from one pipeline branch and inject them into another.
​
ProducerProcessor
ProducerProcessor examines frames flowing through the pipeline, applies a filter to decide which frames to share, and optionally transforms these frames before sending them to connected consumers.
​
Constructor Parameters
​
filter
Callable[[Frame], Awaitable[bool]]required
An async function that determines which frames should be sent to consumers. Should return True for frames to be shared.
​
transformer
Callable[[Frame], Awaitable[Frame]]default:"identity_transformer"
Optional async function that transforms frames before sending to consumers. By default, passes frames unchanged.
​
passthrough
booldefault:"True"
When True, passes all frames through the normal pipeline flow. When False, only passes through frames that don’t match the filter.
​
ConsumerProcessor
ConsumerProcessor receives frames from a ProducerProcessor and injects them into its pipeline branch.
​
Constructor Parameters
​
producer
ProducerProcessorrequired
The producer processor that will send frames to this consumer.
​
transformer
Callable[[Frame], Awaitable[Frame]]default:"identity_transformer"
Optional async function that transforms frames before injecting them into the pipeline.
​
direction
FrameDirectiondefault:"FrameDirection.DOWNSTREAM"
The direction in which to push received frames. Usually DOWNSTREAM to send frames forward in the pipeline.
​
Usage Examples
​
Basic Usage: Moving TTS Audio Between Branches

Copy

Ask AI
# Create a producer that captures TTS audio frames
async def is_tts_audio(frame: Frame) -> bool:
    return isinstance(frame, TTSAudioRawFrame)

# Define an async transformer function
async def tts_to_input_audio_transformer(frame: Frame) -> Frame:
    if isinstance(frame, TTSAudioRawFrame):
        # Convert TTS audio to input audio format
        return InputAudioRawFrame(
            audio=frame.audio,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels
        )
    return frame

producer = ProducerProcessor(
    filter=is_tts_audio,
    transformer=tts_to_input_audio_transformer
    passthrough=True  # Keep these frames in original pipeline
)

# Create a consumer to receive the frames
consumer = ConsumerProcessor(
    producer=producer,
    direction=FrameDirection.DOWNSTREAM
)

# Use in a ParallelPipeline
pipeline = Pipeline([
    transport.input(),
    ParallelPipeline(
        # Branch 1: LLM for bot responses
        [
            llm,
            tts,
            producer,  # Capture TTS audio here
        ],
        # Branch 2: Audio processing branch
        [
            consumer,  # Receive TTS audio here
            llm, # Speech-to-Speech LLM (audio in)
        ]
    ),
    transport.output(),
])


Advanced Frame Processors
UserIdleProcessor

Copy page

A processor that monitors user inactivity and triggers callbacks after specified timeout periods

The UserIdleProcessor is a specialized frame processor that monitors user activity in a conversation and executes callbacks when the user becomes idle. It’s particularly useful for maintaining engagement by detecting periods of user inactivity and providing escalating responses to inactivity.
​
Constructor Parameters
​
callback
Union[BasicCallback, RetryCallback]required
An async function that will be called when user inactivity is detected. Can be either:
Basic callback: async def(processor: UserIdleProcessor) -> None
Retry callback: async def(processor: UserIdleProcessor, retry_count: int) -> bool where returning False stops idle monitoring
​
timeout
floatrequired
The number of seconds to wait before considering the user idle.
​
Behavior
The processor starts monitoring for inactivity only after the first conversation activity (either UserStartedSpeakingFrame or BotSpeakingFrame). It manages idle state based on the following rules:
Resets idle timer when user starts or stops speaking
Pauses idle monitoring while user is speaking
Resets idle timer when bot is speaking
Stops monitoring on conversation end or cancellation
Manages a retry count for the retry callback
Stops monitoring when retry callback returns False
​
Properties
​
retry_count
int
The current number of retry attempts made to engage the user.
​
Example Implementations
Here are two example showing how to use the UserIdleProcessor: one with the basic callback and one with the retry callback:
Basic Callback
Retry Callback

Copy

Ask AI
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.user_idle_processor import UserIdleProcessor

async def handle_idle(user_idle: UserIdleProcessor) -> None:
    messages.append({
        "role": "system",
        "content": "Ask the user if they are still there and try to prompt for some input."
    })
    await user_idle.push_frame(LLMMessagesFrame(messages))

# Create the processor
user_idle = UserIdleProcessor(
    callback=handle_idle,
    timeout=5.0
)

# Add to pipeline
pipeline = Pipeline([
    transport.input(),
    user_idle,  # Add the processor to monitor user activity
    context_aggregator.user(),
    # ... rest of pipeline
])
​
Frame Handling
The processor handles the following frame types:
UserStartedSpeakingFrame: Marks user as active, resets idle timer and retry count
UserStoppedSpeakingFrame: Starts idle monitoring
BotSpeakingFrame: Resets idle timer
EndFrame / CancelFrame: Stops idle monitoring
​
Notes
The idle callback won’t be triggered while the user or bot is actively speaking
The processor automatically cleans up its resources when the pipeline ends
Basic callbacks are supported for backward compatibility
