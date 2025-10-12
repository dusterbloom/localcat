# Refactor HotPathMemoryProcessor God Object

**Priority**: Critical (SOLID Violation)
**Effort**: 5 days
**Assigned To**: Memory Systems Specialist

## Problem Statement

HotPathMemoryProcessor is a 1,100+ line "God Object" that violates Single Responsibility Principle by handling 8 different concerns:

1. Frame processing (pipeline integration)
2. Configuration management (40+ environment variables)
3. Memory injection logic
4. Session tracking and headers
5. Context window pruning
6. Background summarization
7. Ephemeral mode management
8. Metrics logging

**Current State**:
- File: `hotpath_processor.py:37-1137`
- Lines: 1,100+ lines in single class
- Responsibilities: 8 distinct concerns
- Testability: Difficult to test individual components
- Cognitive load: High for developers

## Impact

- **Maintainability**: Changes to one concern affect others
- **Testability**: Impossible to test components in isolation
- **Cognitive Load**: Developers must understand all 8 concerns
- **Bug Surface**: Large class increases risk of bugs

## Success Metrics

- ✓ HotPathMemoryProcessor reduced to <200 lines
- ✓ 5 new focused classes (<150 lines each)
- ✓ Each class has single, clear responsibility
- ✓ All existing tests pass
- ✓ New unit tests for each extracted class
- ✓ Performance maintained (<800ms voice latency)

## Implementation Approach

### Target Architecture

```
HotPathMemoryProcessor (150 lines)
├── ConfigurationManager (80 lines)
├── FrameProcessor (120 lines)
├── ContextInjector (100 lines)
├── SessionManager (120 lines)
└── BackgroundSummarizer (100 lines)
```

### Step 1: Extract ConfigurationManager

```python
# server/core/memory/config_manager.py (NEW FILE)
"""
Centralized configuration management for memory systems.

Handles parsing and validation of 40+ environment variables
with type safety and clear defaults.
"""

from dataclasses import dataclass
from typing import Optional, List
import os


def _parse_bool(value: str) -> bool:
    """Parse bool from env var"""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: Optional[str], default: int) -> int:
    """Parse int from env var with fallback"""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float(value: Optional[str], default: float) -> float:
    """Parse float from env var with fallback"""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_list(value: Optional[str], default: List[str]) -> List[str]:
    """Parse comma-separated list from env var"""
    if value is None:
        return default
    return [item.strip() for item in value.split(',') if item.strip()]


@dataclass
class MemoryConfiguration:
    """
    Complete configuration for HotPath memory system.

    Single source of truth for all memory-related settings.
    Replaces 40+ individual env var reads in hotpath_processor.py:116-194
    """

    # Core settings
    enabled: bool = True
    bullets_max: int = 3
    inject_role: str = "system"

    # Retrieval settings
    sources: List[str] = None  # ["convo", "graph"]
    max_turn_pairs: int = 4

    # Quality filtering
    interim_min_words: int = 6
    filter_quality: bool = True

    # Context management
    ctx_max_pairs: int = 4
    ctx_prune_enabled: bool = True
    ctx_inject_header: bool = True

    # Session tracking
    session_tracking_enabled: bool = True
    session_header_enabled: bool = True

    # Background processing
    summarization_enabled: bool = False
    summarization_interval: int = 10

    # Ephemeral mode
    ephemeral_mode: bool = False
    ephemeral_ttl_seconds: int = 3600

    # Performance
    retrieval_timeout_ms: int = 50
    cache_enabled: bool = True

    # Metrics
    metrics_enabled: bool = True
    metrics_log_interval: int = 60

    # Allowlists
    relation_allowlist: List[str] = None  # ["name", "lives_in", "works_at", "has"]

    # Thresholds
    confidence_threshold: float = 0.3
    recency_boost_hours: int = 24

    def __post_init__(self):
        """Set default values for None fields"""
        if self.sources is None:
            self.sources = ["convo", "graph"]
        if self.relation_allowlist is None:
            self.relation_allowlist = ["name", "lives_in", "works_at", "has", "interested_in"]

    @classmethod
    def from_env(cls) -> 'MemoryConfiguration':
        """
        Parse all memory configuration from environment variables.

        Supports both MEMORY_* and legacy HOTMEM_* prefixes.

        Returns:
            MemoryConfiguration with all settings
        """

        def get_env(key: str, legacy_key: str = None) -> Optional[str]:
            """Get env var with fallback to legacy key"""
            value = os.getenv(f"MEMORY_{key}")
            if value is None and legacy_key:
                value = os.getenv(f"HOTMEM_{legacy_key}")
            return value

        return cls(
            # Core settings
            enabled=_parse_bool(get_env("ENABLED", "ENABLED") or "true"),
            bullets_max=_parse_int(get_env("BULLETS_MAX", "BULLETS_MAX"), default=3),
            inject_role=get_env("INJECT_ROLE", "INJECT_ROLE") or "system",

            # Retrieval settings
            sources=_parse_list(get_env("SOURCES", "SOURCES"), default=["convo", "graph"]),
            max_turn_pairs=_parse_int(get_env("MAX_TURN_PAIRS"), default=4),

            # Quality filtering
            interim_min_words=_parse_int(get_env("INTERIM_MIN_WORDS"), default=6),
            filter_quality=_parse_bool(get_env("FILTER_QUALITY") or "true"),

            # Context management
            ctx_max_pairs=_parse_int(get_env("CTX_MAX_PAIRS"), default=4),
            ctx_prune_enabled=_parse_bool(get_env("CTX_PRUNE_ENABLED") or "true"),
            ctx_inject_header=_parse_bool(get_env("CTX_INJECT_HEADER") or "true"),

            # Session tracking
            session_tracking_enabled=_parse_bool(get_env("SESSION_TRACKING") or "true"),
            session_header_enabled=_parse_bool(get_env("SESSION_HEADER") or "true"),

            # Background processing
            summarization_enabled=_parse_bool(get_env("SUMMARIZATION_ENABLED") or "false"),
            summarization_interval=_parse_int(get_env("SUMMARIZATION_INTERVAL"), default=10),

            # Ephemeral mode
            ephemeral_mode=_parse_bool(get_env("EPHEMERAL_MODE") or "false"),
            ephemeral_ttl_seconds=_parse_int(get_env("EPHEMERAL_TTL"), default=3600),

            # Performance
            retrieval_timeout_ms=_parse_int(get_env("RETRIEVAL_TIMEOUT_MS"), default=50),
            cache_enabled=_parse_bool(get_env("CACHE_ENABLED") or "true"),

            # Metrics
            metrics_enabled=_parse_bool(get_env("METRICS_ENABLED") or "true"),
            metrics_log_interval=_parse_int(get_env("METRICS_LOG_INTERVAL"), default=60),

            # Allowlists
            relation_allowlist=_parse_list(
                get_env("RELATION_ALLOWLIST"),
                default=["name", "lives_in", "works_at", "has", "interested_in"]
            ),

            # Thresholds
            confidence_threshold=_parse_float(get_env("CONFIDENCE_THRESHOLD"), default=0.3),
            recency_boost_hours=_parse_int(get_env("RECENCY_BOOST_HOURS"), default=24),
        )

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.

        Returns:
            List of validation warning messages (empty if all valid)
        """
        warnings = []

        if self.bullets_max < 1 or self.bullets_max > 10:
            warnings.append(f"bullets_max={self.bullets_max} outside recommended range [1-10]")

        if self.retrieval_timeout_ms > 100:
            warnings.append(f"retrieval_timeout_ms={self.retrieval_timeout_ms} exceeds 100ms (impacts latency)")

        if not self.sources:
            warnings.append("No retrieval sources configured (memory will not work)")

        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            warnings.append(f"confidence_threshold={self.confidence_threshold} outside valid range [0.0-1.0]")

        return warnings
```

### Step 2: Extract FrameProcessor

```python
# server/core/memory/frame_processor.py (NEW FILE)
"""
Frame processing for Pipecat pipeline integration.

Handles routing of different frame types through the memory system.
"""

from pipecat.frames import Frame, TextFrame, LLMMessagesFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from typing import AsyncIterator


class MemoryFrameProcessor(FrameProcessor):
    """
    Focused frame processor for memory system.

    Responsibilities:
    - Route frames to appropriate handlers
    - Integrate memory injection into pipeline
    - Handle user/assistant turns
    """

    def __init__(
        self,
        config: MemoryConfiguration,
        injector: ContextInjector,
        session_manager: SessionManager,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.injector = injector
        self.session_manager = session_manager

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection
    ) -> AsyncIterator[Frame]:
        """
        Process frame through memory pipeline.

        Routes different frame types to appropriate handlers.
        """

        # Handle user speech start
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            yield frame
            return

        # Handle text frames
        if isinstance(frame, TextFrame):
            await self._handle_text_frame(frame)
            yield frame
            return

        # Handle LLM message frames (context injection point)
        if isinstance(frame, LLMMessagesFrame):
            # Inject memory context before LLM
            modified_frame = await self._inject_memory_context(frame)
            yield modified_frame
            return

        # Pass through other frames
        yield frame

    async def _handle_user_started_speaking(self, frame: UserStartedSpeakingFrame):
        """Handle user starting to speak"""
        # Update session tracking
        await self.session_manager.mark_user_speaking()

    async def _handle_text_frame(self, frame: TextFrame):
        """Handle text frame (observation)"""
        # Text frames are observed by HotMemory directly
        pass

    async def _inject_memory_context(self, frame: LLMMessagesFrame) -> LLMMessagesFrame:
        """Inject memory bullets into LLM context"""
        if not self.config.enabled:
            return frame

        # Delegate to ContextInjector
        modified_frame = await self.injector.inject(frame)
        return modified_frame
```

### Step 3: Extract ContextInjector

```python
# server/core/memory/context_injector.py (NEW FILE)
"""
Context injection for memory bullets into LLM messages.

Handles formatting and injection of memory context into conversation.
"""

from pipecat.frames import LLMMessagesFrame
from typing import List, Dict


class ContextInjector:
    """
    Inject memory bullets into LLM context.

    Responsibilities:
    - Retrieve memory bullets
    - Format bullets for injection
    - Inject into appropriate role (system/user)
    - Track injection metrics
    """

    def __init__(
        self,
        hot_memory,
        config: MemoryConfiguration,
        formatter: ContextFormatter
    ):
        self.hot = hot_memory
        self.config = config
        self.formatter = formatter
        self._injection_count = 0

    async def inject(self, frame: LLMMessagesFrame) -> LLMMessagesFrame:
        """
        Inject memory bullets into LLM messages frame.

        Args:
            frame: Original LLM messages frame

        Returns:
            Modified frame with memory context injected
        """
        if not self.config.enabled:
            return frame

        # Extract latest user query
        user_query = self._extract_user_query(frame.messages)
        if not user_query:
            return frame

        # Retrieve memory bullets
        bullets = await self._retrieve_bullets(user_query)
        if not bullets:
            return frame

        # Format bullets for injection
        memory_text = self.formatter.format_bullets(bullets)

        # Inject into frame
        modified_messages = self._inject_into_messages(
            frame.messages,
            memory_text,
            role=self.config.inject_role
        )

        # Track metrics
        self._injection_count += 1

        # Create new frame with modified messages
        return LLMMessagesFrame(messages=modified_messages)

    async def _retrieve_bullets(self, query: str) -> List[str]:
        """Retrieve memory bullets for query"""
        bullets = self.hot.retrieve_bullets(
            query=query,
            max_bullets=self.config.bullets_max,
            sources=self.config.sources
        )
        return bullets

    def _extract_user_query(self, messages: List[Dict]) -> str:
        """Extract latest user query from messages"""
        # Find last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _inject_into_messages(
        self,
        messages: List[Dict],
        memory_text: str,
        role: str
    ) -> List[Dict]:
        """Inject memory text into messages"""
        if role == "system":
            # Inject as system message (prepend)
            return [{"role": "system", "content": memory_text}] + messages
        else:
            # Inject into user message (append to latest user message)
            modified = messages.copy()
            for i in reversed(range(len(modified))):
                if modified[i].get("role") == "user":
                    modified[i]["content"] += f"\n\n{memory_text}"
                    break
            return modified

    def get_metrics(self) -> Dict:
        """Get injection metrics"""
        return {
            "injection_count": self._injection_count
        }
```

### Step 4: Extract SessionManager

```python
# server/core/memory/session_manager.py (NEW FILE)
"""
Session tracking and header management for memory system.

Handles session metadata injection and turn tracking.
"""

from typing import Dict, Optional
import time


class SessionManager:
    """
    Manage session tracking and headers.

    Responsibilities:
    - Track session metadata (ID, user, agent)
    - Inject session headers into context
    - Track conversation turns
    - Measure session duration
    """

    def __init__(
        self,
        session_id: str,
        user_eid: str,
        agent_eid: str,
        config: MemoryConfiguration
    ):
        self.session_id = session_id
        self.user_eid = user_eid
        self.agent_eid = agent_eid
        self.config = config

        self._turn_count = 0
        self._start_time = time.time()
        self._last_user_speech_time = None

    def get_session_header(self) -> str:
        """
        Generate session header for context injection.

        Returns:
            Formatted session header string
        """
        if not self.config.session_header_enabled:
            return ""

        duration = int(time.time() - self._start_time)

        header = f"[Session: {self.session_id[:8]}... | "
        header += f"User: {self.user_eid} | "
        header += f"Agent: {self.agent_eid} | "
        header += f"Turn: {self._turn_count} | "
        header += f"Duration: {duration}s]"

        return header

    async def mark_user_speaking(self):
        """Mark user started speaking"""
        self._last_user_speech_time = time.time()

    def increment_turn(self):
        """Increment turn counter"""
        self._turn_count += 1

    def get_metrics(self) -> Dict:
        """Get session metrics"""
        return {
            "session_id": self.session_id,
            "turn_count": self._turn_count,
            "duration_seconds": int(time.time() - self._start_time),
            "user_eid": self.user_eid,
            "agent_eid": self.agent_eid
        }
```

### Step 5: Extract BackgroundSummarizer

```python
# server/core/memory/background_summarizer.py (NEW FILE)
"""
Background summarization for conversation turns.

Generates periodic summaries of conversation for long-term memory.
"""

import asyncio
from typing import List, Optional


class BackgroundSummarizer:
    """
    Generate background summaries of conversations.

    Responsibilities:
    - Track turns since last summary
    - Trigger summarization at intervals
    - Generate summaries using LLM
    - Store summaries in memory
    """

    def __init__(
        self,
        hot_memory,
        config: MemoryConfiguration,
        llm_client
    ):
        self.hot = hot_memory
        self.config = config
        self.llm = llm_client

        self._turns_since_summary = 0
        self._summary_task = None

    def should_summarize(self) -> bool:
        """Check if summarization should trigger"""
        if not self.config.summarization_enabled:
            return False

        return self._turns_since_summary >= self.config.summarization_interval

    async def summarize_turns(self, turns: List[str]):
        """
        Generate summary of recent turns.

        Args:
            turns: List of conversation turn texts
        """
        if not turns:
            return

        # Generate summary via LLM
        summary_prompt = self._build_summary_prompt(turns)
        summary = await self._generate_summary(summary_prompt)

        # Store summary in memory
        if summary:
            self.hot.observe(summary, role="system")

        # Reset counter
        self._turns_since_summary = 0

    def _build_summary_prompt(self, turns: List[str]) -> str:
        """Build prompt for LLM summarization"""
        turns_text = "\n".join(f"- {turn}" for turn in turns)

        prompt = f"""Summarize the following conversation turns into 1-2 concise bullet points:

{turns_text}

Summary:"""
        return prompt

    async def _generate_summary(self, prompt: str) -> Optional[str]:
        """Generate summary using LLM"""
        try:
            response = await self.llm.generate(prompt, max_tokens=100)
            return response.strip()
        except Exception as e:
            # Log error but don't crash
            print(f"Summarization error: {e}")
            return None

    def increment_turn(self):
        """Increment turn counter"""
        self._turns_since_summary += 1
```

### Step 6: Refactor HotPathMemoryProcessor

```python
# server/core/memory/hotpath_processor.py (REFACTORED)
"""
HotPath memory processor - now a thin orchestrator.

Coordinates between specialized components for memory pipeline.
"""

from pipecat.processors.frame_processor import FrameProcessor
from .config_manager import MemoryConfiguration
from .frame_processor import MemoryFrameProcessor
from .context_injector import ContextInjector
from .session_manager import SessionManager
from .background_summarizer import BackgroundSummarizer


class HotPathMemoryProcessor(FrameProcessor):
    """
    Orchestrator for HotPath memory system.

    Now focused on coordination between specialized components.
    No longer a God Object - delegates to focused classes.
    """

    def __init__(
        self,
        hot_memory,
        session_id: str,
        user_eid: str,
        agent_eid: str,
        llm_client=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Load configuration
        self.config = MemoryConfiguration.from_env()

        # Validate configuration
        warnings = self.config.validate()
        for warning in warnings:
            print(f"[MEMORY CONFIG WARNING] {warning}")

        # Initialize specialized components
        self.hot = hot_memory

        self.session_manager = SessionManager(
            session_id=session_id,
            user_eid=user_eid,
            agent_eid=agent_eid,
            config=self.config
        )

        self.context_injector = ContextInjector(
            hot_memory=hot_memory,
            config=self.config,
            formatter=ContextFormatter()  # From existing module
        )

        self.frame_processor = MemoryFrameProcessor(
            config=self.config,
            injector=self.context_injector,
            session_manager=self.session_manager
        )

        if self.config.summarization_enabled and llm_client:
            self.summarizer = BackgroundSummarizer(
                hot_memory=hot_memory,
                config=self.config,
                llm_client=llm_client
            )
        else:
            self.summarizer = None

    async def process_frame(self, frame, direction):
        """Delegate frame processing to MemoryFrameProcessor"""
        async for processed_frame in self.frame_processor.process_frame(frame, direction):
            yield processed_frame

    def process_turn(self, text: str, role: str):
        """Process conversation turn (observation)"""
        # Observe in memory
        self.hot.observe(text, role=role)

        # Track turn
        self.session_manager.increment_turn()

        # Check if summarization needed
        if self.summarizer and self.summarizer.should_summarize():
            # Trigger background summarization (non-blocking)
            asyncio.create_task(self._background_summarize())

    async def _background_summarize(self):
        """Background summarization task"""
        recent_turns = self.hot.get_recent_turns(n=self.config.summarization_interval)
        await self.summarizer.summarize_turns(recent_turns)

    def get_metrics(self) -> dict:
        """Get combined metrics from all components"""
        metrics = {
            "config": self.config.__dict__,
            "session": self.session_manager.get_metrics(),
            "injection": self.context_injector.get_metrics(),
        }

        if self.summarizer:
            metrics["summarization"] = {
                "enabled": True,
                "turns_since_summary": self.summarizer._turns_since_summary
            }

        return metrics
```

## Testing Requirements

### Unit Tests for Each Component

```python
# server/tests/unit/memory_components/test_config_manager.py
def test_config_from_env():
    """Test configuration loading from environment"""
    os.environ["MEMORY_ENABLED"] = "true"
    os.environ["MEMORY_BULLETS_MAX"] = "5"

    config = MemoryConfiguration.from_env()

    assert config.enabled == True
    assert config.bullets_max == 5

def test_config_validation():
    """Test configuration validation"""
    config = MemoryConfiguration(bullets_max=100)  # Invalid
    warnings = config.validate()

    assert len(warnings) > 0
    assert "bullets_max" in warnings[0]


# server/tests/unit/memory_components/test_context_injector.py
def test_inject_memory_into_frame():
    """Test memory injection into LLM frame"""
    injector = ContextInjector(hot_memory=mock_hot, config=config, formatter=formatter)

    frame = LLMMessagesFrame(messages=[
        {"role": "user", "content": "What do I like?"}
    ])

    modified = await injector.inject(frame)

    # Should have memory injected
    assert len(modified.messages) > len(frame.messages)


# server/tests/unit/memory_components/test_session_manager.py
def test_session_header_generation():
    """Test session header formatting"""
    manager = SessionManager(
        session_id="test123",
        user_eid="alice",
        agent_eid="bot",
        config=config
    )

    header = manager.get_session_header()

    assert "test123" in header
    assert "alice" in header
    assert "Turn:" in header
```

### Integration Tests

```python
# server/tests/integration/test_refactored_processor.py
def test_end_to_end_memory_processing():
    """Test complete memory processing pipeline"""
    processor = HotPathMemoryProcessor(
        hot_memory=hot,
        session_id="test",
        user_eid="alice",
        agent_eid="bot"
    )

    # Process user turn
    processor.process_turn("I like pizza", role="user")

    # Query memory
    frame = LLMMessagesFrame(messages=[
        {"role": "user", "content": "What do I like?"}
    ])

    modified = await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    # Should have memory injected with "pizza"
    assert any("pizza" in str(msg) for msg in modified.messages)
```

## Files to Create/Modify

### New Files (5)
1. **server/core/memory/config_manager.py** (~150 lines)
2. **server/core/memory/frame_processor.py** (~120 lines)
3. **server/core/memory/context_injector.py** (~100 lines)
4. **server/core/memory/session_manager.py** (~120 lines)
5. **server/core/memory/background_summarizer.py** (~100 lines)

### Modified Files (1)
6. **server/core/memory/hotpath_processor.py** (reduce from 1100 to ~150 lines)

### Test Files (6 new)
7. **server/tests/unit/memory_components/test_config_manager.py**
8. **server/tests/unit/memory_components/test_frame_processor.py**
9. **server/tests/unit/memory_components/test_context_injector.py**
10. **server/tests/unit/memory_components/test_session_manager.py**
11. **server/tests/unit/memory_components/test_background_summarizer.py**
12. **server/tests/integration/test_refactored_processor.py**

## Definition of Done

- [ ] All 5 new component classes created
- [ ] HotPathMemoryProcessor refactored to <200 lines
- [ ] Each component has <150 lines
- [ ] Unit tests for each component (80%+ coverage)
- [ ] Integration tests pass (end-to-end)
- [ ] All existing tests still pass (regression)
- [ ] Performance tests pass (<800ms latency)
- [ ] Configuration validation working
- [ ] Documentation added to all new modules
- [ ] Code review completed

## Performance Validation

```bash
# Voice latency test
pytest server/tests/performance/test_voice_latency.py -v
# Expected: <800ms p95 latency maintained

# Memory retrieval test
pytest server/tests/performance/test_memory_retrieval.py -v
# Expected: <50ms retrieval time maintained
```

## Delegation Command

```bash
# Manager delegates to Memory Systems Specialist
droid exec memory-systems-specialist --auto medium -f tasks/refactor_hotpath_processor_god_object.md
```

---

**Related Issues**: Part of technical debt cleanup (Phase 2, Critical Priority)
**Blocks**: Configuration improvements, easier testing
**References**: Tech debt guardian report - Critical Issue #1 (God Object anti-pattern)
