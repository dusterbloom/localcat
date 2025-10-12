"""
LocalCat: Pipecat processor that injects ultra-fast memory bullets
Place between context_aggregator.user() and llm in your Pipeline
"""

import asyncio
import time
from collections import deque
from typing import List, Optional, Dict, Any
from loguru import logger
import sys
import os
# Add local pipecat to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, TextFrame, StartFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor as BaseProcessor, FrameDirection

from .memory_store import MemoryStore, Paths
from .memory_hotpath import HotMemory
from .context import MemoryContextFrame
from .context_formatter import ContextFormatter
from .session_tracker import SessionTracker

# Import intent service for smart processing
try:
    from ..intent import get_intent_service, IntentExceptionHandler
    INTENT_SERVICE_AVAILABLE = True
except ImportError:
    INTENT_SERVICE_AVAILABLE = False
    logger.warning("Intent service not available - using standard memory processing")

# Ensure we only add a file sink once per process
_HOTMEM_LOG_SINK_ADDED = False


class HotPathMemoryProcessor(BaseProcessor):
    """
    Ultra-fast memory processor for Pipecat:
    - On final ASR segments, extracts UD triples and updates compiled memory
    - Injects ≤ 3 short user-role bullets directly into context
    - Never blocks on disk or LLM calls
    - Target latency: <200ms p95
    """
    
    def __init__(self, 
                 sqlite_path: Optional[str] = None, 
                 lmdb_dir: Optional[str] = None, 
                 user_id: str = "default-user",
                 enable_metrics: bool = True,
                 context_aggregator = None,
                 *,
                 session_tracker: Optional[SessionTracker] = None,
                 agent_id: Optional[str] = None):
        """
        Initialize HotMem processor
        
        Args:
            sqlite_path: Path to SQLite database (default: from env or "memory.db")
            lmdb_dir: Path to LMDB directory (default: from env or "graph.lmdb")
            user_id: User identifier for memory context
            enable_metrics: Whether to track performance metrics
            context_aggregator: Context aggregator for injecting memory bullets
        """
        super().__init__()

        # Optional file logging via Loguru (non-blocking)
        global _HOTMEM_LOG_SINK_ADDED
        if not _HOTMEM_LOG_SINK_ADDED:
            try:
                log_path = os.getenv(
                    "HOTMEM_LOG_FILE",
                    os.path.join(os.path.dirname(__file__), ".logs", "hotmem.log"),
                )
                log_level = os.getenv("HOTMEM_LOG_LEVEL", "DEBUG").upper()
                log_dir = os.path.dirname(log_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                logger.add(
                    log_path,
                    rotation="10 MB",
                    retention="10 days",
                    enqueue=True,  # background thread
                    level=log_level,
                    backtrace=False,
                    diagnose=False,
                )
                _HOTMEM_LOG_SINK_ADDED = True
            except Exception as e:
                # Don't fail hot path on logging issues
                logger.warning(f"HotMem file logging not enabled: {e}")
        
        # Initialize storage
        paths = Paths(
            sqlite_path=sqlite_path,
            lmdb_dir=lmdb_dir
        )
        self.store = MemoryStore(paths)
        
        # Initialize hot memory
        self.hot = HotMemory(self.store)
        # Pre-warm NLP to avoid first-turn latency
        try:
            self.hot.prewarm("en")
        except Exception:
            pass
        
        # Rebuild RAM indices from persistent store
        try:
            self.hot.rebuild_from_store()
        except Exception as e:
            logger.warning(f"Could not rebuild from store (starting fresh): {e}")
        
        # Session tracking
        self._turn_id = 0
        self._user_id = user_id
        self._session_id = f"{user_id}_{int(time.time())}_{os.urandom(4).hex()}"  # Unique session ID per user
        self._session_start = time.time()
        self._enable_metrics = enable_metrics
        self._pending_bullets: List[str] = []
        # Phase 0 state: track one-time interim pre-injection per turn
        self._turn_has_preinjected_bullets: bool = False
        self._last_injected_bullets: List[str] = []
        self._turn_ready_signaled: bool = False
        # Ephemeral mode: when enabled, bypass all storage, extraction, and retrieval
        self._ephemeral: bool = False
        # Excluded phrases (not stored or injected); defaults to enrollment fixed-phrase
        ex_phr = os.getenv("EXCLUDED_MEMORY_PHRASES", "").strip()
        fixed = os.getenv("ENROLLMENT_FIXED_PHRASE", "").strip()
        items = [p.strip() for p in ex_phr.split("||") if p.strip()]
        if fixed:
            items.append(fixed)
        self._excluded_phrases = [p.lower() for p in items]
        # Env-driven controls (Phase 0.5)
        self._enabled: bool = os.getenv("MEMORY_ENABLED", "true").lower() in ("1", "true", "yes")
        try:
            # Support both MEMORY_* and HOTMEM_* for bullets and interim words
            self._bullets_max: int = int(os.getenv("MEMORY_BULLETS_MAX") or os.getenv("HOTMEM_BULLETS_MAX") or "3")
        except Exception:
            self._bullets_max = 3
        try:
            self._interim_min_words: int = int(os.getenv("MEMORY_INTERIM_MIN_WORDS") or os.getenv("HOTMEM_INTERIM_MIN_WORDS") or "6")
        except Exception:
            self._interim_min_words = 6
        # Support both MEMORY_* and legacy HOTMEM_* envs for injection settings
        self._inject_role = (os.getenv("MEMORY_INJECT_ROLE") or os.getenv("HOTMEM_INJECT_ROLE") or "system").strip().lower()
        if self._inject_role not in ("user", "system"):
            self._inject_role = "system"
        self._inject_header = os.getenv("MEMORY_INJECT_HEADER") or os.getenv("HOTMEM_INJECT_HEADER") or "Use the following factual context if helpful."

        # Initialize context formatter for bullet formatting and message building
        self._context_formatter = ContextFormatter(
            max_bullets=self._bullets_max,
            inject_role=self._inject_role,
            inject_header=self._inject_header
        )

        self._trace_frames = os.getenv("MEMORY_TRACE_FRAMES", "false").lower() in ("1", "true", "yes")
        self._handshake_enabled = os.getenv("MEMORY_ENABLE_HANDSHAKE", "true").lower() in ("1", "true", "yes")
        # Retrieval source controls (Phase 2-ready; used now for convo indexing)
        self._memory_sources = [s.strip() for s in os.getenv("MEMORY_SOURCES", "graph").split(",") if s.strip()]
        self._convo_index_enabled = os.getenv("MEMORY_CONVO_INDEX", "false").lower() in ("1", "true", "yes")
        # LLM Summarizer controls (background)
        self._summary_enabled = (
            os.getenv("MEMORY_SUMMARIZER_ENABLED", "false").lower() in ("1", "true", "yes")
        )
        self._summary_base_url = os.getenv("MEMORY_SUMMARIZER_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        self._summary_api_key = os.getenv("MEMORY_SUMMARIZER_API_KEY", "")
        self._summary_model = os.getenv("MEMORY_SUMMARIZER_MODEL", "llama-3.2-3b-instruct")
        try:
            self._summary_interval_secs = float(os.getenv("MEMORY_SUMMARIZER_INTERVAL_SECS", "60"))
        except Exception:
            self._summary_interval_secs = 60.0
        try:
            self._summary_max_tokens = int(os.getenv("MEMORY_SUMMARIZER_MAX_TOKENS", "160"))
        except Exception:
            self._summary_max_tokens = 160
        try:
            self._summary_max_messages = int(os.getenv("MEMORY_SUMMARIZER_MAX_MESSAGES", "10"))
        except Exception:
            self._summary_max_messages = 10
        # Turn-based summarization controls
        self._window_mode = os.getenv("MEMORY_SUMMARIZER_WINDOW_MODE", "turn_pairs").lower()
        try:
            self._turn_pairs = int(os.getenv("MEMORY_SUMMARIZER_TURN_PAIRS", "5"))
        except Exception:
            self._turn_pairs = 5
        self._last_summarized_turn = 0
        self._summary_task: Optional[asyncio.Task] = None
        
        # Store context aggregator reference for direct context injection
        self._context_aggregator = context_aggregator
        self._session_tracker = session_tracker
        self._agent_id = agent_id or os.getenv("AGENT_ID", "locat")
        self._session_header_tag = "[Session Context]"

        # Initialize intent service for smart processing
        self._intent_aware_processing = INTENT_SERVICE_AVAILABLE and os.getenv("INTENT_CLASSIFICATION_ENABLED", "true").lower() == "true"
        if self._intent_aware_processing:
            try:
                self.intent_service = get_intent_service()
                logger.info("[HotMem] Intent-aware processing enabled")
            except Exception as e:
                logger.warning(f"[HotMem] Failed to initialize intent service: {e}")
                self._intent_aware_processing = False
        else:
            self.intent_service = None

        if self._trace_frames:
            logger.debug(f"[HotMem] Frame tracing ENABLED - will log all frames flowing through processor")

        # Performance tracking
        self._last_metrics_log = time.time()
        
        stats = None
        if self._session_tracker:
            stats = self._session_tracker.start_session(self._user_id, self._session_id)
        else:
            logger.warning("[HotMem] No session tracker available during initialization")
        self._ensure_session_header(stats=stats, initial=True)
        # Provide role-aware IDs to HotMemory
        try:
            self.hot.agent_eid = f"agent:{self._agent_id}"
            self.hot.current_user_id = self._user_id
            self.hot.current_session_id = self._session_id
        except Exception:
            pass
        logger.debug(f"HotPathMemoryProcessor initialized for user: {user_id}")

        # Session already registered above - no need to call start_session again

        # Sliding window context settings
        try:
            self._ctx_window_enabled = os.getenv("CONTEXT_SLIDING_WINDOW", "true").lower() in ("1", "true", "yes")
            self._ctx_max_pairs = int(os.getenv("CONTEXT_MAX_TURN_PAIRS", "4"))
        except Exception:
            self._ctx_window_enabled = True
            self._ctx_max_pairs = 4
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process pipeline frames per Critical FrameProcessor rules:
        - ALWAYS call parent first
        - Push StartFrame immediately downstream
        - ALWAYS forward the incoming frame
        - Inject additional frames (memory bullets) as needed
        """
        # REQUIRED: call parent to set initialization state
        await super().process_frame(frame, direction)

        # Log all frames for debugging
        # logger.debug(f"[HotMem] process_frame called: {type(frame).__name__}")

        # If memory is disabled or ephemeral, simply forward
        if not self._enabled or self._ephemeral:
            await self.push_frame(frame, direction)
            return

        # REQUIRED: handle StartFrame immediately
        if isinstance(frame, StartFrame):
            # Start background summarizer only for delta mode (time-based)
            if (not self._ephemeral) and self._summary_enabled and self._window_mode == "delta" and self._summary_task is None:
                try:
                    self._summary_task = asyncio.create_task(self._summary_loop())
                    logger.debug("[HotMem] Background summarizer started (delta mode)")
                except Exception as e:
                    logger.warning(f"[HotMem] Could not start summarizer: {e}")
            elif (not self._ephemeral) and self._summary_enabled and self._window_mode == "turn_pairs":
                logger.debug(f"[HotMem] Turn-based summarization enabled (every {self._turn_pairs} turns)")
            await self.push_frame(frame, direction)
            return

        # # Enhanced frame tracing for debugging live call issues
        # if self._trace_frames:
        #     try:
        #         fname = type(frame).__name__
        #         # Skip audio frames to avoid log flooding
        #         if 'Audio' in fname or fname in ('InputAudioRawFrame', 'OutputAudioRawFrame', 'TTSAudioRawFrame', 'UserSpeakingFrame'):
        #             pass  # Skip audio frames
        #         elif isinstance(frame, TranscriptionFrame):
        #             is_final = getattr(frame, 'is_final', None)
        #             text = getattr(frame, 'text', '') or ''
        #             logger.info(f"[HotMem TRACE] {fname} is_final={is_final} text_len={len(text)} text='{text[:80]}'")
        #         elif isinstance(frame, LLMMessagesFrame):
        #             messages = getattr(frame, 'messages', [])
        #             roles = [m.get('role') for m in messages if isinstance(m, dict)]
        #             user_content = [m.get('content', '') for m in messages if isinstance(m, dict) and m.get('role') == 'user']
        #             logger.info(f"[HotMem TRACE] {fname} roles={roles} user_messages={len(user_content)} first_user='{(user_content[0] if user_content else '')[:80]}'")
        #         else:
        #             # Log non-audio frame types to understand what's flowing through
        #             logger.info(f"[HotMem TRACE] {fname}")
        #     except Exception as e:
        #         logger.warning(f"[HotMem TRACE] Error tracing {type(frame).__name__}: {e}")

        # Phase 0: Interim pre-injection (retrieval-only; once per turn)
        if isinstance(frame, InterimTranscriptionFrame):
            if self._ephemeral:
                # Do not inject any memory in ephemeral mode
                await self.push_frame(frame, direction)
                return
            text = getattr(frame, 'text', '') or ''
            # Basic length threshold; no intent gating in Phase 0
            if not self._turn_has_preinjected_bullets:
                try:
                    # Count words quickly
                    wcount = len([w for w in text.strip().split() if w])
                except Exception:
                    wcount = 0
                if wcount >= self._interim_min_words:
                    try:
                        # Provide identity scope to retriever
                        try:
                            self.hot.current_session_id = self._session_id
                            self.hot.current_user_id = self._user_id
                        except Exception:
                            pass
                        # Note: Interim doesn't have intent yet (happens in _process_transcription)
                        preview = self.hot.retrieve_bullets(text, read_only=True, intent=None)
                    except Exception as e:
                        logger.error(f"[HotMem] Interim retrieval failed: {e}")
                        preview = []
                    if preview:
                        cap = max(0, self._bullets_max)
                        inject_now = preview[:cap]
                        self._pending_bullets = list(inject_now)
                        try:
                            await self._inject_memory_context()
                            self._turn_has_preinjected_bullets = True
                            self._last_injected_bullets = list(inject_now)
                            logger.debug(f"[HotMem] Interim pre-injection completed with {len(self._last_injected_bullets)} bullets")
                            if self._handshake_enabled:
                                try:
                                    await self.push_frame(MemoryContextReadyFrame(), direction)
                                    self._turn_ready_signaled = True
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"[HotMem] Interim pre-injection error: {e}")

        # Process final transcriptions (compute bullets, update store)
        if isinstance(frame, TranscriptionFrame):
            if self._ephemeral:
                # In ephemeral mode, do not process/store/inject; forward only
                await self.push_frame(frame, direction)
                return
            is_final = getattr(frame, 'is_final', None)
            text = getattr(frame, 'text', '') or ''
            logger.info(f"[HotMem] TranscriptionFrame received: is_final={is_final} text_len={len(text)} text='{text[:120]}'")
            if self._is_excluded(text):
                logger.debug("[HotMem] Skipping excluded phrase from memory processing")
                await self.push_frame(frame, direction)
                return
            # WhisperSTTServiceMLX doesn't set is_final, so treat None as final (non-streaming)
            if is_final is True or is_final is None:
                # Provide identity scope to retriever
                try:
                    self.hot.current_session_id = self._session_id
                    self.hot.current_user_id = self._user_id
                except Exception:
                    pass
                logger.info(f"[HotMem] Processing transcription (is_final={is_final}): '{text}'")
                # Process: extract+persist+retrieve for final
                await self._process_transcription(frame, direction)

                # Phase 0: Refresh injection if different from interim
                if self._pending_bullets and self._context_aggregator:
                    try:
                        # Compare with last injected bullets
                        new_bullets = list(self._pending_bullets)
                        if not self._turn_has_preinjected_bullets or new_bullets != self._last_injected_bullets:
                            await self._inject_memory_context()
                            self._last_injected_bullets = new_bullets
                            logger.debug(f"[HotMem] Final injection {'refreshed' if self._turn_has_preinjected_bullets else 'inserted'} with {len(new_bullets)} bullets")
                            if self._handshake_enabled:
                                try:
                                    await self.push_frame(MemoryContextReadyFrame(), direction)
                                    self._turn_ready_signaled = True
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"[HotMem] Final injection error: {e}")
                # Reset pre-injection state for next turn
                self._turn_has_preinjected_bullets = False
                self._last_injected_bullets = []
                self._turn_ready_signaled = False
            else:
                logger.debug(f"[HotMem] Skipping non-final transcription")

        # Legacy LLMMessagesFrame handling removed - now using direct context injection

        # REQUIRED: always forward the original frame
        await self.push_frame(frame, direction)

    def set_ephemeral_mode(self, enabled: bool) -> None:
        """Enable/disable ephemeral mode (no storage/extraction/retrieval)."""
        self._ephemeral = bool(enabled)
        if self._ephemeral:
            logger.info("[HotMem] Ephemeral mode ENABLED: memory storage and retrieval are bypassed for this session")
        else:
            logger.info("[HotMem] Ephemeral mode DISABLED: normal memory processing restored")
        # Refresh header to reflect anonymous display if needed
        try:
            self._ensure_session_header()
        except Exception:
            pass

    def _is_excluded(self, text: str) -> bool:
        if not text or not self._excluded_phrases:
            return False
        tl = text.lower()
        for p in self._excluded_phrases:
            if p and p in tl:
                return True
        return False

    def _is_quality_conversation(self, text: str) -> bool:
        """
        Quality filter for conversation storage (Layer 2 defense)
        Returns True if the text should be stored, False if it should be filtered out.

        Filters out:
        1. Meta-commentary about the conversation itself
        2. Confusion indicators
        3. Overly short or vague utterances
        4. System-like messages

        Example filtered texts:
        - "think it's just confusing"
        - "Context unclear"
        - "I don't know"
        - "that's confusing"
        """
        if not text or not text.strip():
            return False

        t = text.lower().strip()

        # Filter very short utterances (< 10 chars)
        if len(t) < 10:
            return False

        # Filter meta-commentary about confusion or conversation quality
        confusion_patterns = [
            "confus",  # confusing, confused
            "unclear",
            "don't understand",
            "doesn't make sense",
            "not sure what",
            "what do you mean",
            "what are you",
            "i don't know",
            "dunno",
            "no idea",
            "meta",
            "context unclear",
            "repetitive",
        ]

        for pattern in confusion_patterns:
            if pattern in t:
                logger.debug(f"[HotMem] Filtering confused/meta utterance: '{text[:50]}...'")
                return False

        # Filter system-like messages
        system_patterns = [
            "[memory",
            "[session",
            "[system",
            "context]",
            "summary:",
        ]

        for pattern in system_patterns:
            if pattern in t:
                logger.debug(f"[HotMem] Filtering system-like message: '{text[:50]}...'")
                return False

        # Filter utterances that are just questions without assertions
        # (questions don't provide factual context for retrieval)
        if t.endswith("?"):
            # Check if it contains any facts or assertions (has verbs besides question words)
            assertion_verbs = [" am ", " is ", " are ", " was ", " were ", " have ", " has ", " had ", " like ", " love ", " want ", " need "]
            has_assertion = any(verb in f" {t} " for verb in assertion_verbs)
            if not has_assertion:
                logger.debug(f"[HotMem] Filtering pure question without assertions: '{text[:50]}...'")
                return False

        return True

    def set_user_identity(self, user_id: str) -> None:
        """Switch the active user identity for headers and future indexing."""
        try:
            user_id = (user_id or "").strip() or self._user_id
            if user_id != self._user_id:
                # Store original user_id for session tracking to avoid case sensitivity issues
                self._display_user_id = user_id  # For display purposes
                # Keep the original case for session tracking compatibility
                session_user_id = self._normalize_user_id_for_session(user_id)
                
                if session_user_id != self._user_id:
                    logger.info(f"[HotMem] User identity changed from '{self._user_id}' to '{session_user_id}' (display: '{user_id}')")
                    self._user_id = session_user_id
                    # Namespaced 'you' so future facts are user-scoped
                    try:
                        self.hot.user_eid = f"you:{self._user_id}"
                    except Exception:
                        pass
                    # Provide scope for retrieval
                    try:
                        self.hot.current_user_id = self._user_id
                        self.hot.current_session_id = self._session_id
                    except Exception:
                        pass
                    self._ensure_session_header()
        except Exception as e:
            logger.warning(f"[HotMem] Failed to set user identity: {e}")
    
    def _normalize_user_id_for_session(self, user_id: str) -> str:
        """Normalize user_id for session tracking to avoid case sensitivity issues.
        
        This ensures that session tracking works consistently regardless of
        how the user's name is capitalized by speaker recognition.
        """
        # Use the original environment variable value if available
        env_user_id = os.getenv("USER_ID", "")
        if env_user_id.lower() == user_id.lower():
            return env_user_id
        
        # Otherwise use lowercase for consistency
        return user_id.lower()
    
    async def _process_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Process final user transcription with intent awareness"""
        if not getattr(self, "_enabled", True):
            return
        self._turn_id += 1
        text = frame.text or ""
        logger.info(f"[HotMem] _process_transcription called: turn_id={self._turn_id}, text='{text[:50]}...'")

        if not text.strip():
            return

        start = time.perf_counter()

        # Intent classification for smart processing (using refactored service)
        intent_result = None
        if self._intent_aware_processing and self.intent_service:
            try:
                intent_result = await self.intent_service.classify_intent(text)
                logger.info(f"[HotMem] Intent classified: {intent_result['intent']} "
                           f"(confidence: {intent_result['confidence']:.2f}, "
                           f"strategy: {intent_result.get('strategy', 'unknown')}, "
                           f"skip: {intent_result.get('skip_memory', False)})")
            except Exception as e:
                logger.warning(f"[HotMem] Intent classification failed: {e}")
                # Use exception handler for graceful fallback if available
                if INTENT_SERVICE_AVAILABLE:
                    intent_result = IntentExceptionHandler.handle_classification_error(e, text)

        # Smart processing based on intent (using new routing decisions)
        if intent_result and not intent_result.get('fallback', False):
            intent_name = intent_result['intent']
            strategy = intent_result.get('strategy', 'standard')
            skip_memory = intent_result.get('skip_memory', False)

            # Skip memory processing if routing decision says so
            if skip_memory:
                logger.info(f"[HotMem] Skipping memory processing for intent: {intent_name} "
                           f"(reasoning: {intent_result.get('routing_reasoning', 'not provided')})")
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._record_turn_metrics(elapsed_ms)
                # Clear any pending bullets since we're skipping memory
                self._pending_bullets = []
                return

            # Apply strategy-based processing
            logger.debug(f"[HotMem] Using {strategy} strategy for intent: {intent_name}")

            # Enhanced processing for different strategies - ALWAYS pass intent for routing
            if strategy == 'storage_focused':
                bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='storage', intent=intent_result)
            elif strategy == 'retrieval_focused':
                bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='retrieval', intent=intent_result)
            elif strategy == 'deletion_focused':
                bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='deletion', intent=intent_result)
            elif strategy == 'lookup_focused':
                bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='lookup', intent=intent_result)
            elif strategy == 'minimal':
                # Minimal processing - just retrieve context without extraction
                bullets = self.hot.retrieve_bullets(text, read_only=True, intent=intent_result)
                triples = []
            elif strategy == 'contextual':
                # Contextual processing - focus on recent context
                bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='context', intent=intent_result)
            elif strategy == 'recent_context':
                # Recent context processing - for corrections
                bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='recent', intent=intent_result)
            else:
                # Standard processing for other strategies
                bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, intent=intent_result)
        else:
            # Fallback to standard processing if no intent classification or fallback result
            fallback_reason = "no intent classification"
            if intent_result:
                fallback_reason = f"fallback classification ({intent_result.get('reason', 'unknown')})"
            logger.debug(f"[HotMem] Using standard processing ({fallback_reason})")
            bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, intent=None)

        try:
            
            # Log what we extracted
            if triples:
                logger.debug(f"[HotMem] Extracted {len(triples)} facts (showing up to 3): {triples[:3]}")
            
            # Stash bullets to inject just before the aggregated user message
            if bullets:
                logger.debug(f"[HotMem] Prepared {len(bullets)} memory bullets for injection")
                cap = max(0, self._bullets_max)
                self._pending_bullets = bullets[:cap]
            else:
                self._pending_bullets = []

            # Store conversation text for retrieval (needed for summarization and optional FTS)
            # LAYER 2 DEFENSE: Apply quality filter before storing
            try:
                if text.strip() and self._is_quality_conversation(text):
                    now_ts = int(time.time() * 1000)
                    # Store with user_id for FTS indexing and cross-session retrieval
                    # session_id is passed as 4th parameter for session tracking
                    self.store.enqueue_mention(self._user_id, text.strip(), now_ts, self._session_id, self._turn_id)
                    self.store.flush_if_needed()
                elif text.strip():
                    logger.debug(f"[HotMem] Filtered out low-quality conversation: '{text[:50]}...'")
            except Exception as e:
                logger.warning(f"[HotMem] Storing conversation failed: {e}")

            # Trigger turn-based summary if configured
            if self._summary_enabled and self._window_mode == "turn_pairs":
                if self._turn_id > 0 and self._turn_id % self._turn_pairs == 0:
                    logger.info(f"[HotMem] Triggering turn-based summary at turn {self._turn_id}")
                    asyncio.create_task(self._generate_turn_summary())
                else:
                    logger.debug(f"[HotMem] Not triggering summary: turn={self._turn_id}, pairs={self._turn_pairs}, mod={self._turn_id % self._turn_pairs}")

            # Track performance
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if self._enable_metrics:
                self._log_metrics(elapsed_ms)
            # Memory summary lines (observability)
            logger.debug(f"[HotMem] Summary: saved={len(triples)}, pending_bullets={len(self._pending_bullets)}, turn={self._turn_id}")
            self._record_turn_metrics(elapsed_ms)
            try:
                src = "interim" if self._turn_has_preinjected_bullets else "final"
                injected_count = len(self._last_injected_bullets) if self._last_injected_bullets else len(self._pending_bullets)
                logger.debug(
                    f"[HotMem TurnSummary] pre_injected={self._turn_has_preinjected_bullets} ready_signaled={self._turn_ready_signaled} source={src} bullets={injected_count} total_ms={elapsed_ms:.1f}"
                )
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Memory processing failed: {e}")
            # Don't crash the pipeline on memory errors
    
    async def _inject_memory_context(self):
        """Inject memory bullets directly into the context aggregator's context"""
        try:
            if not self._context_aggregator:
                logger.warning("[HotMem] No context aggregator available for injection")
                return

            # Get the context object from the user aggregator
            context = self._context_aggregator.user().context
            messages = list(context.get_messages())

            # Use ContextFormatter for better bullet cleaning and formatting
            bullets = self._context_formatter.format_bullets(
                self._pending_bullets,
                max_bullets=getattr(self, "_bullets_max", 3)
            )
            # Optionally truncate bullets to fit within reasonable context length
            bullets = self._context_formatter.truncate_bullets(bullets, max_length=500)

            # Build the memory message using the formatter
            memory_message = self._context_formatter.build_message(
                self._inject_role,
                self._inject_header,
                bullets
            )

            logger.debug(f"[HotMem] Injecting {len(bullets)} memory bullets directly into context")
            try:
                if len(bullets) <= 5:
                    preview = ", ".join(bullets)
                else:
                    preview = ", ".join(bullets[:3]) + f" ... (+{len(bullets) - 3} more)"
                logger.debug(f"[HotMem] Memory bullets: {preview}")
            except Exception:
                # Fallback to previous logging behavior on any error
                logger.debug(f"[HotMem] Memory bullets: {bullets[:2]}")

            target_idx = self._find_context_message(messages, self._inject_header)
            if bullets and memory_message:
                if target_idx is None:
                    # Insert after persona prompt, before conversation history
                    insert_idx = self._persona_prompt_index(messages)
                    messages.insert(insert_idx, memory_message)
                    logger.debug(f"[HotMem] Inserted memory context at position {insert_idx} (after persona prompt)")
                else:
                    messages[target_idx] = memory_message
                    logger.debug(f"[HotMem] Updated existing memory context at position {target_idx}")
            else:
                if target_idx is not None:
                    messages.pop(target_idx)

            context.set_messages(messages)

            # Prune context window to keep conversation "forever" with a rolling window
            try:
                if self._ctx_window_enabled:
                    pruned = self._prune_context_window(list(context.get_messages()))
                    if pruned is not None:
                        context.set_messages(pruned)
            except Exception:
                pass
            # Also emit a typed frame for downstream processors (future-proof, non-breaking)
            try:
                await self.push_frame(MemoryContextFrame(self._inject_role, self._inject_header, bullets), None)
            except Exception:
                pass
            
            # Clear pending bullets after injection
            self._pending_bullets = []
            
        except Exception as e:
            logger.error(f"[HotMem] Failed to inject memory context: {e}")

    def _prune_context_window(self, messages: list) -> list:
        """Keep all system messages and the last N user/assistant turns.

        N is defined as CONTEXT_MAX_TURN_PAIRS. This keeps the agent responsive
        while allowing the conversation to continue indefinitely.
        """
        try:
            max_pairs = max(int(self._ctx_max_pairs), 0)
        except Exception:
            max_pairs = 4
        if max_pairs <= 0:
            return messages

        # Identify indices of user/assistant messages
        ua_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get('role') in ('user', 'assistant')]
        keep_ua = set(ua_indices[-2 * max_pairs:])

        pruned = []
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            role = m.get('role')
            if role == 'system' or i in keep_ua:
                pruned.append(m)
        return pruned
    
    def _log_metrics(self, elapsed_ms: float):
        """Log performance metrics periodically"""
        now = time.time()
        
        # Log every 30 seconds
        if now - self._last_metrics_log > 30:
            metrics = self.hot.get_metrics()
            store_metrics = self.store.get_metrics()
            
            logger.debug(f"HotMem metrics - Total: {elapsed_ms:.1f}ms")
            
            for key, stats in metrics.items():
                if isinstance(stats, dict) and 'p95' in stats:
                    logger.debug(f"  {key}: p95={stats['p95']:.1f}ms, mean={stats['mean']:.1f}ms")
                else:
                    logger.debug(f"  {key}: {stats}")
            
            for key, stats in store_metrics.items():
                if isinstance(stats, dict) and 'p95' in stats:
                    logger.debug(f"  Store {key}: p95={stats['p95']:.1f}ms")
            
            self._last_metrics_log = now
            
            # Warn if we're exceeding budget
            if 'total_ms' in metrics and metrics['total_ms'].get('p95', 0) > 200:
                logger.warning(f"HotMem exceeding 200ms budget: p95={metrics['total_ms']['p95']:.1f}ms")

    def _record_turn_metrics(self, elapsed_ms: float) -> None:
        if not self._session_tracker:
            return
        stats = self._session_tracker.record_turn(self._user_id, self._session_id, elapsed_ms / 1000.0)
        self._ensure_session_header(stats=stats)

    def _ensure_session_header(self, *, stats: Optional[Dict[str, Any]] = None, initial: bool = False) -> None:
        # In ephemeral/anonymous mode, do not inject or keep a session header
        if getattr(self, "_ephemeral", False):
            try:
                if not self._context_aggregator:
                    return
                context = self._context_aggregator.user().context
                messages = list(context.get_messages())
                # Remove any existing session header
                filtered = []
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system":
                        content = m.get("content", "")
                        if isinstance(content, str) and content.startswith(self._session_header_tag):
                            # skip
                            continue
                    filtered.append(m)
                context.set_messages(filtered)
            except Exception:
                pass
            return

        if not self._context_aggregator:
            return
        if self._session_tracker is None and stats is None:
            return
        context = self._context_aggregator.user().context
        messages = list(context.get_messages())
        
        # Get session stats if not provided
        if stats is None and self._session_tracker is not None:
            stats = self._session_tracker.get_stats(self._user_id, self._session_id)
        elif self._session_tracker is None:
            logger.debug("[HotMem] No session tracker available")
        
        header_message = self._build_session_header(stats)
        if not header_message:
            return
        idx = self._find_context_message(messages, self._session_header_tag)
        if idx is None:
            messages.insert(0, header_message)
        else:
            messages[idx] = header_message
        context.set_messages(messages)

    def _build_session_header(self, stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            current_turn = int(stats.get("session_turns", self._turn_id))
            total_turns = int(stats.get("total_turns", current_turn))
            total_sessions = int(stats.get("total_sessions", 1))
            session_start = stats.get(
                "session_start_iso",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._session_start)),
            )
            session_elapsed = float(stats.get("session_elapsed", time.time() - self._session_start))
            total_time = float(stats.get("total_time_seconds", session_elapsed))
            
            # Debug logging (can be removed after fixing the regression)
            if total_sessions == 0:
                logger.warning(f"[HotMem] Session regression detected - total_sessions=0 for {self._user_id}/{self._session_id}")
                logger.debug(f"[HotMem] Available stats: {stats}")
            else:
                logger.debug(f"[HotMem] Building session header - total_sessions: {total_sessions}, current_session: {stats.get('current_session')}")
            # Use coarse-grained values to preserve LLM KV cache
            # Only show date without time to avoid cache invalidation
            system_date = time.strftime("%Y-%m-%d")

            # Round session durations to nearest 5 minutes to reduce cache invalidation
            session_minutes = int(session_elapsed / 60)
            session_minutes_rounded = (session_minutes // 5) * 5  # Round to nearest 5min
            total_minutes = int(total_time / 60)
            total_minutes_rounded = (total_minutes // 5) * 5

            # Show anonymous label in ephemeral mode
            if getattr(self, "_ephemeral", False):
                display_user = "anonymous"
            elif hasattr(self, "_display_user_id"):
                display_user = self._display_user_id  # Use the original case for display
            else:
                display_user = self._user_id

            lines = [
                self._session_header_tag,
                f"Date: {system_date}",
                f"User: {display_user}",
                f"Session #{int(stats.get('current_session', total_sessions))}",
                f"Total sessions: {total_sessions}",
            ]

            # Only add timing info if significant (>= 5 min)
            if session_minutes_rounded >= 5:
                lines.append(f"Session: ~{session_minutes_rounded}min")
            if total_minutes_rounded >= 5 and total_minutes_rounded != session_minutes_rounded:
                lines.append(f"Total time: ~{total_minutes_rounded}min")
            return {"role": "system", "content": "\n".join(lines)}
        except Exception as e:
            logger.warning(f"[HotMem] Failed to build session header: {e}")
            return None

    def _find_context_message(self, messages: List[dict], prefix: str) -> Optional[int]:
        for idx, msg in enumerate(messages):
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if msg.get("role") == "system" and isinstance(content, str) and content.startswith(prefix):
                return idx
        return None

    def _session_header_index(self, messages: List[dict]) -> int:
        idx = self._find_context_message(messages, self._session_header_tag)
        if idx is not None:
            return idx + 1
        return 1 if len(messages) > 1 else len(messages)

    def _persona_prompt_index(self, messages: List[dict]) -> int:
        """Find insertion point after persona prompt, before conversation history.

        Returns the index where memory context should be inserted:
        - After the AI persona/system prompt
        - Before the conversation history (user/assistant exchanges)

        P0.4: Explicit ordering replaces keyword heuristics for robustness.
        Correct order: Session Header → Persona Prompt → [Memory Context] → Conversation
        """
        # P0.4: Use explicit ordering instead of keyword matching
        # Phase 1: Identify system message indices (excluding known special messages)
        session_header_idx = None
        memory_context_idx = None
        persona_prompt_idx = None

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get('role') != 'system':
                continue

            content = msg.get('content', '').strip()

            # Identify special system messages by their headers
            if content.startswith(self._session_header_tag):
                session_header_idx = i
            elif content.startswith(self._inject_header):
                memory_context_idx = i
            elif persona_prompt_idx is None:
                # First unrecognized system message is the persona prompt
                # (assumes persona prompt doesn't start with session/memory headers)
                persona_prompt_idx = i

        # Phase 2: Determine insertion point based on explicit ordering
        # Memory context should go after session header and persona prompt
        if persona_prompt_idx is not None:
            # Insert immediately after persona prompt
            return persona_prompt_idx + 1
        elif session_header_idx is not None:
            # No persona found, insert after session header
            return session_header_idx + 1

        # Phase 3: Fallback - insert before first user message (after all system messages)
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return i

        # Last resort: append to end (empty conversation)
        return len(messages)

    def _format_duration(self, seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}h {mins}m {secs}s"
        if mins:
            return f"{mins}m {secs}s"
        return f"{secs}s"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return {
            'turn_id': self._turn_id,
            'session_id': self._session_id,
            'hot_metrics': self.hot.get_metrics(),
            'store_metrics': self.store.get_metrics()
        }

    def refresh_session_header(self):
        """Public helper to refresh session header on demand."""
        self._ensure_session_header()

    async def cleanup(self):
        """Cleanup when processor is destroyed"""
        try:
            # Generate final summary if we have unsummarized turns
            if (self._summary_enabled and
                self._turn_id > 1 and
                self._turn_id > self._last_summarized_turn):
                logger.info(f"[HotMem] Generating final summary for session (turns {self._last_summarized_turn+1} to {self._turn_id})")
                try:
                    await asyncio.wait_for(self._generate_turn_summary(), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("[HotMem] Final summary generation timed out")

            # Final flush to ensure all data is persisted
            self.store.flush()
            logger.debug("HotPathMemoryProcessor cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

        # Stop summarizer task (for delta mode)
        try:
            if self._summary_task is not None:
                self._summary_task.cancel()
                self._summary_task = None
        except Exception:
            pass
        if self._session_tracker:
            try:
                self._session_tracker.end_session(self._user_id, self._session_id)
            except Exception:
                pass

    # ---------------------
    # Background summarizer
    # ---------------------

    async def _call_summarizer_llm(self, text: str) -> Optional[str]:
        """Call the summarizer LLM and return the summary content"""
        import json
        import urllib.request
        import urllib.error

        # LAYER 3 DEFENSE: Improved prompt to extract facts, not confusion
        sys_prompt = """You are a fact-focused summarizer. Extract only clear, actionable facts from the user's utterances.

RULES:
1. Extract ONLY positive facts (what they like, want, have, do)
2. IGNORE negations, confusion, and meta-commentary
3. Skip phrases like "I don't know", "confusing", "unclear"
4. Focus on names, preferences, experiences, and concrete details
5. Keep under 400 characters

EXAMPLES:
Input: "I'm not interested in this classic detail, it's confusing"
Output: [skip - only negation and confusion]

Input: "My name is John and I love hiking in Colorado"
Output: "User's name is John. Loves hiking in Colorado."

Input: "I have a dog named Max and I'm not sure about cats"
Output: "Has a dog named Max."

Provide ONLY the final summary or nothing if no clear facts exist."""

        # Build OpenAI-compatible chat request
        payload = {
            "model": self._summary_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": self._summary_max_tokens,
            "temperature": 0.2,
            "stream": False,
        }

        url = f"{self._summary_base_url}/chat/completions"
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._summary_api_key:
            req.add_header("Authorization", f"Bearer {self._summary_api_key}")

        data = json.dumps(payload).encode("utf-8")
        try:
            timeout = 5  # Use a short timeout for LLM calls
            with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
                resp_data = resp.read().decode("utf-8")
            j = json.loads(resp_data)
            content = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content if content else None
        except urllib.error.URLError as e:
            logger.warning(f"[HotMem] Summarizer LLM call failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"[HotMem] Summarizer LLM error: {e}")
            return None

    async def _generate_turn_summary(self):
        """Generate summary for recent turns"""
        logger.info(f"[HotMem] _generate_turn_summary started")
        try:
            # Calculate how many messages to summarize
            messages_to_get = self._turn_pairs * 2  # Each turn has user + assistant
            logger.info(f"[HotMem] Getting {messages_to_get} recent messages for session {self._session_id}")

            # Get recent messages
            recent = self.store.get_recent_chunks_by_eid(self._session_id, limit=messages_to_get)
            logger.debug(f"[HotMem] Found {len(recent)} messages to summarize")
            if not recent:
                logger.debug("[HotMem] No recent messages to summarize")
                return

            # Combine text (limit to 1200 chars)
            text = "; ".join(t for (t, _ts) in recent if t)[:1200]
            logger.info(f"[HotMem] Combined text for summary ({len(text)} chars): {text[:100]}...")
            if not text.strip():
                logger.info("[HotMem] No text content to summarize")
                return

            # Call LLM to generate summary
            content = await self._call_summarizer_llm(text)
            if content:
                now_ms = int(time.time() * 1000)
                note = f"Summary: {content}"
                self.store.enqueue_mention("summary", note, now_ms, self._session_id, self._turn_id)
                self.store.flush_if_needed()
                logger.debug(f"[HotMem] Stored turn-based summary at turn {self._turn_id}")
                self._last_summarized_turn = self._turn_id
        except Exception as e:
            logger.warning(f"[HotMem] Turn summary generation failed: {e}")

    async def _summary_loop(self):
        """Periodic background task for delta mode - generates LLM summaries periodically."""
        while True:
            try:
                await asyncio.sleep(self._summary_interval_secs)
                # Collect recent user utterances
                recent = self.store.get_recent_chunks_by_eid(self._session_id, limit=self._summary_max_messages)
                if not recent:
                    continue
                text = "; ".join(t for (t, _ts) in recent if t)[:1200]
                if not text.strip():
                    continue

                # Call LLM to generate summary
                content = await self._call_summarizer_llm(text)
                if content:
                    now_ms = int(time.time() * 1000)
                    note = f"Summary: {content}"
                    self.store.enqueue_mention("summary", note, now_ms, self._session_id, self._turn_id)
                    self.store.flush_if_needed()
                    logger.debug("[HotMem] Stored time-based summary (delta mode)")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[HotMem] Summary loop error: {e}")


# For backward compatibility and testing
class TestMemoryProcessor(HotPathMemoryProcessor):
    """Test version with additional debugging"""
    
    def __init__(self, **kwargs):
        kwargs['enable_metrics'] = True
        super().__init__(**kwargs)
        logger.debug("TestMemoryProcessor initialized with debugging enabled")
    
    async def _process_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Enhanced processing with detailed logging"""
        logger.debug(f"Processing: '{frame.text}'")
        await super()._process_transcription(frame, direction)
        
        # Log current memory state
        stats = self.get_memory_stats()
        logger.debug(f"Memory state: {stats['hot_metrics'].get('entities', 0)} entities tracked")


# Optional handshake frame indicating memory context is ready for the turn
class MemoryContextReadyFrame(Frame):
    pass
