"""
LocalCat: Pipecat processor that injects ultra-fast memory bullets
Place between context_aggregator.user() and llm in your Pipeline
"""

import time
from typing import List, Optional, Dict, Any
from loguru import logger

import sys
import os
# Add local pipecat to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, TextFrame, StartFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor as BaseProcessor, FrameDirection

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory
from memory.context import format_bullets as _fmt_bullets, build_message as _build_msg, MemoryContextFrame

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
                 context_aggregator = None):
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
        self._session_id = user_id
        self._enable_metrics = enable_metrics
        self._pending_bullets: List[str] = []
        # Phase 0 state: track one-time interim pre-injection per turn
        self._turn_has_preinjected_bullets: bool = False
        self._last_injected_bullets: List[str] = []
        # Env-driven controls (Phase 0.5)
        self._enabled: bool = os.getenv("ENABLE_MEMORY", "true").lower() in ("1", "true", "yes")
        try:
            self._bullets_max: int = int(os.getenv("HOTMEM_BULLETS_MAX", "3"))
        except Exception:
            self._bullets_max = 3
        try:
            self._interim_min_words: int = int(os.getenv("HOTMEM_INTERIM_MIN_WORDS", "6"))
        except Exception:
            self._interim_min_words = 6
        self._inject_role = os.getenv("HOTMEM_INJECT_ROLE", "user").strip().lower()
        if self._inject_role not in ("user", "system"):
            self._inject_role = "user"
        self._inject_header = os.getenv("HOTMEM_INJECT_HEADER", "[Memory context]")
        self._trace_frames = os.getenv("HOTMEM_TRACE_FRAMES", "false").lower() in ("1", "true", "yes")
        self._handshake_enabled = os.getenv("HOTMEM_ENABLE_HANDSHAKE", "true").lower() in ("1", "true", "yes")
        # Retrieval source controls (Phase 2-ready; used now for convo indexing)
        self._memory_sources = [s.strip() for s in os.getenv("MEMORY_SOURCES", "graph").split(",") if s.strip()]
        self._convo_index_enabled = os.getenv("MEMORY_CONVO_INDEX", "false").lower() in ("1", "true", "yes")
        
        # Store context aggregator reference for direct context injection
        self._context_aggregator = context_aggregator
        
        if self._trace_frames:
            logger.info(f"[HotMem] Frame tracing ENABLED - will log all frames flowing through processor")
        
        # Performance tracking
        self._last_metrics_log = time.time()
        
        logger.info(f"HotPathMemoryProcessor initialized for user: {user_id}")
    
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

        # If memory is disabled, simply forward
        if not self._enabled:
            await self.push_frame(frame, direction)
            return

        # REQUIRED: handle StartFrame immediately
        if isinstance(frame, StartFrame):
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
                        preview = self.hot.retrieve_bullets(text, read_only=True)
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
                            logger.info(f"[HotMem] Interim pre-injection completed with {len(self._last_injected_bullets)} bullets")
                            if self._handshake_enabled:
                                try:
                                    await self.push_frame(MemoryContextReadyFrame(), direction)
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"[HotMem] Interim pre-injection error: {e}")

        # Process final transcriptions (compute bullets, update store)
        if isinstance(frame, TranscriptionFrame):
            is_final = getattr(frame, 'is_final', None)
            text = getattr(frame, 'text', '') or ''
            logger.info(f"[HotMem] TranscriptionFrame received: is_final={is_final} text_len={len(text)} text='{text[:120]}'")
            # WhisperSTTServiceMLX doesn't set is_final, so treat None as final (non-streaming)
            if is_final is True or is_final is None:
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
                            logger.info(f"[HotMem] Final injection {'refreshed' if self._turn_has_preinjected_bullets else 'inserted'} with {len(new_bullets)} bullets")
                            if self._handshake_enabled:
                                try:
                                    await self.push_frame(MemoryContextReadyFrame(), direction)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"[HotMem] Final injection error: {e}")
                # Reset pre-injection state for next turn
                self._turn_has_preinjected_bullets = False
                self._last_injected_bullets = []
            else:
                logger.info(f"[HotMem] Skipping non-final transcription")

        # Legacy LLMMessagesFrame handling removed - now using direct context injection

        # REQUIRED: always forward the original frame
        await self.push_frame(frame, direction)
    
    async def _process_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Process final user transcription"""
        if not getattr(self, "_enabled", True):
            return
        self._turn_id += 1
        text = frame.text or ""
        
        if not text.strip():
            return
        
        start = time.perf_counter()
        
        try:
            # Extract facts and retrieve relevant memories
            bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id)
            
            # Log what we extracted
            if triples:
                logger.info(f"[HotMem] Extracted {len(triples)} facts (showing up to 3): {triples[:3]}")
            
            # Stash bullets to inject just before the aggregated user message
            if bullets:
                logger.info(f"[HotMem] Prepared {len(bullets)} memory bullets for injection")
                cap = max(0, self._bullets_max)
                self._pending_bullets = bullets[:cap]

            # Optional: index conversation text into FTS for convo retrieval
            try:
                if self._convo_index_enabled and text.strip():
                    now_ts = int(time.time() * 1000)
                    self.store.enqueue_mention(self._session_id, text.strip(), now_ts, self._session_id, self._turn_id)
                    self.store.flush_if_needed()
            except Exception as e:
                logger.warning(f"[HotMem] Convo index failed: {e}")
            
            # Track performance
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if self._enable_metrics:
                self._log_metrics(elapsed_ms)
            # Memory summary line
            logger.info(f"[HotMem] Summary: saved={len(triples)}, pending_bullets={len(self._pending_bullets)}, turn={self._turn_id}")
                
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
            # Normalize bullets with current cap
            bullets = _fmt_bullets(self._pending_bullets, max_bullets=getattr(self, "_bullets_max", 3))
            memory_message = _build_msg(self._inject_role, self._inject_header, bullets)
            
            logger.info(f"[HotMem] Injecting {len(self._pending_bullets)} memory bullets directly into context")
            logger.info(f"[HotMem] Memory bullets: {bullets[:2]}")
            
            # Add memory message to context before the user message gets added
            context.add_message(memory_message)
            # Also emit a typed frame for downstream processors (future-proof, non-breaking)
            try:
                await self.push_frame(MemoryContextFrame(self._inject_role, self._inject_header, bullets), None)
            except Exception:
                pass
            
            # Clear pending bullets after injection
            self._pending_bullets = []
            
        except Exception as e:
            logger.error(f"[HotMem] Failed to inject memory context: {e}")
    
    def _log_metrics(self, elapsed_ms: float):
        """Log performance metrics periodically"""
        now = time.time()
        
        # Log every 30 seconds
        if now - self._last_metrics_log > 30:
            metrics = self.hot.get_metrics()
            store_metrics = self.store.get_metrics()
            
            logger.info(f"HotMem metrics - Total: {elapsed_ms:.1f}ms")
            
            for key, stats in metrics.items():
                if isinstance(stats, dict) and 'p95' in stats:
                    logger.info(f"  {key}: p95={stats['p95']:.1f}ms, mean={stats['mean']:.1f}ms")
                else:
                    logger.info(f"  {key}: {stats}")
            
            for key, stats in store_metrics.items():
                if isinstance(stats, dict) and 'p95' in stats:
                    logger.info(f"  Store {key}: p95={stats['p95']:.1f}ms")
            
            self._last_metrics_log = now
            
            # Warn if we're exceeding budget
            if 'total_ms' in metrics and metrics['total_ms'].get('p95', 0) > 200:
                logger.warning(f"HotMem exceeding 200ms budget: p95={metrics['total_ms']['p95']:.1f}ms")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return {
            'turn_id': self._turn_id,
            'session_id': self._session_id,
            'hot_metrics': self.hot.get_metrics(),
            'store_metrics': self.store.get_metrics()
        }
    
    async def cleanup(self):
        """Cleanup when processor is destroyed"""
        try:
            # Final flush to ensure all data is persisted
            self.store.flush()
            logger.info("HotPathMemoryProcessor cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# For backward compatibility and testing
class TestMemoryProcessor(HotPathMemoryProcessor):
    """Test version with additional debugging"""
    
    def __init__(self, **kwargs):
        kwargs['enable_metrics'] = True
        super().__init__(**kwargs)
        logger.info("TestMemoryProcessor initialized with debugging enabled")
    
    async def _process_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Enhanced processing with detailed logging"""
        logger.info(f"Processing: '{frame.text}'")
        await super()._process_transcription(frame, direction)
        
        # Log current memory state
        stats = self.get_memory_stats()
        logger.info(f"Memory state: {stats['hot_metrics'].get('entities', 0)} entities tracked")


# Optional handshake frame indicating memory context is ready for the turn
class MemoryContextReadyFrame(Frame):
    pass
