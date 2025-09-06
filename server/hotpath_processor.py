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

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, TextFrame, StartFrame
try:
    # Prefer update frame for rotating system message (Option B)
    from pipecat.frames.frames import LLMMessagesUpdateFrame
except Exception:
    LLMMessagesUpdateFrame = None  # type: ignore
from pipecat.processors.frame_processor import FrameProcessor as BaseProcessor, FrameDirection

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory

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
        # Canonicalize storage paths relative to this file when relative
        base_dir = os.path.dirname(__file__)
        if sqlite_path and not os.path.isabs(sqlite_path):
            sqlite_path = os.path.abspath(os.path.join(base_dir, sqlite_path))
        if lmdb_dir and not os.path.isabs(lmdb_dir):
            lmdb_dir = os.path.abspath(os.path.join(base_dir, lmdb_dir))
        paths = Paths(sqlite_path=sqlite_path, lmdb_dir=lmdb_dir)
        self.store = MemoryStore(paths)
        try:
            logger.info(f"HotMem storage: sqlite={self.store.paths.sqlite_path} lmdb={self.store.paths.lmdb_dir}")
        except Exception:
            pass
        
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
        self._inject_role = os.getenv("HOTMEM_INJECT_ROLE", "system").strip().lower()
        if self._inject_role not in ("user", "system"):
            self._inject_role = "user"
        self._inject_header = os.getenv("HOTMEM_INJECT_HEADER", "Use the following factual context if helpful.")
        # Caps
        try:
            self._bullets_max = int(os.getenv("HOTMEM_BULLETS_MAX", "5"))
        except Exception:
            self._bullets_max = 5
        self._trace_frames = os.getenv("HOTMEM_TRACE_FRAMES", "false").lower() in ("1", "true", "yes")
        
        # Store context aggregator reference for direct context injection
        self._context_aggregator = context_aggregator
        
        if self._trace_frames:
            logger.info(f"[HotMem] Frame tracing ENABLED - will log all frames flowing through processor")
        
        # Performance tracking
        self._last_metrics_log = time.time()
        
        logger.info(f"HotPathMemoryProcessor initialized for user: {user_id}")

    def _human_time_str(self, ts_ms: int) -> str:
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(ts_ms / 1000).astimezone()
            def _ord(n: int) -> str:
                if 10 <= n % 100 <= 20:
                    suf = "th"
                else:
                    suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                return f"{n}{suf}"
            time_str = dt.strftime('%I:%M %p').lstrip('0').lower()
            return f"{time_str}, {dt.strftime('%A')} {_ord(dt.day)} {dt.strftime('%B')} {dt.year}"
        except Exception:
            return str(ts_ms)

    def build_session_summary(self, max_bullets: int = 5) -> str:
        """Build a compact, human-readable session summary from recent facts.

        Returns a multi-line string suitable for FTS/semantic indexing.
        """
        items = list(self.hot.recency_buffer)
        if not items:
            return ""
        start_ts = items[0].timestamp
        end_ts = items[-1].timestamp
        # Deduplicate triples preserving recency
        seen = set()
        bullets = []
        for it in reversed(items):
            key = (it.s, it.r, it.d)
            if key in seen:
                continue
            seen.add(key)
            bullets.append(self.hot._format_memory_bullet(it.s, it.r, it.d))
            if len(bullets) >= max_bullets:
                break
        bullets = list(reversed(bullets))
        header = f"Session summary ({self._human_time_str(start_ts)} → {self._human_time_str(end_ts)}):"
        body = "\n".join(bullets)
        return f"{header}\n{body}"

    def persist_session_summary(self) -> Optional[str]:
        """Persist the session summary into FTS via mention; return text if written."""
        try:
            text = self.build_session_summary()
            if not text:
                return None
            now_ts = int(time.time() * 1000)
            eid = f"session:{self._session_id}"
            self.store.enqueue_mention(eid=eid, text=text, ts=now_ts, sid=self._session_id, tid=self._turn_id)
            self.store.flush()
            logger.info("[HotMem] Persisted session summary to FTS")
            return text
        except Exception as e:
            logger.warning(f"[HotMem] Failed to persist session summary: {e}")
            return None
    
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

        # Process final transcriptions (compute bullets, update store)  
        if isinstance(frame, TranscriptionFrame):
            is_final = getattr(frame, 'is_final', None)
            text = getattr(frame, 'text', '') or ''
            logger.info(f"[HotMem] TranscriptionFrame received: is_final={is_final} text_len={len(text)} text='{text[:120]}'")
            # WhisperSTTServiceMLX doesn't set is_final, so treat None as final (non-streaming)
            if is_final is True or is_final is None:
                # Recap-now command: inject latest summary on demand
                try:
                    t = text.strip().lower()
                    if any(kw in t for kw in ("recap", "summary", "summarize", "quick recap")):
                        await self._inject_summary_context(direction)
                except Exception as e:
                    logger.warning(f"[HotMem] Recap injection failed: {e}")
                logger.info(f"[HotMem] Processing transcription (is_final={is_final}): '{text}'")
                await self._process_transcription(frame, direction)
                
                # Inject memory bullets directly into context before the aggregator processes the frame
                if self._pending_bullets and self._context_aggregator:
                    await self._inject_memory_context(direction)
            else:
                logger.info(f"[HotMem] Skipping non-final transcription")

        # Legacy LLMMessagesFrame handling removed - now using direct context injection

        # REQUIRED: always forward the original frame
        await self.push_frame(frame, direction)
    
    async def _process_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Process final user transcription"""
        self._turn_id += 1
        text = frame.text or ""
        
        if not text.strip():
            return
        
        start = time.perf_counter()
        
        try:
            # Check for correction intent before normal processing
            correction_result = await self._handle_correction_intent(text)
            
            # Extract facts and retrieve relevant memories
            bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id)
            
            # Log what we extracted
            if triples:
                logger.info(f"[HotMem] Extracted {len(triples)} facts (showing up to 3): {triples[:3]}")
            
            # Add correction feedback to bullets if correction was applied
            if correction_result and correction_result.get('success'):
                correction_bullet = f"• ✅ Correction applied: {correction_result.get('explanation', 'Memory updated')}"
                bullets = [correction_bullet] + bullets[:4]  # Keep within 5 bullet limit
            
            # Stash bullets to inject just before the aggregated user message
            if bullets:
                logger.info(f"[HotMem] Prepared {len(bullets)} memory bullets for injection")
                # Dynamic 1..5 already applied by HotMemory; cap by env
                self._pending_bullets = bullets[: self._bullets_max]
            
            # Track performance
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if self._enable_metrics:
                self._log_metrics(elapsed_ms)
            # Memory summary line
            logger.info(f"[HotMem] Summary: saved={len(triples)}, pending_bullets={len(self._pending_bullets)}, turn={self._turn_id}")
                
        except Exception as e:
            logger.error(f"Memory processing failed: {e}")
            # Don't crash the pipeline on memory errors
    
    async def _handle_correction_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """Handle correction intent detection and application"""
        try:
            from memory_correction import get_corrector
            corrector = get_corrector()
            
            # Detect correction intent
            instruction = corrector.detect_correction_intent(text)
            if not instruction:
                return None
                
            logger.info(f"[HotMem] Detected {instruction.correction_type.value} correction in {instruction.language}")
            
            # Apply correction to memory
            result = corrector.apply_correction(instruction, self.hot)
            
            if result.get('success'):
                logger.info(f"[HotMem] Correction applied: {result.get('explanation')}")
            else:
                logger.warning(f"[HotMem] Correction failed: {result.get('error')}")
                
            return result
            
        except ImportError:
            logger.warning("[HotMem] Correction system not available - spaCy not installed")
            return None
        except Exception as e:
            logger.error(f"[HotMem] Correction handling failed: {e}")
            return None
    
    async def _inject_memory_context(self, direction: FrameDirection):
        """Inject memory bullets directly into the context aggregator's context"""
        try:
            if not self._context_aggregator:
                logger.warning("[HotMem] No context aggregator available for injection")
                return
                
            # Only inject bullets relevant to this turn (no sliding window)
            bullets_final = self._pending_bullets[: self._bullets_max]

            # Compose content with freshness header in human-friendly local format
            from datetime import datetime
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
            human_updated = f"{time_str}, {weekday} {day_ordinal} {month} {year}"
            memory_content = "\n".join(bullets_final)
            header = f"{self._inject_header}\nContext for this turn (updated: {human_updated}):"
            memory_message = {"role": self._inject_role, "content": f"{header}\n{memory_content}"}

            logger.info(f"[HotMem] Injecting rotating memory context with {len(bullets_final)} bullets")

            # Option B: Replace/update single tagged system message via update frame if available
            if LLMMessagesUpdateFrame is not None:
                context = self._context_aggregator.user().context
                # Read current messages
                try:
                    messages = list(getattr(context, 'messages', []))
                except Exception:
                    messages = []
                # Remove any prior HotMem message
                filtered = []
                for m in messages:
                    if isinstance(m, dict) and m.get('role') == self._inject_role:
                        content = m.get('content', '') or ''
                        if isinstance(content, str) and content.startswith(self._inject_header):
                            continue
                    filtered.append(m)
                # Insert memory_message right after the first system message so it precedes conversation
                insert_idx = 0
                for i, m in enumerate(filtered):
                    if isinstance(m, dict) and m.get('role') == 'system':
                        insert_idx = i + 1
                        break
                new_messages = filtered[:insert_idx] + [memory_message] + filtered[insert_idx:]
                try:
                    await self.push_frame(LLMMessagesUpdateFrame(new_messages), direction)
                except Exception as e:
                    logger.warning(f"[HotMem] LLMMessagesUpdateFrame failed ({e}); falling back to add_message")
                    context.add_message(memory_message)
            else:
                # Fallback: simply add a new message (may accumulate)
                context = self._context_aggregator.user().context
                context.add_message(memory_message)
            
            # Clear pending bullets after injection
            self._pending_bullets = []
            
        except Exception as e:
            logger.error(f"[HotMem] Failed to inject memory context: {e}")

    async def _inject_summary_context(self, direction: FrameDirection):
        """Fetch and inject the latest periodic/session summary as a system hint."""
        try:
            # Fetch latest summary from mention table
            cur = getattr(self.store, 'sql', None).cursor()
            sid = self._session_id
            rows = cur.execute(
                """
                SELECT text FROM mention
                WHERE eid IN (?, ?)
                ORDER BY ts DESC LIMIT 1
                """,
                (f"summary:{sid}", f"session:{sid}")
            ).fetchall()
            if not rows:
                return
            summary_text = str(rows[0][0] or '').strip()
            if not summary_text:
                return
            # Insert as a system message right after the first system
            context = self._context_aggregator.user().context
            try:
                messages = list(getattr(context, 'messages', []))
            except Exception:
                messages = []
            recap_header = "Recap from recent conversation:" 
            recap_message = {"role": "system", "content": f"{recap_header}\n{summary_text}"}
            # Remove any prior recap
            filtered = []
            for m in messages:
                if isinstance(m, dict) and m.get('role') == 'system':
                    content = m.get('content', '') or ''
                    if isinstance(content, str) and content.startswith(recap_header):
                        continue
                filtered.append(m)
            insert_idx = 0
            for i, m in enumerate(filtered):
                if isinstance(m, dict) and m.get('role') == 'system':
                    insert_idx = i + 1
                    break
            new_messages = filtered[:insert_idx] + [recap_message] + filtered[insert_idx:]
            if LLMMessagesUpdateFrame is not None:
                await self.push_frame(LLMMessagesUpdateFrame(new_messages), direction)
            else:
                context.add_message(recap_message)
            logger.info("[HotMem] Injected recap summary into context")
        except Exception as e:
            logger.warning(f"[HotMem] Failed to inject recap: {e}")
    
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
