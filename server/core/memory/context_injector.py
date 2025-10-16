"""
Context injection for memory bullets into LLM messages.

Handles formatting and injection of memory context into conversation.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from .config_manager import MemoryConfiguration
from .context_formatter import ContextFormatter


class ContextInjector:
    """
    Inject memory bullets into LLM context.

    Responsibilities:
    - Retrieve memory bullets
    - Format bullets for injection
    - Inject into appropriate role (system/user)
    - Track injection metrics
    - Handle context window pruning
    """

    def __init__(
        self,
        hot_memory,
        config: MemoryConfiguration,
        formatter: Optional[ContextFormatter] = None,
        context_aggregator=None
    ):
        self.hot = hot_memory
        self.config = config
        self.formatter = formatter or ContextFormatter(
            max_bullets=config.bullets_max,
            inject_role=config.inject_role,
            inject_header=config.inject_header
        )
        self.context_aggregator = context_aggregator
        
        # Injection metrics
        self._injection_count = 0
        self._last_injected_bullets: List[str] = []
        self._turn_has_preinjected_bullets: bool = False
        self._turn_ready_signaled: bool = False
        self._pending_bullets: List[str] = []

    async def inject_memory_context(self) -> bool:
        """
        Inject pending memory bullets directly into the context aggregator.

        Returns:
            True if injection was successful, False otherwise
        """
        try:
            if not self.context_aggregator:
                logger.warning("[ContextInjector] No context aggregator available for injection")
                return False

            if not self._pending_bullets:
                logger.debug("[ContextInjector] No pending bullets to inject")
                return False

            # Get the context object from the user aggregator
            context = self.context_aggregator.user().context
            messages = list(context.get_messages())

            # Use ContextFormatter for better bullet cleaning and formatting
            bullets = self.formatter.format_bullets(
                self._pending_bullets,
                max_bullets=self.config.bullets_max
            )
            
            # Optionally truncate bullets to fit within reasonable context length
            bullets = self.formatter.truncate_bullets(bullets, max_length=self.config.token_budget)

            # Build the memory message using the formatter
            memory_message = self.formatter.build_message(
                self.config.inject_role,
                self.config.inject_header,
                bullets
            )

            logger.debug(f"[ContextInjector] Injecting {len(bullets)} memory bullets directly into context")
            
            # Log bullet preview
            try:
                if len(bullets) <= 5:
                    preview = ", ".join(bullets)
                else:
                    preview = ", ".join(bullets[:3]) + f" ... (+{len(bullets) - 3} more)"
                logger.debug(f"[ContextInjector] Memory bullets: {preview}")
            except Exception:
                # Fallback to previous logging behavior on any error
                logger.debug(f"[ContextInjector] Memory bullets: {bullets[:2]}")

            # Find or create insertion point for memory context
            target_idx = self._find_context_message(messages, self.config.inject_header)
            if bullets and memory_message:
                if target_idx is None:
                    # Insert after persona prompt, before conversation history
                    insert_idx = self._persona_prompt_index(messages)
                    messages.insert(insert_idx, memory_message)
                    logger.debug(f"[ContextInjector] Inserted memory context at position {insert_idx} (after persona prompt)")
                else:
                    messages[target_idx] = memory_message
                    logger.debug(f"[ContextInjector] Updated existing memory context at position {target_idx}")
            else:
                # Remove existing memory context if no bullets
                if target_idx is not None:
                    messages.pop(target_idx)
                    logger.debug(f"[ContextInjector] Removed empty memory context at position {target_idx}")

            context.set_messages(messages)

            # Prune context window to keep conversation "forever" with a rolling window
            try:
                if self.config.ctx_window_enabled:
                    pruned = self._prune_context_window(list(context.get_messages()))
                    if pruned is not None:
                        context.set_messages(pruned)
                        logger.debug(f"[ContextInjector] Pruned context window")
            except Exception as e:
                logger.debug(f"[ContextInjector] Context pruning failed: {e}")

            # Clear pending bullets after injection
            self._pending_bullets = []
            self._injection_count += 1
            
            # Track injection state
            self._last_injected_bullets = list(bullets)
            self._turn_has_preinjected_bullets = True

            return True
            
        except Exception as e:
            logger.error(f"[ContextInjector] Failed to inject memory context: {e}")
            return False

    def inject_into_messages(self, messages: List[dict]) -> List[dict]:
        """
        Inject pending memory bullets into a provided messages list and return the updated list.

        Mirrors inject_memory_context but operates purely on the given messages,
        allowing callers with an LLMMessagesFrame to modify it in-place.
        """
        try:
            if not self._pending_bullets:
                return messages

            # Use ContextFormatter to prepare bullets
            bullets = self.formatter.format_bullets(
                self._pending_bullets,
                max_bullets=self.config.bullets_max
            )
            bullets = self.formatter.truncate_bullets(bullets, max_length=self.config.token_budget)

            memory_message = self.formatter.build_message(
                self.config.inject_role,
                self.config.inject_header,
                bullets
            )

            target_idx = self._find_context_message(messages, self.config.inject_header)
            if bullets and memory_message:
                if target_idx is None:
                    insert_idx = self._persona_prompt_index(messages)
                    messages.insert(insert_idx, memory_message)
                else:
                    messages[target_idx] = memory_message
            else:
                if target_idx is not None:
                    messages.pop(target_idx)

            # Update metrics and clear pending
            self._injection_count += 1
            self._last_injected_bullets = list(bullets)
            self._pending_bullets = []
            return messages
        except Exception as e:
            logger.debug(f"[ContextInjector] inject_into_messages failed: {e}")
            return messages

    async def retrieve_and_prepare_bullets(self, query: str, read_only: bool = True, intent: Optional[Dict] = None) -> List[str]:
        """
        Retrieve memory bullets for query and prepare them for injection.
        
        Args:
            query: Query text
            read_only: Whether to retrieve in read-only mode (no updates)
            intent: Optional intent classification result
            
        Returns:
            List of memory bullets
        """
        try:
            # Provide identity scope to retriever
            try:
                if hasattr(self.hot, 'current_session_id'):
                    self.hot.current_session_id = getattr(self.hot, 'current_session_id', None)
                if hasattr(self.hot, 'current_user_id'):
                    self.hot.current_user_id = getattr(self.hot, 'current_user_id', None)
            except Exception:
                pass

            # Retrieve bullets
            bullets = self.hot.retrieve_bullets(
                query=query,
                read_only=read_only,
                intent=intent
            )

            # Prepare bullets for injection
            if bullets:
                # Limit to configured maximum
                cap = max(0, self.config.bullets_max)
                prepared_bullets = bullets[:cap]
                self._pending_bullets = list(prepared_bullets)
                
                logger.debug(f"[ContextInjector] Prepared {len(prepared_bullets)} memory bullets for injection")
                return prepared_bullets
            else:
                self._pending_bullets = []
                return []

        except Exception as e:
            logger.error(f"[ContextInjector] Failed to retrieve bullets: {e}")
            self._pending_bullets = []
            return []

    def set_pending_bullets(self, bullets: List[str]) -> None:
        """
        Set pending bullets directly (useful for interim pre-injection).
        
        Args:
            bullets: List of bullets to set as pending
        """
        cap = max(0, self.config.bullets_max)
        self._pending_bullets = bullets[:cap]
        logger.debug(f"[ContextInjector] Set {len(self._pending_bullets)} pending bullets")

    def clear_pending_bullets(self) -> None:
        """Clear pending bullets"""
        self._pending_bullets = []
        logger.debug("[ContextInjector] Cleared pending bullets")

    def reset_turn_state(self) -> None:
        """Reset turn-specific state"""
        self._turn_has_preinjected_bullets = False
        self._last_injected_bullets = []
        self._turn_ready_signaled = False

    def _find_context_message(self, messages: List[dict], prefix: str) -> Optional[int]:
        """Find message with given prefix in content"""
        for idx, msg in enumerate(messages):
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if msg.get("role") == "system" and isinstance(content, str) and content.startswith(prefix):
                return idx
        return None

    def _persona_prompt_index(self, messages: List[dict]) -> int:
        """
        Find insertion point after persona prompt, before conversation history.

        Returns the index where memory context should be inserted:
        - After the AI persona/system prompt
        - Before the conversation history (user/assistant exchanges)
        """
        # Phase 1: Identify system message indices (excluding known special messages)
        session_header_idx = None
        memory_context_idx = None
        persona_prompt_idx = None

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get('role') != 'system':
                continue

            content = msg.get('content', '').strip()

            # Identify special system messages by their headers
            if content.startswith("[Session Context]"):
                session_header_idx = i
            elif content.startswith(self.config.inject_header):
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

    def _prune_context_window(self, messages: list) -> list:
        """
        Token-aware context pruning: Keep system messages + recent turns that fit budget.

        Strategy:
        1. Calculate token budget from config
        2. Keep all system messages (session, persona, memory)
        3. Calculate remaining budget for conversation history
        4. Keep most recent turn pairs that fit within budget
        5. Always keep minimum turns for coherence

        This prevents performance degradation in long conversations while maintaining
        context coherence through the memory system.
        """
        try:
            from .token_estimator import TokenEstimator

            max_tokens = self.config.llm_context_max_tokens or 3000
            prune_threshold = self.config.llm_context_prune_threshold or 0.70
            min_turns = max(self.config.llm_context_min_turns or 3, 1)

            # Calculate available budget (with safety margin)
            available_budget = int(max_tokens * prune_threshold)

            # Separate system vs user/assistant messages
            system_msgs = []
            ua_msgs = []

            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role')
                if role == 'system':
                    system_msgs.append(msg)
                elif role in ('user', 'assistant'):
                    ua_msgs.append(msg)

            # Calculate tokens in system messages (always kept)
            system_tokens = sum(TokenEstimator.estimate_message_tokens(m) for m in system_msgs)

            # Remaining budget for conversation history
            remaining_budget = available_budget - system_tokens

            if remaining_budget <= 0:
                logger.warning(f"[ContextInjector] System messages exceed token budget!")
                return system_msgs + ua_msgs[-2 * min_turns:]  # Keep minimum turns

            # Keep most recent turn pairs that fit within budget
            kept_ua = []
            current_tokens = 0

            # Iterate from most recent to oldest
            for msg in reversed(ua_msgs):
                msg_tokens = TokenEstimator.estimate_message_tokens(msg)
                if current_tokens + msg_tokens <= remaining_budget or len(kept_ua) < (2 * min_turns):
                    kept_ua.insert(0, msg)  # Prepend to maintain order
                    current_tokens += msg_tokens
                else:
                    break

            # Ensure we keep pairs (user + assistant)
            if len(kept_ua) % 2 == 1:
                kept_ua = kept_ua[:-1]  # Remove incomplete pair

            pruned_count = len(ua_msgs) - len(kept_ua)
            if pruned_count > 0:
                logger.info(f"[ContextInjector] Pruned {pruned_count} messages "
                           f"({system_tokens + current_tokens}/{available_budget} tokens used)")

            return system_msgs + kept_ua

        except Exception as e:
            logger.error(f"[ContextInjector] Token-aware pruning failed: {e}, using fallback")
            # Fallback to original message-count based pruning
            max_pairs = max(int(self.config.ctx_max_pairs), 0) or 4
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

    def get_injection_metrics(self) -> Dict[str, Any]:
        """Get injection metrics"""
        return {
            "injection_count": self._injection_count,
            "pending_bullets_count": len(self._pending_bullets),
            "last_injected_bullets_count": len(self._last_injected_bullets),
            "turn_has_preinjected": self._turn_has_preinjected_bullets,
            "turn_ready_signaled": self._turn_ready_signaled
        }

    def should_refresh_injection(self) -> bool:
        """Check if injection should be refreshed (different from last)"""
        if not self._pending_bullets:
            return False
            
        # Compare with last injected bullets
        new_bullets = list(self._pending_bullets)
        if not self._turn_has_preinjected_bullets or new_bullets != self._last_injected_bullets:
            return True
            
        return False

    def get_pending_bullets_count(self) -> int:
        """Get count of pending bullets"""
        return len(self._pending_bullets)

    def get_last_injected_bullets(self) -> List[str]:
        """Get last injected bullets"""
        return list(self._last_injected_bullets)

    def get_injection_count(self) -> int:
        """Get total number of successful injections performed."""
        return self._injection_count

    def mark_turn_ready(self) -> None:
        """Indicate that the turn-ready handshake has been signaled."""
        self._turn_ready_signaled = True

    def has_signaled_turn_ready(self) -> bool:
        """Return True if a turn-ready handshake was signaled this turn."""
        return self._turn_ready_signaled
