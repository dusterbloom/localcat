import os
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from .token_counter import get_global_counter
from .budget_manager import get_global_budget
from .memory_config import get_global_config
from .exceptions import PackingError, ValidationError


# Use centralized token counter
_token_counter = get_global_counter()

def _count_tokens_from_messages(msgs: List[Dict[str, str]]) -> int:
    """Count tokens in messages using the global token counter"""
    return _token_counter.count_messages(msgs)


def _first_system_index(messages: List[Dict[str, Any]]) -> int:
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "system":
            return i
    return 0


def _filter_old_injections(messages: List[Dict[str, Any]], headers: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = str(m.get("content", "") or "")
        if role in {"system", "user", "assistant"}:
            if any(content.startswith(h) for h in headers):
                # drop old injected block
                continue
        out.append(m)
    return out


def _clean_memory_context_from_content(content: str, inject_header: str) -> str:
    """Remove old memory context sections from system message content to prevent duplication."""
    import re

    # More comprehensive pattern to match memory context sections with variations
    # This handles both the exact format and variations with different endings
    memory_section_pattern = re.compile(
        f'({re.escape(inject_header)}.*?(?:Memory Guidance:.*?(?:as references|as user statements)[^\\n]*(?:\\n|$)|$))',
        re.DOTALL | re.IGNORECASE
    )

    # Find all memory context sections
    sections = list(memory_section_pattern.finditer(content))

    if len(sections) <= 1:
        # No duplicates found, return original content
        return content

    # Keep only the last (most recent) memory context section
    last_match = sections[-1]
    last_section = last_match.group(0)

    # Remove all memory context sections
    cleaned_content = memory_section_pattern.sub('', content).strip()

    # Find where to insert the last section (before Reasoning Guidance or at the end)
    reasoning_guidance_match = re.search(r'Reasoning Guidance:', cleaned_content)
    if reasoning_guidance_match:
        insert_pos = reasoning_guidance_match.start()
        # Insert two newlines before the section for proper spacing
        cleaned_content = (cleaned_content[:insert_pos].rstrip() + '\n\n' +
                          last_section + '\n\n' +
                          cleaned_content[insert_pos:])
    else:
        # Add at the end
        cleaned_content = cleaned_content.rstrip() + '\n\n' + last_section

    return cleaned_content


def _build_memory_message(bullets: List[str], header: str, role: str, progressive_mode: bool = True) -> Optional[Dict[str, str]]:
    if not bullets:
        return None

    body = "\n".join(bullets)
    config = get_global_config()

    # In progressive mode, add memory policies when memory is actually used
    if progressive_mode:
        memory_guidance = config.get_memory_guidance_text()
        content = f"{header}\n{body}{memory_guidance}"
    else:
        content = f"{header}\n{body}"

    return {"role": role, "content": content}


def _build_summary_message(summary_text: Optional[str], role: str) -> Optional[Dict[str, str]]:
    if not summary_text:
        return None
    text = (summary_text or "").replace("\n", " ").strip()
    if not text:
        return None
    # Keep snippet short to fit budget repeatedly
    if len(text) > 400:
        text = text[:400].rstrip() + "…"
    header = "Summary Context (recent):"
    return {"role": role, "content": f"{header}\n{text}"}


def _extract_session_info(memory_bullets: List[str]) -> Tuple[Optional[str], List[str]]:
    """Extract session context from memory bullets and return clean bullets."""
    if not memory_bullets:
        return None, []

    session_info = None
    clean_bullets = []

    for bullet in memory_bullets:
        if bullet.startswith("Session Context:"):
            session_info = bullet
        else:
            clean_bullets.append(bullet)

    return session_info, clean_bullets


def pack_context(
    messages: List[Dict[str, Any]],
    memory_bullets: List[str],
    summary_text: Optional[str],
    budget_tokens: int,
    inject_role: str = "system",
    inject_header: str = "Use the following factual context if helpful.",
    system_hint: Optional[str] = None,
    progressive_mode: bool = True,
    max_memory_tokens: int = 300,  # New: Pre-retrieval token cap for memory
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Pack context with strict token budget and clear section order.
    
    Order:
      1) System instruction (keep first system message)
      2) Memory Context (bullets) - only if memory_bullets is not empty and progressive_mode is True
      3) Summary Context (latest snippet) - only if summary_text exists and progressive_mode is True
      4) Conversation tail (last N messages within remainder)
    
    Args:
        progressive_mode: If True, only inject memory/summary headers when content exists
        max_memory_tokens: Hard cap on memory section tokens (prevents retrieval bloat)
    
    Returns: (messages, stats)
    
    Raises:
        PackingError: If context packing fails
        ValidationError: If inputs are invalid
    """
    try:
        # Input validation
        if messages is None:
            raise ValidationError("Messages cannot be None")

        if not isinstance(messages, list):
            raise ValidationError("Messages must be a list")

        if memory_bullets is None:
            memory_bullets = []

        if not isinstance(memory_bullets, list):
            raise ValidationError("Memory bullets must be a list")

        if budget_tokens is None or budget_tokens <= 0:
            raise ValidationError("Budget tokens must be a positive integer")

        if max_memory_tokens < 0:
            raise ValidationError("Max memory tokens cannot be negative")

        if not inject_role or inject_role not in ("system", "user", "assistant"):
            raise ValidationError("Inject role must be one of: system, user, assistant")

        if not inject_header or not inject_header.strip():
            raise ValidationError("Inject header cannot be empty")

        # Hard safety
        msgs = list(messages)
        if not msgs:
            return [], {"tokens_total": 0}

    except Exception as e:
        if isinstance(e, (ValidationError, PackingError)):
            raise
        raise PackingError(f"Failed to validate inputs: {e}")

    # Remove any prior injected memory/summary blocks
    prior_headers = [inject_header, "Summary Context (recent):", "Recap from recent conversation:"]
    msgs = _filter_old_injections(msgs, prior_headers)

    # Identify first system and split
    sys_idx = _first_system_index(msgs)
    system_msg = msgs[sys_idx]
    before = msgs[: sys_idx + 1]
    dialogue = msgs[sys_idx + 1 :]

    # Use centralized budget management
    budget_manager = get_global_budget()
    # Override total budget if provided
    if budget_tokens and budget_tokens != budget_manager.total_budget:
        budget_manager.total_budget = budget_tokens
    allocations = budget_manager.get_allocations()

    target_system = allocations.system
    target_memory = min(allocations.memory, max_memory_tokens)  # Enforce pre-retrieval cap
    target_summary = allocations.summary
    target_dialogue = allocations.dialogue

    # Extract session info from memory bullets first
    session_info, clean_memory_bullets = _extract_session_info(memory_bullets or [])

    # Build memory and summary blocks conditionally using clean bullets
    if progressive_mode:
        # Only build memory message if we have clean bullets and use conditional header
        mem_msg = _build_memory_message(clean_memory_bullets, f"{inject_header}\nMemory Context:", inject_role, progressive_mode) if clean_memory_bullets else None
        # Only build summary message if we have summary text
        sum_msg = _build_summary_message(summary_text, inject_role) if summary_text and summary_text.strip() else None
    else:
        # Legacy behavior: always build even if empty (using clean bullets)
        mem_msg = _build_memory_message(clean_memory_bullets, f"{inject_header}\nMemory Context:", inject_role, progressive_mode)
        sum_msg = _build_summary_message(summary_text, inject_role)

    packed: List[Dict[str, Any]] = []

    # Build unified system message with truly progressive context sections
    if before:
        # Start with original system content and inject session info at the top
        base_content = before[0]["content"]

        # Remove old memory context sections from base_content to prevent duplication
        base_content = _clean_memory_context_from_content(base_content, inject_header)

        # Inject session info right after Agent/User ID but before persona
        if session_info:
            # Parse session context and format it compactly at the top
            lines = base_content.split('\n')
            header_lines = []
            persona_lines = []

            # Find where persona starts (after time/date line)
            persona_started = False
            for line in lines:
                if not persona_started and ('Agent ID:' in line or 'User ID:' in line or 'It is' in line):
                    header_lines.append(line)
                else:
                    persona_started = True
                    persona_lines.append(line)

            # Format session info compactly
            session_lines = session_info.replace('Session Context:\n', '').replace('Session Context:', '').strip()
            session_compact = session_lines.replace('- ', '').replace('\n', ', ').strip()

            # Reconstruct with session info at the top
            unified_content = '\n'.join(header_lines)
            if session_compact:
                unified_content += f"\nSession: {session_compact}"
            unified_content += '\n' + '\n'.join(persona_lines)
        else:
            unified_content = base_content

        # Only add memory context if clean bullets exist (progressive)
        if mem_msg and clean_memory_bullets:
            mem_tokens = _count_tokens_from_messages([mem_msg])
            if mem_tokens > target_memory:
                # Trim bullets aggressively to fit cap
                kept: List[str] = []
                for b in clean_memory_bullets:
                    tmp_bullets = kept + [b]
                    tmp_msg = _build_memory_message(tmp_bullets, f"{inject_header}\nMemory Context:", inject_role, progressive_mode)
                    if _count_tokens_from_messages([tmp_msg]) > target_memory:
                        break
                    kept.append(b)
                if kept != clean_memory_bullets:
                    logger.warning(f"Memory capped: {len(kept)}/{len(clean_memory_bullets)} bullets to fit {target_memory} tokens")
                    mem_msg = _build_memory_message(kept, f"{inject_header}\nMemory Context:", inject_role, progressive_mode)

            # Append memory content to unified message
            unified_content += "\n\n" + mem_msg["content"]

        # Only add summary if conversation was actually truncated (check if dialogue < full conversation)
        dialogue_count = len([m for m in dialogue if m.get('role') in ['user', 'assistant']])
        original_count = len([m for m in messages if m.get('role') in ['user', 'assistant']])
        conversation_truncated = dialogue_count < original_count

        if sum_msg and summary_text and summary_text.strip() and conversation_truncated:
            if _count_tokens_from_messages([sum_msg]) <= target_summary:
                unified_content += "\n\n" + sum_msg["content"]

        # Add system hint if available
        if system_hint and system_hint.strip():
            unified_content += f"\n\nReasoning Guidance:\n{system_hint.strip()}"

        # Create single unified system message
        unified_msg = {"role": "system", "content": unified_content}
        packed.append(unified_msg)

    stats = {
        "tokens_total": 0,
        "tokens_system": _count_tokens_from_messages(packed),
        "tokens_memory": _count_tokens_from_messages([mem_msg]) if mem_msg else 0,
        "tokens_summary": _count_tokens_from_messages([sum_msg]) if sum_msg else 0,
        "tokens_dialogue": 0,
        "bullets_injected": len(clean_memory_bullets or []),
        "memory_capped": False,
    }

    # 4) Dialogue tail within remainder
    rem = target_dialogue - (stats["tokens_system"] + stats["tokens_memory"] + stats.get("tokens_summary", 0) - allocations.system)
    rem = max(rem, int(allocations.total * 0.5)) if not mem_msg and not sum_msg else rem
    # Always keep the last user message; back-fill previous messages from the end
    tail: List[Dict[str, Any]] = []
    for m in reversed(dialogue):
        candidate = [m] + tail
        if _count_tokens_from_messages(candidate) > rem:
            break
        tail = candidate
    stats["tokens_dialogue"] = _count_tokens_from_messages(tail)
    packed.extend(tail)

    stats["tokens_total"] = _count_tokens_from_messages(packed)
    return packed, stats