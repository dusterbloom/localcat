import os
from typing import List, Dict, Any, Optional, Tuple
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


def pack_context(
    messages: List[Dict[str, Any]],
    memory_bullets: List[str],
    summary_text: Optional[str],
    budget_tokens: int,
    inject_role: str = "system",
    inject_header: str = "Use the following factual context if helpful.",
    system_hint: Optional[str] = None,
    progressive_mode: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Pack context with strict token budget and clear section order.

    Order:
      1) System instruction (keep first system message)
      2) Memory Context (bullets) - only if memory_bullets is not empty and progressive_mode is True
      3) Summary Context (latest snippet) - only if summary_text exists and progressive_mode is True
      4) Conversation tail (last N messages within remainder)

    Args:
        progressive_mode: If True, only inject memory/summary headers when content exists

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
    target_memory = allocations.memory
    target_summary = allocations.summary
    target_dialogue = allocations.dialogue

    # Build memory and summary blocks conditionally
    if progressive_mode:
        # Only build memory message if we have bullets and use conditional header
        mem_msg = _build_memory_message(memory_bullets, f"{inject_header}\nMemory Context:", inject_role, progressive_mode) if memory_bullets else None
        # Only build summary message if we have summary text
        sum_msg = _build_summary_message(summary_text, inject_role) if summary_text and summary_text.strip() else None
    else:
        # Legacy behavior: always build even if empty
        mem_msg = _build_memory_message(memory_bullets, f"{inject_header}\nMemory Context:", inject_role, progressive_mode)
        sum_msg = _build_summary_message(summary_text, inject_role)

    packed: List[Dict[str, Any]] = []
    # 1) System (as-is) + optional reasoning hint
    packed.extend(before)
    if system_hint and system_hint.strip():
        hint_msg = {"role": "system", "content": f"Reasoning Guidance:\n{system_hint.strip()}"}
        packed.append(hint_msg)

    stats = {
        "tokens_total": 0,
        "tokens_system": _count_tokens_from_messages(packed),
        "tokens_memory": 0,
        "tokens_summary": 0,
        "tokens_dialogue": 0,
        "bullets_injected": len(memory_bullets or []),
    }

    # 2) Memory within target slice
    if mem_msg:
        # Trim bullets to fit memory budget if needed
        if memory_bullets:
            # Recompute with incremental fitting
            kept: List[str] = []
            for b in memory_bullets:
                tmp = _build_memory_message(kept + [b], f"{inject_header}\nMemory Context:", inject_role, progressive_mode)
                if _count_tokens_from_messages([tmp]) > target_memory:
                    break
                kept.append(b)
            mem_msg = _build_memory_message(kept, f"{inject_header}\nMemory Context:", inject_role, progressive_mode)
        stats["tokens_memory"] = _count_tokens_from_messages([mem_msg])
        packed.append(mem_msg)  # type: ignore[arg-type]

    # 3) Summary within target
    if sum_msg:
        if _count_tokens_from_messages([sum_msg]) <= target_summary:
            packed.append(sum_msg)
            stats["tokens_summary"] = _count_tokens_from_messages([sum_msg])

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
