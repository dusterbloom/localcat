import os
from typing import List, Dict, Any, Optional, Tuple


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # Fast heuristic: ~4 chars per token
    return max(1, (len(text) + 3) // 4)


def _estimate_tokens_from_messages(msgs: List[Dict[str, str]]) -> int:
    total = 0
    for m in msgs:
        total += _estimate_tokens_from_text(str(m.get("content", "") or ""))
    return total


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


def _build_memory_message(bullets: List[str], header: str, role: str) -> Optional[Dict[str, str]]:
    if not bullets:
        return None
    body = "\n".join(bullets)
    return {"role": role, "content": f"{header}\n{body}"}


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
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Pack context with strict token budget and clear section order.

    Order:
      1) System instruction (keep first system message)
      2) Memory Context (bullets)
      3) Summary Context (latest snippet)
      4) Conversation tail (last N messages within remainder)

    Returns: (messages, stats)
    """
    # Hard safety
    msgs = list(messages or [])
    if not msgs:
        return [], {"tokens_total": 0}

    # Remove any prior injected memory/summary blocks
    prior_headers = [inject_header, "Summary Context (recent):", "Recap from recent conversation:"]
    msgs = _filter_old_injections(msgs, prior_headers)

    # Identify first system and split
    sys_idx = _first_system_index(msgs)
    system_msg = msgs[sys_idx]
    before = msgs[: sys_idx + 1]
    dialogue = msgs[sys_idx + 1 :]

    # Budget slices (tunable)
    B = max(512, int(budget_tokens or 4096))
    target_system = min(int(B * 0.12), 512)
    target_memory = min(int(B * 0.15), 600)
    target_summary = min(int(B * 0.10), 400)
    # tools slice reserved for future
    target_dialogue = B - (target_system + target_memory + target_summary)

    # Build memory and summary blocks
    mem_msg = _build_memory_message(memory_bullets, f"{inject_header}\nMemory Context:", inject_role)
    sum_msg = _build_summary_message(summary_text, inject_role)

    packed: List[Dict[str, Any]] = []
    # 1) System (as-is) + optional reasoning hint
    packed.extend(before)
    if system_hint and system_hint.strip():
        hint_msg = {"role": "system", "content": f"Reasoning Guidance:\n{system_hint.strip()}"}
        packed.append(hint_msg)

    stats = {
        "tokens_total": 0,
        "tokens_system": _estimate_tokens_from_messages(packed),
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
                tmp = _build_memory_message(kept + [b], f"{inject_header}\nMemory Context:", inject_role)
                if _estimate_tokens_from_messages([tmp]) > target_memory:
                    break
                kept.append(b)
            mem_msg = _build_memory_message(kept, f"{inject_header}\nMemory Context:", inject_role)
        stats["tokens_memory"] = _estimate_tokens_from_messages([mem_msg])
        packed.append(mem_msg)  # type: ignore[arg-type]

    # 3) Summary within target
    if sum_msg:
        if _estimate_tokens_from_messages([sum_msg]) <= target_summary:
            packed.append(sum_msg)
            stats["tokens_summary"] = _estimate_tokens_from_messages([sum_msg])

    # 4) Dialogue tail within remainder
    rem = B - (stats["tokens_system"] + stats["tokens_memory"] + stats.get("tokens_summary", 0))
    rem = max(rem, int(B * 0.5)) if not mem_msg and not sum_msg else rem
    # Always keep the last user message; back-fill previous messages from the end
    tail: List[Dict[str, Any]] = []
    for m in reversed(dialogue):
        candidate = [m] + tail
        if _estimate_tokens_from_messages(candidate) > rem:
            break
        tail = candidate
    stats["tokens_dialogue"] = _estimate_tokens_from_messages(tail)
    packed.extend(tail)

    stats["tokens_total"] = _estimate_tokens_from_messages(packed)
    return packed, stats
