import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


def _filter_by_roles(messages: List[Dict[str, Any]], include_user: bool, include_assistant: bool) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        role = m.get("role", "user")
        if role == "user" and not include_user:
            continue
        if role == "assistant" and not include_assistant:
            continue
        out.append(m)
    return out


def _last_turn_pairs(messages: List[Dict[str, Any]], pairs: int) -> List[Dict[str, Any]]:
    """Return the last N (user, assistant) pairs (preserving original order)."""
    if pairs <= 0:
        return messages
    out: List[Dict[str, Any]] = []
    pair_count = 0
    i = len(messages) - 1
    # Walk backwards collecting assistant then user for a pair
    while i >= 0 and pair_count < pairs:
        # Find last assistant
        while i >= 0 and messages[i].get("role") != "assistant":
            i -= 1
        if i < 0:
            break
        a_idx = i
        i -= 1
        # Find preceding user
        while i >= 0 and messages[i].get("role") != "user":
            i -= 1
        if i < 0:
            break
        u_idx = i
        i -= 1
        # Prepend this pair (user, assistant) to the front
        out = messages[u_idx:a_idx+1] + out
        pair_count += 1
    return out if out else messages


def _build_summary_prompt(messages: List[Dict[str, Any]], max_messages: int = 16,
                          include_user: bool = True, include_assistant: bool = True) -> List[Dict[str, str]]:
    """Build a compact chat prompt for summarization from the latest messages."""
    # Take the last N messages (system/user/assistant). Strip long system lines.
    trimmed: List[Dict[str, str]] = []
    for m in messages[-max_messages:]:
        role = m.get("role", "user")
        content = str(m.get("content", "") or "").strip()
        if role == "system":
            # Keep only the first 200 chars of system content to reduce noise
            content = content[:200]
        trimmed.append({"role": role, "content": content})

    system = {
        "role": "system",
        "content": (
            "You are a real-time session summarizer. Every 30 seconds, produce a concise update that includes: "
            "(1) 3-5 factual bullet points from the last segment of conversation, "
            "(2) one brief narrative sentence describing what's happening, and "
            "(3) up to 2 follow-up items or open loops. "
            "Never invent facts; only use the provided content. Keep under 120 words. "
            "Include key entities and facts verbatim (names, places, numbers) to aid downstream retrieval."
        ),
    }

    return [system] + trimmed


def _http_chat_completion(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], max_tokens: int = 160, session_id: str = None) -> Optional[str]:
    """Minimal OpenAI-compatible chat.completions call using stdlib."""
    import urllib.request
    import urllib.error

    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    
    # Add session isolation for LM Studio - prevents context pollution between sessions
    if session_id:
        payload["session_id"] = f"hotmem_{session_id}"
    # Reasoning controls for Ollama/OpenAI-compatible servers (best-effort)
    # Prefer a single "think" parameter supported by Ollama (true|false|low|medium|high)
    think_raw = (os.getenv("SUMMARIZER_THINK") or os.getenv("SUMMARIZER_REASONING_EFFORT") or "").strip().lower()
    if think_raw in ("low", "medium", "high"):
        payload["think"] = think_raw
    elif think_raw in ("1", "true", "yes"):
        payload["think"] = True
    elif think_raw in ("0", "false", "no"):
        payload["think"] = False
    # Optional token cap for reasoning implementations that support it
    rtoks = os.getenv("SUMMARIZER_REASONING_TOKENS")
    if rtoks and rtoks.isdigit():
        payload.setdefault("reasoning", {})["tokens"] = int(rtoks)
    # Ollama options passthrough (optional)
    num_ctx = os.getenv("SUMMARIZER_NUM_CTX")
    if num_ctx and num_ctx.isdigit():
        payload.setdefault("options", {})["num_ctx"] = int(num_ctx)
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
            choice = (obj.get("choices") or [{}])[0]
            msg = (choice.get("message") or {}).get("content")
            return msg
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        logger.warning(f"[Summarizer] HTTP error: {e.code} {err}")
    except Exception as e:
        logger.warning(f"[Summarizer] Request failed: {e}")
    return None


async def periodic_summarizer(context_aggregator, memory_processor, interval_sec: int = 30) -> None:
    """Run periodic LLM summaries every interval_sec.

    - context_aggregator: the LLM context aggregator created in bot.py
    - memory_processor: HotPathMemoryProcessor instance (for store + session info)
    """
    # Prefer dedicated summarizer endpoint if provided (e.g., LM Studio at http://127.0.0.1:1234/v1)
    model = os.getenv("SUMMARIZER_MODEL", os.getenv("OPENAI_MODEL", "qwen3:4b"))
    base_url = os.getenv("SUMMARIZER_BASE_URL") or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.getenv("SUMMARIZER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    max_tokens = int(os.getenv("SUMMARIZER_MAX_TOKENS", "160"))
    # Windowing controls
    window_mode = os.getenv("SUMMARIZER_WINDOW_MODE", "delta").lower()  # delta|turn_pairs|tail
    turn_pairs = int(os.getenv("SUMMARIZER_TURN_PAIRS", "2"))
    max_messages = int(os.getenv("SUMMARIZER_MAX_MESSAGES", "16"))
    include_user = os.getenv("SUMMARIZER_INCLUDE_USER", "true").lower() in ("1", "true", "yes")
    include_assistant = os.getenv("SUMMARIZER_INCLUDE_ASSISTANT", "true").lower() in ("1", "true", "yes")

    # Track last summarized message count to avoid repeating identical segments
    last_len = 0
    logger.info(f"[Summarizer] Enabled (model={model}, base={base_url}, interval={interval_sec}s)")

    try:
        while True:
            await asyncio.sleep(interval_sec)
            try:
                context = context_aggregator.user().context
                messages = list(getattr(context, "messages", []))
            except Exception as e:
                logger.warning(f"[Summarizer] Could not read context: {e}")
                continue

            if not messages or len(messages) == last_len:
                continue  # nothing new
            # Choose window
            msgs = messages
            if window_mode == "delta":
                # Summarize only the new messages since the last run
                msgs = messages[last_len:]
                if not msgs:
                    continue
            elif window_mode == "turn_pairs":
                msgs = _last_turn_pairs(messages, turn_pairs)
            else:  # tail
                pass

            msgs = _filter_by_roles(msgs, include_user=include_user, include_assistant=include_assistant)
            if not msgs:
                continue

            prompt = _build_summary_prompt(msgs, max_messages=max_messages,
                                           include_user=include_user, include_assistant=include_assistant)
            # Get session ID for isolation
            session_id = getattr(memory_processor, '_session_id', 'default')
            summary = await asyncio.get_event_loop().run_in_executor(
                None, _http_chat_completion, base_url, api_key, model, prompt, max_tokens, session_id
            )
            if summary:
                try:
                    ts = int(time.time() * 1000)
                    eid = f"summary:{getattr(memory_processor, '_session_id', 'default')}"
                    memory_processor.store.enqueue_mention(eid=eid, text=summary, ts=ts,
                                                           sid=getattr(memory_processor, '_session_id', 'default'),
                                                           tid=getattr(memory_processor, '_turn_id', 0))
                    memory_processor.store.flush()
                    logger.info("[Summarizer] Stored periodic session summary to FTS")
                    # Optional snippet logging for visibility (PII-aware: disabled by default)
                    if os.getenv("SUMMARIZER_LOG_SUMMARIES", "false").lower() in ("1", "true", "yes"):
                        try:
                            n = int(os.getenv("SUMMARIZER_LOG_SUMMARY_CHARS", "160"))
                        except Exception:
                            n = 160
                        snippet = summary.replace("\n", " ")[:n]
                        logger.info(f"[Summarizer] Summary snippet: {snippet}")
                except Exception as e:
                    logger.warning(f"[Summarizer] Failed to store summary: {e}")
                last_len = len(messages)
            
            # Also persist the most recent assistant reply verbatim into the session store
            try:
                # Find the last assistant message in the new window
                last_assistant_msg = None
                for m in reversed(msgs):
                    if m.get('role') == 'assistant':
                        last_assistant_msg = str(m.get('content', '') or '')
                        break
                if last_assistant_msg:
                    # Store via memory processor helper (links to current session/turn)
                    memory_processor.store_assistant_response(last_assistant_msg)
            except Exception as e:
                logger.debug(f"[Summarizer] Skipped storing assistant verbatim: {e}")
    except asyncio.CancelledError:
        logger.info("[Summarizer] Cancelled")
    except Exception as e:
        logger.warning(f"[Summarizer] Stopped with error: {e}")


def start_periodic_summarizer(context_aggregator, memory_processor, interval_sec: int = 30):
    """Helper to create the asyncio task for periodic summaries."""
    return asyncio.create_task(periodic_summarizer(context_aggregator, memory_processor, interval_sec))
