#!/usr/bin/env python3
"""
Real Conversation E2E with True Summaries
========================================

Runs a realistic, mixed conversation, extracts facts with HotMem, calls a real LLM
summarizer (LM Studio or OpenAI-compatible server) to produce periodic summaries,
stores them into FTS, and then evaluates retrieval across KG + summary layers.

Requirements:
- LM Studio (or other OpenAI-compatible server) running
- SUMMARIZER_BASE_URL, SUMMARIZER_MODEL set (see server/.env.example)
"""
import os
import sys
import time
import tempfile
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_store import MemoryStore, Paths
from memory_hotpath import HotMemory
from summarizer import _build_summary_prompt, _http_chat_completion


CONVO: List[Dict[str, str]] = [
    # A mix of short/long, simple/complex, varied topics
    {"role": "user", "content": "Hi, I'm Sarah. I live in Seattle and work at Microsoft in the Azure team."},
    {"role": "assistant", "content": "Hi Sarah! Seattle is beautiful. How's life at Microsoft?"},

    {"role": "user", "content": "Pretty exciting! I'm leading a project on NLP using Python and TensorFlow, with BERT fine-tuning for sentiment analysis."},
    {"role": "assistant", "content": "That sounds great. Fine-tuning often boosts accuracy on domain data."},

    {"role": "user", "content": "By the way, my brother Tom lives in Portland and teaches philosophy at Reed College."},
    {"role": "assistant", "content": "Nice! Portland has a vibrant academic scene."},

    {"role": "user", "content": "I drive a Tesla Model 3; it makes the Seattle commute much nicer."},
    {"role": "assistant", "content": "Great choice for a tech city commute."},

    {"role": "user", "content": "My dog Luna is a golden retriever who loves hiking; we visit Mount Rainier often."},
    {"role": "assistant", "content": "Golden retrievers make perfect hiking companions. Mount Rainier is stunning."},

    {"role": "user", "content": "Luna is 3 years old and knows over 20 commands."},
    {"role": "assistant", "content": "Impressive training!"},

    {"role": "user", "content": "My favorite color is purple and my lucky number is 7."},
    {"role": "assistant", "content": "Purple is a classic, and 7 is a popular lucky number."},

    {"role": "user", "content": "Yesterday I met a colleague who transitioned from research to product; she said it was challenging yet rewarding."},
    {"role": "assistant", "content": "That transition can be tough indeed but brings a broader impact."},
]


def generate_and_store_summary(messages: List[Dict[str, str]], store: MemoryStore, session_id: str, turn_id: int) -> bool:
    base_url = os.getenv("SUMMARIZER_BASE_URL") or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.getenv("SUMMARIZER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    model = os.getenv("SUMMARIZER_MODEL", os.getenv("OPENAI_MODEL", "qwen3:4b"))
    max_tokens = int(os.getenv("SUMMARIZER_MAX_TOKENS", "160"))
    # Build prompt
    prompt = _build_summary_prompt(messages, max_messages=16, include_user=True, include_assistant=True)
    # Call LLM
    summary = _http_chat_completion(base_url, api_key, model, prompt, max_tokens, session_id=session_id)
    if not summary:
        print("❌ Summarizer did not return a summary. Ensure LM Studio/OpenAI-compatible server is running.")
        return False
    # Store summary into FTS
    ts = int(time.time() * 1000)
    eid = f"summary:{session_id}"
    store.enqueue_mention(eid=eid, text=summary, ts=ts, sid=session_id, tid=turn_id)
    store.flush()
    print("📝 Stored summary:")
    print(summary[:320].replace('\n', ' ') + ("..." if len(summary) > 320 else ""))
    return True


def search_summaries(store: MemoryStore, query: str, limit: int = 3) -> List[str]:
    res = store.search_fts(query, limit=limit)
    return [text for (text, eid) in res if eid.startswith('summary:')]


def build_leann_index_for_summaries(store: MemoryStore, session_id: str, index_path: str) -> int:
    try:
        from leann.api import LeannBuilder  # type: ignore
    except Exception:
        return 0
    cur = store.sql.cursor()
    rows = cur.execute(
        "SELECT text FROM mention WHERE eid=? ORDER BY ts ASC",
        (f"summary:{session_id}",)
    ).fetchall()
    if not rows:
        return 0
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    builder = LeannBuilder(backend_name=os.getenv('LEANN_BACKEND','hnsw'))
    count = 0
    for (text,) in rows:
        if text:
            builder.add_text(str(text), {'type': 'summary', 'sid': session_id})
            count += 1
    builder.build_index(index_path)
    return count


def semantic_search_summaries(index_path: str, query: str, k: int = 3) -> List[str]:
    try:
        from leann.api import LeannSearcher  # type: ignore
    except Exception:
        return []
    if not os.path.exists(index_path):
        return []
    s = LeannSearcher(index_path)
    results = s.search(query, top_k=k, complexity=int(os.getenv('HOTMEM_LEANN_COMPLEXITY','16')))
    out = []
    for r in results:
        t = getattr(r, 'text', None)
        if t:
            out.append(str(t))
    return out


def run():
    with tempfile.TemporaryDirectory() as tdir:
        store = MemoryStore(Paths(
            sqlite_path=os.path.join(tdir, 'memory.db'),
            lmdb_dir=os.path.join(tdir, 'graph.lmdb')
        ))
        hot = HotMemory(store)
        sid = "real-e2e"

        messages: List[Dict[str, str]] = []
        print("\n===== Real Conversation E2E with True Summaries =====")

        for i, m in enumerate(CONVO, start=1):
            messages.append(m)
            # Extract/store via HotMem on user turns to simulate fact ingestion
            if m.get('role') == 'user':
                bullets, triples = hot.process_turn(m['content'], session_id=sid, turn_id=i)
                if triples:
                    print(f"🔎 Turn {i} extracted {len(triples)} facts; bullets={len(bullets)}")
        
            # Periodically generate a true LLM summary and store it
            if i % 6 == 0:
                print(f"\n📋 Generating summary after turn {i}")
                generate_and_store_summary(messages, store, sid, i)

        # Final summary at the end
        print("\n📋 Generating final summary")
        generate_and_store_summary(messages, store, sid, len(CONVO))

        # Optionally build semantic index over summaries
        use_sem = os.getenv('HOTMEM_USE_LEANN_SUMMARIES','true').lower() in ('1','true','yes')
        idx_path = os.path.join(tdir, 'summary_vectors.leann')
        if use_sem:
            n = build_leann_index_for_summaries(store, sid, idx_path)
            if n:
                print(f"🔧 Built LEANN summary index with {n} summaries")

        # Retrieval across KG + Summaries (lexical + optional semantic)
        queries = [
            "who works at microsoft",
            "what car does sarah drive",
            "how old is luna",
            "where does tom teach",
            "summarize the session highlights",
        ]
        for q in queries:
            ents = [w.lower() for w in q.split() if len(w) > 2 and w.isalpha()][:6]
            kg = hot._retrieve_context(q, ents, turn_id=999, intent=None)
            fts = search_summaries(store, q, limit=2)
            sem = semantic_search_summaries(idx_path, q, k=2) if use_sem else []
            print(f"\n🔎 Query: {q}")
            if kg:
                print("  KG Bullets:")
                for b in kg[:3]:
                    print("   •", b)
            if fts:
                print("  Summary Hits:")
                for s in fts:
                    print("   ⤷", s[:200].replace('\n',' ') + ("..." if len(s) > 200 else ""))
            if sem:
                print("  Semantic Summary Hits:")
                for s in sem:
                    print("   ⤷", s[:200].replace('\n',' ') + ("..." if len(s) > 200 else ""))
            if not kg and not fts:
                print("  (no results — ensure facts and summaries were stored)")


if __name__ == '__main__':
    run()
