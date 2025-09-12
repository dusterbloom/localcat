#!/usr/bin/env python3
"""
LLM-only A/B tester for Localcat on macOS.

Compares two variants (A/B) by sending identical prompts to a local OpenAI-compatible API
and measuring latency (TTFB, total), output length, and simple keyword hits.

Usage examples:
  uv run server/scripts/ab_llm_ab.py --runs 10 \
    --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" \
    --variant-a base --variant-b free

  # Custom system prompts from files
  uv run server/scripts/ab_llm_ab.py --runs 10 \
    --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" \
    --variant-a-file path/to/sys_a.txt --variant-b-file path/to/sys_b.txt
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

try:
    # openai>=1.0
    from openai import OpenAI
except Exception:
    OpenAI = None  # Fallback handled later


def load_system_prompt(variant: str, file_path: str | None) -> str:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    # Load Localcat defaults from server/core/bot.py
    # Ensure server/ is on path
    here = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, here)
    try:
        from core.bot import SYSTEM_INSTRUCTION_BASE, SYSTEM_INSTRUCTION_BASE_FREE
        return SYSTEM_INSTRUCTION_BASE if variant == "base" else SYSTEM_INSTRUCTION_BASE_FREE
    except Exception:
        # Safe fallback if import fails
        if variant == "base":
            return "You are a helpful assistant. Keep responses concise."
        else:
            return (
                "You are a helpful and thoughtful assistant.\n"
                "- Prefer concise answers; expand on request.\n"
                "- Personalize only when clearly useful."
            )


def keyword_hit_rate(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = text.lower()
    hits = sum(1 for k in keywords if k.lower() in t)
    return hits / len(keywords)


def run_once(client: OpenAI, system_prompt: str, user_text: str, stream: bool = True) -> Tuple[float, float, str]:
    start = time.perf_counter()
    ttfb = None
    content_parts: List[str] = []

    try:
        if stream:
            stream_resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                stream=True,
            )
            for chunk in stream_resp:
                if ttfb is None:
                    ttfb = (time.perf_counter() - start) * 1000.0
                if chunk and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)
            total = (time.perf_counter() - start) * 1000.0
            return (ttfb or total), total, "".join(content_parts)
        else:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            total = (time.perf_counter() - start) * 1000.0
            text = resp.choices[0].message.content if resp and resp.choices else ""
            return total, total, text
    except Exception as e:
        return 0.0, 0.0, f"__ERROR__: {e}"


def main():
    parser = argparse.ArgumentParser(description="LLM-only A/B tester for Localcat")
    parser.add_argument("--base-url", required=False, default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=False, default=os.getenv("OPENAI_MODEL"), help="Model name")
    parser.add_argument("--api-key", required=False, default=os.getenv("OPENAI_API_KEY", "lm-studio"), help="API key (dummy ok for local)")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per variant")
    parser.add_argument("--variant-a", choices=["base", "free"], default="base")
    parser.add_argument("--variant-b", choices=["base", "free"], default="free")
    parser.add_argument("--variant-a-file", default=None, help="Path to system prompt for A (overrides variant-a)")
    parser.add_argument("--variant-b-file", default=None, help="Path to system prompt for B (overrides variant-b)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming; measure total only")
    parser.add_argument("--keywords", nargs="*", default=[], help="Keywords to check in responses for hit-rate")
    parser.add_argument("--out", default=None, help="CSV output path (default: server/data/ab_results_llm_<ts>.csv)")
    args = parser.parse_args()

    if OpenAI is None:
        print("openai package not available; install per server/requirements.txt")
        sys.exit(1)

    if not args.base_url or not args.model:
        print("Please set --base-url and --model or OPENAI_BASE_URL/OPENAI_MODEL envs.")
        sys.exit(1)

    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_MODEL"] = args.model
    os.environ.setdefault("OPENAI_API_KEY", args.api_key)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url)

    sys_a = load_system_prompt(args.variant_a, args.variant_a_file)
    sys_b = load_system_prompt(args.variant_b, args.variant_b_file)

    # Simple rotating prompts; edit as needed
    prompts = [
        "Summarize this in one sentence: I need to plan a trip to Kyoto in spring.",
        "What are two pros and cons of working from home?",
        "Act as a helpful assistant. Give me three actionable productivity tips.",
        "In one sentence, explain what a vector database does.",
    ]

    rows: List[Dict[str, str]] = []

    def run_block(label: str, sys_prompt: str):
        print(f"\n=== {label} ===")
        ttfb_list: List[float] = []
        total_list: List[float] = []
        hit_list: List[float] = []
        for i in range(args.runs):
            p = prompts[i % len(prompts)]
            ttfb, total, text = run_once(client, sys_prompt, p, stream=not args.no_stream)
            hits = keyword_hit_rate(text, args.keywords)
            ttfb_list.append(ttfb)
            total_list.append(total)
            hit_list.append(hits)
            print(f"Run {i+1}/{args.runs} | ttfb={ttfb:.1f}ms total={total:.1f}ms hits={hits:.2f} len={len(text)}")
            rows.append({
                "variant": label,
                "run": str(i+1),
                "prompt": p,
                "ttfb_ms": f"{ttfb:.1f}",
                "total_ms": f"{total:.1f}",
                "hit_rate": f"{hits:.3f}",
                "resp_len": str(len(text)),
            })
        def avg(x):
            return sum(x)/len(x) if x else 0.0
        print(f"Avg ttfb={avg(ttfb_list):.1f}ms | Avg total={avg(total_list):.1f}ms | Avg hits={avg(hit_list):.2f}")

    run_block("A", sys_a)
    run_block("B", sys_b)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"ab_results_llm_{ts}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()

