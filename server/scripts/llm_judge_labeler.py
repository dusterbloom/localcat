#!/usr/bin/env python3
"""
LLM Judge Labeler (LM Studio / OpenAI-compatible)

Reads gray-zone records and asks an LLM to accept/reject triples per sentence.
Designed for LM Studio running an OpenAI-compatible server (default localhost:1234).

Usage (from server/):
  LLM_JUDGE_BASE_URL=http://127.0.0.1:1234/v1 \
  LLM_JUDGE_MODEL=llama-3.2-3b-instruct \
  .venv/bin/python -m scripts.llm_judge_labeler \
    --log data/judge_grayzone.jsonl \
    --out data/judge_labels.jsonl --batch_size 6

Output:
  JSONL with {text, triples: [[s,r,d],...], labels: [0/1,...], raw}
  You can then convert labels into supervised training examples for distillation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def prompt_for(text: str, triples: List[Tuple[str, str, str]]) -> str:
    # Compact JSON-style prompt to encourage structured output
    triples_json = json.dumps([[s, r, d] for s, r, d in triples], ensure_ascii=False)
    return (
        "You are a precise graph-triple judge. Given a sentence and candidate triples, output JSON with a 'labels' array of 0/1 indicating accept(1) or reject(0) for each triple in order.\n"
        "Rules: Accept only if the triple is explicitly supported by the sentence, with correct arguments and useful object content. Reject generic/empty objects or unsupported arguments.\n"
        f"Sentence: {text}\n"
        f"Triples: {triples_json}\n"
        "Return JSON only: {\"labels\": [..]}"
    )


def call_llm(base_url: str, api_key: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    # Use python-openai client (already in requirements via openai)
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    return resp.choices[0].message.content or "{}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=6)
    args = ap.parse_args()

    base_url = os.getenv("LLM_JUDGE_BASE_URL", "http://127.0.0.1:1234/v1").strip()
    api_key = os.getenv("LLM_JUDGE_API_KEY", "not-needed").strip()
    model = os.getenv("LLM_JUDGE_MODEL", "llama-3.2-3b-instruct").strip()

    rows = load_jsonl(Path(args.log))
    if not rows:
        raise SystemExit(f"No gray-zone records at {args.log}")

    # Group triples per sentence
    grouped: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for r in rows:
        text = r.get("text", "")
        tri = r.get("triple", ["", "", ""][0:3])
        if text and tri and len(tri) == 3:
            grouped[text].append(tuple(tri))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    for text, triples in grouped.items():
        # chunk triples to keep prompt small
        for i in range(0, len(triples), args.batch_size):
            batch = triples[i:i + args.batch_size]
            p = prompt_for(text, batch)
            try:
                content = call_llm(base_url, api_key, model, p)
                data = json.loads(content)
                labels = data.get("labels", [])
                if not isinstance(labels, list):
                    labels = []
            except Exception as e:
                data = {"error": str(e)}
                labels = []
            rec = {
                "text": text,
                "triples": batch,
                "labels": labels,
                "raw": data,
            }
            with outp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} labeled batches to {outp}")


if __name__ == "__main__":
    main()

