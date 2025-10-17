#!/usr/bin/env python3
"""
Simple test of DSPy edge extraction
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["DSPY_MODEL"] = "openai/llama-3.2-3b-instruct"
os.environ["DSPY_BASE_URL"] = "http://127.0.0.1:1234/v1"
os.environ["OPENAI_API_KEY"] = "dummy"

from core.memory.dspy_extractor import DSPyEdgeExtractor

# Test with llama-3.2-3b-instruct via LM Studio (3x larger model)
extractor = DSPyEdgeExtractor(
    model="openai/llama-3.2-3b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy"
)

text = "I'm Alice, a software engineer at Google who loves Python"
existing_edges = [
    ("you", "is", "alice"),
    ("alice", "also_known_as", "software engineer")
]

print(f"Text: {text}")
print(f"\nExisting edges:")
for e in existing_edges:
    print(f"  {e}")

print(f"\nCalling DSPy...")
try:
    result = extractor.extract(text=text, existing_edges="\n".join([f"({s}, {r}, {d})" for s, r, d in existing_edges]))
    print(f"\nDSPy raw result:")
    print(f"  missing_edges: {result.missing_edges}")
    print(f"  rationale: {getattr(result, 'rationale', 'N/A')}")

    missing_edges = extractor._parse_edges(result.missing_edges)
    print(f"\nParsed edges: {len(missing_edges)}")
    for e in missing_edges:
        print(f"  {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()