#!/usr/bin/env python3
import sys
sys.path.insert(0, "/Users/peppi/Dev/localcat/server")

from core.memory.nlp_manager import SharedNLPManager

manager = SharedNLPManager()
nlp = manager.get_model("en")

text = "I enjoyed the Italian restaurant last night"
doc = nlp(text)

print("="*70)
print("DEPENDENCY PARSE ANALYSIS")
print("="*70)
print(f"\nText: '{text}'")
print(f"\nTokens with dependencies:")
for token in doc:
    print(f"  {token.text:20} pos={token.pos_:6} dep={token.dep_:12} head={token.head.text}")

print(f"\nNoun chunks:")
for chunk in doc.noun_chunks:
    print(f"  '{chunk.text}' (root: {chunk.root.text}, dep: {chunk.root.dep_})")
