#!/usr/bin/env python3
import sys
sys.path.insert(0, "/Users/peppi/Dev/localcat/server")

from core.memory.nlp_manager import SharedNLPManager

manager = SharedNLPManager()
nlp = manager.get_model("en")

text = "I enjoyed the Italian restaurant last night"
doc = nlp(text)

print("="*70)
print("TEMPORAL ANALYSIS: 'last night'")
print("="*70)

print(f"\nFull parse:")
for token in doc:
    print(f"  {token.text:15} pos={token.pos_:6} dep={token.dep_:12} head={token.head.text:12} like_num={token.like_num}")

print(f"\nTemporal candidates:")
for tok in doc:
    # Check if it matches the temporal pattern (obl/nmod attached to VERB)
    if tok.dep_ in {"obl", "nmod", "npadvmod"} and tok.head and tok.head.pos_ in {"VERB", "AUX"}:
        print(f"  ✓ '{tok.text}' - dep={tok.dep_}, head='{tok.head.text}' ({tok.head.pos_})")
        # Check for nummod children
        for ch in tok.children:
            if ch.dep_ == "nummod":
                print(f"    - has nummod child: '{ch.text}'")

print(f"\nWhy 'last night' is NOT extracted:")
print(f"  - 'night' has dep=npadvmod (not obl/nmod)")
print(f"  - npadvmod is not in the temporal pattern check (line 1282)")
print(f"  - Temporal extraction only looks for obl/nmod with nummod children")
print(f"  - 'last' is an adjective modifier (amod), not nummod")
