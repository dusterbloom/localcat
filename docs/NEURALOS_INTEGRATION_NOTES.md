# NeuralOS Integration Notes (LocalCat)

This document mirrors the high-level plan in
`memory-production/docs/NEURALOS_LOCALCAT_INTEGRATION_PLAN.md` and
summarises what LocalCat needs to do to treat NeuralOS as its primary
memory OS.

The intent is to keep one mental model for memory (NeuralOS) while
reusing LocalCat's strengths (SLM stack, LMDB/SQLite storage,
voice/TTS pipeline).

---

## 1. Where NeuralOS fits

- **Reasoning layer**: LocalCat continues to own the main SLM
  (Gemma/Qwen3/OpenAI/etc.) and the tool router.
- **Memory OS**: NeuralOS becomes the single high-level memory system
  for:
  - Personal identity and preferences (signed likes/dislikes).
  - Meta-memory (“Did I tell you X?”, “Did we ever talk about Y?”).
  - Entity summaries (“What do you know about my dog?”).
  - General facts/episodes/documents (topics: music, food, books,
    location, pets, etc.).
- **Storage backend**: LocalCat's LMDB/SQLite `MemoryStore` should be
  treated as a persistence/index backend for NeuralOS, not a separate
  memory universe.

---

## 2. Immediate integration steps

These are the concrete hooks now available in NeuralOS:

1. **Inject the main SLM (Brain)**
   - NeuralOS now accepts an injected `Brain` instance:
     ```python
     from neuralos.core import NeuralOS
     from neuralos.brain import Brain

     class LocalCatBrain(Brain):
         def __init__(self, llm_service):
             self.llm = llm_service

         def generate(self, prompt: str, *, logits_processors=None, max_new_tokens: int = 128) -> str:
             # Map to LocalCat's existing generation API.
             return self.llm.generate(prompt, max_tokens=max_new_tokens)

     neuralos = NeuralOS(
         brain=LocalCatBrain(llm_service),
         use_controller=False,
         physics_mode="off",
         use_cognitive_core=False,
         use_doc_rag=False,
     )
     ```
   - If `brain` is not provided, NeuralOS still defaults to its
     internal `QwenBrain`, so existing workflows keep working.

2. **Inject a custom Vault backend**
   - NeuralOS now accepts an injected `vault`:
     ```python
     from neuralos.vault import Vault  # type reference

     class LocalCatVault(Vault):
         # TODO: adapt LocalCat MemoryStore here.
         ...

     neuralos = NeuralOS(
         brain=LocalCatBrain(llm_service),
         vault=LocalCatVault(...),
     )
     ```
   - When `vault` is omitted, NeuralOS preserves the existing
     `Vault` / `FalkorDBVault` selection based on config.

3. **(Optional) Inject a specialised FactExtractor**
   - NeuralOS also accepts a `fact_extractor` argument:
     - This is where LocalCat can plug in a tiny SLM (e.g.
       Qwen2.5‑0.5B) + spaCy for robust, cheap fact extraction.
   - If not provided, NeuralOS constructs a default `FactExtractor`
     bound to the main `Brain`.

---

## 3. Tool wiring in LocalCat

The LocalCat SLM should see NeuralOS via a small tool set:

- `neuralos_remember(text, user_id)`
  - Route to NeuralOS TEACH path.
- `neuralos_query(query, user_id)`
  - Route to NeuralOS ASK path (JIT engine + meta + prefs).
- `neuralos_introspect(query, user_id)`
  - Route to NeuralOS meta-memory path.
- Optionally `neuralos_forget(id_or_text, user_id)`
  - Drive correction/superseding behaviour.

These tools should gradually replace the existing HotMem-style tools
for new memory queries, with LMDB remaining the persistence layer
under a future `LocalCatVault`.

---

## 4. Next review checkpoints

When work resumes on this integration, suggested checkpoints:

1. Implement `LocalCatBrain` and wire a single `NeuralOS` instance
   into the server startup.
2. Add a minimal `LocalCatVault` prototype that mirrors a subset of
   `MemoryItem` fields into `MemoryStore`.
3. Register NeuralOS tools in the LocalCat tool router and A/B them
   against existing memory responses for:
   - Identity (“What is my name?”, “Where do I live?”).
   - Preferences (“What is my favorite music/food?”, dislikes).
   - Meta-memory (“Did we ever talk about X?”, “Did I tell you Y?”).
4. Iterate on performance (latency) and correctness before fully
   deprecating parallel memory stacks.

