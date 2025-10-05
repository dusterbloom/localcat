Prosody Pass-Through Integration (HotMem)

Goal: Attach per-turn prosody (certainty) to edges and confidence scoring.

API hooks
- HotMemService.set_prosody_for_turn(prosody)
  - prosody: ProsodyFeatures or dict with `certainty_modifier`
- HotMemory.set_prosody(prosody_features)

Usage
1) In your audio pipeline, after extracting ProsodyFeatures for the user’s utterance and before calling HotMem to process text:

   service.set_prosody_for_turn(prosody_features)
   service._store_messages([{ "role": "user", "content": user_text }])

2) Confidence scoring (ProsodyAwareConfidence) uses Context.prosody_features.
3) Edge meta persists `prosody_certainty` (with surface/morph/polarity/lang).

Telemetry (optional)
- MEMORY_STORE_META_TELEMETRY=true → logs `[EdgeMeta] (s,r,d) -> {...}`
- MEMORY_TENSE_AWARE=true|false (default true)
- MEMORY_POLARITY_AWARE=true|false (default true)
- MEMORY_RETRIEVAL_TELEMETRY=true → logs tense/polarity boosts
- MEMORY_DEBUG_BULLETS_META=true → appends compact meta to [graph] bullets

Notes
- Prosody provided via API takes precedence over any best-effort logfile scraping.
- LMDB adjacency remains unchanged; meta stored in SQLite `edge.meta` JSON.
