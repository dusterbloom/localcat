# Memory Prosody + Context Upgrade

Owner: Memory/Voice Team  
Status: Proposed  
Goal: Improve recall quality without hurting latency by (1) activating prosody signals, (2) adopting key OpenMemory lessons (sectors, decay, waypoint, explainability), and (3) restructuring context per Anthropic’s effective context engineering.

## Outcomes
- Prosody actively influences fact confidence, retrieval, and summaries.
- Sector-aware, decayed, and explainable recall with optional single-hop waypoint expansion.
- Anthropic-aligned context assembly: compact, typed memory headers and consistent token-aware sliding window for “infinite” chats.
- Optional OpenMemory integration as a cold/augment source (strict budget/timeout) without impacting hot-path latency.

## Scope
- Prosody: capture → store → use in edge confidence and retrieval scoring; bias summarization.
- Sectorization: rule-based classifier + intent mapping; sector prior folded into scoring.
- Salience/Decay/Reinforcement: factor in edge usage/weight, recency, and sector decay lambdas during scoring.
- Waypoint expansion: single strongest 1-hop expansion under strict budget.
- Explainability: enriched candidate metadata and component logging; compact headers in context.
- ContextBuilder: single assembly point with Anthropic ordering and token-aware pruning for voice and typed paths.
- Optional OpenMemory: add as retrieval source with strict timeout and tiny budget.

## Phased Plan (PRs)

### PR1: Prosody (Enable + Wire)
- Env defaults
  -  (initial; tune 0.1–0.2)
  - 
  - 
- HotPath confidence
  - In , pass  into  so edge confidence uses prosody-aware fusion when available.
- Retrieval scoring
  - Keep existing convo prosody component (already implemented behind ).
  - Extend to graph/summary: compute averaged prosody certainty from provenance ( + ) and include as  when enabled.
- Summarization bias
  - Use  to prioritize high-certainty turns and filter very low certainty chatter in background summaries.
- Safeguards
  - Neutral default certainty = 0.5 when missing; keep prosody weights modest; enable  only during QA.

### PR2: Sectorization + Decay + Waypoint + Explainability
- Sectorization
  - Add  (rule-based + intent mapping) to tag query/memories with primary + candidate sectors.
  - Fold a small sector prior into  to prefer source types that match sector (e.g., semantic→graph, episodic→convo/summary).
- Salience/Decay/Reinforcement
  - Use existing , , and .
  - Compute sector-specific decay at scoring time using / with env lambdas:
    - 
    - 
    - 
    - 
    - 
- Waypoint expansion (strict)
  - Add  to allow a single strongest 1-hop neighbor under caps/timeout.
  - Env: , .
- Explainability
  - Enrich  with , , , , .
  - Keep compact, typed headers (conf/pro/rec/use); show components in logs when .

### PR3: ContextBuilder (Anthropic Ordering) + Sliding Window
- Add  and wire in  to assemble final messages per turn in one pass:
  1) System: persona/guardrails (single block)  
  2) System: session header (identity, capabilities, constraints)  
  3) System: memory context (headers mode with scalar brackets; auto-expand only when score below threshold)  
  4) Tool/observation summaries (optional)  
  5) Minimal chat history window (token-aware)  
  6) Current user turn
- Defaults and budgets
  - , 
-  - , 
- Make  a provider (does not mutate the aggregator directly); the builder owns ordering and budgets.
- Sliding-window verification (must-do)
  - Use the existing token-aware pruning (TokenEstimator) inside ContextBuilder:
    - Keep all system messages; fit most recent user/assistant turns into  with  minimum.
    - Fallback to pair-based limit if estimator unavailable: , .
  - Ensure identical behavior for voice pipeline and  (typed) paths.
  - Add a small test/harness verifying: system messages persist; recent turns capped by budget; tightening budgets reduces tokens; voice and typed paths match.

### PR4: Optional OpenMemory Source (Cold/Augment)
- Add  retrieval source with strict timeout and tiny budget share:
  - Env:  (default), , .
  - Sidecar client (mirroring ) invoked from retrieval when enabled.
  - Apply sector filters from classifier; limit to 1 candidate; log latency.

## Env Knobs (initial)
- Prosody: , , .
- Sectors/decay: the five  vars above.
- Waypoint: , .
- Context: , , , .
- Sliding window: , , , , .
- Optional OpenMemory: , , .

## Validation & Metrics
- Latency: measure retrieval time, total turn time, and TTFB deltas; keep hot path unchanged or improved.
- Quality: A/B  (0.0 vs 0.1–0.2); track acceptance rate of prosody-influenced picks.
- Sliding window: assert token count stays within budget; persona/header/memory blocks always present.
- Logging: enable  in QA to inspect conf/pro/rec/use/sector components.

## Risks & Mitigations
- Over-weighting prosody → set small weights and neutral defaults; fall back to 0.5 certainty when missing.
- Latency creep from waypoint/OpenMemory → strict caps, timeouts, and small budgets; feature flags.
- Context drift → centralize assembly in ContextBuilder with fixed order and budgets.

## Rollout
- PR1 → PR2 → PR3 (verify sliding window) → PR4 (optional) with feature flags for safe toggling.
- Start with headers mode off in production if needed; enable per agent/tenant via env.

## Acceptance Criteria
- Prosody actively affects retrieval (and edge confidence) with measurable uplift and no latency regressions.
- Sector prior, decay, and optional waypoint improve relevance on targeted tests.
- ContextBuilder produces Anthropic-aligned message order with verified sliding-window behavior across voice and typed paths.
- Optional OpenMemory source can be toggled on to augment recall without affecting hot-path latency.

## References
- Anthropic, Effective Context Engineering for AI Agents: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- OpenMemory Architecture (HMD v2; sectors, salience/decay, waypoint): 
- Current code touchpoints: retrieval scoring (), hot path processor (), store (), confidence strategies (), context injector ().
