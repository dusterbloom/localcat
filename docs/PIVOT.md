# Pivot: localcat as a web UI for nanobot-rs

## Why

localcat grew from an ~800-line Pipecat demo into a ~90k-line Python codebase
(voice pipeline + memory research + eval frameworks + a Tauri app + two Next.js
clients). Meanwhile [nanobot-rs](https://github.com/dusterbloom/nanobot-rs)
already covers the agent side better: single Rust binary, LLM-provider
agnostic, on-device voice (Whisper + TTS), tools, skills, persistent memory,
MLX inference, and an OpenAI-compatible server. Maintaining two local
voice-agent stacks is pure duplication.

## What localcat keeps (its durable assets)

1. **`client/`** — the Pipecat voice-ui-kit web client. This becomes the
   product: a polished browser voice UI. nanobot-rs has no web front end.
2. **`server/core/memory/` (HotMem) + `evals/`** — active memory research.
   This is research, not app code; it should ship to nanobot as a feature
   (port to Rust) or run as a small sidecar/MCP server nanobot can call.

Everything else — the Pipecat Python pipeline, TTS sidecars, STT engines,
intent classifiers — is superseded by nanobot-rs.

## Target architecture

```
browser (client/, WebRTC/WS audio + chat UI)
        │
        ▼
nanobot-rs (single binary: agent loop, LLM, tools, voice, memory)
```

Migration steps, in order, each independently shippable:

1. **Done here:** delete archived/dead code, committed eval outputs, the
   Tauri app (lives on `tauri-app-next`), and the debug client.
2. Point `client/` at nanobot-rs: add a WebSocket/OpenAI-compatible transport
   path so the UI works against nanobot's gateway with no Python server
   running. Keep the existing Pipecat WebRTC path as a fallback until parity.
3. Extract HotMem + evals into their own module (or repo) with a thin HTTP/MCP
   interface; nanobot calls it for memory read/write. Delete `server/core`'s
   pipeline code once the client no longer needs it.
4. When step 2 reaches parity (voice in/out with interruption via nanobot),
   delete `server/` entirely. localcat = web UI + memory research.

## Non-goals

- No new Python services. Anything new goes in nanobot-rs or the client.
- No committed model/eval outputs — `.gitignore` now blocks them.
