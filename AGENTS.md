# Repository Guidelines

## Project Structure & Modules
- `server/`: Python 3.12 code for the local voice agent, HotMem components, and utilities. Tests live in `server/tests/` and `server/test_*.py`.
- `client/`: Next.js (React + TypeScript) debug console UI for local WebRTC.
- `docs/`, `assets/`, `scripts/`, `utils/`, `backlog/`: Design notes, images, helper scripts, and research.
- Config: copy `server/config/env.example` to an `.env` the server code loads via `python-dotenv`.

## Build, Test, and Dev Commands
- Server (uv): `cd server && uv run bot.py` – starts the local agent (models may download on first run).
- Server (venv/pip): `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python bot.py`.
- Client: `cd client && npm i && npm run dev` – launches the web UI.
- Tests (Python): run files directly, e.g. `uv run python server/tests/test_integration.py` or `python server/tests/test_integration.py`.
- Optional speedup: after first model download, `HF_HUB_OFFLINE=1 uv run bot.py`.

## Coding Style & Naming
- Python: PEP 8, 4‑space indents, type hints where helpful. Files/functions `snake_case`, classes `PascalCase`.
- TypeScript/React: follow Next.js defaults; components `PascalCase`, hooks `useX`.
- Linting: `client` uses ESLint (`npm run lint`). No enforced Python formatter; keep imports tidy and modules small.

## Testing Guidelines
- Python tests live under `server/tests/` and ad‑hoc `server/test_*.py` modules.
- Name tests `test_*.py`; prefer small, deterministic unit tests around components (e.g., HotMem extractor/retriever).
- To validate integrations, run `server/tests/test_integration.py` directly.

## Commit & Pull Requests
- Commit style: Conventional Commits used in history (e.g., `feat(hotmem): …`, `fix(tts): …`, `docs: …`, `refactor(...): …`).
- PRs should include: clear description, linked issue (if any), affected areas (`server`, `client`, `docs`), test notes (commands and results), and screenshots for UI changes.

## Security & Configuration
- Environment: copy and edit `server/config/env.example` (LLM endpoints, HotMem toggles, storage paths). Keep secrets out of git.
- Local LLMs: use LM Studio/Ollama per README; verify ports and models match your `.env`.
- Data: local stores live under `server/` (e.g., `memory.db`, `graph.lmdb`). Avoid committing generated DBs.

