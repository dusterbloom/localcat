# Local A/B Testing Guide

This guide shows how to compare prompt/config variants on your Mac (M‑series) using two scripts added to this repo.

## LLM‑Only A/B (fast)
- Script: `server/scripts/ab_llm_ab.py`
- Compares two system prompts (`base` vs `free` or custom files) against your local OpenAI‑compatible server (LM Studio/Ollama).
- Measures: `ttfb_ms`, `total_ms`, response length, optional keyword hit‑rate.

Example:
```
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_MODEL=<your_local_model>
uv run server/scripts/ab_llm_ab.py --runs 10 --variant-a base --variant-b free --keywords memory concise
```
Output CSV is saved to `server/data/ab_results_llm_<timestamp>.csv`.

## Server‑Level A/B (end‑to‑end)
- Script: `server/scripts/ab_server_ab.py`
- Spawns the bot with different env overrides, waits for `/api/health`, records `/api/metrics` while you talk in the client, and summarizes.
- Env hooks supported in `server/core/bot.py`:
  - `VAD_STOP_SECS` (e.g., `0.10` vs `0.20`)
  - `WHISPER_STT_MODEL` (e.g., `SMALL`, `MEDIUM`)
  - `TTS_MODEL`, `TTS_VOICE`, `TTS_SAMPLE_RATE`
  - `OPENAI_THINK` (existing), `SUMMARIZER_INTERVAL_SECS` (existing)

Example:
```
uv run server/scripts/ab_server_ab.py \
  --a OPENAI_THINK=false SUMMARIZER_INTERVAL_SECS=30 VAD_STOP_SECS=0.20 WHISPER_STT_MODEL=MEDIUM \
  --b OPENAI_THINK=true  SUMMARIZER_INTERVAL_SECS=15 VAD_STOP_SECS=0.10 WHISPER_STT_MODEL=SMALL \
  --duration 60 --port 7860
```
Per‑variant JSON logs are saved in `server/data/`. The script prints averages/max for common metrics. Adjust the summarizer keys in the script if you expose more fields.

Tips:
- Set `HF_HUB_OFFLINE=1` after first model downloads to reduce startup latency.
- Keep other background loads minimal for consistent comparisons.
- Run each variant multiple times to smooth variance.
