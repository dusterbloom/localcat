Scripts and diagnostics

- debug/: one-off debug utilities (e.g., ASI YAML inspection, processor debug)
- compare/: head-to-head and quality comparison scripts for manual validation

Run with the project virtualenv, from the `server/` directory, e.g.:

```
. .venv/bin/activate
python scripts/compare/test_default_strategy.py
python scripts/compare/test_level3_vs_enhanced.py
```

