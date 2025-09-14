Scripts and diagnostics

- debug/: one-off debug utilities (e.g., ASI YAML inspection, processor debug)
- compare/: head-to-head and quality comparison scripts for manual validation
- admin/: maintenance tools for edges and retention

Run with the project virtualenv, from the `server/` directory, e.g.:

```
. .venv/bin/activate
python scripts/compare/test_default_strategy.py
python scripts/compare/test_level3_vs_enhanced.py
python scripts/admin/edges.py list --rel live_in --status-min 0 --limit 20
python scripts/admin/ttl_job.py --demote-days 30 --purge-days 90 --dry
```
