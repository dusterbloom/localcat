Example environment presets

- `.env.performance_fixes`: Safe tweaks for improved latency without heavy tradeoffs.
- `.env.performance_optimized`: Aggressive performance settings; verify on your hardware.

Usage:
- Copy one of these files to your server root as `.env`, then adjust as needed.
- Or selectively copy variables into your existing `.env`.

Example:
  cp server/config/examples/.env.performance_fixes server/.env
