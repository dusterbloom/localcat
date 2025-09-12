#!/usr/bin/env python3
"""
Backup and sandbox your HotMem store (SQLite + LMDB) safely.

Creates a timestamped backup copy of the current store, and (optionally)
creates a sandbox copy you can point the bot at via environment variables.

Usage examples:

  # Backup from current env and create a sandbox copy under server/data/sandbox
  uv run python server/scripts/backup_store.py --make-sandbox

  # Explicit paths
  uv run python server/scripts/backup_store.py \
      --sqlite /Users/you/Dev/localcat/data/memory.db \
      --lmdb   /Users/you/Dev/localcat/data/graph.lmdb \
      --make-sandbox

Then export these before starting the bot to use the sandboxed store:

  export HOTMEM_SQLITE=/path/to/sandbox/memory.db
  export HOTMEM_LMDB_DIR=/path/to/sandbox/graph.lmdb

Notes:
- Stop any running bot/process that writes to the DB before backing up, to avoid partial copies.
- LMDB must be copied as a directory (we do a recursive copy).
"""

import argparse
import os
import sys
import shutil
import time
import sqlite3
from pathlib import Path


def copy_lmdb_dir(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"LMDB dir not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", default=os.getenv("HOTMEM_SQLITE", os.path.join("server", "data", "memory.db")))
    p.add_argument("--lmdb", default=os.getenv("HOTMEM_LMDB_DIR", os.path.join("server", "data", "graph.lmdb")))
    p.add_argument("--out-dir", default=os.path.join("server", "data", "backups"))
    p.add_argument("--make-sandbox", action="store_true", help="Also create a sandbox copy under server/data/sandbox")
    args = p.parse_args()

    sqlite_path = Path(args.sqlite).expanduser().resolve()
    lmdb_dir = Path(args.lmdb).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = out_dir / f"store_backup_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Copy SQLite (use online backup to include WAL/shm changes)
    if not sqlite_path.exists():
        print(f"[!] SQLite not found: {sqlite_path}")
        sys.exit(1)
    dst_sqlite = backup_dir / "memory.db"
    try:
        src_conn = sqlite3.connect(str(sqlite_path))
        dst_conn = sqlite3.connect(str(dst_sqlite))
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        try:
            src_conn.close()
        except Exception:
            pass
        try:
            dst_conn.close()
        except Exception:
            pass

    # Copy LMDB dir
    if not lmdb_dir.exists():
        print(f"[!] LMDB dir not found: {lmdb_dir}")
        sys.exit(1)
    copy_lmdb_dir(lmdb_dir, backup_dir / "graph.lmdb")

    print(f"\n✅ Backup complete → {backup_dir}")

    if args.make_sandbox:
        sandbox_dir = Path("server") / "data" / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        # Overwrite sandbox with fresh copies (use online backup for SQLite)
        try:
            src_conn = sqlite3.connect(str(sqlite_path))
            dst_conn = sqlite3.connect(str(sandbox_dir / "memory.db"))
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            try:
                src_conn.close()
            except Exception:
                pass
            try:
                dst_conn.close()
            except Exception:
                pass
        copy_lmdb_dir(lmdb_dir, sandbox_dir / "graph.lmdb")
        print("\n🧪 Sandbox ready:")
        print(f"  SQLite: {sandbox_dir / 'memory.db'}")
        print(f"  LMDB:   {sandbox_dir / 'graph.lmdb'}")
        print("\nExport these before starting the bot to use the sandbox:")
        print(f"  export HOTMEM_SQLITE={sandbox_dir / 'memory.db'}")
        print(f"  export HOTMEM_LMDB_DIR={sandbox_dir / 'graph.lmdb'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
