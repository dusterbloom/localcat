#!/usr/bin/env python3
"""
Wrapper for archived MegaFlow runner.

Delegates to server/archive/hotmem_evolution_phase2/scripts/test_megaflow.py
so existing commands keep working after archiving.
"""
import os
import sys
import asyncio


def main():
    base = os.path.dirname(__file__)
    archived_pkg = os.path.join(base, '..', 'server', 'archive', 'hotmem_evolution_phase2')
    # Ensure package path for fully-qualified import
    sys.path.append(os.path.abspath(os.path.join(base, '..')))
    # Import archived runner using package path
    try:
        from server.archive.hotmem_evolution_phase2.scripts.test_megaflow import main as archived_main  # type: ignore
    except Exception as e:
        print(f"ERROR: Could not import archived megaflow runner: {e}")
        print(f"Expected at: {archived_pkg}")
        raise
    return asyncio.run(archived_main())


if __name__ == '__main__':
    main()

