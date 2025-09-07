#!/usr/bin/env python3
"""
Wrapper for archived MegaFlow runner.

Delegates to server/archive/hotmem_evolution_phase2/scripts/test_megaflow.py
so existing commands keep working after archiving.
"""
import os
import sys
import asyncio
from dotenv import load_dotenv
import importlib.util


def main():
    base = os.path.dirname(__file__)
    # Load server/.env so tests pick up LM Studio / assisted settings
    env_path = os.path.abspath(os.path.join(base, '..', 'server', '.env'))
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    arch_scripts = os.path.abspath(os.path.join(base, '..', 'server', 'archive', 'hotmem_evolution_phase2', 'scripts'))
    arch_file = os.path.join(arch_scripts, 'test_megaflow.py')
    # Ensure local server/ is importable for archived modules (memory_store, etc.)
    sys.path.append(os.path.abspath(os.path.join(base, '..', 'server')))
    # Make archived scripts importable as top-level modules for their local imports
    sys.path.insert(0, arch_scripts)
    if not os.path.exists(arch_file):
        print(f"ERROR: Archived megaflow runner not found at: {arch_file}")
        raise SystemExit(2)
    # Load the archived module from file location to avoid name clash
    spec = importlib.util.spec_from_file_location("archived_test_megaflow", arch_file)
    if spec is None or spec.loader is None:
        print("ERROR: Could not load archived megaflow module spec")
        raise SystemExit(2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    # Expect archived module to define `main()`
    if not hasattr(mod, 'main'):
        print("ERROR: archived test_megaflow.py does not define main()")
        raise SystemExit(2)
    return asyncio.run(mod.main())


if __name__ == '__main__':
    main()
