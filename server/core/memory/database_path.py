"""
Centralized Database Path Resolution

Single source of truth for memory database location.
Prevents split-brain scenarios where writes go to one database and reads from another.

Architecture:
- Environment-aware (bundle vs development)
- Always returns absolute paths
- Priority: explicit config → legacy vars → auto-detect → default
- Validation to prevent multiple databases
"""

import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def _is_bundled_app() -> bool:
    """
    Detect if running from macOS bundle or development environment.

    Returns:
        True if running from .app bundle, False if development
    """
    # Check if running from .app bundle
    if getattr(sys, 'frozen', False):
        return True

    # Check if parent directories include .app/Contents
    executable_path = Path(sys.executable).resolve()
    return '.app/Contents' in str(executable_path)


def _get_application_support_path() -> Path:
    """
    Get the standard Application Support directory path.

    Returns:
        Path to ~/Library/Application Support/LocalCat/data/
    """
    home = Path.home()
    app_support = home / 'Library' / 'Application Support' / 'LocalCat' / 'data'

    # Ensure directory exists
    app_support.mkdir(parents=True, exist_ok=True)

    return app_support


def _get_development_path() -> Path:
    """
    Get the development database path (project-local).

    Returns:
        Path to server/data/ in project root
    """
    # Find project root by looking for server directory
    current = Path(__file__).resolve()

    # Walk up to find server directory
    while current.parent != current:
        if (current / 'server').exists():
            dev_path = current / 'server' / 'data'
            dev_path.mkdir(parents=True, exist_ok=True)
            return dev_path
        current = current.parent

    # Fallback: use current working directory
    fallback = Path.cwd() / 'data'
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_database_path(filename: str = "memory.db") -> Path:
    """
    Get the absolute path to the memory database.

    This is the SINGLE SOURCE OF TRUTH for database location.
    All components must use this function to get the database path.

    Priority order:
    1. MEMORY_DB_PATH env var (explicit override)
    2. HOTMEM_SQLITE env var (legacy compatibility)
    3. MEMORY_SQLITE_PATH env var (alternative legacy)
    4. Auto-detect based on environment (bundle vs dev)
    5. Default fallback

    Args:
        filename: Database filename (default: memory.db)

    Returns:
        Absolute Path to database file

    Raises:
        RuntimeError: If path cannot be determined or is invalid
    """

    # Priority 1: Explicit MEMORY_DB_PATH
    explicit_path = os.getenv("MEMORY_DB_PATH")
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        logger.info(f"[DatabasePath] Using explicit MEMORY_DB_PATH: {path}")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Log loudly for debugging
        logger.critical(f"[DatabasePath] ✓ RESOLVED: {path}")
        return path

    # Priority 2: Legacy HOTMEM_SQLITE
    legacy_hotmem = os.getenv("HOTMEM_SQLITE")
    if legacy_hotmem:
        path = Path(legacy_hotmem).expanduser().resolve()
        logger.warning(f"[DatabasePath] Using legacy HOTMEM_SQLITE: {path}")
        logger.warning("[DatabasePath] Consider migrating to MEMORY_DB_PATH for clarity")

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.critical(f"[DatabasePath] ✓ RESOLVED: {path}")
        return path

    # Priority 3: Alternative legacy MEMORY_SQLITE_PATH
    legacy_memory = os.getenv("MEMORY_SQLITE_PATH")
    if legacy_memory:
        path = Path(legacy_memory).expanduser().resolve()
        logger.warning(f"[DatabasePath] Using legacy MEMORY_SQLITE_PATH: {path}")
        logger.warning("[DatabasePath] Consider migrating to MEMORY_DB_PATH for clarity")

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.critical(f"[DatabasePath] ✓ RESOLVED: {path}")
        return path

    # Priority 4: Auto-detect based on environment
    is_bundled = _is_bundled_app()

    if is_bundled:
        # Running from .app bundle → use Application Support
        base_path = _get_application_support_path()
        path = base_path / filename
        logger.info(f"[DatabasePath] Auto-detected BUNDLE mode: {path}")
    else:
        # Running in development → use project-local
        base_path = _get_development_path()
        path = base_path / filename
        logger.info(f"[DatabasePath] Auto-detected DEVELOPMENT mode: {path}")

    # CRITICAL: Always log final resolved path
    logger.critical(f"[DatabasePath] ✓ RESOLVED: {path}")
    logger.warning(
        f"[DatabasePath] Using auto-detected path. "
        f"Set MEMORY_DB_PATH in .env for explicit control."
    )

    return path


def validate_database_paths(paths: list[Path]) -> None:
    """
    Validate that all components are using the same database path.

    This prevents split-brain scenarios where different components
    read/write to different databases.

    Args:
        paths: List of database paths from different components

    Raises:
        RuntimeError: If multiple different paths detected (split-brain)
    """
    unique_paths = set(str(p.resolve()) for p in paths)

    if len(unique_paths) == 0:
        logger.warning("[DatabasePath] No database paths to validate")
        return

    if len(unique_paths) == 1:
        logger.info(f"[DatabasePath] ✓ Validation passed: All components use {list(unique_paths)[0]}")
        return

    # FATAL: Multiple databases detected
    error_msg = (
        f"[DatabasePath] ❌ SPLIT-BRAIN DETECTED!\n"
        f"Multiple database paths in use:\n"
    )
    for i, path in enumerate(unique_paths, 1):
        error_msg += f"  {i}. {path}\n"

    error_msg += (
        "\nThis will cause data loss and retrieval failures!\n"
        "Fix: Set MEMORY_DB_PATH in .env to explicit path."
    )

    logger.critical(error_msg)
    raise RuntimeError("Database split-brain detected! See logs for details.")


def get_lmdb_path(filename: str = "memory.lmdb") -> Optional[Path]:
    """
    Get the path to LMDB directory (adjacency index).

    LMDB is REQUIRED for graph retrieval performance. Without it, the neighbors()
    method will fall back to slower SQLite queries.

    Priority order:
    1. MEMORY_LMDB_PATH env var (explicit override)
    2. HOTMEM_LMDB_DIR env var (legacy compatibility)
    3. Auto-detect: Same directory as SQLite database
    4. None if explicitly disabled via MEMORY_USE_LMDB=false

    Args:
        filename: LMDB directory name (default: memory.lmdb)

    Returns:
        Path to LMDB directory, or None if explicitly disabled
    """

    # Priority 1: Explicit MEMORY_LMDB_PATH
    explicit_path = os.getenv("MEMORY_LMDB_PATH")
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        logger.info(f"[DatabasePath] Using explicit MEMORY_LMDB_PATH: {path}")

        # Ensure directory exists
        path.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Log loudly for debugging
        logger.critical(f"[DatabasePath] ✓ LMDB RESOLVED: {path}")
        return path

    # Priority 2: Legacy HOTMEM_LMDB_DIR
    legacy_path = os.getenv("HOTMEM_LMDB_DIR")
    if legacy_path:
        path = Path(legacy_path).expanduser().resolve()
        logger.warning(f"[DatabasePath] Using legacy HOTMEM_LMDB_DIR: {path}")
        logger.warning("[DatabasePath] Consider migrating to MEMORY_LMDB_PATH for clarity")

        path.mkdir(parents=True, exist_ok=True)
        logger.critical(f"[DatabasePath] ✓ LMDB RESOLVED: {path}")
        return path

    # Priority 3: Check if explicitly disabled
    lmdb_disabled = os.getenv("MEMORY_USE_LMDB", "").lower() in ("0", "false", "no", "off")
    if lmdb_disabled:
        logger.warning("[DatabasePath] LMDB explicitly disabled via MEMORY_USE_LMDB=false")
        logger.warning("[DatabasePath] Graph retrieval will use slower SQLite fallback")
        return None

    # Priority 4: Auto-detect - use same base directory as SQLite (RECOMMENDED)
    sqlite_path = get_database_path()
    lmdb_path = sqlite_path.parent / filename

    logger.info(f"[DatabasePath] Auto-detected LMDB path (same dir as SQLite): {lmdb_path}")
    logger.warning(
        f"[DatabasePath] Using auto-detected LMDB path. "
        f"Set MEMORY_LMDB_PATH in .env for explicit control."
    )

    # Ensure directory exists
    lmdb_path.mkdir(parents=True, exist_ok=True)

    # CRITICAL: Log loudly for debugging
    logger.critical(f"[DatabasePath] ✓ LMDB RESOLVED: {lmdb_path}")

    return lmdb_path


# Module-level cache for path (avoid re-computing)
_cached_database_path: Optional[Path] = None


def get_cached_database_path() -> Path:
    """
    Get cached database path (computed once per process).

    Returns:
        Absolute Path to database file
    """
    global _cached_database_path

    if _cached_database_path is None:
        _cached_database_path = get_database_path()

    return _cached_database_path
