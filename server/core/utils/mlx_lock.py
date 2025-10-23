"""
Global MLX lock to serialize Metal operations.

CRITICAL for macOS Sequoia: Prevents concurrent Metal initialization
which causes process killing when MLX STT and TTS load simultaneously.
"""

import threading
import asyncio
from functools import wraps
from loguru import logger


# Global lock shared by all MLX-based components (STT/TTS/LLM)
# to prevent concurrent Metal driver access during initialization and inference.
MLX_GLOBAL_LOCK = threading.Lock()


def with_mlx_lock(func):
    """Decorator to wrap synchronous functions with MLX global lock"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with MLX_GLOBAL_LOCK:
            return func(*args, **kwargs)
    return wrapper


def with_mlx_lock_async(func):
    """Decorator to wrap async functions with MLX global lock"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        # Acquire lock in executor to avoid blocking event loop
        await loop.run_in_executor(None, MLX_GLOBAL_LOCK.acquire)
        try:
            return await func(*args, **kwargs)
        finally:
            MLX_GLOBAL_LOCK.release()
    return wrapper
