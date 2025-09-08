import os
import asyncio
from typing import List, Dict, Any
from loguru import logger


async def rebuild_leann_index(index_path: str, docs: List[Dict[str, Any]]) -> None:
    """Rebuild a LEANN index from provided docs.

    Each doc: {'text': str, 'metadata': dict}
    This is best-effort and no-ops if LEANN is unavailable.
    """
    try:
        # Import inside function to keep optional
        from leann.api import LeannBuilder  # type: ignore
    except Exception as e:
        logger.info(f"LEANN not available (optional): {e}")
        return

    try:
        backend = os.getenv("LEANN_BACKEND", "hnsw")
        builder = LeannBuilder(backend_name=backend)
        for doc in docs:
            text = str(doc.get('text', '') or '')
            meta = doc.get('metadata') or {}
            if text.strip():
                builder.add_text(text, meta)
        # Build in a thread to avoid blocking event loop if backend is CPU-heavy
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, builder.build_index, index_path)
        logger.info(f"LEANN index rebuilt at {index_path}")
    except Exception as e:
        logger.warning(f"LEANN rebuild failed: {e}")

