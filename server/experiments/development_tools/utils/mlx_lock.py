"""Global MLX lock to serialize Metal operations."""

import threading


# Coordinating lock shared by MLX-based components (STT/TTS) to
# avoid concurrent Metal driver access.
MLX_GLOBAL_LOCK = threading.Lock()
