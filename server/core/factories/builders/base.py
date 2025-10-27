from typing import Any

from config import VoiceAgentConfig


class BaseBuilder:
    def __init__(self, config: VoiceAgentConfig):
        self.config = config

    def build(self) -> Any:  # pragma: no cover - interface only
        raise NotImplementedError

