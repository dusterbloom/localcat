"""
HotMem v3 Component

A revolutionary self-improving AI system with dual graph architecture,
active learning, and real-time streaming extraction capabilities.

This component provides:
- Real-time knowledge graph construction
- Dual memory architecture (working + long-term)
- Active learning with pattern detection
- Streaming extraction for voice conversations
- Production-ready optimization for Apple Silicon
"""

from .core.hotmem_v3 import HotMemV3
from .core.interfaces import HotMemV3Interface
from .extraction.streaming_extraction import StreamingExtractor
from .training.active_learning import ActiveLearningSystem
from .integration.production_integration import HotMemIntegration

__version__ = "3.0.0"
__all__ = [
    "HotMemV3",
    "HotMemV3Interface", 
    "StreamingExtractor",
    "ActiveLearningSystem",
    "HotMemIntegration"
]