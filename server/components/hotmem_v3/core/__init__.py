"""
HotMem v3 Core Module

Core functionality for the revolutionary self-improving AI system.
"""

from .hotmem_v3 import HotMemV3
from .dual_graph_architecture import DualGraphArchitecture
from .interfaces import HotMemV3Interface
from .inference import HotMemInference, InferenceResult

__all__ = [
    "HotMemV3",
    "DualGraphArchitecture", 
    "HotMemV3Interface",
    "HotMemInference",
    "InferenceResult"
]