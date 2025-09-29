"""
Text Processing Pipeline

Strategy pattern implementation for composable text processing.
Addresses Open/Closed and Dependency Inversion principles.
"""

from .base import TextProcessor, ProcessorChain
from .coreference import CoreferenceProcessor

__all__ = [
    "TextProcessor",
    "ProcessorChain",
    "CoreferenceProcessor"
]