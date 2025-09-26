"""
Base Text Processing Interfaces

Implements strategy pattern for extensible text processing following
Open/Closed and Dependency Inversion principles.

This allows adding new text processing capabilities (like coreference resolution)
without modifying existing extraction code.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, TYPE_CHECKING
from loguru import logger
import time

if TYPE_CHECKING:
    import spacy


class TextProcessor(ABC):
    """
    Abstract base class for text processing strategies.

    Following the Strategy pattern, each processor implements a specific
    text transformation while maintaining a consistent interface.

    Responsibilities:
    - Process spaCy documents
    - Maintain processing metrics
    - Handle errors gracefully
    """

    def __init__(self, name: str):
        self.name = name
        self.metrics = []

    @abstractmethod
    def process(self, doc: "spacy.Doc") -> "spacy.Doc":
        """
        Process a spaCy document and return the modified version.

        Args:
            doc: Input spaCy document

        Returns:
            Processed spaCy document (may be the same object or a new one)

        Note:
            Implementations should handle errors gracefully and return
            the original document if processing fails.
        """
        pass

    def _record_metric(self, elapsed_ms: float, success: bool, details: Optional[str] = None) -> None:
        """Record processing metrics for observability."""
        metric = {
            "timestamp": time.time(),
            "elapsed_ms": elapsed_ms,
            "success": success,
            "details": details
        }
        self.metrics.append(metric)

        # Keep only recent metrics to prevent memory growth
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-500:]

    def get_metrics_summary(self) -> dict:
        """Get summary of processing metrics."""
        if not self.metrics:
            return {"processor": self.name, "total_calls": 0}

        successful = [m for m in self.metrics if m["success"]]
        failed = [m for m in self.metrics if not m["success"]]

        if successful:
            latencies = [m["elapsed_ms"] for m in successful]
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        else:
            avg_latency = p95_latency = 0

        return {
            "processor": self.name,
            "total_calls": len(self.metrics),
            "successful_calls": len(successful),
            "failed_calls": len(failed),
            "success_rate": len(successful) / len(self.metrics) if self.metrics else 0,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency
        }


class ProcessorChain:
    """
    Composable chain of text processors.

    Implements the Chain of Responsibility pattern to allow
    flexible composition of text processing steps.

    Responsibilities:
    - Execute processors in sequence
    - Handle processor failures gracefully
    - Maintain overall processing metrics
    """

    def __init__(self, processors: Optional[List[TextProcessor]] = None):
        self.processors = processors or []

    def add_processor(self, processor: TextProcessor) -> None:
        """Add a processor to the end of the chain."""
        self.processors.append(processor)

    def process(self, doc: "spacy.Doc") -> "spacy.Doc":
        """
        Process document through the entire chain.

        Args:
            doc: Input spaCy document

        Returns:
            Document processed by all processors in sequence

        Note:
            If any processor fails, the chain continues with the last
            successful result, ensuring robustness.
        """
        if not self.processors:
            return doc

        current_doc = doc
        start_time = time.perf_counter()

        for processor in self.processors:
            try:
                processor_start = time.perf_counter()
                processed_doc = processor.process(current_doc)
                elapsed_ms = (time.perf_counter() - processor_start) * 1000

                # Only update if processing succeeded (returned a valid doc)
                if processed_doc is not None:
                    current_doc = processed_doc
                    processor._record_metric(elapsed_ms, True)
                    logger.debug(f"Processor {processor.name} completed in {elapsed_ms:.1f}ms")
                else:
                    processor._record_metric(elapsed_ms, False, "returned None")
                    logger.warning(f"Processor {processor.name} returned None, continuing with previous doc")

            except Exception as e:
                elapsed_ms = (time.perf_counter() - processor_start) * 1000 if 'processor_start' in locals() else 0
                processor._record_metric(elapsed_ms, False, str(e))
                logger.warning(f"Processor {processor.name} failed: {e}, continuing with previous doc")

        total_elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(f"ProcessorChain completed {len(self.processors)} processors in {total_elapsed:.1f}ms")

        return current_doc

    def get_metrics_summary(self) -> List[dict]:
        """Get metrics summary for all processors in the chain."""
        return [processor.get_metrics_summary() for processor in self.processors]

    def clear_metrics(self) -> None:
        """Clear metrics for all processors (useful for testing)."""
        for processor in self.processors:
            processor.metrics.clear()


class NoOpProcessor(TextProcessor):
    """
    No-operation processor for testing and fallback scenarios.

    This processor does nothing but provides a concrete implementation
    of the TextProcessor interface for testing purposes.
    """

    def __init__(self):
        super().__init__("noop")

    def process(self, doc: "spacy.Doc") -> "spacy.Doc":
        """Return the document unchanged."""
        return doc