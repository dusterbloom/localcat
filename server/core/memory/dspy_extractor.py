"""
DSPy-based edge extractor for complex sentences

Uses LLM to extract additional edges that spaCy dependency parsing misses
in complex compound sentences.

NOTE: This module depends on the `dspy` library. If it is unavailable,
imports will fail and callers should handle ImportError gracefully.
HotMemory guards access to this extractor and treats failures as disabled.
"""

import os
from typing import List, Tuple, Optional

from loguru import logger

try:
    import dspy
except Exception as e:  # pragma: no cover - optional dependency
    dspy = None
    logger.warning(f"DSPy not available: {e}")


if dspy is not None:
    class EdgeExtraction(dspy.Signature):  # type: ignore[no-redef]
        """Extract knowledge graph edges from text as subject-relation-object triples.

        Input:
        - text: The sentence to extract from
        - existing_edges: Edges already found

        Output (missing_edges): NEW edges only, one per line in format: (subject, relation, object)

        Important:
        - ALWAYS include subject, relation, AND object (3 parts)
        - Use "you" for first-person subjects (I, me, my)
        - Only output edges NOT already in existing_edges
        - Focus on factual information
        """

        text: str = dspy.InputField()
        existing_edges: str = dspy.InputField()
        missing_edges: str = dspy.OutputField()


class DSPyEdgeExtractor:
    """LLM-based edge extractor using DSPy for complex sentences"""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ):
        if dspy is None:
            raise ImportError("dspy is not installed")

        self.model = model
        self.max_tokens = max_tokens

        # Configure DSPy LLM (new API: dspy.LM)
        lm_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add API key if needed (OpenAI)
        if "openai" in (model or "").lower():
            lm_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")

        # Add base URL for local LLMs
        if base_url:
            lm_kwargs["api_base"] = base_url
            if not api_key:
                lm_kwargs["api_key"] = "dummy"  # Local LLMs don't need real API key

        lm = dspy.LM(**lm_kwargs)
        dspy.settings.configure(lm=lm)

        # Create predictor - use Predict (not ChainOfThought) for better local LLM compatibility
        if dspy is not None:
            self.extract = dspy.Predict(EdgeExtraction)  # type: ignore[name-defined]
        else:  # pragma: no cover
            self.extract = None

        # Add few-shot examples to guide the model
        self._add_examples()

        logger.debug(f"DSPy edge extractor initialized with model: {model}")

    def _add_examples(self) -> None:
        if dspy is None:
            return
        examples = [
            dspy.Example(
                text="My name is Bob and I work at Microsoft",
                existing_edges="(you, is, bob)",
                missing_edges="(you, works_at, microsoft)",
            ).with_inputs("text", "existing_edges"),
            dspy.Example(
                text="I'm Alice, a software engineer at Google who loves Python",
                existing_edges="(you, is, alice)\n(alice, is, software engineer)",
                missing_edges="(alice, works_at, google)\n(alice, loves, python)",
            ).with_inputs("text", "existing_edges"),
            dspy.Example(
                text="I live in San Francisco and enjoy hiking",
                existing_edges="(you, lives_in, san francisco)",
                missing_edges="(you, enjoys, hiking)",
            ).with_inputs("text", "existing_edges"),
        ]
        self.extract.demos = examples  # type: ignore[union-attr]

    def extract_missing_edges(
        self,
        text: str,
        existing_edges: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str, str]]:
        if dspy is None:
            return []
        existing_str = "\n".join([f"({s}, {r}, {d})" for s, r, d in existing_edges]) or "(none)"
        try:
            result = self.extract(text=text, existing_edges=existing_str)  # type: ignore[union-attr]
            missing = self._parse_edges(getattr(result, "missing_edges", ""))
            return [edge for edge in missing if edge not in existing_edges]
        except Exception as e:
            logger.error(f"DSPy extraction failed: {e}")
            return []

    def _parse_edges(self, edge_text: str) -> List[Tuple[str, str, str]]:
        edges: List[Tuple[str, str, str]] = []
        for line in (edge_text or "").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith("(") and line.endswith(")"):
                    parts = line[1:-1].split(",")
                    if len(parts) == 3:
                        s, r, d = [p.strip().strip('"').strip("'") for p in parts]
                        edges.append((s, r, d))
                        continue
                if "--[" in line and "]-->" in line:
                    s, rest = line.split("--[", 1)
                    r, d = rest.split("]-->", 1)
                    edges.append((s.strip(), r.strip(), d.strip()))
                    continue
                parts = line.split(",")
                if len(parts) == 3:
                    s, r, d = [p.strip().strip('"').strip("'") for p in parts]
                    edges.append((s, r, d))
                    continue
                logger.warning(f"Could not parse edge: {line}")
            except Exception as e:
                logger.warning(f"Failed to parse edge '{line}': {e}")
        return edges


def create_dspy_extractor(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> DSPyEdgeExtractor:
    """Factory for DSPy edge extractor (env-configured)."""
    model = model or os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    base_url = base_url or os.getenv("DSPY_BASE_URL")
    return DSPyEdgeExtractor(model=model, base_url=base_url)

