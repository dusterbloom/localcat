"""
DSPy-based edge extractor for complex sentences

Uses LLM to extract additional edges that spaCy dependency parsing misses
in complex compound sentences.
"""

import os
from typing import List, Tuple, Optional, Any
import dspy
from loguru import logger


class EdgeExtraction(dspy.Signature):
    """Extract knowledge graph edges from text as subject-relation-object triples.

    Input:
    - text: The sentence to extract from
    - existing_edges: Edges already found

    Output (missing_edges): NEW edges only, one per line in format: (subject, relation, object)

    Examples of good edges:
    (alice, works_at, google)
    (alice, loves, python)
    (you, lives_in, san francisco)

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
        temperature: float = 0.0
    ):
        """
        Initialize DSPy edge extractor

        Args:
            model: Model name (default: openai/gpt-4o-mini for speed)
            base_url: Optional base URL for local LLM (uses api_base kwarg)
            api_key: Optional API key (uses env var if not provided)
            max_tokens: Max tokens for response
            temperature: Sampling temperature (0 = deterministic)
        """
        self.model = model
        self.max_tokens = max_tokens

        # Configure DSPy LLM (new API: dspy.LM)
        lm_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add API key if needed (OpenAI)
        if "openai" in model.lower():
            lm_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")

        # Add base URL for local LLMs
        if base_url:
            lm_kwargs["api_base"] = base_url
            if not api_key:
                lm_kwargs["api_key"] = "dummy"  # Local LLMs don't need real API key

        lm = dspy.LM(**lm_kwargs)
        dspy.settings.configure(lm=lm)

        # Create predictor - use Predict (not ChainOfThought) for better local LLM compatibility
        self.extract = dspy.Predict(EdgeExtraction)

        # Add few-shot examples to guide the model
        self._add_examples()

        logger.debug(f"DSPy edge extractor initialized with model: {model}")

    def _add_examples(self):
        """Add few-shot examples to improve extraction quality"""
        examples = [
            dspy.Example(
                text="My name is Bob and I work at Microsoft",
                existing_edges="(you, is, bob)",
                missing_edges="(you, works_at, microsoft)"
            ).with_inputs("text", "existing_edges"),

            dspy.Example(
                text="I'm Alice, a software engineer at Google who loves Python",
                existing_edges="(you, is, alice)\n(alice, is, software engineer)",
                missing_edges="(alice, works_at, google)\n(alice, loves, python)"
            ).with_inputs("text", "existing_edges"),

            dspy.Example(
                text="I live in San Francisco and enjoy hiking",
                existing_edges="(you, lives_in, san francisco)",
                missing_edges="(you, enjoys, hiking)"
            ).with_inputs("text", "existing_edges"),
        ]

        # Set examples for the predictor
        self.extract.demos = examples

    def extract_missing_edges(
        self,
        text: str,
        existing_edges: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract additional edges missed by rule-based extraction

        Args:
            text: Original text
            existing_edges: Edges already extracted by spaCy

        Returns:
            List of additional (subject, relation, object) triples
        """
        # Format existing edges for prompt
        existing_str = "\n".join([
            f"({src}, {rel}, {dst})"
            for src, rel, dst in existing_edges
        ])

        if not existing_str:
            existing_str = "(none)"

        try:
            # Call DSPy predictor
            result = self.extract(
                text=text,
                existing_edges=existing_str
            )

            # Parse output
            missing_edges = self._parse_edges(result.missing_edges)

            # Filter out duplicates of existing edges
            missing_edges = [
                edge for edge in missing_edges
                if edge not in existing_edges
            ]

            logger.debug(
                f"DSPy extracted {len(missing_edges)} additional edges "
                f"(existing: {len(existing_edges)})"
            )

            return missing_edges

        except Exception as e:
            logger.error(f"DSPy extraction failed: {e}")
            return []

    def _parse_edges(self, edge_text: str) -> List[Tuple[str, str, str]]:
        """
        Parse edge text into list of triples

        Expected formats:
        - (subject, relation, object)
        - subject, relation, object
        - subject --[relation]--> object
        """
        edges = []

        for line in edge_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                # Try format: (subject, relation, object)
                if line.startswith("(") and line.endswith(")"):
                    parts = line[1:-1].split(",")
                    if len(parts) == 3:
                        src, rel, dst = [p.strip().strip('"').strip("'") for p in parts]
                        edges.append((src, rel, dst))
                        continue

                # Try format: subject --[relation]--> object
                if "--[" in line and "]-->" in line:
                    src, rest = line.split("--[", 1)
                    rel, dst = rest.split("]-->", 1)
                    src, rel, dst = src.strip(), rel.strip(), dst.strip()
                    edges.append((src, rel, dst))
                    continue

                # Try format: subject, relation, object
                parts = line.split(",")
                if len(parts) == 3:
                    src, rel, dst = [p.strip().strip('"').strip("'") for p in parts]
                    edges.append((src, rel, dst))
                    continue

                logger.warning(f"Could not parse edge: {line}")

            except Exception as e:
                logger.warning(f"Failed to parse edge '{line}': {e}")
                continue

        return edges

    def extract_with_rationale(
        self,
        text: str,
        existing_edges: List[Tuple[str, str, str]]
    ) -> Tuple[List[Tuple[str, str, str]], str]:
        """
        Extract edges and return reasoning

        Returns:
            (missing_edges, rationale)
        """
        existing_str = "\n".join([
            f"({src}, {rel}, {dst})"
            for src, rel, dst in existing_edges
        ])

        if not existing_str:
            existing_str = "(none)"

        try:
            result = self.extract(text=text, existing_edges=existing_str)
            missing_edges = self._parse_edges(result.missing_edges)

            # Get rationale from chain of thought
            rationale = getattr(result, "rationale", "No rationale provided")

            return missing_edges, rationale

        except Exception as e:
            logger.error(f"DSPy extraction with rationale failed: {e}")
            return [], str(e)


def create_dspy_extractor(
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> DSPyEdgeExtractor:
    """
    Factory function to create DSPy extractor with environment config

    Args:
        model: Override model (defaults to env or gpt-4o-mini)
        base_url: Override base URL (defaults to env)

    Returns:
        Configured DSPyEdgeExtractor
    """
    model = model or os.getenv("DSPY_MODEL", "openai/gpt-4o-mini")
    base_url = base_url or os.getenv("DSPY_BASE_URL")

    return DSPyEdgeExtractor(model=model, base_url=base_url)