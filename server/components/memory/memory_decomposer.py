"""
Lightweight clause decomposition utilities for complex sentences.

Design goals:
- Fast and optional: only used when HOTMEM_DECOMPOSE_CLAUSES=true
- Return a small set of clause spans likely to contain atomic facts
- Keep complexity low; leave advanced handling for later iterations
"""
from typing import List
from loguru import logger

try:
    from spacy.tokens import Doc, Span, Token  # type: ignore
except Exception:  # pragma: no cover
    Doc = object  # type: ignore
    Span = object  # type: ignore
    Token = object  # type: ignore


def decompose(doc: "Doc") -> List["Span"]:
    """Return a list of clause spans in the given doc.

    Heuristics:
    - Extract subordinate/compliment clauses via dep in {ccomp, xcomp, advcl}
    - If nothing found, return [] so caller can use the original doc
    - Intentionally conservative to avoid over-splitting
    """
    spans: List[Span] = []
    try:
        for tok in doc:  # type: ignore[attr-defined]
            if tok.dep_ in ("ccomp", "xcomp", "advcl"):
                try:
                    subtree = list(tok.subtree)
                    if subtree:
                        start = subtree[0].i
                        end = subtree[-1].i + 1
                        # Create a span; as_doc() can be used by caller
                        span = doc[start:end]
                        spans.append(span)
                except Exception:
                    continue

        # Special case: "Did I tell you that X ..." → extract after "that"
        try:
            lower = doc.text.lower()
            if " tell you that " in lower:
                idx = lower.find(" tell you that ")
                after = doc.char_span(idx + len(" tell you that "))
                if after is not None and len(after) > 3:
                    spans.append(after)
        except Exception:
            pass

        # Deduplicate by character offsets
        uniq = []
        seen = set()
        for s in spans:
            key = (s.start_char, s.end_char)
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq
    except Exception as e:  # pragma: no cover
        logger.debug(f"decompose() failed, skipping: {e}")
        return []

