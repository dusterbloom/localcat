"""
Triple canonicalization utilities.

Purpose: Normalize (subject, relation, object) triples to a canonical
lexicalization so evaluation and downstream components align.

Canonicalizations:
- Lowercase and strip determiners/pronouns (delegated to callers as needed)
- Unify copula: "is" -> "be"
- Fold common verb+preposition constructions into lexicalized relations:
  work at/in/on -> works_at/works_in/works_on
  live in/at    -> lives_in
  focus on      -> focus_on
  agree on      -> agree_on
  result in     -> result_in
  lead to       -> lead_to
  enter into    -> enter_into
  comply with   -> comply_with
  benefit from  -> benefit_from
  adhere to     -> adhere_to
  apply to      -> apply_to
  consist of/in -> consist_of / consist_in

The prepositional particle is removed from the object (e.g., "at google" -> "google").
"""

from __future__ import annotations

from typing import Tuple


def _strip_prep(obj: str) -> Tuple[str, str]:
    """Return (prep, target) if object starts with a known preposition, else ("", obj)."""
    if not obj:
        return "", obj
    t = obj.strip().lower()
    preps = ("at", "in", "on", "to", "into", "with", "from", "of", "upon")
    parts = t.split()
    if parts and parts[0] in preps:
        return parts[0], " ".join(parts[1:]).strip()
    return "", t


def _verb_prep_fold(r: str, obj: str) -> Tuple[str, str]:
    """Fold verb+preposition into a lexicalized relation when whitelisted."""
    base = r.strip().lower()
    prep, target = _strip_prep(obj)

    # Normalize some already-lexicalized variants (e.g., work_at -> works_at)
    VARIANTS = {
        "work_at": "works_at",
        "work_in": "works_in",
        "work_on": "works_on",
        "live_in": "lives_in",
        "live_at": "lives_in",
        "focus_on": "focus_on",
        "agree_on": "agree_on",
        "result_in": "result_in",
        "lead_to": "lead_to",
        "enter_into": "enter_into",
        "comply_with": "comply_with",
        "benefit_from": "benefit_from",
        "adhere_to": "adhere_to",
        "apply_to": "apply_to",
        "consist_of": "consist_of",
        "consist_in": "consist_in",
    }
    if base in VARIANTS:
        return VARIANTS[base], target or obj

    # Whitelist mapping for verb + preposition
    VP = {
        "work": {"at": "works_at", "in": "works_in", "on": "works_on"},
        "live": {"in": "lives_in", "at": "lives_in"},
        "focus": {"on": "focus_on"},
        "agree": {"on": "agree_on"},
        "result": {"in": "result_in"},
        "lead": {"to": "lead_to"},
        "enter": {"into": "enter_into"},
        "comply": {"with": "comply_with"},
        "benefit": {"from": "benefit_from"},
        "adhere": {"to": "adhere_to"},
        "apply": {"to": "apply_to"},
        "consist": {"of": "consist_of", "in": "consist_in"},
    }

    if prep and base in VP and prep in VP[base]:
        return VP[base][prep], target
    return base, obj


def canonicalize_triple(s: str, r: str, o: str) -> Tuple[str, str, str]:
    """Return a canonicalized (s, r, o). Assumes s/o are already lowercased and stripped
    of determiners/pronouns by the caller; r is lowercased here.
    """
    rel = (r or "").strip().lower()
    # Unify copula
    if rel == "is":
        rel = "be"

    # Fold verb+preposition into relation when whitelisted
    rel2, obj2 = _verb_prep_fold(rel, (o or "").strip().lower())

    return s, rel2, obj2

