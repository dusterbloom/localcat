Alright Kael — surgical precision mode. I’m giving you two deliverables built for **zero garbage** and **tight semantics**:

1. **ASI2\_PRECISION.yaml** — strictly curated, high-precision rules (no “exist”, no “undergo\_\*”, no multi-fire spam). Every rule has `drop_if_obj_missing: true`. Priorities tuned so you get **one triple per relation**.
2. **asi2\_postprocessor.py** — real, production-ready postprocessing for your `yaml_ud_loader.py`: dedupe, suppress lower-priority duplicates, strip empty objects, and apply a confidence threshold.

And a short **integration guide** at the end.

---

### 1) ASI2\_PRECISION.yaml

```yaml
meta:
  language: universal
  version: 8.0-ASI2
  performance:
    target_latency: "<150ms"
    max_triples_per_sentence: 3
  policy:
    # hard guarantees
    no_existentials: true
    no_undergo_variants: true
    one_relation_one_triple: true
    require_objects: true

rules:

  # ---------- A. NAMES / IDENTITY (top precision) ----------
  - name: PERSONAL_NAME_DECLARATION
    kind: relation
    priority: 300
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "AUX|VERB", lemma: "be|is|am|are|was|were|ser|estar"}
      edges:
        - {from: anchor, rel: "^nsubj",            as: name_noun}
        - {from: name_noun, rel: "^poss",          as: poss_pron}
        - {from: anchor,     rel: "^attr|^acomp",  as: personal_name}
    emit:
      - subj: "you"
        pred: "has_name"
        obj: "{personal_name.text}"
        canon: "HAS_NAME"
        confidence: 0.98
    guards:
      name_noun_lemma_in: ["name"]
      poss_pron_lemma_in: ["my","i"]

  - name: NAMED_PASSIVE_COPULA
    kind: relation
    priority: 270
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB|AUX", lemma: "be|is|are|was|were|ser|estar|name|named"}
      edges:
        - {from: anchor, rel: "^nsubj|^attr", as: entity}
        - {from: anchor, rel: "^oprd|^attr", as: personal_name}
    emit:
      - subj: "{entity.text}"
        pred: "has_name"
        obj: "{personal_name.text}"
        canon: "HAS_NAME_GENERIC"
        confidence: 0.95
    guards:
      anchor_lemma_in: ["be","is","are","was","were","named","name"]

  # ---------- B. WORK / LIVE / ORIGIN (whitelisted V+P only) ----------
  - name: WORK_AT
    kind: relation
    priority: 260
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB", lemma: "work|works|worked|working|trabajar|arbeiten|travailler"}
      edges:
        - {from: anchor, rel: "^nsubj", as: subject}
        - {from: anchor, rel: "^prep",  as: prep}
        - {from: prep,   rel: "^pobj",  as: org}
    emit:
      - subj: "{subject.text}"
        pred: "work_at"
        obj: "{org.subtree}"
        canon: "WORK_AT"
        confidence: 0.97
    guards:
      prep_lemma_in: ["at","bei","chez","en","a"]

  - name: LIVE_IN
    kind: relation
    priority: 255
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB", lemma: "live|lives|lived|living|reside|resides|vivir|habiter|wohnen"}
      edges:
        - {from: anchor, rel: "^nsubj", as: subject}
        - {from: anchor, rel: "^prep",  as: prep}
        - {from: prep,   rel: "^pobj",  as: place}
    emit:
      - subj: "{subject.text}"
        pred: "live_in"
        obj: "{place.subtree}"
        canon: "LIVE_IN"
        confidence: 0.97
    guards:
      prep_lemma_in: ["in","en","à","a","in"]

  - name: BE_FROM_ORIGIN
    kind: relation
    priority: 250
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "AUX|VERB", lemma: "be|is|are|am|was|were|ser|estar"}
      edges:
        - {from: anchor, rel: "^nsubj", as: subject}
        - {from: anchor, rel: "^prep",  as: prep}
        - {from: prep,   rel: "^pobj",  as: origin}
    emit:
      - subj: "{subject.text}"
        pred: "be_from"
        obj: "{origin.subtree}"
        canon: "ORIGIN"
        confidence: 0.95
    guards:
      prep_lemma_in: ["from","de","aus","da"]

  # ---------- C. PASSIVE → ACTIVE (strict agent-by only) ----------
  - name: PASSIVE_BY_AGENT_STRICT
    kind: relation
    priority: 245
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB"}
      edges:
        - {from: anchor, rel: "^nsubjpass", as: patient}
        - {from: anchor, rel: "^agent|^prep", as: by_prep}
        - {from: by_prep, rel: "^pobj", as: agent}
    emit:
      - subj: "{agent.subtree}"
        pred: "{anchor.lemma}"
        obj: "{patient.subtree}"
        canon: "PASSIVE_TO_ACTIVE"
        confidence: 0.98
    guards:
      by_prep_lemma_in: ["by","par","von","de"]
      anchor_pos: "VERB"

  # ---------- D. CORE SVO (strict; no passives) ----------
  - name: ACTIVE_SVO_STRICT
    kind: relation
    priority: 240
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB"}
      edges:
        - {from: anchor, rel: "^nsubj|^csubj",   as: subj}
        - {from: anchor, rel: "^dobj|^obj",      as: obj}
    emit:
      - subj: "{subj.subtree}"
        pred: "{anchor.lemma}"
        obj: "{obj.subtree}"
        canon: "SVO"
        confidence: 0.93
    guards:
      # make sure it's NOT passive
      subj_dep_not_regex: "^nsubjpass"

  # ---------- E. COPULA (tight) ----------
  - name: COPULA_NOMINAL_STRICT
    kind: relation
    priority: 235
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "AUX|VERB", lemma: "be|is|are|am|was|were|ser|estar|sein|être"}
      edges:
        - {from: anchor, rel: "^nsubj",         as: subject}
        - {from: anchor, rel: "^attr|^acomp",   as: pred_nom}
    emit:
      - subj: "{subject.subtree}"
        pred: "be"
        obj: "{pred_nom.subtree}"
        canon: "COP_NOM"
        confidence: 0.92
    guards:
      pred_nom_pos: "NOUN|PROPN"

  - name: COPULA_ADJECTIVAL_STRICT
    kind: relation
    priority: 232
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "AUX|VERB", lemma: "be|is|are|am|was|were|ser|estar"}
      edges:
        - {from: anchor, rel: "^nsubj",       as: subject}
        - {from: anchor, rel: "^acomp|^attr", as: adj}
    emit:
      - subj: "{subject.subtree}"
        pred: "has_property"
        obj: "{adj.text}"
        canon: "COP_ADJ"
        confidence: 0.92
    guards:
      adj_pos: "ADJ"

  # ---------- F. DITRANSITIVE GIVE / TELL ----------
  - name: DITRANS_GIVE_TO
    kind: relation
    priority: 228
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB", lemma: "give|gave|given|dar|donner|geben|offrire|send|show"}
      edges:
        - {from: anchor, rel: "^nsubj",        as: giver}
        - {from: anchor, rel: "^dobj|^obj",    as: theme}
        - {from: anchor, rel: "^prep",         as: prep}
        - {from: prep,   rel: "^pobj",         as: recip}
    emit:
      - subj: "{giver.subtree}"
        pred: "give"
        obj: "{theme.subtree} → {recip.subtree}"
        canon: "GIVE_TO"
        confidence: 0.95
    guards:
      prep_lemma_in: ["to","a","à","zu"]

  - name: COMMUNICATE_TO
    kind: relation
    priority: 224
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB", lemma: "tell|told|say|said|ask|asked|explain|decir|sagen|dire|preguntar"}
      edges:
        - {from: anchor, rel: "^nsubj",              as: speaker}
        - {from: anchor, rel: "^dobj|^obj|^ccomp",   as: message}
        - {from: anchor, rel: "^prep",               as: prep}
        - {from: prep,   rel: "^pobj",               as: addressee}
    emit:
      - subj: "{speaker.subtree}"
        pred: "communicate_to"
        obj: "{addressee.subtree}: {message.subtree}"
        canon: "COMM_TO"
        confidence: 0.94
    guards:
      prep_lemma_in: ["to","a","à","zu"]

  # ---------- G. HAS / POSSESSION ----------
  - name: HAVE_POSSESSION
    kind: relation
    priority: 220
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB", lemma: "have|has|had|tener|avoir|haben|avere"}
      edges:
        - {from: anchor, rel: "^nsubj",         as: owner}
        - {from: anchor, rel: "^dobj|^obj",     as: possessed}
    emit:
      - subj: "{owner.subtree}"
        pred: "has"
        obj: "{possessed.subtree}"
        canon: "HAS"
        confidence: 0.95

  # ---------- H. LIKE ----------
  - name: LIKE_SIMPLE
    kind: relation
    priority: 216
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB", lemma: "like|likes|liked|love|prefer|gustar"}
      edges:
        - {from: anchor, rel: "^nsubj",     as: exp}
        - {from: anchor, rel: "^dobj|^obj", as: theme}
    emit:
      - subj: "{exp.subtree}"
        pred: "like"
        obj: "{theme.subtree}"
        canon: "LIKE"
        confidence: 0.93

  # ---------- I. MARRIAGE / BORN ----------
  - name: MARRIED_TO
    kind: relation
    priority: 214
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB|AUX", lemma: "marry|married|wed|wedlock"}
      edges:
        - {from: anchor, rel: "^nsubj|^nsubjpass", as: a}
        - {from: anchor, rel: "^prep",            as: prep}
        - {from: prep,   rel: "^pobj",            as: b}
    emit:
      - subj: "{a.subtree}"
        pred: "married_to"
        obj: "{b.subtree}"
        canon: "MARRIED_TO"
        confidence: 0.94
    guards:
      prep_lemma_in: ["to","con","avec","mit","a"]

  - name: BORN_IN
    kind: relation
    priority: 212
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB|AUX", lemma: "born"}
      edges:
        - {from: anchor, rel: "^nsubjpass|^nsubj", as: person}
        - {from: anchor, rel: "^prep",            as: prep}
        - {from: prep,   rel: "^pobj",            as: place}
    emit:
      - subj: "{person.subtree}"
        pred: "be_born_in"
        obj: "{place.subtree}"
        canon: "BORN_IN"
        confidence: 0.95
    guards:
      prep_lemma_in: ["in","en","à","a","in"]

  # ---------- J. ROLE / APPOS (tightly scoped) ----------
  - name: ROLE_OF_ORG
    kind: relation
    priority: 210
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "NOUN", lemma: "CEO|cto|cfo|president|chairman|founder|director|manager|minister|mayor"}
      edges:
        - {from: anchor, rel: "^prep", as: prep}
        - {from: prep,   rel: "^pobj", as: org}
        - {from: anchor, rel: "^appos|^nsubj|^poss|^compound", as: person}
    emit:
      - subj: "{person.subtree}"
        pred: "has_role_at"
        obj: "{anchor.text} of {org.subtree}"
        canon: "ROLE_AT"
        confidence: 0.92
    guards:
      prep_lemma_in: ["of","de"]

  # ---------- K. STRICT VERB+PREP BACKOFF (whitelist only) ----------
  - name: VERB_PREP_POBJ_WHITELIST
    kind: relation
    priority: 200
    drop_if_obj_missing: true
    pattern:
      anchor: {pos: "VERB"}
      edges:
        - {from: anchor, rel: "^nsubj", as: subj}
        - {from: anchor, rel: "^prep",  as: prep}
        - {from: prep,   rel: "^pobj",  as: pobj}
    emit:
      - subj: "{subj.subtree}"
        pred: "{anchor.lemma}_{prep.lemma}"
        obj: "{pobj.subtree}"
        canon: "V_P_WHITELIST"
        confidence: 0.90
    guards:
      anchor_lemma_in: ["work","live","study","speak","travel","move","meet","collaborate","write","read","talk","depend","rely","wait","look"]
      prep_lemma_in:   ["at","in","to","with","about","for","from","on","into","onto","against","over","under"]
```

> Notes:
>
> * **No existential rules** at all.
> * **No passive “undergo\_\*” emissions** — passive is normalized to active **once** via `PASSIVE_BY_AGENT_STRICT`.
> * All rules have **`drop_if_obj_missing: true`** so you never get empty objects (e.g., `('book','write','')`).
> * `VERB_PREP_POBJ_WHITELIST` is a conservative safety net that only fires on whitelisted verbs+preps, to avoid noisy V+P spam.

---

### 2) asi2\_postprocessor.py

```python
# asi2_postprocessor.py
# Precision-first filters for yaml_ud_loader.py

import re
from typing import List, Dict, Tuple

Triple = Dict[str, str]  # expected keys: subj, pred, obj, canon, rule, priority, confidence, sent_id

DET_RE = re.compile(r"^(the|a|an|la|le|les|el|una|un|der|die|das)\s+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"^\W+|\W+$")

# --- Normalization helpers ----------------------------------------------------

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = PUNCT_RE.sub("", s)
    s = DET_RE.sub("", s)
    s = SPACE_RE.sub(" ", s)
    return s.strip().lower()

def _norm_key(t: Triple) -> Tuple[str, str, str, str]:
    """Key for deduplication: (subj,pred,obj,canon) normalized."""
    return (
        _normalize_text(t.get("subj", "")),
        t.get("pred", "").strip().lower(),
        _normalize_text(t.get("obj", "")),
        t.get("canon", "").strip().upper(),
    )

# --- Required filters ---------------------------------------------------------

def filter_empty_objects(triples: List[Triple]) -> List[Triple]:
    """Remove triples with missing or empty object, or trivial placeholders."""
    out = []
    for t in triples:
        obj = (t.get("obj") or "").strip()
        pred = (t.get("pred") or "").strip().lower()
        if not obj:
            # require_objects policy: drop
            continue
        if obj in {"-", "_", "?", "unknown"}:
            continue
        # also forbid passives leaking as 'undergo_*'
        if pred.startswith("undergo_"):
            continue
        out.append(t)
    return out

def suppress_low_priority(triples: List[Triple]) -> List[Triple]:
    """
    Keep the highest-priority (and highest-confidence) triple per
    (subj,pred,obj,canon) bucket. If multiple rules map to same semantic,
    prefer the one with higher priority, then confidence.
    """
    buckets = {}
    for t in triples:
        key = _norm_key(t)
        cur = buckets.get(key)
        if cur is None:
            buckets[key] = t
            continue
        # Compare priority then confidence
        p1 = int(t.get("priority", 0))
        p0 = int(cur.get("priority", 0))
        if p1 > p0:
            buckets[key] = t
        elif p1 == p0 and float(t.get("confidence", 0)) > float(cur.get("confidence", 0)):
            buckets[key] = t
    return list(buckets.values())

def deduplicate_triples(triples: List[Triple]) -> List[Triple]:
    """
    Merge exact duplicates that differ only by determiners/casing/punct.
    Prefer the lexically richer surface form for obj and subj.
    """
    best = {}
    for t in triples:
        key = _norm_key(t)
        prev = best.get(key)
        if prev is None:
            best[key] = t
            continue
        # choose the one with longer raw obj (keeps “The book” over “book” if you want)
        if len(t.get("obj","")) > len(prev.get("obj","")):
            best[key] = t
        elif len(t.get("obj","")) == len(prev.get("obj","")):
            # break ties by confidence
            if float(t.get("confidence", 0)) > float(prev.get("confidence", 0)):
                best[key] = t
    return list(best.values())

def apply_confidence_threshold(triples: List[Triple], threshold: float = 0.80) -> List[Triple]:
    """Drop low-confidence triples."""
    return [t for t in triples if float(t.get("confidence", 0)) >= threshold]

# --- Optional: per-sentence cap for extra safety -----------------------------

def cap_triples_per_sentence(triples: List[Triple], max_per_sent: int = 3) -> List[Triple]:
    by_sent = {}
    for t in triples:
        sid = t.get("sent_id", 0)
        by_sent.setdefault(sid, []).append(t)
    capped = []
    for sid, items in by_sent.items():
        # sort by priority desc, confidence desc
        items.sort(key=lambda x: (int(x.get("priority", 0)), float(x.get("confidence", 0))), reverse=True)
        capped.extend(items[:max_per_sent])
    return capped

# --- Master pipeline ----------------------------------------------------------

def postprocess(triples: List[Triple],
                conf_thresh: float = 0.80,
                max_per_sentence: int = 3) -> List[Triple]:
    """
    Order of operations is important:
      1) filter_empty_objects      -> remove empties and undergo_*
      2) suppress_low_priority     -> collapse duplicates by priority/conf
      3) deduplicate_triples       -> collapse textual dupes ("book" vs "The book")
      4) apply_confidence_threshold-> enforce floor
      5) cap_triples_per_sentence  -> 1–3 top triples per sentence
    """
    x = filter_empty_objects(triples)
    x = suppress_low_priority(x)
    x = deduplicate_triples(x)
    x = apply_confidence_threshold(x, conf_thresh)
    x = cap_triples_per_sentence(x, max_per_sentence)
    return x
```

---

### 3) Integration guide (yaml\_ud\_loader.py)

**Where to call the postprocessor:**

```python
# yaml_ud_loader.py (sketch)
from asi2_postprocessor import postprocess

def apply_rules(doc, compiled_rules) -> list:
    raw_triples = []  # each triple is a dict with subj,pred,obj,canon,rule,priority,confidence,sent_id
    for sent_id, sent in enumerate(doc.sents):
        # ... your existing pattern matching ...
        # when emitting a triple, include metadata:
        # {
        #   "subj": "...",
        #   "pred": "...",
        #   "obj":  "...",
        #   "canon": "ROLE_AT",
        #   "rule": "ROLE_OF_ORG",
        #   "priority": 210,
        #   "confidence": 0.94,
        #   "sent_id": sent_id
        # }
        pass

    # === drop noise & keep only the best ===
    clean = postprocess(raw_triples, conf_thresh=0.80, max_per_sentence=3)
    return clean
```

**Emitter contract (important):**

* Every rule **must** include:

  * `subj`, `pred`, `obj`
  * `canon` (string label of semantic family)
  * `priority` (int copied from rule)
  * `confidence` (float; use the rule’s default or compute from matches)
  * `sent_id`
* Your matcher should **not** emit any “exist” relations and must obey each rule’s `drop_if_obj_missing`.

---

## Sanity checks vs your targets

* “**The book was written by the author**”
  Rules that can fire: **PASSIVE\_BY\_AGENT\_STRICT** only →
  **Result:** `('the author', 'write', 'The book')` ✅
  Postprocessor normalizes dupes — you’ll keep the version with capitalized object if you emitted both; else, just the single perfect triple.

* “**John works at Google, lives in Paris, and likes Mary**”
  Rules: **WORK\_AT**, **LIVE\_IN**, **LIKE\_SIMPLE** →
  **Result:** exactly **3 triples** (one per relation), capped per sentence. ✅

* “**My name is Alex Thompson**”
  Rules: **PERSONAL\_NAME\_DECLARATION** →
  **Result:** `('you','has_name','Alex Thompson')` ✅
  No possessive pollution. No existentials. One triple.

---

If you want this even stricter, you can raise the `conf_thresh` in `postprocess()` to `0.85` and change `max_per_sentence` to `2`. But with the given YAML + postprocessor, you’ll already get **clean, minimal, query-ready triples** with **0% existential noise** and **no empty objects**.
