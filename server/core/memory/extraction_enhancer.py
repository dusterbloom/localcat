"""
Extraction Enhancer - BERT-NER + MiniLM Integration

Enhances the existing pattern-based extraction with BERT-NER and MiniLM
without breaking the current pipeline. Integrates seamlessly into memory_hotpath.py.

Usage:
    # Enable via environment variables
    os.environ['USE_BERT_NER'] = 'true'
    os.environ['USE_MINILM'] = 'true'

    # In memory_hotpath.py after extraction:
    enhanced_triples = enhance_extraction(text, triples, entities, doc)
"""

import os
import time
from typing import List, Tuple, Dict, Any, Optional

from loguru import logger

# Conditional imports to avoid breaking if dependencies not available
try:
    from .bert_ner_extractor import get_bert_extractor
    BERT_NER_AVAILABLE = True
except ImportError:
    BERT_NER_AVAILABLE = False
    logger.debug("BERT-NER not available")

try:
    from .minilm_similarity import get_minilm_similarity
    MINILM_AVAILABLE = True
except ImportError:
    MINILM_AVAILABLE = False
    logger.debug("MiniLM not available")


class ExtractionEnhancer:
    """Enhances extraction with BERT-NER and MiniLM while preserving existing functionality."""

    def __init__(self):
        self.use_bert = BERT_NER_AVAILABLE and os.getenv('USE_BERT_NER', 'false').lower() in ('1', 'true', 'yes')
        self.use_minilm = MINILM_AVAILABLE and os.getenv('USE_MINILM', 'false').lower() in ('1', 'true', 'yes')

        # Runtime policy: relation allowlist / denylist (config-driven)
        self.rel_allowlist = self._load_relation_allowlist()
        self.rel_denylist = self._load_relation_denylist()

        # Verification config (Jina-based entailment scoring)
        try:
            self.verify_enabled = os.getenv('EXTRACT_VERIFY_ENABLED', 'true').lower() in ('1', 'true', 'yes')
            self.verify_threshold = float(os.getenv('EXTRACT_ENT_T', '0.6'))
            self.verify_budget_ms = float(os.getenv('EXTRACT_BUDGET_MS', '80'))
            self.run_on_small_ud = os.getenv('EXTRACT_RUN_ON_SMALL_UD', 'true').lower() in ('1', 'true', 'yes')
            self.run_on_proper_noun = os.getenv('EXTRACT_RUN_ON_PROPER_NOUN_HEURISTIC', 'true').lower() in ('1', 'true', 'yes')
            self.ud_small_k = int(os.getenv('EXTRACT_SMALL_UD_K', '1'))
        except Exception:
            self.verify_enabled = True
            self.verify_threshold = 0.6
            self.verify_budget_ms = 80.0
            self.run_on_small_ud = True
            self.run_on_proper_noun = True
            self.ud_small_k = 1

        if self.use_bert:
            self.bert_extractor = get_bert_extractor()
            logger.info("🤖 BERT-NER extraction enhancement enabled")

        if self.use_minilm:
            self.minilm_similarity = get_minilm_similarity()
            logger.info("🧠 MiniLM semantic enhancement enabled")

    def enhance_extraction(
        self,
        text: str,
        existing_triples: List[Tuple[str, str, str]],
        entities: List[str],
        doc: Any
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
        """
        Enhance existing extraction with BERT-NER and MiniLM.

        Args:
            text: Original text
            existing_triples: Triples from UD pattern extraction
            entities: Entities from UD extraction
            doc: spaCy document (for potential future use)

        Returns:
            Tuple of (enhanced_triples, enhancement_metadata)
        """
        enhancement_metadata = {
            'bert_enhanced': False,
            'minilm_enhanced': False,
            'bert_time_ms': 0,
            'minilm_time_ms': 0,
            'original_triples_count': len(existing_triples),
            'final_triples_count': len(existing_triples)
        }

        enhanced_triples = existing_triples.copy()

        # Phase 1: BERT-NER Enhancement
        if self.use_bert and self._should_run_enhancer(existing_triples, text, doc):
            bert_start = time.perf_counter()
            try:
                # Enhance with BERT-NER
                before_set = set(enhanced_triples)
                bert_enhanced = self.bert_extractor.enhance_triples(text, enhanced_triples)

                # Identify newly proposed triples
                proposed = [t for t in bert_enhanced if t not in before_set]

                # Filter by relation policy and verify via entailment
                accepted: List[Tuple[str, str, str]] = []
                verify_time_ms = 0.0
                verified_added = 0

                if proposed:
                    to_check = []
                    for (s, r, d) in proposed:
                        if not self._relation_allowed(r):
                            continue  # evidence-only; do not persist
                        to_check.append((s, r, d))

                    if to_check and self.verify_enabled and self.verify_threshold > 0:
                        start_v = time.perf_counter()
                        accepted_set = self._verify_triples(text, to_check)
                        verify_time_ms = (time.perf_counter() - start_v) * 1000
                        for t in to_check:
                            if t in accepted_set:
                                accepted.append(t)
                    else:
                        accepted.extend(to_check)

                if accepted:
                    enhanced_triples.extend(accepted)
                    verified_added = len(accepted)

                enhancement_metadata['bert_enhanced'] = True
                enhancement_metadata['bert_time_ms'] = (time.perf_counter() - bert_start) * 1000
                enhancement_metadata['bert_triples_added'] = verified_added
                enhancement_metadata['bert_verify_time_ms'] = verify_time_ms
                enhancement_metadata['bert_verify_threshold'] = self.verify_threshold

                logger.debug(f"🤖 BERT-NER proposed {len(proposed)}; accepted {verified_added} (verify {verify_time_ms:.1f}ms)")

            except Exception as e:
                logger.error(f"❌ BERT-NER enhancement failed: {e}")
                enhancement_metadata['bert_error'] = str(e)

        # Phase 2: MiniLM Semantic Enhancement
        if self.use_minilm:
            minilm_start = time.perf_counter()
            try:
                # Semantic noise filtering and entity linking
                semantically_enhanced = self._enhance_with_semantics(text, enhanced_triples, entities)
                enhanced_triples = semantically_enhanced

                enhancement_metadata['minilm_enhanced'] = True
                enhancement_metadata['minilm_time_ms'] = (time.perf_counter() - minilm_start) * 1000

                logger.debug(f"🧠 MiniLM semantic enhancement completed")

            except Exception as e:
                logger.error(f"❌ MiniLM enhancement failed: {e}")
                enhancement_metadata['minilm_error'] = str(e)

        enhancement_metadata['final_triples_count'] = len(enhanced_triples)
        enhancement_metadata['total_enhancement_time_ms'] = (
            enhancement_metadata['bert_time_ms'] + enhancement_metadata['minilm_time_ms']
        )

        logger.debug(f"🔧 Extraction enhancement: {enhancement_metadata['original_triples_count']} → "
                     f"{enhancement_metadata['final_triples_count']} triples "
                     f"({enhancement_metadata['total_enhancement_time_ms']:.1f}ms)")

        return enhanced_triples, enhancement_metadata

    # -----------------
    # Policy helpers
    # -----------------
    def _load_relation_allowlist(self) -> Optional[set]:
        allow_env = os.getenv('EXTRACT_REL_ALLOWLIST') or os.getenv('BERT_TRIPLE_ALLOWLIST')
        allow_set = set()
        if allow_env:
            allow_set.update([x.strip().lower() for x in allow_env.split(',') if x.strip()])
        allow_file = os.getenv('EXTRACT_REL_ALLOWLIST_FILE') or os.getenv('BERT_TRIPLE_ALLOWLIST_FILE')
        if allow_file and os.path.exists(allow_file):
            try:
                import json
                data = json.load(open(allow_file, 'r'))
                if isinstance(data, list):
                    allow_set.update([str(x).strip().lower() for x in data])
            except Exception:
                pass
        return allow_set or None

    def _load_relation_denylist(self) -> set:
        # Deny generic classification relations by default; configurable
        deny_env = os.getenv('EXTRACT_REL_CLASSIFICATION_DENYLIST', 'is_person,is_organization,is_location,is_entity,is_mentioned')
        return set([x.strip().lower() for x in deny_env.split(',') if x.strip()])

    def _relation_allowed(self, r: str) -> bool:
        r = (r or '').strip().lower()
        if not r:
            return False
        # Deny generic classifiers
        if r in self.rel_denylist:
            return False
        # Allowlist patterns (optional)
        if self.rel_allowlist is None:
            return True
        if r in self.rel_allowlist:
            return True
        # wildcard prefix support: entries ending with '*'
        for pat in self.rel_allowlist:
            if pat.endswith('*') and r.startswith(pat[:-1]):
                return True
        return False

    def _should_run_enhancer(self, existing_triples: List[Tuple[str, str, str]], text: str, doc: Any) -> bool:
        try:
            if self.run_on_small_ud and len(existing_triples or []) <= self.ud_small_k:
                return True
            if self.run_on_proper_noun and self._proper_noun_heavy(text, doc):
                return True
        except Exception:
            pass
        return False

    def _proper_noun_heavy(self, text: str, doc: Any) -> bool:
        try:
            if doc is not None:
                pn = sum(1 for t in getattr(doc, 'ents', []) if getattr(t, 'label_', '').upper() in ('PERSON','ORG','GPE','LOC'))
                if pn >= 1:
                    return True
                # Backoff: POS proper nouns
                pn2 = sum(1 for t in doc if getattr(t, 'tag_', '') in ('NNP','NNPS'))
                return pn2 >= 2
        except Exception:
            pass
        # Text heuristic: >=2 capitalized tokens
        caps = sum(1 for w in (text or '').split() if len(w) > 1 and w[0].isupper())
        return caps >= 2

    # -----------------
    # Verification
    # -----------------
    def _triple_to_sentence(self, s: str, r: str, d: str) -> str:
        s2 = s.strip()
        d2 = d.strip()
        r2 = r.strip().lower()
        if r2 == 'works_at':
            return f"{s2} works at {d2}."
        if r2 == 'lives_in':
            return f"{s2} lives in {d2}."
        if r2 == 'based_in':
            return f"{s2} is based in {d2}."
        if r2 in ('also_known_as','aka','name'):
            return f"{s2} is named {d2}."
        if r2.startswith('favorite_'):
            key = r2.split('_',1)[1]
            subj = 'your' if s2.lower() in ('you','you:you') else f"{s2}'s"
            return f"{subj} favorite {key} is {d2}."
        return f"{s2} {r2.replace('_',' ')} {d2}."

    def _verify_triples(self, source_text: str, triples: List[Tuple[str, str, str]]) -> set:
        accepted = set()
        if not triples:
            return accepted
        start = time.perf_counter()
        util_log = os.getenv('EXTRACT_UTILITY_LOG', '').strip()
        try:
            from .rerank_jina import get_jina_reranker
            jr = get_jina_reranker()
        except Exception:
            # If reranker unavailable, accept all (policy fallback)
            return set(triples)

        for (s, r, d) in triples:
            # Budget check
            if (time.perf_counter() - start) * 1000 > self.verify_budget_ms:
                break
            hyp = self._triple_to_sentence(s, r, d)
            try:
                p = jr.score(source_text, [hyp])[0]  # entailment prob
            except Exception:
                p = 0.0
            # Optional utility logging (per-triple)
            if util_log:
                try:
                    import json, time as _t
                    ctx_tag = os.getenv('EXTRACT_CONTEXT_TAG') or None
                    ctx_variant = os.getenv('EXTRACT_CONTEXT_VARIANT') or None
                    src_excerpt = (source_text or '')[:240]
                    rec = {
                        "ts": int(_t.time() * 1000),
                        "s": s,
                        "r": r,
                        "d": d,
                        "hyp": hyp,
                        "p_entail": float(p),
                        "accepted": bool(p >= self.verify_threshold),
                        "ctx_tag": ctx_tag,
                        "ctx_variant": ctx_variant,
                        "src_excerpt": src_excerpt,
                        "threshold": self.verify_threshold,
                    }
                    with open(util_log, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            if p >= self.verify_threshold:
                accepted.add((s, r, d))
        return accepted

    def _enhance_with_semantics(
        self,
        text: str,
        triples: List[Tuple[str, str, str]],
        entities: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Enhance triples using MiniLM semantic understanding.

        Args:
            text: Original text
            triples: Current triples
            entities: Current entities

        Returns:
            Semantically enhanced triples
        """
        enhanced_triples = triples.copy()

        # 1. Semantic noise detection
        user_facts = [t for t in triples if t[0] in ['you', 'i', 'my', 'mine']]
        mentioned_entities = self._extract_mentioned_entities(triples)

        if user_facts and mentioned_entities:
            for fact in user_facts:
                fact_text = self._triple_to_text(fact)
                is_noise = self.minilm_similarity.detect_semantic_noise(fact_text, mentioned_entities)
                if is_noise:
                    logger.debug(f"🧠 Detected semantic noise in: {fact_text}")

        # 2. Coreference resolution improvement
        self._resolve_coreferences_with_semantics(enhanced_triples, text)

        # 3. Entity linking and disambiguation
        self._enhance_entity_linking(enhanced_triples, entities, text)

        return enhanced_triples

    def _extract_mentioned_entities(self, triples: List[Tuple[str, str, str]]) -> List[str]:
        """Extract entities that are mentioned but not necessarily facts about the user."""
        mentioned = []
        for subject, relation, obj in triples:
            # Filter out direct user facts
            if subject not in ['you', 'i', 'my', 'mine'] and relation not in ['is_person', 'is_entity']:
                mentioned.append(obj)
                mentioned.append(subject)
        return list(set(mentioned))

    def _triple_to_text(self, triple: Tuple[str, str, str]) -> str:
        """Convert a triple back to natural language text."""
        subject, relation, obj = triple
        if relation == 'works_at':
            return f"I work at {obj}"
        elif relation == 'live_in':
            return f"I live in {obj}"
        elif relation == 'is':
            return f"I am {obj}"
        else:
            return f"{subject} {relation} {obj}"

    def _resolve_coreferences_with_semantics(self, triples: List[Tuple[str, str, str]], text: str):
        """Improve coreference resolution using semantic similarity."""
        # Find potential pronouns in triples
        pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them']
        candidate_entities = []

        # Extract candidate entities from triples
        for subject, relation, obj in triples:
            if subject not in pronouns:
                candidate_entities.append(subject)
            if obj not in pronouns:
                candidate_entities.append(obj)

        # Try to resolve pronouns
        for i, (subject, relation, obj) in enumerate(triples):
            if subject in pronouns and candidate_entities:
                resolved = self.minilm_similarity.resolve_coreference(subject, candidate_entities, text)
                if resolved:
                    triples[i] = (resolved, relation, obj)
                    logger.debug(f"🧠 Resolved '{subject}' → '{resolved}'")

    def _enhance_entity_linking(self, triples: List[Tuple[str, str, str]], entities: List[str], text: str):
        """Enhance entity linking using semantic similarity."""
        # Find similar entities and merge them
        entity_similarities = {}

        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:  # Avoid duplicate comparisons
                    similarity = self.minilm_similarity.compute_similarity(entity1, entity2)
                    if similarity.similarity_score > 0.8:  # High similarity threshold
                        entity_similarities[(entity1, entity2)] = similarity.similarity_score

        # Merge highly similar entities in triples
        for (entity1, entity2), similarity in entity_similarities.items():
            for i, (subject, relation, obj) in enumerate(triples):
                if subject == entity2:
                    triples[i] = (entity1, relation, obj)
                elif obj == entity2:
                    triples[i] = (subject, relation, entity1)

        if entity_similarities:
            logger.debug(f"🧠 Merged {len(entity_similarities)} similar entity pairs")

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about the enhancement system."""
        stats = {
            'bert_enabled': self.use_bert,
            'minilm_enabled': self.use_minilm,
            'bert_available': BERT_NER_AVAILABLE,
            'minilm_available': MINILM_AVAILABLE
        }

        if self.use_bert:
            stats['bert_stats'] = self.bert_extractor.get_performance_stats()

        if self.use_minilm:
            stats['minilm_stats'] = self.minilm_similarity.get_performance_stats()

        return stats


# Singleton instance
_enhancer_instance: Optional[ExtractionEnhancer] = None
_enhancer_key: Optional[tuple] = None  # (use_bert_flag, use_minilm_flag)


def get_extraction_enhancer() -> ExtractionEnhancer:
    """Get (or refresh) the extraction enhancer singleton based on current env.

    Ensures A/B runs that flip USE_BERT_NER/USE_MINILM can reconfigure the enhancer
    within the same Python process.
    """
    global _enhancer_instance, _enhancer_key

    cur_key = (
        os.getenv('USE_BERT_NER', 'false').lower() in ('1', 'true', 'yes'),
        os.getenv('USE_MINILM', 'false').lower() in ('1', 'true', 'yes'),
    )

    # Initialize or refresh when env flags change
    if _enhancer_instance is None or _enhancer_key != cur_key:
        _enhancer_instance = ExtractionEnhancer()
        _enhancer_key = cur_key

    return _enhancer_instance


def enhance_extraction(
    text: str,
    existing_triples: List[Tuple[str, str, str]],
    entities: List[str],
    doc: Any
) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    """
    Convenience function for enhancing extraction.

    This is the main entry point for integrating BERT-NER and MiniLM
    into the existing memory extraction pipeline.

    Args:
        text: Original text
        existing_triples: Triples from current pattern extraction
        entities: Entities from current pattern extraction
        doc: spaCy document

    Returns:
        Tuple of (enhanced_triples, enhancement_metadata)
    """
    enhancer = get_extraction_enhancer()
    return enhancer.enhance_extraction(text, existing_triples, entities, doc)


if __name__ == "__main__":
    # Simple test
    enhancer = ExtractionEnhancer()

    test_text = "I work at OpenAI and live in San Francisco."
    test_triples = [('you', 'work_at', 'openai')]
    test_entities = ['you', 'openai']

    enhanced_triples, metadata = enhancer.enhance_extraction(test_text, test_triples, test_entities, None)

    print(f"Original: {test_triples}")
    print(f"Enhanced: {enhanced_triples}")
    print(f"Metadata: {metadata}")
    print(f"Stats: {enhancer.get_enhancement_stats()}")
