#!/usr/bin/env python3
"""
Multilingual Graph Extractor - Enhanced ReLiK replacement for HotMem
Leverages spaCy's multilingual transformers and dependency parsing
to extract rich semantic graphs from text in any language.
"""

import os
import time
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import spacy
from spacy.tokens import Doc, Token, Span
from loguru import logger
import torch

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.info("sentence-transformers not available, using rule-based only")


@dataclass
class RelationPattern:
    """Represents a relation extraction pattern"""
    relation: str
    verb_lemmas: Set[str]
    preps: Optional[Set[str]] = None
    confidence: float = 0.8
    
    
class MultilingualGraphExtractor:
    """
    Advanced multilingual relation extractor using:
    1. Cross-lingual verb embeddings for relation mapping
    2. Enhanced dependency parsing patterns
    3. Semantic role labeling integration
    4. Contextual relation inference
    """
    
    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        """Initialize extractor with optional semantic model."""
        self.model_id = model_id or "multilingual-graph"
        self.device = device
        self.nlp_cache: Dict[str, spacy.Language] = {}
        self._ready = False
        
        # Load multilingual sentence transformer if available
        self.semantic_model = None
        if SBERT_AVAILABLE:
            try:
                # Use same model as HotMem for consistency
                self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.semantic_model.to(device)
                logger.info("Loaded multilingual semantic model")
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}")
        
        # Initialize relation patterns with multilingual verbs
        self._init_relation_patterns()
        
        # Cache for verb embeddings
        self.verb_embeddings: Dict[str, torch.Tensor] = {}
        self.relation_embeddings: Dict[str, torch.Tensor] = {}
        
        # Initialize relation embeddings if semantic model available
        if self.semantic_model:
            self._init_relation_embeddings()
    
    def _init_relation_patterns(self):
        """Initialize multilingual relation patterns."""
        self.patterns = [
            # Location relations
            RelationPattern("lives_in", 
                {"live", "reside", "dwell", "stay", "inhabit", 
                 "vivir", "residir", "habitar",  # Spanish
                 "vivre", "habiter", "résider",  # French
                 "wohnen", "leben",  # German
                 "vivere", "abitare", "risiedere"},  # Italian
                {"in", "at", "on", "en", "à", "dans", "auf", "bei", "a"}),
            
            RelationPattern("works_at",
                {"work", "employ", "labor",
                 "trabajar", "laborar",
                 "travailler", "bosser",
                 "arbeiten", "jobben",
                 "lavorare", "operare"},
                {"at", "for", "in", "para", "chez", "bei", "presso"}),
            
            RelationPattern("born_in",
                {"born", "birth",
                 "nacer", "nacido",
                 "naître", "né",
                 "geboren", "geburt",
                 "nascere", "nato"},
                {"in", "at", "en", "à", "in"}),
            
            # Creation/founding relations
            RelationPattern("founded",
                {"found", "establish", "create", "start", "launch", "build",
                 "fundar", "establecer", "crear",
                 "fonder", "établir", "créer",
                 "gründen", "etablieren",
                 "fondare", "stabilire", "creare"}),
            
            RelationPattern("invented",
                {"invent", "discover", "develop", "pioneer",
                 "inventar", "descubrir", "desarrollar",
                 "inventer", "découvrir", "développer",
                 "erfinden", "entdecken", "entwickeln",
                 "inventare", "scoprire", "sviluppare"}),
            
            # Ownership/possession
            RelationPattern("owns",
                {"own", "possess", "have", "hold",
                 "poseer", "tener",
                 "posséder", "avoir",
                 "besitzen", "haben",
                 "possedere", "avere"}),
            
            # Educational relations
            RelationPattern("studied_at",
                {"study", "attend", "graduate",
                 "estudiar", "asistir",
                 "étudier", "assister",
                 "studieren", "besuchen",
                 "studiare", "frequentare"},
                {"at", "in", "from", "en", "à", "bei", "a"}),
            
            RelationPattern("teaches_at",
                {"teach", "teaches", "lecture", "instruct", "educate",
                 "enseñar", "instruir",
                 "enseigner", "instruire",
                 "lehren", "unterrichten",
                 "insegnare", "istruire"},
                {"at", "in", "en", "à", "bei", "a"}),
        ]
        
        # Build verb-to-relation index for fast lookup
        self.verb_to_relations: Dict[str, List[RelationPattern]] = defaultdict(list)
        for pattern in self.patterns:
            for verb in pattern.verb_lemmas:
                self.verb_to_relations[verb.lower()].append(pattern)
    
    def _init_relation_embeddings(self):
        """Pre-compute embeddings for target relations."""
        if not self.semantic_model:
            return
            
        relation_descriptions = {
            "lives_in": "person lives in location",
            "works_at": "person works at organization",
            "born_in": "person was born in location",
            "founded": "person founded organization",
            "invented": "person invented something",
            "owns": "person owns something",
            "studied_at": "person studied at institution",
            "teaches_at": "person teaches at institution",
            "married_to": "person is married to another person",
            "child_of": "person is child of parent",
            "parent_of": "person is parent of child",
        }
        
        for rel, desc in relation_descriptions.items():
            self.relation_embeddings[rel] = self.semantic_model.encode(desc, convert_to_tensor=True)
    
    def _get_nlp(self, lang: str = "en") -> spacy.Language:
        """Get or cache spaCy model for language."""
        if lang not in self.nlp_cache:
            # Try transformer models first for better accuracy
            transformer_models = {
                "en": "en_core_web_trf",
                "de": "de_dep_news_trf", 
                "fr": "fr_dep_news_trf",
                "es": "es_dep_news_trf",
                "zh": "zh_core_web_trf"
            }
            
            # Fallback to smaller models
            small_models = {
                "en": "en_core_web_sm",
                "es": "es_core_news_sm",
                "fr": "fr_core_news_sm", 
                "de": "de_core_news_sm",
                "it": "it_core_news_sm",
                "pt": "pt_core_news_sm",
                "nl": "nl_core_news_sm",
                "zh": "zh_core_web_sm"
            }
            
            model_name = None
            # Try transformer model first
            if lang in transformer_models:
                try:
                    model_name = transformer_models[lang]
                    self.nlp_cache[lang] = spacy.load(model_name)
                    logger.info(f"Loaded transformer model {model_name}")
                except:
                    model_name = None
            
            # Fallback to small model
            if not model_name and lang in small_models:
                try:
                    model_name = small_models[lang]
                    self.nlp_cache[lang] = spacy.load(model_name)
                    logger.info(f"Loaded model {model_name}")
                except:
                    model_name = None
            
            # Final fallback to English
            if not model_name:
                try:
                    self.nlp_cache[lang] = spacy.load("en_core_web_sm")
                    logger.warning(f"No model for {lang}, using English")
                except:
                    return None
                    
        return self.nlp_cache.get(lang)
    
    def _extract_subject_object_pairs(self, token: Token) -> List[Tuple[Token, Token, Optional[Token]]]:
        """Extract subject-object pairs from a verb token, including prepositions."""
        pairs = []
        
        # Find all subjects (including passive)
        subjects = []
        objects = []
        
        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
                subjects.append(child)
            # Handle agent in passive voice ("was founded by")
            elif child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by" and token.dep_ == "ROOT"):
                for grandchild in child.children:
                    if grandchild.dep_ in {"pobj", "obj"}:
                        # Check for conjunction (Steve Jobs and Steve Wozniak)
                        agent_subjects = [grandchild]
                        for conj_child in grandchild.children:
                            if conj_child.dep_ == "conj":
                                agent_subjects.append(conj_child)
                        subjects.extend(agent_subjects)
        
        # If passive voice with no explicit subject, the nsubjpass is the object
        if token.tag_ in {"VBN", "VBD"} and any(c.dep_ == "auxpass" for c in token.children):
            # In passive, nsubjpass is actually the object
            actual_subjects = []
            actual_objects = []
            for child in token.children:
                if child.dep_ == "nsubjpass":
                    actual_objects.append(child)
                elif child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by"):
                    for gc in child.children:
                        if gc.dep_ in {"pobj", "obj"}:
                            actual_subjects.append(gc)
                            # Check for conjunctions
                            for conj in gc.children:
                                if conj.dep_ == "conj":
                                    actual_subjects.append(conj)
            
            # Swap subjects and objects for passive
            if actual_subjects and actual_objects:
                subjects = actual_subjects
                objects = actual_objects
        
        # Find direct objects
        if not objects:
            for child in token.children:
                if child.dep_ in {"dobj", "obj", "attr"}:
                    objects.append(child)
        
        # Create pairs for direct objects
        for obj in objects:
            for subj in subjects:
                pairs.append((subj, obj, None))
        
        # Find prepositional objects
        for child in token.children:
            if child.dep_ == "prep" and child.text.lower() != "by":  # Skip "by" (already handled)
                for grandchild in child.children:
                    if grandchild.dep_ in {"pobj", "obj"}:
                        for subj in subjects:
                            pairs.append((subj, grandchild, child))
        
        # Handle compound verbs (phrasal verbs)
        for child in token.children:
            if child.dep_ == "prt":  # particle
                for sibling in token.children:
                    if sibling.dep_ == "prep":
                        for grandchild in sibling.children:
                            if grandchild.dep_ in {"pobj", "obj"}:
                                for subj in subjects:
                                    pairs.append((subj, grandchild, sibling))
        
        return pairs
    
    def _get_full_entity_text(self, token: Token) -> str:
        """Get full entity text including compounds and modifiers."""
        # For noun chunks, get the full chunk
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        
        # Otherwise get the subtree
        subtree = list(token.subtree)
        subtree.sort(key=lambda t: t.i)
        return " ".join(t.text for t in subtree)
    
    def _match_relation_semantic(self, verb_text: str, prep: Optional[str] = None) -> Tuple[str, float]:
        """Match verb to relation using semantic similarity."""
        if not self.semantic_model or not self.relation_embeddings:
            return None, 0.0
        
        # Create context string
        context = verb_text
        if prep:
            context = f"{verb_text} {prep}"
        
        # Get embedding for context
        if context not in self.verb_embeddings:
            self.verb_embeddings[context] = self.semantic_model.encode(context, convert_to_tensor=True)
        
        verb_emb = self.verb_embeddings[context]
        
        # Find most similar relation
        best_relation = None
        best_score = 0.0
        
        for rel, rel_emb in self.relation_embeddings.items():
            similarity = torch.cosine_similarity(verb_emb.unsqueeze(0), rel_emb.unsqueeze(0)).item()
            if similarity > best_score and similarity > 0.5:  # Threshold
                best_score = similarity
                best_relation = rel
        
        return best_relation, best_score
    
    def _match_relation_pattern(self, verb_lemma: str, prep: Optional[str] = None) -> Tuple[str, float]:
        """Match verb and preposition to relation using patterns."""
        verb_lower = verb_lemma.lower()
        
        # Check patterns that match this verb
        matching_patterns = self.verb_to_relations.get(verb_lower, [])
        
        for pattern in matching_patterns:
            # If pattern has no prep requirement, match
            if pattern.preps is None:
                return pattern.relation, pattern.confidence
            
            # If pattern requires prep and we have matching prep
            if prep and prep.lower() in pattern.preps:
                return pattern.relation, pattern.confidence
        
        return None, 0.0
    
    def _extract_from_dependencies(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract relations using advanced dependency parsing."""
        triples = []
        
        for token in doc:
            # Process verbs (including coordinated verbs)
            if token.pos_ == "VERB":
                pairs = self._extract_subject_object_pairs(token)
                
                for subj_token, obj_token, prep_token in pairs:
                    # Get full entity texts
                    subject = self._get_full_entity_text(subj_token)
                    obj = self._get_full_entity_text(obj_token)
                    prep = prep_token.text if prep_token else None
                    
                    # Try pattern matching first
                    relation, conf = self._match_relation_pattern(token.lemma_, prep)
                    
                    # If no pattern match, try semantic matching
                    if not relation and self.semantic_model:
                        relation, conf = self._match_relation_semantic(token.text, prep)
                    
                    # If still no match, create generic relation
                    if not relation:
                        if prep:
                            relation = f"{token.lemma_}_{prep}"
                        else:
                            relation = token.lemma_
                        conf = 0.5
                    
                    triples.append((subject, relation, obj, conf))
                
                # Handle coordinated verbs (e.g., "lives in X and teaches at Y")
                for conj_verb in token.children:
                    if conj_verb.dep_ == "conj" and conj_verb.pos_ == "VERB":
                        conj_pairs = self._extract_subject_object_pairs(conj_verb)
                        
                        # If no explicit subject, inherit from main verb
                        if not conj_pairs:
                            # Try to find objects for the conj verb
                            for child in conj_verb.children:
                                if child.dep_ == "prep":
                                    for gc in child.children:
                                        if gc.dep_ in {"pobj", "obj"}:
                                            # Use subjects from main verb
                                            for subj_token, _, _ in pairs[:1]:  # Take first subject
                                                conj_pairs.append((subj_token, gc, child))
                        
                        for subj_token, obj_token, prep_token in conj_pairs:
                            subject = self._get_full_entity_text(subj_token)
                            obj = self._get_full_entity_text(obj_token)
                            prep = prep_token.text if prep_token else None
                            
                            relation, conf = self._match_relation_pattern(conj_verb.lemma_, prep)
                            if not relation and self.semantic_model:
                                relation, conf = self._match_relation_semantic(conj_verb.text, prep)
                            if not relation:
                                if prep:
                                    relation = f"{conj_verb.lemma_}_{prep}"
                                else:
                                    relation = conj_verb.lemma_
                                conf = 0.5
                            
                            triples.append((subject, relation, obj, conf))
            
            # Process copula constructions (X is Y)
            elif token.lemma_ == "be" and token.dep_ == "ROOT":
                subjects = [c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}]
                attrs = [c for c in token.children if c.dep_ in {"attr", "acomp"}]
                
                for subj in subjects:
                    for attr in attrs:
                        subj_text = self._get_full_entity_text(subj)
                        attr_text = self._get_full_entity_text(attr)
                        
                        # Check for special patterns
                        if attr.pos_ == "PROPN" or attr.ent_type_ == "PERSON":
                            triples.append((subj_text, "also_known_as", attr_text, 0.8))
                        elif "ceo" in attr_text.lower() or "founder" in attr_text.lower():
                            # Extract organization from "CEO of X"
                            for child in attr.children:
                                if child.dep_ == "prep" and child.text.lower() in ["of", "at"]:
                                    for gc in child.children:
                                        if gc.dep_ == "pobj":
                                            org = self._get_full_entity_text(gc)
                                            triples.append((subj_text, "works_at", org, 0.8))
                        else:
                            triples.append((subj_text, "is", attr_text, 0.6))
        
        return triples
    
    def _extract_noun_phrases_relations(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract relations from noun phrase patterns."""
        triples = []
        
        for chunk in doc.noun_chunks:
            # Pattern: "X of Y" (e.g., "CEO of Apple")
            for token in chunk:
                if token.dep_ == "prep" and token.text.lower() in ["of", "from", "at"]:
                    # Get the head noun
                    head = chunk.root
                    head_text = " ".join(t.text for t in chunk if t.i <= token.i)
                    
                    # Get the object of preposition
                    for child in token.children:
                        if child.dep_ in {"pobj", "obj"}:
                            obj_text = self._get_full_entity_text(child)
                            
                            # Map common patterns
                            head_lower = head_text.lower()
                            if any(role in head_lower for role in ["ceo", "director", "manager", "president", "founder"]):
                                triples.append((head_text, "works_at", obj_text, 0.7))
                            elif any(rel in head_lower for rel in ["brother", "sister", "son", "daughter", "father", "mother"]):
                                triples.append((head_text, "related_to", obj_text, 0.7))
                            elif "capital" in head_lower:
                                triples.append((obj_text, "capital", head_text, 0.7))
        
        return triples
    
    def _extract_clausal_relations(self, doc: Doc) -> List[Tuple[str, str, str, float]]:
        """Extract relations from clausal constructions."""
        triples = []
        
        for token in doc:
            # Adjectival clauses (e.g., "the company that Steve founded")
            if token.dep_ == "acl":
                # Get the noun being modified
                head_noun = self._get_full_entity_text(token.head)
                
                # Look for subject/object in the clause
                for child in token.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        subj = self._get_full_entity_text(child)
                        relation = token.lemma_
                        triples.append((subj, relation, head_noun, 0.7))
                    elif child.dep_ in {"dobj", "obj"}:
                        obj = self._get_full_entity_text(child)
                        relation = token.lemma_
                        # Infer subject from context if available
                        for s in token.children:
                            if s.dep_ in {"nsubj", "nsubjpass"}:
                                subj = self._get_full_entity_text(s)
                                triples.append((subj, relation, obj, 0.7))
        
        return triples
    
    def extract(self, text: str, lang: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """
        Extract semantic graph from text.
        Returns list of (subject, relation, object, confidence) tuples.
        """
        if not text:
            return []
        
        # Auto-detect language if not provided
        if not lang:
            # Simple heuristic - could be improved with langdetect
            lang = "en"
        
        # Get spaCy model
        nlp = self._get_nlp(lang)
        if not nlp:
            return []
        
        # Process text
        doc = nlp(text)
        
        # Extract using multiple strategies
        all_triples = []
        
        # Dependency-based extraction
        all_triples.extend(self._extract_from_dependencies(doc))
        
        # Noun phrase patterns
        all_triples.extend(self._extract_noun_phrases_relations(doc))
        
        # Clausal relations
        all_triples.extend(self._extract_clausal_relations(doc))
        
        # Deduplicate and keep highest confidence
        seen = {}
        for s, r, o, conf in all_triples:
            # Normalize entities
            s = s.strip()
            o = o.strip()
            key = (s.lower(), r, o.lower())
            
            if key not in seen or seen[key][3] < conf:
                seen[key] = (s, r, o, conf)
        
        return list(seen.values())
    
    async def extract_async(self, text: str, lang: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """Async wrapper for compatibility with ReLiK interface."""
        return self.extract(text, lang)


def test_extractor():
    """Test the multilingual graph extractor."""
    extractor = MultilingualGraphExtractor()
    
    test_cases = [
        ("My brother Tom lives in Portland and teaches at Reed College.", "en"),
        ("Apple was founded by Steve Jobs and Steve Wozniak in Cupertino in 1976.", "en"),
        ("Marie Curie discovered radium and polonium with her husband Pierre.", "en"),
        ("Barcelona es la capital de Cataluña.", "es"),
        ("Elon Musk ist der CEO von Tesla und SpaceX.", "de"),
        ("Paris est la capitale de la France.", "fr"),
    ]
    
    print("=== Multilingual Graph Extractor Test ===\n")
    
    for text, lang in test_cases:
        print(f"Text: {text}")
        print(f"Language: {lang}")
        
        start = time.perf_counter()
        relations = extractor.extract(text, lang)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"Time: {elapsed:.1f}ms")
        print("Relations:")
        for s, r, o, conf in relations:
            print(f"  ({s}, {r}, {o}) conf={conf:.2f}")
        print()


if __name__ == "__main__":
    test_extractor()