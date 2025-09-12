"""
Utilities for UD-based reasoning and future extensions
Kept separate so you can extend with GLiNER/GLiREL ONNX fallbacks later
"""

import re
from typing import Iterable, Set, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger


def norm(s: str) -> str:
    """Normalize string for consistent matching"""
    return re.sub(r"\s+", " ", s.strip().lower())


def sent_has_neg_root(sent) -> bool:
    """Check if sentence root has negation"""
    try:
        root = sent.root
        return any(ch.dep_ == "neg" for ch in root.children)
    except Exception:
        return False


def contains_lemma(span, lemmas: Iterable[str]) -> bool:
    """Check if span contains any of the specified lemmas"""
    lemma_set = set(lemmas)
    try:
        for token in span:
            if token.lemma_.lower() in lemma_set:
                return True
    except Exception:
        pass
    return False


def eid(tok) -> str:
    """Get entity ID from token"""
    if tok:
        return norm(tok.text)
    return ""


def find_child_with_dep(head, dep_labels: Set[str]):
    """Find child token with specific dependency label(s)"""
    if isinstance(dep_labels, str):
        dep_labels = {dep_labels}
    
    for child in head.children:
        if child.dep_ in dep_labels:
            return child
    return None


def get_token_span(token, window: int = 3) -> str:
    """Get text span around token for context"""
    try:
        start = max(0, token.i - window)
        end = min(len(token.doc), token.i + window + 1)
        return " ".join([t.text for t in token.doc[start:end]])
    except:
        return token.text if token else ""


@dataclass
class ExtractedRelation:
    """Structured representation of extracted relation"""
    subject: str
    relation: str
    object: str
    confidence: float = 0.5
    source_text: str = ""
    dependency_path: str = ""


class UDPatternMatcher:
    """
    Advanced UD pattern matching for future extensions
    Can be extended with GLiNER/GLiREL integration
    """
    
    def __init__(self):
        self.patterns = []
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize all 27 UD patterns for relation extraction"""
        
        # Core grammatical relations
        self.patterns.extend([
            {
                'name': 'nominal_subject',
                'deps': [('nsubj',)],
                'extract': self._extract_nsubj
            },
            {
                'name': 'passive_subject', 
                'deps': [('nsubjpass',)],
                'extract': self._extract_nsubjpass
            },
            {
                'name': 'direct_object',
                'deps': [('dobj',)],
                'extract': self._extract_dobj
            },
            {
                'name': 'indirect_object',
                'deps': [('iobj',)],
                'extract': self._extract_iobj
            },
            {
                'name': 'object',
                'deps': [('obj',)],
                'extract': self._extract_obj
            }
        ])
        
        # Clausal relations
        self.patterns.extend([
            {
                'name': 'adnominal_clause',
                'deps': [('acl',)],
                'extract': self._extract_acl
            },
            {
                'name': 'adverbial_clause',
                'deps': [('advcl',)],
                'extract': self._extract_advcl
            },
            {
                'name': 'clausal_complement',
                'deps': [('ccomp',)],
                'extract': self._extract_ccomp
            },
            {
                'name': 'clausal_subject',
                'deps': [('csubj',)],
                'extract': self._extract_csubj
            },
            {
                'name': 'open_clausal_comp',
                'deps': [('xcomp',)],
                'extract': self._extract_xcomp
            }
        ])
        
        # Modifier relations
        self.patterns.extend([
            {
                'name': 'adverbial_modifier',
                'deps': [('advmod',)],
                'extract': self._extract_advmod
            },
            {
                'name': 'adjectival_modifier',
                'deps': [('amod',)],
                'extract': self._extract_amod
            },
            {
                'name': 'nominal_modifier',
                'deps': [('nmod',)],
                'extract': self._extract_nmod
            },
            {
                'name': 'numeric_modifier',
                'deps': [('nummod',)],
                'extract': self._extract_nummod
            }
        ])
        
        # Function word relations
        self.patterns.extend([
            {
                'name': 'auxiliary',
                'deps': [('aux',)],
                'extract': self._extract_aux
            },
            {
                'name': 'passive_auxiliary',
                'deps': [('auxpass',)],
                'extract': self._extract_auxpass
            },
            {
                'name': 'case_marker',
                'deps': [('case',)],
                'extract': self._extract_case
            },
            {
                'name': 'coordination',
                'deps': [('cc',)],
                'extract': self._extract_cc
            },
            {
                'name': 'copula',
                'deps': [('cop',)],
                'extract': self._extract_copula
            },
            {
                'name': 'determiner',
                'deps': [('det',)],
                'extract': self._extract_det
            },
            {
                'name': 'marker',
                'deps': [('mark',)],
                'extract': self._extract_mark
            },
            {
                'name': 'negation',
                'deps': [('neg',)],
                'extract': self._extract_neg
            }
        ])
        
        # Special relations
        self.patterns.extend([
            {
                'name': 'agent',
                'deps': [('agent',)],
                'extract': self._extract_agent
            },
            {
                'name': 'attribute',
                'deps': [('attr',)],
                'extract': self._extract_attr
            },
            {
                'name': 'compound',
                'deps': [('compound',)],
                'extract': self._extract_compound
            },
            {
                'name': 'conjunction',
                'deps': [('conj',)],
                'extract': self._extract_conj
            },
            {
                'name': 'possessive',
                'deps': [('poss',)],
                'extract': self._extract_possession
            },
            {
                'name': 'prepositional_object',
                'deps': [('pobj',)],
                'extract': self._extract_pobj
            },
            {
                'name': 'preposition',
                'deps': [('prep',)],
                'extract': self._extract_prep
            },
            {
                'name': 'object_predicate',
                'deps': [('oprd',)],
                'extract': self._extract_oprd
            },
            {
                'name': 'root',
                'deps': [('root',)],
                'extract': self._extract_root
            }
        ])
    
    def match(self, doc) -> List[ExtractedRelation]:
        """Match all patterns against document"""
        relations = []
        
        for sent in doc.sents:
            for pattern in self.patterns:
                try:
                    extracted = pattern['extract'](sent)
                    relations.extend(extracted)
                except Exception as e:
                    logger.debug(f"Pattern {pattern['name']} failed: {e}")
        
        return relations
    
    def _extract_copula(self, sent) -> List[ExtractedRelation]:
        """Extract copula-based relations"""
        relations = []
        root = sent.root
        
        # Check for copula construction
        if any(ch.dep_ == "cop" for ch in root.children):
            subj = find_child_with_dep(root, {"nsubj", "nsubj:pass"})
            attr = find_child_with_dep(root, {"attr", "acomp", "amod"})
            
            if subj and attr:
                # Determine relation type
                if contains_lemma(sent, {"name", "nome", "nombre", "nom"}):
                    rel_type = "name"
                    confidence = 0.9
                else:
                    rel_type = "attr_of"
                    confidence = 0.7
                
                relations.append(ExtractedRelation(
                    subject=eid(subj),
                    relation=rel_type,
                    object=eid(attr),
                    confidence=confidence,
                    source_text=sent.text,
                    dependency_path=f"{subj.dep_}-{root.dep_}-{attr.dep_}"
                ))
        
        return relations
    
    def _extract_svo(self, sent) -> List[ExtractedRelation]:
        """Extract subject-verb-object relations"""
        relations = []
        root = sent.root
        
        if root.pos_ == "VERB":
            subj = find_child_with_dep(root, {"nsubj", "nsubj:pass"})
            obj = find_child_with_dep(root, {"obj", "dobj", "iobj"})
            
            if subj and obj:
                relations.append(ExtractedRelation(
                    subject=eid(subj),
                    relation=f"v:{root.lemma_}",
                    object=eid(obj),
                    confidence=0.75,
                    source_text=sent.text,
                    dependency_path=f"{subj.dep_}-{root.pos_}-{obj.dep_}"
                ))
        
        return relations
    
    def _extract_possession(self, sent) -> List[ExtractedRelation]:
        """Extract possessive relations"""
        relations = []
        
        for token in sent:
            if token.dep_ in {"nmod:poss", "poss"}:
                # Look for possessed item
                head = token.head
                
                # Determine possessor
                if token.text.lower() in {"my", "mine"}:
                    possessor = "you"
                else:
                    possessor = eid(token)
                
                if possessor and head:
                    relations.append(ExtractedRelation(
                        subject=possessor,
                        relation="has",
                        object=eid(head),
                        confidence=0.8,
                        source_text=get_token_span(token),
                        dependency_path=f"poss-{head.pos_}"
                    ))
        
        return relations
    
    # Core grammatical relations extraction methods
    def _extract_nsubj(self, sent) -> List[ExtractedRelation]:
        """Extract nominal subject relations"""
        relations = []
        for token in sent:
            if token.dep_ == "nsubj":
                relations.append(ExtractedRelation(
                    subject=eid(token),
                    relation="subject_of",
                    object=eid(token.head),
                    confidence=0.9,
                    source_text=sent.text,
                    dependency_path=f"{token.dep_}-{token.head.dep_}"
                ))
        return relations
    
    def _extract_nsubjpass(self, sent) -> List[ExtractedRelation]:
        """Extract passive subject relations"""
        relations = []
        for token in sent:
            if token.dep_ == "nsubjpass":
                relations.append(ExtractedRelation(
                    subject=eid(token),
                    relation="passive_subject_of",
                    object=eid(token.head),
                    confidence=0.9,
                    source_text=sent.text,
                    dependency_path=f"{token.dep_}-{token.head.dep_}"
                ))
        return relations
    
    def _extract_dobj(self, sent) -> List[ExtractedRelation]:
        """Extract direct object relations"""
        relations = []
        for token in sent:
            if token.dep_ == "dobj":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation=f"verb:{token.head.lemma_}",
                    object=eid(token),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_iobj(self, sent) -> List[ExtractedRelation]:
        """Extract indirect object relations"""
        relations = []
        for token in sent:
            if token.dep_ == "iobj":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation=f"indirect_object_of:{token.head.lemma_}",
                    object=eid(token),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_obj(self, sent) -> List[ExtractedRelation]:
        """Extract object relations (unified)"""
        relations = []
        for token in sent:
            if token.dep_ == "obj":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation=f"object_of:{token.head.lemma_}",
                    object=eid(token),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    # Clausal relations extraction methods
    def _extract_acl(self, sent) -> List[ExtractedRelation]:
        """Extract adnominal clause relations"""
        relations = []
        for token in sent:
            if token.dep_ == "acl":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="modified_by",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_advcl(self, sent) -> List[ExtractedRelation]:
        """Extract adverbial clause relations"""
        relations = []
        for token in sent:
            if token.dep_ == "advcl":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="adverbially_modified_by",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_ccomp(self, sent) -> List[ExtractedRelation]:
        """Extract clausal complement relations"""
        relations = []
        for token in sent:
            if token.dep_ == "ccomp":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="complemented_by",
                    object=eid(token),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_csubj(self, sent) -> List[ExtractedRelation]:
        """Extract clausal subject relations"""
        relations = []
        for token in sent:
            if token.dep_ == "csubj":
                relations.append(ExtractedRelation(
                    subject=eid(token),
                    relation="clausal_subject_of",
                    object=eid(token.head),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.dep_}-{token.head.dep_}"
                ))
        return relations
    
    def _extract_xcomp(self, sent) -> List[ExtractedRelation]:
        """Extract open clausal complement relations"""
        relations = []
        for token in sent:
            if token.dep_ == "xcomp":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="open_complement",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    # Modifier relations extraction methods
    def _extract_advmod(self, sent) -> List[ExtractedRelation]:
        """Extract adverbial modifier relations"""
        relations = []
        for token in sent:
            if token.dep_ == "advmod":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="adverbially_modified_by",
                    object=eid(token),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_amod(self, sent) -> List[ExtractedRelation]:
        """Extract adjectival modifier relations"""
        relations = []
        for token in sent:
            if token.dep_ == "amod":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="has_property",
                    object=eid(token),
                    confidence=0.9,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_nmod(self, sent) -> List[ExtractedRelation]:
        """Extract nominal modifier relations"""
        relations = []
        for token in sent:
            if token.dep_ == "nmod":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="nominally_modified_by",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_nummod(self, sent) -> List[ExtractedRelation]:
        """Extract numeric modifier relations"""
        relations = []
        for token in sent:
            if token.dep_ == "nummod":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="has_quantity",
                    object=eid(token),
                    confidence=0.9,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    # Function word relations extraction methods
    def _extract_aux(self, sent) -> List[ExtractedRelation]:
        """Extract auxiliary relations"""
        relations = []
        for token in sent:
            if token.dep_ == "aux":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="auxiliary_verb",
                    object=eid(token),
                    confidence=0.6,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_auxpass(self, sent) -> List[ExtractedRelation]:
        """Extract passive auxiliary relations"""
        relations = []
        for token in sent:
            if token.dep_ == "auxpass":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="passive_auxiliary",
                    object=eid(token),
                    confidence=0.6,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_case(self, sent) -> List[ExtractedRelation]:
        """Extract case marker relations"""
        relations = []
        for token in sent:
            if token.dep_ == "case":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="case",
                    object=eid(token),
                    confidence=0.5,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_cc(self, sent) -> List[ExtractedRelation]:
        """Extract coordination relations"""
        relations = []
        for token in sent:
            if token.dep_ == "cc":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="coordinated_with",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_det(self, sent) -> List[ExtractedRelation]:
        """Extract determiner relations"""
        relations = []
        for token in sent:
            if token.dep_ == "det":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="determined_by",
                    object=eid(token),
                    confidence=0.6,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_mark(self, sent) -> List[ExtractedRelation]:
        """Extract marker relations"""
        relations = []
        for token in sent:
            if token.dep_ == "mark":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="marked_by",
                    object=eid(token),
                    confidence=0.6,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_neg(self, sent) -> List[ExtractedRelation]:
        """Extract negation relations"""
        relations = []
        for token in sent:
            if token.dep_ == "neg":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="negated",
                    object=eid(token),
                    confidence=0.9,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    # Special relations extraction methods
    def _extract_agent(self, sent) -> List[ExtractedRelation]:
        """Extract agent relations"""
        relations = []
        for token in sent:
            if token.dep_ == "agent":
                relations.append(ExtractedRelation(
                    subject=eid(token),
                    relation="agent_of",
                    object=eid(token.head),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.dep_}-{token.head.dep_}"
                ))
        return relations
    
    def _extract_attr(self, sent) -> List[ExtractedRelation]:
        """Extract attribute relations"""
        relations = []
        for token in sent:
            if token.dep_ == "attr":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="has_attribute",
                    object=eid(token),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_compound(self, sent) -> List[ExtractedRelation]:
        """Extract compound relations"""
        relations = []
        for token in sent:
            if token.dep_ == "compound":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="compound_with",
                    object=eid(token),
                    confidence=0.9,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_conj(self, sent) -> List[ExtractedRelation]:
        """Extract conjunction relations"""
        relations = []
        for token in sent:
            if token.dep_ == "conj":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="conjugated_with",
                    object=eid(token),
                    confidence=0.8,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_pobj(self, sent) -> List[ExtractedRelation]:
        """Extract prepositional object relations"""
        relations = []
        for token in sent:
            if token.dep_ == "pobj":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="prepositional_object_of",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_prep(self, sent) -> List[ExtractedRelation]:
        """Extract preposition relations"""
        relations = []
        for token in sent:
            if token.dep_ == "prep":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation=f"preposition:{token.text}",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_oprd(self, sent) -> List[ExtractedRelation]:
        """Extract object predicate relations"""
        relations = []
        for token in sent:
            if token.dep_ == "oprd":
                relations.append(ExtractedRelation(
                    subject=eid(token.head),
                    relation="object_predicate",
                    object=eid(token),
                    confidence=0.7,
                    source_text=sent.text,
                    dependency_path=f"{token.head.dep_}-{token.dep_}"
                ))
        return relations
    
    def _extract_root(self, sent) -> List[ExtractedRelation]:
        """Extract root relations"""
        relations = []
        for token in sent:
            if token.dep_ == "root":
                relations.append(ExtractedRelation(
                    subject=eid(token),
                    relation="root",
                    object="sentence",
                    confidence=1.0,
                    source_text=sent.text,
                    dependency_path=f"{token.dep_}"
                ))
        return relations


# Future: GLiNER integration placeholder
class GLiNEREntityExtractor:
    """
    Placeholder for future GLiNER ONNX integration
    Would provide zero-shot NER when UD extraction is uncertain
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        
        if model_path:
            logger.info(f"GLiNER integration prepared (model: {model_path})")
    
    def extract(self, text: str, entity_types: List[str] = None) -> List[Tuple[str, str, float]]:
        """
        Extract entities with GLiNER
        Returns: List of (entity_text, entity_type, confidence)
        """
        # Placeholder for ONNX inference
        logger.debug("GLiNER extraction not yet implemented")
        return []


# Future: GLiREL integration placeholder  
class GLiRELRelationExtractor:
    """
    Placeholder for future GLiREL ONNX integration
    Would provide zero-shot relation extraction when UD patterns are ambiguous
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        
        if model_path:
            logger.info(f"GLiREL integration prepared (model: {model_path})")
    
    def extract(self, text: str, entity_pairs: List[Tuple[str, str]]) -> List[ExtractedRelation]:
        """
        Extract relations between entity pairs
        Returns: List of ExtractedRelation objects
        """
        # Placeholder for ONNX inference
        logger.debug("GLiREL extraction not yet implemented")
        return []


def extract_all_ud_patterns(text: str, nlp=None) -> List[ExtractedRelation]:
    """
    Convenience function to extract all UD patterns from text
    Returns: List of ExtractedRelation objects
    """
    import spacy
    
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    matcher = UDPatternMatcher()
    return matcher.match(doc)


# Export convenience functions
__all__ = [
    'norm',
    'eid',
    'sent_has_neg_root',
    'contains_lemma',
    'find_child_with_dep',
    'get_token_span',
    'ExtractedRelation',
    'UDPatternMatcher',
    'extract_all_ud_patterns',
    'GLiNEREntityExtractor',
    'GLiRELRelationExtractor'
]