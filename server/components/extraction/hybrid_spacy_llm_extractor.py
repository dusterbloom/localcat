#!/usr/bin/env python3
"""
Hybrid spaCy + LLM Relation Extractor for HotMem

Combines the best of both worlds:
1. Fast, accurate spaCy dependency parsing for simple sentences
2. LLM fallback for complex, multi-clause sentences

Based on successful production approach from macos-local-voice-agents project.
"""

import os
import json
import time
import re
import spacy
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from loguru import logger
import asyncio
import aiohttp


@dataclass
class ComplexityScore:
    """Sentence complexity assessment"""
    score: float
    clause_count: int
    entity_count: int
    conjunction_count: int
    depth: int
    length: int


class HybridRelationExtractor:
    """
    Production-grade hybrid extractor combining spaCy and LLM.
    Drop-in replacement for ReLiK with much better quality.
    """
    
    # Complexity threshold for LLM fallback
    COMPLEXITY_THRESHOLD = float(os.getenv("HOTMEM_COMPLEXITY_THRESHOLD", "0.6"))
    
    # LLM prompt for structured extraction
    EXTRACTION_PROMPT = """Extract facts from this text as simple (subject, relation, object) triples.

Text: "{text}"

Rules:
- Use "user" for first-person references (I, my, me)
- Keep relations simple: has, works_at, lives_in, founded, discovered, teaches_at, etc.
- Handle conjunctions properly: "X and Y founded Z" → two separate facts
- Resolve pronouns to their antecedents
- Output JSON array of triples

Example output:
[
  ["Steve Jobs", "founded", "Apple"],
  ["Steve Wozniak", "founded", "Apple"],
  ["user", "has", "brother"],
  ["brother", "lives_in", "Portland"]
]

Output only the JSON array, no explanation:"""
    
    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        """Initialize with ReLiK-compatible interface."""
        self.model_id = model_id or "hybrid-spacy-llm"
        self.device = device
        self._ready = False
        
        # Load spaCy model
        self.nlp = self._load_spacy_model()
        
        # LLM configuration
        self.llm_enabled = os.getenv("HOTMEM_LLM_ASSISTED", "true").lower() in ("1", "true", "yes")
        self.llm_model = os.getenv("HOTMEM_LLM_ASSISTED_MODEL", "qwen2.5-coder:3b")
        self.llm_base_url = os.getenv("HOTMEM_LLM_ASSISTED_BASE_URL", 
                                      os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"))
        self.llm_timeout_ms = int(os.getenv("HOTMEM_LLM_ASSISTED_TIMEOUT_MS", "500"))
        
        logger.info(f"[Hybrid Extractor] Initialized with LLM={self.llm_enabled}, threshold={self.COMPLEXITY_THRESHOLD}")
    
    def _load_spacy_model(self) -> spacy.Language:
        """Load best available spaCy model."""
        models = ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
        for model_name in models:
            try:
                nlp = spacy.load(model_name)
                logger.info(f"[Hybrid] Loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                continue
        raise RuntimeError("No spaCy model available")
    
    def _assess_complexity(self, doc) -> ComplexityScore:
        """Assess sentence complexity to decide extraction strategy."""
        # Count clauses
        clause_count = len([t for t in doc if t.dep_ in ["ccomp", "xcomp", "advcl", "acl", "relcl"]])
        
        # Count entities
        entity_count = len(doc.ents)
        
        # Count conjunctions
        conjunction_count = len([t for t in doc if t.dep_ == "conj"])
        
        # Parse tree depth
        depths = []
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            depths.append(depth)
        max_depth = max(depths) if depths else 0
        
        # Sentence length
        length = len(doc)
        
        # Calculate complexity score (0-1)
        score = 0.0
        score += min(clause_count * 0.2, 0.4)  # Up to 0.4 for clauses
        score += min(conjunction_count * 0.15, 0.3)  # Up to 0.3 for conjunctions
        score += min(max_depth / 10, 0.2)  # Up to 0.2 for depth
        score += min(length / 50, 0.1)  # Up to 0.1 for length
        
        return ComplexityScore(
            score=min(score, 1.0),
            clause_count=clause_count,
            entity_count=entity_count,
            conjunction_count=conjunction_count,
            depth=max_depth,
            length=length
        )
    
    def _extract_spacy_simple(self, doc) -> List[Tuple[str, str, str, float]]:
        """Fast spaCy extraction for simple sentences."""
        triples = []
        
        for token in doc:
            # Handle main verbs
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = self._get_full_np(child)
                        break
                
                if not subject:
                    continue
                
                # Normalize first-person to "user"
                subject = self._normalize_subject(subject)
                
                # Handle passive voice
                if token.tag_ in ["VBN", "VBD"] and any(c.dep_ == "auxpass" for c in token.children):
                    # Look for agent
                    for child in token.children:
                        if child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by"):
                            for gc in child.children:
                                if gc.dep_ in ["pobj", "obj"]:
                                    agent = self._get_full_np(gc)
                                    # Handle conjunctions in agent
                                    agents = [agent]
                                    for conj in gc.children:
                                        if conj.dep_ == "conj":
                                            agents.append(self._get_full_np(conj))
                                    
                                    obj = subject  # In passive, subject is the object
                                    for a in agents:
                                        triples.append((self._normalize_subject(a), token.lemma_, obj, 0.8))
                
                # Handle active voice
                else:
                    # Direct objects
                    for child in token.children:
                        if child.dep_ in ["dobj", "obj"]:
                            obj = self._get_full_np(child)
                            relation = self._normalize_relation(token.lemma_)
                            triples.append((subject, relation, obj, 0.8))
                    
                    # Prepositional objects
                    for child in token.children:
                        if child.dep_ == "prep":
                            prep = child.text.lower()
                            for gc in child.children:
                                if gc.dep_ in ["pobj", "obj"]:
                                    obj = self._get_full_np(gc)
                                    relation = self._map_verb_prep(token.lemma_, prep)
                                    if relation:
                                        triples.append((subject, relation, obj, 0.85))
                
                # Handle coordinated verbs
                for conj in token.children:
                    if conj.dep_ == "conj" and conj.pos_ == "VERB":
                        # Inherit subject if not explicit
                        conj_subj = subject
                        for child in conj.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                conj_subj = self._normalize_subject(self._get_full_np(child))
                                break
                        
                        # Extract from conjunction
                        for child in conj.children:
                            if child.dep_ in ["dobj", "obj"]:
                                obj = self._get_full_np(child)
                                relation = self._normalize_relation(conj.lemma_)
                                triples.append((conj_subj, relation, obj, 0.8))
                            elif child.dep_ == "prep":
                                prep = child.text.lower()
                                for gc in child.children:
                                    if gc.dep_ in ["pobj", "obj"]:
                                        obj = self._get_full_np(gc)
                                        relation = self._map_verb_prep(conj.lemma_, prep)
                                        if relation:
                                            triples.append((conj_subj, relation, obj, 0.85))
            
            # Handle copula (X is Y)
            elif token.lemma_ == "be" and token.dep_ == "ROOT":
                subject = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = self._normalize_subject(self._get_full_np(child))
                        break
                
                if subject:
                    # Attributes
                    for child in token.children:
                        if child.dep_ in ["attr", "acomp"]:
                            attr = self._get_full_np(child)
                            # Check for special patterns
                            if "ceo" in attr.lower() or "founder" in attr.lower():
                                # Extract organization
                                for gc in child.children:
                                    if gc.dep_ == "prep" and gc.text.lower() in ["of", "at"]:
                                        for ggc in gc.children:
                                            if ggc.dep_ in ["pobj", "obj"]:
                                                org = self._get_full_np(ggc)
                                                triples.append((subject, "works_at", org, 0.75))
                            else:
                                triples.append((subject, "is", attr, 0.7))
        
        return triples
    
    async def _extract_llm_complex(self, text: str) -> List[Tuple[str, str, str, float]]:
        """LLM extraction for complex sentences."""
        if not self.llm_enabled:
            return []
        
        prompt = self.EXTRACTION_PROMPT.format(text=text)
        
        try:
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}"
            }
            
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 256,
                "response_format": {"type": "json_object"} if "gpt" in self.llm_model else None
            }
            
            # Make async request
            timeout = aiohttp.ClientTimeout(total=self.llm_timeout_ms / 1000.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.llm_base_url}/chat/completions"
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "[]")
                        
                        # Parse JSON response
                        content = content.strip()
                        # Remove code fences if present
                        content = re.sub(r"^```[a-zA-Z]*\n|```$", "", content, flags=re.MULTILINE)
                        
                        try:
                            result = json.loads(content)
                            if isinstance(result, list):
                                # Convert to tuples with confidence
                                triples = []
                                for item in result:
                                    if isinstance(item, list) and len(item) >= 3:
                                        s, r, o = item[0], item[1], item[2]
                                        conf = item[3] if len(item) > 3 else 0.9
                                        triples.append((s, r, o, conf))
                                return triples
                        except json.JSONDecodeError:
                            logger.debug(f"[Hybrid] Failed to parse LLM JSON: {content[:100]}")
                            
        except asyncio.TimeoutError:
            logger.debug(f"[Hybrid] LLM timeout after {self.llm_timeout_ms}ms")
        except Exception as e:
            logger.debug(f"[Hybrid] LLM extraction failed: {e}")
        
        return []
    
    def extract(self, text: str, lang: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """
        Extract relations using hybrid approach.
        Compatible with ReLiK interface.
        """
        if not text or len(text) > 480:  # Respect text limit
            return []
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Assess complexity
        complexity = self._assess_complexity(doc)
        logger.debug(f"[Hybrid] Complexity: {complexity.score:.2f} (clauses={complexity.clause_count}, conj={complexity.conjunction_count})")
        
        # Choose extraction strategy
        if complexity.score < self.COMPLEXITY_THRESHOLD:
            # Simple sentence - use fast spaCy
            triples = self._extract_spacy_simple(doc)
            logger.debug(f"[Hybrid] Using spaCy for simple sentence, extracted {len(triples)} triples")
        else:
            # Complex sentence - use LLM
            logger.debug(f"[Hybrid] Using LLM for complex sentence")
            # Run async LLM extraction in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                triples = loop.run_until_complete(self._extract_llm_complex(text))
                logger.debug(f"[Hybrid] LLM extracted {len(triples)} triples")
            finally:
                loop.close()
            
            # Fallback to spaCy if LLM fails
            if not triples:
                triples = self._extract_spacy_simple(doc)
                logger.debug(f"[Hybrid] Fallback to spaCy, extracted {len(triples)} triples")
        
        # Post-process and deduplicate
        seen = set()
        unique = []
        for s, r, o, conf in triples:
            key = (s.lower().strip(), r.lower(), o.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append((s, r, o, conf))
        
        return unique
    
    def _get_full_np(self, token) -> str:
        """Get full noun phrase including modifiers."""
        # Try noun chunks first
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        
        # Fallback to subtree
        subtree = sorted(token.subtree, key=lambda t: t.i)
        return " ".join(t.text for t in subtree)
    
    def _normalize_subject(self, subject: str) -> str:
        """Normalize first-person pronouns to 'user'."""
        if subject.lower() in ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours"]:
            return "user"
        return subject
    
    def _normalize_relation(self, verb: str) -> str:
        """Normalize verb to standard relation."""
        verb = verb.lower()
        
        # Direct mappings
        mappings = {
            "found": "founded",
            "establish": "founded",
            "create": "founded",
            "start": "founded",
            "build": "founded",
            "invent": "invented",
            "discover": "discovered",
            "develop": "developed",
            "own": "owns",
            "possess": "owns",
            "have": "has",
            "teach": "teaches",
            "study": "studied",
            "graduate": "graduated",
            "marry": "married_to",
            "know": "knows",
        }
        
        return mappings.get(verb, verb)
    
    def _map_verb_prep(self, verb: str, prep: str) -> Optional[str]:
        """Map verb + preposition to relation."""
        verb = verb.lower()
        prep = prep.lower()
        
        patterns = {
            ("live", "in"): "lives_in",
            ("work", "at"): "works_at",
            ("work", "for"): "works_at",
            ("teach", "at"): "teaches_at",
            ("study", "at"): "studied_at",
            ("graduate", "from"): "graduated_from",
            ("born", "in"): "born_in",
            ("move", "from"): "moved_from",
            ("move", "to"): "moved_to",
            ("come", "from"): "from",
            ("go", "to"): "went_to",
        }
        
        return patterns.get((verb, prep), f"{verb}_{prep}")
    
    async def extract_async(self, text: str, lang: Optional[str] = None) -> List[Tuple[str, str, str, float]]:
        """Async wrapper for compatibility."""
        return self.extract(text, lang)


def test_hybrid_extractor():
    """Test the hybrid extractor with various sentences."""
    extractor = HybridRelationExtractor()
    
    test_cases = [
        # Simple sentences (should use spaCy)
        "I work at Google.",
        "My dog's name is Luna.",
        
        # Complex sentences (should use LLM)
        "My brother Tom, who graduated from MIT last year, now works at Apple as a senior engineer and lives in Cupertino.",
        "Marie Curie discovered radium and polonium with her husband Pierre, for which they won the Nobel Prize in Physics.",
        "Apple was founded by Steve Jobs and Steve Wozniak in a garage in Los Altos, California, on April 1, 1976.",
        
        # Medium complexity
        "I have a cat named Whiskers and a dog named Buddy.",
        "Barcelona, which is the capital of Catalonia, is the second largest city in Spain.",
    ]
    
    print("=== Hybrid Relation Extractor Test ===\n")
    
    for text in test_cases:
        print(f"Text: {text}")
        
        start = time.perf_counter()
        relations = extractor.extract(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"Time: {elapsed:.1f}ms")
        print("Relations:")
        for s, r, o, conf in relations:
            print(f"  ({s}, {r}, {o}) conf={conf:.2f}")
        print()


if __name__ == "__main__":
    test_hybrid_extractor()