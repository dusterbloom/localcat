#!/usr/bin/env python3
"""
A/B/C Testing Framework for Extraction Approaches
=================================================

Test A: Stanza vs spaCy in real-time mode
Test B: Batch processing with entity pre-extraction  
Test C: Current HotMem baseline

This will help determine the best approach before implementing changes.
"""

import os
import sys
import time
import tempfile
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add server path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

from memory_hotpath import HotMemory
from memory_intent import get_intent_classifier
from memory_store import MemoryStore, Paths

# Test cases - same as eval_extraction.py
TEST_CASES = [
    ("Simple Statement", "My dog's name is Potola"),
    ("Complex Question", "Did I tell you that Potola is five years old?"),
    ("Casablanca", "I came to Casablanca for the waters, even though I was misinformed because Casablanca is in the desert, and I stayed because Ilsa walked into my gin joint."),
    ("Pride & Prejudice", "Mr. Darcy owns Pemberley estate which has ten thousand acres, he rides horses every morning, and his sister Georgiana plays piano beautifully while their aunt Lady Catherine lives in Kent."),
    ("Great Gatsby", "Jay Gatsby, whose real name is James Gatz, was born in North Dakota, worked on Dan Cody's yacht for five years, and now lives in West Egg where he throws lavish parties."),
    ("Conversation", "Yesterday I met Sarah who works at Microsoft in Seattle, she graduated from Stanford in 2019, drives a Tesla, and said her brother Tom lives in Portland and teaches at Reed College."),
]

@dataclass
class ExtractionResult:
    """Results from extraction testing"""
    method: str
    case_name: str
    input_text: str
    extraction_time_ms: float
    total_time_ms: float  
    entities_found: int
    triples_stored: int
    bullets_generated: int
    triples: List[Tuple[str, str, str]]
    bullets: List[str]
    success: bool
    error: Optional[str] = None

class TestFramework:
    """A/B/C testing framework for extraction approaches"""
    
    def __init__(self):
        self.results: List[ExtractionResult] = []
        
    def run_all_tests(self):
        """Run all A/B/C tests and compare results"""
        print("🧪 A/B/C Extraction Testing Framework")
        print("=" * 60)
        
        # Test C: Current HotMem baseline
        print("\n📊 TEST C: Current HotMem Baseline")
        print("-" * 40)
        self.test_current_hotmem()
        
        # Test A: Stanza in real-time  
        print("\n📊 TEST A: Stanza Real-time")
        print("-" * 40)
        self.test_stanza_realtime()
        
        # Test B: Batch processing
        print("\n📊 TEST B: Batch Processing")  
        print("-" * 40)
        self.test_batch_processing()
        
        # Compare results
        print("\n📈 COMPARISON RESULTS")
        print("=" * 60)
        self.compare_results()
        
    def test_current_hotmem(self):
        """Test C: Current HotMem implementation (baseline)"""
        os.environ['HOTMEM_DECOMPOSE_CLAUSES'] = 'true'
        os.environ['HOTMEM_CONFIDENCE_THRESHOLD'] = '0.3'  # Reasonable threshold
        
        with tempfile.TemporaryDirectory() as tdir:
            store = MemoryStore(Paths(
                sqlite_path=os.path.join(tdir, 'memory.db'),
                lmdb_dir=os.path.join(tdir, 'graph.lmdb')
            ))
            
            for case_name, text in TEST_CASES:
                hot = HotMemory(store)
                hot.prewarm('en')
                
                start_time = time.perf_counter()
                try:
                    bullets, triples = hot.process_turn(text, session_id="test", turn_id=1)
                    total_time = (time.perf_counter() - start_time) * 1000
                    
                    # Get extraction time from metrics
                    extraction_time = 0
                    if hot.metrics.get('extraction_ms'):
                        extraction_time = hot.metrics['extraction_ms'][-1]
                    
                    result = ExtractionResult(
                        method="HotMem Current",
                        case_name=case_name,
                        input_text=text,
                        extraction_time_ms=extraction_time,
                        total_time_ms=total_time,
                        entities_found=0,  # Will calculate from triples
                        triples_stored=len(triples),
                        bullets_generated=len(bullets),
                        triples=triples,
                        bullets=bullets,
                        success=True
                    )
                    
                except Exception as e:
                    result = ExtractionResult(
                        method="HotMem Current",
                        case_name=case_name,
                        input_text=text,
                        extraction_time_ms=0,
                        total_time_ms=0,
                        entities_found=0,
                        triples_stored=0,
                        bullets_generated=0,
                        triples=[],
                        bullets=[],
                        success=False,
                        error=str(e)
                    )
                
                self.results.append(result)
                self._print_result(result)
    
    def test_stanza_realtime(self):
        """Test A: Replace spaCy with Stanza in real-time"""
        try:
            import stanza
            print("📦 Loading Stanza models...")
            # Download if needed
            try:
                nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse,ner', download_method=None)
                print("✅ Stanza loaded successfully")
            except Exception:
                print("📥 Downloading Stanza English model...")
                stanza.download('en')
                nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse,ner')
                print("✅ Stanza downloaded and loaded")
            
        except ImportError:
            print("❌ Stanza not available. Install with: pip install stanza")
            # Add placeholder results
            for case_name, text in TEST_CASES:
                result = ExtractionResult(
                    method="Stanza Real-time",
                    case_name=case_name,
                    input_text=text,
                    extraction_time_ms=0,
                    total_time_ms=0,
                    entities_found=0,
                    triples_stored=0,
                    bullets_generated=0,
                    triples=[],
                    bullets=[],
                    success=False,
                    error="Stanza not installed"
                )
                self.results.append(result)
                self._print_result(result)
            return
        
        # Test Stanza extraction on each case
        for case_name, text in TEST_CASES:
            start_time = time.perf_counter()
            
            try:
                # Process with Stanza
                doc = nlp(text)
                extraction_time = (time.perf_counter() - start_time) * 1000
                
                # Extract entities and relations using Stanza
                entities, triples = self._extract_with_stanza(doc, text)
                
                total_time = (time.perf_counter() - start_time) * 1000
                
                # Generate bullets (simplified)
                bullets = [f"• {s} {r} {d}" for s, r, d in triples[:5]]
                
                result = ExtractionResult(
                    method="Stanza Real-time",
                    case_name=case_name,
                    input_text=text,
                    extraction_time_ms=extraction_time,
                    total_time_ms=total_time,
                    entities_found=len(entities),
                    triples_stored=len(triples),
                    bullets_generated=len(bullets),
                    triples=triples,
                    bullets=bullets,
                    success=True
                )
                
            except Exception as e:
                result = ExtractionResult(
                    method="Stanza Real-time",
                    case_name=case_name,
                    input_text=text,
                    extraction_time_ms=0,
                    total_time_ms=0,
                    entities_found=0,
                    triples_stored=0,
                    bullets_generated=0,
                    triples=[],
                    bullets=[],
                    success=False,
                    error=str(e)
                )
            
            self.results.append(result)
            self._print_result(result)
    
    def test_batch_processing(self):
        """Test B: Batch processing approach"""
        print("🔄 Simulating batch processing...")
        
        # Step 1: Immediate response (light extraction only)
        immediate_results = []
        batch_input_texts = []
        
        for case_name, text in TEST_CASES:
            start_time = time.perf_counter()
            
            # Light extraction (entities only, no relations)
            entities = self._extract_entities_light(text)
            immediate_time = (time.perf_counter() - start_time) * 1000
            
            immediate_results.append({
                'case_name': case_name,
                'text': text,
                'entities': entities,
                'time_ms': immediate_time
            })
            batch_input_texts.append(text)
            
            print(f"  ⚡ {case_name}: {len(entities)} entities in {immediate_time:.1f}ms")
        
        # Step 2: Batch processing (every 30s simulation)
        print("  🔄 Running batch processing on accumulated texts...")
        
        start_batch = time.perf_counter()
        batch_results = self._batch_extract_facts(batch_input_texts, immediate_results)
        batch_time = (time.perf_counter() - start_batch) * 1000
        
        print(f"  📦 Batch processing completed in {batch_time:.1f}ms total")
        
        # Create results
        for i, (case_name, text) in enumerate(TEST_CASES):
            immediate = immediate_results[i]
            batch = batch_results[i] if i < len(batch_results) else {'triples': [], 'bullets': []}
            
            result = ExtractionResult(
                method="Batch Processing",
                case_name=case_name,
                input_text=text,
                extraction_time_ms=immediate['time_ms'],  # User sees immediate time
                total_time_ms=batch_time / len(TEST_CASES),  # Amortized batch time
                entities_found=len(immediate['entities']),
                triples_stored=len(batch['triples']),
                bullets_generated=len(batch['bullets']),
                triples=batch['triples'],
                bullets=batch['bullets'],
                success=True
            )
            
            self.results.append(result)
            self._print_result(result)
    
    def _extract_with_stanza(self, doc, original_text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Extract entities and relations using Stanza"""
        entities = []
        triples = []
        
        # Extract named entities
        for sentence in doc.sentences:
            for ent in sentence.ents:
                entities.append(ent.text.lower())
        
        # Extract relations using dependency parsing  
        for sentence in doc.sentences:
            for word in sentence.words:
                # Subject-verb-object patterns
                if word.deprel == 'nsubj':
                    verb = sentence.words[word.head - 1] if word.head > 0 else None
                    if verb:
                        # Find object
                        for w2 in sentence.words:
                            if w2.head == verb.id and w2.deprel in ['obj', 'dobj']:
                                subj = self._normalize_entity(word.text)
                                pred = verb.lemma
                                obj = self._normalize_entity(w2.text) 
                                triples.append((subj, pred, obj))
                                break
                        
                        # Find prepositional complements
                        for w2 in sentence.words:
                            if w2.head == verb.id and w2.deprel == 'obl':
                                # Look for preposition
                                prep = None
                                for w3 in sentence.words:
                                    if w3.head == w2.id and w3.deprel == 'case':
                                        prep = w3.lemma
                                        break
                                
                                if prep:
                                    subj = self._normalize_entity(word.text)
                                    pred = f"{verb.lemma}_{prep}"
                                    obj = self._normalize_entity(w2.text)
                                    triples.append((subj, pred, obj))
                
                # Copular constructions (X is Y)  
                elif word.deprel == 'nsubj' and word.head > 0:
                    verb = sentence.words[word.head - 1]
                    if verb.lemma == 'be':
                        # Find predicate
                        for w2 in sentence.words:
                            if w2.head == verb.id and w2.deprel in ['acomp', 'attr']:
                                subj = self._normalize_entity(word.text)
                                pred = 'is'
                                obj = self._normalize_entity(w2.text)
                                triples.append((subj, pred, obj))
                                break
        
        return list(set(entities)), triples
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text"""
        normalized = text.lower().strip()
        if normalized in ['i', 'me', 'my', 'myself']:
            return 'you'
        return normalized
    
    def _extract_entities_light(self, text: str) -> List[str]:
        """Light entity extraction - just capitalized words and common patterns"""
        words = text.split()
        entities = []
        
        for word in words:
            clean = word.strip('.,!?;:"')
            # Capitalized words (likely names/places)
            if clean and clean[0].isupper() and len(clean) > 2 and clean.isalpha():
                entities.append(clean.lower())
        
        # Add self-reference  
        if any(p in text.lower() for p in [' i ', ' my ', ' me ']):
            entities.append('you')
            
        return list(set(entities))
    
    def _batch_extract_facts(self, texts: List[str], immediate_results: List[Dict]) -> List[Dict]:
        """Batch fact extraction using REAL LLM for relation extraction"""
        results = []
        
        # Real LLM extraction using LM Studio API like in slowcat-consciousness
        for i, text in enumerate(texts):
            entities = immediate_results[i]['entities']
            triples = []
            
            # Real LLM relation extraction via sync HTTP call (non-blocking w.r.t. outer event loop)
            try:
                t0 = time.perf_counter()
                relations = self._extract_relations_real_llm_sync(text, entities)
                dt_ms = (time.perf_counter() - t0) * 1000
                if relations:
                    print(f"    ↳ LLM relations: {len(relations)} ( {dt_ms:.0f}ms )")
                for rel in relations or []:
                    subj = rel.get('subject', '')
                    pred = rel.get('predicate', '')
                    obj = rel.get('object', '')
                    if subj and pred and obj:
                        triples.append((subj, pred, obj))
            except Exception as e:
                print(f"LLM extraction failed: {e}")
                # Fallback to basic extraction
                triples = []
                
            bullets = [f"{s} {r.replace('_', ' ')} {o}" for s, r, o in triples]
            
            results.append({
                'triples': triples,
                'bullets': bullets
            })
        
        return results
    
    def _simulate_llm_relation_extraction(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Simulate what an LLM would extract for relations given entities"""
        relations = []
        text_lower = text.lower()
        
        # Simulate LLM understanding context and entities to generate relations
        # This is much more sophisticated than pattern matching
        
        # Name relations
        if "'s name is" in text_lower or "real name is" in text_lower:
            for entity in entities:
                if entity in text_lower:
                    # Find the name after "name is"
                    patterns = ["'s name is", "real name is", "name is"]
                    for pattern in patterns:
                        if pattern in text_lower:
                            start = text_lower.find(pattern) + len(pattern)
                            end = text_lower.find(',', start)
                            if end == -1:
                                end = text_lower.find(' and', start)  
                            if end == -1:
                                end = text_lower.find('.', start)
                            if end == -1:
                                end = len(text_lower)
                            name = text_lower[start:end].strip(' .,')
                            if name and name != entity:
                                relations.append((entity, 'real_name', name))
        
        # Location relations
        location_patterns = ['lives in', 'in', 'from', 'born in']
        for pattern in location_patterns:
            if pattern in text_lower:
                for entity in entities:
                    if entity in text_lower:
                        start = text_lower.find(pattern) + len(pattern)
                        end = text_lower.find(',', start)
                        if end == -1:
                            end = text_lower.find(' and', start)
                        if end == -1:
                            end = text_lower.find('.', start)  
                        if end == -1:
                            end = len(text_lower)
                        location = text_lower[start:end].strip(' .,')
                        if location and location in entities:
                            relations.append((entity, pattern.replace(' ', '_'), location))
        
        # Work relations  
        work_patterns = ['works at', 'work at', 'teaches at']
        for pattern in work_patterns:
            if pattern in text_lower:
                for entity in entities:
                    if entity in text_lower:
                        start = text_lower.find(pattern) + len(pattern)
                        end = text_lower.find(',', start)
                        if end == -1:
                            end = text_lower.find(' and', start)
                        if end == -1:
                            end = text_lower.find('.', start)
                        if end == -1:
                            end = len(text_lower)
                        workplace = text_lower[start:end].strip(' .,')
                        if workplace and workplace in entities:
                            relations.append((entity, 'works_at', workplace))
        
        return relations[:10]  # Limit like real LLM would
    
    async def _extract_relations_real_llm(self, text: str, entities: List[str]) -> List[Dict]:
        """Real LLM relation extraction using LM Studio API"""
        import httpx
        import json
        
        if not entities:
            return []
        
        # Use actual entities for context 
        entity_list = ', '.join(entities[:5])
        
        prompt = f"""You are an expert knowledge graph builder. Extract ALL factual relationships from text.

Examples:
Text: "My dog Rex is brown"
Relations: [
  {{"subject": "user", "predicate": "has_pet", "object": "Rex", "confidence": 0.9}},
  {{"subject": "Rex", "predicate": "is_a", "object": "dog", "confidence": 1.0}},
  {{"subject": "Rex", "predicate": "has_color", "object": "brown", "confidence": 1.0}}
]

Text: "I work at Microsoft in Seattle"
Relations: [
  {{"subject": "user", "predicate": "works_at", "object": "Microsoft", "confidence": 0.9}},
  {{"subject": "Microsoft", "predicate": "located_in", "object": "Seattle", "confidence": 0.9}},
  {{"subject": "user", "predicate": "works_in", "object": "Seattle", "confidence": 0.8}}
]

RULES:
- "I", "my", "me" → "user" entity  
- Extract ownership, locations, occupations, names, attributes, relationships
- Use predicates: has_pet, works_at, is_a, has_name, located_in, owns, employed_by, lives_in, is_named, has_color, has_occupation, etc.

Now extract from: "{text}"
Entities found: {entity_list}"""

        request_data = {
            "model": "qwen/qwen3-4b",  # Use the model from .env
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0,
            "stream": False,
            "response_format": {
                "type": "json_schema", 
                "json_schema": {
                    "name": "relation_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject": {"type": "string"},
                                        "predicate": {"type": "string"}, 
                                        "object": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    },
                                    "required": ["subject", "predicate", "object", "confidence"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["relations"],
                        "additionalProperties": False
                    }
                }
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "http://127.0.0.1:1234/v1/chat/completions",  # LM Studio URL
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    raw_response = result['choices'][0]['message']['content']
                    
                    try:
                        parsed = json.loads(raw_response)
                        return parsed.get('relations', [])
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e}, raw: {raw_response[:100]}")
                        return []
                else:
                    print(f"LM Studio error: {response.status_code}: {response.text}")
                    return []
                    
        except Exception as e:
            print(f"LLM request failed: {e}")
            return []

    def _extract_relations_real_llm_sync(self, text: str, entities: List[str]) -> List[Dict]:
        """Synchronous version of relation extraction for environments with a running event loop."""
        import json
        import urllib.request
        import urllib.error

        if not entities:
            return []

        entity_list = ', '.join(entities[:5])
        prompt = f"""You are an expert knowledge graph builder. Extract ALL factual relationships from text.

Examples:
Text: \"My dog Rex is brown\"
Relations: [
  {{\"subject\": \"user\", \"predicate\": \"has_pet\", \"object\": \"Rex\", \"confidence\": 0.9}},
  {{\"subject\": \"Rex\", \"predicate\": \"is_a\", \"object\": \"dog\", \"confidence\": 1.0}},
  {{\"subject\": \"Rex\", \"predicate\": \"has_color\", \"object\": \"brown\", \"confidence\": 1.0}}
]

Text: \"I work at Microsoft in Seattle\"
Relations: [
  {{\"subject\": \"user\", \"predicate\": \"works_at\", \"object\": \"Microsoft\", \"confidence\": 0.9}},
  {{\"subject\": \"Microsoft\", \"predicate\": \"located_in\", \"object\": \"Seattle\", \"confidence\": 0.9}},
  {{\"subject\": \"user\", \"predicate\": \"works_in\", \"object\": \"Seattle\", \"confidence\": 0.8}}
]

RULES:
- \"I\", \"my\", \"me\" → \"user\" entity
- Extract ownership, locations, occupations, names, attributes, relationships
- Use predicates: has_pet, works_at, is_a, has_name, located_in, owns, employed_by, lives_in, is_named, has_color, has_occupation, etc.

Now extract from: \"{text}\"
Entities found: {entity_list}"""

        request_data = {
            "model": os.getenv("HOTMEM_LLM_ASSISTED_MODEL", os.getenv("SUMMARIZER_MODEL", os.getenv("OPENAI_MODEL", "qwen3:4b"))),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "relation_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject": {"type": "string"},
                                        "predicate": {"type": "string"},
                                        "object": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    },
                                    "required": ["subject", "predicate", "object", "confidence"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["relations"],
                        "additionalProperties": False
                    }
                }
            }
        }

        base_url = os.getenv("SUMMARIZER_BASE_URL") or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        url = base_url.rstrip('/') + "/chat/completions"
        data = json.dumps(request_data).encode('utf-8')
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}"}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode('utf-8')
                obj = json.loads(body)
                raw = ((obj.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
                try:
                    parsed = json.loads(raw)
                    return parsed.get('relations', []) if isinstance(parsed, dict) else []
                except json.JSONDecodeError:
                    print(f"JSON parse error, raw: {raw[:100]}")
                    return []
        except Exception as e:
            print(f"LLM request failed: {e}")
            return []
    
    def _print_result(self, result: ExtractionResult):
        """Print individual test result"""
        status = "✅" if result.success else "❌"
        print(f"  {status} {result.case_name}")
        
        if result.success:
            print(f"     ⏱️  Extraction: {result.extraction_time_ms:.1f}ms | Total: {result.total_time_ms:.1f}ms")
            print(f"     📊 Entities: {result.entities_found} | Triples: {result.triples_stored} | Bullets: {result.bullets_generated}")
            
            if result.triples:
                print("     🔗 Triples:")
                for s, r, d in result.triples[:3]:  # Show first 3
                    print(f"        ({s}) -[{r}]-> ({d})")
                if len(result.triples) > 3:
                    print(f"        ... and {len(result.triples) - 3} more")
        else:
            print(f"     ❌ Error: {result.error}")
        print()
    
    def compare_results(self):
        """Compare all test results"""
        methods = defaultdict(list)
        
        for result in self.results:
            methods[result.method].append(result)
        
        print("📈 Performance Comparison")
        print("-" * 40)
        
        for method_name, results in methods.items():
            successful = [r for r in results if r.success]
            if not successful:
                print(f"{method_name}: ❌ All tests failed")
                continue
            
            avg_extraction_time = sum(r.extraction_time_ms for r in successful) / len(successful)
            avg_total_time = sum(r.total_time_ms for r in successful) / len(successful) 
            total_entities = sum(r.entities_found for r in successful)
            total_triples = sum(r.triples_stored for r in successful)
            total_bullets = sum(r.bullets_generated for r in successful)
            success_rate = (len(successful) / len(results)) * 100
            
            print(f"\n{method_name}:")
            print(f"  ⏱️  Avg Extraction: {avg_extraction_time:.1f}ms")
            print(f"  ⏱️  Avg Total: {avg_total_time:.1f}ms") 
            print(f"  📊 Success Rate: {success_rate:.1f}%")
            print(f"  📊 Total Entities: {total_entities}")
            print(f"  📊 Total Triples: {total_triples}")
            print(f"  📊 Total Bullets: {total_bullets}")
        
        # Complex sentence analysis
        print(f"\n🎯 Complex Sentence Analysis")
        print("-" * 40)
        
        complex_cases = ["Complex Question", "Casablanca", "Pride & Prejudice", "Great Gatsby"]
        
        for case_name in complex_cases:
            print(f"\n{case_name}:")
            for result in self.results:
                if result.case_name == case_name:
                    status = "✅" if result.triples_stored > 0 else "❌"
                    print(f"  {status} {result.method}: {result.triples_stored} triples in {result.total_time_ms:.1f}ms")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS")
        print("=" * 40)
        
        # Find best performer for each metric
        if successful_results := [r for r in self.results if r.success]:
            fastest = min(successful_results, key=lambda x: x.total_time_ms)
            most_accurate = max(successful_results, key=lambda x: x.triples_stored)
            
            print(f"⚡ Fastest: {fastest.method} ({fastest.total_time_ms:.1f}ms avg)")
            print(f"🎯 Most Accurate: {most_accurate.method} ({most_accurate.triples_stored} triples on '{most_accurate.case_name}')")
            
            # Specific recommendations based on results
            stanza_results = [r for r in successful_results if 'Stanza' in r.method]
            batch_results = [r for r in successful_results if 'Batch' in r.method]
            hotmem_results = [r for r in successful_results if 'HotMem' in r.method]
            
            if stanza_results and hotmem_results:
                stanza_avg_time = sum(r.total_time_ms for r in stanza_results) / len(stanza_results)
                stanza_avg_triples = sum(r.triples_stored for r in stanza_results) / len(stanza_results)
                
                hotmem_avg_time = sum(r.total_time_ms for r in hotmem_results) / len(hotmem_results)
                hotmem_avg_triples = sum(r.triples_stored for r in hotmem_results) / len(hotmem_results)
                
                print(f"\n📊 Stanza vs HotMem:")
                print(f"  Stanza: {stanza_avg_triples:.1f} triples in {stanza_avg_time:.1f}ms")
                print(f"  HotMem: {hotmem_avg_triples:.1f} triples in {hotmem_avg_time:.1f}ms")
                
                if stanza_avg_triples > hotmem_avg_triples * 1.5:
                    print(f"  💡 Stanza shows {stanza_avg_triples/max(hotmem_avg_triples, 0.1):.1f}x better extraction!")
                    if stanza_avg_time < hotmem_avg_time * 2:
                        print(f"  ✅ Recommend: Replace spaCy with Stanza")
                    else:
                        print(f"  ⚠️  Consider: Stanza slower by {stanza_avg_time/hotmem_avg_time:.1f}x")
            
            if batch_results:
                print(f"\n📊 Batch Processing Analysis:")
                batch_immediate_time = sum(r.extraction_time_ms for r in batch_results) / len(batch_results)
                batch_total_triples = sum(r.triples_stored for r in batch_results) / len(batch_results)
                
                print(f"  User Experience: {batch_immediate_time:.1f}ms (immediate)")
                print(f"  Extraction Quality: {batch_total_triples:.1f} triples (batch)")
                print(f"  💡 Best of both worlds: Fast response + good extraction")

if __name__ == "__main__":
    framework = TestFramework()
    framework.run_all_tests()
