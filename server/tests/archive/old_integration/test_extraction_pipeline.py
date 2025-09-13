#!/usr/bin/env python3
"""
Comprehensive Extraction Pipeline Inspector

This script provides detailed visibility into every step of the memory extraction pipeline,
helping identify quality issues in entity and edge extraction.

Usage:
    python test_extraction_pipeline.py "Your test sentence here"
    python test_extraction_pipeline.py --file test_sentences.txt
    python test_extraction_pipeline.py --examples
    
Options:
    --json              Output results as JSON
    --csv               Output results as CSV
    --compare           Compare with previous run
    --no-color          Disable colored output
    --verbose           Show all extraction details
    --methods [list]    Only test specific methods (ud,srl,relik,onnx)
"""

import argparse
import json
import csv
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

# Import memory components
from components.memory.memory_store import MemoryStore, Paths
from components.memory.memory_hotpath import HotMemory
from components.memory.memory_intent import get_intent_classifier, get_quality_filter, IntentType
from components.retrieval.memory_retriever import MemoryRetriever
from components.extraction.memory_extractor import MemoryExtractor
from components.context.context_orchestrator import pack_context

# Rich console for better output
console = Console()

@dataclass
class ExtractionStage:
    """Results from a single extraction stage"""
    name: str
    entities: List[str]
    triples: List[Tuple[str, str, str]]
    timing_ms: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class QualityMetrics:
    """Quality metrics for extraction results"""
    entity_coverage: float  # % of expected entities found
    relation_quality: float  # Average quality score of relations
    redundancy_ratio: float  # % of duplicate/redundant triples
    noise_ratio: float      # % of low-quality triples
    missing_patterns: List[str]  # Patterns that should have been extracted
    
class ExtractionPipelineInspector:
    """Main inspector for the extraction pipeline"""
    
    def __init__(self, verbose: bool = False, use_color: bool = True):
        self.verbose = verbose
        self.use_color = use_color
        
        # Load environment
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)
        
        # Initialize components
        self.initialize_components()
        
        # Test sentences for common patterns
        self.test_sentences = [
            "My brother Tom lives in Portland and teaches at Reed College.",
            "I met Sarah yesterday at the coffee shop where we discussed the new project.",
            "The company announced that revenue increased by 25% last quarter.",
            "John's car broke down so he had to take the bus to work.",
            "Marie Curie discovered radium and polonium while working in Paris.",
            "I think we should probably meet tomorrow at 3pm to review the proposal.",
            "Steve Jobs founded Apple in 1976 with Steve Wozniak in a garage.",
            "The weather is beautiful today but it might rain tomorrow.",
            "Can you remember what we talked about last week regarding the budget?",
            "My daughter is studying computer science at MIT and loves it.",
        ]
        
    def initialize_components(self):
        """Initialize memory and extraction components"""
        # Create temporary store for testing
        paths = Paths(
            sqlite_path="/tmp/extraction_test.db",
            lmdb_dir="/tmp/extraction_test.lmdb"
        )
        
        self.store = MemoryStore(paths=paths)
        self.hotmem = HotMemory(self.store)
        
        # Create extractor config
        extractor_config = {
            'use_srl': os.getenv('HOTMEM_USE_SRL', 'true').lower() in ('1', 'true', 'yes'),
            'use_onnx_ner': os.getenv('HOTMEM_USE_ONNX_NER', 'true').lower() in ('1', 'true', 'yes'),
            'use_onnx_srl': os.getenv('HOTMEM_USE_ONNX_SRL', 'false').lower() in ('1', 'true', 'yes'),
            'use_relik': os.getenv('HOTMEM_USE_RELIK', 'true').lower() in ('1', 'true', 'yes'),
            'use_dspy': os.getenv('HOTMEM_USE_DSPY', 'false').lower() in ('1', 'true', 'yes'),
        }
        
        self.extractor = MemoryExtractor(extractor_config)
        self.intent_classifier = get_intent_classifier()
        self.quality_filter = get_quality_filter()
        
        # Prewarm models
        console.print("[yellow]Prewarming models...[/yellow]")
        self.hotmem.prewarm('en')
        
    def inspect_sentence(self, text: str) -> Dict[str, Any]:
        """Inspect extraction pipeline for a single sentence"""
        results = {
            'input': text,
            'stages': [],
            'metrics': None,
            'timeline': []
        }
        
        console.print(Panel(f"[bold]Inspecting:[/bold] {text}", style="cyan"))
        
        # Stage 1: Intent Classification
        stage_intent = self.extract_intent(text)
        results['stages'].append(stage_intent)
        results['intent'] = stage_intent.metadata.get('intent_type')
        
        # Stage 2: Entity Extraction (Multiple Methods)
        stage_entities = self.extract_entities_detailed(text)
        results['stages'].extend(stage_entities)
        
        # Stage 3: Triple Extraction (Multiple Methods)
        stage_triples = self.extract_triples_detailed(text)
        results['stages'].extend(stage_triples)
        
        # Stage 4: Refinement
        all_triples = []
        for stage in stage_triples:
            all_triples.extend(stage.triples)
        
        stage_refined = self.refine_triples(text, all_triples)
        results['stages'].append(stage_refined)
        
        # Stage 5: Coreference Resolution
        stage_coref = self.apply_coreference(text, stage_refined.triples)
        results['stages'].append(stage_coref)
        
        # Stage 6: Quality Assessment
        all_entities = set()
        for stage in stage_entities:
            all_entities.update(stage.entities)
        
        metrics = self.calculate_quality_metrics(
            text, 
            list(all_entities),
            stage_coref.triples
        )
        results['metrics'] = metrics
        
        # Stage 7: Retrieval Test
        stage_retrieval = self.test_retrieval(text, list(all_entities), stage_coref.triples)
        results['stages'].append(stage_retrieval)
        
        # Stage 8: Context Building
        stage_context = self.build_context(text, stage_retrieval.metadata.get('bullets', []))
        results['stages'].append(stage_context)
        
        # Display results
        self.display_results(results)
        
        return results
    
    def extract_intent(self, text: str) -> ExtractionStage:
        """Extract intent from text"""
        start = time.perf_counter()
        
        intent = self.intent_classifier.analyze(text, 'en')
        intent_type = getattr(intent.intent, 'name', 'unknown') if intent else 'unknown'
        
        timing = (time.perf_counter() - start) * 1000
        
        return ExtractionStage(
            name="Intent Classification",
            entities=[],
            triples=[],
            timing_ms=timing,
            metadata={'intent_type': intent_type, 'intent': intent}
        )
    
    def extract_entities_detailed(self, text: str) -> List[ExtractionStage]:
        """Extract entities using multiple methods"""
        stages = []
        
        # Load spaCy doc
        from components.memory.memory_hotpath import _load_nlp
        nlp = _load_nlp('en')
        doc = nlp(text) if text else None
        
        if not doc:
            return stages
        
        # Method 1: spaCy NER
        start = time.perf_counter()
        spacy_entities = [ent.text for ent in doc.ents]
        timing = (time.perf_counter() - start) * 1000
        stages.append(ExtractionStage(
            name="spaCy NER",
            entities=spacy_entities,
            triples=[],
            timing_ms=timing,
            metadata={'entity_types': {ent.text: ent.label_ for ent in doc.ents}}
        ))
        
        # Method 2: Noun Chunks
        start = time.perf_counter()
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        timing = (time.perf_counter() - start) * 1000
        stages.append(ExtractionStage(
            name="Noun Chunks",
            entities=noun_chunks,
            triples=[],
            timing_ms=timing,
            metadata={'chunk_roots': {chunk.text: chunk.root.text for chunk in doc.noun_chunks}}
        ))
        
        # Method 3: ONNX NER (if available)
        if self.hotmem.use_onnx_ner and self.hotmem._onnx_ner:
            start = time.perf_counter()
            try:
                onnx_results = self.hotmem._onnx_ner.extract(text)
                onnx_entities = [span_text for span_text, label, score, span in onnx_results]
                timing = (time.perf_counter() - start) * 1000
                stages.append(ExtractionStage(
                    name="ONNX NER",
                    entities=onnx_entities,
                    triples=[],
                    timing_ms=timing,
                    metadata={'scores': {e[0]: e[2] for e in onnx_results}}
                ))
            except Exception as e:
                logger.debug(f"ONNX NER failed: {e}")
        
        # Method 4: Combined Entity Map (what HotMem actually uses)
        start = time.perf_counter()
        entities_set = set()
        entity_map = self.hotmem._build_entity_map(doc, entities_set)
        combined_entities = list(entities_set)
        timing = (time.perf_counter() - start) * 1000
        stages.append(ExtractionStage(
            name="Combined Entity Map",
            entities=combined_entities,
            triples=[],
            timing_ms=timing,
            metadata={'map_size': len(entity_map)}
        ))
        
        return stages
    
    def extract_triples_detailed(self, text: str) -> List[ExtractionStage]:
        """Extract triples using multiple methods"""
        stages = []
        
        # Load spaCy doc
        from components.memory.memory_hotpath import _load_nlp
        nlp = _load_nlp('en')
        doc = nlp(text) if text else None
        
        # Method 1: UD Patterns (27 dependency patterns)
        if doc:
            start = time.perf_counter()
            ud_entities, ud_triples, neg_count = self.hotmem._extract_from_doc(doc)
            timing = (time.perf_counter() - start) * 1000
            stages.append(ExtractionStage(
                name="UD Patterns",
                entities=ud_entities,
                triples=ud_triples,
                timing_ms=timing,
                metadata={'negation_count': neg_count, 'pattern_count': 27}
            ))
        
        # Method 2: SRL (Semantic Role Labeling)
        if self.hotmem.use_srl:
            try:
                from components.processing.semantic_roles import SRLExtractor
                if not self.hotmem._srl:
                    self.hotmem._srl = SRLExtractor(use_normalizer=True)
                
                start = time.perf_counter()
                preds = self.hotmem._srl.doc_to_predications(doc, 'en') if doc else []
                srl_triples = self.hotmem._srl.predications_to_triples(preds)
                timing = (time.perf_counter() - start) * 1000
                
                stages.append(ExtractionStage(
                    name="SRL",
                    entities=[],
                    triples=srl_triples,
                    timing_ms=timing,
                    metadata={'predications': len(preds)}
                ))
            except Exception as e:
                logger.debug(f"SRL extraction failed: {e}")
        
        # Method 3: ReLiK (if available)
        if self.hotmem.use_relik and self.hotmem._relik:
            try:
                start = time.perf_counter()
                relik_results = self.hotmem._relik.extract(text) or []
                relik_triples = [(s, r, d) for s, r, d, conf in relik_results]
                timing = (time.perf_counter() - start) * 1000
                
                stages.append(ExtractionStage(
                    name="ReLiK",
                    entities=[],
                    triples=relik_triples,
                    timing_ms=timing,
                    metadata={'confidences': {(s,r,d): conf for s,r,d,conf in relik_results}}
                ))
            except Exception as e:
                logger.debug(f"ReLiK extraction failed: {e}")
        
        # Method 4: ONNX SRL (if available)
        if self.hotmem.use_onnx_srl and self.hotmem._onnx_srl:
            try:
                start = time.perf_counter()
                roles_list = self.hotmem._onnx_srl.extract(text)
                onnx_triples = []
                for roles in roles_list:
                    pred = roles.get('predicate', '')
                    agent = roles.get('agent', '')
                    patient = roles.get('patient', '') or roles.get('destination', '')
                    if agent and patient:
                        onnx_triples.append((agent, pred, patient))
                timing = (time.perf_counter() - start) * 1000
                
                stages.append(ExtractionStage(
                    name="ONNX SRL",
                    entities=[],
                    triples=onnx_triples,
                    timing_ms=timing,
                    metadata={'roles_count': len(roles_list)}
                ))
            except Exception as e:
                logger.debug(f"ONNX SRL extraction failed: {e}")
        
        return stages
    
    def refine_triples(self, text: str, triples: List[Tuple[str, str, str]]) -> ExtractionStage:
        """Apply refinement to triples"""
        start = time.perf_counter()
        
        # Use HotMem's refinement
        from components.memory.memory_hotpath import _load_nlp
        nlp = _load_nlp('en')
        doc = nlp(text) if text else None
        intent = self.intent_classifier.analyze(text, 'en')
        
        refined = self.hotmem._refine_triples(text, triples, doc, intent, 'en')
        
        timing = (time.perf_counter() - start) * 1000
        
        return ExtractionStage(
            name="Refinement",
            entities=[],
            triples=refined,
            timing_ms=timing,
            metadata={
                'input_count': len(triples),
                'output_count': len(refined),
                'reduction': f"{(1 - len(refined)/max(1, len(triples)))*100:.1f}%"
            }
        )
    
    def apply_coreference(self, text: str, triples: List[Tuple[str, str, str]]) -> ExtractionStage:
        """Apply coreference resolution"""
        start = time.perf_counter()
        
        from components.memory.memory_hotpath import _load_nlp
        nlp = _load_nlp('en')
        doc = nlp(text) if text else None
        
        # Try neural coref first, fall back to lite
        if self.hotmem.use_coref and self.hotmem._coref_model:
            try:
                resolved = self.hotmem._apply_coref_neural(triples, doc)
                method = "neural"
            except:
                resolved = self.hotmem._apply_coref_lite(triples, doc)
                method = "lite"
        else:
            resolved = self.hotmem._apply_coref_lite(triples, doc)
            method = "lite"
        
        timing = (time.perf_counter() - start) * 1000
        
        # Find what changed
        changes = []
        for orig, res in zip(triples, resolved):
            if orig != res:
                changes.append({'from': orig, 'to': res})
        
        return ExtractionStage(
            name="Coreference Resolution",
            entities=[],
            triples=resolved,
            timing_ms=timing,
            metadata={
                'method': method,
                'changes': changes[:5],  # Show first 5 changes
                'change_count': len(changes)
            }
        )
    
    def calculate_quality_metrics(self, text: str, entities: List[str], triples: List[Tuple[str, str, str]]) -> QualityMetrics:
        """Calculate quality metrics for extraction results"""
        
        # Expected entities (simple heuristic: proper nouns and pronouns)
        import spacy
        from components.memory.memory_hotpath import _load_nlp
        nlp = _load_nlp('en')
        doc = nlp(text)
        
        expected_entities = set()
        for token in doc:
            if token.pos_ in ['PROPN', 'NOUN'] and not token.is_stop:
                expected_entities.add(token.text.lower())
        for ent in doc.ents:
            expected_entities.add(ent.text.lower())
        
        found_entities = set(e.lower() for e in entities)
        
        # Entity coverage
        entity_coverage = len(found_entities & expected_entities) / max(1, len(expected_entities))
        
        # Relation quality (using quality filter)
        quality_scores = []
        intent = self.intent_classifier.analyze(text, 'en')
        for s, r, d in triples:
            should_store, conf = self.quality_filter.should_store_fact(s, r, d, intent)
            quality_scores.append(conf if should_store else 0.0)
        
        relation_quality = sum(quality_scores) / max(1, len(quality_scores)) if quality_scores else 0.0
        
        # Redundancy detection
        unique_triples = set(triples)
        redundancy_ratio = 1.0 - (len(unique_triples) / max(1, len(triples)))
        
        # Noise detection (low quality triples)
        noise_count = sum(1 for score in quality_scores if score < 0.3)
        noise_ratio = noise_count / max(1, len(triples)) if triples else 0.0
        
        # Missing patterns (heuristic)
        missing_patterns = []
        
        # Check for common patterns that might be missing
        if "lives in" in text.lower() or "live in" in text.lower():
            if not any("lives" in r or "resides" in r for _, r, _ in triples):
                missing_patterns.append("location/residence relation")
        
        if "works at" in text.lower() or "employed by" in text.lower():
            if not any("works" in r or "employed" in r for _, r, _ in triples):
                missing_patterns.append("employment relation")
        
        if "'s" in text:  # Possessive
            if not any("has" in r or "owns" in r or "possesses" in r for _, r, _ in triples):
                missing_patterns.append("possession relation")
        
        return QualityMetrics(
            entity_coverage=entity_coverage,
            relation_quality=relation_quality,
            redundancy_ratio=redundancy_ratio,
            noise_ratio=noise_ratio,
            missing_patterns=missing_patterns
        )
    
    def test_retrieval(self, text: str, entities: List[str], triples: List[Tuple[str, str, str]]) -> ExtractionStage:
        """Test retrieval with extracted entities"""
        start = time.perf_counter()
        
        # Add some triples to memory for testing
        for s, r, d in triples[:5]:  # Add first 5 triples
            self.store.observe_edge(s, r, d, 0.8, now_ts=int(time.time() * 1000))
            # Also update hotmem's entity index for retrieval
            self.hotmem.entity_index[s].add((s, r, d))
            self.hotmem.entity_index[d].add((s, r, d))
        
        # Test retrieval
        bullets = self.hotmem._retrieve_context(text, entities, turn_id=1, intent=None)
        
        timing = (time.perf_counter() - start) * 1000
        
        return ExtractionStage(
            name="Retrieval Test",
            entities=entities[:5],  # Show first 5 entities used
            triples=[],
            timing_ms=timing,
            metadata={
                'bullets': bullets,
                'bullet_count': len(bullets),
                'entity_count': len(entities)
            }
        )
    
    def build_context(self, text: str, bullets: List[str]) -> ExtractionStage:
        """Build final context for assistant"""
        start = time.perf_counter()
        
        # Simple context building - just join bullets for now
        context = "\n".join(bullets) if bullets else "No memory context available."
        
        timing = (time.perf_counter() - start) * 1000
        
        return ExtractionStage(
            name="Context Building",
            entities=[],
            triples=[],
            timing_ms=timing,
            metadata={
                'context': context[:500],  # First 500 chars
                'total_length': len(context),
                'bullet_count': len(bullets)
            }
        )
    
    def display_results(self, results: Dict[str, Any]):
        """Display results in a nice format"""
        
        # Summary table
        table = Table(title="Extraction Pipeline Summary", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Entities", style="green")
        table.add_column("Triples", style="blue")
        table.add_column("Time (ms)", style="yellow")
        table.add_column("Notes", style="white")
        
        for stage in results['stages']:
            notes = ""
            if stage.metadata:
                if 'reduction' in stage.metadata:
                    notes = f"Reduced by {stage.metadata['reduction']}"
                elif 'change_count' in stage.metadata:
                    notes = f"{stage.metadata['change_count']} changes"
                elif 'intent_type' in stage.metadata:
                    notes = f"Intent: {stage.metadata['intent_type']}"
            
            table.add_row(
                stage.name,
                str(len(stage.entities)),
                str(len(stage.triples)),
                f"{stage.timing_ms:.1f}",
                notes
            )
        
        console.print(table)
        
        # Quality metrics
        if results['metrics']:
            metrics = results['metrics']
            
            metrics_table = Table(title="Quality Metrics", show_header=True)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            metrics_table.add_column("Status", style="yellow")
            
            # Entity coverage
            coverage_status = "✅" if metrics.entity_coverage > 0.7 else "⚠️" if metrics.entity_coverage > 0.4 else "❌"
            metrics_table.add_row(
                "Entity Coverage",
                f"{metrics.entity_coverage:.1%}",
                coverage_status
            )
            
            # Relation quality
            quality_status = "✅" if metrics.relation_quality > 0.6 else "⚠️" if metrics.relation_quality > 0.3 else "❌"
            metrics_table.add_row(
                "Relation Quality",
                f"{metrics.relation_quality:.1%}",
                quality_status
            )
            
            # Redundancy
            redundancy_status = "✅" if metrics.redundancy_ratio < 0.2 else "⚠️" if metrics.redundancy_ratio < 0.4 else "❌"
            metrics_table.add_row(
                "Redundancy",
                f"{metrics.redundancy_ratio:.1%}",
                redundancy_status
            )
            
            # Noise
            noise_status = "✅" if metrics.noise_ratio < 0.2 else "⚠️" if metrics.noise_ratio < 0.4 else "❌"
            metrics_table.add_row(
                "Noise Ratio",
                f"{metrics.noise_ratio:.1%}",
                noise_status
            )
            
            console.print(metrics_table)
            
            if metrics.missing_patterns:
                console.print(Panel(
                    "\n".join(f"• {pattern}" for pattern in metrics.missing_patterns),
                    title="[red]Missing Patterns[/red]",
                    style="red"
                ))
        
        # Show some example triples
        if self.verbose:
            for stage in results['stages']:
                if stage.triples:
                    console.print(f"\n[bold]{stage.name} Triples:[/bold]")
                    for i, (s, r, d) in enumerate(stage.triples[:5], 1):
                        console.print(f"  {i}. ({s}, {r}, {d})")
                    if len(stage.triples) > 5:
                        console.print(f"  ... and {len(stage.triples) - 5} more")
    
    def export_json(self, results: Dict[str, Any], filename: str):
        """Export results to JSON"""
        # Convert dataclasses to dicts
        export_data = {
            'input': results['input'],
            'intent': results.get('intent'),
            'stages': [asdict(stage) for stage in results['stages']],
            'metrics': asdict(results['metrics']) if results['metrics'] else None
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        console.print(f"[green]Results exported to {filename}[/green]")
    
    def export_csv(self, results: Dict[str, Any], filename: str):
        """Export results to CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Stage', 'Entity Count', 'Triple Count', 'Time (ms)', 'Notes'])
            
            # Data
            for stage in results['stages']:
                notes = json.dumps(stage.metadata) if stage.metadata else ""
                writer.writerow([
                    stage.name,
                    len(stage.entities),
                    len(stage.triples),
                    f"{stage.timing_ms:.1f}",
                    notes
                ])
        
        console.print(f"[green]Results exported to {filename}[/green]")
    
    def run_examples(self):
        """Run inspection on example sentences"""
        all_results = []
        
        for sentence in self.test_sentences:
            console.print("\n" + "="*80 + "\n")
            results = self.inspect_sentence(sentence)
            all_results.append(results)
            
            # Brief pause between sentences
            time.sleep(0.1)
        
        # Summary statistics
        self.print_summary_statistics(all_results)
        
        return all_results
    
    def print_summary_statistics(self, all_results: List[Dict[str, Any]]):
        """Print summary statistics across all test sentences"""
        console.print("\n" + "="*80)
        console.print(Panel("[bold]Summary Statistics[/bold]", style="cyan"))
        
        # Aggregate metrics
        avg_coverage = sum(r['metrics'].entity_coverage for r in all_results if r['metrics']) / len(all_results)
        avg_quality = sum(r['metrics'].relation_quality for r in all_results if r['metrics']) / len(all_results)
        avg_redundancy = sum(r['metrics'].redundancy_ratio for r in all_results if r['metrics']) / len(all_results)
        avg_noise = sum(r['metrics'].noise_ratio for r in all_results if r['metrics']) / len(all_results)
        
        # Stage timings
        stage_timings = defaultdict(list)
        for result in all_results:
            for stage in result['stages']:
                stage_timings[stage.name].append(stage.timing_ms)
        
        # Print aggregate metrics
        metrics_table = Table(title="Average Quality Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Average", style="green")
        
        metrics_table.add_row("Entity Coverage", f"{avg_coverage:.1%}")
        metrics_table.add_row("Relation Quality", f"{avg_quality:.1%}")
        metrics_table.add_row("Redundancy Ratio", f"{avg_redundancy:.1%}")
        metrics_table.add_row("Noise Ratio", f"{avg_noise:.1%}")
        
        console.print(metrics_table)
        
        # Print timing statistics
        timing_table = Table(title="Average Stage Timings", show_header=True)
        timing_table.add_column("Stage", style="cyan")
        timing_table.add_column("Avg Time (ms)", style="yellow")
        timing_table.add_column("Min", style="green")
        timing_table.add_column("Max", style="red")
        
        for stage_name, timings in stage_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                min_time = min(timings)
                max_time = max(timings)
                timing_table.add_row(
                    stage_name,
                    f"{avg_time:.1f}",
                    f"{min_time:.1f}",
                    f"{max_time:.1f}"
                )
        
        console.print(timing_table)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Extraction Pipeline Inspector")
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--file', help='File with test sentences (one per line)')
    parser.add_argument('--examples', action='store_true', help='Run built-in examples')
    parser.add_argument('--json', help='Export results to JSON file')
    parser.add_argument('--csv', help='Export results to CSV file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--methods', nargs='+', help='Only test specific methods', 
                       choices=['ud', 'srl', 'relik', 'onnx'])
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = ExtractionPipelineInspector(
        verbose=args.verbose,
        use_color=not args.no_color
    )
    
    # Run appropriate mode
    if args.examples:
        results = inspector.run_examples()
        if args.json:
            # Export all results
            with open(args.json, 'w') as f:
                json.dump([{
                    'input': r['input'],
                    'stages': [asdict(s) for s in r['stages']],
                    'metrics': asdict(r['metrics']) if r['metrics'] else None
                } for r in results], f, indent=2, default=str)
    elif args.file:
        with open(args.file, 'r') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        all_results = []
        for sentence in sentences:
            console.print("\n" + "="*80 + "\n")
            results = inspector.inspect_sentence(sentence)
            all_results.append(results)
        
        inspector.print_summary_statistics(all_results)
    elif args.text:
        results = inspector.inspect_sentence(args.text)
        
        if args.json:
            inspector.export_json(results, args.json)
        if args.csv:
            inspector.export_csv(results, args.csv)
    else:
        # Interactive mode
        console.print("[bold cyan]Extraction Pipeline Inspector - Interactive Mode[/bold cyan]")
        console.print("Enter sentences to analyze (or 'quit' to exit, 'examples' to run test suite):\n")
        
        while True:
            try:
                text = input("> ").strip()
                if text.lower() == 'quit':
                    break
                elif text.lower() == 'examples':
                    inspector.run_examples()
                elif text:
                    inspector.inspect_sentence(text)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("Error in interactive mode")

if __name__ == "__main__":
    main()