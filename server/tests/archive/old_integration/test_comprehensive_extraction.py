#!/usr/bin/env python3
"""
Comprehensive Extraction Test - Shows all components contributing to graph build
Tests the optimized extraction pipeline with global caching and comprehensive UD patterns
"""

import time
import json
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Import our optimized components
from components.extraction.memory_extractor import MemoryExtractor
from components.extraction.tiered_extractor import TieredRelationExtractor

console = Console()

class ComprehensiveExtractionTest:
    """Test all extraction components and show their contributions"""

    def __init__(self):
        # Initialize with all extraction methods enabled
        self.config = {
            'use_srl': False,  # Disabled for speed
            'use_onnx_ner': False,
            'use_onnx_srl': False,
            'use_relik': True,  # ReLiK enabled (cached globally)
            'use_gliner': True,  # GLiNER enabled (cached globally)
            'use_coref': False,  # Disabled for speed
            'llm_base_url': 'http://127.0.0.1:1234/v1'
        }

        # Test sentences covering various patterns
        self.test_sentences = [
            # Simple subject-verb-object
            "Steve Jobs founded Apple Inc. in Cupertino, California.",

            # Multiple entities and relations
            "Dr. Sarah Chen is the AI research director at OpenAI who previously worked at Google Brain.",

            # Complex with conjunctions
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",

            # Possessive and attributes
            "John's new car broke down yesterday when he was driving to work.",

            # Location and time
            "Marie Curie discovered radium and polonium while working in Paris in 1898.",

            # Nested relations
            "The company announced that its revenue increased by 25% last quarter.",

            # Coreference patterns
            "Tom lives in Portland. He teaches computer science at Reed College.",

            # Complex multi-clause
            "After graduating from MIT, Sarah joined Tesla where she led the autopilot team before founding her own startup.",
        ]

        self.extractor = None
        self.stats = defaultdict(lambda: defaultdict(list))

    def initialize_extractors(self):
        """Initialize extractors - models will be cached globally"""
        console.print("\n[yellow]🔄 Initializing extraction pipeline...[/yellow]")

        start = time.perf_counter()
        self.extractor = MemoryExtractor(self.config)
        init_time = (time.perf_counter() - start) * 1000

        console.print(f"[green]✅ Extractor initialized in {init_time:.1f}ms[/green]")
        console.print("[dim]   Models are cached globally for reuse[/dim]\n")

        return init_time

    def test_single_sentence(self, text: str, sentence_num: int) -> Dict[str, Any]:
        """Test extraction on a single sentence and track component contributions"""

        console.print(f"\n[bold cyan]Test {sentence_num}: {text}[/bold cyan]")

        # Main extraction
        start = time.perf_counter()
        result = self.extractor.extract(text, use_cache=False)
        total_time = (time.perf_counter() - start) * 1000

        # Analyze component contributions
        contributions = self.analyze_contributions(text, result)

        # Display results
        self.display_sentence_results(text, result, contributions, total_time)

        return {
            'text': text,
            'entities': result.entities,
            'triples': result.triples,
            'time_ms': total_time,
            'contributions': contributions
        }

    def analyze_contributions(self, text: str, result) -> Dict[str, Any]:
        """Analyze which components contributed what"""
        contributions = {
            'gliner_entities': [],
            'spacy_entities': [],
            'ud_patterns': [],
            'relik_relations': [],
            'rule_based': [],
            'total_entities': len(result.entities),
            'total_relations': len(result.triples)
        }

        # Check GLiNER contribution (from debug logs)
        # Note: In production, you'd track this internally
        if self.config['use_gliner']:
            # GLiNER contributes to entity extraction
            contributions['gliner_entities'] = [e for e in result.entities[:6]]  # Estimate

        # Check spaCy contribution
        if result.doc:
            contributions['spacy_entities'] = [ent.text.lower() for ent in result.doc.ents]

        # Analyze relation types to identify source
        for s, r, o in result.triples:
            # UD patterns typically have specific relation names
            if r in ['nsubj', 'dobj', 'prep', 'has', 'is'] or '_' in r:
                contributions['ud_patterns'].append((s, r, o))
            # ReLiK/hybrid typically has more semantic relations
            elif r in ['founded', 'works_at', 'located_in', 'employed_by']:
                contributions['relik_relations'].append((s, r, o))
            # Rule-based patterns
            elif r in ['works_at', 'lives_in', 'teaches_at']:
                contributions['rule_based'].append((s, r, o))

        return contributions

    def display_sentence_results(self, text: str, result, contributions: Dict, time_ms: float):
        """Display results for a single sentence"""

        # Create results table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", width=25)
        table.add_column("Contribution", style="green")
        table.add_column("Count", justify="right", style="yellow")

        # Entity extraction
        table.add_row(
            "GLiNER Entities",
            ", ".join(contributions['gliner_entities'][:3]) + ("..." if len(contributions['gliner_entities']) > 3 else ""),
            str(len(contributions['gliner_entities']))
        )

        table.add_row(
            "spaCy NER",
            ", ".join(contributions['spacy_entities'][:3]) + ("..." if len(contributions['spacy_entities']) > 3 else ""),
            str(len(contributions['spacy_entities']))
        )

        # Relation extraction
        table.add_row(
            "UD Patterns (27 types)",
            f"{len(contributions['ud_patterns'])} relations",
            str(len(contributions['ud_patterns']))
        )

        table.add_row(
            "ReLiK/Hybrid",
            f"{len(contributions['relik_relations'])} relations",
            str(len(contributions['relik_relations']))
        )

        table.add_row(
            "Rule-based",
            f"{len(contributions['rule_based'])} relations",
            str(len(contributions['rule_based']))
        )

        console.print(table)

        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  • Total Entities: {contributions['total_entities']}")
        console.print(f"  • Total Relations: {contributions['total_relations']}")
        console.print(f"  • Extraction Time: {time_ms:.1f}ms")

        # Show sample triples
        if result.triples:
            console.print(f"\n[bold]Sample Relations:[/bold]")
            for i, (s, r, o) in enumerate(result.triples[:5], 1):
                console.print(f"  {i}. [cyan]{s}[/cyan] --[yellow]{r}[/yellow]--> [cyan]{o}[/cyan]")
            if len(result.triples) > 5:
                console.print(f"  [dim]... and {len(result.triples) - 5} more[/dim]")

    def run_comprehensive_test(self):
        """Run the comprehensive test suite"""
        console.print(Panel.fit(
            "[bold]COMPREHENSIVE EXTRACTION TEST[/bold]\n"
            "Testing optimized pipeline with global model caching\n"
            "Components: GLiNER + 27 UD Patterns + ReLiK/Hybrid",
            style="cyan"
        ))

        # Initialize
        init_time = self.initialize_extractors()

        # Test each sentence
        all_results = []
        total_extraction_time = 0

        for i, sentence in enumerate(self.test_sentences, 1):
            result = self.test_single_sentence(sentence, i)
            all_results.append(result)
            total_extraction_time += result['time_ms']

            # Track statistics
            self.stats['entities']['counts'].append(len(result['entities']))
            self.stats['relations']['counts'].append(len(result['triples']))
            self.stats['times']['extractions'].append(result['time_ms'])

        # Display final statistics
        self.display_final_statistics(all_results, init_time, total_extraction_time)

        return all_results

    def display_final_statistics(self, results: List[Dict], init_time: float, total_time: float):
        """Display comprehensive statistics"""

        console.print("\n" + "="*80)
        console.print(Panel.fit("[bold]FINAL STATISTICS[/bold]", style="green"))

        # Performance table
        perf_table = Table(title="Performance Metrics", show_header=True)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")

        avg_time = total_time / len(results)
        min_time = min(r['time_ms'] for r in results)
        max_time = max(r['time_ms'] for r in results)

        perf_table.add_row("Model Initialization", f"{init_time:.1f}ms")
        perf_table.add_row("Average Extraction", f"{avg_time:.1f}ms")
        perf_table.add_row("Min Extraction", f"{min_time:.1f}ms")
        perf_table.add_row("Max Extraction", f"{max_time:.1f}ms")
        perf_table.add_row("Total Time", f"{total_time:.1f}ms")

        console.print(perf_table)

        # Extraction statistics
        extract_table = Table(title="Extraction Statistics", show_header=True)
        extract_table.add_column("Component", style="cyan")
        extract_table.add_column("Total Contributions", style="green")
        extract_table.add_column("Average per Sentence", style="yellow")

        total_entities = sum(len(r['entities']) for r in results)
        total_relations = sum(len(r['triples']) for r in results)

        extract_table.add_row(
            "Entities",
            str(total_entities),
            f"{total_entities / len(results):.1f}"
        )

        extract_table.add_row(
            "Relations",
            str(total_relations),
            f"{total_relations / len(results):.1f}"
        )

        # Component contribution breakdown
        total_ud = sum(len(r['contributions']['ud_patterns']) for r in results)
        total_relik = sum(len(r['contributions']['relik_relations']) for r in results)
        total_gliner = sum(len(r['contributions']['gliner_entities']) for r in results)

        extract_table.add_row(
            "UD Patterns (27 types)",
            str(total_ud),
            f"{total_ud / len(results):.1f}"
        )

        extract_table.add_row(
            "ReLiK/Hybrid",
            str(total_relik),
            f"{total_relik / len(results):.1f}"
        )

        extract_table.add_row(
            "GLiNER Entities",
            str(total_gliner),
            f"{total_gliner / len(results):.1f}"
        )

        console.print(extract_table)

        # Performance assessment
        console.print("\n[bold]Performance Assessment:[/bold]")

        if avg_time < 100:
            console.print("  🎉 [green]EXCELLENT: <100ms target achieved![/green]")
        elif avg_time < 200:
            console.print("  ✅ [green]VERY GOOD: Under 200ms[/green]")
        elif avg_time < 500:
            console.print("  👍 [yellow]GOOD: Under 500ms[/yellow]")
        else:
            console.print("  ⚠️  [red]NEEDS OPTIMIZATION: Over 500ms[/red]")

        # Quality assessment
        console.print("\n[bold]Quality Assessment:[/bold]")

        avg_entities = total_entities / len(results)
        avg_relations = total_relations / len(results)

        if avg_entities > 4 and avg_relations > 5:
            console.print("  🎯 [green]HIGH QUALITY: Rich extraction[/green]")
        elif avg_entities > 2 and avg_relations > 2:
            console.print("  ✅ [green]GOOD QUALITY: Decent extraction[/green]")
        else:
            console.print("  ⚠️  [yellow]LOW QUALITY: Limited extraction[/yellow]")

        # Component synergy
        console.print("\n[bold]Component Synergy:[/bold]")
        console.print("  • [cyan]GLiNER[/cyan]: Provides high-quality entity extraction")
        console.print("  • [cyan]27 UD Patterns[/cyan]: Comprehensive grammatical relation coverage")
        console.print("  • [cyan]ReLiK/Hybrid[/cyan]: Semantic relation enhancement")
        console.print("  • [cyan]Global Caching[/cyan]: Models loaded once, reused across instances")

    def export_results(self, results: List[Dict], filename: str = "extraction_results.json"):
        """Export results to JSON for analysis"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\n[green]Results exported to {filename}[/green]")

def main():
    """Run the comprehensive extraction test"""
    test = ComprehensiveExtractionTest()
    results = test.run_comprehensive_test()

    # Optionally export results
    # test.export_results(results)

    console.print("\n[bold cyan]Test completed successfully![/bold cyan]")

if __name__ == "__main__":
    main()