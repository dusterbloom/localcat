#!/usr/bin/env python3
"""Three-way comparison: Our SRL vs ASI2 vs ASI1 8.2.3"""

import time
import warnings
warnings.filterwarnings('ignore')

# Suppress debug output
original_print = print
def quiet_print(*args, **kwargs):
    text = str(args[0]) if args else ""
    if "DEBUG:" in text or "⚠️" in text or "✅" in text or "📊" in text or "🔍" in text:
        return
    return original_print(*args, **kwargs)

import builtins
builtins.print = quiet_print

def run_three_way_comparison():
    # Import processors
    from asi1_processor import ULTRAGROKSpacyV821Processor

    # A: Our SRL (find the right one)
    try:
        from components.extraction.memory_extractor import MemoryExtractor
        processor_a = MemoryExtractor()
        processor_a_name = "Our SRL (MemoryExtractor)"
    except:
        try:
            from components.extraction.tiered_extractor import TieredExtractor
            processor_a = TieredExtractor()
            processor_a_name = "Our SRL (TieredExtractor)"
        except:
            processor_a = None
            processor_a_name = "Our SRL (Not Found)"

    # B: ASI2 Last Working (Original ULTRAGROK)
    try:
        processor_b = ULTRAGROKSpacyV821Processor('ULTRAGROK_V8.2.1_SPACY.yaml')
        processor_b_name = "ASI2 ULTRAGROK V8.2.1"
    except:
        processor_b = None
        processor_b_name = "ASI2 (Not Found)"

    # C: ASI1 V8.2.3 Integrated
    try:
        processor_c = ULTRAGROKSpacyV821Processor('ASI1_8_2_3.yaml')
        processor_c_name = "ASI1 V8.2.3 Integrated"
    except:
        processor_c = None
        processor_c_name = "ASI1 (Not Found)"

    # Test sentences - variety of complexity
    test_cases = [
        "John works at Google",
        "Mary gave the book to her friend",
        "The CEO announced quarterly results during the meeting",
        "Alice bought books and Tom bought magazines",
        "Scientists discovered artifacts using advanced technology"
    ]

    # Restore print for results
    builtins.print = original_print

    print('🏁 THREE-WAY SEMANTIC EXTRACTION COMPARISON')
    print('=' * 70)
    print('A = Our SRL System')
    print('B = ASI2 ULTRAGROK V8.2.1 (Last Working)')
    print('C = ASI1 V8.2.3 (Integrated)')
    print('=' * 70)

    results = {'A': [], 'B': [], 'C': []}
    times = {'A': [], 'B': [], 'C': []}

    for i, text in enumerate(test_cases, 1):
        print(f'\\n{i}. INPUT: "{text}"')
        print('-' * 50)

        # Test A: Our SRL
        if processor_a:
            try:
                start_time = time.time()
                if hasattr(processor_a, 'extract'):
                    result_a = processor_a.extract(text)
                    relations_a = getattr(result_a, 'relations', []) or []
                else:
                    relations_a = []
                end_time = time.time()
                times['A'].append((end_time - start_time) * 1000)
                results['A'].append(len(relations_a))

                print(f'   A ({processor_a_name}):')
                print(f'     Relations: {len(relations_a)}')
                print(f'     Speed: {times["A"][-1]:.1f}ms')
                for j, rel in enumerate(relations_a[:2], 1):
                    if hasattr(rel, 'subject'):
                        print(f'     {j}. {rel.subject} | {rel.predicate} | {rel.object}')
                    else:
                        print(f'     {j}. {rel}')
            except Exception as e:
                print(f'   A ({processor_a_name}): ERROR - {e}')
                results['A'].append(0)
                times['A'].append(0)
        else:
            print(f'   A: Not Available')
            results['A'].append(0)
            times['A'].append(0)

        # Test B: ASI2
        if processor_b:
            try:
                start_time = time.time()
                result_b = processor_b.process_spacy_semantics(text)
                triples_b = result_b.get('triples', [])
                end_time = time.time()
                times['B'].append((end_time - start_time) * 1000)
                results['B'].append(len(triples_b))

                print(f'   B ({processor_b_name}):')
                print(f'     Relations: {len(triples_b)}')
                print(f'     Speed: {times["B"][-1]:.1f}ms')
                for j, triple in enumerate(triples_b[:2], 1):
                    subj = getattr(triple, 'subject', 'N/A')
                    pred = getattr(triple, 'predicate', 'N/A')
                    obj = getattr(triple, 'object', 'N/A')
                    print(f'     {j}. {subj} | {pred} | {obj}')
            except Exception as e:
                print(f'   B ({processor_b_name}): ERROR - {e}')
                results['B'].append(0)
                times['B'].append(0)
        else:
            print(f'   B: Not Available')
            results['B'].append(0)
            times['B'].append(0)

        # Test C: ASI1
        if processor_c:
            try:
                start_time = time.time()
                result_c = processor_c.process_spacy_semantics(text)
                triples_c = result_c.get('triples', [])
                end_time = time.time()
                times['C'].append((end_time - start_time) * 1000)
                results['C'].append(len(triples_c))

                print(f'   C ({processor_c_name}):')
                print(f'     Relations: {len(triples_c)}')
                print(f'     Speed: {times["C"][-1]:.1f}ms')
                for j, triple in enumerate(triples_c[:2], 1):
                    subj = getattr(triple, 'subject', 'N/A')
                    pred = getattr(triple, 'predicate', 'N/A')
                    obj = getattr(triple, 'object', 'N/A')
                    print(f'     {j}. {subj} | {pred} | {obj}')
            except Exception as e:
                print(f'   C ({processor_c_name}): ERROR - {e}')
                results['C'].append(0)
                times['C'].append(0)
        else:
            print(f'   C: Not Available')
            results['C'].append(0)
            times['C'].append(0)

    # Final Analysis
    print(f'\\n🏆 FINAL PERFORMANCE ANALYSIS')
    print('=' * 45)
    print(f'Test Cases: {len(test_cases)}')
    print()

    for system in ['A', 'B', 'C']:
        total_relations = sum(results[system])
        avg_relations = total_relations / len(test_cases)
        avg_speed = sum(times[system]) / len(times[system]) if times[system] else 0

        system_name = {
            'A': processor_a_name,
            'B': processor_b_name,
            'C': processor_c_name
        }[system]

        print(f'{system} ({system_name}):')
        print(f'  Total Relations: {total_relations}')
        print(f'  Average/Sentence: {avg_relations:.1f}')
        print(f'  Average Speed: {avg_speed:.1f}ms')
        print()

    # Speed Ranking
    speed_ranking = []
    for system in ['A', 'B', 'C']:
        avg_speed = sum(times[system]) / len(times[system]) if times[system] else float('inf')
        speed_ranking.append((system, avg_speed))

    speed_ranking.sort(key=lambda x: x[1])

    print('🚀 SPEED RANKING (Fastest to Slowest):')
    for i, (system, speed) in enumerate(speed_ranking, 1):
        if speed == float('inf'):
            print(f'  {i}. {system}: Not Available')
        else:
            print(f'  {i}. {system}: {speed:.1f}ms')

    # Quality Ranking
    quality_ranking = []
    for system in ['A', 'B', 'C']:
        total_relations = sum(results[system])
        quality_ranking.append((system, total_relations))

    quality_ranking.sort(key=lambda x: x[1], reverse=True)

    print('\\n📊 EXTRACTION VOLUME RANKING (Most to Least):')
    for i, (system, relations) in enumerate(quality_ranking, 1):
        print(f'  {i}. {system}: {relations} total relations')

if __name__ == "__main__":
    run_three_way_comparison()