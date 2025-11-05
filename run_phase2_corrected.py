

import os
import sys
import json
import subprocess
from datetime import datetime

def run_single_benchmark(name, pickle_path):
    
    print(f"\n{'='*80}")
    print(f"Running {name}...")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'main_alldiff_cop.py',
        '--experiment', name,
        '--phase1_pickle', pickle_path,
        '--alpha', '0.42',
        '--theta_max', '0.9',
        '--theta_min', '0.1',
        '--max_queries', '500',
        '--timeout', '600'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    queries = None
    time_taken = None
    target = None
    learned = None
    correct = None
    missing = None
    spurious = None
    
    for line in output.split('\n'):
        if 'Total queries:' in line:
            try:
                queries = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Total time:' in line and 'Final Statistics' in output[max(0, output.index(line)-200):output.index(line)]:
            try:
                time_taken = float(line.split(':')[1].strip().replace('s', ''))
            except:
                pass
        elif 'Target AllDifferent constraints:' in line:
            try:
                target = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Learned AllDifferent constraints:' in line:
            try:
                learned = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Correct:' in line:
            try:
                correct = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Missing:' in line:
            try:
                missing = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Spurious:' in line:
            try:
                spurious = int(line.split(':')[1].strip())
            except:
                pass
    
    precision = correct / learned if learned and learned > 0 else 0
    recall = correct / target if target and target > 0 else 0
    
    return {
        'name': name,
        'queries': queries,
        'time': time_taken,
        'target': target,
        'learned': learned,
        'correct': correct,
        'missing': missing,
        'spurious': spurious,
        'precision': precision,
        'recall': recall,
        'status': 'SUCCESS' if result.returncode == 0 else 'FAILED'
    }

def main():
    benchmarks = [
        ('sudoku', 'phase1_output/sudoku_phase1.pkl'),
        ('sudoku_gt', 'phase1_output/sudoku_gt_phase1.pkl'),
        ('examtt_v1', 'phase1_output/examtt_v1_phase1.pkl'),
        ('examtt_v2', 'phase1_output/examtt_v2_phase1.pkl')
    ]
    
    results = []
    
    for name, pickle_path in benchmarks:
        result = run_single_benchmark(name, pickle_path)
        results.append(result)
        print(f"\n[DONE] {name}: {result['queries']} queries, {result['recall']:.1%} recall")

    output = {
        'experiment_date': datetime.now().isoformat(),
        'description': 'Phase 2 with CORRECTED query counting (includes disambiguation)',
        'results': results,
        'summary': {
            'total_queries': sum(r['queries'] for r in results if r['queries']),
            'avg_precision': sum(r['precision'] for r in results) / len(results),
            'avg_recall': sum(r['recall'] for r in results) / len(results),
            'perfect_recall_count': sum(1 for r in results if r['recall'] == 1.0)
        }
    }
    
    with open('phase2_corrected_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY (CORRECTED QUERY COUNTS)")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['name']:<15} Queries: {r['queries']:>4}  Precision: {r['precision']:>6.1%}  Recall: {r['recall']:>6.1%}")
    print(f"{'='*80}\n")
    print(f"Results saved to: phase2_corrected_results.json")

if __name__ == '__main__':
    main()

