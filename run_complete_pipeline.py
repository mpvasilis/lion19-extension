

import os
import sys
import subprocess
import json
from datetime import datetime


def run_command(cmd, description):
    
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        return False
    
    return True


def run_phase2(experiment, phase1_pickle):
    
    cmd = [
        'python', 'main_alldiff_cop.py',
        '--experiment', experiment,
        '--phase1_pickle', phase1_pickle,
        '--max_queries', '5000',
        '--timeout', '1200'
    ]
    
    return run_command(cmd, f"Phase 2: {experiment}")


def run_phase3(experiment, phase2_pickle):
    
    cmd = [
        'python', 'run_phase3.py',
        '--experiment', experiment,
        '--phase2_pickle', phase2_pickle,
    ]
    
    return run_command(cmd, f"Phase 3: {experiment}")


def main():
    
    
    benchmarks = [
        {
            'name': 'sudoku',
            'phase1_pickle': 'phase1_output/sudoku_phase1.pkl'
        },
        {
            'name': 'sudoku_gt',
            'phase1_pickle': 'phase1_output/sudoku_gt_phase1.pkl'
        },
        {
            'name': 'jsudoku',
            'phase1_pickle': 'phase1_output/jsudoku_phase1.pkl'
        },
        {
            'name': 'latin_square',
            'phase1_pickle': 'phase1_output/latin_square_phase1.pkl'
        },
        {
            'name': 'graph_coloring_register',
            'phase1_pickle': 'phase1_output/graph_coloring_register_phase1.pkl'
        },
        {
            'name': 'examtt_v1',
            'phase1_pickle': 'phase1_output/examtt_v1_phase1.pkl'
        },
        {
            'name': 'examtt_v2',
            'phase1_pickle': 'phase1_output/examtt_v2_phase1.pkl'
        }
    ]
    
    results = []
    
    for benchmark in benchmarks:
        name = benchmark['name']
        phase1_pickle = benchmark['phase1_pickle']


        if not os.path.exists(phase1_pickle):
            print(f"[ERROR] Phase 1 pickle not found: {phase1_pickle}")
            print(f"Please run Phase 1 first: python run_phase1_experiments.py")
            results.append({
                'benchmark': name,
                'phase2_success': False,
                'phase3_success': False,
                'error': 'Phase 1 pickle not found'
            })
            continue

        phase2_success = run_phase2(name, phase1_pickle)
        phase2_pickle = f"phase2_output/{name}_phase2.pkl"
        
        if not phase2_success:
            print(f"[ERROR] Phase 2 failed for {name}")
            results.append({
                'benchmark': name,
                'phase2_success': False,
                'phase3_success': False,
                'error': 'Phase 2 failed'
            })
            continue

        phase3_success = run_phase3(name, phase2_pickle)
        
        if not phase3_success:
            print(f"[ERROR] Phase 3 failed for {name}")
            results.append({
                'benchmark': name,
                'phase2_success': True,
                'phase3_success': False,
                'error': 'Phase 3 failed'
            })
            continue

        results.append({
            'benchmark': name,
            'phase2_success': True,
            'phase3_success': True,
            'error': None
        })
        
        print(f"\n[SUCCESS] Complete pipeline finished for {name}")

    print(f"\n\n{'='*80}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*80}\n")
    
    for r in results:
        status = "[OK]" if r['phase2_success'] and r['phase3_success'] else "[FAIL]"
        error_msg = f" ({r['error']})" if r['error'] else ""
        print(f"{status} {r['benchmark']:<15} Phase2: {r['phase2_success']}, Phase3: {r['phase3_success']}{error_msg}")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'total_benchmarks': len(benchmarks),
        'successful': sum(1 for r in results if r['phase2_success'] and r['phase3_success'])
    }
    
    with open('pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SAVED] Summary saved to: pipeline_summary.json")
    print(f"\nTotal: {summary['successful']}/{summary['total_benchmarks']} benchmarks completed successfully")


if __name__ == "__main__":
    main()

