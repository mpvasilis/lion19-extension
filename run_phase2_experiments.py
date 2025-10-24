"""
Run Phase 2 for multiple benchmark variants with comprehensive logging.

This script runs Phase 2 COP-based refinement for all Phase 1 outputs:
1. Regular Sudoku
2. Greater-Than Sudoku
3. Exam Timetabling Variant 1
4. Exam Timetabling Variant 2

All output is logged to files for later analysis.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path


def run_phase2_for_benchmark(benchmark_name, phase1_pickle_path, output_dir, log_file):
    """
    Run Phase 2 for a single benchmark and log all output.
    
    Args:
        benchmark_name: Name of benchmark
        phase1_pickle_path: Path to Phase 1 pickle file
        output_dir: Directory for output files
        log_file: Path to log file for this run
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Running Phase 2 for: {benchmark_name}")
    print(f"{'='*80}")
    print(f"Phase 1 pickle: {phase1_pickle_path}")
    print(f"Log file: {log_file}")
    
    start_time = time.time()
    
    # Construct command
    cmd = [
        'python',
        'main_alldiff_cop.py',
        '--experiment', benchmark_name,
        '--phase1_pickle', phase1_pickle_path,
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run command and capture output
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write(f"PHASE 2 EXPERIMENT LOG\n")
            f.write(f"Benchmark: {benchmark_name}\n")
            f.write(f"Phase 1 pickle: {phase1_pickle_path}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            f.flush()
            
            # Run subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output to both console and file
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                    output_lines.append(line.strip())
            
            process.wait()
            return_code = process.returncode
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse output for key metrics
        results = parse_phase2_output(output_lines, benchmark_name)
        results['return_code'] = return_code
        results['duration'] = duration
        results['log_file'] = log_file
        
        # Append summary to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Status: {'SUCCESS' if return_code == 0 else 'FAILED'}\n")
            f.write("="*80 + "\n")
        
        if return_code == 0:
            print(f"\n[SUCCESS] {benchmark_name} completed in {duration:.2f}s")
            return results
        else:
            print(f"\n[FAILED] {benchmark_name} returned error code {return_code}")
            results['status'] = 'FAILED'
            return results
    
    except Exception as e:
        print(f"\n[ERROR] Exception running {benchmark_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'benchmark': benchmark_name,
            'status': 'ERROR',
            'error': str(e),
            'duration': duration,
            'log_file': log_file
        }


def parse_phase2_output(output_lines, benchmark_name):
    """
    Parse Phase 2 output to extract key metrics.
    
    Args:
        output_lines: List of output lines
        benchmark_name: Name of benchmark
        
    Returns:
        Dictionary with parsed metrics
    """
    results = {
        'benchmark': benchmark_name,
        'status': 'UNKNOWN',
        'queries': None,
        'time': None,
        'validated': None,
        'rejected': None,
        'target_count': None,
        'learned_count': None,
        'correct': None,
        'missing': None,
        'spurious': None,
        'precision': None,
        'recall': None
    }
    
    # Parse output lines
    for line in output_lines:
        if 'Total queries:' in line:
            try:
                results['queries'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Total time:' in line:
            try:
                time_str = line.split(':')[1].strip().replace('s', '')
                results['time'] = float(time_str)
            except:
                pass
        
        elif 'Validated constraints:' in line:
            try:
                results['validated'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Rejected constraints:' in line:
            try:
                results['rejected'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Target AllDifferent constraints:' in line:
            try:
                results['target_count'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif 'Learned AllDifferent constraints:' in line:
            try:
                results['learned_count'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif line.startswith('Correct:'):
            try:
                results['correct'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif line.startswith('Missing:'):
            try:
                results['missing'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif line.startswith('Spurious:'):
            try:
                results['spurious'] = int(line.split(':')[1].strip())
            except:
                pass
        
        elif '[SUCCESS] Perfect learning!' in line:
            results['status'] = 'SUCCESS'
    
    # Calculate precision and recall
    if results['learned_count'] is not None and results['learned_count'] > 0:
        if results['correct'] is not None:
            results['precision'] = results['correct'] / results['learned_count']
    
    if results['target_count'] is not None and results['target_count'] > 0:
        if results['correct'] is not None:
            results['recall'] = results['correct'] / results['target_count']
    
    # Determine status if not already set
    if results['status'] == 'UNKNOWN':
        if results['correct'] == results['target_count'] and results['spurious'] == 0:
            results['status'] = 'SUCCESS'
        elif results['queries'] is not None:
            results['status'] = 'COMPLETED'
        else:
            results['status'] = 'FAILED'
    
    return results


def create_summary_report(all_results, output_dir):
    """
    Create a comprehensive summary report of all Phase 2 experiments.
    
    Args:
        all_results: List of result dictionaries
        output_dir: Output directory for report
    """
    summary_file = os.path.join(output_dir, 'phase2_summary.txt')
    json_file = os.path.join(output_dir, 'phase2_results.json')
    
    # Write text summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2 EXPERIMENTS - COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total benchmarks: {len(all_results)}\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        success_count = sum(1 for r in all_results if r['status'] == 'SUCCESS')
        completed_count = sum(1 for r in all_results if r['status'] in ['SUCCESS', 'COMPLETED'])
        failed_count = len(all_results) - completed_count
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Successful (Perfect Learning): {success_count}\n")
        f.write(f"Completed (Partial Learning): {completed_count - success_count}\n")
        f.write(f"Failed/Error: {failed_count}\n")
        f.write("\n")
        
        # Per-benchmark results
        f.write("PER-BENCHMARK RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for result in all_results:
            f.write(f"Benchmark: {result['benchmark']}\n")
            f.write(f"  Status: {result['status']}\n")
            
            if result.get('queries') is not None:
                f.write(f"  Queries: {result['queries']}\n")
            if result.get('time') is not None:
                f.write(f"  Time: {result['time']:.2f}s\n")
            if result.get('duration') is not None:
                f.write(f"  Total Duration: {result['duration']:.2f}s\n")
            
            f.write(f"\n  Constraint Learning:\n")
            if result.get('target_count') is not None:
                f.write(f"    Target constraints: {result['target_count']}\n")
            if result.get('learned_count') is not None:
                f.write(f"    Learned constraints: {result['learned_count']}\n")
            if result.get('validated') is not None:
                f.write(f"    Validated: {result['validated']}\n")
            if result.get('rejected') is not None:
                f.write(f"    Rejected: {result['rejected']}\n")
            
            f.write(f"\n  Accuracy:\n")
            if result.get('correct') is not None:
                f.write(f"    Correct: {result['correct']}\n")
            if result.get('missing') is not None:
                f.write(f"    Missing: {result['missing']}\n")
            if result.get('spurious') is not None:
                f.write(f"    Spurious: {result['spurious']}\n")
            if result.get('precision') is not None:
                f.write(f"    Precision: {result['precision']:.2%}\n")
            if result.get('recall') is not None:
                f.write(f"    Recall: {result['recall']:.2%}\n")
            
            if result.get('log_file'):
                f.write(f"\n  Log file: {result['log_file']}\n")
            
            if result.get('error'):
                f.write(f"\n  Error: {result['error']}\n")
            
            f.write("\n" + "-"*80 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Benchmark':<20} {'Status':<12} {'Queries':<10} {'Time(s)':<10} {'Precision':<12} {'Recall':<10}\n")
        f.write("-"*80 + "\n")
        
        for result in all_results:
            queries_str = str(result.get('queries', 'N/A'))
            time_str = f"{result.get('time', 0):.1f}" if result.get('time') else 'N/A'
            prec_str = f"{result.get('precision', 0):.2%}" if result.get('precision') is not None else 'N/A'
            rec_str = f"{result.get('recall', 0):.2%}" if result.get('recall') is not None else 'N/A'
            
            f.write(f"{result['benchmark']:<20} {result['status']:<12} {queries_str:<10} {time_str:<10} {prec_str:<12} {rec_str:<10}\n")
        
        f.write("="*80 + "\n")
    
    # Write JSON results
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSummary report saved to: {summary_file}")
    print(f"JSON results saved to: {json_file}")
    
    return summary_file, json_file


def main():
    """Run Phase 2 for all benchmarks."""
    
    benchmarks = [
        {
            'name': 'sudoku',
            'pickle': 'phase1_output/sudoku_phase1.pkl',
            'description': 'Regular 9x9 Sudoku'
        },
        {
            'name': 'sudoku_gt',
            'pickle': 'phase1_output/sudoku_gt_phase1.pkl',
            'description': 'Sudoku with Greater-Than Constraints'
        },
        {
            'name': 'examtt_v1',
            'pickle': 'phase1_output/examtt_v1_phase1.pkl',
            'description': 'Exam Timetabling Variant 1 (Small)'
        },
        {
            'name': 'examtt_v2',
            'pickle': 'phase1_output/examtt_v2_phase1.pkl',
            'description': 'Exam Timetabling Variant 2 (Large)'
        }
    ]
    
    output_dir = 'phase2_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("PHASE 2 BATCH RUNNER")
    print("="*80)
    print(f"\nRunning Phase 2 for {len(benchmarks)} benchmarks:")
    for i, bench in enumerate(benchmarks, 1):
        print(f"  {i}. {bench['name']}: {bench['description']}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)
    
    all_results = []
    
    for i, bench in enumerate(benchmarks, 1):
        print(f"\n\n{'='*80}")
        print(f"BENCHMARK {i}/{len(benchmarks)}: {bench['name']}")
        print(f"Description: {bench['description']}")
        print(f"{'='*80}\n")
        
        # Check if Phase 1 pickle exists
        if not os.path.exists(bench['pickle']):
            print(f"[ERROR] Phase 1 pickle not found: {bench['pickle']}")
            all_results.append({
                'benchmark': bench['name'],
                'status': 'ERROR',
                'error': f"Phase 1 pickle not found: {bench['pickle']}"
            })
            continue
        
        # Create log file path
        log_file = os.path.join(output_dir, f"{bench['name']}_phase2.log")
        
        # Run Phase 2
        result = run_phase2_for_benchmark(
            benchmark_name=bench['name'],
            phase1_pickle_path=bench['pickle'],
            output_dir=output_dir,
            log_file=log_file
        )
        
        all_results.append(result)
    
    # Create summary report
    print("\n\n" + "="*80)
    print("CREATING SUMMARY REPORT")
    print("="*80)
    
    summary_file, json_file = create_summary_report(all_results, output_dir)
    
    # Print final summary to console
    print("\n" + "="*80)
    print("PHASE 2 BATCH RUNNER - FINAL SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in all_results if r['status'] == 'SUCCESS')
    completed_count = sum(1 for r in all_results if r['status'] in ['SUCCESS', 'COMPLETED'])
    failed_count = len(all_results) - completed_count
    
    print(f"\nTotal benchmarks: {len(all_results)}")
    print(f"  [+] Successful (Perfect Learning): {success_count}")
    print(f"  [~] Completed (Partial Learning): {completed_count - success_count}")
    print(f"  [-] Failed/Error: {failed_count}")
    
    print("\nDetailed results:")
    for result in all_results:
        status_symbol = "[+]" if result['status'] == 'SUCCESS' else "[-]" if result['status'] in ['FAILED', 'ERROR'] else "[~]"
        print(f"  {status_symbol} {result['benchmark']}: {result['status']}")
        if result.get('queries'):
            print(f"      Queries: {result['queries']}")
        if result.get('precision') is not None and result.get('recall') is not None:
            print(f"      Precision: {result['precision']:.2%}, Recall: {result['recall']:.2%}")
    
    print(f"\nFull summary: {summary_file}")
    print(f"JSON results: {json_file}")
    print("="*80)
    
    # Return success if all completed
    if failed_count == 0:
        print("\nALL BENCHMARKS COMPLETED!")
        return 0
    else:
        print(f"\n{failed_count} BENCHMARK(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

