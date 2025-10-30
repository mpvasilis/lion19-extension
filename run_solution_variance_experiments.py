"""
Solution Variance Experiments
==============================
This script runs experiments with varying numbers of solutions and inversely 
proportional mock constraints. As the number of solutions increases, the number 
of mock constraints decreases.

Configurations:
- 2 solutions  -> 50 mock constraints
- 5 solutions  -> 20 mock constraints  
- 10 solutions -> 10 mock constraints
- 50 solutions -> 2 mock constraints
"""

import os
import sys
import json
import time
import pickle
import subprocess
from datetime import datetime

# Benchmark display names for formatted output
BENCHMARK_DISPLAY_NAMES = {
    'sudoku': 'Sudoku',
    'sudoku_gt': 'Sudoku-GT',
    'jsudoku': 'JSudoku',
    'latin_square': 'Latin Square',
    'graph_coloring_register': 'Graph Coloring',
    'examtt_v1': 'ExamTT-V1',
    'examtt_v2': 'ExamTT-V2',
    'nurse': 'Nurse'
}

# Target constraint counts for each benchmark
TARGET_CONSTRAINTS = {
    'sudoku': 27,
    'sudoku_gt': 37,
    'jsudoku': 31,
    'latin_square': 18,
    'graph_coloring_register': 5,
    'examtt_v1': 7,
    'examtt_v2': 9,
    'nurse': 13  # Adjust based on actual
}


def run_command(cmd, description):
    """Run a command and capture output."""
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
        return False, None
    
    return True, result.stdout


def run_phase1_with_timing(experiment, num_examples, num_overfitted, output_dir='solution_variance_output'):
    """Run Phase 1 with timing and return the time taken."""
    
    # Create custom output directory for this configuration
    config_output_dir = f"{output_dir}/{experiment}_sol{num_examples}_mock{num_overfitted}"
    os.makedirs(config_output_dir, exist_ok=True)
    
    cmd = [
        'python', 'phase1_passive_learning.py',
        '--benchmark', experiment,
        '--output_dir', config_output_dir,
        '--num_examples', str(num_examples),
        '--num_overfitted', str(num_overfitted)
    ]
    
    start_time = time.time()
    success, output = run_command(cmd, 
        f"Phase 1: {experiment} | Solutions={num_examples}, Mocks={num_overfitted}")
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    if success:
        print(f"\n[TIMING] Phase 1 completed in {elapsed_time:.2f} seconds")
        phase1_pickle = f"{config_output_dir}/{experiment}_phase1.pkl"
        return True, phase1_pickle, elapsed_time
    else:
        print(f"\n[ERROR] Phase 1 failed after {elapsed_time:.2f} seconds")
        return False, None, elapsed_time


def run_phase2(experiment, phase1_pickle, max_queries=500, timeout=600):
    """Run Phase 2 with the given Phase 1 pickle."""
    
    cmd = [
        'python', 'main_alldiff_cop.py',
        '--experiment', experiment,
        '--phase1_pickle', phase1_pickle,
        '--max_queries', str(max_queries),
        '--timeout', str(timeout)
    ]
    
    success, _ = run_command(cmd, f"Phase 2: {experiment}")
    
    if success:
        # Derive phase2 pickle path from phase1 pickle path
        phase1_dir = os.path.dirname(phase1_pickle)
        phase2_pickle = phase1_pickle.replace('_phase1.pkl', '_phase2.pkl')
        # Phase 2 outputs to phase2_output by default, need to adjust
        phase2_pickle = f"phase2_output/{experiment}_phase2.pkl"
        return True, phase2_pickle
    else:
        return False, None


def run_phase3(experiment, phase2_pickle):
    """Run Phase 3 with the given Phase 2 pickle."""
    
    cmd = [
        'python', 'run_phase3.py',
        '--experiment', experiment,
        '--phase2_pickle', phase2_pickle,
    ]
    
    success, _ = run_command(cmd, f"Phase 3: {experiment}")
    return success


def load_phase1_pickle(pickle_path):
    """Load Phase 1 pickle data."""
    if not os.path.exists(pickle_path):
        return None
    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load Phase 1 pickle: {e}")
        return None


def load_phase3_results(benchmark_name):
    """Load Phase 3 JSON results."""
    json_path = f"phase3_output/{benchmark_name}_phase3_results.json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load Phase 3 results: {e}")
        return None


def extract_metrics(benchmark_name, num_solutions, phase1_pickle_path, phase1_time):
    """Extract all metrics from phase outputs for the results table."""
    
    # Load Phase 1 data
    phase1_data = load_phase1_pickle(phase1_pickle_path)
    if phase1_data is None:
        return None
    
    # Load Phase 3 results
    phase3_results = load_phase3_results(benchmark_name)
    if phase3_results is None:
        return None
    
    metrics = {}
    
    # Problem name
    metrics['Prob.'] = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)
    
    # Number of solutions
    metrics['Sols'] = num_solutions
    
    # StartC: Starting candidate constraints from Phase 1
    StartC = len(phase1_data.get('CG', []))
    metrics['StartC'] = StartC
    
    # CT: Target constraint count
    metrics['CT'] = TARGET_CONSTRAINTS.get(benchmark_name, 'N/A')
    
    # Bias: Size of generated bias
    metrics['Bias'] = phase3_results.get('phase1', {}).get('B_fixed_size', 0)
    
    # ViolQ: Violation queries from Phase 2
    metrics['ViolQ'] = phase3_results.get('phase2', {}).get('queries', 0)
    
    # InvC: Invalid constraints (StartC - validated)
    validated_count = phase3_results.get('phase2', {}).get('validated_globals', 0)
    metrics['InvC'] = StartC - validated_count
    
    # MQuQ: MQuAcq queries from Phase 3
    metrics['MQuQ'] = phase3_results.get('phase3', {}).get('queries', 0)
    
    # TQ: Total queries
    metrics['TQ'] = metrics['ViolQ'] + metrics['MQuQ']
    
    # Phase 1 Time (passive learning time)
    metrics['P1T(s)'] = round(phase1_time, 2)
    
    # VT(s): Phase 2 violation time
    metrics['VT(s)'] = round(phase3_results.get('phase2', {}).get('time', 0), 2)
    
    # MQuT(s): Phase 3 MQuAcq time
    metrics['MQuT(s)'] = round(phase3_results.get('phase3', {}).get('time', 0), 2)
    
    # TT(s): Total time (P1T + VT + MQuT)
    metrics['TT(s)'] = round(metrics['P1T(s)'] + metrics['VT(s)'] + metrics['MQuT(s)'], 2)
    
    # Baseline metrics (not applicable for this experiment)
    metrics['ALQ'] = 'N/A'
    metrics['PAQ'] = 'N/A'
    metrics['ALT(s)'] = 'N/A'
    metrics['PAT(s)'] = 'N/A'
    
    # Evaluation metrics
    eval_data = phase3_results.get('evaluation', {})
    constraint_level = eval_data.get('constraint_level', {})
    solution_level = eval_data.get('solution_level', {})
    
    metrics['precision'] = round(constraint_level.get('precision', 0) * 100, 2)
    metrics['recall'] = round(constraint_level.get('recall', 0) * 100, 2)
    metrics['s_precision'] = round(solution_level.get('s_precision', 0) * 100, 2)
    metrics['s_recall'] = round(solution_level.get('s_recall', 0) * 100, 2)
    
    return metrics


def calculate_mock_constraints(num_solutions):
    """
    Calculate the number of mock constraints inversely proportional to solutions.
    Formula: More solutions -> Fewer mock constraints
    """
    # Inverse relationship mapping
    mapping = {
        2: 50,
        5: 20,
        10: 10,
        50: 2
    }
    
    return mapping.get(num_solutions, 10)  # Default to 10 if not in mapping


def main():
    """Run the complete solution variance experiment."""
    
    print(f"\n{'='*80}")
    print(f"SOLUTION VARIANCE EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Solutions: [2, 5, 10, 50]")
    print(f"  Mock Constraints (inverse): [50, 20, 10, 2]")
    print(f"{'='*80}\n")
    
    # Define benchmarks to test
    benchmarks = [
        'sudoku',
        'sudoku_gt',
        # 'jsudoku',
        'latin_square',
        'graph_coloring_register',
        'examtt_v1',
        'examtt_v2',
        'nurse'
    ]
    
    # Define solution variance configurations
    solution_configs = [2, 5, 10, 50]
    
    # Results storage
    all_results = []
    
    # Storage for collected metrics
    all_metrics = []
    
    # Run experiments for each benchmark and configuration
    for benchmark in benchmarks:
        print(f"\n\n{'='*80}")
        print(f"BENCHMARK: {benchmark}")
        print(f"{'='*80}\n")
        
        benchmark_results = {
            'benchmark': benchmark,
            'configurations': []
        }
        
        for num_solutions in solution_configs:
            num_mocks = calculate_mock_constraints(num_solutions)
            
            print(f"\n{'-'*80}")
            print(f"Configuration: {num_solutions} solutions, {num_mocks} mock constraints")
            print(f"{'-'*80}\n")
            
            config_result = {
                'num_solutions': num_solutions,
                'num_mock_constraints': num_mocks,
                'phase1_success': False,
                'phase1_time': None,
                'phase1_pickle': None,
                'phase2_success': False,
                'phase3_success': False,
                'error': None
            }
            
            # Run Phase 1 with timing
            phase1_success, phase1_pickle, phase1_time = run_phase1_with_timing(
                benchmark, num_solutions, num_mocks
            )
            
            config_result['phase1_success'] = phase1_success
            config_result['phase1_time'] = phase1_time
            config_result['phase1_pickle'] = phase1_pickle
            
            if not phase1_success:
                config_result['error'] = 'Phase 1 failed'
                benchmark_results['configurations'].append(config_result)
                continue
            
            # Run Phase 2
            phase2_success, phase2_pickle = run_phase2(benchmark, phase1_pickle)
            config_result['phase2_success'] = phase2_success
            
            if not phase2_success:
                config_result['error'] = 'Phase 2 failed'
                benchmark_results['configurations'].append(config_result)
                continue
            
            # Run Phase 3
            phase3_success = run_phase3(benchmark, phase2_pickle)
            config_result['phase3_success'] = phase3_success
            
            if not phase3_success:
                config_result['error'] = 'Phase 3 failed'
                benchmark_results['configurations'].append(config_result)
                continue
            
            # Extract metrics if all phases succeeded
            metrics = extract_metrics(benchmark, num_solutions, phase1_pickle, phase1_time)
            if metrics:
                all_metrics.append(metrics)
                print(f"\n[METRICS] Successfully extracted metrics for {benchmark} with {num_solutions} solutions")
            else:
                print(f"\n[WARNING] Could not extract metrics for {benchmark} with {num_solutions} solutions")
            
            benchmark_results['configurations'].append(config_result)
            
            print(f"\n[SUMMARY] Configuration complete:")
            print(f"  Solutions: {num_solutions}, Mocks: {num_mocks}")
            print(f"  Phase 1: {'✓' if phase1_success else '✗'} ({phase1_time:.2f}s)")
            print(f"  Phase 2: {'✓' if phase2_success else '✗'}")
            print(f"  Phase 3: {'✓' if phase3_success else '✗'}")
        
        all_results.append(benchmark_results)
    
    # Generate summary report
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    output_dir = 'solution_variance_output'
    os.makedirs(output_dir, exist_ok=True)
    
    if not all_metrics:
        print("[WARNING] No metrics collected. Check for errors in pipeline execution.")
    
    # Generate formatted text report (matching HCAR format)
    report_path = f"{output_dir}/variance_results.txt"
    with open(report_path, 'w') as f:
        f.write("Solution Variance Experiment Results\n")
        f.write("="*130 + "\n\n")
        
        # Header line
        f.write(f"{'Prob.':<15} {'Sols':<6} {'StartC':<8} {'InvC':<6} {'CT':<5} {'Bias':<6} ")
        f.write(f"{'ViolQ':<7} {'MQuQ':<7} {'TQ':<6} {'ALQ':<5} {'PAQ':<5} ")
        f.write(f"{'P1T(s)':<8} {'VT(s)':<8} {'MQuT(s)':<9} {'TT(s)':<8} {'ALT(s)':<7} {'PAT(s)':<7}\n")
        
        # Data rows
        for m in all_metrics:
            f.write(f"{m['Prob.']:<15} {m['Sols']:<6} {m['StartC']:<8} {m['InvC']:<6} ")
            f.write(f"{str(m['CT']):<5} {m['Bias']:<6} ")
            f.write(f"{m['ViolQ']:<7} {m['MQuQ']:<7} {m['TQ']:<6} {m['ALQ']:<5} {m['PAQ']:<5} ")
            f.write(f"{m['P1T(s)']:<8} {m['VT(s)']:<8} {m['MQuT(s)']:<9} {m['TT(s)']:<8} ")
            f.write(f"{m['ALT(s)']:<7} {m['PAT(s)']:<7}\n")
        
        f.write("\n" + "="*130 + "\n")
        f.write("Legend:\n")
        f.write("  Sols: Number of given solutions (positive examples)\n")
        f.write("  StartC: Number of candidate constraints from passive learning\n")
        f.write("  InvC: Number of constraints invalidated by refinement\n")
        f.write("  CT: Number of AllDifferent constraints in target model\n")
        f.write("  Bias: Size of generated bias\n")
        f.write("  ViolQ: Violation queries (Phase 2)\n")
        f.write("  MQuQ: Active learning queries (Phase 3)\n")
        f.write("  TQ: Total queries (ViolQ + MQuQ)\n")
        f.write("  ALQ: Queries for purely Active Learning baseline\n")
        f.write("  PAQ: Queries for Passive+Active baseline\n")
        f.write("  P1T(s): Phase 1 passive learning time\n")
        f.write("  VT(s): Duration of violation phase\n")
        f.write("  MQuT(s): Duration of active learning phase\n")
        f.write("  TT(s): Overall runtime (P1T + VT + MQuT)\n")
        f.write("  ALT(s): Runtime for Active Learning baseline\n")
        f.write("  PAT(s): Runtime for Passive+Active baseline\n")
    
    print(f"[SAVED] Formatted results saved to: {report_path}")
    
    # Print to console
    print(f"\n{'Prob.':<15} {'Sols':<6} {'StartC':<8} {'InvC':<6} {'CT':<5} {'Bias':<6} ")
    print(f"{'ViolQ':<7} {'MQuQ':<7} {'TQ':<6} {'P1T(s)':<8} {'VT(s)':<8} {'MQuT(s)':<9} {'TT(s)':<8}")
    print("="*110)
    for m in all_metrics:
        print(f"{m['Prob.']:<15} {m['Sols']:<6} {m['StartC']:<8} {m['InvC']:<6} ", end="")
        print(f"{str(m['CT']):<5} {m['Bias']:<6} ", end="")
        print(f"{m['ViolQ']:<7} {m['MQuQ']:<7} {m['TQ']:<6} ", end="")
        print(f"{m['P1T(s)']:<8} {m['VT(s)']:<8} {m['MQuT(s)']:<9} {m['TT(s)']:<8}")
    
    # Generate CSV output
    csv_path = f"{output_dir}/variance_results.csv"
    with open(csv_path, 'w') as f:
        # Header
        f.write("Prob.,Sols,StartC,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,")
        f.write("P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),")
        f.write("precision,recall,s_precision,s_recall\n")
        
        # Data
        for m in all_metrics:
            f.write(f"{m['Prob.']},{m['Sols']},{m['StartC']},{m['InvC']},")
            f.write(f"{m['CT']},{m['Bias']},{m['ViolQ']},{m['MQuQ']},{m['TQ']},")
            f.write(f"{m['ALQ']},{m['PAQ']},")
            f.write(f"{m['P1T(s)']},{m['VT(s)']},{m['MQuT(s)']},{m['TT(s)']},")
            f.write(f"{m['ALT(s)']},{m['PAT(s)']},")
            f.write(f"{m['precision']},{m['recall']},{m['s_precision']},{m['s_recall']}\n")
    
    print(f"[SAVED] CSV results saved to: {csv_path}")
    
    # Save detailed JSON
    json_path = f"{output_dir}/variance_experiment_detailed.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'metrics': all_metrics,
        'raw_results': all_results,
        'total_benchmarks': len(benchmarks),
        'solution_configurations': solution_configs,
        'mock_constraints_mapping': {
            str(sol): calculate_mock_constraints(sol) for sol in solution_configs
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[SAVED] Detailed JSON saved to: {json_path}")
    
    # Calculate statistics
    total_configs = sum(len(br['configurations']) for br in all_results)
    successful_configs = len(all_metrics)
    
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total configurations tested: {total_configs}")
    print(f"Successful completions: {successful_configs}/{total_configs}")
    if total_configs > 0:
        print(f"Success rate: {100*successful_configs/total_configs:.1f}%")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

