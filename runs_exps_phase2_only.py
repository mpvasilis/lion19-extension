"""
Solution Variance Experiments - COP vs LION Comparison (Phase 2 Only)
=======================================================================
This script runs experiments with varying numbers of solutions and inversely 
proportional overfitted constraints, comparing both COP and LION approaches.
As the number of solutions increases, the number of overfitted constraints decreases.

This version runs experiments only through Phase 2 (skips Phase 3).

Configurations:
Per-benchmark overfitted constraints (InvC) used during passive learning:
  - S9x9: 2→20, 5→8, 10→6, 50→3
  - GTS: 2→17, 5→8, 20→5, 200→1
  - JSud: 2→41, 20→41, 200→32, 500→25
  - ET1: 2→25, 5→20, 10→11, 50→5
  - ET2: 2→44, 5→28, 10→9, 50→4
Benchmarks without explicit mappings fall back to the default inverse relationship
described in `DEFAULT_OVERFITTED_CONSTRAINTS`.

Each configuration is tested with both:
- COP approach (main_alldiff_cop.py)
- LION approach (main_alldiff_lion19.py)
"""

import os
import sys
import json
import time
import pickle
import shutil
import subprocess
import statistics
from datetime import datetime
from threading import Lock

PYTHON_EXECUTABLE = sys.executable or 'python3'


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


TARGET_CONSTRAINTS = {
    'sudoku': 27,
    'sudoku_gt': 37,
    'jsudoku': 27,
    'latin_square': 18,
    'graph_coloring_register': 5,
    'examtt_v1': 7,
    'examtt_v2': 9,
    'nurse': 13  
}




BENCHMARK_OVERFITTED_CONSTRAINTS = {
    'sudoku': {
        2: 20,
        5: 8,
        10: 6,
        50: 3,
    },
    'sudoku_gt': {
        2: 17,
        5: 8,
        20: 5,
        200: 1,
    },
    'jsudoku': {
        2: 15,
        20: 10,
        200: 7,
        500: 2,
    },
    'examtt_v1': {
        2: 25,
        5: 20,
        10: 11,
        50: 5,
    },
    'examtt_v2': {
        2: 44,
        5: 28,
        10: 9,
        50: 4,
    },
}


DEFAULT_OVERFITTED_CONSTRAINTS = {
    2: 50,
    5: 20,
    10: 10,
    50: 2,
}

DEFAULT_OVERFITTED_VALUE = 10

DEFAULT_SOLUTION_CONFIGS = sorted(DEFAULT_OVERFITTED_CONSTRAINTS.keys())


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
    """Run Phase 1 with timing and return the time taken. Skip if pickle already exists."""
    

    config_output_dir = f"{output_dir}/{experiment}_sol{num_examples}_overfitted{num_overfitted}"
    os.makedirs(config_output_dir, exist_ok=True)
    

    phase1_pickle = f"{config_output_dir}/{experiment}_phase1.pkl"
    
    if os.path.exists(phase1_pickle):

        try:
            with open(phase1_pickle, 'rb') as f:
                phase1_data = pickle.load(f)
            

            phase1_time = 0.0
            if isinstance(phase1_data, dict) and 'metadata' in phase1_data:
                phase1_time = phase1_data['metadata'].get('phase1_time', 0.0)
            
            print(f"\n{'='*80}")
            print(f"[SKIP] Phase 1 pickle already exists: {phase1_pickle}")
            print(f"[SKIP] Reusing existing Phase 1 results")
            if phase1_time > 0:
                print(f"[SKIP] Phase 1 time from cache: {phase1_time:.2f} seconds")
            else:
                print(f"[SKIP] Phase 1 time not recorded in cache (using 0.0)")
            print(f"{'='*80}\n")
            return True, phase1_pickle, phase1_time
        except Exception as e:
            print(f"\n[WARNING] Existing Phase 1 pickle is corrupted: {e}")
            print(f"[WARNING] Re-running Phase 1...")
    
    cmd = [
        PYTHON_EXECUTABLE, 'phase1_passive_learning.py',
        '--benchmark', experiment,
        '--output_dir', config_output_dir,
        '--num_examples', str(num_examples),
        '--num_overfitted', str(num_overfitted)
    ]
    
    start_time = time.time()
    success, output = run_command(cmd, 
        f"Phase 1: {experiment} | Solutions={num_examples}, overfitted={num_overfitted}")
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    if success:

        try:
            if os.path.exists(phase1_pickle):
                with open(phase1_pickle, 'rb') as f:
                    phase1_data = pickle.load(f)
                

                if isinstance(phase1_data, dict):
                    if 'metadata' not in phase1_data:
                        phase1_data['metadata'] = {}
                    phase1_data['metadata']['phase1_time'] = elapsed_time
                    

                    with open(phase1_pickle, 'wb') as f:
                        pickle.dump(phase1_data, f)
                    print(f"[INFO] Stored Phase 1 execution time ({elapsed_time:.2f}s) in pickle metadata")
        except Exception as e:
            print(f"[WARNING] Could not store Phase 1 time in pickle metadata: {e}")
        
        print(f"\n[TIMING] Phase 1 completed in {elapsed_time:.2f} seconds")
        return True, phase1_pickle, elapsed_time
    else:
        print(f"\n[ERROR] Phase 1 failed after {elapsed_time:.2f} seconds")
        return False, None, elapsed_time


def run_phase2(
    experiment,
    phase1_pickle,
    *,
    approach='cop',
    max_queries=5000,
    timeout=1200,
    config_tag=None,
):
    """Run Phase 2 with the given Phase 1 pickle using specified approach (cop or lion)."""
    

    script_map = {
        'cop': 'main_alldiff_cop.py',
        'lion': 'main_alldiff_lion19.py'
    }
    
    script = script_map.get(approach.lower(), 'main_alldiff_cop.py')
    



    filename_map = {
        'cop': f"{experiment}_phase2.pkl",
        'lion': f"{experiment}_lion19_phase2.pkl"
    }
    
    cmd = [
        PYTHON_EXECUTABLE, script,
        '--experiment', experiment,
        '--phase1_pickle', phase1_pickle,
        '--max_queries', str(max_queries),
        '--timeout', str(timeout)
    ]
    
    success, _ = run_command(cmd, f"Phase 2 ({approach.upper()}): {experiment}")
    
    if success:

        default_output = "phase2_output"
        source_pickle = os.path.join(default_output, filename_map[approach.lower()])


        base_output_dir = f"phase2_output_{approach.lower()}"
        if config_tag:
            target_dir = os.path.join(base_output_dir, experiment)
            file_suffix_map = {
                'cop': 'phase2.pkl',
                'lion': 'lion19_phase2.pkl'
            }
            dest_filename = f"{experiment}_{config_tag}_{file_suffix_map[approach.lower()]}"
        else:
            target_dir = base_output_dir
            dest_filename = filename_map[approach.lower()]

        os.makedirs(target_dir, exist_ok=True)
        target_pickle = os.path.join(target_dir, dest_filename)


        if os.path.exists(source_pickle):
            shutil.move(source_pickle, target_pickle)
            print(f"\n[INFO] Moved {source_pickle} to {target_pickle}")
            return True, target_pickle
        else:
            print(f"\n[WARNING] Expected output file not found: {source_pickle}")
            return True, source_pickle
    else:
        return False, None


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


def load_phase2_pickle(path):
    """Load Phase 2 pickle with error handling."""
    if not path or not os.path.exists(path):
        print(f"[WARNING] Phase 2 pickle not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load Phase 2 pickle ({path}): {e}")
        return None


def extract_metrics(
    benchmark_name,
    num_solutions,
    phase1_pickle_path,
    phase1_time,
    *,
    approach='cop',
    config_tag=None,
    phase2_pickle_path=None,
):
    """Extract all metrics from phase outputs for the results table (Phase 2 only)."""
    

    phase1_data = load_phase1_pickle(phase1_pickle_path)
    if phase1_data is None:
        return None
    

    phase2_data = load_phase2_pickle(phase2_pickle_path) if phase2_pickle_path else None
    if phase2_data is None:
        print(f"[WARNING] Skipping metrics extraction due to missing Phase 2 data: {phase2_pickle_path}")
        return None
    
    phase2_stats = phase2_data.get('phase2_stats', {})
    validated_globals = phase2_stats.get('validated', len(phase2_data.get('C_validated', [])))
    cp_implication = phase2_stats.get('cp_implication', {}) if isinstance(phase2_stats, dict) else {}
    implied_constraints = None
    not_implied_constraints = None
    if isinstance(cp_implication, dict):
        implied_constraints = cp_implication.get('implied_count')
        not_implied_constraints = cp_implication.get('not_implied_count')
    
    metrics = {}
    

    metrics['Prob.'] = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)
    

    metrics['Approach'] = approach.upper()
    

    metrics['Sols'] = num_solutions
    

    StartC = len(phase1_data.get('CG', []))
    metrics['StartC'] = StartC
    

    metrics['Implied'] = implied_constraints if implied_constraints is not None else 'N/A'
    metrics['NotImplied'] = not_implied_constraints if not_implied_constraints is not None else 'N/A'
    

    metrics['CT'] = TARGET_CONSTRAINTS.get(benchmark_name, 'N/A')
    

    raw_bias = len(phase2_data.get('B_fixed', []))
    metrics['Bias'] = raw_bias
    

    metrics['ViolQ'] = phase2_stats.get('queries', 0)
    

    validated_count = validated_globals
    metrics['InvC'] = StartC - validated_count
    

    metrics['MQuQ'] = 0
    

    metrics['TQ'] = metrics['ViolQ'] + metrics['MQuQ']
    

    metrics['P1T(s)'] = round(phase1_time, 2)
    

    phase2_time = phase2_stats.get('time', 0)
    metrics['VT(s)'] = round(phase2_time, 2)
    

    metrics['MQuT(s)'] = 0.0
    

    metrics['TT(s)'] = round(metrics['P1T(s)'] + metrics['VT(s)'] + metrics['MQuT(s)'], 2)
    

    metrics['ALQ'] = 'N/A'
    metrics['PAQ'] = 'N/A'
    metrics['ALT(s)'] = 'N/A'
    metrics['PAT(s)'] = 'N/A'
    

    metrics['precision'] = 0.0
    metrics['recall'] = 0.0
    metrics['s_precision'] = 0.0
    metrics['s_recall'] = 0.0
    
    return metrics


def get_solution_counts_for_benchmark(benchmark):
    """Return the list of solution counts to use for a benchmark."""
    benchmark_mapping = BENCHMARK_OVERFITTED_CONSTRAINTS.get(benchmark)
    if benchmark_mapping:
        return sorted(benchmark_mapping.keys())
    return DEFAULT_SOLUTION_CONFIGS


def calculate_overfitted_constraints(benchmark, num_solutions):
    """Return the number of overfitted constraints (InvC) for Phase 1."""
    benchmark_mapping = BENCHMARK_OVERFITTED_CONSTRAINTS.get(benchmark, {})
    if num_solutions in benchmark_mapping:
        return benchmark_mapping[num_solutions]
    return DEFAULT_OVERFITTED_CONSTRAINTS.get(num_solutions, DEFAULT_OVERFITTED_VALUE)


def process_benchmark_config(benchmark, num_solutions, approaches):
    """
    Worker function to process a single benchmark configuration.
    Runs Phase 1 once, then Phase 2 for both COP and LION approaches.
    Phase 3 is skipped.
    Returns list of metrics collected.
    """
    num_overfitteds = calculate_overfitted_constraints(benchmark, num_solutions)
    config_tag = f"sol{num_solutions}_of{num_overfitteds}"
    
    print(f"\n{'='*80}")
    print(f"[TASK] Processing: {benchmark} | Solutions={num_solutions}, overfitted={num_overfitteds}")
    print(f"{'='*80}\n")
    
    config_metrics = []
    

    phase1_success, phase1_pickle, phase1_time = run_phase1_with_timing(
        benchmark, num_solutions, num_overfitteds
    )
    
    if not phase1_success:
        print(f"\n[THREAD ERROR] Phase 1 failed for {benchmark} with {num_solutions} solutions")
        return config_metrics
    

    for approach in approaches:
        print(f"\n{'-'*60}")
        print(f"[TASK] Running {approach.upper()} approach for {benchmark} ({config_tag})")
        print(f"[TASK] Phase 2 only (Phase 3 skipped)")
        print(f"{'-'*60}\n")
        

        phase2_success, phase2_pickle = run_phase2(
            benchmark,
            phase1_pickle,
            approach=approach,
            config_tag=config_tag,
        )
        
        if not phase2_success:
            print(f"\n[TASK ERROR] Phase 2 ({approach.upper()}) failed for {benchmark}")
            continue
        

        try:
            metrics = extract_metrics(
                benchmark,
                num_solutions,
                phase1_pickle,
                phase1_time,
                approach=approach,
                config_tag=config_tag,
                phase2_pickle_path=phase2_pickle,
            )
            if metrics:
                config_metrics.append(metrics)
                print(f"\n[TASK SUCCESS] Extracted Phase 2 metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}")
            else:
                print(f"\n[TASK WARNING] Could not extract metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}")
        except Exception as e:
            print(f"\n[TASK ERROR] Failed to extract metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}: {e}")
            import traceback
            traceback.print_exc()
    
    return config_metrics


def append_metrics_to_csv(metrics_list, csv_path, metrics_lock):
    """Append metrics to CSV file in a thread-safe manner."""
    if not metrics_list:
        return
    
    with metrics_lock:

        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a') as f:

            if not file_exists:
                f.write("Prob.,Approach,Sols,StartC,Implied,NotImplied,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,")
                f.write("P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),")
                f.write("precision,recall,s_precision,s_recall\n")
            

            for m in metrics_list:
                f.write(f"{m['Prob.']},{m['Approach']},{m['Sols']},{m['StartC']},{m['Implied']},{m['NotImplied']},{m['InvC']},")
                f.write(f"{m['CT']},{m['Bias']},{m['ViolQ']},{m['MQuQ']},{m['TQ']},")
                f.write(f"{m['ALQ']},{m['PAQ']},")
                f.write(f"{m['P1T(s)']},{m['VT(s)']},{m['MQuT(s)']},{m['TT(s)']},")
                f.write(f"{m['ALT(s)']},{m['PAT(s)']},")
                f.write(f"{m['precision']},{m['recall']},{m['s_precision']},{m['s_recall']}\n")


def update_progress_file(completed, total, progress_path, metrics_lock):
    """Update progress tracking file."""
    with metrics_lock:
        with open(progress_path, 'w') as f:
            f.write(f"Experiment Progress (Phase 2 Only)\n")
            f.write(f"{'='*60}\n")
            f.write(f"Completed: {completed}/{total} tasks\n")
            f.write(f"Progress: {100*completed/total:.1f}%\n")
            f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")


def aggregate_metrics_across_runs(aggregated_metrics, num_runs):
    """Aggregate metrics across multiple runs, computing averages and standard deviations."""
    aggregated_results = []

    for (benchmark, sols, approach), metrics_list in aggregated_metrics.items():
        if len(metrics_list) == 0:
            continue

        # Initialize aggregated metric with the same structure
        agg_metric = {
            'Prob.': benchmark,
            'Approach': approach,
            'Sols': sols,
            'Runs': len(metrics_list)
        }

        # Numeric fields to aggregate
        numeric_fields = [
            'StartC', 'Implied', 'NotImplied', 'InvC', 'CT', 'Bias', 'ViolQ', 'MQuQ', 'TQ',
            'P1T(s)', 'VT(s)', 'MQuT(s)', 'TT(s)', 'ALT(s)', 'PAT(s)',
            'precision', 'recall', 's_precision', 's_recall'
        ]

        # For each numeric field, compute mean and std
        for field in numeric_fields:
            values = []
            for m in metrics_list:
                val = m.get(field, 0)
                # Skip N/A values
                if isinstance(val, (int, float)) and not (isinstance(val, str) and val == 'N/A'):
                    values.append(float(val))

            if values:
                mean_val = statistics.mean(values)
                if len(values) > 1:
                    std_val = statistics.stdev(values)
                else:
                    std_val = 0.0

                # Store as formatted strings for display
                if field in ['P1T(s)', 'VT(s)', 'MQuT(s)', 'TT(s)', 'ALT(s)', 'PAT(s)',
                           'precision', 'recall', 's_precision', 's_recall']:
                    agg_metric[field] = f"{mean_val:.2f}±{std_val:.2f}"
                    agg_metric[f"{field}_mean"] = mean_val
                    agg_metric[f"{field}_std"] = std_val
                else:
                    agg_metric[field] = f"{mean_val:.1f}±{std_val:.1f}"
                    agg_metric[f"{field}_mean"] = mean_val
                    agg_metric[f"{field}_std"] = std_val
            else:
                agg_metric[field] = 'N/A'

        aggregated_results.append(agg_metric)

    return aggregated_results


def main(num_runs=10):
    """Run the complete solution variance experiment through Phase 2 only."""

    print(f"\n{'='*80}")
    print(f"SOLUTION VARIANCE EXPERIMENTS - COP vs LION COMPARISON (PHASE 2 ONLY)")
    print(f"Running each experiment {num_runs} times")
    print(f"{'='*80}")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"NOTE: This script runs experiments only through Phase 2 (Phase 3 is skipped)")
    print(f"{'='*80}\n")
    

    benchmarks = [
'sudoku',
'sudoku_gt',
"graph_coloring_register",
"examtt_v1",
"examtt_v2",
"nurse",
"jsudoku",





    ]
    

    approaches = ['cop','lion']
    

    benchmark_solution_map = {
        benchmark: get_solution_counts_for_benchmark(benchmark)
        for benchmark in benchmarks
    }

    print("Benchmark-specific solution counts (solutions → overfitted constraints):")
    for benchmark in benchmarks:
        display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        solution_counts = benchmark_solution_map[benchmark]
        mapped_constraints = {
            sol: calculate_overfitted_constraints(benchmark, sol)
            for sol in solution_counts
        }
        print(f"  - {display_name}: {mapped_constraints}")
    print(f"{'='*80}\n")


    all_metrics = []
    metrics_lock = Lock()

    # Storage for aggregating results across runs
    aggregated_metrics = {}  # Key: (benchmark, num_solutions, approach), Value: list of metrics
    

    output_dir = 'solution_variance_output_phase2'
    os.makedirs(output_dir, exist_ok=True)
    
    intermediate_csv_path = f"{output_dir}/intermediate_results.csv"
    progress_path = f"{output_dir}/progress.txt"
    

    with open(intermediate_csv_path, 'w') as f:
        f.write("Prob.,Approach,Sols,StartC,Implied,NotImplied,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,")
        f.write("P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),")
        f.write("precision,recall,s_precision,s_recall\n")
    
    print(f"[INFO] Intermediate results will be saved to: {intermediate_csv_path}")
    print(f"[INFO] Progress tracking file: {progress_path}\n")
    

    # Create list of all tasks (benchmark, solution_config, run_number combinations)
    tasks = []
    for benchmark, solution_counts in benchmark_solution_map.items():
        for num_solutions in solution_counts:
            for run_num in range(1, num_runs + 1):
                tasks.append((benchmark, num_solutions, approaches, run_num))

    total_tasks = len(tasks)
    print(f"Total tasks to process: {total_tasks}")
    print(f"Each task runs Phase 1 followed by COP (Phase 2) and LION (Phase 2)")
    print(f"Phase 3 is skipped for all experiments")
    print(f"Each experiment configuration will be run {num_runs} times")
    print(f"Processing sequentially...\n")
    

    update_progress_file(0, total_tasks, progress_path, metrics_lock)
    

    for index, (benchmark, num_solutions, approach_list, run_num) in enumerate(tasks, start=1):
        print(f"\n{'='*80}")
        print(f"[TASK {index}/{total_tasks}] Processing: {benchmark} | Solutions={num_solutions} | Run {run_num}")
        print(f"{'='*80}\n")

        try:
            config_metrics = process_benchmark_config(benchmark, num_solutions, approach_list)

            # Store metrics for aggregation
            for metric in config_metrics:
                key = (metric['Prob.'], metric['Sols'], metric['Approach'])
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(metric)

            # Thread-safe addition to all_metrics (includes all individual runs)
            with metrics_lock:
                all_metrics.extend(config_metrics)

            # Write intermediate results immediately
            append_metrics_to_csv(config_metrics, intermediate_csv_path, metrics_lock)

            # Update progress tracking
            update_progress_file(index, total_tasks, progress_path, metrics_lock)

            print(f"\n{'='*80}")
            print(f"[PROGRESS] Completed {index}/{total_tasks}: {benchmark} with {num_solutions} solutions (Run {run_num})")
            print(f"[PROGRESS] Collected {len(config_metrics)} metric sets from this task")
            print(f"[PROGRESS] Results appended to: {intermediate_csv_path}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n[ERROR] Task failed for {benchmark} with {num_solutions} solutions (Run {run_num}): {e}")
            import traceback
            traceback.print_exc()

            # Still update progress even on failure
            update_progress_file(index, total_tasks, progress_path, metrics_lock)
    

    print(f"\n{'='*80}")
    print(f"ALL TASKS COMPLETED")
    print(f"{'='*80}")
    print(f"Total metrics collected: {len(all_metrics)}")
    print(f"Total configurations tested: {len(aggregated_metrics)}")
    print(f"{'='*80}\n")

    # Aggregate results across runs
    print(f"Aggregating results across {num_runs} runs per configuration...\n")
    aggregated_results = aggregate_metrics_across_runs(aggregated_metrics, num_runs)

    # Generate summary report
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT SUMMARY (PHASE 2 ONLY)")
    print(f"{'='*80}\n")

    if not all_metrics:
        print("[WARNING] No metrics collected. Check for errors in pipeline execution.")

    print(f"\n[INFO] Intermediate results during execution: {intermediate_csv_path}")
    print(f"[INFO] Generating final summary reports...\n")
    

    # Generate formatted text report for aggregated results (matching HCAR format)
    report_path = f"{output_dir}/variance_results_phase2_aggregated.txt"
    with open(report_path, 'w') as f:
        f.write(f"Solution Variance Experiment Results - Phase 2 Only (COP vs LION Comparison) - AGGREGATED OVER {num_runs} RUNS\n")
        f.write("="*150 + "\n\n")

        # Header line
        f.write(f"{'Prob.':<15} {'Approach':<9} {'Sols':<6} {'Runs':<6} {'StartC':<10} {'Implied':<11} {'NotImp.':<11} {'InvC':<8} {'CT':<5} {'Bias':<8} ")
        f.write(f"{'ViolQ':<9} {'MQuQ':<9} {'TQ':<8} ")
        f.write(f"{'P1T(s)':<12} {'VT(s)':<12} {'MQuT(s)':<13} {'TT(s)':<12}\n")

        # Data rows
        for m in sorted(aggregated_results, key=lambda x: (x['Prob.'], x['Sols'], x['Approach'])):
            f.write(f"{m['Prob.']:<15} {m['Approach']:<9} {m['Sols']:<6} {m['Runs']:<6} {m['StartC']:<10} {str(m['Implied']):<11} {str(m['NotImplied']):<11} {m['InvC']:<8} ")
            f.write(f"{str(m['CT']):<5} {m['Bias']:<8} ")
            f.write(f"{m['ViolQ']:<9} {m['MQuQ']:<9} {m['TQ']:<8} ")
            f.write(f"{m['P1T(s)']:<12} {m['VT(s)']:<12} {m['MQuT(s)']:<13} {m['TT(s)']:<12}\n")

        f.write("\n" + "="*150 + "\n")
        f.write("Legend:\n")
        f.write("  Approach: COP or LION methodology\n")
        f.write("  Sols: Number of given solutions (positive examples)\n")
        f.write("  Runs: Number of experimental runs aggregated\n")
        f.write("  StartC: Number of candidate constraints from passive learning (mean±std)\n")
        f.write("  InvC: Number of constraints invalidated by refinement (mean±std)\n")
        f.write("  CT: Number of AllDifferent constraints in target model\n")
        f.write("  Bias: Size of generated bias (mean±std)\n")
        f.write("  NotImp.: Validated constraints not implied by target model (mean±std)\n")
        f.write("  ViolQ: Violation queries (Phase 2) (mean±std)\n")
        f.write("  MQuQ: Active learning queries (Phase 3) - NOT RUN (always 0)\n")
        f.write("  TQ: Total queries (ViolQ + MQuQ) (mean±std)\n")
        f.write("  P1T(s): Phase 1 passive learning time (mean±std)\n")
        f.write("  VT(s): Duration of violation phase (mean±std)\n")
        f.write("  MQuT(s): Duration of active learning phase - NOT RUN (always 0)\n")
        f.write("  TT(s): Overall runtime (P1T + VT + MQuT) (mean±std)\n")
        f.write("\nNote: All numeric values show mean±standard_deviation across runs\n")
        f.write("NOTE: This experiment runs only through Phase 2. Phase 3 metrics are not available.\n")

    # Generate detailed text report with all individual runs
    detailed_report_path = f"{output_dir}/variance_results_phase2_detailed.txt"
    with open(detailed_report_path, 'w') as f:
        f.write("Solution Variance Experiment Results - Phase 2 Only (COP vs LION Comparison) - ALL INDIVIDUAL RUNS\n")
        f.write("="*140 + "\n\n")

        # Header line
        f.write(f"{'Prob.':<15} {'Approach':<9} {'Sols':<6} {'StartC':<8} {'Implied':<9} {'NotImp.':<9} {'InvC':<6} {'CT':<5} {'Bias':<6} ")
        f.write(f"{'ViolQ':<7} {'MQuQ':<7} {'TQ':<6} {'ALQ':<5} {'PAQ':<5} ")
        f.write(f"{'P1T(s)':<8} {'VT(s)':<8} {'MQuT(s)':<9} {'TT(s)':<8} {'ALT(s)':<7} {'PAT(s)':<7}\n")

        # Data rows
        for m in all_metrics:
            f.write(f"{m['Prob.']:<15} {m['Approach']:<9} {m['Sols']:<6} {m['StartC']:<8} {str(m['Implied']):<9} {str(m['NotImplied']):<9} {m['InvC']:<6} ")
            f.write(f"{str(m['CT']):<5} {m['Bias']:<6} ")
            f.write(f"{m['ViolQ']:<7} {m['MQuQ']:<7} {m['TQ']:<6} {m['ALQ']:<5} {m['PAQ']:<5} ")
            f.write(f"{m['P1T(s)']:<8} {m['VT(s)']:<8} {m['MQuT(s)']:<9} {m['TT(s)']:<8} ")
            f.write(f"{m['ALT(s)']:<7} {m['PAT(s)']:<7}\n")

        f.write("\n" + "="*140 + "\n")
        f.write("Legend: Same as aggregated report\n")
        f.write(f"Total individual runs: {len(all_metrics)}\n")
        f.write("NOTE: This experiment runs only through Phase 2. Phase 3 metrics are not available.\n")
    
    print(f"[SAVED] Aggregated results saved to: {report_path}")
    print(f"[SAVED] Detailed results (all runs) saved to: {detailed_report_path}")

    # Print aggregated results to console
    print(f"\nAGGREGATED RESULTS (across {num_runs} runs per configuration):")
    print(f"{'Prob.':<15} {'Approach':<9} {'Sols':<6} {'Runs':<6} {'StartC':<10} {'Implied':<11} {'NotImp.':<11} {'InvC':<8} {'CT':<5} {'Bias':<8} ")
    print(f"{'ViolQ':<9} {'MQuQ':<9} {'TQ':<8} {'P1T(s)':<12} {'VT(s)':<12} {'MQuT(s)':<13} {'TT(s)':<12}")
    print("="*140)
    for m in sorted(aggregated_results, key=lambda x: (x['Prob.'], x['Sols'], x['Approach'])):
        print(f"{m['Prob.']:<15} {m['Approach']:<9} {m['Sols']:<6} {m['Runs']:<6} {m['StartC']:<10} {str(m['Implied']):<11} {str(m['NotImplied']):<11} {m['InvC']:<8} ", end="")
        print(f"{str(m['CT']):<5} {m['Bias']:<8} ", end="")
        print(f"{m['ViolQ']:<9} {m['MQuQ']:<9} {m['TQ']:<8} ", end="")
        print(f"{m['P1T(s)']:<12} {m['VT(s)']:<12} {m['MQuT(s)']:<13} {m['TT(s)']:<12}")
    

    # Generate CSV output for all individual runs
    csv_path = f"{output_dir}/variance_results_phase2_all_runs.csv"
    with open(csv_path, 'w') as f:
        f.write("Prob.,Approach,Sols,StartC,Implied,NotImplied,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,")
        f.write("P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),")
        f.write("precision,recall,s_precision,s_recall\n")

        for m in all_metrics:
            f.write(f"{m['Prob.']},{m['Approach']},{m['Sols']},{m['StartC']},{m['Implied']},{m['NotImplied']},{m['InvC']},")
            f.write(f"{m['CT']},{m['Bias']},{m['ViolQ']},{m['MQuQ']},{m['TQ']},")
            f.write(f"{m['ALQ']},{m['PAQ']},")
            f.write(f"{m['P1T(s)']},{m['VT(s)']},{m['MQuT(s)']},{m['TT(s)']},")
            f.write(f"{m['ALT(s)']},{m['PAT(s)']},")
            f.write(f"{m['precision']},{m['recall']},{m['s_precision']},{m['s_recall']}\n")

    print(f"[SAVED] CSV results (all runs) saved to: {csv_path}")

    # Generate CSV output for aggregated results
    agg_csv_path = f"{output_dir}/variance_results_phase2_aggregated.csv"
    with open(agg_csv_path, 'w') as f:
        f.write("Prob.,Approach,Sols,Runs,StartC,StartC_std,Implied,Implied_std,NotImplied,NotImplied_std,InvC,InvC_std,CT,Bias,Bias_std,")
        f.write("ViolQ,ViolQ_std,MQuQ,MQuQ_std,TQ,TQ_std,")
        f.write("P1T(s),P1T_std,VT(s),VT_std,MQuT(s),MQuT_std,TT(s),TT_std,")
        f.write("precision,precision_std,recall,recall_std,s_precision,s_precision_std,s_recall,s_recall_std\n")

        for m in sorted(aggregated_results, key=lambda x: (x['Prob.'], x['Sols'], x['Approach'])):
            f.write(f"{m['Prob.']},{m['Approach']},{m['Sols']},{m['Runs']},")
            f.write(f"{m['StartC_mean']:.3f},{m['StartC_std']:.3f},")
            f.write(f"{m['Implied_mean']:.3f},{m['Implied_std']:.3f},")
            f.write(f"{m['NotImplied_mean']:.3f},{m['NotImplied_std']:.3f},")
            f.write(f"{m['InvC_mean']:.3f},{m['InvC_std']:.3f},")
            f.write(f"{m['CT']},{m['Bias_mean']:.3f},{m['Bias_std']:.3f},")
            f.write(f"{m['ViolQ_mean']:.3f},{m['ViolQ_std']:.3f},")
            f.write(f"{m['MQuQ_mean']:.3f},{m['MQuQ_std']:.3f},")
            f.write(f"{m['TQ_mean']:.3f},{m['TQ_std']:.3f},")
            f.write(f"{m['P1T(s)_mean']:.3f},{m['P1T(s)_std']:.3f},")
            f.write(f"{m['VT(s)_mean']:.3f},{m['VT(s)_std']:.3f},")
            f.write(f"{m['MQuT(s)_mean']:.3f},{m['MQuT(s)_std']:.3f},")
            f.write(f"{m['TT(s)_mean']:.3f},{m['TT(s)_std']:.3f},")
            f.write(f"{m['precision_mean']:.3f},{m['precision_std']:.3f},")
            f.write(f"{m['recall_mean']:.3f},{m['recall_std']:.3f},")
            f.write(f"{m['s_precision_mean']:.3f},{m['s_precision_std']:.3f},")
            f.write(f"{m['s_recall_mean']:.3f},{m['s_recall_std']:.3f}\n")

    print(f"[SAVED] CSV results (aggregated) saved to: {agg_csv_path}")
    

    # Generate comparison summary (COP vs LION side-by-side) using aggregated results
    comparison_path = f"{output_dir}/cop_vs_lion_comparison_phase2_aggregated.txt"
    with open(comparison_path, 'w') as f:
        f.write(f"COP vs LION Comparison Summary (Phase 2 Only) - Aggregated over {num_runs} runs\n")
        f.write("="*140 + "\n\n")

        # Group by benchmark and solution count
        grouped = {}
        for m in aggregated_results:
            key = (m['Prob.'], m['Sols'])
            if key not in grouped:
                grouped[key] = {}
            grouped[key][m['Approach']] = m

        for (benchmark, sols), approaches in sorted(grouped.items()):
            f.write(f"\n{benchmark} - {sols} solutions:\n")
            f.write("-" * 120 + "\n")

            cop = approaches.get('COP', {})
            lion = approaches.get('LION', {})

            if cop and lion:
                f.write(f"{'Metric':<25} {'COP (mean±std)':<20} {'LION (mean±std)':<20} {'LION - COP':<15}\n")
                f.write("-" * 120 + "\n")

                # Query metrics
                f.write(f"{'ViolQ':<25} {cop.get('ViolQ', 'N/A'):<20} {lion.get('ViolQ', 'N/A'):<20} ")
                if 'ViolQ_mean' in cop and 'ViolQ_mean' in lion:
                    diff = lion['ViolQ_mean'] - cop['ViolQ_mean']
                    f.write(f"{diff:<15.1f}\n")
                else:
                    f.write(f"{'N/A':<15}\n")

                f.write(f"{'MQuQ':<25} {cop.get('MQuQ', 'N/A'):<20} {lion.get('MQuQ', 'N/A'):<20} ")
                if 'MQuQ_mean' in cop and 'MQuQ_mean' in lion:
                    diff = lion['MQuQ_mean'] - cop['MQuQ_mean']
                    f.write(f"{diff:<15.1f}\n")
                else:
                    f.write(f"{'N/A':<15}\n")

                f.write(f"{'Total Queries':<25} {cop.get('TQ', 'N/A'):<20} {lion.get('TQ', 'N/A'):<20} ")
                if 'TQ_mean' in cop and 'TQ_mean' in lion:
                    diff = lion['TQ_mean'] - cop['TQ_mean']
                    f.write(f"{diff:<15.1f}\n")
                else:
                    f.write(f"{'N/A':<15}\n")

                # Time metrics
                f.write(f"{'VT(s)':<25} {cop.get('VT(s)', 'N/A'):<20} {lion.get('VT(s)', 'N/A'):<20} ")
                if 'VT(s)_mean' in cop and 'VT(s)_mean' in lion:
                    diff = lion['VT(s)_mean'] - cop['VT(s)_mean']
                    f.write(f"{diff:<15.2f}\n")
                else:
                    f.write(f"{'N/A':<15}\n")

                f.write(f"{'MQuT(s)':<25} {cop.get('MQuT(s)', 'N/A'):<20} {lion.get('MQuT(s)', 'N/A'):<20} ")
                if 'MQuT(s)_mean' in cop and 'MQuT(s)_mean' in lion:
                    diff = lion['MQuT(s)_mean'] - cop['MQuT(s)_mean']
                    f.write(f"{diff:<15.2f}\n")
                else:
                    f.write(f"{'N/A':<15}\n")

                f.write(f"{'Total Time (s)':<25} {cop.get('TT(s)', 'N/A'):<20} {lion.get('TT(s)', 'N/A'):<20} ")
                if 'TT(s)_mean' in cop and 'TT(s)_mean' in lion:
                    diff = lion['TT(s)_mean'] - cop['TT(s)_mean']
                    f.write(f"{diff:<15.2f}\n")
                else:
                    f.write(f"{'N/A':<15}\n")

                # Accuracy metrics (N/A for Phase 2 only)
                f.write(f"{'Precision (%)':<25} {'N/A':<20} {'N/A':<20} {'N/A':<15}\n")
                f.write(f"{'Recall (%)':<25} {'N/A':<20} {'N/A':<20} {'N/A':<15}\n")

                # Implied constraints
                cop_implied = cop.get('Implied_mean', 'N/A')
                lion_implied = lion.get('Implied_mean', 'N/A')
                if isinstance(cop_implied, (int, float)) and isinstance(lion_implied, (int, float)):
                    implied_diff = lion_implied - cop_implied
                    diff_str = f"{implied_diff:<15.1f}"
                else:
                    diff_str = f"{'N/A':<15}"
                f.write(f"{'Implied (mean)':<25} {str(cop_implied):<20} {str(lion_implied):<20} {diff_str}\n")

                # Winner determination based on mean total queries
                f.write("\nWinner (based on mean total queries): ")
                if 'TQ_mean' in cop and 'TQ_mean' in lion:
                    if cop['TQ_mean'] < lion['TQ_mean']:
                        f.write("COP (fewer queries)\n")
                    elif lion['TQ_mean'] < cop['TQ_mean']:
                        f.write("LION (fewer queries)\n")
                    else:
                        f.write("TIE (same queries)\n")
                else:
                    f.write("Cannot determine (missing data)\n")
            else:
                if cop:
                    f.write(f"  COP: Available ({cop.get('Runs', 0)} runs)\n")
                if lion:
                    f.write(f"  LION: Available ({lion.get('Runs', 0)} runs)\n")
                if not cop:
                    f.write("  COP: Missing\n")
                if not lion:
                    f.write("  LION: Missing\n")
    
    print(f"[SAVED] COP vs LION comparison saved to: {comparison_path}")
    

    json_path = f"{output_dir}/variance_experiment_detailed_phase2.json"
    agg_json_path = f"{output_dir}/variance_experiment_aggregated_phase2.json"
    summary_overfitted_mapping = {
        benchmark: {
            str(sol): calculate_overfitted_constraints(benchmark, sol)
            for sol in benchmark_solution_map[benchmark]
        }
        for benchmark in benchmarks
    }

    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_runs_per_config': num_runs,
        'experiment_type': 'phase2_only',
        'metrics_all_runs': all_metrics,
        'metrics_aggregated': aggregated_results,
        'total_benchmarks': len(benchmarks),
        'solution_configurations': benchmark_solution_map,
        'overfitted_constraints_mapping': summary_overfitted_mapping,
        'note': 'This experiment runs only through Phase 2. Phase 3 is skipped.',
        'execution': {
            'max_workers': 1,
            'total_tasks': len(tasks),
            'total_unique_configs': len(aggregated_metrics)
        }
    }

    # Save aggregated JSON
    agg_summary = {
        'timestamp': datetime.now().isoformat(),
        'num_runs_per_config': num_runs,
        'experiment_type': 'phase2_only_aggregated',
        'metrics_aggregated': aggregated_results,
        'total_benchmarks': len(benchmarks),
        'total_unique_configs': len(aggregated_results),
        'note': 'Aggregated results across multiple runs. This experiment runs only through Phase 2. Phase 3 is skipped.',
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    with open(agg_json_path, 'w') as f:
        json.dump(agg_summary, f, indent=2)

    print(f"[SAVED] Detailed JSON (all runs) saved to: {json_path}")
    print(f"[SAVED] Aggregated JSON saved to: {agg_json_path}")
    
    total_expected_configs = len(tasks) * len(approaches)  # Total individual runs
    successful_runs = len(all_metrics)
    unique_configs = len(aggregated_metrics)  # Unique (benchmark, sols, approach) combinations

    cop_results = [m for m in all_metrics if m['Approach'] == 'COP']
    lion_results = [m for m in all_metrics if m['Approach'] == 'LION']

    cop_aggregated = [m for m in aggregated_results if m['Approach'] == 'COP']
    lion_aggregated = [m for m in aggregated_results if m['Approach'] == 'LION']

    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS (PHASE 2 ONLY - MULTI-RUN)")
    print(f"{'='*80}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Total unique configurations: {unique_configs}")
    print(f"Total individual runs expected: {total_expected_configs}")
    print(f"Successful individual runs: {successful_runs}/{total_expected_configs}")
    print(f"  - COP approach: {len(cop_results)} successful runs ({len(cop_aggregated)} aggregated configs)")
    print(f"  - LION approach: {len(lion_results)} successful runs ({len(lion_aggregated)} aggregated configs)")
    if total_expected_configs > 0:
        print(f"Success rate: {100*successful_runs/total_expected_configs:.1f}%")
    print(f"\n{'='*80}")
    print(f"OUTPUT FILES GENERATED")
    print(f"{'='*80}")
    print(f"Real-time monitoring (updated during execution):")
    print(f"  - {intermediate_csv_path}")
    print(f"  - {progress_path}")
    print(f"\nAggregated results (across {num_runs} runs per config):")
    print(f"  - {report_path}")
    print(f"  - {agg_csv_path}")
    print(f"  - {comparison_path}")
    print(f"\nDetailed results (all individual runs):")
    print(f"  - {detailed_report_path}")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"  - {agg_json_path}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(num_runs)
