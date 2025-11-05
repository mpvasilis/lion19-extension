"""
Solution Variance Experiments - COP vs LION Comparison
=======================================================
This script runs experiments with varying numbers of solutions and inversely 
proportional overfitted constraints, comparing both COP and LION approaches.
As the number of solutions increases, the number of overfitted constraints decreases.

Configurations:
- 2 solutions  -> 50 overfitted constraints
- 5 solutions  -> 20 overfitted constraints  
- 10 solutions -> 10 overfitted constraints
- 50 solutions -> 2 overfitted constraints

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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
    config_output_dir = f"{output_dir}/{experiment}_sol{num_examples}_overfitted{num_overfitted}"
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
        f"Phase 1: {experiment} | Solutions={num_examples}, overfitted={num_overfitted}")
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    if success:
        print(f"\n[TIMING] Phase 1 completed in {elapsed_time:.2f} seconds")
        phase1_pickle = f"{config_output_dir}/{experiment}_phase1.pkl"
        return True, phase1_pickle, elapsed_time
    else:
        print(f"\n[ERROR] Phase 1 failed after {elapsed_time:.2f} seconds")
        return False, None, elapsed_time


def run_phase2(experiment, phase1_pickle, approach='cop', max_queries=5000, timeout=1200):
    """Run Phase 2 with the given Phase 1 pickle using specified approach (cop or lion)."""
    
    # Select the appropriate script based on approach
    script_map = {
        'cop': 'main_alldiff_cop.py',
        'lion': 'main_alldiff_lion19.py'
    }
    
    script = script_map.get(approach.lower(), 'main_alldiff_cop.py')
    
    # Different scripts output different filenames
    # COP outputs: {experiment}_phase2.pkl
    # LION outputs: {experiment}_lion19_phase2.pkl
    filename_map = {
        'cop': f"{experiment}_phase2.pkl",
        'lion': f"{experiment}_lion19_phase2.pkl"
    }
    
    cmd = [
        'python', script,
        '--experiment', experiment,
        '--phase1_pickle', phase1_pickle,
        '--max_queries', str(max_queries),
        '--timeout', str(timeout)
    ]
    
    success, _ = run_command(cmd, f"Phase 2 ({approach.upper()}): {experiment}")
    
    if success:
        # Both scripts output to "phase2_output" directory
        default_output = "phase2_output"
        source_pickle = os.path.join(default_output, filename_map[approach.lower()])
        
        # Move to approach-specific directory for organization
        target_dir = f"phase2_output_{approach.lower()}"
        os.makedirs(target_dir, exist_ok=True)
        target_pickle = os.path.join(target_dir, filename_map[approach.lower()])
        
        # Move the file
        if os.path.exists(source_pickle):
            shutil.move(source_pickle, target_pickle)
            print(f"\n[INFO] Moved {source_pickle} to {target_pickle}")
            return True, target_pickle
        else:
            print(f"\n[WARNING] Expected output file not found: {source_pickle}")
            return True, source_pickle
    else:
        return False, None


def run_phase3(experiment, phase2_pickle, approach='cop'):
    """Run Phase 3 with the given Phase 2 pickle."""
    
    cmd = [
        'python', 'run_phase3.py',
        '--experiment', experiment,
        '--phase2_pickle', phase2_pickle
    ]
    
    success, _ = run_command(cmd, f"Phase 3 ({approach.upper()}): {experiment}")
    
    if success:
        # Phase 3 outputs to "phase3_output" directory by default
        default_output = "phase3_output"
        
        # Move outputs to approach-specific directory
        target_dir = f"phase3_output_{approach.lower()}"
        os.makedirs(target_dir, exist_ok=True)
        
        # Files to move
        results_json = f"{experiment}_phase3_results.json"
        final_model_pkl = f"{experiment}_final_model.pkl"
        
        for filename in [results_json, final_model_pkl]:
            source = os.path.join(default_output, filename)
            target = os.path.join(target_dir, filename)
            
            if os.path.exists(source):
                shutil.move(source, target)
                print(f"[INFO] Moved {source} to {target}")
            else:
                print(f"[WARNING] Expected file not found: {source}")
    
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


def load_phase3_results(benchmark_name, approach='cop'):
    """Load Phase 3 JSON results from approach-specific directory."""
    output_dir = f"phase3_output_{approach.lower()}"
    json_path = f"{output_dir}/{benchmark_name}_phase3_results.json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load Phase 3 results: {e}")
        return None


def extract_metrics(benchmark_name, num_solutions, phase1_pickle_path, phase1_time, approach='cop'):
    """Extract all metrics from phase outputs for the results table."""
    
    # Load Phase 1 data
    phase1_data = load_phase1_pickle(phase1_pickle_path)
    if phase1_data is None:
        return None
    
    # Load Phase 3 results from approach-specific directory
    phase3_results = load_phase3_results(benchmark_name, approach=approach)
    if phase3_results is None:
        return None
    
    metrics = {}
    
    # Problem name
    metrics['Prob.'] = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)
    
    # Approach (COP or LION)
    metrics['Approach'] = approach.upper()
    
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


def calculate_overfitted_constraints(num_solutions):
    """
    Calculate the number of overfitted constraints inversely proportional to solutions.
    Formula: More solutions -> Fewer overfitted constraints
    """
    # Inverse relationship mapping
    mapping = {
        2: 50,
        5: 20,
        10: 10,
        50: 2
    }
    
    return mapping.get(num_solutions, 10)  # Default to 10 if not in mapping


def process_benchmark_config(benchmark, num_solutions, approaches):
    """
    Worker function to process a single benchmark configuration.
    Runs Phase 1 once, then both COP and LION approaches.
    Returns list of metrics collected.
    """
    num_overfitteds = calculate_overfitted_constraints(num_solutions)
    
    print(f"\n{'='*80}")
    print(f"[THREAD] Processing: {benchmark} | Solutions={num_solutions}, overfitted={num_overfitteds}")
    print(f"{'='*80}\n")
    
    config_metrics = []
    
    # Run Phase 1 once (shared for both approaches)
    phase1_success, phase1_pickle, phase1_time = run_phase1_with_timing(
        benchmark, num_solutions, num_overfitteds
    )
    
    if not phase1_success:
        print(f"\n[THREAD ERROR] Phase 1 failed for {benchmark} with {num_solutions} solutions")
        return config_metrics
    
    # Run both COP and LION approaches for this configuration
    for approach in approaches:
        print(f"\n{'-'*60}")
        print(f"[THREAD] Running {approach.upper()} approach for {benchmark}")
        print(f"{'-'*60}\n")
        
        # Run Phase 2 with the specified approach
        phase2_success, phase2_pickle = run_phase2(benchmark, phase1_pickle, approach=approach)
        
        if not phase2_success:
            print(f"\n[THREAD ERROR] Phase 2 ({approach.upper()}) failed for {benchmark}")
            continue
        
        # Run Phase 3 with the specified approach
        phase3_success = run_phase3(benchmark, phase2_pickle, approach=approach)
        
        if not phase3_success:
            print(f"\n[THREAD ERROR] Phase 3 ({approach.upper()}) failed for {benchmark}")
            continue
        
        # Extract metrics if all phases succeeded
        metrics = extract_metrics(benchmark, num_solutions, phase1_pickle, phase1_time, approach=approach)
        if metrics:
            config_metrics.append(metrics)
            print(f"\n[THREAD SUCCESS] Extracted metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}")
        else:
            print(f"\n[THREAD WARNING] Could not extract metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}")
    
    return config_metrics


def append_metrics_to_csv(metrics_list, csv_path, metrics_lock):
    """Append metrics to CSV file in a thread-safe manner."""
    if not metrics_list:
        return
    
    with metrics_lock:
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a') as f:
            # Write header if file doesn't exist
            if not file_exists:
                f.write("Prob.,Approach,Sols,StartC,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,")
                f.write("P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),")
                f.write("precision,recall,s_precision,s_recall\n")
            
            # Write data rows
            for m in metrics_list:
                f.write(f"{m['Prob.']},{m['Approach']},{m['Sols']},{m['StartC']},{m['InvC']},")
                f.write(f"{m['CT']},{m['Bias']},{m['ViolQ']},{m['MQuQ']},{m['TQ']},")
                f.write(f"{m['ALQ']},{m['PAQ']},")
                f.write(f"{m['P1T(s)']},{m['VT(s)']},{m['MQuT(s)']},{m['TT(s)']},")
                f.write(f"{m['ALT(s)']},{m['PAT(s)']},")
                f.write(f"{m['precision']},{m['recall']},{m['s_precision']},{m['s_recall']}\n")


def update_progress_file(completed, total, progress_path, metrics_lock):
    """Update progress tracking file."""
    with metrics_lock:
        with open(progress_path, 'w') as f:
            f.write(f"Experiment Progress\n")
            f.write(f"{'='*60}\n")
            f.write(f"Completed: {completed}/{total} tasks\n")
            f.write(f"Progress: {100*completed/total:.1f}%\n")
            f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")


def main():
    """Run the complete solution variance experiment with parallel execution."""
    
    print(f"\n{'='*80}")
    print(f"SOLUTION VARIANCE EXPERIMENTS - COP vs LION COMPARISON (PARALLEL)")
    print(f"{'='*80}")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Solutions: [2, 5, 10, 50]")
    print(f"  Mock Constraints (inverse): [50, 20, 10, 2]")
    print(f"  Approaches: [COP, LION]")
    print(f"  Parallel threads: 4")
    print(f"  Note: Each thread runs Phase 1 once, then both COP and LION for Phase 2+3")
    print(f"{'='*80}\n")
    
    # Define benchmarks to test
    benchmarks = [
        'sudoku',
        'sudoku_gt',
        # 'latin_square',
        # 'graph_coloring_register',
        # 'examtt_v1',
        # 'examtt_v2',
        # 'nurse',
        # 'jsudoku',
    ]
    
    # Define solution variance configurations
    solution_configs = [2, 5, 10, 50]
    
    # Define approaches to compare
    approaches = ['cop', 'lion']
    
    # Storage for collected metrics (thread-safe)
    all_metrics = []
    metrics_lock = Lock()
    
    # Setup output directory and intermediate results files
    output_dir = 'solution_variance_output'
    os.makedirs(output_dir, exist_ok=True)
    
    intermediate_csv_path = f"{output_dir}/intermediate_results.csv"
    progress_path = f"{output_dir}/progress.txt"
    
    # Initialize files (create with headers)
    with open(intermediate_csv_path, 'w') as f:
        f.write("Prob.,Approach,Sols,StartC,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,")
        f.write("P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),")
        f.write("precision,recall,s_precision,s_recall\n")
    
    print(f"[INFO] Intermediate results will be saved to: {intermediate_csv_path}")
    print(f"[INFO] Progress tracking file: {progress_path}\n")
    
    # Create list of all tasks (benchmark, solution_config combinations)
    tasks = []
    for benchmark in benchmarks:
        for num_solutions in solution_configs:
            tasks.append((benchmark, num_solutions, approaches))
    
    print(f"Total tasks to process: {len(tasks)}")
    print(f"Each task will run Phase 1 + COP (Phase 2+3) + LION (Phase 2+3)")
    print(f"Processing with 4 parallel threads...\n")
    
    # Initialize progress tracking
    update_progress_file(0, len(tasks), progress_path, metrics_lock)
    
    # Process tasks in parallel using ThreadPoolExecutor
    completed_tasks = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_benchmark_config, benchmark, num_sols, approaches): (benchmark, num_sols)
            for benchmark, num_sols, approaches in tasks
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_task):
            benchmark, num_sols = future_to_task[future]
            completed_tasks += 1
            
            try:
                config_metrics = future.result()
                
                # Thread-safe addition to all_metrics
                with metrics_lock:
                    all_metrics.extend(config_metrics)
                
                # Write intermediate results immediately
                append_metrics_to_csv(config_metrics, intermediate_csv_path, metrics_lock)
                
                # Update progress tracking
                update_progress_file(completed_tasks, len(tasks), progress_path, metrics_lock)
                
                print(f"\n{'='*80}")
                print(f"[PROGRESS] Completed {completed_tasks}/{len(tasks)}: {benchmark} with {num_sols} solutions")
                print(f"[PROGRESS] Collected {len(config_metrics)} metric sets from this task")
                print(f"[PROGRESS] Results appended to: {intermediate_csv_path}")
                print(f"{'='*80}\n")
                
            except Exception as e:
                print(f"\n[ERROR] Task failed for {benchmark} with {num_sols} solutions: {e}")
                import traceback
                traceback.print_exc()
                
                # Still update progress even on failure
                update_progress_file(completed_tasks, len(tasks), progress_path, metrics_lock)
    
    # All tasks completed
    print(f"\n{'='*80}")
    print(f"ALL PARALLEL TASKS COMPLETED")
    print(f"{'='*80}")
    print(f"Total metrics collected: {len(all_metrics)}")
    print(f"{'='*80}\n")
    
    # Generate summary report
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    if not all_metrics:
        print("[WARNING] No metrics collected. Check for errors in pipeline execution.")
    
    print(f"\n[INFO] Intermediate results during execution: {intermediate_csv_path}")
    print(f"[INFO] Generating final summary reports...\n")
    
    # Generate formatted text report (matching HCAR format)
    report_path = f"{output_dir}/variance_results.txt"
    with open(report_path, 'w') as f:
        f.write("Solution Variance Experiment Results (COP vs LION Comparison)\n")
        f.write("="*140 + "\n\n")
        
        # Header line
        f.write(f"{'Prob.':<15} {'Approach':<9} {'Sols':<6} {'StartC':<8} {'InvC':<6} {'CT':<5} {'Bias':<6} ")
        f.write(f"{'ViolQ':<7} {'MQuQ':<7} {'TQ':<6} {'ALQ':<5} {'PAQ':<5} ")
        f.write(f"{'P1T(s)':<8} {'VT(s)':<8} {'MQuT(s)':<9} {'TT(s)':<8} {'ALT(s)':<7} {'PAT(s)':<7}\n")
        
        # Data rows
        for m in all_metrics:
            f.write(f"{m['Prob.']:<15} {m['Approach']:<9} {m['Sols']:<6} {m['StartC']:<8} {m['InvC']:<6} ")
            f.write(f"{str(m['CT']):<5} {m['Bias']:<6} ")
            f.write(f"{m['ViolQ']:<7} {m['MQuQ']:<7} {m['TQ']:<6} {m['ALQ']:<5} {m['PAQ']:<5} ")
            f.write(f"{m['P1T(s)']:<8} {m['VT(s)']:<8} {m['MQuT(s)']:<9} {m['TT(s)']:<8} ")
            f.write(f"{m['ALT(s)']:<7} {m['PAT(s)']:<7}\n")
        
        f.write("\n" + "="*140 + "\n")
        f.write("Legend:\n")
        f.write("  Approach: COP or LION methodology\n")
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
    print(f"\n{'Prob.':<15} {'Approach':<9} {'Sols':<6} {'StartC':<8} {'InvC':<6} {'CT':<5} {'Bias':<6} ")
    print(f"{'ViolQ':<7} {'MQuQ':<7} {'TQ':<6} {'P1T(s)':<8} {'VT(s)':<8} {'MQuT(s)':<9} {'TT(s)':<8}")
    print("="*120)
    for m in all_metrics:
        print(f"{m['Prob.']:<15} {m['Approach']:<9} {m['Sols']:<6} {m['StartC']:<8} {m['InvC']:<6} ", end="")
        print(f"{str(m['CT']):<5} {m['Bias']:<6} ", end="")
        print(f"{m['ViolQ']:<7} {m['MQuQ']:<7} {m['TQ']:<6} ", end="")
        print(f"{m['P1T(s)']:<8} {m['VT(s)']:<8} {m['MQuT(s)']:<9} {m['TT(s)']:<8}")
    
    # Generate CSV output
    csv_path = f"{output_dir}/variance_results.csv"
    with open(csv_path, 'w') as f:
        # Header
        f.write("Prob.,Approach,Sols,StartC,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,")
        f.write("P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),")
        f.write("precision,recall,s_precision,s_recall\n")
        
        # Data
        for m in all_metrics:
            f.write(f"{m['Prob.']},{m['Approach']},{m['Sols']},{m['StartC']},{m['InvC']},")
            f.write(f"{m['CT']},{m['Bias']},{m['ViolQ']},{m['MQuQ']},{m['TQ']},")
            f.write(f"{m['ALQ']},{m['PAQ']},")
            f.write(f"{m['P1T(s)']},{m['VT(s)']},{m['MQuT(s)']},{m['TT(s)']},")
            f.write(f"{m['ALT(s)']},{m['PAT(s)']},")
            f.write(f"{m['precision']},{m['recall']},{m['s_precision']},{m['s_recall']}\n")
    
    print(f"[SAVED] CSV results saved to: {csv_path}")
    
    # Generate comparison summary (COP vs LION side-by-side)
    comparison_path = f"{output_dir}/cop_vs_lion_comparison.txt"
    with open(comparison_path, 'w') as f:
        f.write("COP vs LION Comparison Summary\n")
        f.write("="*120 + "\n\n")
        
        # Group by benchmark and solution count
        grouped = {}
        for m in all_metrics:
            key = (m['Prob.'], m['Sols'])
            if key not in grouped:
                grouped[key] = {}
            grouped[key][m['Approach']] = m
        
        for (benchmark, sols), approaches in sorted(grouped.items()):
            f.write(f"\n{benchmark} - {sols} solutions:\n")
            f.write("-" * 100 + "\n")
            
            cop = approaches.get('COP', {})
            lion = approaches.get('LION', {})
            
            if cop and lion:
                f.write(f"{'Metric':<20} {'COP':<15} {'LION':<15} {'Difference':<15}\n")
                f.write("-" * 100 + "\n")
                
                # Query metrics
                f.write(f"{'ViolQ':<20} {cop.get('ViolQ', 0):<15} {lion.get('ViolQ', 0):<15} ")
                f.write(f"{lion.get('ViolQ', 0) - cop.get('ViolQ', 0):<15}\n")
                
                f.write(f"{'MQuQ':<20} {cop.get('MQuQ', 0):<15} {lion.get('MQuQ', 0):<15} ")
                f.write(f"{lion.get('MQuQ', 0) - cop.get('MQuQ', 0):<15}\n")
                
                f.write(f"{'Total Queries':<20} {cop.get('TQ', 0):<15} {lion.get('TQ', 0):<15} ")
                f.write(f"{lion.get('TQ', 0) - cop.get('TQ', 0):<15}\n")
                
                # Time metrics
                f.write(f"{'VT(s)':<20} {cop.get('VT(s)', 0):<15} {lion.get('VT(s)', 0):<15} ")
                f.write(f"{lion.get('VT(s)', 0) - cop.get('VT(s)', 0):<15.2f}\n")
                
                f.write(f"{'MQuT(s)':<20} {cop.get('MQuT(s)', 0):<15} {lion.get('MQuT(s)', 0):<15} ")
                f.write(f"{lion.get('MQuT(s)', 0) - cop.get('MQuT(s)', 0):<15.2f}\n")
                
                f.write(f"{'Total Time (s)':<20} {cop.get('TT(s)', 0):<15} {lion.get('TT(s)', 0):<15} ")
                f.write(f"{lion.get('TT(s)', 0) - cop.get('TT(s)', 0):<15.2f}\n")
                
                # Accuracy metrics
                f.write(f"{'Precision (%)':<20} {cop.get('precision', 0):<15} {lion.get('precision', 0):<15} ")
                f.write(f"{lion.get('precision', 0) - cop.get('precision', 0):<15.2f}\n")
                
                f.write(f"{'Recall (%)':<20} {cop.get('recall', 0):<15} {lion.get('recall', 0):<15} ")
                f.write(f"{lion.get('recall', 0) - cop.get('recall', 0):<15.2f}\n")
                
                # Winner determination
                f.write("\nWinner: ")
                if cop.get('TQ', 0) < lion.get('TQ', 0):
                    f.write("COP (fewer queries)\n")
                elif lion.get('TQ', 0) < cop.get('TQ', 0):
                    f.write("LION (fewer queries)\n")
                else:
                    f.write("TIE (same queries)\n")
            else:
                if cop:
                    f.write("  COP: Available\n")
                if lion:
                    f.write("  LION: Available\n")
                if not cop:
                    f.write("  COP: Missing\n")
                if not lion:
                    f.write("  LION: Missing\n")
    
    print(f"[SAVED] COP vs LION comparison saved to: {comparison_path}")
    
    # Save detailed JSON
    json_path = f"{output_dir}/variance_experiment_detailed.json"
    summary = {
        'timestamp': datetime.now().isoformat(),
        'metrics': all_metrics,
        'total_benchmarks': len(benchmarks),
        'solution_configurations': solution_configs,
        'overfitted_constraints_mapping': {
            str(sol): calculate_overfitted_constraints(sol) for sol in solution_configs
        },
        'parallel_execution': {
            'max_workers': 4,
            'total_tasks': len(tasks)
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[SAVED] Detailed JSON saved to: {json_path}")
    
    total_expected_configs = len(benchmarks) * len(solution_configs) * len(approaches)
    successful_configs = len(all_metrics)
    
    cop_results = [m for m in all_metrics if m['Approach'] == 'COP']
    lion_results = [m for m in all_metrics if m['Approach'] == 'LION']
    
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS (PARALLEL EXECUTION)")
    print(f"{'='*80}")
    print(f"Total configurations expected: {total_expected_configs}")
    print(f"Successful completions: {successful_configs}/{total_expected_configs}")
    print(f"  - COP approach: {len(cop_results)} successful")
    print(f"  - LION approach: {len(lion_results)} successful")
    if total_expected_configs > 0:
        print(f"Success rate: {100*successful_configs/total_expected_configs:.1f}%")
    print(f"\n{'='*80}")
    print(f"OUTPUT FILES GENERATED")
    print(f"{'='*80}")
    print(f"Real-time monitoring (updated during execution):")
    print(f"  - {intermediate_csv_path}")
    print(f"  - {progress_path}")
    print(f"\nFinal summary reports:")
    print(f"  - {report_path}")
    print(f"  - {csv_path}")
    print(f"  - {comparison_path}")
    print(f"  - {json_path}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

