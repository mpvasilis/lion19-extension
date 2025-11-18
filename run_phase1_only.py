"""
Run Phase 1 (Passive Learning) only for specific configurations and log times.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

PYTHON_EXECUTABLE = sys.executable or 'python3'

# Benchmark display names
BENCHMARK_DISPLAY_NAMES = {
    'sudoku': 'Sudoku',
    'sudoku_gt': 'Sudoku-GT',
}

# Benchmark-specific overfitted constraint counts (InvC) keyed by solution count
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
}

# Configurations to run (Problem, Sols)
CONFIGURATIONS = [
    ('Sudoku', 50),
    ('Sudoku', 2),
    ('Sudoku-GT', 2),
    ('Sudoku-GT', 20),
    ('Sudoku-GT', 200),
    ('Sudoku', 5),
]

# Map display names to internal benchmark names
DISPLAY_TO_INTERNAL = {
    'Sudoku': 'sudoku',
    'Sudoku-GT': 'sudoku_gt',
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
    
    # Phase 1 pickle path
    phase1_pickle = f"{config_output_dir}/{experiment}_phase1.pkl"
    
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
        print(f"\n[TIMING] Phase 1 completed in {elapsed_time:.2f} seconds")
        return True, elapsed_time
    else:
        print(f"\n[ERROR] Phase 1 failed after {elapsed_time:.2f} seconds")
        return False, elapsed_time


def calculate_overfitted_constraints(benchmark, num_solutions):
    """Return the number of overfitted constraints (InvC) for Phase 1."""
    benchmark_mapping = BENCHMARK_OVERFITTED_CONSTRAINTS.get(benchmark, {})
    if num_solutions in benchmark_mapping:
        return benchmark_mapping[num_solutions]
    return 10  # Default


def main():
    """Run Phase 1 only for specified configurations and log times."""
    
    print(f"\n{'='*80}")
    print(f"PHASE 1 PASSIVE LEARNING - TIME LOGGING")
    print(f"{'='*80}")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Setup output directory and log file
    output_dir = 'solution_variance_output'
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = f"{output_dir}/phase1_times.log"
    csv_file = f"{output_dir}/phase1_times.csv"
    
    # Initialize CSV file with header
    with open(csv_file, 'w') as f:
        f.write("Problem,Benchmark,Solutions,Overfitted,Phase1_Time(s),Status,Timestamp\n")
    
    print(f"[INFO] Phase 1 times will be logged to: {log_file}")
    print(f"[INFO] CSV results will be saved to: {csv_file}\n")
    
    results = []
    
    # Process each configuration
    for idx, (problem_display, num_solutions) in enumerate(CONFIGURATIONS, start=1):
        benchmark = DISPLAY_TO_INTERNAL.get(problem_display)
        if not benchmark:
            print(f"[ERROR] Unknown problem: {problem_display}")
            continue
        
        num_overfitted = calculate_overfitted_constraints(benchmark, num_solutions)
        
        print(f"\n{'='*80}")
        print(f"[CONFIG {idx}/{len(CONFIGURATIONS)}] {problem_display} | Solutions={num_solutions}, Overfitted={num_overfitted}")
        print(f"{'='*80}\n")
        
        # Run Phase 1
        success, phase1_time = run_phase1_with_timing(
            benchmark, 
            num_solutions, 
            num_overfitted,
            output_dir=output_dir
        )
        
        status = "SUCCESS" if success else "FAILED"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Store result
        result = {
            'problem': problem_display,
            'benchmark': benchmark,
            'solutions': num_solutions,
            'overfitted': num_overfitted,
            'time': phase1_time,
            'status': status,
            'timestamp': timestamp
        }
        results.append(result)
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {problem_display} | Solutions={num_solutions}, Overfitted={num_overfitted} | ")
            f.write(f"Phase1_Time={phase1_time:.2f}s | Status={status}\n")
        
        # Append to CSV
        with open(csv_file, 'a') as f:
            f.write(f"{problem_display},{benchmark},{num_solutions},{num_overfitted},")
            f.write(f"{phase1_time:.2f},{status},{timestamp}\n")
        
        print(f"\n[RESULT] {problem_display} | Solutions={num_solutions} | Phase1_Time={phase1_time:.2f}s | Status={status}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"PHASE 1 EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total configurations: {len(CONFIGURATIONS)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'FAILED')}")
    print(f"\nPhase 1 Times:")
    for r in results:
        print(f"  {r['problem']:<15} Sols={r['solutions']:<4} Overfitted={r['overfitted']:<3} Time={r['time']:>8.2f}s  {r['status']}")
    
    total_time = sum(r['time'] for r in results)
    print(f"\nTotal Phase 1 time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nLog file: {log_file}")
    print(f"CSV file: {csv_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

