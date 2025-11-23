"""
Experiment: 10 Different 4x4 Greater-Than Sudoku Variants
==========================================================

This script creates 10 different 4x4 GT Sudoku problems with varying
greater-than constraint configurations, then runs Phase 1 and Phase 2 (COP)
for each variant and analyzes the results.

Author: Generated for research
Date: November 2025
"""

import os
import sys
import time
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle, ProblemInstance, absvar

# Import Phase 1 function
from phase1_passive_learning import run_phase1


# ============================================================================
# VARIANT GENERATORS - 10 Different 4x4 GT Sudoku Configurations
# ============================================================================

def construct_sudoku_4x4_gt_variant(variant_id, block_size_row=2, block_size_col=2, grid_size=4):
    """
    Construct a 4x4 Greater-Than Sudoku variant with different GT configurations.
    
    variant_id determines which greater-than constraint pattern to use:
    - 0: Horizontal GT only (right bias)
    - 1: Vertical GT only (down bias)
    - 2: Mixed horizontal and vertical (balanced)
    - 3: Dense GT (many constraints)
    - 4: Sparse GT (few constraints)
    - 5: Diagonal GT pattern
    - 6: Checkerboard pattern
    - 7: Border-focused GT
    - 8: Center-focused GT
    - 9: Random scattered GT
    """
    
    parameters = {
        "block_size_row": block_size_row,
        "block_size_col": block_size_col,
        "grid_size": grid_size,
        "variant_id": variant_id
    }
    
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")
    model = cp.Model()
    
    # Standard Sudoku constraints (same for all variants)
    # Row constraints
    for row in grid:
        model += cp.AllDifferent(row)
    
    # Column constraints
    for col in grid.T:
        model += cp.AllDifferent(col)
    
    # Block constraints
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col])
    
    # Variant-specific greater-than constraints
    if variant_id == 0:
        # Horizontal GT only (cells greater than right neighbor)
        gt_constraints = [
            (0, 0, 0, 1),  # grid[0,0] > grid[0,1]
            (1, 1, 1, 2),  # grid[1,1] > grid[1,2]
            (2, 0, 2, 1),  # grid[2,0] > grid[2,1]
            (3, 2, 3, 3),  # grid[3,2] > grid[3,3]
        ]
    
    elif variant_id == 1:
        # Vertical GT only (cells greater than below neighbor)
        gt_constraints = [
            (0, 1, 1, 1),  # grid[0,1] > grid[1,1]
            (1, 2, 2, 2),  # grid[1,2] > grid[2,2]
            (0, 3, 1, 3),  # grid[0,3] > grid[1,3]
            (2, 0, 3, 0),  # grid[2,0] > grid[3,0]
        ]
    
    elif variant_id == 2:
        # Mixed horizontal and vertical (balanced)
        gt_constraints = [
            (0, 0, 0, 1),  # Horizontal
            (1, 1, 1, 2),  # Horizontal
            (0, 2, 1, 2),  # Vertical
            (2, 3, 3, 3),  # Vertical
            (3, 0, 3, 1),  # Horizontal
        ]
    
    elif variant_id == 3:
        # Dense GT (many constraints)
        gt_constraints = [
            (0, 0, 0, 1),  # H
            (0, 2, 0, 3),  # H
            (1, 0, 1, 1),  # H
            (1, 2, 1, 3),  # H
            (2, 1, 2, 2),  # H
            (3, 0, 3, 1),  # H
            (0, 1, 1, 1),  # V
            (1, 0, 2, 0),  # V
            (0, 3, 1, 3),  # V
            (2, 2, 3, 2),  # V
        ]
    
    elif variant_id == 4:
        # Sparse GT (minimal constraints)
        gt_constraints = [
            (0, 0, 0, 1),  # H
            (2, 3, 3, 3),  # V
        ]
    
    elif variant_id == 5:
        # Diagonal pattern
        gt_constraints = [
            (0, 0, 0, 1),  # Main diagonal area
            (0, 0, 1, 0),  # Main diagonal area
            (1, 1, 1, 2),  # Main diagonal
            (1, 1, 2, 1),  # Main diagonal
            (2, 2, 2, 3),  # Main diagonal
            (2, 2, 3, 2),  # Main diagonal
        ]
    
    elif variant_id == 6:
        # Checkerboard pattern (alternating)
        gt_constraints = [
            (0, 0, 0, 1),  # Row 0
            (0, 2, 0, 3),  # Row 0
            (1, 1, 1, 2),  # Row 1 (offset)
            (2, 0, 2, 1),  # Row 2
            (2, 2, 2, 3),  # Row 2
            (3, 1, 3, 2),  # Row 3 (offset)
        ]
    
    elif variant_id == 7:
        # Border-focused (constraints on edges)
        gt_constraints = [
            (0, 0, 0, 1),  # Top edge
            (0, 2, 0, 3),  # Top edge
            (3, 0, 3, 1),  # Bottom edge
            (3, 2, 3, 3),  # Bottom edge
            (0, 0, 1, 0),  # Left edge
            (2, 0, 3, 0),  # Left edge
            (0, 3, 1, 3),  # Right edge
            (2, 3, 3, 3),  # Right edge
        ]
    
    elif variant_id == 8:
        # Center-focused (constraints in middle)
        gt_constraints = [
            (1, 1, 1, 2),  # Center block
            (1, 1, 2, 1),  # Center block
            (1, 2, 2, 2),  # Center block
            (2, 1, 2, 2),  # Center block
            (1, 0, 1, 1),  # Near center
            (1, 2, 1, 3),  # Near center
        ]
    
    elif variant_id == 9:
        # Random scattered (asymmetric)
        gt_constraints = [
            (0, 1, 0, 2),  # H
            (1, 3, 2, 3),  # V
            (2, 0, 2, 1),  # H
            (3, 2, 3, 3),  # H
            (0, 2, 1, 2),  # V
            (3, 0, 3, 1),  # H
        ]
    
    else:
        raise ValueError(f"Unknown variant_id: {variant_id}. Must be 0-9.")
    
    # Apply the greater-than constraints
    for r1, c1, r2, c2 in gt_constraints:
        if r1 < grid_size and r2 < grid_size and c1 < grid_size and c2 < grid_size:
            model += (grid[r1, c1] > grid[r2, c2])
    
    # Extract target constraints
    C_T = list(set(toplevel_list(model.constraints)))
    
    # No overfitted constraints in benchmark definition
    overfitted_constraints = []
    
    # Language for constraint acquisition
    AV = absvar(2)
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1]
    ]
    
    # Create problem instance
    instance = ProblemInstance(
        variables=grid,
        params=parameters,
        language=lang,
        name=f"sudoku_4x4_gt_variant{variant_id}"
    )
    
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle, overfitted_constraints


def get_variant_description(variant_id):
    """Return a human-readable description of each variant."""
    descriptions = {
        0: "Horizontal GT only (right bias)",
        1: "Vertical GT only (down bias)",
        2: "Mixed horizontal and vertical (balanced)",
        3: "Dense GT (many constraints)",
        4: "Sparse GT (minimal constraints)",
        5: "Diagonal pattern",
        6: "Checkerboard pattern (alternating)",
        7: "Border-focused (constraints on edges)",
        8: "Center-focused (constraints in middle)",
        9: "Random scattered (asymmetric)"
    }
    return descriptions.get(variant_id, f"Unknown variant {variant_id}")


# ============================================================================
# PHASE 1 RUNNER
# ============================================================================

def run_phase1_for_variant(variant_id, output_dir, num_examples=5, num_overfitted=10):
    """
    Run Phase 1 for a specific variant.
    
    This modifies the global benchmark registry temporarily to include our variant.
    """
    
    print(f"\n{'='*80}")
    print(f"PHASE 1 - VARIANT {variant_id}")
    print(f"{'='*80}")
    print(f"Description: {get_variant_description(variant_id)}")
    print(f"{'='*80}\n")
    
    # Temporarily inject variant constructor into the benchmarks_global module
    import benchmarks_global
    
    # Create the variant constructor with a unique name
    variant_name = f"sudoku_4x4_gt_variant{variant_id}"
    
    def temp_constructor(block_size_row=2, block_size_col=2, grid_size=4):
        return construct_sudoku_4x4_gt_variant(variant_id, block_size_row, block_size_col, grid_size)
    
    # Register the constructor
    setattr(benchmarks_global, f'construct_sudoku_4x4_gt_variant{variant_id}', temp_constructor)
    
    # Also temporarily modify the construct_instance function in phase1_passive_learning
    from phase1_passive_learning import construct_instance
    
    original_construct = construct_instance
    
    def modified_construct_instance(benchmark_name):
        if benchmark_name == variant_name:
            print(f"Constructing 4x4 Sudoku GT Variant {variant_id}...")
            return construct_sudoku_4x4_gt_variant(variant_id, 2, 2, 4)
        else:
            return original_construct(benchmark_name)
    
    # Monkey-patch temporarily
    import phase1_passive_learning
    phase1_passive_learning.construct_instance = modified_construct_instance
    
    try:
        # Run Phase 1 with the variant name
        output_path = run_phase1(
            benchmark_name=variant_name,
            output_dir=output_dir,
            num_examples=num_examples,
            num_overfitted=num_overfitted
        )
        
        return output_path
    
    finally:
        # Restore original function
        phase1_passive_learning.construct_instance = original_construct


# ============================================================================
# PHASE 2 RUNNER
# ============================================================================

def run_phase2_for_variant(variant_id, phase1_pickle_path, output_dir, log_file):
    """
    Run Phase 2 (COP version) for a specific variant.
    """
    
    print(f"\n{'='*80}")
    print(f"PHASE 2 - VARIANT {variant_id}")
    print(f"{'='*80}")
    print(f"Description: {get_variant_description(variant_id)}")
    print(f"Phase 1 pickle: {phase1_pickle_path}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")
    
    variant_name = f"sudoku_4x4_gt_variant{variant_id}"
    
    start_time = time.time()
    
    cmd = [
        sys.executable,  # Use the same Python interpreter
        'main_alldiff_cop.py',
        '--experiment', variant_name,
        '--phase1_pickle', phase1_pickle_path,
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"PHASE 2 EXPERIMENT LOG - VARIANT {variant_id}\n")
            f.write(f"Description: {get_variant_description(variant_id)}\n")
            f.write(f"Phase 1 pickle: {phase1_pickle_path}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
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
        
        # Parse results
        results = parse_phase2_output(output_lines, variant_name, variant_id)
        results['return_code'] = return_code
        results['duration'] = duration
        results['log_file'] = log_file
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Status: {'SUCCESS' if return_code == 0 else 'FAILED'}\n")
            f.write("="*80 + "\n")
        
        if return_code == 0:
            print(f"\n[SUCCESS] Variant {variant_id} completed in {duration:.2f}s")
        else:
            print(f"\n[FAILED] Variant {variant_id} returned error code {return_code}")
            results['status'] = 'FAILED'
        
        return results
    
    except Exception as e:
        print(f"\n[ERROR] Exception running variant {variant_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'variant_id': variant_id,
            'benchmark': variant_name,
            'status': 'ERROR',
            'error': str(e),
            'duration': duration,
            'log_file': log_file
        }


def parse_phase2_output(output_lines, benchmark_name, variant_id):
    """Parse Phase 2 output to extract key metrics."""
    
    results = {
        'variant_id': variant_id,
        'benchmark': benchmark_name,
        'description': get_variant_description(variant_id),
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
    
    # Determine status
    if results['status'] == 'UNKNOWN':
        if results['correct'] == results['target_count'] and results['spurious'] == 0:
            results['status'] = 'SUCCESS'
        elif results['queries'] is not None:
            results['status'] = 'COMPLETED'
        else:
            results['status'] = 'FAILED'
    
    return results


# ============================================================================
# RESULT ANALYSIS
# ============================================================================

def create_summary_report(all_results, output_dir, problem_definitions):
    """Create comprehensive summary report."""
    
    summary_file = os.path.join(output_dir, 'summary_report.txt')
    json_file = os.path.join(output_dir, 'results.json')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("10 VARIANTS - 4x4 GREATER-THAN SUDOKU EXPERIMENT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total variants: {len(all_results)}\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        success_count = sum(1 for r in all_results if r['status'] == 'SUCCESS')
        completed_count = sum(1 for r in all_results if r['status'] in ['SUCCESS', 'COMPLETED'])
        failed_count = len(all_results) - completed_count
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Successful (Perfect Learning): {success_count}/{len(all_results)}\n")
        f.write(f"Completed (Partial Learning): {completed_count - success_count}/{len(all_results)}\n")
        f.write(f"Failed/Error: {failed_count}/{len(all_results)}\n")
        f.write("\n")
        
        # Average metrics
        avg_queries = sum(r.get('queries', 0) for r in all_results if r.get('queries')) / len(all_results) if all_results else 0
        avg_time = sum(r.get('time', 0) for r in all_results if r.get('time')) / len(all_results) if all_results else 0
        avg_precision = sum(r.get('precision', 0) for r in all_results if r.get('precision') is not None) / len([r for r in all_results if r.get('precision') is not None]) if any(r.get('precision') is not None for r in all_results) else 0
        avg_recall = sum(r.get('recall', 0) for r in all_results if r.get('recall') is not None) / len([r for r in all_results if r.get('recall') is not None]) if any(r.get('recall') is not None for r in all_results) else 0
        
        f.write("AVERAGE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Average queries: {avg_queries:.1f}\n")
        f.write(f"Average time: {avg_time:.2f}s\n")
        f.write(f"Average precision: {avg_precision:.2%}\n")
        f.write(f"Average recall: {avg_recall:.2%}\n")
        f.write("\n")
        
        # Variant definitions
        f.write("VARIANT DEFINITIONS\n")
        f.write("-"*80 + "\n")
        for variant_id, desc in problem_definitions.items():
            f.write(f"Variant {variant_id}: {desc}\n")
        f.write("\n")
        
        # Per-variant detailed results
        f.write("PER-VARIANT RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for result in sorted(all_results, key=lambda x: x.get('variant_id', 999)):
            f.write(f"Variant {result.get('variant_id', '?')}: {result.get('description', 'Unknown')}\n")
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
        f.write(f"{'Variant':<10} {'Description':<40} {'Status':<12} {'Queries':<10} {'Precision':<12} {'Recall':<10}\n")
        f.write("-"*80 + "\n")
        
        for result in sorted(all_results, key=lambda x: x.get('variant_id', 999)):
            var_str = f"V{result.get('variant_id', '?')}"
            desc_str = result.get('description', 'Unknown')[:38]
            queries_str = str(result.get('queries', 'N/A'))
            prec_str = f"{result.get('precision', 0):.2%}" if result.get('precision') is not None else 'N/A'
            rec_str = f"{result.get('recall', 0):.2%}" if result.get('recall') is not None else 'N/A'
            
            f.write(f"{var_str:<10} {desc_str:<40} {result['status']:<12} {queries_str:<10} {prec_str:<12} {rec_str:<10}\n")
        
        f.write("="*80 + "\n")
    
    # Save JSON results
    json_data = {
        'metadata': {
            'experiment': '10 Variants - 4x4 GT Sudoku',
            'timestamp': datetime.now().isoformat(),
            'total_variants': len(all_results),
            'success_count': success_count,
            'failed_count': failed_count
        },
        'problem_definitions': problem_definitions,
        'results': all_results,
        'summary': {
            'avg_queries': avg_queries,
            'avg_time': avg_time,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nSummary report saved to: {summary_file}")
    print(f"JSON results saved to: {json_file}")
    
    return summary_file, json_file


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    """Main experiment runner."""
    
    print("="*80)
    print("EXPERIMENT: 10 DIFFERENT 4x4 GREATER-THAN SUDOKU VARIANTS")
    print("="*80)
    print()
    print("This experiment will:")
    print("  1. Generate 10 different 4x4 GT Sudoku problem variants")
    print("  2. Run Phase 1 (Passive Learning) for each variant")
    print("  3. Run Phase 2 (Active Learning - COP) for each variant")
    print("  4. Collect and analyze results")
    print()
    print("="*80)
    
    # Configuration
    NUM_VARIANTS = 10
    NUM_EXAMPLES = 5
    NUM_OVERFITTED = 10
    OUTPUT_DIR = 'experiment_10variants_output'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Store problem definitions
    problem_definitions = {}
    for i in range(NUM_VARIANTS):
        problem_definitions[i] = get_variant_description(i)
    
    # Save problem definitions
    with open(os.path.join(OUTPUT_DIR, 'problem_definitions.json'), 'w') as f:
        json.dump(problem_definitions, f, indent=2)
    
    print("\nProblem Variants:")
    print("-"*80)
    for variant_id, desc in problem_definitions.items():
        print(f"  Variant {variant_id}: {desc}")
    print("-"*80)
    
    print("\nStarting experiment...\n")
    
    # Track all results
    all_results = []
    
    # Run experiments for each variant
    for variant_id in range(NUM_VARIANTS):
        print(f"\n\n{'#'*80}")
        print(f"# VARIANT {variant_id}/{NUM_VARIANTS-1}: {get_variant_description(variant_id)}")
        print(f"{'#'*80}\n")
        
        variant_name = f"sudoku_4x4_gt_variant{variant_id}"
        
        try:
            # ===== PHASE 1 =====
            phase1_start = time.time()
            phase1_output = run_phase1_for_variant(
                variant_id=variant_id,
                output_dir=OUTPUT_DIR,
                num_examples=NUM_EXAMPLES,
                num_overfitted=NUM_OVERFITTED
            )
            phase1_duration = time.time() - phase1_start
            
            if not phase1_output or not os.path.exists(phase1_output):
                print(f"\n[ERROR] Phase 1 failed for variant {variant_id}")
                all_results.append({
                    'variant_id': variant_id,
                    'benchmark': variant_name,
                    'description': get_variant_description(variant_id),
                    'status': 'ERROR',
                    'error': 'Phase 1 failed'
                })
                continue
            
            print(f"\n[SUCCESS] Phase 1 completed in {phase1_duration:.2f}s")
            print(f"  Output: {phase1_output}")
            
            # Save the instance for reference
            instance_file = os.path.join(OUTPUT_DIR, f"variant{variant_id}_instance.pkl")
            instance, oracle, _ = construct_sudoku_4x4_gt_variant(variant_id, 2, 2, 4)
            with open(instance_file, 'wb') as f:
                pickle.dump({'instance': instance, 'oracle': oracle}, f)
            print(f"  Instance saved: {instance_file}")
            
            # ===== PHASE 2 =====
            log_file = os.path.join(OUTPUT_DIR, f"variant{variant_id}_phase2.log")
            
            phase2_result = run_phase2_for_variant(
                variant_id=variant_id,
                phase1_pickle_path=phase1_output,
                output_dir=OUTPUT_DIR,
                log_file=log_file
            )
            
            all_results.append(phase2_result)
            
            # Save intermediate results
            with open(os.path.join(OUTPUT_DIR, f'variant{variant_id}.pkl'), 'wb') as f:
                pickle.dump(phase2_result, f)
            
            print(f"\n[COMPLETED] Variant {variant_id}")
            print(f"  Status: {phase2_result['status']}")
            if phase2_result.get('queries'):
                print(f"  Queries: {phase2_result['queries']}")
            if phase2_result.get('precision') is not None:
                print(f"  Precision: {phase2_result['precision']:.2%}")
            if phase2_result.get('recall') is not None:
                print(f"  Recall: {phase2_result['recall']:.2%}")
        
        except Exception as e:
            print(f"\n[ERROR] Exception for variant {variant_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                'variant_id': variant_id,
                'benchmark': variant_name,
                'description': get_variant_description(variant_id),
                'status': 'ERROR',
                'error': str(e)
            })
    
    # ===== FINAL ANALYSIS =====
    print(f"\n\n{'='*80}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*80}\n")
    
    summary_file, json_file = create_summary_report(all_results, OUTPUT_DIR, problem_definitions)
    
    # Print final summary to console
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in all_results if r['status'] == 'SUCCESS')
    failed_count = sum(1 for r in all_results if r['status'] in ['FAILED', 'ERROR'])
    
    print(f"\nTotal variants: {NUM_VARIANTS}")
    print(f"  [+] Successful: {success_count}")
    print(f"  [-] Failed: {failed_count}")
    
    print(f"\nResults summary:")
    for result in sorted(all_results, key=lambda x: x.get('variant_id', 999)):
        status_symbol = "[+]" if result['status'] == 'SUCCESS' else "[-]"
        var_id = result.get('variant_id', '?')
        desc = result.get('description', 'Unknown')[:50]
        queries = result.get('queries', 'N/A')
        print(f"  {status_symbol} V{var_id}: {desc} | Queries: {queries}")
    
    print(f"\nFull report: {summary_file}")
    print(f"JSON data: {json_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    if success_count == NUM_VARIANTS:
        print("ALL VARIANTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"{failed_count} VARIANT(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
