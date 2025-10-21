"""
Phase 3: Active Learning with MQuAcq-2

Loads Phase 2 outputs (validated global constraints + Phase 1 inputs)
and runs MQuAcq-2 to learn remaining fixed-arity constraints.

According to HCAR methodology:
- C'_G: Validated global constraints from Phase 2
- B_fixed: Pruned fixed-arity bias from Phase 1
- MQuAcq-2: Learns remaining binary constraints with pruned bias
- Output: C_final = C'_G ∪ C_L
"""

import os
import sys
import pickle
import time
import json
from datetime import datetime
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables
from pycona import MQuAcq2, ProblemInstance

# Import benchmark constructors
from benchmarks_global import construct_sudoku
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks import construct_sudoku_binary, construct_examtt_simple


def load_phase2_data(pickle_path):
    """Load Phase 2 outputs."""
    print(f"Loading Phase 2 data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  - Validated constraints: {len(data['C_validated'])}")
    print(f"  - Phase 2 queries: {data['phase2_stats']['queries']}")
    print(f"  - Phase 2 time: {data['phase2_stats']['time']:.2f}s")
    
    return data


def construct_instance(experiment_name):
    """Construct both global and binary instances."""
    if 'sudoku' in experiment_name.lower():
        print(f"Constructing Sudoku instances...")
        n = 9
        instance_binary, oracle_binary = construct_sudoku_binary(3, 3, 9)
        result = construct_sudoku(3, 3, 9)
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'examtt' in experiment_name.lower():
        print(f"Constructing ExamTT instances...")
        n = 6
        result1 = construct_examtt_simple(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14)
        instance_binary, oracle_binary = (result1[0], result1[1]) if len(result1) == 3 else result1
        result2 = ces_global(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14)
        instance_global, oracle_global = (result2[0], result2[1]) if len(result2) == 3 else result2
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return instance_binary, oracle_binary, instance_global, oracle_global


def decompose_global_constraints(global_constraints):
    """
    Decompose global constraints (AllDifferent, Sum, Count) into binary constraints.
    
    Returns:
        list: Binary constraints that form CL_init
    """
    binary_constraints = []
    
    for c in global_constraints:
        if hasattr(c, 'name') and c.name == "alldifferent":
            # Decompose AllDifferent to binary not-equals
            decomposed = c.decompose()
            if decomposed and len(decomposed) > 0:
                binary_constraints.extend(decomposed[0])
                print(f"  Decomposed {c} -> {len(decomposed[0])} binary constraints")
        elif 'sum' in str(c).lower() or 'count' in str(c).lower():
            # For sum/count, we can't directly decompose, but add as is
            # MQuAcq-2 will use them as they are
            print(f"  Keeping global constraint: {c}")
            binary_constraints.append(c)
        else:
            # Other constraints (if any)
            binary_constraints.append(c)
    
    return binary_constraints


def prune_bias_with_globals(bias_fixed, global_constraints):
    """
    Prune B_fixed using validated global constraints.
    
    According to methodology:
    - When a global constraint is validated, prune contradictory constraints from B_fixed
    - Example: If AllDifferent([x1, x2]) is validated, remove (x1 == x2) from B_fixed
    
    Returns:
        list: Pruned bias
    """
    pruned_bias = []
    contradictions_removed = 0
    
    # Decompose globals to get implied binary constraints
    implied_binaries = []
    for gc in global_constraints:
        if hasattr(gc, 'name') and gc.name == "alldifferent":
            decomposed = gc.decompose()
            if decomposed and len(decomposed) > 0:
                implied_binaries.extend(decomposed[0])
    
    print(f"\n  Implied binary constraints from globals: {len(implied_binaries)}")
    implied_strs = set(str(c) for c in implied_binaries)
    
    # Check each bias constraint
    for b in bias_fixed:
        # Check if bias constraint contradicts any implied binary
        # Example: If (x != y) is implied, remove (x == y) from bias
        b_str = str(b)
        is_contradictory = False
        
        # Simple check: if exact string match in implied constraints
        if b_str in implied_strs:
            # Already implied by global, can remove from bias
            contradictions_removed += 1
            is_contradictory = True
        
        if not is_contradictory:
            pruned_bias.append(b)
    
    print(f"  Removed {contradictions_removed} constraints from bias (already implied by globals)")
    print(f"  Pruned bias size: {len(pruned_bias)}")
    
    return pruned_bias


def run_phase3(experiment_name, phase2_pickle_path, max_queries=1000, timeout=600):
    """
    Run Phase 3: Active Learning with MQuAcq-2
    
    Args:
        experiment_name: Name of the benchmark
        phase2_pickle_path: Path to Phase 2 pickle file
        max_queries: Maximum queries for MQuAcq-2
        timeout: Timeout in seconds
    """
    print(f"\n{'='*80}")
    print(f"Phase 3: Active Learning with MQuAcq-2")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Max queries: {max_queries}")
    print(f"Timeout: {timeout}s")
    print(f"{'='*80}\n")
    
    # Load Phase 2 outputs
    phase2_data = load_phase2_data(phase2_pickle_path)
    C_validated = phase2_data['C_validated']  # Validated global constraints
    phase1_data = phase2_data.get('phase1_data', None)
    
    # Check if we have Phase 1 data
    if phase1_data is None:
        print("\n[ERROR] No Phase 1 data found in Phase 2 pickle!")
        print("Phase 3 requires B_fixed from Phase 1")
        sys.exit(1)
    
    B_fixed = phase1_data.get('B_fixed', [])
    E_plus = phase1_data.get('E_plus', [])
    
    print(f"\nPhase 1 inputs:")
    print(f"  - B_fixed (fixed-arity bias): {len(B_fixed)} constraints")
    print(f"  - E_plus (positive examples): {len(E_plus)} examples")
    
    # Construct instances
    instance_binary, oracle_binary, instance_global, oracle_global = construct_instance(experiment_name)
    
    # Setup oracle
    oracle_binary.variables_list = cpm_array(instance_binary.X)
    oracle_global.variables_list = cpm_array(instance_global.X)
    
    print(f"\nBenchmark info:")
    print(f"  - Variables: {len(instance_binary.X)}")
    print(f"  - Target constraints (binary): {len(oracle_binary.constraints)}")
    print(f"  - Target constraints (global): {len(oracle_global.constraints)}")
    
    # Step 1: Decompose validated global constraints
    print(f"\n{'='*60}")
    print(f"Step 1: Decompose Validated Global Constraints")
    print(f"{'='*60}")
    print(f"Validated global constraints: {len(C_validated)}")
    for c in C_validated:
        print(f"  - {c}")
    
    CL_init = decompose_global_constraints(C_validated)
    print(f"\nInitial CL (decomposed): {len(CL_init)} constraints")
    
    # Step 2: Prune B_fixed using validated globals
    print(f"\n{'='*60}")
    print(f"Step 2: Prune B_fixed Using Validated Globals")
    print(f"{'='*60}")
    print(f"Original B_fixed: {len(B_fixed)} constraints")
    
    B_pruned = prune_bias_with_globals(B_fixed, C_validated)
    
    # Step 3: Run MQuAcq-2
    print(f"\n{'='*60}")
    print(f"Step 3: Run MQuAcq-2 on Pruned Bias")
    print(f"{'='*60}")
    print(f"Initial CL: {len(CL_init)}")
    print(f"Pruned bias: {len(B_pruned)}")
    
    # Create fresh ProblemInstance for MQuAcq-2 (fixes variable initialization issue)
    variables_for_mquacq = get_variables(CL_init + B_pruned)
    mquacq_instance = ProblemInstance(
        variables=cpm_array(variables_for_mquacq),
        init_cl=CL_init,
        name=f"{experiment_name}_phase3",
        bias=B_pruned
    )
    
    print(f"Created MQuAcq-2 instance:")
    print(f"  Variables: {len(mquacq_instance.variables)}")
    print(f"  Initial CL: {len(mquacq_instance.cl)}")
    print(f"  Bias: {len(mquacq_instance.bias)}")
    
    # Run MQuAcq-2
    ca_system = MQuAcq2()
    
    phase3_start = time.time()
    try:
        learned_instance = ca_system.learn(mquacq_instance, oracle=oracle_binary, verbose=3)
    except Exception as e:
        print(f"\n[ERROR] MQuAcq-2 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    phase3_time = time.time() - phase3_start
    
    # Get final model
    final_model = learned_instance.get_cpmpy_model()
    final_constraints = final_model.constraints
    
    phase3_queries = ca_system.env.metrics.total_queries
    
    print(f"\n{'='*60}")
    print(f"Phase 3 Results")
    print(f"{'='*60}")
    print(f"MQuAcq-2 queries: {phase3_queries}")
    print(f"MQuAcq-2 time: {phase3_time:.2f}s")
    print(f"Final model constraints: {len(final_constraints)}")
    
    # Combine all phases
    phase2_queries = phase2_data['phase2_stats']['queries']
    phase2_time = phase2_data['phase2_stats']['time']
    
    total_queries = phase2_queries + phase3_queries
    total_time = phase2_time + phase3_time
    
    print(f"\n{'='*60}")
    print(f"Complete HCAR Pipeline Results")
    print(f"{'='*60}")
    print(f"Phase 1: Passive Learning (0 queries)")
    print(f"Phase 2: Interactive Refinement ({phase2_queries} queries, {phase2_time:.2f}s)")
    print(f"Phase 3: Active Learning ({phase3_queries} queries, {phase3_time:.2f}s)")
    print(f"{'='*60}")
    print(f"TOTAL: {total_queries} queries, {total_time:.2f}s")
    print(f"{'='*60}")
    
    # Evaluate against target
    print(f"\n{'='*60}")
    print(f"Evaluation Against Target Model")
    print(f"{'='*60}")
    
    # For evaluation, we check if the learned model is solution-equivalent
    # to the target model (this requires checking all solutions, which is
    # expensive, so we do a proxy check)
    
    target_constraints_str = set(str(c) for c in oracle_binary.constraints)
    learned_constraints_str = set(str(c) for c in final_constraints)
    
    correct = len(target_constraints_str & learned_constraints_str)
    missing = len(target_constraints_str - learned_constraints_str)
    spurious = len(learned_constraints_str - target_constraints_str)
    
    print(f"Target model size: {len(oracle_binary.constraints)}")
    print(f"Learned model size: {len(final_constraints)}")
    print(f"Correct constraints: {correct}")
    print(f"Missing constraints: {missing}")
    print(f"Spurious constraints: {spurious}")
    
    if correct == len(oracle_binary.constraints) and spurious == 0:
        print(f"\n[SUCCESS] Perfect model learning!")
    
    # Calculate metrics
    precision = correct / len(final_constraints) if len(final_constraints) > 0 else 0
    recall = correct / len(oracle_binary.constraints) if len(oracle_binary.constraints) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1-Score: {f1:.2%}")
    
    # Save results
    results = {
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'phase1': {
            'queries': 0,
            'time': 0,
            'E_plus_size': len(E_plus),
            'B_fixed_size': len(B_fixed)
        },
        'phase2': {
            'queries': phase2_queries,
            'time': phase2_time,
            'validated_globals': len(C_validated),
            'validated_globals_list': [str(c) for c in C_validated]
        },
        'phase3': {
            'queries': phase3_queries,
            'time': phase3_time,
            'initial_cl': len(CL_init),
            'pruned_bias': len(B_pruned),
            'final_model_size': len(final_constraints)
        },
        'total': {
            'queries': total_queries,
            'time': total_time
        },
        'evaluation': {
            'target_size': len(oracle_binary.constraints),
            'learned_size': len(final_constraints),
            'correct': correct,
            'missing': missing,
            'spurious': spurious,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    # Save to JSON
    output_dir = "phase3_output"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{experiment_name}_phase3_results.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Phase 3: Active Learning with MQuAcq-2')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--phase2_pickle', type=str, required=True, help='Path to Phase 2 pickle file')
    parser.add_argument('--max_queries', type=int, default=1000, help='Maximum queries for MQuAcq-2')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    run_phase3(
        experiment_name=args.experiment,
        phase2_pickle_path=args.phase2_pickle,
        max_queries=args.max_queries,
        timeout=args.timeout
    )

