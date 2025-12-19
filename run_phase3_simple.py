#!/usr/bin/env python3
"""
Simplified Phase 3: MQuAcq2 Active Learning

This script runs MQuAcq2 using:
- Bias from Phase 2 (B_fixed)
- Decomposed learned AllDifferent constraints as initial CL

No validation checks, no debug output - just straightforward MQuAcq2 learning.
"""

import os
import sys
import pickle
import time
import json
import argparse
from datetime import datetime

from cpmpy import Model, cpm_array
from cpmpy.transformations.get_variables import get_variables
from pycona import MQuAcq2, ProblemInstance
from pycona.ca_environment import ActiveCAEnv
from pycona.query_generation import PQGen
from pycona.find_constraint.findc import FindC

# Benchmark constructors (global version for oracle)
from benchmarks_global import (
    construct_sudoku, construct_jsudoku, construct_latin_square,
    construct_graph_coloring_register, construct_graph_coloring_scheduling,
    construct_examtt_simple, construct_sudoku_4x4_gt
)
from benchmarks_global.sudoku_greater_than import construct_sudoku_greater_than


def load_phase2_pickle(pickle_path):
    """Load phase 2 pickle and extract key data."""
    if not os.path.exists(pickle_path):
        print(f"[ERROR] Phase 2 pickle not found: {pickle_path}")
        sys.exit(1)
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Extract oracle from pickle (stored during Phase 2)
    oracle = data.get('oracle', None)
    
    # Extract validated constraints
    C_validated = data.get('C_validated', [])
    
    # Extract phase1 data (contains B_fixed)
    phase1_data = data.get('phase1_data', {})
    B_fixed = phase1_data.get('B_fixed', []) if phase1_data else []
    
    # Extract phase2 stats
    phase2_stats = data.get('phase2_stats', {'queries': 0, 'time': 0})
    
    # Extract variables from pickle if available
    all_variables = data.get('all_variables', None)
    
    print(f"Loaded Phase 2 pickle: {pickle_path}")
    print(f"  - Validated constraints: {len(C_validated)}")
    print(f"  - Bias (B_fixed): {len(B_fixed)}")
    print(f"  - Phase 2 queries: {phase2_stats.get('queries', 0)}")
    print(f"  - Oracle from pickle: {'YES' if oracle is not None else 'NO'}")
    
    return C_validated, B_fixed, phase2_stats, oracle, all_variables


def construct_benchmark(experiment_name):
    """Construct the benchmark instance and oracle."""
    exp_lower = experiment_name.lower()
    
    if 'graph_coloring_register' in exp_lower or exp_lower == 'register':
        result = construct_graph_coloring_register()
    elif 'graph_coloring_scheduling' in exp_lower or exp_lower == 'scheduling':
        result = construct_graph_coloring_scheduling()
    elif 'latin_square' in exp_lower or 'latin' in exp_lower:
        result = construct_latin_square(n=9)
    elif 'jsudoku' in exp_lower:
        result = construct_jsudoku(grid_size=9)
    elif 'sudoku_4x4_gt' in exp_lower:
        result = construct_sudoku_4x4_gt(2, 2, 4)
    elif 'sudoku_gt' in exp_lower or 'sudoku_greater' in exp_lower:
        result = construct_sudoku_greater_than(3, 3, 9)
    elif 'sudoku' in exp_lower:
        result = construct_sudoku(3, 3, 9)
    elif 'examtt' in exp_lower:
        result = construct_examtt_simple(
            nsemesters=30, courses_per_semester=20, 
            slots_per_day=18, days_for_exams=35
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    # Handle different return formats (some return 2, some return 3 values)
    if len(result) == 3:
        instance, oracle, _ = result
    else:
        instance, oracle = result
    
    return instance, oracle


def decompose_alldifferent(constraints):
    """
    Decompose AllDifferent constraints to binary != constraints.
    Keep other constraints as-is.
    """
    binary_constraints = []
    
    for c in constraints:
        c_str = str(c)
        
        # Check if it's an AllDifferent constraint
        if hasattr(c, 'name') and c.name == "alldifferent":
            # Skip if contains arithmetic operations
            if any(op in c_str for op in ['//', '/', '*', '+', '%']):
                continue
            
            # Decompose to binary !=
            decomposed = c.decompose()
            if decomposed and len(decomposed) > 0:
                binary_constraints.extend(decomposed[0])
        else:
            # Keep non-AllDifferent constraints as-is
            binary_constraints.append(c)
    
    # Remove duplicates
    unique = {}
    for c in binary_constraints:
        unique[str(c)] = c
    
    return list(unique.values())




def get_scope(constraint):
    """Extract variables from a constraint."""
    import cpmpy
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


def prune_bias_to_oracle_scopes(bias, oracle_constraints):
    oracle_scopes = set()
    for c in oracle_constraints:
        scope = get_scope(c)
        if len(scope) >= 2:
            scope_key = frozenset(v.name for v in scope)
            oracle_scopes.add(scope_key)
    
    print(f"  Oracle has constraints on {len(oracle_scopes)} unique variable pair scopes")
    
    pruned_bias = []
    removed_count = 0
    
    for c in bias:
        scope = get_scope(c)
        if len(scope) >= 2:
            scope_key = frozenset(v.name for v in scope)
            if scope_key in oracle_scopes:
                pruned_bias.append(c)
            else:
                removed_count += 1
        else:
            pruned_bias.append(c)
    
    print(f"  Removed {removed_count} bias constraints on non-oracle scopes")
    print(f"  Pruned bias size: {len(pruned_bias)}")
    
    return pruned_bias


def run_growacq_simple(
    experiment_name,
    phase2_pickle_path,
    max_queries=1000,
    verbose=1
):
    """
    Simple MQuAcq2 runner.
    
    Args:
        experiment_name: Name of the benchmark
        phase2_pickle_path: Path to phase2 pickle file
        max_queries: Maximum queries for MQuAcq2
        verbose: Verbosity level (0-3)
    
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Phase 3: MQuAcq2 Active Learning (Simple)")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*70}\n")
    
    # 1. Load Phase 2 data (including oracle)
    C_validated, B_fixed, phase2_stats, oracle_from_pickle, variables_from_pickle = load_phase2_pickle(phase2_pickle_path)
    
    # 2. Use oracle from pickle if available, otherwise construct new one
    if oracle_from_pickle is not None:
        print(f"Using oracle from Phase 2 pickle (consistent with Phase 2)")
        oracle = oracle_from_pickle
        # Construct instance just to get the ProblemInstance structure
        instance, _ = construct_benchmark(experiment_name)
    else:
        print(f"WARNING: No oracle in pickle, constructing new instance")
        instance, oracle = construct_benchmark(experiment_name)
    
    print(f"\nBenchmark: {experiment_name}")
    print(f"  - Variables: {len(instance.X)}")
    print(f"  - Oracle constraints: {len(oracle.constraints)}")
    
    # 3. Decompose validated constraints (AllDifferent -> binary !=)
    CL_init = decompose_alldifferent(C_validated)
    print(f"\nInitial CL (decomposed from validated globals): {len(CL_init)}")
    
    # 4. Use oracle directly from pickle (no decomposition)
    oracle.variables_list = cpm_array(instance.X)
    
    print(f"Oracle: {len(oracle.constraints)} constraints")
    
    # 5. Set up MQuAcq2
    variables = get_variables(CL_init + B_fixed) if (CL_init or B_fixed) else list(instance.X.flat)
    
    ca_instance = ProblemInstance(
        variables=cpm_array(variables),
        init_cl=CL_init,
        name=f"{experiment_name}_phase3",
        bias=B_fixed
    )
    
    print(f"\nMQuAcq2 Setup:")
    print(f"  - Variables: {len(ca_instance.variables)}")
    print(f"  - Initial CL: {len(ca_instance.cl)}")
    print(f"  - Bias: {len(ca_instance.bias)}")
    
    # Create MQuAcq2 algorithm
    findc = FindC(time_limit=1)
    qgen = PQGen(time_limit=2)
    ca_env = ActiveCAEnv(qgen=qgen, findc=findc)
    
    mquacq2 = MQuAcq2(ca_env=ca_env)
    
    # 6. Run MQuAcq2
    print(f"\n{'='*70}")
    print(f"Starting MQuAcq2...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    learned_instance = mquacq2.learn(
        ca_instance,
        oracle=oracle,
        verbose=verbose
    )
    
    phase3_time = time.time() - start_time
    phase3_queries = mquacq2.env.metrics.total_queries
    
    # 7. Results
    learned_constraints = learned_instance.cl
    
    print(f"\n{'='*70}")
    print(f"MQuAcq2 Complete")
    print(f"{'='*70}")
    print(f"  - Queries: {phase3_queries}")
    print(f"  - Time: {phase3_time:.2f}s")
    print(f"  - Learned constraints: {len(learned_constraints)}")
    
    # 8. Evaluate against oracle
    target_strs = set(str(c) for c in oracle.constraints)
    learned_strs = set(str(c) for c in learned_constraints)
    
    correct = len(target_strs & learned_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)
    
    precision = correct / len(learned_constraints) if learned_constraints else 0
    recall = correct / len(oracle.constraints) if oracle.constraints else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nEvaluation:")
    print(f"  - Target constraints: {len(oracle.constraints)}")
    print(f"  - Learned constraints: {len(learned_constraints)}")
    print(f"  - Correct: {correct}")
    print(f"  - Missing: {missing}")
    print(f"  - Spurious: {spurious}")
    print(f"  - Precision: {precision:.2%}")
    print(f"  - Recall: {recall:.2%}")
    print(f"  - F1: {f1:.2%}")
    
    # Total queries and time
    total_queries = phase2_stats.get('queries', 0) + phase3_queries
    total_time = phase2_stats.get('time', 0) + phase3_time
    
    print(f"\n{'='*70}")
    print(f"Total (Phase 2 + Phase 3)")
    print(f"{'='*70}")
    print(f"  - Total queries: {total_queries}")
    print(f"  - Total time: {total_time:.2f}s")
    
    # 9. Save results
    results = {
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'phase2': {
            'queries': phase2_stats.get('queries', 0),
            'time': phase2_stats.get('time', 0),
            'validated_globals': len(C_validated)
        },
        'phase3': {
            'queries': phase3_queries,
            'time': phase3_time,
            'initial_cl': len(CL_init),
            'bias': len(B_fixed),
            'learned_constraints': len(learned_constraints)
        },
        'total': {
            'queries': total_queries,
            'time': total_time
        },
        'evaluation': {
            'target_size': len(oracle.constraints),
            'learned_size': len(learned_constraints),
            'correct': correct,
            'missing': missing,
            'spurious': spurious,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    output_dir = "phase3_simple_output"
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results: {results_path}")
    
    # Save learned model
    # Handle both numpy arrays and lists for instance.X
    if hasattr(instance.X, 'flat'):
        variables_list = list(instance.X.flat)
    else:
        variables_list = list(instance.X) if not isinstance(instance.X, list) else instance.X
    
    model_data = {
        'experiment': experiment_name,
        'learned_constraints': learned_constraints,
        'variables': variables_list,
        'phase3_queries': phase3_queries,
        'phase3_time': phase3_time
    }
    
    model_path = os.path.join(output_dir, f"{experiment_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[SAVED] Model: {model_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simple Phase 3: MQuAcq2 with decomposed AllDifferent CL'
    )
    parser.add_argument(
        '--experiment', type=str, default='sudoku',
        help='Experiment name (sudoku, jsudoku, latin_square, etc.)'
    )
    parser.add_argument(
        '--phase2_pickle', type=str, default=None,
        help='Path to Phase 2 pickle (default: phase2_output/{experiment}_phase2.pkl)'
    )
    parser.add_argument(
        '--max_queries', type=int, default=1000,
        help='Max queries for MQuAcq2'
    )
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='Verbosity level (0-3)'
    )
    
    args = parser.parse_args()
    
    # Default phase2 pickle path
    if args.phase2_pickle is None:
        args.phase2_pickle = f"phase2_output/{args.experiment}_phase2.pkl"
    
    run_growacq_simple(
        experiment_name=args.experiment,
        phase2_pickle_path=args.phase2_pickle,
        max_queries=args.max_queries,
        verbose=args.verbose
    )

