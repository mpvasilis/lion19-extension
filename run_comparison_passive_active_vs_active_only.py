#!/usr/bin/env python3
"""
Comparison Script: Passive+Active vs Active-Only Learning

This script compares two constraint acquisition approaches:
1. Passive+Active (Hybrid): Phase 1 (passive) + Phase 3 (active) - skipping Phase 2 refinement
2. Active-Only: Pure active learning with no passive initialization

For each benchmark, it runs both approaches and compares:
- Total queries
- Learning time
- Model quality (precision, recall, F1)
- Solution-space metrics
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
from pycona.ca_environment import ActiveCAEnv
from pycona.query_generation import PQGen

from resilient_findc import ResilientFindC
from resilient_mquacq2 import ResilientMQuAcq2
from resilient_growacq import ResilientGrowAcq
from resilient_pqgen import ResilientPQGen

from benchmarks_global import construct_sudoku, construct_jsudoku, construct_latin_square
from benchmarks_global import construct_graph_coloring_register, construct_graph_coloring_scheduling
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering

from benchmarks import construct_sudoku_binary, construct_jsudoku_binary, construct_latin_square_binary
from benchmarks import construct_graph_coloring_binary_register, construct_graph_coloring_binary_scheduling
from benchmarks import construct_examtt_simple
from benchmarks import construct_nurse_rostering as construct_nurse_rostering_binary

from cpmpy.expressions.utils import all_pairs
from utils import get_scope


def generate_binary_bias(variables, language):
    """Generate binary constraint bias for all pairs of variables."""
    
    print(f"[GENERATE BIAS] Variables: {len(variables)}, Language: {language}")
    
    bias_constraints = []
    
    for v1, v2 in all_pairs(variables):
        for relation in language:
            if relation == '==':
                bias_constraints.append(v1 == v2)
            elif relation == '!=':
                bias_constraints.append(v1 != v2)
            elif relation == '<':
                bias_constraints.append(v1 < v2)
            elif relation == '>':
                bias_constraints.append(v1 > v2)
            elif relation == '<=':
                bias_constraints.append(v1 <= v2)
            elif relation == '>=':
                bias_constraints.append(v1 >= v2)
    
    print(f"[GENERATE BIAS] Generated {len(bias_constraints)} binary constraints")
    return bias_constraints


def construct_instance(experiment_name):
    """Construct both binary and global constraint instances for a benchmark."""
    
    if 'graph_coloring_register' in experiment_name.lower() or experiment_name.lower() == 'register':
        instance_binary, oracle_binary = construct_graph_coloring_binary_register()
        result = construct_graph_coloring_register()
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'graph_coloring_scheduling' in experiment_name.lower() or experiment_name.lower() == 'scheduling':
        instance_binary, oracle_binary = construct_graph_coloring_binary_scheduling()
        result = construct_graph_coloring_scheduling()
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'latin_square' in experiment_name.lower() or 'latin' in experiment_name.lower():
        instance_binary, oracle_binary = construct_latin_square_binary(n=9)
        result = construct_latin_square(n=9)
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'jsudoku' in experiment_name.lower():
        instance_binary, oracle_binary = construct_jsudoku_binary(grid_size=9)
        result = construct_jsudoku(grid_size=9)
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'sudoku_gt' in experiment_name.lower():
        # Use sudoku_greater_than for both binary and global
        instance_binary, oracle_binary = construct_sudoku_binary(3, 3, 9)
        result = construct_sudoku_greater_than(3, 3, 9)
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'sudoku' in experiment_name.lower():
        instance_binary, oracle_binary = construct_sudoku_binary(3, 3, 9)
        result = construct_sudoku(3, 3, 9)
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'examtt_v1' in experiment_name.lower():
        result1 = construct_examtt_simple(nsemesters=6, courses_per_semester=5, 
                                           slots_per_day=6, days_for_exams=10)
        instance_binary, oracle_binary = (result1[0], result1[1]) if len(result1) == 3 else result1
        result2 = construct_examtt_variant1(nsemesters=6, courses_per_semester=5, 
                                           slots_per_day=6, days_for_exams=10)
        instance_global, oracle_global = (result2[0], result2[1]) if len(result2) == 3 else result2
    elif 'examtt_v2' in experiment_name.lower():
        result1 = construct_examtt_simple(nsemesters=30, courses_per_semester=25, 
                                           slots_per_day=15, days_for_exams=40)
        instance_binary, oracle_binary = (result1[0], result1[1]) if len(result1) == 3 else result1
        result2 = construct_examtt_variant2(nsemesters=30, courses_per_semester=25, 
                                           slots_per_day=15, days_for_exams=40)
        instance_global, oracle_global = (result2[0], result2[1]) if len(result2) == 3 else result2
    elif 'nurse' in experiment_name.lower():
        instance_binary, oracle_binary = construct_nurse_rostering_binary()
        result = construct_nurse_rostering()
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return instance_binary, oracle_binary, instance_global, oracle_global


def decompose_global_constraints(global_constraints):
    """Decompose global constraints (like AllDifferent) into binary constraints."""
    
    binary_constraints = []
    
    for c in global_constraints:
        if hasattr(c, 'name') and c.name == "alldifferent":
            c_str = str(c)
            if '//' in c_str or '/' in c_str or '*' in c_str or '+' in c_str or '%' in c_str:
                print(f"  [SKIP] Constraint with transformations: {c}")
                continue
            
            decomposed = c.decompose()
            if decomposed and len(decomposed) > 0:
                binary_constraints.extend(decomposed[0])
                print(f"  [DECOMPOSE] {c} -> {len(decomposed[0])} binary constraints")
        else:
            binary_constraints.append(c)
    
    # Remove duplicates
    unique_constraints = []
    seen_strs = set()
    
    for c in binary_constraints:
        c_str = str(c)
        if c_str not in seen_strs:
            seen_strs.add(c_str)
            unique_constraints.append(c)
    
    # Validate scopes
    validated_constraints = []
    for c in unique_constraints:
        try:
            scope = get_scope(c)
            if len(scope) >= 2:
                validated_constraints.append(c)
        except Exception as e:
            print(f"  [WARNING] Invalid constraint scope: {c}")
    
    return validated_constraints


def compute_metrics(learned_constraints, target_constraints):
    """Compute precision, recall, and F1 for constraint-level evaluation."""
    
    target_strs = set(str(c) for c in target_constraints)
    learned_strs = set(str(c) for c in learned_constraints)
    
    correct = len(target_strs & learned_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)
    
    precision = correct / len(learned_constraints) if len(learned_constraints) > 0 else 0
    recall = correct / len(target_constraints) if len(target_constraints) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'target_size': len(target_constraints),
        'learned_size': len(learned_constraints),
        'correct': correct,
        'missing': missing,
        'spurious': spurious,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def run_active_learning(instance_binary, oracle_binary, init_cl, bias, experiment_name, algorithm='mquacq2'):
    """
    Run active learning (Phase 3) with MQuAcq-2 or GrowAcq.
    
    Args:
        instance_binary: Binary constraint problem instance
        oracle_binary: Oracle for answering queries
        init_cl: Initial constraints (empty for active-only, decomposed globals for passive+active)
        bias: Constraint bias (full bias for active-only, pruned bias for passive+active)
        experiment_name: Name of the experiment
        algorithm: 'mquacq2' or 'growacq'
    
    Returns:
        Dictionary with results
    """
    
    variables_for_ca = get_variables(init_cl + bias) if (init_cl or bias) else instance_binary.X
    
    ca_instance = ProblemInstance(
        variables=cpm_array(variables_for_ca),
        init_cl=init_cl,
        name=f"{experiment_name}_active",
        bias=bias
    )
    
    print(f"\n[ACTIVE LEARNING] Using {algorithm.upper()}")
    print(f"  Variables: {len(ca_instance.variables)}")
    print(f"  Initial CL: {len(ca_instance.cl)}")
    print(f"  Bias: {len(ca_instance.bias)}")
    
    # Setup resilient components
    resilient_findc = ResilientFindC(time_limit=1)
    qgen = ResilientPQGen(time_limit=2)
    custom_env = ActiveCAEnv(qgen=qgen, findc=resilient_findc)
    
    # Select algorithm
    if algorithm.lower() == 'growacq':
        inner_mquacq2 = ResilientMQuAcq2(ca_env=custom_env)
        ca_system = ResilientGrowAcq(ca_env=custom_env, inner_algorithm=inner_mquacq2)
    else:
        ca_system = ResilientMQuAcq2(ca_env=custom_env)
    
    start_time = time.time()
    try:
        learned_instance = ca_system.learn(
            ca_instance, 
            oracle=oracle_binary, 
            verbose=2
        )
    except Exception as e:
        print(f"\n[ERROR] Active learning failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    learning_time = time.time() - start_time
    
    learned_constraints = learned_instance.cl
    total_queries = ca_system.env.metrics.total_queries
    
    print(f"\n[ACTIVE LEARNING COMPLETE]")
    print(f"  Queries: {total_queries}")
    print(f"  Time: {learning_time:.2f}s")
    print(f"  Learned constraints: {len(learned_constraints)}")
    
    return {
        'learned_constraints': learned_constraints,
        'queries': total_queries,
        'time': learning_time,
        'algorithm': algorithm.upper()
    }


def run_passive_active_hybrid(experiment_name, phase1_pickle_path, algorithm='mquacq2'):
    """
    Run Passive+Active approach: Phase 1 (passive) + Phase 3 (active), skipping Phase 2.
    
    This loads Phase 1 results and directly proceeds to active learning without
    the Phase 2 interactive refinement step.
    """
    
    print(f"\n{'='*80}")
    print(f"PASSIVE+ACTIVE HYBRID (Phase 1 + Phase 3, skipping Phase 2)")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Phase 1 pickle: {phase1_pickle_path}")
    print(f"{'='*80}\n")
    
    # Load Phase 1 data
    if not os.path.exists(phase1_pickle_path):
        print(f"[ERROR] Phase 1 pickle not found: {phase1_pickle_path}")
        return None
    
    with open(phase1_pickle_path, 'rb') as f:
        phase1_data = pickle.load(f)
    
    B_fixed = phase1_data.get('B_fixed', [])
    print(f"\n[PHASE 1 LOADED]")
    print(f"  B_fixed: {len(B_fixed)} constraints")
    
    # Construct instances
    instance_binary, oracle_binary, instance_global, oracle_global = construct_instance(experiment_name)
    oracle_binary.variables_list = cpm_array(instance_binary.X)
    
    print(f"\n[BENCHMARK INFO]")
    print(f"  Variables: {len(instance_binary.X)}")
    print(f"  Target constraints: {len(oracle_binary.constraints)}")
    
    # For passive+active, we skip Phase 2, so no validated globals
    # We go directly to active learning with empty init_cl and full bias from Phase 1
    init_cl = []  # No initial constraints since we're skipping Phase 2
    bias = B_fixed  # Use the full bias from Phase 1
    
    print(f"\n[PHASE 3 SETUP] (Direct from Phase 1, skipping Phase 2)")
    print(f"  Initial CL: {len(init_cl)} (empty - no Phase 2 refinement)")
    print(f"  Bias: {len(bias)} (from Phase 1)")
    
    # Run active learning
    result = run_active_learning(instance_binary, oracle_binary, init_cl, bias, 
                                 experiment_name, algorithm)
    
    if result is None:
        return None
    
    # Evaluate results
    metrics = compute_metrics(result['learned_constraints'], oracle_binary.constraints)
    
    print(f"\n[EVALUATION]")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1-Score: {metrics['f1']:.2%}")
    
    return {
        'approach': 'Passive+Active',
        'experiment': experiment_name,
        'phase1': {
            'bias_size': len(B_fixed)
        },
        'phase3': {
            'queries': result['queries'],
            'time': result['time'],
            'learned_size': len(result['learned_constraints'])
        },
        'total': {
            'queries': result['queries'],  # Phase 1 has 0 queries
            'time': result['time']
        },
        'evaluation': metrics,
        'algorithm': result['algorithm']
    }


def run_active_only(experiment_name, algorithm='mquacq2'):
    """
    Run Active-Only approach: Pure active learning with no passive initialization.
    
    This starts from scratch with only the bias (no Phase 1 passive learning).
    """
    
    print(f"\n{'='*80}")
    print(f"ACTIVE-ONLY (Phase 3 only, no passive learning)")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Construct instances
    instance_binary, oracle_binary, instance_global, oracle_global = construct_instance(experiment_name)
    oracle_binary.variables_list = cpm_array(instance_binary.X)
    
    print(f"\n[BENCHMARK INFO]")
    print(f"  Variables: {len(instance_binary.X)}")
    print(f"  Target constraints: {len(oracle_binary.constraints)}")
    
    # For active-only, we need to construct the bias from scratch
    # Generate full binary bias using all relational operators
    language = ['==', '!=', '<', '>', '<=', '>=']
    bias = generate_binary_bias(instance_binary.X, language)
    
    print(f"\n[ACTIVE-ONLY SETUP]")
    print(f"  Initial CL: 0 (starting from scratch)")
    print(f"  Bias: {len(bias)} (full bias, no passive learning)")
    
    init_cl = []
    
    # Run active learning
    result = run_active_learning(instance_binary, oracle_binary, init_cl, bias, 
                                 experiment_name, algorithm)
    
    if result is None:
        return None
    
    # Evaluate results
    metrics = compute_metrics(result['learned_constraints'], oracle_binary.constraints)
    
    print(f"\n[EVALUATION]")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1-Score: {metrics['f1']:.2%}")
    
    return {
        'approach': 'Active-Only',
        'experiment': experiment_name,
        'phase3': {
            'queries': result['queries'],
            'time': result['time'],
            'learned_size': len(result['learned_constraints'])
        },
        'total': {
            'queries': result['queries'],
            'time': result['time']
        },
        'evaluation': metrics,
        'algorithm': result['algorithm']
    }


def run_comparison_for_benchmark(experiment_name, phase1_pickle_path, algorithm='mquacq2'):
    """Run both approaches for a single benchmark and compare results."""
    
    print(f"\n\n{'#'*80}")
    print(f"# BENCHMARK: {experiment_name}")
    print(f"{'#'*80}\n")
    
    # Run Passive+Active
    print(f"\n{'='*80}")
    print(f"Running APPROACH 1: Passive+Active (Phase 1 + Phase 3)")
    print(f"{'='*80}")
    passive_active_result = run_passive_active_hybrid(experiment_name, phase1_pickle_path, algorithm)
    
    # Run Active-Only
    print(f"\n{'='*80}")
    print(f"Running APPROACH 2: Active-Only (Phase 3 only)")
    print(f"{'='*80}")
    active_only_result = run_active_only(experiment_name, algorithm)
    
    # Compare results
    if passive_active_result and active_only_result:
        print(f"\n{'='*80}")
        print(f"COMPARISON SUMMARY: {experiment_name}")
        print(f"{'='*80}")
        print(f"\n{'Approach':<20} {'Queries':<12} {'Time (s)':<12} {'F1-Score':<12}")
        print(f"{'-'*56}")
        print(f"{'Passive+Active':<20} {passive_active_result['total']['queries']:<12} "
              f"{passive_active_result['total']['time']:<12.2f} "
              f"{passive_active_result['evaluation']['f1']:<12.2%}")
        print(f"{'Active-Only':<20} {active_only_result['total']['queries']:<12} "
              f"{active_only_result['total']['time']:<12.2f} "
              f"{active_only_result['evaluation']['f1']:<12.2%}")
        print(f"{'-'*56}")
        
        # Calculate improvements
        query_diff = passive_active_result['total']['queries'] - active_only_result['total']['queries']
        query_improvement = (query_diff / active_only_result['total']['queries'] * 100) if active_only_result['total']['queries'] > 0 else 0
        
        print(f"\nPassive+Active vs Active-Only:")
        print(f"  Query difference: {query_diff:+d} ({query_improvement:+.1f}%)")
        print(f"  F1 difference: {passive_active_result['evaluation']['f1'] - active_only_result['evaluation']['f1']:+.2%}")
    
    return {
        'experiment': experiment_name,
        'passive_active': passive_active_result,
        'active_only': active_only_result
    }


def main():
    """Main function to run comparison experiments on all benchmarks."""
    
    # Define benchmarks
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
    
    # Choose algorithm (can be modified via command line)
    import argparse
    parser = argparse.ArgumentParser(description='Compare Passive+Active vs Active-Only approaches')
    parser.add_argument('--algorithm', type=str, default='mquacq2', 
                       choices=['mquacq2', 'growacq'],
                       help='Active learning algorithm to use')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=None,
                       help='Specific benchmarks to run (default: all)')
    args = parser.parse_args()
    
    # Filter benchmarks if specified
    if args.benchmarks:
        benchmarks = [b for b in benchmarks if b['name'] in args.benchmarks]
    
    print(f"\n{'#'*80}")
    print(f"# COMPARISON: Passive+Active vs Active-Only")
    print(f"# Algorithm: {args.algorithm.upper()}")
    print(f"# Benchmarks: {len(benchmarks)}")
    print(f"{'#'*80}\n")
    
    for b in benchmarks:
        print(f"  - {b['name']}")
    
    all_results = []
    
    # Run comparison for each benchmark
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n\n{'#'*80}")
        print(f"# PROGRESS: {i}/{len(benchmarks)}")
        print(f"{'#'*80}")
        
        try:
            result = run_comparison_for_benchmark(
                benchmark['name'], 
                benchmark['phase1_pickle'],
                args.algorithm
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed on {benchmark['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'experiment': benchmark['name'],
                'error': str(e),
                'passive_active': None,
                'active_only': None
            })
    
    # Generate final summary
    print(f"\n\n{'#'*80}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*80}\n")
    
    print(f"{'Benchmark':<25} {'Approach':<20} {'Queries':<12} {'Time (s)':<12} {'F1-Score':<12}")
    print(f"{'='*81}")
    
    for result in all_results:
        if 'error' in result:
            print(f"{result['experiment']:<25} {'ERROR':<20} {'-':<12} {'-':<12} {'-':<12}")
            continue
        
        exp = result['experiment']
        if result['passive_active']:
            pa = result['passive_active']
            print(f"{exp:<25} {'Passive+Active':<20} {pa['total']['queries']:<12} "
                  f"{pa['total']['time']:<12.2f} {pa['evaluation']['f1']:<12.2%}")
        
        if result['active_only']:
            ao = result['active_only']
            print(f"{'':<25} {'Active-Only':<20} {ao['total']['queries']:<12} "
                  f"{ao['total']['time']:<12.2f} {ao['evaluation']['f1']:<12.2%}")
        
        print(f"{'-'*81}")
    
    # Save results to file
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"passive_active_vs_active_only_{args.algorithm}_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'algorithm': args.algorithm.upper(),
            'benchmarks': len(benchmarks),
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n[SAVED] Results saved to: {results_file}")
    
    # Generate summary statistics
    successful_results = [r for r in all_results if 'error' not in r and r['passive_active'] and r['active_only']]
    
    if successful_results:
        print(f"\n{'='*80}")
        print(f"AGGREGATE STATISTICS ({len(successful_results)} benchmarks)")
        print(f"{'='*80}")
        
        total_pa_queries = sum(r['passive_active']['total']['queries'] for r in successful_results)
        total_ao_queries = sum(r['active_only']['total']['queries'] for r in successful_results)
        avg_pa_f1 = sum(r['passive_active']['evaluation']['f1'] for r in successful_results) / len(successful_results)
        avg_ao_f1 = sum(r['active_only']['evaluation']['f1'] for r in successful_results) / len(successful_results)
        
        print(f"\nTotal Queries:")
        print(f"  Passive+Active: {total_pa_queries}")
        print(f"  Active-Only: {total_ao_queries}")
        print(f"  Difference: {total_pa_queries - total_ao_queries:+d}")
        
        print(f"\nAverage F1-Score:")
        print(f"  Passive+Active: {avg_pa_f1:.2%}")
        print(f"  Active-Only: {avg_ao_f1:.2%}")
        print(f"  Difference: {avg_pa_f1 - avg_ao_f1:+.2%}")
        
        print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

