#!/usr/bin/env python3
"""
Solution Variance Comparison: Passive+Active with Different Numbers of Solutions

This script runs Phase 3 (active learning) using Phase 1 results with different
numbers of solutions to analyze how the quantity of training examples affects:
- Query count in active learning
- Model quality (precision, recall, F1)
- Learning time
- Overall convergence

The script uses pre-computed Phase 1 pickles from solution_variance_output22/
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
from benchmarks_global import construct_graph_coloring_register, construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering

from benchmarks import construct_sudoku_binary, construct_jsudoku_binary
from benchmarks import construct_graph_coloring_binary_register
from benchmarks import construct_examtt_simple
from benchmarks import construct_nurse_rostering as construct_nurse_rostering_binary

from utils import get_scope


def construct_instance(experiment_name):
    """Construct both binary and global constraint instances for a benchmark."""
    
    if 'graph_coloring_register' in experiment_name.lower():
        instance_binary, oracle_binary = construct_graph_coloring_binary_register()
        result = construct_graph_coloring_register()
        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'jsudoku' in experiment_name.lower():
        instance_binary, oracle_binary = construct_jsudoku_binary(grid_size=9)
        result = construct_jsudoku(grid_size=9)
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


def run_active_learning_phase3(instance_binary, oracle_binary, init_cl, bias, 
                                experiment_name, num_solutions, algorithm='mquacq2'):
    """
    Run Phase 3 (active learning) with MQuAcq-2 or GrowAcq.
    """
    
    variables_for_ca = get_variables(init_cl + bias) if (init_cl or bias) else instance_binary.X
    
    ca_instance = ProblemInstance(
        variables=cpm_array(variables_for_ca),
        init_cl=init_cl,
        name=f"{experiment_name}_sol{num_solutions}_phase3",
        bias=bias
    )
    
    print(f"\n[PHASE 3 ACTIVE LEARNING]")
    print(f"  Algorithm: {algorithm.upper()}")
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
    
    print(f"\n[PHASE 3 COMPLETE]")
    print(f"  Queries: {total_queries}")
    print(f"  Time: {learning_time:.2f}s")
    print(f"  Learned constraints: {len(learned_constraints)}")
    
    return {
        'learned_constraints': learned_constraints,
        'queries': total_queries,
        'time': learning_time,
        'algorithm': algorithm.upper()
    }


def run_experiment_with_solution_count(benchmark_name, num_solutions, phase1_pickle_path, 
                                       base_dir, algorithm='mquacq2'):
    """
    Run passive+active experiment with specific number of solutions from Phase 1.
    """
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {benchmark_name} with {num_solutions} solutions")
    print(f"{'='*80}")
    print(f"Phase 1 pickle: {phase1_pickle_path}")
    
    # Load Phase 1 data
    if not os.path.exists(phase1_pickle_path):
        print(f"[ERROR] Phase 1 pickle not found: {phase1_pickle_path}")
        return None
    
    with open(phase1_pickle_path, 'rb') as f:
        phase1_data = pickle.load(f)
    
    B_fixed = phase1_data.get('B_fixed', [])
    E_plus = phase1_data.get('E+', [])
    CG = phase1_data.get('CG', set())
    
    print(f"\n[PHASE 1 LOADED]")
    print(f"  Solutions used: {num_solutions}")
    print(f"  E+ (examples): {len(E_plus)}")
    print(f"  CG (candidate globals): {len(CG)}")
    print(f"  B_fixed (pruned bias): {len(B_fixed)}")
    
    # Construct instances
    instance_binary, oracle_binary, instance_global, oracle_global = construct_instance(benchmark_name)
    oracle_binary.variables_list = cpm_array(instance_binary.X)
    
    print(f"\n[BENCHMARK INFO]")
    print(f"  Variables: {len(instance_binary.X)}")
    print(f"  Target constraints: {len(oracle_binary.constraints)}")
    
    # For this comparison, we skip Phase 2 (no validated globals)
    # We go directly to Phase 3 with empty init_cl and Phase 1 bias
    init_cl = []
    bias = B_fixed
    
    print(f"\n[PHASE 3 SETUP] (Direct from Phase 1, skipping Phase 2)")
    print(f"  Initial CL: {len(init_cl)} (empty - no Phase 2 refinement)")
    print(f"  Bias: {len(bias)} (from Phase 1 with {num_solutions} solutions)")
    
    # Run Phase 3
    result = run_active_learning_phase3(instance_binary, oracle_binary, init_cl, bias, 
                                        benchmark_name, num_solutions, algorithm)
    
    if result is None:
        return None
    
    # Evaluate results
    metrics = compute_metrics(result['learned_constraints'], oracle_binary.constraints)
    
    print(f"\n[EVALUATION]")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1-Score: {metrics['f1']:.2%}")
    print(f"  Queries: {result['queries']}")
    print(f"  Time: {result['time']:.2f}s")
    
    return {
        'benchmark': benchmark_name,
        'num_solutions': num_solutions,
        'phase1': {
            'examples': len(E_plus),
            'candidate_globals': len(CG),
            'pruned_bias': len(B_fixed)
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


def discover_solution_variants(base_dir='solution_variance_output22'):
    """
    Discover all available solution variance experiments from the directory.
    
    Returns a dictionary mapping benchmark names to lists of (num_solutions, pickle_path) tuples.
    """
    
    variants = {}
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Base directory not found: {base_dir}")
        return variants
    
    # Scan directory for solution variance subdirectories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # Parse directory name: {benchmark}_sol{num}_overfitted{num}
        parts = item.split('_sol')
        if len(parts) != 2:
            continue
        
        benchmark_name = parts[0]
        sol_parts = parts[1].split('_overfitted')
        if len(sol_parts) != 2:
            continue
        
        try:
            num_solutions = int(sol_parts[0])
        except ValueError:
            continue
        
        # Find the pickle file
        pickle_name = f"{benchmark_name}_phase1.pkl"
        pickle_path = os.path.join(item_path, pickle_name)
        
        if os.path.exists(pickle_path):
            if benchmark_name not in variants:
                variants[benchmark_name] = []
            variants[benchmark_name].append((num_solutions, pickle_path))
    
    # Sort by number of solutions
    for benchmark in variants:
        variants[benchmark].sort(key=lambda x: x[0])
    
    return variants


def main():
    """Main function to run solution variance comparison experiments."""
    
    import argparse
    parser = argparse.ArgumentParser(
        description='Compare passive+active performance with different numbers of solutions'
    )
    parser.add_argument('--base_dir', type=str, default='solution_variance_output22',
                       help='Base directory containing solution variance pickles')
    parser.add_argument('--algorithm', type=str, default='mquacq2', 
                       choices=['mquacq2', 'growacq'],
                       help='Active learning algorithm to use')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=None,
                       help='Specific benchmarks to run (default: all available)')
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print(f"# SOLUTION VARIANCE COMPARISON")
    print(f"# Algorithm: {args.algorithm.upper()}")
    print(f"# Base directory: {args.base_dir}")
    print(f"{'#'*80}\n")
    
    # Discover available solution variants
    print(f"[DISCOVERY] Scanning {args.base_dir} for solution variance experiments...")
    variants = discover_solution_variants(args.base_dir)
    
    if not variants:
        print(f"[ERROR] No solution variance experiments found in {args.base_dir}")
        sys.exit(1)
    
    # Filter benchmarks if specified
    if args.benchmarks:
        variants = {k: v for k, v in variants.items() if k in args.benchmarks}
    
    print(f"\n[AVAILABLE BENCHMARKS]")
    for benchmark, sol_variants in sorted(variants.items()):
        sol_counts = [num for num, _ in sol_variants]
        print(f"  {benchmark}: {len(sol_variants)} variants - solutions: {sol_counts}")
    
    if not variants:
        print(f"[ERROR] No matching benchmarks found")
        sys.exit(1)
    
    print(f"\n{'#'*80}")
    print(f"# Starting experiments...")
    print(f"{'#'*80}\n")
    
    all_results = []
    
    # Run experiments for each benchmark and solution count
    for benchmark_idx, (benchmark, sol_variants) in enumerate(sorted(variants.items()), 1):
        print(f"\n\n{'#'*80}")
        print(f"# BENCHMARK {benchmark_idx}/{len(variants)}: {benchmark}")
        print(f"# Solution variants: {[num for num, _ in sol_variants]}")
        print(f"{'#'*80}\n")
        
        benchmark_results = []
        
        for sol_idx, (num_solutions, pickle_path) in enumerate(sol_variants, 1):
            print(f"\n{'-'*80}")
            print(f"Variant {sol_idx}/{len(sol_variants)}: {num_solutions} solutions")
            print(f"{'-'*80}")
            
            try:
                result = run_experiment_with_solution_count(
                    benchmark, num_solutions, pickle_path, args.base_dir, args.algorithm
                )
                if result:
                    benchmark_results.append(result)
            except Exception as e:
                print(f"\n[ERROR] Failed on {benchmark} with {num_solutions} solutions: {e}")
                import traceback
                traceback.print_exc()
                benchmark_results.append({
                    'benchmark': benchmark,
                    'num_solutions': num_solutions,
                    'error': str(e)
                })
        
        # Print benchmark summary
        if benchmark_results:
            print(f"\n{'='*80}")
            print(f"BENCHMARK SUMMARY: {benchmark}")
            print(f"{'='*80}")
            print(f"\n{'Solutions':<12} {'Queries':<12} {'Time (s)':<12} {'F1-Score':<12} {'Bias Size':<12}")
            print(f"{'-'*72}")
            
            for res in benchmark_results:
                if 'error' not in res:
                    print(f"{res['num_solutions']:<12} {res['total']['queries']:<12} "
                          f"{res['total']['time']:<12.2f} {res['evaluation']['f1']:<12.2%} "
                          f"{res['phase1']['pruned_bias']:<12}")
            print(f"{'-'*72}")
            
            all_results.extend(benchmark_results)
    
    # Generate final summary
    print(f"\n\n{'#'*80}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*80}\n")
    
    # Group results by benchmark
    results_by_benchmark = {}
    for result in all_results:
        if 'error' in result:
            continue
        benchmark = result['benchmark']
        if benchmark not in results_by_benchmark:
            results_by_benchmark[benchmark] = []
        results_by_benchmark[benchmark].append(result)
    
    # Print summary table
    print(f"{'Benchmark':<25} {'Solutions':<12} {'Queries':<12} {'Time (s)':<12} {'F1-Score':<12}")
    print(f"{'='*73}")
    
    for benchmark in sorted(results_by_benchmark.keys()):
        results = sorted(results_by_benchmark[benchmark], key=lambda x: x['num_solutions'])
        for i, res in enumerate(results):
            bench_name = benchmark if i == 0 else ''
            print(f"{bench_name:<25} {res['num_solutions']:<12} {res['total']['queries']:<12} "
                  f"{res['total']['time']:<12.2f} {res['evaluation']['f1']:<12.2%}")
        print(f"{'-'*73}")
    
    # Save results
    output_dir = "solution_variance_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"solution_variance_{args.algorithm}_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'algorithm': args.algorithm.upper(),
            'base_dir': args.base_dir,
            'benchmarks': len(results_by_benchmark),
            'total_experiments': len(all_results),
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n[SAVED] Results saved to: {results_file}")
    
    # Generate analysis
    print(f"\n{'='*80}")
    print(f"ANALYSIS: Effect of Number of Solutions")
    print(f"{'='*80}\n")
    
    for benchmark in sorted(results_by_benchmark.keys()):
        results = sorted(results_by_benchmark[benchmark], key=lambda x: x['num_solutions'])
        if len(results) < 2:
            continue
        
        print(f"\n{benchmark}:")
        min_sol_res = results[0]
        max_sol_res = results[-1]
        
        query_reduction = min_sol_res['total']['queries'] - max_sol_res['total']['queries']
        query_reduction_pct = (query_reduction / min_sol_res['total']['queries'] * 100) if min_sol_res['total']['queries'] > 0 else 0
        f1_improvement = max_sol_res['evaluation']['f1'] - min_sol_res['evaluation']['f1']
        
        print(f"  From {min_sol_res['num_solutions']} to {max_sol_res['num_solutions']} solutions:")
        print(f"    Query reduction: {query_reduction:+d} ({query_reduction_pct:+.1f}%)")
        print(f"    F1 improvement: {f1_improvement:+.2%}")
        print(f"    Time change: {max_sol_res['total']['time'] - min_sol_res['total']['time']:+.2f}s")
        print(f"    Bias change: {min_sol_res['phase1']['pruned_bias'] - max_sol_res['phase1']['pruned_bias']:+d} constraints")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

