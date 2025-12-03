#!/usr/bin/env python3
"""
Run Phase 3 (Active Learning Only) for all benchmarks using GrowAcq.

This script runs pure active learning with pycona's GrowAcq algorithm
directly on each benchmark oracle - no Phase 1/2 pickle files needed.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime

from cpmpy import *
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.utils import all_pairs
from pycona import ProblemInstance
from pycona.ca_environment import ActiveCAEnv

from resilient_findc import ResilientFindC
from resilient_mquacq2 import ResilientMQuAcq2
from resilient_growacq import ResilientGrowAcq
from resilient_pqgen import ResilientPQGen

# Import benchmarks
from benchmarks import (
    construct_sudoku_binary, 
    construct_jsudoku_binary, 
    construct_latin_square_binary,
    construct_graph_coloring_binary_register, 
    construct_graph_coloring_binary_scheduling,
    construct_examtt_simple,
    construct_nurse_rostering as construct_nurse_rostering_binary,
    construct_sudoku_4x4_gt_binary
)

from benchmarks_global import (
    construct_sudoku, 
    construct_jsudoku, 
    construct_latin_square,
    construct_graph_coloring_register, 
    construct_graph_coloring_scheduling,
    construct_sudoku_greater_than,
    construct_examtt_variant1, 
    construct_examtt_variant2,
    construct_nurse_rostering,
    construct_sudoku_4x4_gt
)

from utils import get_scope


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

EXPERIMENTS = {
    'sudoku_4x4': {
        'description': '4x4 Sudoku',
        'construct_binary': lambda: construct_sudoku_binary(2, 2, 4),
        'construct_global': lambda: construct_sudoku(2, 2, 4),
        'language': ['!='],
    },
    'sudoku_4x4_gt': {
        'description': '4x4 Sudoku with Greater-Than constraints',
        'construct_binary': lambda: construct_sudoku_4x4_gt_binary(2, 2, 4),
        'construct_global': lambda: construct_sudoku_4x4_gt(2, 2, 4),
        'language': ['!=', '<', '>'],
    },
    'sudoku_9x9': {
        'description': '9x9 Sudoku',
        'construct_binary': lambda: construct_sudoku_binary(3, 3, 9),
        'construct_global': lambda: construct_sudoku(3, 3, 9),
        'language': ['!='],
    },
    'sudoku_gt': {
        'description': '9x9 Sudoku with Greater-Than constraints',
        'construct_binary': lambda: construct_sudoku_binary(3, 3, 9),
        'construct_global': lambda: construct_sudoku_greater_than(3, 3, 9),
        'language': ['!=', '<', '>'],
    },
    'jsudoku': {
        'description': 'Jigsaw Sudoku 9x9',
        'construct_binary': lambda: construct_jsudoku_binary(grid_size=9),
        'construct_global': lambda: construct_jsudoku(grid_size=9),
        'language': ['!='],
    },
    'latin_square': {
        'description': 'Latin Square 9x9',
        'construct_binary': lambda: construct_latin_square_binary(n=9),
        'construct_global': lambda: construct_latin_square(n=9),
        'language': ['!='],
    },
    'graph_coloring_register': {
        'description': 'Graph Coloring (Register Allocation)',
        'construct_binary': lambda: construct_graph_coloring_binary_register(),
        'construct_global': lambda: construct_graph_coloring_register(),
        'language': ['!='],
    },
    'graph_coloring_scheduling': {
        'description': 'Graph Coloring (Scheduling)',
        'construct_binary': lambda: construct_graph_coloring_binary_scheduling(),
        'construct_global': lambda: construct_graph_coloring_scheduling(),
        'language': ['!='],
    },
    'nurse_rostering': {
        'description': 'Nurse Rostering',
        'construct_binary': lambda: construct_nurse_rostering_binary(),
        'construct_global': lambda: construct_nurse_rostering(),
        'language': ['!=', '=='],
    },
    'examtt_v1': {
        'description': 'Exam Timetabling Variant 1 (small)',
        'construct_binary': lambda: construct_examtt_simple(
            nsemesters=6, courses_per_semester=5, 
            slots_per_day=6, days_for_exams=10
        ),
        'construct_global': lambda: construct_examtt_variant1(
            nsemesters=6, courses_per_semester=5, 
            slots_per_day=6, days_for_exams=10
        ),
        'language': ['!='],
    },
    'examtt_v2': {
        'description': 'Exam Timetabling Variant 2 (medium)',
        'construct_binary': lambda: construct_examtt_simple(
            nsemesters=12, courses_per_semester=10, 
            slots_per_day=10, days_for_exams=20
        ),
        'construct_global': lambda: construct_examtt_variant2(
            nsemesters=12, courses_per_semester=10, 
            slots_per_day=10, days_for_exams=20
        ),
        'language': ['!='],
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_logging(output_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, f"active_learning_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('active_learning_runner')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def generate_binary_bias(variables, language):
    """Generate binary constraint bias for all pairs of variables."""
    
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
    
    return bias_constraints


def compute_metrics(learned_constraints, target_constraints):
    """Compute precision, recall, F1 for learned vs target constraints."""
    
    target_strs = set(str(c) for c in target_constraints)
    learned_strs = set(str(c) for c in learned_constraints)
    
    correct = len(target_strs & learned_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)
    
    precision = correct / len(learned_strs) if len(learned_strs) > 0 else 0
    recall = correct / len(target_strs) if len(target_strs) > 0 else 0
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


def compute_solution_metrics(learned_constraints, target_constraints, variables, 
                             max_solutions=100, timeout_per_model=300):
    """Compute solution-space metrics (S-Precision, S-Recall, S-F1)."""
    
    def enumerate_solutions(constraints, variables, max_sols, label):
        # Filter out boolean constraints
        filtered_constraints = [c for c in constraints if not c.is_bool()]
        
        solutions = set()
        start_time = time.time()
        count = 0
        incomplete = False
        
        try:
            while count < max_sols:
                model = Model(filtered_constraints)
                if time.time() - start_time > timeout_per_model:
                    incomplete = True
                    break
                
                result = model.solve()
                if not result:
                    break
                
                sol_tuple = tuple(v.value() for v in variables)
                solutions.add(sol_tuple)
                count += 1
                
                # Add exclusion constraint
                exclusion = [v != v.value() for v in variables]
                filtered_constraints.append(any(exclusion))
            
            if count >= max_sols:
                incomplete = True
                
            return solutions, incomplete
            
        except Exception as e:
            print(f"  [WARNING] Enumeration failed for {label}: {e}")
            return solutions, True

    learned_sols, learned_incomplete = enumerate_solutions(
        learned_constraints, variables, max_solutions, "Learned"
    )
    target_sols, target_incomplete = enumerate_solutions(
        target_constraints, variables, max_solutions, "Target"
    )

    intersection = learned_sols & target_sols
    
    s_precision = len(intersection) / len(learned_sols) if len(learned_sols) > 0 else 0.0
    s_recall = len(intersection) / len(target_sols) if len(target_sols) > 0 else 0.0
    s_f1 = 2 * s_precision * s_recall / (s_precision + s_recall) if (s_precision + s_recall) > 0 else 0.0
    
    return {
        's_precision': s_precision,
        's_recall': s_recall,
        's_f1': s_f1,
        'learned_solutions': len(learned_sols),
        'target_solutions': len(target_sols),
        'intersection_solutions': len(intersection),
        'is_complete': not (learned_incomplete or target_incomplete)
    }


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_active_learning(experiment_name, exp_config, output_dir, algorithm='growacq', 
                        verbose=2, logger=None):
    """
    Run pure active learning (GrowAcq or MQuAcq2) on a benchmark.
    
    Args:
        experiment_name: Name of the experiment
        exp_config: Experiment configuration dict
        output_dir: Directory for output files
        algorithm: 'growacq' or 'mquacq2'
        verbose: Verbosity level
        logger: Logger instance
    
    Returns:
        Dictionary with results
    """
    
    log_file = os.path.join(output_dir, f"{experiment_name}_active_learning.log")
    
    # Redirect stdout/stderr to log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        with open(log_file, 'w') as f:
            class TeeWriter:
                def __init__(self, file, stream):
                    self.file = file
                    self.stream = stream
                
                def write(self, data):
                    self.file.write(data)
                    self.stream.write(data)
                    self.file.flush()
                
                def flush(self):
                    self.file.flush()
                    self.stream.flush()
            
            sys.stdout = TeeWriter(f, original_stdout)
            sys.stderr = TeeWriter(f, original_stderr)
            
            return _run_active_learning_impl(
                experiment_name, exp_config, output_dir, algorithm, verbose, logger
            )
            
    except Exception as e:
        if logger:
            logger.error(f"Experiment {experiment_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'experiment': experiment_name,
            'status': 'failed',
            'error': str(e)
        }
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def _run_active_learning_impl(experiment_name, exp_config, output_dir, algorithm, verbose, logger):
    """Implementation of active learning run."""
    
    print(f"\n{'='*80}")
    print(f"Active Learning: {experiment_name}")
    print(f"Description: {exp_config['description']}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"{'='*80}\n")
    
    # Construct instance and oracle
    print("[1/5] Constructing benchmark instance...")
    result_binary = exp_config['construct_binary']()
    instance_binary, oracle_binary = result_binary[:2]
    
    # Set oracle variables
    oracle_binary.variables_list = cpm_array(instance_binary.X)
    
    print(f"  Variables: {len(instance_binary.X)}")
    print(f"  Target constraints: {len(oracle_binary.constraints)}")
    
    # Generate bias
    print("\n[2/5] Generating constraint bias...")
    language = exp_config.get('language', ['!='])
    bias = generate_binary_bias(instance_binary.X, language)
    print(f"  Language: {language}")
    print(f"  Bias size: {len(bias)}")
    
    # Create problem instance for active learning
    print("\n[3/5] Setting up active learning...")
    ca_instance = ProblemInstance(
        variables=cpm_array(instance_binary.X),
        init_cl=[],  # Start with empty CL
        name=f"{experiment_name}_active",
        bias=bias
    )
    
    print(f"  Problem instance created")
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
    
    # Run active learning
    print(f"\n[4/5] Running {algorithm.upper()}...")
    start_time = time.time()
    
    try:
        learned_instance = ca_system.learn(
            ca_instance, 
            oracle=oracle_binary, 
            verbose=verbose
        )
    except Exception as e:
        print(f"\n[ERROR] Active learning failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'experiment': experiment_name,
            'status': 'failed',
            'error': str(e)
        }
    
    learning_time = time.time() - start_time
    
    # Get results
    learned_constraints = learned_instance.cl
    total_queries = ca_system.env.metrics.total_queries
    
    print(f"\n[5/5] Computing metrics...")
    
    # Constraint-level metrics
    metrics = compute_metrics(learned_constraints, oracle_binary.constraints)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {experiment_name}")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Total queries: {total_queries}")
    print(f"Learning time: {learning_time:.2f}s")
    print(f"\nConstraint-Level Metrics:")
    print(f"  Target size: {metrics['target_size']}")
    print(f"  Learned size: {metrics['learned_size']}")
    print(f"  Correct: {metrics['correct']}")
    print(f"  Missing: {metrics['missing']}")
    print(f"  Spurious: {metrics['spurious']}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")
    
    # Solution-level metrics (for smaller problems)
    if len(instance_binary.X) <= 81:  # Skip for very large problems
        print(f"\nComputing solution-space metrics...")
        sol_metrics = compute_solution_metrics(
            learned_constraints, oracle_binary.constraints, 
            instance_binary.X, max_solutions=100
        )
        print(f"\nSolution-Level Metrics:")
        print(f"  Learned solutions: {sol_metrics['learned_solutions']}")
        print(f"  Target solutions: {sol_metrics['target_solutions']}")
        print(f"  Intersection: {sol_metrics['intersection_solutions']}")
        print(f"  S-Precision: {sol_metrics['s_precision']:.2%}")
        print(f"  S-Recall: {sol_metrics['s_recall']:.2%}")
        print(f"  S-F1: {sol_metrics['s_f1']:.2%}")
    else:
        print(f"\n[INFO] Skipping solution-space metrics (problem too large)")
        sol_metrics = None
    
    # Build results dictionary
    results = {
        'experiment': experiment_name,
        'description': exp_config['description'],
        'status': 'success',
        'algorithm': algorithm.upper(),
        'timestamp': datetime.now().isoformat(),
        'problem_size': {
            'variables': len(instance_binary.X),
            'target_constraints': len(oracle_binary.constraints),
            'bias_size': len(bias)
        },
        'learning': {
            'queries': total_queries,
            'time': learning_time
        },
        'constraint_metrics': metrics,
        'solution_metrics': sol_metrics
    }
    
    # Save individual results
    results_file = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results: {results_file}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Active Learning (GrowAcq/MQuAcq2) for all benchmarks'
    )
    parser.add_argument(
        '--output_dir', type=str, default='phase3_output',
        help='Directory to store outputs'
    )
    parser.add_argument(
        '--algorithm', type=str, default='growacq',
        choices=['mquacq2', 'growacq'],
        help='Active learning algorithm to use'
    )
    parser.add_argument(
        '--experiments', type=str, nargs='*', default=None,
        help='Specific experiments to run (default: all)'
    )
    parser.add_argument(
        '--skip', type=str, nargs='*', default=None,
        help='Experiments to skip'
    )
    parser.add_argument(
        '--verbose', type=int, default=2,
        help='Verbosity level (0-3)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='List experiments without running them'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all available experiments'
    )
    
    args = parser.parse_args()
    
    # List experiments and exit
    if args.list:
        print("\nAvailable experiments:")
        print("-" * 60)
        for name, config in EXPERIMENTS.items():
            print(f"  {name:<30} - {config['description']}")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger, main_log_file = setup_logging(args.output_dir)
    
    logger.info("=" * 80)
    logger.info("Active Learning Runner - All Experiments")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Main log file: {main_log_file}")
    logger.info("=" * 80)
    
    # Determine experiments to run
    experiments_to_run = []
    for name, config in EXPERIMENTS.items():
        if args.skip and name in args.skip:
            logger.info(f"  Skipping: {name}")
            continue
        if args.experiments and name not in args.experiments:
            continue
        experiments_to_run.append((name, config))
    
    logger.info(f"\nExperiments to run: {len(experiments_to_run)}")
    for i, (name, config) in enumerate(experiments_to_run, 1):
        logger.info(f"  {i}. {name} - {config['description']}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Exiting without running experiments")
        return 0
    
    # Run experiments
    results_summary = []
    total_start_time = time.time()
    
    for i, (exp_name, exp_config) in enumerate(experiments_to_run, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{len(experiments_to_run)}] Running: {exp_name}")
        logger.info(f"{'='*80}")
        
        exp_start_time = time.time()
        
        results = run_active_learning(
            experiment_name=exp_name,
            exp_config=exp_config,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            verbose=args.verbose,
            logger=logger
        )
        
        exp_elapsed = time.time() - exp_start_time
        
        if results.get('status') == 'success':
            logger.info(f"[SUCCESS] {exp_name} completed in {exp_elapsed:.2f}s")
            logger.info(f"  Queries: {results['learning']['queries']}")
            logger.info(f"  F1 Score: {results['constraint_metrics']['f1']:.2%}")
        else:
            logger.error(f"[FAILED] {exp_name}: {results.get('error', 'Unknown error')}")
        
        results['wall_clock_time'] = exp_elapsed
        results_summary.append(results)
    
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    successful = [r for r in results_summary if r.get('status') == 'success']
    failed = [r for r in results_summary if r.get('status') != 'success']
    
    logger.info(f"Total experiments: {len(results_summary)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total wall-clock time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    
    if successful:
        logger.info(f"\nSuccessful Experiments:")
        logger.info(f"{'Experiment':<30} {'Queries':>10} {'Time':>10} {'F1':>10}")
        logger.info("-" * 60)
        for r in successful:
            logger.info(f"{r['experiment']:<30} {r['learning']['queries']:>10} "
                       f"{r['learning']['time']:>10.2f}s {r['constraint_metrics']['f1']:>10.2%}")
    
    if failed:
        logger.info(f"\nFailed Experiments:")
        for r in failed:
            logger.info(f"  - {r['experiment']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"active_learning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'config': {
                'algorithm': args.algorithm,
                'output_dir': args.output_dir
            },
            'total_experiments': len(results_summary),
            'successful': len(successful),
            'failed': len(failed),
            'total_wall_clock_time': total_elapsed,
            'results': results_summary
        }, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")
    
    # Save CSV
    csv_file = os.path.join(args.output_dir, f"active_learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, 'w') as f:
        f.write("experiment,status,variables,target_constraints,bias_size,queries,time,precision,recall,f1\n")
        for r in results_summary:
            if r.get('status') == 'success':
                f.write(f"{r['experiment']},{r['status']},{r['problem_size']['variables']},"
                        f"{r['problem_size']['target_constraints']},{r['problem_size']['bias_size']},"
                        f"{r['learning']['queries']},{r['learning']['time']:.2f},"
                        f"{r['constraint_metrics']['precision']:.4f},"
                        f"{r['constraint_metrics']['recall']:.4f},"
                        f"{r['constraint_metrics']['f1']:.4f}\n")
            else:
                f.write(f"{r['experiment']},{r.get('status', 'failed')},,,,,,,,\n")
    
    logger.info(f"CSV results saved to: {csv_file}")
    logger.info(f"Main log saved to: {main_log_file}")
    
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
