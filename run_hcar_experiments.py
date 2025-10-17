"""
Run HCAR Experiments
====================

Example script demonstrating how to run HCAR experiments on benchmarks.
This integrates the advanced HCAR framework with existing benchmark problems.

Usage:
    python run_hcar_experiments.py --benchmark sudoku --method HCAR-Advanced
    python run_hcar_experiments.py --all  # Run all experiments
"""

import argparse
import json
import os
import sys
from typing import Dict, List
import logging

from hcar_advanced import (
    HCARFramework,
    HCARConfig,
    ExperimentRunner,
    OracleResponse
)

# Import existing benchmarks (adapt to your structure)
try:
    # Import benchmarks with global constraints
    from benchmarks_global import sudoku as sudoku_global
    from benchmarks_global import uefa as uefa_global
    from benchmarks_global import vm_allocation as vm_allocation_global
    from benchmarks_global import exam_timetabling as exam_timetabling_global
    from benchmarks_global import nurse_rostering as nurse_rostering_global
    
    # Import benchmarks with only fixed-arity constraints
    from benchmarks import sudoku as sudoku_binary
    from benchmarks import uefa as uefa_binary
    from benchmarks import vm_allocation as vm_allocation_binary
    from benchmarks import exam_timetabling as exam_timetabling_binary
    from benchmarks import nurse_rostering as nurse_rostering_binary
    
    from pycona import ConstraintOracle
except ImportError as e:
    logging.warning(f"Could not import benchmarks: {e}")
    logging.warning("Please ensure benchmark modules are available")


logger = logging.getLogger(__name__)


class BenchmarkConfig:
    """Configuration for each benchmark problem."""
    
    def __init__(self, name: str, constructor_func, num_examples: int = 5, **constructor_params):
        self.name = name
        self.constructor_func = constructor_func
        self.num_examples = num_examples
        self.constructor_params = constructor_params
    
    def load(self):
        """Load benchmark problem definition."""
        logger.info(f"Loading benchmark: {self.name}")
        
        try:
            # Call the constructor function to get the ProblemInstance and Oracle
            instance, oracle = self.constructor_func(**self.constructor_params)
            
            # Extract variables from the ProblemInstance
            # instance.variables is a CPMpy array/variable structure
            variables_array = instance.variables
            
            # Flatten variables to a list if it's an array
            import numpy as np
            if isinstance(variables_array, np.ndarray):
                variables_list = list(variables_array.flatten())
            elif hasattr(variables_array, '__iter__'):
                variables_list = list(variables_array)
            else:
                variables_list = [variables_array]
            
            # Create variables dict (name -> variable mapping)
            variables = {}
            for var in variables_list:
                if hasattr(var, 'name'):
                    variables[var.name] = var
            
            # Extract domains from variables
            domains = {}
            for var_name, var in variables.items():
                if hasattr(var, 'lb') and hasattr(var, 'ub'):
                    # lb and ub are properties, not methods
                    lb_val = var.lb if not callable(var.lb) else var.lb()
                    ub_val = var.ub if not callable(var.ub) else var.ub()
                    domains[var_name] = (lb_val, ub_val)
                else:
                    # Default domain for boolean variables
                    domains[var_name] = (0, 1)
            
            # Get target model constraints from oracle
            # The ConstraintOracle stores constraints in .constraints attribute
            if hasattr(oracle, 'C_T'):
                target_model = oracle.C_T
            elif hasattr(oracle, 'constraints'):
                target_model = oracle.constraints
            else:
                target_model = []
            
            # Create oracle function compatible with HCAR
            def oracle_func(assignment: Dict) -> OracleResponse:
                """
                Oracle function that checks if assignment is valid using a fresh CP model.

                This creates a new CP model with:
                1. All target constraints (ground truth)
                2. Assignment constraints (fixing variables to assigned values)

                Then checks if the model is satisfiable.
                """
                try:
                    from cpmpy import Model

                    # Create a fresh model with target constraints
                    # This ensures we're checking against the ground truth, not learned constraints
                    oracle_model = Model(target_model)

                    # Add assignment constraints (fix variables to their assigned values)
                    for var_name, value in assignment.items():
                        if var_name in variables:
                            var = variables[var_name]
                            oracle_model += (var == value)

                    # Solve the model
                    # If SAT: assignment is consistent with target constraints → VALID
                    # If UNSAT: assignment violates at least one target constraint → INVALID
                    is_sat = oracle_model.solve()

                    if is_sat:
                        # Double-check: verify the solution actually matches the assignment
                        # (this should always be true if the solver is correct)
                        all_match = True
                        mismatches = []

                        for var_name, expected_value in assignment.items():
                            if var_name in variables:
                                actual_value = variables[var_name].value()
                                if actual_value != expected_value:
                                    all_match = False
                                    mismatches.append(f"{var_name}: expected={expected_value}, actual={actual_value}")

                        if not all_match:
                            # This should never happen - indicates a solver bug
                            logger.error(f"ORACLE ERROR: Solution doesn't match assignment!")
                            logger.error(f"  Mismatches (first 5): {mismatches[:5]}")
                            return OracleResponse.INVALID

                        logger.debug(f"Oracle: Assignment is VALID (satisfies all target constraints)")
                        return OracleResponse.VALID
                    else:
                        logger.debug(f"Oracle: Assignment is INVALID (violates target constraints)")
                        return OracleResponse.INVALID

                except Exception as e:
                    logger.error(f"Error in oracle query: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return OracleResponse.INVALID
            
            # Generate initial positive examples
            import cpmpy as cp
            positive_examples = []

            # Special handling for Sudoku: use predefined diverse examples
            # to ensure comprehensive constraint coverage in passive learning
            if self.name == "Sudoku":
                try:
                    from sudoku_example_cache import get_sudoku_examples
                    positive_examples = get_sudoku_examples(grid_size=9)
                    logger.info(f"Using {len(positive_examples)} predefined Sudoku examples")
                    logger.info("Predefined examples ensure discovery of all row/column/block constraints")
                except ImportError:
                    logger.warning("Could not import sudoku_example_cache, falling back to random generation")

            # If Sudoku cache loaded successfully, skip random generation
            if positive_examples:
                pass  # Use cached examples

            # Otherwise, generate random examples
            else:
                try:
                    # Create a model with the target constraints
                    model = cp.Model(target_model)

                    # Try to solve the model once to verify it's satisfiable
                    logger.info(f"Testing if benchmark is satisfiable...")
                    if not model.solve():
                        logger.error(f"Benchmark {self.name} is UNSAT - cannot generate examples")
                        return None

                    # First solution found, extract it
                    example = {}
                    exclusion_constraints = []

                    for var_name, var in variables.items():
                        if hasattr(var, 'value'):
                            val = var.value()
                            if val is not None:
                                example[var_name] = val
                                exclusion_constraints.append(var != val)

                    if example:
                        positive_examples.append(example)
                        logger.info(f"Generated example 1/{self.num_examples}")

                    # Generate additional solutions
                    for i in range(1, self.num_examples):
                        if not exclusion_constraints:
                            logger.warning("No variables to constrain for diversity")
                            break

                        try:
                            # Add constraint to exclude previous solution
                            model += cp.any(exclusion_constraints)
                        except Exception as e:
                            logger.warning(f"Could not add exclusion constraint: {e}")
                            logger.info(f"Will use {len(positive_examples)} examples instead of {self.num_examples}")
                            break

                        # Try to find another solution
                        if model.solve():
                            example = {}
                            new_exclusion = []

                            for var_name, var in variables.items():
                                if hasattr(var, 'value'):
                                    val = var.value()
                                    if val is not None:
                                        example[var_name] = val
                                        new_exclusion.append(var != val)

                            if example:
                                positive_examples.append(example)
                                exclusion_constraints = new_exclusion
                                logger.info(f"Generated example {len(positive_examples)}/{self.num_examples}")
                        else:
                            logger.info(f"No more solutions found after {len(positive_examples)} examples")
                            break
                
                except Exception as e:
                    logger.error(f"Error during example generation: {e}", exc_info=True)
                    # If we failed completely, return None
                    if not positive_examples:
                        return None
            
            if len(positive_examples) < self.num_examples:
                logger.warning(f"Only generated {len(positive_examples)}/{self.num_examples} examples")
            
            # Need at least 1 example to proceed
            if not positive_examples:
                logger.error(f"Could not generate any positive examples for {self.name}")
                return None
            
            return {
                'variables': variables,
                'domains': domains,
                'target_model': target_model,
                'oracle_func': oracle_func,
                'positive_examples': positive_examples
            }
        
        except Exception as e:
            logger.error(f"Error loading benchmark {self.name}: {e}", exc_info=True)
            return None


def get_benchmark_configs(use_global_constraints=True):
    """
    Get all benchmark configurations.
    
    Args:
        use_global_constraints: If True, use benchmarks_global/ (with global constraints),
                              otherwise use benchmarks/ (only fixed-arity constraints)
    """
    configs = []
    
    # Select the appropriate benchmark modules
    if use_global_constraints:
        logger.info("Using benchmarks with GLOBAL constraints (benchmarks_global/)")
        benchmarks = [
            ('Sudoku', sudoku_global.construct_sudoku, {'block_size_row': 3, 'block_size_col': 3, 'grid_size': 9}),
            ('UEFA', uefa_global.construct_uefa, {
                'teams_data': {
                    'Team1': {'country': 'A', 'coefficient': 90},
                    'Team2': {'country': 'B', 'coefficient': 85},
                    'Team3': {'country': 'C', 'coefficient': 80},
                    'Team4': {'country': 'D', 'coefficient': 75},
                    'Team5': {'country': 'A', 'coefficient': 70},
                    'Team6': {'country': 'B', 'coefficient': 65},
                    'Team7': {'country': 'E', 'coefficient': 60},
                    'Team8': {'country': 'F', 'coefficient': 55},
                },
                'n_groups': 2,
                'teams_per_group': 4
            }),
            ('VM_Allocation', vm_allocation_global.construct_vm_allocation, {
                'pm_data': {
                    'PM1': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
                    'PM2': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
                    'PM3': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
                },
                'vm_data': {
                    'VM1': {'demand_cpu': 2, 'demand_memory': 4, 'demand_disk': 20, 'availability_zone': 'AZ1', 'priority': 1},
                    'VM2': {'demand_cpu': 3, 'demand_memory': 6, 'demand_disk': 30, 'availability_zone': 'AZ1', 'priority': 2},
                    'VM3': {'demand_cpu': 2, 'demand_memory': 4, 'demand_disk': 20, 'availability_zone': 'AZ2', 'priority': 1},
                    'VM4': {'demand_cpu': 4, 'demand_memory': 8, 'demand_disk': 40, 'availability_zone': 'AZ2', 'priority': 2},
                }
            }),
            ('Exam_Timetabling', exam_timetabling_global.construct_examtt_simple, {
                'nsemesters': 3,
                'courses_per_semester': 3,
                'slots_per_day': 9,
                'days_for_exams': 7
            }),
            ('Nurse_Rostering', nurse_rostering_global.construct_nurse_rostering, {
                'shifts_per_day': 3,
                'num_days': 7,
                'num_nurses': 8,
                'nurses_per_shift': 2,
                'max_workdays': 6  # Fixed: Was 5, which made problem UNSAT (40 capacity < 42 needed)
            }),
        ]
    else:
        logger.info("Using benchmarks with FIXED-ARITY constraints only (benchmarks/)")
        benchmarks = [
            ('Sudoku', sudoku_binary.construct_sudoku_binary, {'block_size_row': 3, 'block_size_col': 3, 'grid_size': 9}),
            ('UEFA', uefa_binary.construct_uefa, {
                'teams_data': {
                    'Team1': {'country': 'A', 'coefficient': 90},
                    'Team2': {'country': 'B', 'coefficient': 85},
                    'Team3': {'country': 'C', 'coefficient': 80},
                    'Team4': {'country': 'D', 'coefficient': 75},
                    'Team5': {'country': 'A', 'coefficient': 70},
                    'Team6': {'country': 'B', 'coefficient': 65},
                    'Team7': {'country': 'E', 'coefficient': 60},
                    'Team8': {'country': 'F', 'coefficient': 55},
                },
                'n_groups': 2,
                'teams_per_group': 4
            }),
            ('VM_Allocation', vm_allocation_binary.construct_vm_allocation, {
                'pm_data': {
                    'PM1': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
                    'PM2': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
                    'PM3': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
                },
                'vm_data': {
                    'VM1': {'demand_cpu': 2, 'demand_memory': 4, 'demand_disk': 20, 'availability_zone': 'AZ1', 'priority': 1},
                    'VM2': {'demand_cpu': 3, 'demand_memory': 6, 'demand_disk': 30, 'availability_zone': 'AZ1', 'priority': 2},
                    'VM3': {'demand_cpu': 2, 'demand_memory': 4, 'demand_disk': 20, 'availability_zone': 'AZ2', 'priority': 1},
                    'VM4': {'demand_cpu': 4, 'demand_memory': 8, 'demand_disk': 40, 'availability_zone': 'AZ2', 'priority': 2},
                }
            }),
            ('Exam_Timetabling', exam_timetabling_binary.construct_examtt_simple, {
                'nsemesters': 3,
                'courses_per_semester': 3,
                'slots_per_day': 9,
                'days_for_exams': 7
            }),
            ('Nurse_Rostering', nurse_rostering_binary.construct_nurse_rostering, {
                'shifts_per_day': 3,
                'num_days': 7,
                'num_nurses': 8,
                'nurses_per_shift': 2,
                'max_workdays': 6  # Fixed: Was 5, which made problem UNSAT (40 capacity < 42 needed)
            }),
        ]
    
    for name, constructor_func, params in benchmarks:
        try:
            configs.append(BenchmarkConfig(name, constructor_func, num_examples=5, **params))
        except Exception as e:
            logger.warning(f"Could not configure {name}: {e}")
    
    return configs


def run_single_experiment(
    benchmark_name: str,
    method_name: str,
    config: HCARConfig,
    num_runs: int = 1,
    use_global_constraints: bool = True
):
    """
    Run a single experiment configuration.
    
    Args:
        benchmark_name: Name of benchmark (e.g., "Sudoku")
        method_name: Method variant (e.g., "HCAR-Advanced")
        config: HCAR configuration
        num_runs: Number of repetitions
        use_global_constraints: Whether to use benchmarks with global constraints
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {benchmark_name} | {method_name}")
    logger.info(f"{'='*70}\n")
    
    # Load benchmark
    benchmark_configs = {bc.name: bc for bc in get_benchmark_configs(use_global_constraints)}
    
    if benchmark_name not in benchmark_configs:
        logger.error(f"Benchmark {benchmark_name} not found")
        return None
    
    benchmark_data = benchmark_configs[benchmark_name].load()
    if benchmark_data is None:
        logger.error(f"Failed to load benchmark {benchmark_name}")
        return None
    
    # Run experiment
    runner = ExperimentRunner(output_dir="hcar_results")
    
    results = runner.run_experiment(
        benchmark_name=benchmark_name,
        method_name=method_name,
        positive_examples=benchmark_data['positive_examples'],
        oracle_func=benchmark_data['oracle_func'],
        variables=benchmark_data['variables'],
        domains=benchmark_data['domains'],
        target_model=benchmark_data['target_model'],
        num_runs=num_runs,
        config=config  # Pass the config through
    )
    
    # Print summary
    print_results_summary(benchmark_name, method_name, results)
    
    return results


def print_results_summary(benchmark: str, method: str, results: Dict):
    """Print formatted results summary."""
    print(f"\n{'='*70}")
    print(f"Results: {benchmark} | {method}")
    print(f"{'='*70}")
    
    if not results:
        print("No results available")
        return
    
    # Model Quality
    print("\nModel Quality:")
    print(f"  S-Precision: {results.get('s_precision_mean', 0):.1f}% "
          f"(+/-{results.get('s_precision_std', 0):.1f})")
    print(f"  S-Recall:    {results.get('s_recall_mean', 0):.1f}% "
          f"(+/-{results.get('s_recall_std', 0):.1f})")

    # Query Efficiency
    print("\nQuery Efficiency:")
    print(f"  Phase 2 (Q2): {results.get('queries_phase2_mean', 0):.0f} "
          f"(+/-{results.get('queries_phase2_std', 0):.0f})")
    print(f"  Phase 3 (Q3): {results.get('queries_phase3_mean', 0):.0f} "
          f"(+/-{results.get('queries_phase3_std', 0):.0f})")
    print(f"  Total (QSum):  {results.get('queries_total_mean', 0):.0f} "
          f"(+/-{results.get('queries_total_std', 0):.0f})")

    # Computational Cost
    print("\nComputational Cost:")
    print(f"  Time: {results.get('time_seconds_mean', 0):.1f}s "
          f"(+/-{results.get('time_seconds_std', 0):.1f}s)")

    # Model Size
    print("\nLearned Model:")
    print(f"  Global constraints: {results.get('num_global_constraints_mean', 0):.0f}")
    print(f"  Fixed constraints:  {results.get('num_fixed_constraints_mean', 0):.0f}")
    print(f"  Total:             {results.get('total_constraints_mean', 0):.0f}")
    
    print(f"\n{'='*70}\n")


def create_method_config(method_name: str) -> HCARConfig:
    """Create configuration for specific method variant."""
    config = HCARConfig(
        total_budget=500,
        max_time_seconds=1800.0,
        query_timeout=30.0,
        theta_min=0.15,
        theta_max=0.85,
        alpha=0.1,
        max_subset_depth=3,
        base_budget_per_constraint=10,
        uncertainty_weight=0.5,
        enable_ml_prior=True,
        use_intelligent_subsets=True,
        inject_overfitted=False  # Set to True to test overfitting detection
    )

    # Adjust config based on method variant
    if method_name == "HCAR-Advanced":
        # Use all advanced features (default)
        # - Intelligent subset exploration with culprit scores
        # - ML prior estimation
        # Enable overfitting injection to demonstrate Phase 2 correction
        config.inject_overfitted = True
        logger.info("Injecting overfitted constraints to demonstrate Phase 2 correction")

    elif method_name == "HCAR-Heuristic":
        # Use positional heuristics for subset exploration (baseline)
        config.use_intelligent_subsets = False
        logger.info("Using positional heuristic subset exploration (first/middle/last)")

    elif method_name == "HCAR-NoRefine":
        # Skip Phase 2 entirely (ablation study)
        config.total_budget = 0
        # Enable overfitting injection to show what happens without Phase 2
        config.inject_overfitted = True
        logger.info("Skipping interactive refinement phase (overfitted constraints will remain)")

    elif method_name == "MQuAcq-2":
        # Pure active learning (different code path)
        logger.info("Using pure active learning baseline")

    else:
        logger.warning(f"Unknown method: {method_name}, using default config")

    return config


def run_full_comparison(num_runs: int = 3, use_global_constraints: bool = True):
    """
    Run full experimental comparison as described in the paper.
    
    Compares:
    - HCAR-Advanced (proposed)
    - HCAR-Heuristic (baseline)
    - HCAR-NoRefine (ablation)
    - MQuAcq-2 (external baseline)
    
    On all benchmarks:
    - Sudoku (9x9)
    - UEFA Champions League Scheduling
    - Cloud VM Allocation
    - University Exam Timetabling
    - Nurse Rostering
    
    Args:
        num_runs: Number of runs per configuration
        use_global_constraints: Whether to use benchmarks with global constraints
    """
    benchmarks = [
        "Sudoku",
        "UEFA",
        "VM_Allocation",
        "Exam_Timetabling",
        "Nurse_Rostering"
    ]
    
    methods = [
        "HCAR-Advanced",
        "HCAR-Heuristic",
        "HCAR-NoRefine",
        # "MQuAcq-2"  # Optional - may timeout on complex problems
    ]
    
    logger.info(f"\n{'#'*70}")
    logger.info("HCAR Full Experimental Comparison")
    logger.info(f"{'#'*70}")
    logger.info(f"Benchmarks: {len(benchmarks)}")
    logger.info(f"Methods: {len(methods)}")
    logger.info(f"Runs per configuration: {num_runs}")
    logger.info(f"Total experiments: {len(benchmarks) * len(methods)}")
    logger.info(f"{'#'*70}\n")
    
    all_results = {}
    
    for benchmark in benchmarks:
        all_results[benchmark] = {}
        
        for method in methods:
            config = create_method_config(method)
            
            try:
                results = run_single_experiment(
                    benchmark_name=benchmark,
                    method_name=method,
                    config=config,
                    num_runs=num_runs,
                    use_global_constraints=use_global_constraints
                )
                all_results[benchmark][method] = results
            
            except Exception as e:
                logger.error(f"Experiment failed: {benchmark} | {method}")
                logger.error(f"Error: {e}")
                all_results[benchmark][method] = None
    
    # Save comprehensive results
    output_dir = "hcar_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "full_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✅ Full results saved to: {output_file}")
    
    # Generate comparison table
    generate_comparison_table(all_results)
    
    return all_results


def generate_comparison_table(results: Dict):
    """Generate LaTeX-style comparison table from results."""
    print("\n" + "="*80)
    print("COMPARISON TABLE (as in paper Table 1)")
    print("="*80 + "\n")
    
    # Header
    print(f"{'Benchmark':<20} | {'Method':<18} | {'S-Prec':<7} | {'S-Rec':<7} | "
          f"{'Q₂':<6} | {'Q₃':<6} | {'Q_Σ':<6} | {'T(s)':<7}")
    print("-" * 80)
    
    # Data rows
    for benchmark, methods in results.items():
        first_row = True
        for method, data in methods.items():
            if data is None:
                continue
            
            bench_name = benchmark if first_row else ""
            first_row = False
            
            print(f"{bench_name:<20} | {method:<18} | "
                  f"{data.get('s_precision_mean', 0):>6.0f}% | "
                  f"{data.get('s_recall_mean', 0):>6.0f}% | "
                  f"{data.get('queries_phase2_mean', 0):>6.0f} | "
                  f"{data.get('queries_phase3_mean', 0):>6.0f} | "
                  f"{data.get('queries_total_mean', 0):>6.0f} | "
                  f"{data.get('time_seconds_mean', 0):>7.1f}")
        
        print("-" * 80)
    
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run HCAR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python run_hcar_experiments.py --benchmark Sudoku --method HCAR-Advanced
  
  # Run full comparison (all benchmarks x all methods)
  python run_hcar_experiments.py --all --runs 3
  
  # Run specific benchmark with all methods
  python run_hcar_experiments.py --benchmark UEFA --compare
        """
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        help='Benchmark name (Sudoku, UEFA, VM_Allocation, etc.)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='HCAR-Advanced',
        choices=['HCAR-Advanced', 'HCAR-Heuristic', 'HCAR-NoRefine', 'MQuAcq-2'],
        help='Method variant to run'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run full experimental comparison'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all methods on specified benchmark'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of runs per configuration'
    )
    
    parser.add_argument(
        '--use-global',
        action='store_true',
        default=True,
        help='Use benchmarks with global constraints (benchmarks_global/)'
    )
    
    parser.add_argument(
        '--use-binary',
        action='store_true',
        help='Use benchmarks with only fixed-arity constraints (benchmarks/)'
    )
    
    args = parser.parse_args()
    
    # Determine which benchmark type to use
    if args.use_binary:
        use_global_constraints = False
    else:
        use_global_constraints = args.use_global
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute based on arguments
    if args.all:
        # Run full comparison
        run_full_comparison(num_runs=args.runs, use_global_constraints=use_global_constraints)
    
    elif args.compare and args.benchmark:
        # Compare all methods on one benchmark
        methods = ['HCAR-Advanced', 'HCAR-Heuristic', 'HCAR-NoRefine']
        
        results = {}
        for method in methods:
            config = create_method_config(method)
            results[method] = run_single_experiment(
                benchmark_name=args.benchmark,
                method_name=method,
                config=config,
                num_runs=args.runs,
                use_global_constraints=use_global_constraints
            )
        
        # Show comparison
        generate_comparison_table({args.benchmark: results})
    
    elif args.benchmark:
        # Run single experiment
        config = create_method_config(args.method)
        run_single_experiment(
            benchmark_name=args.benchmark,
            method_name=args.method,
            config=config,
            num_runs=args.runs,
            use_global_constraints=use_global_constraints
        )
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

