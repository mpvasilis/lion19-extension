"""
Test the S-Precision calculation to identify the bug.
"""

import cpmpy as cp
from benchmarks_global.sudoku import construct_sudoku
from hcar_advanced import ExperimentRunner
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("="*80)
print("S-PRECISION CALCULATION TEST")
print("="*80)

# Load Sudoku benchmark
instance, oracle = construct_sudoku(3, 3, 9)
grid = instance.variables

# Get variables dict
import numpy as np
variables_list = list(grid.flatten())
variables = {}
for var in variables_list:
    if hasattr(var, 'name'):
        variables[var.name] = var

print(f"\nVariables: {len(variables)}")
print(f"Sample variable names: {list(variables.keys())[:5]}")

# Get target model (regular constraints only - no mocks)
target_constraints = oracle.constraints
print(f"\nTarget constraints: {len(target_constraints)}")

# Create a simple learned model (just the first 10 constraints)
learned_model = target_constraints[:10]
print(f"Learned model constraints: {len(learned_model)}")

# Get domains
domains = {}
for var_name, var in variables.items():
    domains[var_name] = (var.lb, var.ub)

print("\n" + "="*80)
print("TESTING SOLUTION SAMPLING")
print("="*80)

runner = ExperimentRunner(output_dir="test_output")

# Test 1: Sample solutions from target model
print("\n1. Sampling solutions from TARGET model...")
target_solutions = runner._sample_solutions(target_constraints, variables, domains, num_samples=5)
print(f"   Generated {len(target_solutions)} target solutions")

if target_solutions:
    sol = target_solutions[0]
    print(f"   Sample solution has {len(sol)} variables")
    print(f"   Sample keys: {list(sol.keys())[:5]}")

# Test 2: Sample solutions from learned model
print("\n2. Sampling solutions from LEARNED model...")
learned_solutions = runner._sample_solutions(learned_model, variables, domains, num_samples=5)
print(f"   Generated {len(learned_solutions)} learned solutions")

if learned_solutions:
    sol = learned_solutions[0]
    print(f"   Sample solution has {len(sol)} variables")
    print(f"   Sample keys: {list(sol.keys())[:5]}")

print("\n" + "="*80)
print("TESTING SOLUTION VALIDATION")
print("="*80)

# Set the variables for validation
runner.variables = variables
print(f"\nrunner.variables set: {len(runner.variables)} variables")
print(f"Sample runner.variables keys: {list(runner.variables.keys())[:5]}")

if learned_solutions and target_solutions:
    # Test 3: Validate a learned solution against target
    print("\n3. Validating LEARNED solutions against TARGET model...")
    for i, sol in enumerate(learned_solutions[:3]):
        is_valid = runner._is_valid_solution(sol, target_constraints)
        print(f"   Solution {i+1}: {is_valid}")

        # Debug: check if variable names match
        sol_keys = set(sol.keys())
        runner_keys = set(runner.variables.keys())
        if sol_keys != runner_keys:
            missing_in_runner = sol_keys - runner_keys
            missing_in_sol = runner_keys - sol_keys
            print(f"      [WARNING] Variable mismatch!")
            if missing_in_runner:
                print(f"      Missing in runner.variables: {list(missing_in_runner)[:5]}")
            if missing_in_sol:
                print(f"      Missing in solution: {list(missing_in_sol)[:5]}")

    # Test 4: Validate a target solution against target
    print("\n4. Validating TARGET solutions against TARGET model (should all be True)...")
    for i, sol in enumerate(target_solutions[:3]):
        is_valid = runner._is_valid_solution(sol, target_constraints)
        print(f"   Solution {i+1}: {is_valid}")

print("\n" + "="*80)
print("TESTING FULL EVALUATION")
print("="*80)

# Test 5: Run full evaluation
print("\n5. Running evaluate_model_quality()...")
result = runner.evaluate_model_quality(
    learned_model=learned_model,
    target_model=target_constraints,
    variables=variables,
    domains=domains,
    num_samples=10
)

print(f"\nResults:")
print(f"   S-Precision: {result['s_precision']:.1f}%")
print(f"   S-Recall: {result['s_recall']:.1f}%")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if result['s_precision'] == 0:
    print("\n[ERROR] S-Precision is 0%!")
    print("This means ALL learned solutions are being marked as INVALID.")
    print("\nPossible causes:")
    print("1. Variable name mismatch between solution dict and runner.variables")
    print("2. _is_valid_solution is failing due to constraint building issue")
    print("3. The learned model is genuinely under-constrained")

    print("\nDebugging steps:")
    print("- Check if solution dict keys match runner.variables keys")
    print("- Add logging to _is_valid_solution to see where validation fails")
    print("- Verify CPMpy constraint building is correct")
else:
    print(f"\n[OK] S-Precision is {result['s_precision']:.1f}%")

print("="*80)
