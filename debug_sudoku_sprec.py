"""
Debug script to investigate why Sudoku has 0% S-Prec.
S-Prec = 0% means the learned model accepts invalid solutions.
This suggests the learned model is under-constrained (missing critical constraints).
"""

import cpmpy as cp
from benchmarks_global.sudoku import construct_sudoku
from cpmpy.transformations.normalize import toplevel_list
import numpy as np

# Create the benchmark
instance, oracle = construct_sudoku(3, 3, 9)
grid = instance.variables

print("="*80)
print("SUDOKU S-PREC DEBUG")
print("="*80)

# Get all constraints from the oracle (target model)
target_constraints = oracle.constraints
print(f"\nTarget Model Constraints ({len(target_constraints)}):")
for i, c in enumerate(target_constraints):
    print(f"  {i+1}. {c}")

# Separate regular constraints from mock constraints
regular_constraints = []
mock_constraints = []

for c in target_constraints:
    c_str = str(c)
    # Check if it's a diagonal constraint (mock)
    if 'grid[0,0]' in c_str and 'grid[8,8]' in c_str:
        mock_constraints.append(("Main diagonal", c))
    elif 'grid[0,8]' in c_str and 'grid[8,0]' in c_str:
        mock_constraints.append(("Anti-diagonal", c))
    else:
        regular_constraints.append(c)

print(f"\nRegular Constraints (ground truth): {len(regular_constraints)}")
print(f"Mock Constraints (overfitted): {len(mock_constraints)}")

for name, c in mock_constraints:
    print(f"  - {name}: {c}")

# Now test: Generate 5 positive examples from the FULL model (with mocks)
print("\n" + "="*80)
print("GENERATING 5 POSITIVE EXAMPLES (from full model with mocks)")
print("="*80)

full_model = cp.Model(target_constraints)
examples = []

for i in range(5):
    if full_model.solve():
        example = {}
        exclusion = []
        for var in grid.flatten():
            val = var.value()
            example[var.name] = val
            exclusion.append(var != val)
        examples.append(example)
        print(f"\nExample {i+1} generated.")

        # Add exclusion constraint for next solve
        if i < 4:
            full_model += cp.any(exclusion)
    else:
        print(f"\nNo more solutions after {i} examples")
        break

print(f"\nGenerated {len(examples)} examples")

# Now simulate what passive learning would do
print("\n" + "="*80)
print("SIMULATING PASSIVE LEARNING")
print("="*80)

# Mock constraints should be learned since they're consistent with the 5 examples
print("\nPassive Learning would detect these mock constraints as consistent:")
for name, c in mock_constraints:
    # Check if constraint is satisfied by all examples
    satisfied_count = 0
    for ex in examples:
        # Build a test model with just this constraint
        test_model = cp.Model()

        # Set variable values from example
        assignments = []
        for var in grid.flatten():
            assignments.append(var == ex[var.name])

        test_model += assignments
        test_model += c

        if test_model.solve():
            satisfied_count += 1

    print(f"  - {name}: satisfied by {satisfied_count}/{len(examples)} examples")

# Now test: What if we learn ONLY the mock constraints (worst case)?
print("\n" + "="*80)
print("SCENARIO 1: Learned Model = ONLY Mock Constraints (worst case)")
print("="*80)

learned_model_worst = cp.Model([c for _, c in mock_constraints])
print(f"\nLearned model has {len(mock_constraints)} constraints (only mocks)")

# Try to generate a solution from this learned model
if learned_model_worst.solve():
    print("\n[OK] Learned model is SAT (can generate solutions)")

    # Get a solution
    learned_solution = {}
    for var in grid.flatten():
        learned_solution[var.name] = var.value()

    # Check if this solution is valid according to the TARGET
    target_model = cp.Model(regular_constraints)  # Ground truth (no mocks)
    assignments = []
    for var in grid.flatten():
        assignments.append(var == learned_solution[var.name])
    target_model += assignments

    is_valid = target_model.solve()
    print(f"\nIs learned solution valid in TARGET? {is_valid}")

    if not is_valid:
        print("\n[ERROR] PROBLEM IDENTIFIED:")
        print("   Learned model (mocks only) accepts solutions that violate")
        print("   the regular Sudoku constraints (rows/cols/blocks)")
        print("   This would cause S-Prec = 0%")
else:
    print("\n[FAIL] Learned model is UNSAT (over-constrained)")

# Test: What if we learn SOME regular constraints + mocks?
print("\n" + "="*80)
print("SCENARIO 2: Learned Model = Some Regular + Mock Constraints")
print("="*80)

# Suppose we learn 10 regular constraints + 2 mock constraints
learned_mixed = cp.Model(regular_constraints[:10] + [c for _, c in mock_constraints])
print(f"\nLearned model has {10 + len(mock_constraints)} constraints")

if learned_mixed.solve():
    print("\n[OK] Learned model is SAT")

    # Generate 10 solutions from learned model
    learned_solutions = []
    for i in range(10):
        if learned_mixed.solve():
            sol = {}
            exclusion = []
            for var in grid.flatten():
                val = var.value()
                sol[var.name] = val
                exclusion.append(var != val)
            learned_solutions.append(sol)
            learned_mixed += cp.any(exclusion)
        else:
            break

    print(f"Generated {len(learned_solutions)} solutions from learned model")

    # Check how many are valid in target
    valid_count = 0
    for sol in learned_solutions:
        target_check = cp.Model(regular_constraints)
        assignments = []
        for var in grid.flatten():
            assignments.append(var == sol[var.name])
        target_check += assignments

        if target_check.solve():
            valid_count += 1

    s_prec = valid_count / len(learned_solutions) * 100 if learned_solutions else 0
    print(f"\nS-Precision: {valid_count}/{len(learned_solutions)} = {s_prec:.1f}%")

    if s_prec < 100:
        print("\n[WARNING] Some learned solutions are invalid in the target model")
        print("   This happens when learned model is missing critical constraints")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The 0% S-Prec issue occurs when:
1. Passive learning from 5 examples MISSES critical constraints
2. The learned model becomes under-constrained
3. It generates solutions that violate the ground truth

Solution: Ensure passive learning generates examples that force
discovery of ALL critical constraints (rows, cols, blocks).
Currently, the 5 examples might be too similar/biased.
""")
