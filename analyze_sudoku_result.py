"""
Analyze Sudoku experimental result to understand why S-Prec = 0%
"""

import json

# Load results
with open('hcar_results/Sudoku_HCAR-Advanced.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("SUDOKU RESULT ANALYSIS")
print("="*80)

run_results = results['runs'][0]

print("\nMetrics:")
print(f"  S-Precision: {run_results['s_precision']:.1f}%")
print(f"  S-Recall: {run_results['s_recall']:.1f}%")
print(f"  Total Queries: {run_results['total_queries']}")
print(f"  Q2 (Refinement): {run_results['queries_phase2']}")
print(f"  Q3 (Active): {run_results['queries_phase3']}")

print("\nLearned Model:")
print(f"  Global constraints: {run_results['num_global_constraints']}")
print(f"  Fixed constraints: {run_results['num_fixed_constraints']}")
print(f"  Total: {run_results['total_constraints']}")

# Check if the learned model is over-constrained
if run_results['s_recall'] == 100:
    print("\n[OK] S-Recall = 100% -> Model accepts all valid solutions")
else:
    print("\n[ERROR] S-Recall < 100% -> Model rejects some valid solutions (over-constrained)")

if run_results['s_precision'] == 0:
    print("[ERROR] S-Prec = 0% -> Model accepts ONLY invalid solutions")
    print("   This is very unusual and suggests a serious issue.")
elif run_results['s_precision'] < 100:
    print(f"[WARNING] S-Prec = {run_results['s_precision']:.1f}% -> Model accepts some invalid solutions")
    print("   Model is under-constrained (missing critical constraints)")
else:
    print("[OK] S-Prec = 100% -> Model accepts only valid solutions")

# Check constraint counts
print("\nConstraint Analysis:")
print(f"  Expected global constraints: 27 (9 rows + 9 cols + 9 blocks)")
print(f"  Learned global constraints: {run_results['num_global_constraints']}")

if run_results['num_global_constraints'] > 27:
    print(f"  [WARNING] Learned {run_results['num_global_constraints'] - 27} extra global constraints")
    print("  These might be the mock diagonal constraints or other spurious patterns")

print(f"\n  Expected fixed-arity constraints: ~0-20 (for grid structure)")
print(f"  Learned fixed-arity constraints: {run_results['num_fixed_constraints']}")

if run_results['num_fixed_constraints'] > 100:
    print(f"  [WARNING] Learned {run_results['num_fixed_constraints']} fixed constraints")
    print("  This suggests the fixed-arity bias was NOT properly pruned")

# Diagnostic conclusion
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if run_results['s_precision'] == 0 and run_results['s_recall'] == 100:
    print("""
The learned model accepts ALL solutions (both valid and invalid).
This indicates the model is essentially EMPTY or contains only trivial constraints.

Possible causes:
1. Global constraints were rejected during Phase 2
2. Fixed-arity bias was not learned properly in Phase 3
3. Model merging issue: learned constraints are not being enforced

Check:
- Were the 27 AllDifferent constraints learned in Phase 1?
- Were they kept (not rejected) in Phase 2?
- Was Phase 3 (MQuAcq-2) executed correctly?
""")

print("="*80)
