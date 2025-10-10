"""
Debug script to investigate Nurse_Rostering S-Rec = 0% issue.
"""

import logging
import cpmpy as cp
from hcar_advanced import HCARFramework, HCARConfig
from run_hcar_experiments import get_benchmark_configs

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_nurse_rostering_satisfiability():
    """Test if Nurse_Rostering benchmark and learned models are satisfiable."""

    print("="*80)
    print("NURSE ROSTERING DEBUG TEST")
    print("="*80)

    # Load the benchmark
    configs = get_benchmark_configs(use_global_constraints=True)
    nurse_config = None
    for bc in configs:
        if bc.name == "Nurse_Rostering":
            nurse_config = bc
            break

    if not nurse_config:
        print("ERROR: Could not find Nurse_Rostering config")
        return

    print(f"\n1. Loading Nurse_Rostering benchmark...")
    benchmark_data = nurse_config.load()

    if not benchmark_data:
        print("ERROR: Failed to load benchmark")
        return

    # Extract data
    target_model = benchmark_data['target_model']
    variables = benchmark_data['variables']
    positive_examples = benchmark_data['positive_examples']

    print(f"   Variables: {len(variables)}")
    print(f"   Target constraints: {len(target_model)}")
    print(f"   Positive examples: {len(positive_examples)}")

    # Test 1: Is target model satisfiable?
    print(f"\n2. Testing TARGET model satisfiability...")
    target_cpm = cp.Model(target_model)
    target_sat = target_cpm.solve()
    print(f"   Target model SAT: {target_sat}")

    if target_sat:
        print(f"   Sample solution from target:")
        for i, (var_name, var_obj) in enumerate(list(variables.items())[:10]):
            if hasattr(var_obj, 'value'):
                val = var_obj.value()
                print(f"      {var_name} = {val}")

    # Test 2: Check if positive examples satisfy target model
    print(f"\n3. Validating positive examples against target model...")
    for idx, example in enumerate(positive_examples):
        # Create model with target constraints + example assignment
        test_model = cp.Model(target_model)
        example_constraints = []
        for var_name, value in example.items():
            if var_name in variables:
                example_constraints.append(variables[var_name] == value)
        test_model += example_constraints

        is_sat = test_model.solve()
        print(f"   Example {idx+1}: {'VALID' if is_sat else 'INVALID'}")

        if not is_sat:
            print(f"      WARNING: Example {idx+1} does not satisfy target model!")

    # Test 3: Run HCAR and check learned model
    print(f"\n4. Running HCAR-Advanced to learn model...")

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
        inject_overfitted=True
    )

    framework = HCARFramework(config=config)

    final_constraints, metrics = framework.run(
        positive_examples=positive_examples,
        oracle_func=benchmark_data['oracle_func'],
        variables=variables,
        domains=benchmark_data['domains'],
        target_model=target_model
    )

    # Extract learned global and fixed constraints
    learned_global = [c for c in final_constraints if hasattr(c, 'constraint_type') and c.constraint_type in ['AllDifferent', 'Sum', 'Count']]
    learned_fixed = [c for c in final_constraints if c not in learned_global]

    print(f"\n5. Analyzing LEARNED model...")
    print(f"   Global constraints learned: {len(learned_global)}")
    print(f"   Fixed constraints learned: {len(learned_fixed)}")
    print(f"   Total constraints: {len(final_constraints)}")

    # Print learned global constraints
    if learned_global:
        print(f"\n   Learned global constraints:")
        for idx, c in enumerate(learned_global[:10]):
            print(f"      {idx+1}. {c}")

    # Test 4: Is learned model satisfiable?
    print(f"\n6. Testing LEARNED model satisfiability...")
    learned_cpm = cp.Model()

    for c in final_constraints:
        if hasattr(c, 'constraint') and c.constraint is not None:
            learned_cpm += c.constraint

    learned_sat = learned_cpm.solve()
    print(f"   Learned model SAT: {learned_sat}")

    if learned_sat:
        print(f"   Sample solution from learned model:")
        for i, (var_name, var_obj) in enumerate(list(variables.items())[:10]):
            if hasattr(var_obj, 'value'):
                val = var_obj.value()
                print(f"      {var_name} = {val}")

        # Test 5: Does learned model solution satisfy target?
        print(f"\n7. Checking if learned model solution satisfies TARGET...")
        learned_solution = {}
        for var_name, var_obj in variables.items():
            if hasattr(var_obj, 'value'):
                val = var_obj.value()
                if val is not None:
                    learned_solution[var_name] = val

        # Validate with oracle
        oracle_response = benchmark_data['oracle_func'](learned_solution)
        print(f"   Oracle says: {oracle_response}")

        if oracle_response.value == "Invalid":
            print(f"\n   DIAGNOSIS: Learned model is SAT but produces INVALID solutions!")
            print(f"   This explains S-Prec = 100%, S-Rec = 0%")
            print(f"   The learned model is MISSING some target constraints.")
    else:
        print(f"\n   DIAGNOSIS: Learned model is UNSAT!")
        print(f"   This means the learned model is over-constrained.")
        print(f"   Possible causes:")
        print(f"   - Conflicting constraints")
        print(f"   - Incorrect constraint parameters")
        print(f"   - Over-fitted constraints")

    # Test 6: Compare target vs learned constraints
    print(f"\n8. Constraint comparison:")
    print(f"   Target model has {len(target_model)} constraints")
    print(f"   Learned model has {len(final_constraints)} constraints")
    print(f"   - Global: {len(learned_global)}")
    print(f"   - Fixed: {len(learned_fixed)}")

    # Print metrics
    print(f"\n9. Metrics from framework:")
    print(f"   S-Precision: {metrics.get('s_precision', 0):.1f}%")
    print(f"   S-Recall: {metrics.get('s_recall', 0):.1f}%")
    print(f"   Queries Phase 2: {metrics.get('queries_phase2', 0)}")
    print(f"   Queries Phase 3: {metrics.get('queries_phase3', 0)}")

    print("\n" + "="*80)

if __name__ == "__main__":
    test_nurse_rostering_satisfiability()
