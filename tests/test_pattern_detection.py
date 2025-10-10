"""
Test pattern detection for Nurse_Rostering to understand why constraints are missed.
"""

import logging
from hcar_advanced import PassiveCandidateGenerator, HCARConfig
from run_hcar_experiments import get_benchmark_configs

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_pattern_detection():
    """Test if pattern detection correctly identifies all constraints."""

    print("="*80)
    print("PATTERN DETECTION TEST - Nurse Rostering")
    print("="*80)

    # Load benchmark
    configs = get_benchmark_configs(use_global_constraints=True)
    nurse_config = None
    for bc in configs:
        if bc.name == "Nurse_Rostering":
            nurse_config = bc
            break

    if not nurse_config:
        print("ERROR: Could not find Nurse_Rostering config")
        return

    print(f"\n1. Loading benchmark...")
    benchmark_data = nurse_config.load()

    if not benchmark_data:
        print("ERROR: Failed to load benchmark")
        return

    target_model = benchmark_data['target_model']
    variables = benchmark_data['variables']
    domains = benchmark_data['domains']
    positive_examples = benchmark_data['positive_examples']

    print(f"   Variables: {len(variables)}")
    print(f"   Target constraints: {len(target_model)}")
    print(f"   Positive examples: {len(positive_examples)}")

    # Print target model constraints
    print(f"\n2. TARGET MODEL CONSTRAINTS:")
    for idx, c in enumerate(target_model):
        print(f"   {idx+1}. {c}")

    # Run passive candidate generation
    print(f"\n3. Running passive candidate generation...")
    config = HCARConfig()
    generator = PassiveCandidateGenerator(config)

    B_globals, B_fixed = generator.generate_candidates(
        positive_examples=positive_examples,
        variables=variables,
        domains=domains
    )

    print(f"\n4. PASSIVE LEARNING RESULTS:")
    print(f"   B_globals: {len(B_globals)} candidates")
    print(f"   B_fixed: {len(B_fixed)} candidates")

    # Print global candidates
    print(f"\n5. GLOBAL CONSTRAINT CANDIDATES:")
    for idx, c in enumerate(B_globals):
        print(f"   {idx+1}. {c}")

    # Analyze what's missing
    print(f"\n6. ANALYSIS:")

    # Count AllDifferent candidates
    alldiff_candidates = [c for c in B_globals if c.constraint_type == 'AllDifferent']
    print(f"   AllDifferent candidates: {len(alldiff_candidates)}")
    for c in alldiff_candidates:
        print(f"      - {c.id}: scope size = {len(c.scope)}")

    # Count Count candidates
    count_candidates = [c for c in B_globals if c.constraint_type == 'Count']
    print(f"   Count candidates: {len(count_candidates)}")
    for c in count_candidates:
        print(f"      - {c.id}")

    # Check if we're detecting patterns for all days
    print(f"\n7. DAY-BY-DAY PATTERN CHECK:")
    num_days = 7
    for day in range(num_days):
        day_vars = [f"var[{day},0,0]", f"var[{day},0,1]", f"var[{day},1,0]",
                   f"var[{day},1,1]", f"var[{day},2,0]", f"var[{day},2,1]"]

        # Check if there's an AllDifferent for this day
        found = False
        for c in alldiff_candidates:
            if set(c.scope) == set(day_vars):
                found = True
                break

        status = "FOUND" if found else "MISSING"
        print(f"   Day {day}: {status}")

    # Check examples to see if all days have different values
    print(f"\n8. CHECKING EXAMPLES FOR PATTERNS:")
    for ex_idx, example in enumerate(positive_examples[:2]):  # Check first 2 examples
        print(f"\n   Example {ex_idx+1}:")
        for day in range(num_days):
            day_values = []
            for shift in range(3):
                for pos in range(2):
                    var_name = f"var[{day},{shift},{pos}]"
                    if var_name in example:
                        day_values.append(example[var_name])

            is_alldiff = len(day_values) == len(set(day_values))
            print(f"      Day {day}: {day_values} - AllDiff: {is_alldiff}")

    print("\n" + "="*80)

if __name__ == "__main__":
    test_pattern_detection()
