"""
Quick test to verify Phase 1 diversity fix works for 9x9 variants.
Runs Phase 1 for variants 0, 1, and 2, then compares their overfitted constraints.
"""

import os
import pickle
from cpmpy.transformations.get_variables import get_variables

def main():
    print("="*80)
    print("TESTING 9x9 PHASE 1 DIVERSITY")
    print("="*80)
    print("\nRunning Phase 1 for 9x9 variants 0, 1, 2 to verify different constraints...")
    
    from experiment_10_variants_9x9 import run_phase1_for_variant
    
    output_dir = 'test_9x9_diversity_output'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for variant_id in [0, 1, 2]:
        print(f"\n{'='*80}")
        print(f"VARIANT {variant_id}")
        print(f"{'='*80}")
        
        output_path = run_phase1_for_variant(
            variant_id=variant_id,
            output_dir=output_dir,
            num_examples=5,
            num_overfitted=18
        )
        
        # Load and analyze
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract constraint signatures
        signatures = set()
        for c in data['CG']:
            scope_vars = get_variables([c])
            var_names = tuple(sorted([v.name for v in scope_vars]))
            signatures.add(var_names)
        
        results[variant_id] = {
            'total': len(signatures),
            'target': data['metadata']['num_target_alldiffs_dedup'],
            'overfitted': data['metadata']['num_overfitted_alldiffs_dedup'],
            'signatures': signatures
        }
        
        print(f"\n[SUMMARY] Variant {variant_id}:")
        print(f"  Total constraints: {results[variant_id]['total']}")
        print(f"  Target constraints: {results[variant_id]['target']}")
        print(f"  Overfitted constraints: {results[variant_id]['overfitted']}")
    
    # Compare results
    print(f"\n\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    # Compare 0 vs 1
    common_01 = results[0]['signatures'] & results[1]['signatures']
    unique_0 = results[0]['signatures'] - results[1]['signatures']
    unique_1 = results[1]['signatures'] - results[0]['signatures']
    
    print("Variant 0 vs Variant 1:")
    print(f"  Common constraints: {len(common_01)}")
    print(f"  Unique to V0: {len(unique_0)}")
    print(f"  Unique to V1: {len(unique_1)}")
    
    # Compare 1 vs 2
    common_12 = results[1]['signatures'] & results[2]['signatures']
    unique_1_vs_2 = results[1]['signatures'] - results[2]['signatures']
    unique_2 = results[2]['signatures'] - results[1]['signatures']
    
    print("\nVariant 1 vs Variant 2:")
    print(f"  Common constraints: {len(common_12)}")
    print(f"  Unique to V1: {len(unique_1_vs_2)}")
    print(f"  Unique to V2: {len(unique_2)}")
    
    # Compare 0 vs 2
    common_02 = results[0]['signatures'] & results[2]['signatures']
    unique_0_vs_2 = results[0]['signatures'] - results[2]['signatures']
    unique_2_vs_0 = results[2]['signatures'] - results[0]['signatures']
    
    print("\nVariant 0 vs Variant 2:")
    print(f"  Common constraints: {len(common_02)}")
    print(f"  Unique to V0: {len(unique_0_vs_2)}")
    print(f"  Unique to V2: {len(unique_2_vs_0)}")
    
    # Check if all have differences
    has_differences = (
        (len(unique_0) > 0 or len(unique_1) > 0) and
        (len(unique_1_vs_2) > 0 or len(unique_2) > 0) and
        (len(unique_0_vs_2) > 0 or len(unique_2_vs_0) > 0)
    )
    
    print(f"\n{'='*80}")
    if has_differences:
        print("✅ SUCCESS: All three variants have different overfitted constraints!")
        print("\nThe Phase 1 diversity fix is working correctly for 9x9 problems.")
        return True
    else:
        print("✗ FAILURE: Some variants have identical constraints.")
        print("\nThe Phase 1 diversity fix may not be working correctly.")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*80}")
    if success:
        print("TEST PASSED ✓")
        print("\nYou can now re-run the full 9x9 experiment with:")
        print("  python experiment_10_variants_9x9.py")
    else:
        print("TEST FAILED ✗")
    print(f"{'='*80}\n")


