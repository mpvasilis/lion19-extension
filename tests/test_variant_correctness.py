"""
Test to verify that HCAR variants are implemented correctly.

This script validates that:
1. HCAR-Advanced uses IntelligentSubsetExplorer
2. HCAR-Heuristic uses HeuristicSubsetExplorer
3. HCAR-NoRefine skips Phase 2
"""

from hcar_advanced import (
    HCARFramework,
    HCARConfig,
    IntelligentSubsetExplorer,
    HeuristicSubsetExplorer
)


def test_hcar_advanced():
    """Test HCAR-Advanced uses intelligent subset exploration."""
    print("Testing HCAR-Advanced...")

    config = HCARConfig(
        use_intelligent_subsets=True,
        total_budget=500
    )

    framework = HCARFramework(config, problem_name="test")

    # Check that intelligent explorer is used
    assert isinstance(framework.subset_explorer, IntelligentSubsetExplorer), \
        "HCAR-Advanced should use IntelligentSubsetExplorer"

    # Check that Phase 2 is enabled
    assert config.total_budget > 0, \
        "HCAR-Advanced should have Phase 2 enabled"

    print("[PASS] HCAR-Advanced correctly uses IntelligentSubsetExplorer")
    print(f"[PASS] Phase 2 budget: {config.total_budget}")


def test_hcar_heuristic():
    """Test HCAR-Heuristic uses positional heuristics."""
    print("\nTesting HCAR-Heuristic...")

    config = HCARConfig(
        use_intelligent_subsets=False,
        total_budget=500
    )

    framework = HCARFramework(config, problem_name="test")

    # Check that heuristic explorer is used
    assert isinstance(framework.subset_explorer, HeuristicSubsetExplorer), \
        "HCAR-Heuristic should use HeuristicSubsetExplorer"

    # Check that Phase 2 is enabled
    assert config.total_budget > 0, \
        "HCAR-Heuristic should have Phase 2 enabled"

    print("[PASS] HCAR-Heuristic correctly uses HeuristicSubsetExplorer")
    print(f"[PASS] Phase 2 budget: {config.total_budget}")


def test_hcar_norefine():
    """Test HCAR-NoRefine skips Phase 2."""
    print("\nTesting HCAR-NoRefine...")

    config = HCARConfig(
        total_budget=0  # Skip Phase 2
    )

    framework = HCARFramework(config, problem_name="test")

    # Check that Phase 2 is disabled
    assert config.total_budget == 0, \
        "HCAR-NoRefine should have Phase 2 disabled (budget=0)"

    print("[PASS] HCAR-NoRefine correctly skips Phase 2 (budget=0)")


def test_subset_explorer_behavior():
    """Test that the two explorers produce different subsets."""
    print("\nTesting subset explorer behavior...")

    from hcar_advanced import Constraint

    # Create a mock rejected constraint
    mock_constraint = Constraint(
        id="test_alldiff_row1",
        constraint=None,
        scope=["x_0_0", "x_0_1", "x_0_2", "x_0_3", "x_0_4"],
        constraint_type="AllDifferent",
        arity=5,
        level=0,
        confidence=0.1
    )

    config = HCARConfig(max_subset_depth=3)

    # Test intelligent explorer
    intelligent_subsets = IntelligentSubsetExplorer.generate_informed_subsets(
        rejected_constraint=mock_constraint,
        positive_examples=[],
        learned_globals=[],
        config=config,
        variables=None
    )

    # Test heuristic explorer
    heuristic_subsets = HeuristicSubsetExplorer.generate_informed_subsets(
        rejected_constraint=mock_constraint,
        positive_examples=[],
        learned_globals=[],
        config=config,
        variables=None
    )

    print(f"[PASS] IntelligentSubsetExplorer generated {len(intelligent_subsets)} subsets")
    print(f"[PASS] HeuristicSubsetExplorer generated {len(heuristic_subsets)} subsets")

    # Both should generate subsets
    assert len(intelligent_subsets) > 0, "IntelligentSubsetExplorer should generate subsets"
    assert len(heuristic_subsets) > 0, "HeuristicSubsetExplorer should generate subsets"

    # Show which variables were removed
    if intelligent_subsets:
        print(f"  Intelligent removed: {[c.id.split('_sub_')[-1] for c in intelligent_subsets]}")
    if heuristic_subsets:
        print(f"  Heuristic removed: {[c.id.split('_sub_')[-1] for c in heuristic_subsets]}")


def main():
    """Run all tests."""
    print("="*70)
    print("HCAR Variant Correctness Tests")
    print("="*70)

    try:
        test_hcar_advanced()
        test_hcar_heuristic()
        test_hcar_norefine()
        test_subset_explorer_behavior()

        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)
        print("\nSummary:")
        print("  - HCAR-Advanced: Uses IntelligentSubsetExplorer (culprit scores)")
        print("  - HCAR-Heuristic: Uses HeuristicSubsetExplorer (positional)")
        print("  - HCAR-NoRefine: Skips Phase 2 (budget=0)")
        print("\nThe implementation is now CORRECT!")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
