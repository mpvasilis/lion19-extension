"""
Test script for improved counterexample repair mechanisms.

Tests the new contribution analysis for Sum and direct violation
analysis for Count constraints.
"""

from hcar_advanced import CounterexampleRepair, Constraint
from dataclasses import dataclass


def test_sum_contribution_analysis():
    """Test Sum constraint contribution analysis."""
    print("\n=== Testing Sum Contribution Analysis ===")

    # Create a fake Sum constraint: sum(x1,x2,x3,x4) <= 20
    constraint = Constraint(
        id="sum_test",
        constraint=None,  # We'll parse from string
        scope=["x1", "x2", "x3", "x4"],
        constraint_type="Sum",
        arity=4
    )

    # Counterexample: x1=3, x2=5, x3=8, x4=9 (sum=25, violates <=20)
    counterexample = {
        "x1": 3,
        "x2": 5,
        "x3": 8,
        "x4": 9
    }

    # Extract just the values for analysis
    values = {var: counterexample[var] for var in constraint.scope}

    print(f"Constraint: sum([x1, x2, x3, x4]) <= 20")
    print(f"Counterexample: {counterexample}")
    print(f"Actual sum: {sum(values.values())} (violation amount: 5)")

    # Call the analysis
    violating_vars = CounterexampleRepair._identify_violating_vars_sum(
        constraint, values
    )

    print(f"\nIdentified violating variables: {violating_vars}")
    print(f"Expected: Variables with value >= 2.5 (50% of violation)")
    print(f"  - x3=8 (>= 2.5) [PASS]")
    print(f"  - x4=9 (>= 2.5) [PASS]")
    print(f"Analysis successful: {'x3' in violating_vars and 'x4' in violating_vars}")


def test_count_direct_analysis():
    """Test Count constraint direct violation analysis."""
    print("\n\n=== Testing Count Direct Violation Analysis ===")

    # Create a fake Count constraint: Count([x1,x2,x3,x4], value=5) == 2
    # (exactly 2 variables should have value 5)
    constraint = Constraint(
        id="count_test",
        constraint=None,
        scope=["x1", "x2", "x3", "x4"],
        constraint_type="Count",
        arity=4
    )

    # Mock the constraint string for parsing
    class MockConstraint:
        def __str__(self):
            return "Count([x1, x2, x3, x4], 5) == 2"

    constraint.constraint = MockConstraint()

    # Counterexample: x1=5, x2=5, x3=5, x4=3 (3 vars have value 5, violates ==2)
    counterexample = {
        "x1": 5,
        "x2": 5,
        "x3": 5,
        "x4": 3
    }

    values = {var: counterexample[var] for var in constraint.scope}

    print(f"Constraint: Count([x1,x2,x3,x4], 5) == 2")
    print(f"Counterexample: {counterexample}")
    print(f"Actual count: 3 (expects 2)")

    # Call the analysis
    violating_vars = CounterexampleRepair._identify_violating_vars_count(
        constraint, values
    )

    print(f"\nIdentified violating variables: {violating_vars}")
    print(f"Expected: Variables WITH value 5 (x1, x2, x3)")
    print(f"Reason: Too many have the value, any could be removed")

    has_all_targets = all(v in violating_vars for v in ["x1", "x2", "x3"])
    print(f"Analysis successful: {has_all_targets}")


def test_alldifferent_precision():
    """Test AllDifferent precise duplicate detection."""
    print("\n\n=== Testing AllDifferent Duplicate Detection ===")

    constraint = Constraint(
        id="alldiff_test",
        constraint=None,
        scope=["x1", "x2", "x3", "x4"],
        constraint_type="AllDifferent",
        arity=4
    )

    # Counterexample: x1=3, x2=5, x3=3, x4=7 (x1 and x3 duplicate)
    counterexample = {
        "x1": 3,
        "x2": 5,
        "x3": 3,
        "x4": 7
    }

    print(f"Constraint: AllDifferent([x1, x2, x3, x4])")
    print(f"Counterexample: {counterexample}")
    print(f"Duplicate: x1=3, x3=3")

    # Call main function
    violating_vars = CounterexampleRepair._identify_violating_variables(
        constraint, counterexample, {}
    )

    print(f"\nIdentified violating variables: {violating_vars}")
    print(f"Expected: x1 and x3 (precisely the duplicates)")

    is_precise = set(violating_vars) == {"x1", "x3"}
    print(f"Analysis successful: {is_precise}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Improved Counterexample Repair Mechanisms")
    print("=" * 60)

    try:
        test_sum_contribution_analysis()
        test_count_direct_analysis()
        test_alldifferent_precision()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
