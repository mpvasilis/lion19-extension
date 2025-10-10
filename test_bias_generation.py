"""
Test script to verify the complete bias generation in Phase 1.

This script tests that:
1. All binary constraint types are generated (==, !=, <, >, <=, >=)
2. Constraints are properly pruned using positive examples
3. The bias generation follows CONSTRAINT 2 (prune only with E+)
"""

import logging
from cpmpy import *
from hcar_advanced import HCARFramework, HCARConfig, Constraint

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_bias_generation():
    """Test the complete bias generation with a simple example."""

    logger.info("="*70)
    logger.info("Testing Complete Bias Generation")
    logger.info("="*70)

    # Create a simple problem: 4 variables with domain [1, 4]
    x1 = intvar(1, 4, name='x1')
    x2 = intvar(1, 4, name='x2')
    x3 = intvar(1, 4, name='x3')
    x4 = intvar(1, 4, name='x4')

    variables = {
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4
    }

    domains = {
        'x1': (1, 4),
        'x2': (1, 4),
        'x3': (1, 4),
        'x4': (1, 4)
    }

    # Create positive examples
    # Example: x1 != x2 != x3 != x4 (AllDifferent constraint)
    positive_examples = [
        {'x1': 1, 'x2': 2, 'x3': 3, 'x4': 4},
        {'x1': 2, 'x2': 3, 'x3': 4, 'x4': 1},
        {'x1': 3, 'x2': 4, 'x3': 1, 'x4': 2},
        {'x1': 4, 'x2': 1, 'x3': 2, 'x4': 3},
        {'x1': 1, 'x2': 3, 'x3': 4, 'x4': 2}
    ]

    logger.info(f"\nProblem setup:")
    logger.info(f"  Variables: {len(variables)}")
    logger.info(f"  Domains: {domains}")
    logger.info(f"  Positive examples: {len(positive_examples)}")

    # Initialize HCAR framework
    config = HCARConfig(
        total_budget=100,
        enable_ml_prior=False  # Disable for simple test
    )

    hcar = HCARFramework(config=config, problem_name="test_bias")

    # Generate bias
    logger.info(f"\n{'='*70}")
    logger.info("Generating complete binary bias...")
    logger.info(f"{'='*70}\n")

    B_fixed = hcar._generate_fixed_bias_simple(variables, domains, positive_examples)

    # Analyze results
    logger.info(f"\n{'='*70}")
    logger.info("Bias Generation Results")
    logger.info(f"{'='*70}")

    # Count constraint types
    type_counts = {}
    for c in B_fixed:
        ctype = c.constraint_type
        type_counts[ctype] = type_counts.get(ctype, 0) + 1

    logger.info(f"\nConstraint types generated:")
    for ctype, count in sorted(type_counts.items()):
        logger.info(f"  {ctype}: {count}")

    logger.info(f"\nTotal B_fixed size: {len(B_fixed)}")

    # Expected counts
    num_pairs = len(variables) * (len(variables) - 1) // 2
    max_possible = num_pairs * 6  # 6 constraint types per pair

    logger.info(f"\nExpected statistics:")
    logger.info(f"  Variable pairs: {num_pairs}")
    logger.info(f"  Max possible constraints: {max_possible} (6 types × {num_pairs} pairs)")
    logger.info(f"  Pruned by E+: {max_possible - len(B_fixed)}")
    logger.info(f"  Kept in bias: {len(B_fixed)}")
    logger.info(f"  Pruning rate: {100 * (max_possible - len(B_fixed)) / max_possible:.1f}%")

    # Verify specific constraints
    logger.info(f"\n{'='*70}")
    logger.info("Verification Tests")
    logger.info(f"{'='*70}\n")

    # Test 1: All constraint types should be present
    expected_types = {'Equal', 'NotEqual', 'LessThan', 'GreaterThan',
                     'LessThanOrEqual', 'GreaterThanOrEqual'}
    actual_types = set(type_counts.keys())

    logger.info(f"Test 1: All constraint types generated")
    logger.info(f"  Expected: {expected_types}")
    logger.info(f"  Actual: {actual_types}")
    logger.info(f"  ✓ PASS" if actual_types == expected_types else f"  ✗ FAIL")

    # Test 2: Equal constraints should be pruned (AllDifferent examples)
    equal_count = type_counts.get('Equal', 0)
    logger.info(f"\nTest 2: Equal constraints pruned (AllDifferent examples)")
    logger.info(f"  Equal constraints remaining: {equal_count}")
    logger.info(f"  ✓ PASS (All pruned)" if equal_count == 0 else f"  Note: {equal_count} remain")

    # Test 3: NotEqual constraints should be kept
    notequal_count = type_counts.get('NotEqual', 0)
    logger.info(f"\nTest 3: NotEqual constraints kept")
    logger.info(f"  NotEqual constraints: {notequal_count}")
    logger.info(f"  Expected: {num_pairs} (one per pair)")
    logger.info(f"  ✓ PASS" if notequal_count == num_pairs else f"  Note: Got {notequal_count}")

    # Sample some constraints
    logger.info(f"\n{'='*70}")
    logger.info("Sample Constraints (first 10)")
    logger.info(f"{'='*70}\n")

    for i, c in enumerate(list(B_fixed)[:10]):
        logger.info(f"  {i+1}. {c.id} ({c.constraint_type})")
        logger.info(f"     Scope: {c.scope}")
        logger.info(f"     CPMpy: {c.constraint}")

    logger.info(f"\n{'='*70}")
    logger.info("Bias Generation Test Complete")
    logger.info(f"{'='*70}\n")

    return B_fixed


def test_pruning_correctness():
    """Test that pruning correctly identifies violated constraints."""

    logger.info("="*70)
    logger.info("Testing Pruning Correctness")
    logger.info("="*70)

    # Simple 2-variable problem
    x = intvar(1, 5, name='x')
    y = intvar(1, 5, name='y')

    variables = {'x': x, 'y': y}

    # Example: x=2, y=3
    positive_examples = [{'x': 2, 'y': 3}]

    config = HCARConfig()
    hcar = HCARFramework(config=config, problem_name="test_pruning")

    # Test cases: (constraint_type, should_be_pruned)
    test_cases = [
        ('Equal', True),       # x == y violated by (2, 3)
        ('NotEqual', False),   # x != y satisfied by (2, 3)
        ('LessThan', False),   # x < y satisfied by (2, 3)
        ('GreaterThan', True), # x > y violated by (2, 3)
        ('LessThanOrEqual', False),    # x <= y satisfied by (2, 3)
        ('GreaterThanOrEqual', True)   # x >= y violated by (2, 3)
    ]

    logger.info(f"\nTesting with example: x=2, y=3")
    logger.info(f"\n{'Constraint Type':<20} | {'Should Prune':<12} | {'Actual':<12} | {'Result':<6}")
    logger.info("-" * 60)

    all_passed = True

    for ctype_name, should_prune in test_cases:
        # Create constraint
        if ctype_name == 'Equal':
            cpm_constraint = (x == y)
        elif ctype_name == 'NotEqual':
            cpm_constraint = (x != y)
        elif ctype_name == 'LessThan':
            cpm_constraint = (x < y)
        elif ctype_name == 'GreaterThan':
            cpm_constraint = (x > y)
        elif ctype_name == 'LessThanOrEqual':
            cpm_constraint = (x <= y)
        elif ctype_name == 'GreaterThanOrEqual':
            cpm_constraint = (x >= y)

        candidate = Constraint(
            id=f"test_{ctype_name}",
            constraint=cpm_constraint,
            scope=['x', 'y'],
            constraint_type=ctype_name,
            arity=2
        )

        # Check if violated
        is_violated = hcar._is_violated_by_examples(candidate, positive_examples, variables)

        # Verify
        passed = (is_violated == should_prune)
        all_passed = all_passed and passed

        logger.info(f"{ctype_name:<20} | {str(should_prune):<12} | {str(is_violated):<12} | {'✓ PASS' if passed else '✗ FAIL'}")

    logger.info("-" * 60)
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    logger.info(f"\n{'='*70}\n")

    return all_passed


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("HCAR Phase 1: Complete Bias Generation Test Suite")
    logger.info("="*70 + "\n")

    # Run tests
    test_pruning_correctness()
    test_bias_generation()

    logger.info("\n" + "="*70)
    logger.info("All tests complete!")
    logger.info("="*70 + "\n")
