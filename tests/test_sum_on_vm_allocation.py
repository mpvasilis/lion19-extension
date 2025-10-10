"""
Test Sum Constraint Contribution Analysis on VM Allocation Benchmark

This script runs VM Allocation (which has Sum constraints) to verify
that the contribution analysis works correctly with real CPMpy objects.
"""

import logging
import sys
from hcar_advanced import HCARFramework, HCARConfig

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sum_contribution_test.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)


def test_vm_allocation_sum_analysis():
    """
    Test VM Allocation benchmark with counterexample repair enabled.

    VM Allocation has Sum constraints (resource allocation), so we'll
    see the contribution analysis in action when constraints are refuted.
    """

    print("=" * 70)
    print("Testing Sum Contribution Analysis on VM Allocation Benchmark")
    print("=" * 70)

    try:
        # Import benchmark
        from benchmarks_global import vm_allocation as vm_allocation_global

        logger.info("Loading VM Allocation benchmark...")

        # Define PM and VM data (resource allocation problem)
        pm_data = {
            'PM1': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
            'PM2': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
            'PM3': {'capacity_cpu': 10, 'capacity_memory': 16, 'capacity_disk': 100},
        }

        vm_data = {
            'VM1': {'demand_cpu': 2, 'demand_memory': 4, 'demand_disk': 20, 'availability_zone': 'AZ1', 'priority': 1},
            'VM2': {'demand_cpu': 3, 'demand_memory': 6, 'demand_disk': 30, 'availability_zone': 'AZ1', 'priority': 2},
            'VM3': {'demand_cpu': 2, 'demand_memory': 4, 'demand_disk': 20, 'availability_zone': 'AZ2', 'priority': 1},
            'VM4': {'demand_cpu': 4, 'demand_memory': 8, 'demand_disk': 40, 'availability_zone': 'AZ2', 'priority': 2},
        }

        instance, oracle = vm_allocation_global.construct_vm_allocation(pm_data, vm_data)

        # Extract variables
        import numpy as np
        variables_array = instance.variables

        if isinstance(variables_array, np.ndarray):
            variables_list = list(variables_array.flatten())
        elif hasattr(variables_array, '__iter__'):
            variables_list = list(variables_array)
        else:
            variables_list = [variables_array]

        variables = {}
        for var in variables_list:
            if hasattr(var, 'name'):
                variables[var.name] = var

        # Extract domains
        domains = {}
        for var_name, var in variables.items():
            if hasattr(var, 'lb') and hasattr(var, 'ub'):
                lb_val = var.lb if not callable(var.lb) else var.lb()
                ub_val = var.ub if not callable(var.ub) else var.ub()
                domains[var_name] = list(range(int(lb_val), int(ub_val) + 1))
            else:
                domains[var_name] = [0, 1]

        logger.info(f"Problem has {len(variables)} variables")

        # Try to get constraints count
        if hasattr(instance, 'constraints'):
            logger.info(f"Target model has {len(instance.constraints)} constraints")
        elif hasattr(instance, 'target_model'):
            logger.info(f"Target model has {len(instance.target_model)} constraints")

        # Generate positive examples
        from cpmpy import Model

        # Get constraints (try different attributes)
        constraints = None
        if hasattr(instance, 'constraints'):
            constraints = instance.constraints
        elif hasattr(instance, 'target_model'):
            constraints = instance.target_model
        else:
            logger.error("Could not find constraints in instance!")
            return

        model = Model(constraints)

        positive_examples = []
        for i in range(5):
            solution = model.solve()
            if solution:
                example = {var.name: var.value() for var in variables.values()}
                positive_examples.append(example)
                logger.info(f"Generated example {i+1}/5")
            else:
                logger.error("Failed to generate example!")
                return

        # Configure HCAR with counterexample repair ENABLED
        config = HCARConfig(
            total_budget=100,
            max_time_seconds=300,
            use_counterexample_repair=True,  # Enable new approach!
            theta_min=0.2,
            theta_max=0.8,
            max_subset_depth=2
        )

        logger.info("\n" + "=" * 70)
        logger.info("Starting HCAR with COUNTEREXAMPLE REPAIR enabled")
        logger.info("=" * 70 + "\n")

        # Create framework
        framework = HCARFramework(config)

        # Oracle wrapper
        def oracle_func(assignment):
            from hcar_advanced import OracleResponse
            # Convert to format expected by oracle
            is_valid = oracle.ask_query(assignment)
            return OracleResponse.VALID if is_valid else OracleResponse.INVALID

        # Run learning
        result = framework.learn_constraint_model(
            positive_examples=positive_examples,
            variables=variables,
            domains=domains,
            oracle_func=oracle_func
        )

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Learned model: {len(result['learned_constraints'])} constraints")
        print(f"  - Global: {len([c for c in result['learned_constraints'] if c.arity > 3])}")
        print(f"  - Fixed-arity: {len([c for c in result['learned_constraints'] if c.arity <= 3])}")
        print(f"\nQueries used:")
        print(f"  - Phase 2 (refinement): {result['queries_phase2']}")
        print(f"  - Phase 3 (active): {result['queries_phase3']}")
        print(f"  - TOTAL: {result['total_queries']}")
        print(f"\nTime: {result['time_seconds']:.2f}s")

        print("\n" + "=" * 70)
        print("Check 'sum_contribution_test.log' for detailed analysis!")
        print("Look for lines containing:")
        print("  - 'Counterexample analysis: violating variables'")
        print("  - 'Using counterexample-driven minimal repair'")
        print("=" * 70)

    except ImportError as e:
        print(f"Error: Could not import VM Allocation benchmark: {e}")
        print("Please ensure benchmarks_global/vm_allocation.py exists")
        return

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vm_allocation_sum_analysis()
