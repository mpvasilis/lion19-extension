from cpmpy import *

from hcar_advanced import (
    HCARFramework,
    HCARConfig,
    OracleResponse,
    Constraint
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_sudoku_4x4():
    """
    Create a simple 4x4 Sudoku problem for testing.
    
    4x4 Sudoku has:
    - 4 rows, each with AllDifferent constraint
    - 4 columns, each with AllDifferent constraint
    - 4 2x2 blocks, each with AllDifferent constraint
    Total: 12 AllDifferent constraints
    """

    
    variables = {}
    domains = {}
    
    for i in range(4):
        for j in range(4):
            var_name = f"x_{i}_{j}"
            variables[var_name] = intvar(1, 4, name=var_name)
            domains[var_name] = [1, 2, 3, 4]
    
    # Ground truth model: AllDifferent constraints
    target_model = []
    
    # Row constraints
    for i in range(4):
        row_vars = [variables[f"x_{i}_{j}"] for j in range(4)]
        target_model.append(AllDifferent(row_vars))
    
    # Column constraints
    for j in range(4):
        col_vars = [variables[f"x_{i}_{j}"] for i in range(4)]
        target_model.append(AllDifferent(col_vars))
    
    # Block constraints (2x2 blocks)
    for block_row in range(2):
        for block_col in range(2):
            block_vars = []
            for i in range(2):
                for j in range(2):
                    var_name = f"x_{block_row*2 + i}_{block_col*2 + j}"
                    block_vars.append(variables[var_name])
            target_model.append(AllDifferent(block_vars))
    
    logger.info(f"Created 4x4 Sudoku with {len(target_model)} constraints")
    
    return {
        'variables': variables,
        'domains': domains,
        'target_model': target_model
    }


def generate_sudoku_solutions(model, variables, num_solutions=5):

    
    solutions = []
    cpm_model = Model(model)
    
    # Generate multiple solutions
    for _ in range(num_solutions):
        if cpm_model.solve():
            solution = {}
            for var_name, var_obj in variables.items():
                solution[var_name] = int(var_obj.value())
            solutions.append(solution)
            
            # Add blocking clause to get different solution
            blocking = []
            for var_name, var_obj in variables.items():
                blocking.append(var_obj != var_obj.value())
            cpm_model += sum(blocking) > 0
        else:
            break
    
    logger.info(f"Generated {len(solutions)} example solutions")
    return solutions


def create_oracle(target_model, variables):
    """Create an oracle function for the problem."""
    
    def oracle_func(assignment):
        """Check if assignment satisfies target model."""
        try:
            # Create a model with target constraints
            model = Model(target_model)
            
            # Add assignment as constraints
            for var_name, value in assignment.items():
                if var_name in variables:
                    model += (variables[var_name] == value)
            
            # Check if satisfiable
            if model.solve():
                return OracleResponse.VALID
            else:
                return OracleResponse.INVALID
        
        except Exception as e:
            logger.error(f"Oracle error: {e}")
            return OracleResponse.INVALID
    
    return oracle_func


def run_simple_example():
    """Run a simple HCAR example."""
    
    print("\n" + "="*70)
    print("HCAR Simple Example: 4x4 Sudoku")
    print("="*70 + "\n")
    
    # 1. Create problem
    print("Step 1: Creating 4x4 Sudoku problem...")
    problem = create_simple_sudoku_4x4()
    
    if problem is None:
        print("\n❌ Failed to create problem. Please install CPMpy:")
        print("   pip install cpmpy")
        return
    
    variables = problem['variables']
    domains = problem['domains']
    target_model = problem['target_model']
    
    print(f"  [OK] Created problem with {len(variables)} variables")
    print(f"  [OK] Target model has {len(target_model)} constraints")
    
    # 2. Generate initial examples
    print("\nStep 2: Generating initial positive examples...")
    positive_examples = generate_sudoku_solutions(target_model, variables, num_solutions=5)
    
    if not positive_examples:
        print("\n[ERROR] Failed to generate examples")
        return
    
    print(f"  [OK] Generated {len(positive_examples)} example solutions")
    print(f"  Example: {list(positive_examples[0].items())[:4]}...")
    
    # 3. Create oracle
    print("\nStep 3: Creating oracle...")
    oracle_func = create_oracle(target_model, variables)
    
    if oracle_func is None:
        print("\n[ERROR] Failed to create oracle")
        return
    
    print("  [OK] Oracle ready")
    
    # 4. Configure HCAR
    print("\nStep 4: Configuring HCAR...")
    config = HCARConfig(
        total_budget=100,           # Limited budget for quick test
        max_time_seconds=300,       # 5 minutes
        query_timeout=10.0,
        theta_min=0.15,
        theta_max=0.85,
        max_subset_depth=2,
        base_budget_per_constraint=5,
        enable_ml_prior=False       # Disable ML for simplicity
    )
    print("  [OK] HCAR configured")
    
    # 5. Run HCAR
    print("\nStep 5: Running HCAR framework...")
    print("-" * 70)
    
    hcar = HCARFramework(config, problem_name="Sudoku4x4")
    
    try:
        learned_model, metrics = hcar.run(
            positive_examples=positive_examples,
            oracle_func=oracle_func,
            variables=variables,
            domains=domains,
            target_model=target_model
        )
        
        # 6. Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70 + "\n")
        
        print("Performance Metrics:")
        print(f"  Total Queries:    {metrics['queries_total']}")
        print(f"  Phase 2 Queries:  {metrics['queries_phase2']}")
        print(f"  Phase 3 Queries:  {metrics['queries_phase3']}")
        print(f"  Time:             {metrics['time_seconds']:.2f}s")
        
        print("\nLearned Model:")
        print(f"  Global Constraints:  {metrics['num_global_constraints']}")
        print(f"  Fixed Constraints:   {metrics['num_fixed_constraints']}")
        print(f"  Total Constraints:   {metrics['total_constraints']}")
        
        print("\n[SUCCESS] HCAR completed successfully!")
        
        # Compare with target
        print(f"\nTarget Model Size: {len(target_model)} constraints")
        print(f"Learned Model Size: {len(learned_model)} constraints")
        
        if len(learned_model) == len(target_model):
            print("   Perfect match!")
        elif len(learned_model) < len(target_model):
            print("   Under-fitted (learned fewer constraints)")
        else:
            print("   Over-fitted (learned more constraints)")
    
    except Exception as e:
        print(f"\n[ERROR] HCAR failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70 + "\n")


def demonstrate_intelligent_subset_exploration():
    """
    Demonstrate the intelligent subset exploration mechanism.
    This shows how culprit scores work without running full HCAR.
    """
    
    print("\n" + "="*70)
    print("Intelligent Subset Exploration Demo")
    print("="*70 + "\n")
    
    try:
        from hcar_advanced import IntelligentSubsetExplorer, Constraint
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Create a mock over-fitted constraint
    # Suppose we have AllDifferent([x_0_0, x_0_1, x_0_2, x_0_3, x_1_0])
    # where x_1_0 was incorrectly included (from different row)
    
    rejected_constraint = Constraint(
        id="alldiff_overfit",
        constraint=None,
        scope=['x_0_0', 'x_0_1', 'x_0_2', 'x_0_3', 'x_1_0'],  # x_1_0 is culprit
        constraint_type="AllDifferent",
        arity=5,
        level=0,
        confidence=0.1  # Rejected
    )
    
    # Mock positive examples
    positive_examples = [
        {
            'x_0_0': 1, 'x_0_1': 2, 'x_0_2': 3, 'x_0_3': 4,
            'x_1_0': 2, 'x_1_1': 1, 'x_1_2': 4, 'x_1_3': 3
        },
        {
            'x_0_0': 2, 'x_0_1': 3, 'x_0_2': 4, 'x_0_3': 1,
            'x_1_0': 1, 'x_1_1': 2, 'x_1_2': 3, 'x_1_3': 4
        }
    ]
    
    print("Rejected Constraint:")
    print(f"  Scope: {rejected_constraint.scope}")
    print(f"  Type: {rejected_constraint.constraint_type}")
    print("\nCalculating culprit scores...\n")
    
    # Calculate scores manually to show the process
    explorer = IntelligentSubsetExplorer()
    
    scores = {}
    for var in rejected_constraint.scope:
        # Calculate individual score components
        isolation = explorer._structural_isolation(var, rejected_constraint.scope)
        support = explorer._weak_constraint_support(var, [])  # No learned globals yet
        deviation = explorer._value_pattern_deviation(
            var, rejected_constraint.scope, positive_examples, "AllDifferent"
        )
        
        total = 0.4 * isolation + 0.3 * support + 0.3 * deviation
        scores[var] = total
        
        print(f"Variable: {var}")
        print(f"  Structural Isolation:   {isolation:.3f} (weight: 0.4)")
        print(f"  Weak Constraint Support: {support:.3f} (weight: 0.3)")
        print(f"  Value Pattern Deviation: {deviation:.3f} (weight: 0.3)")
        print(f"  → Total Culprit Score:   {total:.3f}")
        print()
    
    # Identify culprit
    culprit = max(scores.items(), key=lambda x: x[1])
    print(f"\n[RESULT] Most Likely Culprit: {culprit[0]} (score: {culprit[1]:.3f})")
    print(f"This variable would be removed to generate new candidate")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    
    print("\n" + "#"*70)
    print("HCAR Framework - Simple Examples")
    print("#"*70)
    
  
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--subset-demo":
        # Just demonstrate subset exploration
        demonstrate_intelligent_subset_exploration()
    else:
        # Run full simple example
        run_simple_example()
        
        # Optionally run subset demo
        print("\n\nWant to see how Intelligent Subset Exploration works?")
        print("Run: python hcar_example_simple.py --subset-demo")


if __name__ == "__main__":
    main()

