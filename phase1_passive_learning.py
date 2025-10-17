"""
Phase 1: Passive Learning for HCAR

This module implements the passive learning phase that:
1. Generates positive examples from target model
2. Detects legitimate AllDifferent patterns
3. Adds synthetic overfitted AllDifferent constraints
4. Generates and prunes fixed-arity binary bias
5. Saves all data to pickle for Phase 2

Author: HCAR Research Team
"""

import argparse
import os
import pickle
import random
import re
import sys
from itertools import combinations
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.expressions.utils import all_pairs
from cpmpy.expressions.globalconstraints import AllDifferent
from cpmpy.transformations.get_variables import get_variables

# Import benchmark construction functions
from benchmarks_global import construct_sudoku
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_nurse_rostering as nr_global


def construct_instance(benchmark_name):
    """
    Construct problem instance and oracle for given benchmark.
    
    Args:
        benchmark_name: Name of benchmark (sudoku, examtt, nurse, uefa)
        
    Returns:
        Tuple (instance, oracle)
    """
    if 'sudoku' in benchmark_name.lower():
        print("Constructing 9x9 Sudoku...")
        instance, oracle = construct_sudoku(3, 3, 9)
    
    elif 'examtt' in benchmark_name.lower():
        print("Constructing Exam Timetabling...")
        instance, oracle = ces_global(nsemesters=9, courses_per_semester=6, 
                                      slots_per_day=9, days_for_exams=14)
    
    elif 'nurse' in benchmark_name.lower():
        print("Constructing Nurse Rostering...")
        instance, oracle = nr_global()
    
    elif 'uefa' in benchmark_name.lower():
        print("Constructing UEFA Champions League...")
        from benchmarks_global.uefa import construct_uefa as construct_uefa_instance
        
        teams_data = {
            "RealMadrid": {"country": "ESP", "coefficient": 134000},
            "BayernMunich": {"country": "GER", "coefficient": 129000},
            "ManchesterCity": {"country": "ENG", "coefficient": 128000},
            "PSG": {"country": "FRA", "coefficient": 112000},
            "Liverpool": {"country": "ENG", "coefficient": 109000},
            "Barcelona": {"country": "ESP", "coefficient": 98000},
            "Juventus": {"country": "ITA", "coefficient": 95000},
            "AtleticoMadrid": {"country": "ESP", "coefficient": 94000},
            "ManchesterUnited": {"country": "ENG", "coefficient": 92000},
            "Chelsea": {"country": "ENG", "coefficient": 91000},
            "BorussiaDortmund": {"country": "GER", "coefficient": 88000},
            "Ajax": {"country": "NED", "coefficient": 82000},
            "RB Leipzig": {"country": "GER", "coefficient": 79000},
            "InterMilan": {"country": "ITA", "coefficient": 76000},
            "Sevilla": {"country": "ESP", "coefficient": 75000},
            "Napoli": {"country": "ITA", "coefficient": 74000},
            "Benfica": {"country": "POR", "coefficient": 73000},
            "Porto": {"country": "POR", "coefficient": 72000},
            "Arsenal": {"country": "ENG", "coefficient": 71000},
            "ACMilan": {"country": "ITA", "coefficient": 70000},
            "RedBullSalzburg": {"country": "AUT", "coefficient": 69000},
            "ShakhtarDonetsk": {"country": "UKR", "coefficient": 68000},
            "BayerLeverkusen": {"country": "GER", "coefficient": 67000},
            "Olympiacos": {"country": "GRE", "coefficient": 66000},
            "Celtic": {"country": "SCO", "coefficient": 65000},
            "Rangers": {"country": "SCO", "coefficient": 64000},
            "PSVEindhoven": {"country": "NED", "coefficient": 63000},
            "SportingCP": {"country": "POR", "coefficient": 62000},
            "Marseille": {"country": "FRA", "coefficient": 61000},
            "ClubBrugge": {"country": "BEL", "coefficient": 60000},
            "Galatasaray": {"country": "TUR", "coefficient": 59000},
            "Feyenoord": {"country": "NED", "coefficient": 58000}
        }
        
        instance, oracle = construct_uefa_instance(teams_data)
    
    else:
        print(f"Unknown benchmark: {benchmark_name}")
        sys.exit(1)
    
    return instance, oracle


def generate_positive_examples(oracle, variables, count=5):
    """
    Generate positive examples by solving the target model.
    
    Args:
        oracle: Oracle with target constraints
        variables: CPMpy variables
        count: Number of examples to generate
        
    Returns:
        List of example dicts: [{var_name: value, ...}, ...]
    """
    print(f"\nGenerating {count} positive examples...")
    
    # Create model with target constraints
    model = Model(oracle.constraints)
    
    # Test if satisfiable
    if not model.solve():
        print("ERROR: Target model is UNSAT!")
        return []
    
    positive_examples = []
    exclusion_constraints = []
    
    # Extract first solution
    example = {}
    for var in variables:
        if hasattr(var, 'value'):
            val = var.value()
            if val is not None:
                example[var.name] = val
                exclusion_constraints.append(var != val)
    
    if example:
        positive_examples.append(example)
        print(f"  Generated example 1/{count}")
    
    # Generate additional diverse solutions
    for i in range(1, count):
        if not exclusion_constraints:
            print(f"  Warning: No variables to constrain for diversity")
            break
        
        try:
            # Add constraint to exclude previous solution
            model += any(exclusion_constraints)
        except Exception as e:
            print(f"  Warning: Could not add exclusion constraint: {e}")
            break
        
        # Solve for another solution
        if model.solve():
            example = {}
            new_exclusion = []
            
            for var in variables:
                if hasattr(var, 'value'):
                    val = var.value()
                    if val is not None:
                        example[var.name] = val
                        new_exclusion.append(var != val)
            
            if example:
                positive_examples.append(example)
                exclusion_constraints = new_exclusion
                print(f"  Generated example {len(positive_examples)}/{count}")
        else:
            print(f"  No more solutions found after {len(positive_examples)} examples")
            break
    
    return positive_examples


def detect_structured_patterns(variables, positive_examples, grid_size=9):
    """
    Detect structured AllDifferent patterns (rows, columns, blocks) based on variable naming.
    
    This is optimized for grid-based problems like Sudoku, Exam Timetabling, Nurse Rostering.
    
    Args:
        variables: List of CPMpy variables
        positive_examples: List of example dicts
        grid_size: Grid dimension (default: 9 for Sudoku)
        
    Returns:
        List of detected AllDifferent constraints
    """
    print(f"\n  [Pattern-based detection]")
    detected = []
    var_dict = {var.name: var for var in variables}
    
    # Try to infer problem structure from variable names
    sample_var = variables[0].name
    
    # Case 1: Sudoku-style grid variables (grid[i,j] or similar)
    if '[' in sample_var and ',' in sample_var:
        print(f"  Detected grid structure (e.g., grid[i,j])")
        
        # Detect rows
        for row in range(grid_size):
            row_vars = []
            for col in range(grid_size):
                var_name = f"grid[{row},{col}]"
                if var_name in var_dict:
                    row_vars.append(var_dict[var_name])
            
            if len(row_vars) == grid_size:
                if check_alldiff_in_examples(row_vars, positive_examples):
                    detected.append(AllDifferent(row_vars))
        
        # Detect columns
        for col in range(grid_size):
            col_vars = []
            for row in range(grid_size):
                var_name = f"grid[{row},{col}]"
                if var_name in var_dict:
                    col_vars.append(var_dict[var_name])
            
            if len(col_vars) == grid_size:
                if check_alldiff_in_examples(col_vars, positive_examples):
                    detected.append(AllDifferent(col_vars))
        
        # Detect 3x3 blocks (for Sudoku)
        block_size = int(grid_size ** 0.5)
        if block_size * block_size == grid_size:
            for block_row in range(0, grid_size, block_size):
                for block_col in range(0, grid_size, block_size):
                    block_vars = []
                    for i in range(block_size):
                        for j in range(block_size):
                            var_name = f"grid[{block_row + i},{block_col + j}]"
                            if var_name in var_dict:
                                block_vars.append(var_dict[var_name])
                    
                    if len(block_vars) == grid_size:
                        if check_alldiff_in_examples(block_vars, positive_examples):
                            detected.append(AllDifferent(block_vars))
    
    # Case 2: Multi-dimensional indexed variables (e.g., var[day,shift,nurse])
    elif '[' in sample_var:
        print(f"  Detected multi-dimensional structure")
        
        # Try to infer dimensions by parsing variable names
        dimensions = {}
        for var in variables:
            match = re.match(r'.*\[(\d+),(\d+),?(\d+)?\]', var.name)
            if match:
                d1, d2 = int(match.group(1)), int(match.group(2))
                d3 = int(match.group(3)) if match.group(3) else None
                
                if 'dim1' not in dimensions or d1 > dimensions['dim1']:
                    dimensions['dim1'] = d1
                if 'dim2' not in dimensions or d2 > dimensions['dim2']:
                    dimensions['dim2'] = d2
                if d3 is not None:
                    if 'dim3' not in dimensions or d3 > dimensions['dim3']:
                        dimensions['dim3'] = d3
        
        if dimensions:
            print(f"    Dimensions inferred: {dimensions}")
            
            # For 3D: detect patterns across each dimension
            if 'dim3' in dimensions:
                # Fix first dimension, vary second and third
                for d1 in range(dimensions['dim1'] + 1):
                    vars_slice = []
                    for d2 in range(dimensions['dim2'] + 1):
                        for d3 in range(dimensions['dim3'] + 1):
                            var_name = f"var[{d1},{d2},{d3}]"
                            if var_name in var_dict:
                                vars_slice.append(var_dict[var_name])
                    
                    if len(vars_slice) >= 2:
                        if check_alldiff_in_examples(vars_slice, positive_examples):
                            detected.append(AllDifferent(vars_slice))
    
    print(f"    Found {len(detected)} structured patterns")
    return detected


def check_alldiff_in_examples(var_subset, positive_examples):
    """
    Check if a variable subset satisfies AllDifferent in all examples.
    
    Args:
        var_subset: List of variables
        positive_examples: List of example dicts
        
    Returns:
        True if AllDifferent holds in all examples
    """
    for example in positive_examples:
        values = []
        for var in var_subset:
            if var.name in example:
                values.append(example[var.name])
            else:
                return False
        
        # Check if all values are different
        if len(values) != len(set(values)):
            return False
    
    return True


def detect_alldifferent_patterns(variables, positive_examples, use_structured=True, 
                                 use_combinatorial=False, min_scope=9, max_scope=11):
    """
    Find variable subsets where all values are different across ALL examples.
    
    Uses two strategies:
    1. Pattern-based: Detects structured patterns (rows, columns, blocks)
    2. Combinatorial: Enumerates all subsets (expensive, optional)
    
    Args:
        variables: List of CPMpy variables
        positive_examples: List of example dicts
        use_structured: Use pattern-based detection (default: True)
        use_combinatorial: Use combinatorial search (default: False)
        min_scope: Minimum scope size for combinatorial (default: 7)
        max_scope: Maximum scope size for combinatorial (default: 11)
        
    Returns:
        List of CPMpy AllDifferent constraints
    """
    print(f"\nDetecting AllDifferent patterns...")
    print(f"  Variables: {len(variables)}")
    print(f"  Strategy: ", end="")
    
    strategies = []
    if use_structured:
        strategies.append("pattern-based")
    if use_combinatorial:
        strategies.append("combinatorial")
    print(", ".join(strategies))
    
    detected = []
    
    # Strategy 1: Pattern-based detection
    if use_structured:
        structured_patterns = detect_structured_patterns(variables, positive_examples)
        detected.extend(structured_patterns)
    
    # Strategy 2: Combinatorial search (optional, expensive)
    if use_combinatorial:
        print(f"\n  [Combinatorial search]")
        print(f"    Scope range: {min_scope} to {min(max_scope, len(variables))}")
        
        var_list = list(variables)
        
        for scope_size in range(min_scope, min(max_scope + 1, len(variables) + 1)):
            print(f"    Checking scope size {scope_size}...", end=" ")
            
            count = 0
            # Enumerate all subsets of this size
            for var_subset in combinations(var_list, scope_size):
                if check_alldiff_in_examples(var_subset, positive_examples):
                    constraint = AllDifferent(list(var_subset))
                    detected.append(constraint)
                    count += 1
            
            print(f"found {count} patterns")
    
    print(f"  Total detected: {len(detected)} AllDifferent patterns")
    return detected


def extract_alldifferent_constraints(oracle):
    """
    Extract only AllDifferent constraints from oracle.
    
    Args:
        oracle: Oracle object with constraints attribute
        
    Returns:
        List of AllDifferent constraints
    """
    alldiff_constraints = []
    for c in oracle.constraints:
        if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
            alldiff_constraints.append(c)
    return alldiff_constraints


def generate_overfitted_alldifferent(variables, positive_examples, target_alldiffs, count=4, max_attempts=1000):
    """
    Generate synthetic overfitted AllDifferent constraints.
    
    These constraints satisfy all examples but are NOT in the target model.
    
    Args:
        variables: List of CPMpy variables
        positive_examples: List of example dicts
        target_alldiffs: Target AllDifferent constraints (to avoid duplicates)
        count: Number of overfitted constraints to generate
        max_attempts: Maximum random attempts
        
    Returns:
        List of overfitted AllDifferent constraints
    """
    print(f"\nGenerating {count} overfitted AllDifferent constraints...")
    
    # Convert target constraints to string representations for comparison
    target_strs = set()
    for c in target_alldiffs:
        # Normalize by sorting variable names
        scope_vars = get_variables([c])
        var_names = tuple(sorted([v.name for v in scope_vars]))
        target_strs.add(var_names)
    
    overfitted = []
    var_list = list(variables)
    attempts = 0
    
    while len(overfitted) < count and attempts < max_attempts:
        attempts += 1
        
        # Random scope size between 3 and 7
        scope_size = random.randint(3, min(7, len(var_list)))
        
        # Random subset of variables
        var_subset = random.sample(var_list, scope_size)
        
        # Check if this subset is already in target
        var_names = tuple(sorted([v.name for v in var_subset]))
        if var_names in target_strs:
            continue
        
        # Check if it satisfies AllDifferent in all examples
        is_alldiff_pattern = True
        for example in positive_examples:
            values = []
            for var in var_subset:
                if var.name in example:
                    values.append(example[var.name])
                else:
                    is_alldiff_pattern = False
                    break
            
            if not is_alldiff_pattern:
                break
            
            # Check if all values are different
            if len(values) != len(set(values)):
                is_alldiff_pattern = False
                break
        
        if is_alldiff_pattern:
            # This is overfitted!
            constraint = AllDifferent(var_subset)
            overfitted.append(constraint)
            target_strs.add(var_names)  # Avoid duplicates in overfitted set
            print(f"  Generated overfitted constraint {len(overfitted)}/{count}: scope size = {scope_size}")
    
    if len(overfitted) < count:
        print(f"  Warning: Only generated {len(overfitted)} overfitted constraints (target was {count})")
    
    return overfitted


def generate_binary_bias(variables, language):
    """
    Generate all binary constraints using language relations.
    
    Args:
        variables: List of CPMpy variables
        language: List of relation strings ['==', '!=', '<', '>', '<=', '>=']
        
    Returns:
        List of all pairwise constraints
    """
    print(f"\nGenerating binary bias...")
    print(f"  Variables: {len(variables)}")
    print(f"  Language: {language}")
    
    bias_constraints = []
    
    # Generate all pairs
    for v1, v2 in all_pairs(variables):
        for relation in language:
            if relation == '==':
                bias_constraints.append(v1 == v2)
            elif relation == '!=':
                bias_constraints.append(v1 != v2)
            elif relation == '<':
                bias_constraints.append(v1 < v2)
            elif relation == '>':
                bias_constraints.append(v1 > v2)
            elif relation == '<=':
                bias_constraints.append(v1 <= v2)
            elif relation == '>=':
                bias_constraints.append(v1 >= v2)
    
    print(f"  Generated {len(bias_constraints)} binary constraints")
    return bias_constraints


def prune_bias_with_examples(bias_constraints, positive_examples, variables):
    """
    Remove constraints inconsistent with E+.
    
    For each constraint c:
        - Evaluate c on each example
        - If c is violated by ANY example, remove it
    
    Args:
        bias_constraints: List of binary constraints
        positive_examples: List of example dicts
        variables: List of CPMpy variables
        
    Returns:
        List of pruned bias (only consistent constraints)
    """
    print(f"\nPruning bias with {len(positive_examples)} examples...")
    print(f"  Initial bias size: {len(bias_constraints)}")
    
    # Create variable name to variable mapping
    var_mapping = {var.name: var for var in variables}
    
    pruned_bias = []
    
    for idx, constraint in enumerate(bias_constraints):
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(bias_constraints)} constraints...")
        
        # Check if constraint is consistent with all examples
        is_consistent = True
        
        for example in positive_examples:
            # Assign values to variables from example
            for var_name, value in example.items():
                if var_name in var_mapping:
                    var = var_mapping[var_name]
                    # Set the variable's value temporarily
                    var._value = value
            
            # Evaluate constraint
            try:
                if not constraint.value():
                    # Constraint is violated by this example
                    is_consistent = False
                    break
            except:
                # If evaluation fails, skip this constraint
                is_consistent = False
                break
        
        # Clear variable values
        for var in variables:
            var._value = None
        
        if is_consistent:
            pruned_bias.append(constraint)
    
    print(f"  Pruned bias size: {len(pruned_bias)}")
    print(f"  Removed: {len(bias_constraints) - len(pruned_bias)} constraints")
    
    return pruned_bias


def run_phase1(benchmark_name, output_dir='phase1_output', num_examples=5, num_overfitted=4):
    """
    Run Phase 1 passive learning for a benchmark.
    
    Args:
        benchmark_name: Name of benchmark
        output_dir: Output directory for pickle files
        num_examples: Number of positive examples to generate
        num_overfitted: Number of overfitted constraints to add
        
    Returns:
        Path to output pickle file
    """
    print(f"\n{'='*70}")
    print(f"Phase 1: Passive Learning - {benchmark_name}")
    print(f"{'='*70}")
    
    # 1. Load benchmark
    instance, oracle = construct_instance(benchmark_name)
    
    # Setup oracle
    oracle.variables_list = cpm_array(instance.X)
    
    # 2. Generate positive examples
    positive_examples = generate_positive_examples(oracle, instance.X, count=num_examples)
    
    if len(positive_examples) < num_examples:
        print(f"\nWarning: Only generated {len(positive_examples)} examples (requested {num_examples})")
    
    if len(positive_examples) == 0:
        print(f"\nERROR: Could not generate any positive examples!")
        return None
    
    # 3. Detect legitimate AllDifferent patterns
    detected_alldiffs = detect_alldifferent_patterns(instance.X, positive_examples)
    
    # 4. Extract target AllDifferent for comparison
    target_alldiffs = extract_alldifferent_constraints(oracle)
    print(f"\nTarget model has {len(target_alldiffs)} AllDifferent constraints")
    
    # 5. Generate overfitted AllDifferent constraints
    overfitted_alldiffs = generate_overfitted_alldifferent(
        instance.X, positive_examples, target_alldiffs, count=num_overfitted
    )
    
    # 6. Combine: CG = detected + overfitted
    CG = detected_alldiffs + overfitted_alldiffs
    print(f"\nCombined CG: {len(detected_alldiffs)} detected + {len(overfitted_alldiffs)} overfitted = {len(CG)} total")
    
    # 6b. Create informed priors for each constraint
    # Detected constraints (pattern-based, scope 9) → 0.8 (high confidence)
    # Overfitted constraints (synthetic, scope 3-5) → 0.3 (low confidence)
    initial_probabilities = {}
    for c in detected_alldiffs:
        initial_probabilities[c] = 0.8  # High prior for detected constraints
    for c in overfitted_alldiffs:
        initial_probabilities[c] = 0.3  # Low prior for overfitted constraints
    
    print(f"Initial probabilities: {len(detected_alldiffs)} @ 0.8 (detected), {len(overfitted_alldiffs)} @ 0.3 (overfitted)")
    
    # 7. Generate complete binary bias
    language = ['==', '!=', '<', '>', '<=', '>=']
    B_fixed = generate_binary_bias(instance.X, language)
    
    # 8. Prune B_fixed with E+
    B_fixed_pruned = prune_bias_with_examples(B_fixed, positive_examples, instance.X)
    
    # 9. Save to pickle
    output_data = {
        'CG': CG,  # List of CPMpy AllDifferent constraints
        'B_fixed': B_fixed_pruned,  # List of pruned binary constraints
        'E+': positive_examples,  # List of example dicts
        'variables': instance.X,  # CPMpy variables
        'initial_probabilities': initial_probabilities,  # Informed priors for each constraint
        'metadata': {
            'benchmark': benchmark_name,
            'num_examples': len(positive_examples),
            'num_detected_alldiffs': len(detected_alldiffs),
            'num_overfitted_alldiffs': len(overfitted_alldiffs),
            'num_bias_initial': len(B_fixed),
            'num_bias_pruned': len(B_fixed_pruned),
            'target_alldiff_count': len(target_alldiffs),
            'prior_detected': 0.8,
            'prior_overfitted': 0.3
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{benchmark_name}_phase1.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n{'='*70}")
    print(f"Phase 1 Complete!")
    print(f"{'='*70}")
    print(f"Output saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Positive examples: {len(positive_examples)}")
    print(f"  Detected AllDifferent: {len(detected_alldiffs)}")
    print(f"  Overfitted AllDifferent: {len(overfitted_alldiffs)}")
    print(f"  Total CG: {len(CG)}")
    print(f"  Binary bias (initial): {len(B_fixed)}")
    print(f"  Binary bias (pruned): {len(B_fixed_pruned)}")
    print(f"  Target AllDifferent: {len(target_alldiffs)}")
    print(f"{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 1: Passive Learning for HCAR'
    )
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['sudoku', 'examtt', 'nurse', 'uefa'],
                       help='Benchmark name')
    parser.add_argument('--output_dir', type=str, default='phase1_output',
                       help='Output directory for pickle files')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of positive examples (default: 5)')
    parser.add_argument('--num_overfitted', type=int, default=4,
                       help='Number of overfitted constraints to add (default: 4)')
    
    args = parser.parse_args()
    
    output_path = run_phase1(
        args.benchmark, 
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        num_overfitted=args.num_overfitted
    )
    
    if output_path:
        print(f"✓ Phase 1 complete. Data saved to: {output_path}")
    else:
        print(f"✗ Phase 1 failed!")
        sys.exit(1)

