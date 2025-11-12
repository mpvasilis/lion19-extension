

import argparse
import math
import os
import pickle
import random
import re
import sys
from itertools import combinations
import cpmpy as cp
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.expressions.utils import all_pairs
from cpmpy.expressions.globalconstraints import AllDifferent
from cpmpy.transformations.get_variables import get_variables

from benchmarks_global import construct_sudoku, construct_jsudoku, construct_latin_square
from benchmarks_global import construct_graph_coloring_register, construct_graph_coloring_scheduling
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering as nr_global


def construct_instance(benchmark_name):

    if 'graph_coloring_register' in benchmark_name.lower() or 'register' in benchmark_name.lower():
        print("Constructing Graph Coloring (Register Allocation)...")
        result = construct_graph_coloring_register()
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'graph_coloring_scheduling' in benchmark_name.lower() or benchmark_name.lower() == 'scheduling':
        print("Constructing Graph Coloring (Course Scheduling)...")
        result = construct_graph_coloring_scheduling()
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'latin_square' in benchmark_name.lower() or 'latin' in benchmark_name.lower():
        print("Constructing 9x9 Latin Square...")
        result = construct_latin_square(n=9)
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'jsudoku' in benchmark_name.lower():
        print("Constructing 9x9 JSudoku (Jigsaw Sudoku)...")
        result = construct_jsudoku(grid_size=9)
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'sudoku_gt' in benchmark_name.lower() or 'sudoku_greater' in benchmark_name.lower():
        print("Constructing 9x9 Sudoku with Greater-Than constraints...")
        result = construct_sudoku_greater_than(3, 3, 9)
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'sudoku' in benchmark_name.lower():
        print("Constructing 9x9 Sudoku...")
        result = construct_sudoku(3, 3, 9)
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'examtt_v1' in benchmark_name.lower() or 'examtt_variant1' in benchmark_name.lower():
        print("Constructing Exam Timetabling Variant 1...")
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5, 
                                           slots_per_day=6, days_for_exams=10)
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'examtt_v2' in benchmark_name.lower() or 'examtt_variant2' in benchmark_name.lower():
        print("Constructing Exam Timetabling Variant 2...")
        result = construct_examtt_variant2(nsemesters=8, courses_per_semester=7, 
                                           slots_per_day=8, days_for_exams=12)
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'examtt' in benchmark_name.lower():
        print("Constructing Exam Timetabling...")
        result = ces_global(nsemesters=9, courses_per_semester=6, 
                           slots_per_day=9, days_for_exams=14)
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
    elif 'nurse' in benchmark_name.lower():
        print("Constructing Nurse Rostering...")
        result = nr_global()
        
        if len(result) == 3:
            instance, oracle, overfitted_constraints = result
            print(f"  Received {len(overfitted_constraints)} overfitted constraints from benchmark")
            return instance, oracle, overfitted_constraints
        else:
            instance, oracle = result
            return instance, oracle
    
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

    print(f"\nGenerating {count} positive examples...")
    
    model = Model(oracle.constraints)
    
    if not model.solve():
        print("ERROR: Target model is UNSAT!")
        return []
    
    positive_examples = []
    exclusion_constraints = []
    
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

    for i in range(1, count):
        if not exclusion_constraints:
            print(f"  Warning: No variables to constrain for diversity")
            break
        
        try:

            model += any(exclusion_constraints)
        except Exception as e:
            print(f"  Warning: Could not add exclusion constraint: {e}")
            break
        
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

    print(f"\n  [Pattern-based detection]")
    detected = []
    var_dict = {var.name: var for var in variables}

    sample_var = variables[0].name

    if '[' in sample_var and ',' in sample_var:
        print(f"  Detected grid structure (e.g., grid[i,j])")

        for row in range(grid_size):
            row_vars = []
            for col in range(grid_size):
                var_name = f"grid[{row},{col}]"
                if var_name in var_dict:
                    row_vars.append(var_dict[var_name])
            
            if len(row_vars) == grid_size:
                if check_alldiff_in_examples(row_vars, positive_examples):
                    detected.append(AllDifferent(row_vars))

        for col in range(grid_size):
            col_vars = []
            for row in range(grid_size):
                var_name = f"grid[{row},{col}]"
                if var_name in var_dict:
                    col_vars.append(var_dict[var_name])
            
            if len(col_vars) == grid_size:
                if check_alldiff_in_examples(col_vars, positive_examples):
                    detected.append(AllDifferent(col_vars))

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
    
    elif '[' in sample_var:
        print(f"  Detected multi-dimensional structure")

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

            if 'dim3' in dimensions:

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
    for example in positive_examples:
        values = []
        for var in var_subset:
            if var.name in example:
                values.append(example[var.name])
            else:
                return False
        
        if len(values) != len(set(values)):
            return False
    
    return True


def detect_alldifferent_patterns(variables, positive_examples, use_structured=True, 
                                 use_combinatorial=False, min_scope=9, max_scope=11):

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

    if use_structured:
        structured_patterns = detect_structured_patterns(variables, positive_examples)
        detected.extend(structured_patterns)

    if use_combinatorial:
        print(f"\n  [Combinatorial search]")
        print(f"    Scope range: {min_scope} to {min(max_scope, len(variables))}")
        
        var_list = list(variables)
        
        for scope_size in range(min_scope, min(max_scope + 1, len(variables) + 1)):
            print(f"    Checking scope size {scope_size}...", end=" ")
            
            count = 0

            for var_subset in combinations(var_list, scope_size):
                if check_alldiff_in_examples(var_subset, positive_examples):
                    constraint = AllDifferent(list(var_subset))
                    detected.append(constraint)
                    count += 1
            
            print(f"found {count} patterns")
    
    print(f"  Total detected: {len(detected)} AllDifferent patterns")
    return detected


def extract_alldifferent_constraints(oracle):

    alldiff_constraints = []
    for c in oracle.constraints:
        if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
            alldiff_constraints.append(c)
    return alldiff_constraints


def build_constraint_violation(constraint):

    if isinstance(constraint, AllDifferent):
        variables = []
        if getattr(constraint, "args", None):
            args = constraint.args
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                variables = list(args[0])
            else:
                variables = list(args)
        if not variables:
            variables = list(get_variables(constraint))

        variables = list(variables)
        violation_terms = []
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                violation_terms.append(variables[i] == variables[j])

        if not violation_terms:
            return 0 == 1

        return cp.sum(violation_terms) >= 1

    return ~constraint


def is_constraint_implied(candidate_constraint, base_constraints, variables):

    if not base_constraints:
        return False

    violation_expr = build_constraint_violation(candidate_constraint)

    test_model = Model()
    test_model += base_constraints
    test_model += violation_expr

    has_counterexample = test_model.solve()

    for var in variables:
        if hasattr(var, "_value"):
            var._value = None

    return not has_counterexample


def extract_grid_info(variables):

    pattern = re.compile(r"grid\[(\d+),(\d+)\]")
    entries = []

    for var in variables:
        name = getattr(var, "name", None)
        if not name:
            continue
        match = pattern.match(str(name))
        if not match:
            continue
        row = int(match.group(1))
        col = int(match.group(2))
        entries.append((var, row, col))

    if not entries:
        return None

    max_row = max(entry[1] for entry in entries) + 1
    max_col = max(entry[2] for entry in entries) + 1

    if max_row <= 0 or max_col <= 0:
        return None

    block_size_row = int(math.sqrt(max_row))
    block_size_col = int(math.sqrt(max_col))

    if block_size_row == 0 or block_size_col == 0:
        return None

    if block_size_row * block_size_row != max_row or block_size_col * block_size_col != max_col:
        return None

    coords = {}
    rows = {}
    cols = {}
    blocks = {}

    for var, row, col in entries:
        block_row = row // block_size_row
        block_col = col // block_size_col
        block_id = (block_row, block_col)

        coords[var] = (row, col, block_id)
        rows.setdefault(row, []).append(var)
        cols.setdefault(col, []).append(var)
        blocks.setdefault(block_id, []).append(var)

    return {
        'coords': coords,
        'rows': rows,
        'cols': cols,
        'blocks': blocks,
        'block_size': (block_size_row, block_size_col)
    }


def discover_non_implied_pairs(variables, positive_examples, target_constraints, grid_info=None, max_pairs=25, max_checks=2000):

    if grid_info is None or not positive_examples:
        return []

    coords = grid_info.get('coords', {})
    if not coords:
        return []

    var_list = list(coords.keys())
    unique_pairs = []
    attempts = 0
    rng = random.Random(42)

    while len(unique_pairs) < max_pairs and attempts < max_checks:
        attempts += 1
        v1, v2 = rng.sample(var_list, 2)
        row1, col1, block1 = coords[v1]
        row2, col2, block2 = coords[v2]

        if row1 == row2 or col1 == col2 or block1 == block2:
            continue

        if any(example[v1.name] == example[v2.name] for example in positive_examples if v1.name in example and v2.name in example):
            continue

        candidate = AllDifferent([v1, v2])
        if is_constraint_implied(candidate, target_constraints, variables):
            continue

        unique_pairs.append((v1, v2))

    return unique_pairs


def generate_overfitted_alldifferent(variables, positive_examples, target_alldiffs, count=4, max_attempts=10000, grid_info=None, pair_seeds=None):

    print(f"\nGenerating {count} overfitted AllDifferent constraints...")

    target_strs = set()
    target_sets = []  
    for c in target_alldiffs:

        scope_vars = get_variables([c])
        var_names = tuple(sorted([v.name for v in scope_vars]))
        target_strs.add(var_names)
        target_sets.append(set(var_names))
    
    overfitted = []
    implication_base = list(target_alldiffs)
    var_list = list(variables)
    block_mappings = []
    coords = {}

    if grid_info:
        block_mappings = [(bid, vars_in_block) for bid, vars_in_block in grid_info.get('blocks', {}).items() if len(vars_in_block) > 1]
        coords = grid_info.get('coords', {})
    attempts = 0

    def is_valid_all_diff(var_subset):
        for example in positive_examples:
            values = []
            for var in var_subset:
                if var.name in example:
                    values.append(example[var.name])
                else:
                    return False
            if len(values) != len(set(values)):
                return False
        return True
    
    while len(overfitted) < count and attempts < max_attempts:
        attempts += 1

        var_subset = None
        generation_mode = 'standard'
        roll = random.random()

        if pair_seeds and roll < 0.2:
            generation_mode = 'pair'
        elif block_mappings and roll < 0.4:
            generation_mode = 'block'
        elif grid_info and 0.4 <= roll < 0.6:
            generation_mode = 'sparse'
        elif roll >= 0.6 and roll < 0.8:
            generation_mode = 'wide'

        if generation_mode == 'pair' and pair_seeds:
            base_pair = random.choice(pair_seeds)
            subset_pool = [v for v in var_list if v not in base_pair]
            random.shuffle(subset_pool)
            target_size = random.randint(4, min(9, len(var_list)))
            selection = list(base_pair)
            for candidate_var in subset_pool:
                if candidate_var in selection:
                    continue
                if any(candidate_var.name not in example for example in positive_examples):
                    continue
                duplicate = False
                for example in positive_examples:
                    existing_values = {example[var.name] for var in selection if var.name in example}
                    if candidate_var.name in example and example[candidate_var.name] in existing_values:
                        duplicate = True
                        break
                if duplicate:
                    continue
                selection.append(candidate_var)
                if len(selection) >= target_size:
                    break
            if len(selection) >= 3:
                var_subset = selection
        elif generation_mode == 'block' and block_mappings:
            block_id, block_vars = random.choice(block_mappings)
            other_vars = [v for v in var_list if v not in block_vars]
            if not block_vars:
                continue
            hybrid_target = random.randint(4, min(9, len(block_vars) + len(other_vars)))
            take_from_block = min(len(block_vars), max(2, hybrid_target - max(1, hybrid_target // 3)))
            take_from_rest = max(0, hybrid_target - take_from_block)
            block_selection = random.sample(block_vars, take_from_block)
            remainder_selection = random.sample(other_vars, min(take_from_rest, len(other_vars))) if take_from_rest > 0 and other_vars else []
            var_subset = block_selection + remainder_selection
        elif generation_mode == 'sparse' and coords:
            available = list(coords.keys())
            random.shuffle(available)
            used_rows = set()
            used_cols = set()
            used_blocks = set()
            target_size = random.randint(5, min(9, len(available)))
            selection = []
            for var in available:
                row, col, block_id = coords[var]
                if row in used_rows or col in used_cols or block_id in used_blocks:
                    continue
                selection.append(var)
                used_rows.add(row)
                used_cols.add(col)
                used_blocks.add(block_id)
                if len(selection) >= target_size:
                    break
            if len(selection) >= 4:
                var_subset = selection
        elif generation_mode == 'wide':
            if len(var_list) < 8:
                continue
            var_subset = random.sample(var_list, random.randint(8, min(9, len(var_list))))
        else:
            var_subset = random.sample(var_list, random.randint(4, min(7, len(var_list))))

        if not var_subset or len(var_subset) < 3:
            continue

        scope_size = len(var_subset)

        var_names = tuple(sorted([v.name for v in var_subset]))
        if var_names in target_strs:
            continue

        var_names_set = set(var_names)
        is_subset_of_target = False
        for target_set in target_sets:
            if var_names_set.issubset(target_set):
                is_subset_of_target = True
                break
        
        if is_subset_of_target:
            continue  

        if is_valid_all_diff(var_subset):

            constraint = AllDifferent(var_subset)
            if is_constraint_implied(constraint, implication_base, variables):
                print(f"  [SKIP] Candidate implied by target model; discarding (scope size = {scope_size})")
                continue
            overfitted.append(constraint)
            target_strs.add(var_names)  
            implication_base.append(constraint)
            print(f"  Generated overfitted constraint {len(overfitted)}/{count}: scope size = {scope_size}")

    if len(overfitted) < count:
        print(
            f"  Warning: Only generated {len(overfitted)} overfitted constraints out of {count} after {attempts} attempts."
        )
        print("  Attempting fallback generation allowing subsets of target constraints...")

        fallback_attempts = 0
        remaining_needed = count - len(overfitted)
        fallback_max_attempts = max(max_attempts * 5, remaining_needed * 1000)
        while len(overfitted) < count and fallback_attempts < fallback_max_attempts:
            fallback_attempts += 1

            scope_size = random.randint(3, min(7, len(var_list)))
            var_subset = random.sample(var_list, scope_size)

            var_names = tuple(sorted([v.name for v in var_subset]))
            if var_names in target_strs:
                continue

            if is_valid_all_diff(var_subset):
                constraint = AllDifferent(var_subset)
                if is_constraint_implied(constraint, implication_base, variables):
                    continue
                overfitted.append(constraint)
                target_strs.add(var_names)
                implication_base.append(constraint)
                print(
                    f"  [fallback] Generated overfitted constraint {len(overfitted)}/{count}: scope size = {scope_size}"
                )

        if len(overfitted) < count:
            print(
                f"  Warning: Fallback generation still produced only {len(overfitted)} overfitted constraints (target was {count})."
            )
            print("  Attempting deterministic generation to complete target...")

            deterministic_generated = 0
            for scope_size in range(3, min(8, len(var_list) + 1)):
                if len(overfitted) >= count:
                    break
                for start_idx in range(0, len(var_list) - scope_size + 1):
                    var_subset = var_list[start_idx:start_idx + scope_size]
                    var_names = tuple(sorted([v.name for v in var_subset]))
                    if var_names in target_strs:
                        continue
                    if not is_valid_all_diff(var_subset):
                        continue

                    constraint = AllDifferent(var_subset)
                    if is_constraint_implied(constraint, implication_base, variables):
                        continue
                    overfitted.append(constraint)
                    target_strs.add(var_names)
                    deterministic_generated += 1
                    implication_base.append(constraint)
                    print(
                        f"  [deterministic] Generated overfitted constraint {len(overfitted)}/{count}: scope size = {scope_size}"
                    )
                    if len(overfitted) >= count:
                        break

            if deterministic_generated == 0 and len(overfitted) < count:
                print("  Warning: Deterministic generation could not reach the desired count either.")
    
    return overfitted


def generate_binary_bias(variables, language):
    
    print(f"\nGenerating binary bias...")
    print(f"  Variables: {len(variables)}")
    print(f"  Language: {language}")
    
    bias_constraints = []

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
    
    print(f"\nPruning bias with {len(positive_examples)} examples...")
    print(f"  Initial bias size: {len(bias_constraints)}")

    var_mapping = {var.name: var for var in variables}
    
    pruned_bias = []
    
    for idx, constraint in enumerate(bias_constraints):
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(bias_constraints)} constraints...")

        is_consistent = True
        
        for example in positive_examples:

            for var_name, value in example.items():
                if var_name in var_mapping:
                    var = var_mapping[var_name]

                    var._value = value

            try:
                if not constraint.value():

                    is_consistent = False
                    break
            except:

                is_consistent = False
                break

        for var in variables:
            var._value = None
        
        if is_consistent:
            pruned_bias.append(constraint)
    
    print(f"  Pruned bias size: {len(pruned_bias)}")
    print(f"  Removed: {len(bias_constraints) - len(pruned_bias)} constraints")
    
    return pruned_bias


def run_phase1(benchmark_name, output_dir='phase1_output', num_examples=5, num_overfitted=4):
    
    print(f"\n{'='*70}")
    print(f"Phase 1: Passive Learning - {benchmark_name}")
    print(f"{'='*70}")

    result = construct_instance(benchmark_name)
    
    
    if len(result) == 3:
        instance, oracle, overfitted_constraints_from_benchmark = result
        print(f"Using {len(overfitted_constraints_from_benchmark)} overfitted constraints from benchmark")
    else:
        instance, oracle = result
        overfitted_constraints_from_benchmark = None
        print("No overfitted constraints provided, will generate random overfitted constraints")

    oracle.variables_list = cpm_array(instance.X)

    positive_examples = generate_positive_examples(oracle, instance.X, count=num_examples)
    
    if len(positive_examples) < num_examples:
        print(f"\nWarning: Only generated {len(positive_examples)} examples (requested {num_examples})")
    
    if len(positive_examples) == 0:
        print(f"\nERROR: Could not generate any positive examples!")
        return None

    detected_alldiffs = detect_alldifferent_patterns(instance.X, positive_examples)

    target_alldiffs = extract_alldifferent_constraints(oracle)
    print(f"\nTarget model has {len(target_alldiffs)} AllDifferent constraints")


    detected_strs = set()
    for c in detected_alldiffs:
        scope_vars = get_variables([c])
        var_names = tuple(sorted([v.name for v in scope_vars]))
        detected_strs.add(var_names)
    
    missing_targets = []
    for c in target_alldiffs:
        scope_vars = get_variables([c])
        var_names = tuple(sorted([v.name for v in scope_vars]))
        if var_names not in detected_strs:
            missing_targets.append(c)
            detected_strs.add(var_names)  
    
    if missing_targets:
        print(f"\n[APPEND] Pattern detection missed {len(missing_targets)} target constraints")
        print(f"         Appending them to ensure 100% target coverage:")
        for c in missing_targets:
            print(f"         + {c}")
    else:
        print(f"\n[SUCCESS] Pattern detection found all {len(target_alldiffs)} target constraints!")

    all_target_constraints = detected_alldiffs + missing_targets
    print(f"\nComplete target coverage: {len(detected_alldiffs)} detected + {len(missing_targets)} appended = {len(all_target_constraints)} total target constraints")

    grid_info = extract_grid_info(instance.X)
    if grid_info and grid_info.get('blocks'):
        print(f"\nDetected {len(grid_info.get('blocks', {}))} block groups for overfitted generation")
    pair_seeds = discover_non_implied_pairs(instance.X, positive_examples, all_target_constraints, grid_info=grid_info)
    if pair_seeds:
        print(f"  Identified {len(pair_seeds)} non-implied variable pairs for overfitted seeding")

    if overfitted_constraints_from_benchmark is not None and len(overfitted_constraints_from_benchmark) > 0:

        print(f"\n[MOCK] Received {len(overfitted_constraints_from_benchmark)} overfitted constraints from benchmark")
        
        alldiff_overfitteds = []
        other_overfitteds = []
        for c in overfitted_constraints_from_benchmark:

            if isinstance(c, AllDifferent) or (hasattr(c, 'name') and 'alldifferent' in str(c.name).lower()):
                alldiff_overfitteds.append(c)
            else:
                other_overfitteds.append(c)
        
        if other_overfitteds:
            print(f"       Filtering: Keeping {len(alldiff_overfitteds)} AllDifferent, discarding {len(other_overfitteds)} other types")
            print(f"       Discarded types:")
            for i, c in enumerate(other_overfitteds[:5], 1):  
                print(f"         - {c}")
            if len(other_overfitteds) > 5:
                print(f"         ... and {len(other_overfitteds) - 5} more")
        
        overfitted_constraints = list(alldiff_overfitteds)

        if len(overfitted_constraints) < num_overfitted:
            needed = num_overfitted - len(overfitted_constraints)
            print(
                f"\n[MOCK] Benchmark provided {len(overfitted_constraints)} overfitted constraints; generating {needed} more to reach target of {num_overfitted}."
            )
            additional_overfitteds = generate_overfitted_alldifferent(
                instance.X,
                positive_examples,
                target_alldiffs + overfitted_constraints,
                count=needed,
                grid_info=grid_info,
                pair_seeds=pair_seeds
            )
            overfitted_constraints.extend(additional_overfitteds)
        elif len(overfitted_constraints) > num_overfitted:
            print(
                f"\n[MOCK] Benchmark provided {len(overfitted_constraints)} overfitted constraints; trimming to requested {num_overfitted}."
            )
            overfitted_constraints = overfitted_constraints[:num_overfitted]

        print(f"\n[MOCK] Using {len(overfitted_constraints)} AllDifferent overfitted constraints")
        for i, c in enumerate(overfitted_constraints, 1):
            print(f"       Mock {i}: {c}")
    else:

        overfitted_constraints = generate_overfitted_alldifferent(
            instance.X,
            positive_examples,
            all_target_constraints,
            count=num_overfitted,
            grid_info=grid_info,
            pair_seeds=pair_seeds
        )

    CG = all_target_constraints + overfitted_constraints
    print(f"\nCombined CG (before dedup): {len(all_target_constraints)} target + {len(overfitted_constraints)} overfitted = {len(CG)} total")



    seen_patterns = {}  

    for c in all_target_constraints:
        scope_vars = get_variables([c])
        pattern = tuple(sorted([v.name for v in scope_vars]))
        if pattern not in seen_patterns:
            seen_patterns[pattern] = (c, 0, 'target')  

    for c in overfitted_constraints:
        scope_vars = get_variables([c])
        pattern = tuple(sorted([v.name for v in scope_vars]))
        if pattern not in seen_patterns:
            seen_patterns[pattern] = (c, 1, 'overfitted')  

    CG = set()
    dedup_target_count = 0
    dedup_overfitted_count = 0
    duplicates_removed = len(all_target_constraints) + len(overfitted_constraints) - len(seen_patterns)
    
    for constraint, priority, source in seen_patterns.values():
        CG.add(constraint)  
        if source == 'target':
            dedup_target_count += 1
        else:
            dedup_overfitted_count += 1
    
    print(f"Deduplicated CG (set): {dedup_target_count} target + {dedup_overfitted_count} overfitted = {len(CG)} total")
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate constraint patterns")



    initial_probabilities = {}

    for constraint, priority, source in seen_patterns.values():
        if source == 'target':
            initial_probabilities[constraint] = 0.8  
        else:
            initial_probabilities[constraint] = 0.3  
    
    print(f"Initial probabilities: {dedup_target_count} @ 0.8 (target), {dedup_overfitted_count} @ 0.3 (overfitted)")

    language = ['==', '!=', '<', '>', '<=', '>=']
    B_fixed = generate_binary_bias(instance.X, language)

    B_fixed_pruned = prune_bias_with_examples(B_fixed, positive_examples, instance.X)

    output_data = {
        'CG': CG,  
        'B_fixed': B_fixed_pruned,  
        'E+': positive_examples,  
        'variables': instance.X,  
        'initial_probabilities': initial_probabilities,  
        'metadata': {
            'benchmark': benchmark_name,
            'num_examples': len(positive_examples),
            'num_detected_alldiffs': len(detected_alldiffs),
            'num_appended_alldiffs': len(missing_targets),
            'num_target_alldiffs': len(all_target_constraints),
            'num_overfitted_alldiffs': len(overfitted_constraints),
            'num_target_alldiffs_dedup': dedup_target_count,  
            'num_overfitted_alldiffs_dedup': dedup_overfitted_count,  
            'num_duplicates_removed': duplicates_removed,
            'use_overfitted_constraints': overfitted_constraints_from_benchmark is not None,
            'num_bias_initial': len(B_fixed),
            'num_bias_pruned': len(B_fixed_pruned),
            'target_alldiff_count': len(target_alldiffs),
            'prior_target': 0.8,
            'prior_overfitted': 0.3,
            'target_coverage': '100%'
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
    print(f"  Target AllDifferent: {len(target_alldiffs)}")
    print(f"    - Detected by pattern: {len(detected_alldiffs)}")
    print(f"    - Appended (missed): {len(missing_targets)}")
    print(f"  Overfitted constraints: {len(overfitted_constraints)}")
    print(f"    - Source: {'Mock from benchmark' if overfitted_constraints_from_benchmark else 'Random generation'}")
    print(f"  Total CG: {len(CG)} (TARGET COVERAGE: 100%)")
    print(f"  Binary bias (initial): {len(B_fixed)}")
    print(f"  Binary bias (pruned): {len(B_fixed_pruned)}")
    print(f"{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 1: Passive Learning for HCAR'
    )
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['sudoku', 'sudoku_gt', 'examtt', 'examtt_v1', 'examtt_v2', 'nurse', 'uefa', 
                               'graph_coloring_register', 'graph_coloring_scheduling', 'latin_square', 'jsudoku'],
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
        print(f"[OK] Phase 1 complete. Data saved to: {output_path}")
    else:
        print(f"[ERROR] Phase 1 failed!")
        sys.exit(1)

