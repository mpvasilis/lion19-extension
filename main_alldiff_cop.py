import argparse
import os
import pickle
import time
import sys
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.globalconstraints import AllDifferent
from pycona.utils import get_kappa, get_con_subset
from benchmarks_global import construct_sudoku, construct_jsudoku, construct_latin_square
from benchmarks_global import construct_graph_coloring_register, construct_graph_coloring_scheduling
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering as nr_global


def load_phase1_data(pickle_path):
    
    print(f"\nLoading Phase 1 data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded Phase 1 data:")
    print(f"  CG (global constraints): {len(data['CG'])}")
    print(f"  B_fixed (pruned bias): {len(data['B_fixed'])}")
    print(f"  E+ (positive examples): {len(data['E+'])}")
    print(f"  Initial probabilities: {len(data.get('initial_probabilities', {}))}")

    return data


def display_sudoku_grid(variables, title="Sudoku Grid", debug=False):
    
    print(f"\n{title}")
    print("  " + "-" * 37)

    grid = [[None for _ in range(9)] for _ in range(9)]

    if debug and len(variables) > 0:
        print(f"\n  DEBUG: Checking first 3 variables...")
        for i, var in enumerate(variables[:3]):
            print(f"    Var {i}: name={var.name if hasattr(var, 'name') else 'NO NAME'}, "
                  f"value={var.value() if hasattr(var, 'value') else 'NO VALUE METHOD'}, "
                  f"type={type(var)}")

    for var in variables:
        if hasattr(var, 'name') and 'grid[' in str(var.name):

            try:
                var_name = str(var.name)
                parts = var_name.split('[')[1].split(']')[0].split(',')
                row = int(parts[0])
                col = int(parts[1])

                if callable(getattr(var, 'value', None)):
                    val = var.value()
                elif hasattr(var, '_value'):
                    val = var._value
                else:
                    val = None
                
                if val is not None:
                    grid[row][col] = val
            except Exception as e:
                if debug:
                    print(f"  DEBUG: Error parsing {var.name}: {e}")
                continue

    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("  " + "-" * 37)
        
        row_str = "  |"
        for j in range(9):
            if j > 0 and j % 3 == 0:
                row_str += " |"
            val = grid[i][j]
            if val is None:
                row_str += " . "
            else:
                row_str += f" {val} "
        row_str += "|"
        print(row_str)
    
    print("  " + "-" * 37)

    filled = sum(1 for row in grid for cell in row if cell is not None)
    print(f"  Filled cells: {filled}/81")


def extract_alldifferent_constraints(oracle):
    
    alldiff_constraints = []
    for c in oracle.constraints:
        if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
            alldiff_constraints.append(c)
    return alldiff_constraints


def initialize_probabilities(constraints, prior=0.5):
    
    probabilities = {}
    for c in constraints:
        probabilities[c] = prior
    return probabilities


def update_supporting_evidence(P_c, alpha):
    
    return P_c + (1 - P_c) * (1 - alpha)


def generate_violation_query(CG, C_validated, probabilities, all_variables, oracle=None, B_fixed=None):
    
    import cpmpy as cp
    import time
    
    print(f"  Building COP model: {len(CG)} candidates, {len(C_validated)} validated, {len(all_variables)} variables")

    model = cp.Model()

    # Add all non-AllDifferent constraints from oracle as hard constraints
    if oracle is not None:
        print(f"  Oracle provided: {len(oracle.constraints)} total constraints")
        non_alldiff_constraints = []
        alldiff_count = 0
        for c in oracle.constraints:
            if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
                alldiff_count += 1
            else:
                non_alldiff_constraints.append(c)
        
        print(f"  Oracle breakdown: {alldiff_count} AllDifferent, {len(non_alldiff_constraints)} non-AllDifferent")
        
        # if non_alldiff_constraints:
        #     print(f"  Adding {len(non_alldiff_constraints)} non-AllDifferent constraints as hard constraints")
        #     for c in non_alldiff_constraints:
        #         model += c
        # else:
        #     print(f"  Warning: No non-AllDifferent constraints found in oracle!")
    else:
        print(f"  WARNING: No oracle provided to generate_violation_query!")

    for c in C_validated:
        model += c
    
    # Add relevant fixed-arity bias constraints from Phase 1
    if B_fixed is not None and len(B_fixed) > 0:
        # Get variables involved in the candidate constraints
        decomposed_cg = []
        for c in CG:
            if isinstance(c, AllDifferent):
                decomposed_cg.extend(c.decompose())
            else:
                decomposed_cg.append(c)
        
        S = get_variables(decomposed_cg)
        B_fixed_subset = get_con_subset(B_fixed, S)
        
        if B_fixed_subset:
            print(f"  Adding {len(B_fixed_subset)}/{len(B_fixed)} relevant B_fixed constraints from Phase 1")
            for c in B_fixed_subset:
                model += c
        else:
            print(f"  No relevant B_fixed constraints found for current candidates")

    gamma = {str(c): cp.boolvar(name=f"gamma_{i}") for i, c in enumerate(CG)}

    for c in CG:
        c_str = str(c)
        model += (gamma[c_str] == ~c)
    
    gamma_list = list(gamma.values())
    model += (cp.sum(gamma_list) >= 1)  

    # Objective: minimize sum of probabilities of violated constraints
    # Prefer to violate constraints with low probability (likely incorrect)
    objective = cp.sum([probabilities[c] * gamma[str(c)] for c in CG])

    model.minimize(objective)

    print(f"  Solving COP...")
    solve_start = time.time()

    result = model.solve(time_limit=30)
    solve_time = time.time() - solve_start
    if not result:
        print("UNSAT")
    else:
        violated = []
        for i, c in enumerate(CG):
            gi = gamma[str(c)].value()
            if gi is None:
                print(f"gamma_{i} has no value (solver didn’t assign).")
            elif gi:  
                violated.append((i, c))
        print(f"Violated {len(violated)}/{len(CG)} constraints:")
        for i, c in violated:
            print(f" - gamma_{i} -> VIOLATED: {c}")

    
    if result:
        print(f"  Solved in {solve_time:.2f}s - found violation query")
        Y = get_variables(model.constraints)

        values_set = sum(1 for v in Y if v.value() is not None)
        print(f"  Variables with values: {values_set}/{len(Y)}")
        
        # Map values to original variables if provided
        if all_variables is not None:
            print(f"  Mapping values to {len(all_variables)} original variables")
            # Create a mapping by variable name
            value_map = {}
            for v in Y:
                if hasattr(v, 'name') and v.value() is not None:
                    value_map[str(v.name)] = v.value()
            
            # Set values on original variables
            for orig_var in all_variables:
                if hasattr(orig_var, 'name'):
                    var_name = str(orig_var.name)
                    if var_name in value_map:
                        # Set the value on the original variable
                        if hasattr(orig_var, '_value'):
                            orig_var._value = value_map[var_name]
            
            Y = all_variables

        Viol_e = get_kappa(CG, Y)
        print(f"  Violating {len(Viol_e)}/{len(CG)} constraints")

        
        return Y, Viol_e, "SAT"
    else:
        print(f"  UNSAT after {solve_time:.2f}s - cannot find violation query")
        return None, [], "UNSAT"



def cop_refinement_recursive(CG_cand, C_validated, oracle, probabilities, all_variables,
                             alpha, theta_max, theta_min, max_queries, timeout, 
                             recursion_depth=0, experiment_name="", B_fixed=None):
    """
    Recursive COP-based constraint refinement.
    
    This is the core refinement algorithm that can be called recursively for disambiguation.
    When oracle rejects a query violating multiple constraints, we recursively call this
    on the violated set to determine which constraints are correct.
    
    Algorithm Flow:
    1. Generate violation query that violates subset of CG_cand (minimizing sum of P(c))
    2. Ask oracle about the query
    3. If oracle accepts (TRUE):
         - All violated constraints are INCORRECT (counterexample found)
         - Remove them and continue
    4. If oracle rejects (FALSE):
         - At least one violated constraint is CORRECT
         - If single constraint: must be correct, validate it
         - If multiple constraints: RECURSIVELY call this function on violated set
            to disambiguate which are correct via bisection-like search
            * Recursive call receives filtered C_validated containing only constraints
              whose scope has at least 2 variables in common with violated set (via get_con_subset)
    5. Repeat until budget exhausted or all constraints classified
    
    Args:
        CG_cand: Candidate global constraints to refine
        C_validated: Already validated constraints (used as hard constraints in COP)
        oracle: Oracle for membership queries
        probabilities: Current probability estimates for each constraint
        all_variables: All problem variables
        alpha: Bayesian learning rate (decay factor for refutation)
        theta_max: Acceptance threshold
        theta_min: Rejection threshold
        max_queries: Query budget for this call (including recursive calls)
        timeout: Time budget in seconds
        recursion_depth: Current recursion depth (for logging/indentation)
        experiment_name: Name of experiment (for special handling like sudoku display)
        B_fixed: Fixed-arity bias constraints from Phase 1 (optional)
    
    Returns:
        C_validated: List of validated constraints
        CG_remaining: Set of remaining uncertain constraints  
        probabilities: Updated probability dictionary
        queries_used: Number of queries consumed
    """
    start_time = time.time()
    queries_used = 0
    
    # Make a working copy
    CG = set(CG_cand) if not isinstance(CG_cand, set) else CG_cand.copy()
    C_val = list(C_validated)  # Local validated set
    probs = probabilities.copy()
    
    indent = "  " * recursion_depth
    print(f"\n{indent}{'-'*50}")
    print(f"{indent}COP Refinement [Depth={recursion_depth}]")
    print(f"{indent}{'-'*50}")
    print(f"{indent}Candidates: {len(CG)}, Validated: {len(C_val)}, Budget: {max_queries}q, {timeout}s")
    
    iteration = 0
    consecutive_unsat = 0
    
    while True:
        iteration += 1
        
        # Check termination conditions
        if queries_used >= max_queries:
            print(f"{indent}[STOP] Query budget exhausted ({queries_used}/{max_queries})")
            break
        
        if time.time() - start_time > timeout:
            print(f"{indent}[STOP] Timeout reached")
            break
        
        if not CG:
            print(f"{indent}[STOP] No more candidates")
            break
        
        # Check if all remaining have high confidence
        if len(CG) > 0 and min(probs[c] for c in CG) > theta_max:
            print(f"{indent}[STOP] All remaining P(c) > {theta_max}")
            for c in CG:
                C_val.append(c)
                print(f"{indent}  [ACCEPT] {c} (P={probs[c]:.3f})")
            CG = set()
            break
        
        print(f"\n{indent}[Iter {iteration}] {len(C_val)} validated, {len(CG)} candidates, {queries_used}q used")
        
        # Generate violation query
        print(f"{indent}[QUERY] Generating violation query...")
        Y, Viol_e, status = generate_violation_query(CG, C_val, probs, all_variables, oracle, B_fixed)
        
        if status == "UNSAT":
            consecutive_unsat += 1
            print(f"{indent}[UNSAT] No violation query exists (consecutive: {consecutive_unsat})")
            
            # Accept high-confidence constraints
            for c in list(CG):
                if probs[c] >= 0.7:
                    C_val.append(c)
                    print(f"{indent}  [ACCEPT] {c} (P={probs[c]:.3f})")
            
            CG = {c for c in CG if probs[c] < 0.7}
            
            if not CG or consecutive_unsat >= 2:
                # Final decision on remaining
                for c in list(CG):
                    if probs[c] >= 0.5:
                        C_val.append(c)
                        print(f"{indent}  [FINAL ACCEPT] {c} (P={probs[c]:.3f})")
                    else:
                        print(f"{indent}  [FINAL REJECT] {c} (P={probs[c]:.3f})")
                break
            
            continue
        
        consecutive_unsat = 0
        
        print(f"{indent}Generated query violating {len(Viol_e)} constraints")
        for c in Viol_e:
            print(f"{indent}  - {c} (P={probs[c]:.3f})")
        
        if recursion_depth == 0 and 'sudoku' in experiment_name.lower() and len(all_variables) == 81:
            try:
                display_sudoku_grid(Y, title=f"{indent}Violation Query Assignment", debug=False)
            except Exception as e:
                print(f"{indent}Error displaying grid: {e}")
        
        # Ask oracle
        print(f"{indent}[ORACLE] Asking...")
        answer = oracle.answer_membership_query(Y)
        queries_used += 1
        
        if answer == True:
            # Counterexample: all violated constraints are incorrect
            print(f"{indent}Oracle: YES (valid) - Remove all {len(Viol_e)} violated constraints")
            for c in Viol_e:
                if c in CG:
                    CG.remove(c)
                    probs[c] *= alpha 
                    print(f"{indent}  [REMOVE] {c} (P={probs[c]:.3f})")
        
        else:
            # Oracle: NO (invalid) - At least one violated constraint is correct
            print(f"{indent}Oracle: NO (invalid) - Disambiguate {len(Viol_e)} violated constraints")
            
            if len(Viol_e) == 1:
                # Only one violated: must be correct
                c = list(Viol_e)[0]
                print(f"{indent}  [SINGLE VIOLATION] Must be correct: {c}")
                probs[c] = update_supporting_evidence(probs[c], alpha)
                if c in CG:
                    CG.remove(c)
                    C_val.append(c)
                print(f"{indent}  [VALIDATE] {c} (P={probs[c]:.3f})")
            
            else:
                print(f"{indent}[DISAMBIGUATE] Recursively refining {len(Viol_e)} constraints...")
                
                # S = variables involved in the violated AllDifferent constraints
                decomposed_viol = []
                for c in Viol_e:
                    if isinstance(c, AllDifferent):
                        decomposed_viol.extend(c.decompose())
                    else:
                        decomposed_viol.append(c)
                
                S = get_variables(decomposed_viol)
                print(f"{indent}  Variables in violated constraints: {len(S)}")
                
                C_val_filtered = get_con_subset(C_val, S) if C_val else []
                print(f"{indent}  Relevant validated constraints: {len(C_val_filtered)}/{len(C_val)}")
                
                recursive_budget = min(max_queries - queries_used, max(10, (max_queries - queries_used) // 2))
                recursive_timeout = max(10, (timeout - (time.time() - start_time)) / 2)
                
                print(f"{indent}  Recursive budget: {recursive_budget}q, {recursive_timeout:.0f}s")
                
                C_val_recursive, CG_remaining_recursive, probs_recursive, queries_recursive = \
                    cop_refinement_recursive(
                        CG_cand=list(Viol_e),
                        C_validated=C_val_filtered,  # Pass only relevant validated constraints
                        oracle=oracle,
                        probabilities=probs,
                        all_variables=all_variables,
                        alpha=alpha,
                        theta_max=theta_max,
                        theta_min=theta_min,
                        max_queries=recursive_budget,
                        timeout=recursive_timeout,
                        recursion_depth=recursion_depth + 1,
                        experiment_name=experiment_name,
                        B_fixed=B_fixed
                    )
                
                queries_used += queries_recursive
                print(f"{indent}[DISAMBIGUATE] Recursive call used {queries_recursive}q")
                
                # Update probabilities
                for c in Viol_e:
                    if c in probs_recursive:
                        probs[c] = probs_recursive[c]
                
                # Process results
                ToValidate = [c for c in C_val_recursive if c in Viol_e and c not in C_val]
                ToRemove = [c for c in Viol_e if c not in C_val_recursive and c in CG_remaining_recursive and probs[c] <= theta_min]
                
                print(f"{indent}[DISAMBIGUATE] Results: {len(ToValidate)} validated, {len(ToRemove)} removed")
                
                # Apply results
                for c in ToValidate:
                    if c in CG:
                        CG.remove(c)
                        C_val.append(c)
                    print(f"{indent}  [VALIDATE] {c} (P={probs[c]:.3f})")
                
                for c in ToRemove:
                    if c in CG:
                        CG.remove(c)
                    print(f"{indent}  [REMOVE] {c} (P={probs[c]:.3f})")
    
    duration = time.time() - start_time
    print(f"\n{indent}{'-'*50}")
    print(f"{indent}Refinement [Depth={recursion_depth}] Complete")
    print(f"{indent}Validated: {len(C_val)}, Remaining: {len(CG)}, Queries: {queries_used}, Time: {duration:.2f}s")
    print(f"{indent}{'-'*50}")
    
    return C_val, CG, probs, queries_used


def cop_based_refinement(experiment_name, oracle, candidate_constraints, initial_probabilities,
                         variables, alpha=0.42, theta_max=0.9, theta_min=0.1, 
                         max_queries=500, timeout=600, B_fixed=None):
    """
    Wrapper function for the recursive COP-based refinement.
    
    Implements Algorithm: COP-Based Interactive Refinement with Disambiguation.
    Uses recursive COP refinement for principled disambiguation instead of 
    hard-coded engineering approaches.
    """
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"COP-Based Refinement for {experiment_name}")
    print(f"{'='*60}")
    print(f"Initial candidate constraints: {len(candidate_constraints)}")
    print(f"Parameters: alpha={alpha}, theta_max={theta_max}, theta_min={theta_min}")
    print(f"Budget: {max_queries} queries, {timeout}s timeout\n")
    
    # Call the recursive refinement function
    C_validated, CG_remaining, probabilities_final, queries_used = cop_refinement_recursive(
        CG_cand=candidate_constraints,
        C_validated=[],
        oracle=oracle,
        probabilities=initial_probabilities,
        all_variables=variables,
        alpha=alpha,
        theta_max=theta_max,
        theta_min=theta_min,
        max_queries=max_queries,
        timeout=timeout,
        recursion_depth=0,
        experiment_name=experiment_name,
        B_fixed=B_fixed
    )
    
    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"Refinement Complete")
    print(f"{'='*60}")
    print(f"Validated constraints: {len(C_validated)}")
    print(f"Rejected constraints: {len(candidate_constraints) - len(C_validated)}")
    print(f"Uncertain/remaining: {len(CG_remaining)}")
    print(f"Total queries: {queries_used}")
    print(f"Total time: {total_duration:.2f}s")
    print(f"\nValidated constraints:")
    for c in C_validated:
        print(f"  [OK] {c}")
    
    if CG_remaining:
        print(f"\nRemaining uncertain constraints:")
        for c in CG_remaining:
            print(f"  [?] {c} (P={probabilities_final.get(c, 0.5):.3f})")
    
    stats = {
        'queries': queries_used,
        'time': total_duration,
        'validated': len(C_validated),
        'rejected': len(candidate_constraints) - len(C_validated) - len(CG_remaining),
        'uncertain': len(CG_remaining)
    }
    
    return C_validated, stats


def construct_instance(experiment_name):
    
    if 'graph_coloring_register' in experiment_name.lower() or experiment_name.lower() == 'register':
        result = construct_graph_coloring_register()

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'graph_coloring_scheduling' in experiment_name.lower() or experiment_name.lower() == 'scheduling':
        result = construct_graph_coloring_scheduling()

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'latin_square' in experiment_name.lower() or 'latin' in experiment_name.lower():
        result = construct_latin_square(n=9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'jsudoku' in experiment_name.lower():
        result = construct_jsudoku(grid_size=9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'sudoku_gt' in experiment_name.lower() or 'sudoku_greater' in experiment_name.lower():
        result = construct_sudoku_greater_than(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'sudoku' in experiment_name.lower():
        result = construct_sudoku(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v1' in experiment_name.lower() or 'examtt_variant1' in experiment_name.lower():
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5, 
                                           slots_per_day=6, days_for_exams=10)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v2' in experiment_name.lower() or 'examtt_variant2' in experiment_name.lower():
        result = construct_examtt_variant2(nsemesters=8, courses_per_semester=7, 
                                           slots_per_day=8, days_for_exams=12)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt' in experiment_name.lower():
        result = ces_global(nsemesters=9, courses_per_semester=6, 
                           slots_per_day=9, days_for_exams=14)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'nurse' in experiment_name.lower():
        result = nr_global()
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'uefa' in experiment_name.lower():
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
    
    elif 'vm_allocation' in experiment_name.lower():
        print("Constructing VM Allocation...")
        from benchmarks_global.vm_allocation import construct_vm_allocation as construct_vm_instance
        from vm_allocation_model import PM_DATA, VM_DATA
        
        instance, oracle = construct_vm_instance(PM_DATA, VM_DATA)
    
    else:
        print(f"Unknown experiment: {experiment_name}")
        sys.exit(1)
    
    return instance, oracle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HCAR AllDifferent Phase 2'
    )
    parser.add_argument('--experiment', type=str, default='sudoku',
                       help='Benchmark name (sudoku, examtt, nurse, uefa, vm_allocation)')
    parser.add_argument('--phase1_pickle', type=str, default=None,
                       help='Path to Phase 1 pickle file (optional)')
    parser.add_argument('--alpha', type=float, default=0.42,
                       help='Bayesian learning rate (default: 0.42)')
    parser.add_argument('--theta_max', type=float, default=0.9,
                       help='Acceptance threshold (default: 0.9)')
    parser.add_argument('--theta_min', type=float, default=0.1,
                       help='Rejection threshold (default: 0.1)')
    parser.add_argument('--max_queries', type=int, default=500,
                       help='Maximum total queries (default: 500)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout in seconds (default: 600)')
    parser.add_argument('--prior', type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"HCAR AllDifferent COP Experiment")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Alpha: {args.alpha}")
    print(f"Theta_max: {args.theta_max}")
    print(f"Theta_min: {args.theta_min}")
    print(f"Max queries: {args.max_queries}")
    print(f"Timeout: {args.timeout}s")
    print(f"Prior: {args.prior}")
    print(f"{'='*60}\n")

    instance, oracle = construct_instance(args.experiment)

    oracle.variables_list = cpm_array(instance.X)

    phase1_data = None
    B_fixed = None
    if args.phase1_pickle:

        phase1_data = load_phase1_data(args.phase1_pickle)
        CG = phase1_data['CG']
        
        # Extract B_fixed if available
        if 'B_fixed' in phase1_data:
            B_fixed = phase1_data['B_fixed']
            print(f"\n Loaded B_fixed: {len(B_fixed)} fixed-arity bias constraints")
        else:
            print(f"\n No B_fixed found in Phase 1 data")

        if 'initial_probabilities' in phase1_data:
            probabilities = phase1_data['initial_probabilities']
        else:

            probabilities = initialize_probabilities(CG, prior=args.prior)
            print(f"\n No initial_probabilities in pickle, using uniform prior={args.prior}")
    else:

        CG = extract_alldifferent_constraints(oracle)
        
        # print(f"\nExtracted {len(CG)} AllDifferent constraints from oracle:")
        # for i, c in enumerate(CG, 1):
        #     print(f"  {i}. {c}")

        probabilities = initialize_probabilities(CG, prior=args.prior)
    
    if len(CG) == 0:
        print(f"\n No AllDifferent constraints found")
        sys.exit(0)

    C_validated, stats = cop_based_refinement(
        experiment_name=args.experiment,
        oracle=oracle,
        candidate_constraints=CG,
        initial_probabilities=probabilities,
        variables=instance.X,
        alpha=args.alpha,
        theta_max=args.theta_max,
        theta_min=args.theta_min,
        max_queries=args.max_queries,
        timeout=args.timeout,
        B_fixed=B_fixed
    )

    print(f"\n{'='*60}")
    print(f"Comparison with Target Model")
    print(f"{'='*60}")
    
    target_alldiff = extract_alldifferent_constraints(oracle)
    target_strs = set(str(c) for c in target_alldiff)
    learned_strs = set(str(c) for c in C_validated)
    
    correct = len(learned_strs & target_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)
    
    print(f"Target AllDifferent constraints: {len(target_alldiff)}")
    print(f"Learned AllDifferent constraints: {len(C_validated)}")
    print(f"Correct: {correct}")
    print(f"Missing: {missing}")
    print(f"Spurious: {spurious}")
    
    if correct == len(target_alldiff) and spurious == 0:
        print(f"\n[SUCCESS] Perfect learning!")
    else:
        if missing > 0:
            print(f"\n[ERROR] Missing constraints:")
            for c in target_alldiff:
                if str(c) not in learned_strs:
                    print(f"  - {c}")
        
        if spurious > 0:
            print(f"\n[ERROR] Spurious constraints:")
            for c in C_validated:
                if str(c) not in target_strs:
                    print(f"  - {c}")
    
    print(f"\n{'='*60}")
    print(f"Final Statistics")
    print(f"{'='*60}")
    print(f"Total queries: {stats['queries']}")
    print(f"Total time: {stats['time']:.2f}s")
    print(f"Queries per second: {stats['queries']/stats['time']:.2f}")
    print(f"{'='*60}\n")

    phase2_output = {
        'C_validated': C_validated,  
        'C_validated_strs': [str(c) for c in C_validated],  
        'probabilities': probabilities,  
        'experiment_name': args.experiment,
        'phase2_stats': stats,

        'phase1_data': phase1_data if args.phase1_pickle else None,
        'E_plus': phase1_data['E_plus'] if args.phase1_pickle and 'E_plus' in phase1_data else None,
        'B_fixed': phase1_data['B_fixed'] if args.phase1_pickle and 'B_fixed' in phase1_data else None,
        'all_variables': list(instance.X),
        'metadata': {
            'alpha': args.alpha,
            'theta_max': args.theta_max,
            'theta_min': args.theta_min,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_queries': stats['queries'],
            'total_time': stats['time']
        }
    }

    phase2_output_dir = "phase2_output"
    os.makedirs(phase2_output_dir, exist_ok=True)
    phase2_pickle_path = os.path.join(phase2_output_dir, f"{args.experiment}_phase2.pkl")
    
    with open(phase2_pickle_path, 'wb') as f:
        pickle.dump(phase2_output, f)
    
    print(f"\n Phase 2 outputs saved to: {phase2_pickle_path}")
    print(f"  - Validated constraints: {len(C_validated)}")

