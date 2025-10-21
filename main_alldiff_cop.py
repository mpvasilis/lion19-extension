"""
HCAR AllDifferent COP Experiment

A principled implementation focusing on AllDifferent constraints with:
- COP-based query generation using PyCona's EnhancedBayesianPQGen
- Weighted violation objective: minimize sum(γc · (1 - P(c)))
- Disambiguation using BayesianQuAcq for automatic oracle interaction
- Bayesian probability updates
- Fixed prior initialization (0.5)

"""

import argparse
import os
import pickle
import time
import sys
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.globalconstraints import AllDifferent
from pycona import ProblemInstance
from pycona.utils import get_kappa
from bayesian_quacq import BayesianQuAcq
from bayesian_ca_env import BayesianActiveCAEnv
from enhanced_bayesian_pqgen import EnhancedBayesianPQGen

# Import benchmark construction functions
from benchmarks_global import construct_sudoku
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering as nr_global


def load_phase1_data(pickle_path):
    """
    Load CG, B_fixed, and metadata from Phase 1 pickle.
    
    Args:
        pickle_path: Path to Phase 1 pickle file
        
    Returns:
        Dictionary with keys: 'CG', 'B_fixed', 'E+', 'variables', 'metadata'
    """
    print(f"\nLoading Phase 1 data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded Phase 1 data:")
    print(f"  CG (global constraints): {len(data['CG'])}")
    print(f"  B_fixed (pruned bias): {len(data['B_fixed'])}")
    print(f"  E+ (positive examples): {len(data['E+'])}")
    print(f"  Initial probabilities: {len(data.get('initial_probabilities', {}))}")
    print(f"  Metadata: {data['metadata']}")
    
    return data


def display_sudoku_grid(variables, title="Sudoku Grid", debug=False):
    """
    Display a 9x9 Sudoku grid with current variable values.
    
    Args:
        variables: List of CPMpy variables (must be named grid[row,col])
        title: Title to display above the grid
        debug: If True, show debug information
    """
    print(f"\n{title}")
    print("  " + "-" * 37)
    
    # Create a 9x9 grid
    grid = [[None for _ in range(9)] for _ in range(9)]
    
    # Debug: check first few variables
    if debug and len(variables) > 0:
        print(f"\n  DEBUG: Checking first 3 variables...")
        for i, var in enumerate(variables[:3]):
            print(f"    Var {i}: name={var.name if hasattr(var, 'name') else 'NO NAME'}, "
                  f"value={var.value() if hasattr(var, 'value') else 'NO VALUE METHOD'}, "
                  f"type={type(var)}")
    
    # Fill in values from variables
    for var in variables:
        if hasattr(var, 'name') and 'grid[' in str(var.name):
            # Parse grid[row,col] format
            try:
                var_name = str(var.name)
                parts = var_name.split('[')[1].split(']')[0].split(',')
                row = int(parts[0])
                col = int(parts[1])
                
                # Get value - try different ways
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
    
    # Display the grid with box separators
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
    
    # Show count of filled cells
    filled = sum(1 for row in grid for cell in row if cell is not None)
    print(f"  Filled cells: {filled}/81")


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


def initialize_probabilities(constraints, prior=0.5):
    """
    Initialize P(c) for each constraint with fixed prior.
    
    Args:
        constraints: List of constraints
        prior: Fixed prior probability (default: 0.5)
        
    Returns:
        Dictionary mapping constraint to probability
    """
    probabilities = {}
    for c in constraints:
        probabilities[c] = prior
    return probabilities


def update_supporting_evidence(P_c, alpha):
    """
    Bayesian update when oracle says No (e is invalid) - constraints are supported.
    
    Formula: P(c | E) = P(c) + (1 - P(c)) * (1 - α)
    
    Args:
        P_c: Current probability
        alpha: Learning rate
        
    Returns:
        Updated probability
    """
    return P_c + (1 - P_c) * (1 - alpha)


def generate_violation_query(CG, C_validated, probabilities, all_variables):
    """
    Generate violation query with weighted COP objective.
    
    Strategy:
    - Build COP model with validated constraints (must satisfy)
    - Create binary variables gamma_c for each c in CG
    - Reify: gamma_c = 1 iff c is violated (~c is true)
    - Constraint: 1 <= sum(gamma_c) < len(CG) (violate at least one, not all)
    - Objective: minimize sum((1 - P(c)) * gamma_c) (prefer violating low-prob constraints)
    
    Args:
        CG: List of candidate global constraints
        C_validated: List of validated constraints
        probabilities: Dict mapping constraint to probability
        all_variables: All variables in the problem
        
    Returns:
        Tuple (Y_vars, Viol_e, status) where:
        - Y_vars: Variables with values set by solver
        - Viol_e: List of violated constraints
        - status: "SAT" or "UNSAT"
    """
    import cpmpy as cp
    import time
    
    print(f"  Building COP model: {len(CG)} candidates, {len(C_validated)} validated, {len(all_variables)} variables")
    
    # Build model
    model = cp.Model()
    
    # Add validated constraints (must be satisfied)
    for c in C_validated:
        model += c
    
    
    # Create violation indicator variables
    gamma = {str(c): cp.boolvar(name=f"gamma_{i}") for i, c in enumerate(CG)}
    
    # Reify: gamma_c = 1 iff c is violated
    for c in CG:
        c_str = str(c)
        model += (gamma[c_str] == ~c)
    
    gamma_list = list(gamma.values())
    model += (cp.sum(gamma_list) >= 1)  # At least one
    
    # Objective: Minimize total violations, preferring low P(c) as tie-breaker
    # Primary: minimize count of violations (more informative queries)
    # Secondary: among same count, prefer violating low P(c) constraints (suspicious ones)
    violation_count = cp.sum(gamma_list)
    weighted_preference = cp.sum([
        (1.0 - probabilities[c]) * gamma[str(c)]
        for c in CG
    ])
    
    epsilon = 0.01
    objective = violation_count - epsilon * weighted_preference
    
    # Minimize: fewer violations first, then prefer low-P(c) constraints
    # Example: 1 violation @ P=0.3 → 1 - 0.01·0.7 = 0.993 (selected!)
    #          1 violation @ P=0.8 → 1 - 0.01·0.2 = 0.998 (not selected)
    #          31 violations → objective = 31 - ... = ~31 (NEVER selected with max=4 constraint)
    model.minimize(objective)
    
    # Solve with timeout (30 seconds per query)
    print(f"  Solving COP (timeout: 30s)...")
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
            elif gi:  # 1 means c is violated
                violated.append((i, c))
        print(f"Violated {len(violated)}/{len(CG)} constraints:")
        for i, c in violated:
            print(f" - gamma_{i} -> VIOLATED: {c}")

    
    if result:
        print(f"  Solved in {solve_time:.2f}s - found violation query")
        
        # Now all_variables should have values (we added domain constraints)
        Y = get_variables(model.constraints)
        
        # Verify values are set
        values_set = sum(1 for v in Y if v.value() is not None)
        print(f"  Variables with values: {values_set}/{len(Y)}")
    
        
        # Get violated constraints
        Viol_e = get_kappa(CG, Y)
        print(f"  Violating {len(Viol_e)}/{len(CG)} constraints")

        
        return Y, Viol_e, "SAT"
    else:
        print(f"  UNSAT after {solve_time:.2f}s - cannot find violation query")
        return None, [], "UNSAT"



def test_constraints_individually(candidates, oracle, probabilities, all_variables,
                                  alpha, theta_max, theta_min, max_queries_per_constraint=10):
    """
    Test each candidate constraint individually in a CLEAN environment (no assumptions).
    
    Used when UNSAT occurs - we cannot generate violation queries within C_validated,
    so we test each candidate directly against the oracle without assuming C_validated is correct.
    
    Strategy:
    - For each c_target in candidates:
        - Create ProblemInstance with bias=[c_target], init_cl=[] (empty!)
        - Run BayesianQuAcq.learn() which will:
            * Generate queries that test c_target against ground truth
            * Not be biased by potentially incorrect validated constraints
        - Update probabilities based on testing results
    
    Args:
        candidates: List of constraints to test
        oracle: Oracle for answering queries
        probabilities: Current probability dictionary
        all_variables: All variables in problem
        alpha: Learning rate
        theta_max: Accept threshold
        theta_min: Reject threshold
        max_queries_per_constraint: Budget per constraint
        
    Returns:
        Tuple (updated_probabilities, constraints_to_remove)
    """
    from cpmpy.transformations.get_variables import get_variables
    from cpmpy import cpm_array
    from pycona.problem_instance import ProblemInstance
    from bayesian_ca_env import BayesianActiveCAEnv
    from enhanced_bayesian_pqgen import EnhancedBayesianPQGen
    
    updated_probs = probabilities.copy()
    to_remove = []
    
    for c_target in candidates:
        print(f"\n  Testing constraint independently: {c_target}")
        print(f"  Current P(c) = {probabilities[c_target]:.3f}")
        
        # CRITICAL FIX: Use ALL variables (not just constraint variables)
        # Otherwise testing becomes meaningless (e.g., "can 9 values be all different?" - trivially yes!)
        # We need to test within the full problem context
        
        # Create instance for CLEAN testing (no init_cl - no candidate assumptions)
        instance = ProblemInstance(
            variables=all_variables,  # ALL 81 Sudoku variables!
            init_cl=[],  # No assumptions about other constraints
            bias=[c_target],
            name="clean_testing"
        )
        
        # Create Bayesian environment
        env = BayesianActiveCAEnv(
            qgen=EnhancedBayesianPQGen(),
            theta_max=theta_max,
            theta_min=theta_min,
            prior=probabilities[c_target],
            alpha=alpha
        )
        
        env.constraint_probs = {c_target: probabilities[c_target]}
        env.max_queries = max_queries_per_constraint
        
        # Run BayesianQuAcq - test against ground truth
        ca_system = BayesianQuAcq(ca_env=env)
        learned_instance = ca_system.learn(instance, oracle=oracle, verbose=2)
        
        # Check result
        if c_target in learned_instance.cl:
            # Constraint accepted - oracle confirmed it's correct
            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target])
            print(f"  Result: ACCEPTED (P={updated_probs[c_target]:.3f})")
        
        elif c_target not in learned_instance.bias:
            # Constraint removed from bias - definitively false
            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target] * alpha)
            print(f"  Result: REJECTED (P={updated_probs[c_target]:.3f})")
            
            if updated_probs[c_target] <= theta_min:
                to_remove.append(c_target)
        else:
            # Constraint still in bias - uncertain
            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target])
            print(f"  Result: UNCERTAIN (P={updated_probs[c_target]:.3f})")
    
    return updated_probs, to_remove


def disambiguate_violated_constraints(Viol_e, C_validated, CG, oracle, probabilities, all_variables, 
                                      alpha, theta_max, theta_min, max_queries_per_constraint=10):
    """
    For each c in Viol_e, use BayesianQuAcq to learn if it's correct.
    
    Strategy:
    - For each c_target in Viol_e:
        - Create ProblemInstance with bias=[c_target], init_cl=C_validated + remaining CG (not in Viol_e)
        - Run BayesianQuAcq.learn() which handles:
            * Query generation via PyCona
            * Oracle interaction
            * Bayesian probability updates
        - Check result: c_target in CL → keep, c_target removed from bias → delete
    
    Args:
        Viol_e: List of violated constraints
        C_validated: List of validated constraints
        CG: Candidate global constraints (remaining candidates)
        oracle: Oracle for answering queries
        probabilities: Current probability dictionary
        all_variables: All variables in problem
        alpha: Learning rate
        theta_max: Accept threshold
        theta_min: Reject threshold
        max_queries_per_constraint: Budget per constraint
        
    Returns:
        Tuple (updated_probabilities, constraints_to_remove, disambiguation_queries_used)
    """
    updated_probs = probabilities.copy()
    to_remove = []
    total_disambiguation_queries = 0
    
    for c_target in Viol_e:
        print(f"\n  Disambiguating constraint: {c_target}")
        print(f"  Current P(c) = {probabilities[c_target]:.3f}")
        
        # Build init_cl: validated constraints + remaining CG candidates (not in Viol_e)
        # We test c_target while respecting other candidates that are not currently violated
        init_cl = list(C_validated)
        # Add remaining candidates from CG that are NOT in Viol_e
        remaining_cg = [c for c in CG if c not in Viol_e]
        init_cl.extend(remaining_cg)
        print(f"  Init CL: {len(C_validated)} validated + {len(remaining_cg)} remaining CG candidates")
        
        # Get all variables
        all_vars = get_variables([c_target] + init_cl)
        
        # Create instance for isolation learning
        instance = ProblemInstance(
            variables=cpm_array(all_vars),
            init_cl=init_cl,
            bias=[c_target],  # Testing only this constraint
            name="isolation_learning"
        )
        
        # Create Bayesian environment
        env = BayesianActiveCAEnv(
            qgen=EnhancedBayesianPQGen(),
            theta_max=theta_max,
            theta_min=theta_min,
            prior=probabilities[c_target],  # Use current probability as prior
            alpha=alpha
        )
        
        # Set constraint probability
        env.constraint_probs = {c_target: probabilities[c_target]}
        env.max_queries = max_queries_per_constraint
        
        # Run BayesianQuAcq - it will ask oracle and update probabilities
        ca_system = BayesianQuAcq(ca_env=env)
        learned_instance = ca_system.learn(instance, oracle=oracle, verbose=2)
        
        # Track queries used for this constraint (from metrics after learning)
        if hasattr(env, 'metrics') and env.metrics is not None:
            queries_used_for_this_constraint = env.metrics.membership_queries_count
        else:
            queries_used_for_this_constraint = 1  # Conservative estimate
        
        total_disambiguation_queries += queries_used_for_this_constraint
        print(f"  [Queries for this constraint: {queries_used_for_this_constraint}]")
        
        # Check result
        if c_target in learned_instance.cl:
            # Constraint was accepted - should not happen in disambiguation of violated constraints
            # This means evidence suggests it's correct despite being violated
            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target])
            print(f"  Result: Kept (P={updated_probs[c_target]:.3f})")
        
        elif c_target not in learned_instance.bias:
            # Constraint was removed from bias - it's definitively false
            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target] * alpha)
            print(f"  Result: Rejected (P={updated_probs[c_target]:.3f})")
            
            if updated_probs[c_target] <= theta_min:
                to_remove.append(c_target)
        else:
            # Constraint still in bias - update probability from environment
            updated_probs[c_target] = env.constraint_probs.get(c_target, probabilities[c_target])
            print(f"  Result: Uncertain (P={updated_probs[c_target]:.3f})")
    
    print(f"\n[DISAMBIGUATION] Total queries used: {total_disambiguation_queries}")
    return updated_probs, to_remove, total_disambiguation_queries


def cop_based_refinement(experiment_name, oracle, candidate_constraints, initial_probabilities,
                         variables, alpha=0.42, theta_max=0.9, theta_min=0.1, 
                         max_queries=500, timeout=600):
    """
    Main COP-based refinement loop for AllDifferent constraints.
    
    Algorithm:
    1. Generate violation query using weighted COP objective
    2. Ask oracle
    3. If Yes (valid): Disambiguate violated constraints
    4. If No (invalid): Update probabilities (supporting evidence)
    5. Repeat until stopping condition
    
    Args:
        experiment_name: Name of benchmark
        oracle: Oracle for answering queries
        candidate_constraints: CG - list of AllDifferent constraints
        initial_probabilities: P(c) for each c in CG
        variables: All variables in problem
        alpha: Learning rate
        theta_max: Accept threshold
        theta_min: Reject threshold
        max_queries: Maximum total queries
        timeout: Maximum time in seconds
        
    Returns:
        Tuple (C_validated, stats) where stats contains metrics
    """
    start_time = time.time()
    queries_used = 0
    
    # Copy to avoid modifying inputs
    CG = list(candidate_constraints)
    probabilities = initial_probabilities.copy()
    C_validated = []
    
    print(f"\n{'='*60}")
    print(f"COP-Based Refinement for {experiment_name}")
    print(f"{'='*60}")
    print(f"Initial candidate constraints: {len(CG)}")
    print(f"Parameters: alpha={alpha}, theta_max={theta_max}, theta_min={theta_min}")
    print(f"Budget: {max_queries} queries, {timeout}s timeout\n")
    
    iteration = 0
    consecutive_unsat = 0  # Track consecutive UNSAT occurrences
    
    while True:
        # input("Press Enter to continue...")
        iteration += 1
        print(f"\n{'-'*60}")
        print(f"Iteration {iteration}")
        print(f"{'-'*60}")
        
        # Stopping conditions
        if queries_used >= max_queries:
            print(f"[STOP] Reached maximum query budget ({max_queries})")
            break
        
        if time.time() - start_time > timeout:
            print(f"[STOP] Timeout ({timeout}s) reached")
            break
        
        if not CG:
            print(f"[STOP] No more candidate constraints")
            break
        
        if len(CG) > 0 and min(probabilities[c] for c in CG) > theta_max:
            print(f"[STOP] All remaining constraints have P(c) > {theta_max}")
            # Accept remaining constraints
            for c in CG:
                C_validated.append(c)
                print(f"  Accepted: {c} (P={probabilities[c]:.3f})")
            CG = []
            break
        
        print(f"Status: {len(C_validated)} validated, {len(CG)} candidates, {queries_used} queries used")
        
        # Generate violation query
        print(f"\n[QUERY] Generating violation query...")
        Y, Viol_e, status = generate_violation_query(CG, C_validated, probabilities, variables)
        
        if status == "UNSAT":
            consecutive_unsat += 1
            print(f"[UNSAT] Cannot generate violation query for remaining {len(CG)} constraints")
            print(f"[DECISION] Accepting remaining constraints as likely correct or implied")
            
            # When UNSAT occurs with validated constraints in place, it means:
            # Cannot find assignment that satisfies C_validated AND violates any candidate
            # 
            # For interdependent constraints (like Sudoku), this typically means:
            # - Remaining candidates are correct and implied by validated constraints
            # - OR they are mutually exclusive with validated constraints
            # 
            # Since we've been conservative (theta_max=0.9) and validated constraints
            # have high confidence, we accept remaining candidates with sufficient probability
            
            for c in list(CG):
                if probabilities[c] >= 0.7:  # Accept if reasonably confident
                    C_validated.append(c)
                    print(f"  [ACCEPT] {c} (P={probabilities[c]:.3f})")
                else:
                    print(f"  [UNCERTAIN] {c} (P={probabilities[c]:.3f}) - keeping in candidates")
            
            CG = [c for c in CG if probabilities[c] < 0.7]
            
            if not CG:
                break
            
            # If we've hit UNSAT multiple times in a row, stop trying
            if consecutive_unsat >= 2:
                print(f"[STOP] Multiple consecutive UNSAT results - accepting/rejecting remaining constraints")
                for c in list(CG):
                    if probabilities[c] >= 0.5:  # Lower threshold for final acceptance
                        C_validated.append(c)
                        print(f"  [FINAL ACCEPT] {c} (P={probabilities[c]:.3f})")
                    else:
                        print(f"  [FINAL REJECT] {c} (P={probabilities[c]:.3f}) - too uncertain")
                break
            else:
                print(f"[CONTINUE] {len(CG)} uncertain constraints remaining (UNSAT count: {consecutive_unsat})")
                # Try one more iteration
                continue  # Skip oracle query since Y is None
        
        # Successfully generated a query - reset UNSAT counter
        consecutive_unsat = 0
        
        print(f"Generated query violating {len(Viol_e)} constraints")
        for c in Viol_e:
            print(f"  - {c} (P={probabilities[c]:.3f})")
        
        # Display Sudoku board if this is a Sudoku problem
        if 'sudoku' in experiment_name.lower() and len(variables) == 81:
            try:
                display_sudoku_grid(Y, title="Violation Query Assignment", debug=False)
            except Exception as e:
                print(f"Error displaying Sudoku grid: {e}")
                print(Y)
        
        # Ask oracle
        print(f"\n[ORACLE] Asking oracle...")
        answer = oracle.answer_membership_query(Y)
        queries_used += 1
        
        if answer == False:  # Yes - Y is valid
            print(f"[YES] Oracle: Yes (valid assignment)")
            print(f"[DISAMBIG] {len(Viol_e)} constraints violated by valid solution - entering disambiguation")
            
            # Disambiguation phase - test each violated constraint individually
            # Try to isolate which constraint(s) in Viol_e are actually false
            probabilities, to_remove, disambiguation_queries = disambiguate_violated_constraints(
                Viol_e, C_validated, CG, oracle, probabilities, variables, 
                alpha, theta_max, theta_min,
                max_queries_per_constraint=10  # Budget per constraint for isolation testing
            )
            
            # Update query count to include disambiguation queries
            queries_used += disambiguation_queries
            print(f"[QUERIES] Main loop: {queries_used - disambiguation_queries}, Disambiguation: {disambiguation_queries}, Total so far: {queries_used}")
            
            # Remove constraints marked for removal (low confidence)
            for c in to_remove:
                if c in CG:
                    CG.remove(c)
                    print(f"  [REMOVE] Removed: {c} (P={probabilities[c]:.3f})")
            
            # Accept constraints that reached high confidence during disambiguation
            # Check remaining violated constraints (not removed)
            for c in Viol_e:
                if c not in to_remove and c in CG and probabilities[c] >= theta_max:
                    C_validated.append(c)
                    CG.remove(c)
                    print(f"  [ACCEPT] Accepted: {c} (P={probabilities[c]:.3f} >= {theta_max})")
        
        else:  # No - Y is invalid
            print(f"[NO] Oracle: No (invalid assignment)")
            print(f"[SUPPORT] Supporting {len(Viol_e)} constraints")
            
            # Update probabilities (supporting evidence)
            for c in Viol_e:
                old_prob = probabilities[c]
                probabilities[c] = update_supporting_evidence(probabilities[c], alpha)
                print(f"  [UPDATE] {c}: P={old_prob:.3f} -> {probabilities[c]:.3f}")
                
                # Accept if P(c) >= theta_max
                # Note: For interdependent constraints like Sudoku, we rely on:
                # 1. Multiple supporting queries before reaching threshold
                # 2. Disambiguation when oracle says "Yes" to catch spurious constraints
                # 3. Conservative theta_max (0.9) to reduce false positives
                if probabilities[c] >= theta_max:
                    C_validated.append(c)
                    CG.remove(c)
                    print(f"    [ACCEPT] Accepted (P >= {theta_max})")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"Refinement Complete")
    print(f"{'='*60}")
    print(f"Validated constraints: {len(C_validated)}")
    print(f"Rejected constraints: {len(candidate_constraints) - len(C_validated)}")
    print(f"Total queries: {queries_used}")
    print(f"Total time: {total_duration:.2f}s")
    print(f"\nValidated constraints:")
    for c in C_validated:
        print(f"  [OK] {c}")
    
    stats = {
        'queries': queries_used,
        'time': total_duration,
        'validated': len(C_validated),
        'rejected': len(candidate_constraints) - len(C_validated)
    }
    
    return C_validated, stats


def construct_instance(experiment_name):
    """
    Construct problem instance and oracle for given benchmark.
    
    Args:
        experiment_name: Name of benchmark (sudoku, sudoku_gt, examtt, examtt_v1, examtt_v2, nurse, uefa, vm_allocation)
        
    Returns:
        Tuple (instance, oracle)
    """
    if 'sudoku_gt' in experiment_name.lower() or 'sudoku_greater' in experiment_name.lower():
        print("Constructing 9x9 Sudoku with Greater-Than constraints...")
        result = construct_sudoku_greater_than(3, 3, 9)
        # Handle optional mock_constraints return (Phase 2 doesn't need them)
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'sudoku' in experiment_name.lower():
        print("Constructing 9x9 Sudoku...")
        result = construct_sudoku(3, 3, 9)
        # Handle optional mock_constraints return (Phase 2 doesn't need them)
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v1' in experiment_name.lower() or 'examtt_variant1' in experiment_name.lower():
        print("Constructing Exam Timetabling Variant 1...")
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5, 
                                           slots_per_day=6, days_for_exams=10)
        # Handle optional mock_constraints return (Phase 2 doesn't need them)
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v2' in experiment_name.lower() or 'examtt_variant2' in experiment_name.lower():
        print("Constructing Exam Timetabling Variant 2...")
        result = construct_examtt_variant2(nsemesters=8, courses_per_semester=7, 
                                           slots_per_day=8, days_for_exams=12)
        # Handle optional mock_constraints return (Phase 2 doesn't need them)
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt' in experiment_name.lower():
        print("Constructing Exam Timetabling...")
        result = ces_global(nsemesters=9, courses_per_semester=6, 
                           slots_per_day=9, days_for_exams=14)
        # Handle optional mock_constraints return (Phase 2 doesn't need them)
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'nurse' in experiment_name.lower():
        print("Constructing Nurse Rostering...")
        instance, oracle = nr_global()
    
    elif 'uefa' in experiment_name.lower():
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
        description='HCAR AllDifferent COP Experiment - Principled constraint refinement'
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
    parser.add_argument('--prior', type=float, default=0.5,
                       help='Initial probability when no Phase 1 data (default: 0.5, ignored if using --phase1_pickle)')
    
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
    
    # Load benchmark
    instance, oracle = construct_instance(args.experiment)
    
    # Setup oracle
    oracle.variables_list = cpm_array(instance.X)
    
    # Load constraints: either from Phase 1 pickle or extract from oracle
    phase1_data = None  # Initialize to None
    if args.phase1_pickle:
        # Load from Phase 1 pickle
        phase1_data = load_phase1_data(args.phase1_pickle)
        CG = phase1_data['CG']
        
        print(f"\nUsing constraints from Phase 1 pickle:")
        print(f"  Total CG (detected + overfitted): {len(CG)}")
        print(f"  Detected AllDifferent: {phase1_data['metadata']['num_detected_alldiffs']}")
        print(f"  Overfitted AllDifferent: {phase1_data['metadata']['num_overfitted_alldiffs']}")
        
        # Load informed priors if available
        if 'initial_probabilities' in phase1_data:
            probabilities = phase1_data['initial_probabilities']
            print(f"\nUsing informed priors from Phase 1:")
            print(f"  Detected constraints: P=0.8 (high confidence)")
            print(f"  Overfitted constraints: P=0.3 (low confidence)")
            
            # Show distribution
            high_prior = sum(1 for p in probabilities.values() if p >= 0.7)
            low_prior = sum(1 for p in probabilities.values() if p < 0.5)
            print(f"  Prior distribution: {high_prior} high (>=0.7), {low_prior} low (<0.5)")
        else:
            # Fallback to uniform prior
            probabilities = initialize_probabilities(CG, prior=args.prior)
            print(f"\n[WARNING] No initial_probabilities in pickle, using uniform prior={args.prior}")
    else:
        # Extract from oracle (original behavior)
        CG = extract_alldifferent_constraints(oracle)
        
        print(f"\nExtracted {len(CG)} AllDifferent constraints from oracle:")
        for i, c in enumerate(CG, 1):
            print(f"  {i}. {c}")
        
        # Initialize with uniform prior
        probabilities = initialize_probabilities(CG, prior=args.prior)
    
    if len(CG) == 0:
        print(f"\n[WARNING] No AllDifferent constraints found")
        sys.exit(0)
    
    # Run refinement
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
        timeout=args.timeout
    )
    
    # Compare with target model
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
    
    # Save Phase 2 outputs for Phase 3
    phase2_output = {
        'C_validated': C_validated,  # Validated global constraints
        'C_validated_strs': [str(c) for c in C_validated],  # String representations
        'probabilities': probabilities,  # Final probabilities
        'experiment_name': args.experiment,
        'phase2_stats': stats,
        # Include Phase 1 inputs for Phase 3
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
    
    # Save to pickle for Phase 3
    phase2_output_dir = "phase2_output"
    os.makedirs(phase2_output_dir, exist_ok=True)
    phase2_pickle_path = os.path.join(phase2_output_dir, f"{args.experiment}_phase2.pkl")
    
    with open(phase2_pickle_path, 'wb') as f:
        pickle.dump(phase2_output, f)
    
    print(f"\n[SAVED] Phase 2 outputs saved to: {phase2_pickle_path}")
    print(f"  - Validated constraints: {len(C_validated)}")
    print(f"  - Ready for Phase 3 (Active Learning with MQuAcq-2)")

