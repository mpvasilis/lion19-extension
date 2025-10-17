<!-- 547a8cf2-e657-4fef-8388-bac087a8474f 3163349e-7008-4143-999a-5dfcaae4df61 -->
# HCAR AllDifferent COP Experiment Implementation

## Overview

Create `main_alldiff_cop.py` - a principled HCAR implementation focusing on AllDifferent constraints with COP-based query generation, PyCona-based disambiguation, and Bayesian updates.

## Key Differences from main.py

1. **Scope**: Only AllDifferent constraints (no Sum/Count)
2. **Query Generation**: COP formulation with weighted violation objective
3. **Objective**: Minimize `sum(γc · (1 - P(c)), c ∈ CG)` where γc = 1 if constraint c is violated
4. **Hard constraint**: `1 <= sum(γc) < len(CG)` (violate at least one, but not all)
5. **Disambiguation**: When oracle says "Yes", use `get_kappa` to find violated constraints, then use PyCona to generate one-by-one isolation queries
6. **Initial probabilities**: Fixed prior 0.5 (no ML classifier)
7. **Bayesian updates**: Correct Bayesian formula for supporting evidence

## Implementation Steps

### 1. File Structure and Imports

- Copy essential imports from `main.py`
- Import `get_kappa` from `pycona.utils`
- Import COP solver components from CPMpy
- Remove unnecessary imports (clustering, subset exploration, etc.)

### 2. Core Function: `cop_based_refinement()`

Create the main refinement loop with signature:

```python
def cop_based_refinement(
    experiment_name,
    oracle,
    candidate_constraints,  # CG - only AllDifferent
    initial_probabilities,  # P(c) for each c in CG
    variables,
    alpha=0.42,
    theta_max=0.9,
    theta_min=0.1,
    max_queries=500,
    timeout=600
)
```

### 3. PyCona-Based Query Generation

Implement `generate_violation_query()` using PyCona (properly integrated):

```python
def generate_violation_query(CG, C_validated, probabilities, all_variables):
    """
    Generate violation query using PyCona's EnhancedBayesianPQGen.
    
    Strategy:
    - Create ProblemInstance with bias=CG, init_cl=C_validated
    - Use EnhancedBayesianPQGen with weighted objective
    - After generation, use get_kappa to find violated constraints
    
    Returns: (Y_vars, Viol_e, status) where Y_vars are variables with values set
    """
    from pycona import ProblemInstance
    from cpmpy import cpm_array
    from enhanced_bayesian_pqgen import EnhancedBayesianPQGen
    from pycona.utils import get_kappa
    
    # Create instance with all variables from CG and C_validated
    instance = ProblemInstance(
        variables=cpm_array(all_variables),
        init_cl=list(C_validated),
        bias=list(CG),
        name="violation_query_generation"
    )
    
    # Create query generator
    qgen = EnhancedBayesianPQGen()
    
    # Create minimal environment for qgen
    class QueryEnv:
        def __init__(self, inst, probs):
            self.instance = inst
            self.constraint_probs = probs.copy()
    
    qgen.env = QueryEnv(instance, probabilities)
    
    # Generate query - returns Y (list of variables with values)
    Y = qgen.generate()
    
    if Y and len(Y) > 0:
        # get_kappa returns constraints that are violated
        # Note: Y has variables with .value() set by the solver
        Viol_e = get_kappa(CG, Y)
        
        return Y, Viol_e, "SAT"
    else:
        return None, [], "UNSAT"
```

### 4. Disambiguation Phase Using BayesianQuAcq

Implement `disambiguate_violated_constraints()` - uses BayesianQuAcq to automatically handle oracle queries and probability updates:

```python
def disambiguate_violated_constraints(Viol_e, C_validated, oracle, probabilities, all_variables, alpha, theta_max, theta_min, max_queries_per_constraint=10):
    """
    For each c in Viol_e, use BayesianQuAcq to learn if it's correct.
    
    Strategy:
    - For each c_target in Viol_e:
        - Create ProblemInstance with bias=[c_target], init_cl=C_validated + other Viol_e constraints
        - Run BayesianQuAcq.learn() which handles:
            * Query generation via PyCona
            * Oracle interaction
            * Bayesian probability updates
        - Check result: c_target in CL → keep, c_target removed from bias → delete
    
    Returns: updated_probabilities, constraints_to_remove
    """
    from pycona import ProblemInstance
    from cpmpy import cpm_array
    from bayesian_quacq import BayesianQuAcq
    from bayesian_ca_env import BayesianActiveCAEnv
    from enhanced_bayesian_pqgen import EnhancedBayesianPQGen
    from cpmpy.transformations.get_variables import get_variables
    
    updated_probs = probabilities.copy()
    to_remove = []
    
    for c_target in Viol_e:
        print(f"\n  Disambiguating constraint: {c_target}")
        print(f"  Current P(c) = {probabilities[c_target]:.3f}")
        
        # Build init_cl: validated + other violated constraints (not c_target)
        init_cl = list(C_validated)
        for c in Viol_e:
            if c != c_target:
                init_cl.append(c)
        
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
    
    return updated_probs, to_remove
```

### 6. Bayesian Update Functions

Implement update formulas:

```python
def update_supporting_evidence(P_c, alpha):
    """When oracle says No (e is invalid) - constraints are supported."""
    return P_c + (1 - P_c) * (1 - alpha)

def update_definitive_refutation(P_c, alpha):
    """When isolated query is Yes - constraint definitively false."""
    return P_c * alpha

def update_ambiguous_refutation(P_c, alpha, prior):
    """When cannot isolate or isolation query is No."""
    update_factor = alpha + (1 - alpha) * prior
    return P_c * update_factor
```

### 7. Main Refinement Loop

Implement the core loop in `cop_based_refinement()`:

```python
while True:
    # Stopping conditions
    if queries_used >= max_queries:
        break
    if time.time() - start_time > timeout:
        break
    if not CG:  # No more candidates
        break
    if min(probabilities.values()) > theta_max:
        print("All constraints confident!")
        break
    
    # Generate violation query
    e, Viol_e, status = generate_violation_query(CG, C_validated, probabilities, variables)
    
    if status == "UNSAT":
        print("No feasible query - stopping")
        break
    
    # Ask oracle
    answer = oracle.ask(e)
    queries_used += 1
    
    if answer:  # Yes - e is valid
        print(f"Oracle: Yes (valid) - {len(Viol_e)} constraints violated")
        
        # Disambiguation phase
        probabilities, to_remove = disambiguate_violated_constraints(
            Viol_e, C_validated, oracle, probabilities, alpha
        )
        
        # Remove constraints with P(c) <= theta_min
        for c in to_remove:
            CG.remove(c)
            print(f"Removed: {c}")
    
    else:  # No - e is invalid
        print(f"Oracle: No (invalid) - supporting {len(Viol_e)} constraints")
        
        # Update probabilities (supporting evidence)
        for c in Viol_e:
            probabilities[c] = update_supporting_evidence(probabilities[c], alpha)
            
            # Accept if P(c) >= theta_max
            if probabilities[c] >= theta_max:
                C_validated.append(c)
                CG.remove(c)
                print(f"Accepted: {c} (P={probabilities[c]:.3f})")
```

### 8. Extract Only AllDifferent Constraints

Implement `extract_alldifferent_constraints()`:

```python
def extract_alldifferent_constraints(oracle):
    """Extract only AllDifferent constraints from oracle."""
    alldiff_constraints = []
    for c in oracle.constraints:
        if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
            alldiff_constraints.append(c)
    return alldiff_constraints
```

### 9. Initialize Probabilities with ML

Implement `initialize_probabilities()`:

```python
def initialize_probabilities(constraints, variables, use_ml=True):
    """Initialize P(c) for each constraint using XGBoost or default."""
    probabilities = {}
    
    if use_ml:
        try:
            import joblib
            xgb_clf = joblib.load("constraint_classifier_xgb.joblib")
            # ... feature extraction logic from main.py
        except:
            use_ml = False
    
    for c in constraints:
        if use_ml:
            # Extract features and predict
            prob = predict_probability(c, variables, xgb_clf)
        else:
            prob = 0.5  # Default prior
        
        probabilities[c] = prob
    
    return probabilities
```

### 10. Main Entry Point

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='sudoku')
    parser.add_argument('--alpha', type=float, default=0.42)
    parser.add_argument('--theta_max', type=float, default=0.9)
    parser.add_argument('--theta_min', type=float, default=0.1)
    parser.add_argument('--max_queries', type=int, default=500)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    
    # Load benchmark
    instance, oracle = construct_instance(args.experiment)
    
    # Extract only AllDifferent constraints
    CG = extract_alldifferent_constraints(oracle)
    
    # Initialize probabilities
    probabilities = initialize_probabilities(CG, instance.X)
    
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
    
    # Report results
    print(f"\n=== Results ===")
    print(f"Validated constraints: {len(C_validated)}")
    print(f"Total queries: {stats['queries']}")
    print(f"Time: {stats['time']:.2f}s")
```

## Key Design Decisions

1. **No subset exploration**: Simplifies the implementation significantly
2. **Direct oracle integration**: Call `oracle.ask(assignment)` directly instead of through PyCona wrapper
3. **Pure COP formulation**: The violation query IS the COP optimization, very clean
4. **get_kappa for verification**: After getting assignment values, use `get_kappa` to verify which constraints are actually violated
5. **Principled Bayesian updates**: Three distinct update rules based on evidence type

## Files to Reference

- `main.py`: Lines 205-237 (ML probability initialization)
- `enhanced_bayesian_pqgen.py`: Lines 8-60 (weighted objective concept)
- `bayesian_quacq.py`: Lines 138-151 (get_kappa usage)
- `utils.py`: Lines 942-944 (get_kappa implementation)

### To-dos

- [ ] Create hcar_alldiff_cop_experiment.py with imports and file header
- [ ] Implement generate_violation_query() with weighted COP objective
- [ ] Implement generate_isolation_query() for disambiguation
- [ ] Implement disambiguate_violated_constraints() with get_kappa integration
- [ ] Implement Bayesian update functions (supporting, definitive, ambiguous)
- [ ] Implement cop_based_refinement() main loop with stopping conditions
- [ ] Implement extract_alldifferent_constraints() to filter only AllDifferent
- [ ] Implement initialize_probabilities() with XGBoost integration
- [ ] Implement main entry point with argument parsing and benchmark loading
- [ ] Test on Sudoku benchmark and verify results