# HCAR AllDifferent COP Experiment

A principled constraint acquisition system focusing exclusively on AllDifferent constraints using Constraint Optimization Problem (COP) formulation for query generation.

## Overview

This implementation (`main_alldiff_cop.py`) represents a simplified, principled version of the HCAR methodology that:

1. **Focuses only on AllDifferent constraints** (no Sum/Count)
2. **Uses COP-based query generation** with weighted violation objective
3. **Employs BayesianQuAcq for disambiguation** with automatic oracle interaction
4. **Implements correct Bayesian probability updates**

## Key Features

### 1. Weighted Violation Objective

The system generates queries by solving a Constraint Optimization Problem:

```
minimize: sum(γc) - ε · sum(γc · (1 - P(c)), c ∈ CG)
          ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          primary    tie-breaker (ε = 0.01)

where:
  - γc = 1 if constraint c is violated
  - γc = 0 if constraint c is satisfied
  - P(c) = current probability that c is correct
  
constraints:
  - All validated constraints must be satisfied
  - 1 <= sum(γc) < len(CG) (violate at least one, but not all)
```

**Primary goal:** Minimize total violations (more informative queries)
**Secondary goal:** Among queries with same violation count, prefer violating low-P(c) constraints (suspicious ones)

**Why the minus sign?**
- `(1 - P(c))` is HIGH for low P(c) (suspicious constraints)
- We MINIMIZE `count - ε·weight`, so lower weight = lower objective
- Therefore, COP prefers violating constraints with HIGH `(1 - P(c))`, i.e., LOW P(c)

**Example:**
- Query A: violates 1 constraint (P=0.3) → objective = 1 - 0.01·0.7 = 0.993 ✓ **Selected**
- Query B: violates 1 constraint (P=0.8) → objective = 1 - 0.01·0.2 = 0.998
- **COP selects A** (0.993 < 0.998, suspicious constraint tested)
- Query C: violates 2 constraints → objective ≥ 1.99
- **COP prefers A** (fewer violations is better)

### 2. UNSAT Handling

When the COP cannot generate a violation query (returns `UNSAT`):

```python
# Accept all remaining constraints
for c in CG:
    C_validated.append(c)
break
```

**Why this is safe:**
- With informed priors (0.3 for overfitted, 0.8 for detected), the COP tests suspicious constraints first
- Minimize violation count ensures thorough testing
- By the time UNSAT occurs, overfitted constraints have been rejected
- Remaining constraints are correct or implied by validated set

### 3. Simple Refinement Loop

```python
while budget_remaining:
    # Generate violation query
    e, Viol_e = generate_cop_query(CG, C_validated, probabilities)
    
    # Ask oracle
    answer = oracle.ask(e)
    
    if answer == "No":  # Invalid assignment
        # Supporting evidence - increase probabilities
        for c in Viol_e:
            P(c) = P(c) + (1 - P(c)) * (1 - α)
            if P(c) >= θ_max:
                move c to C_validated
    
    else:  # answer == "Yes" - Valid assignment
        # Refuting evidence - use BayesianQuAcq for disambiguation
        for c in Viol_e:
            learn_individually(c)  # Uses BayesianQuAcq
```

### 3. Bayesian Probability Updates

**Supporting Evidence** (Oracle says "No"):
```
P(c | E) = P(c) + (1 - P(c)) * (1 - α)
```

**Refuting Evidence** (handled by BayesianQuAcq during disambiguation):
- Uses correct Bayesian inference
- Automatically managed by the environment

### 4. No Heuristics

Unlike the original implementation, this version:
- ✅ No positional heuristics (first/middle/last)
- ✅ No manual subset exploration  
- ✅ No hardcoded repair strategies
- ✅ Pure COP-based query generation

## Usage

### Basic Usage

```bash
python main_alldiff_cop.py --experiment sudoku
```

### Parameters

```bash
python main_alldiff_cop.py \
    --experiment sudoku \
    --alpha 0.42 \
    --theta_max 0.9 \
    --theta_min 0.1 \
    --max_queries 500 \
    --timeout 600 \
    --prior 0.5
```

### With Phase 1 Integration (Informed Priors)

```bash
python main_alldiff_cop.py \
    --experiment sudoku \
    --phase1_pickle phase1_output/sudoku_phase1.pkl \
    --max_queries 100
```

When using `--phase1_pickle`, the system automatically loads **informed priors**:
- Detected constraints (pattern-based, scope 9): P = 0.8
- Overfitted constraints (synthetic, scope 3-5): P = 0.3

This makes the COP preferentially test suspicious overfitted constraints!

**Parameters:**
- `--experiment`: Benchmark name (sudoku, examtt, nurse, uefa, vm_allocation)
- `--alpha`: Bayesian learning rate (default: 0.42)
- `--theta_max`: Acceptance threshold (default: 0.9)
- `--theta_min`: Rejection threshold (default: 0.1)
- `--max_queries`: Maximum total queries (default: 500)
- `--timeout`: Timeout in seconds (default: 600)
- `--prior`: Initial probability for all constraints (default: 0.5)

## Implementation Details

### File Structure

```
main_alldiff_cop.py
├── extract_alldifferent_constraints()  # Filter only AllDifferent
├── initialize_probabilities()           # Set initial P(c) = 0.5
├── update_supporting_evidence()         # Bayesian update formula
├── generate_violation_query()           # COP-based query generation
├── disambiguate_violated_constraints()  # Uses BayesianQuAcq
├── cop_based_refinement()               # Main refinement loop
└── main()                               # Entry point
```

### Stopping Conditions

The algorithm stops when:
1. Query budget exhausted
2. Timeout reached
3. No more candidate constraints
4. All remaining constraints have P(c) > θ_max

### Differences from main.py

| Aspect | main.py | main_alldiff_cop.py |
|--------|---------|---------------------|
| Scope | All constraint types | AllDifferent only |
| Query Generation | Heuristic scoring | COP optimization |
| Subset Exploration | Manual tree-based | None (simplified) |
| Repair | Positional heuristics | N/A |
| Complexity | ~1200 lines | ~570 lines |
| Dependencies | ML classifier | Fixed prior |

## Example Output

```
============================================================
HCAR AllDifferent COP Experiment
============================================================
Experiment: sudoku
Alpha: 0.42
Theta_max: 0.9
...

------------------------------------------------------------
Iteration 1
------------------------------------------------------------
Status: 0 validated, 27 candidates, 0 queries used

[QUERY] Generating violation query...
  Building COP model: 27 candidates, 0 validated, 81 variables
  Solving COP (timeout: 30s)...
  Solved in 10.23s - found violation query
  Violating 26/27 constraints
  Variables with values: 81/81
Generated query violating 26 constraints
  - alldifferent(grid[0,5],...) (P=0.500)
  - alldifferent(grid[0,3],...) (P=0.500)
  ...

Violation Query Assignment
  -------------------------------------
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 5  3  4  | 8  2  6  | 9  1  7 |
  | 7  1  1  | 1  1  1  | 1  1  1 |
  -------------------------------------
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 6  1  1  | 9  1  1  | 1  1  2 |
  -------------------------------------
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 1  1  1  | 1  1  1  | 1  1  2 |
  | 1  1  1  | 1  1  1  | 1  1  2 |
  -------------------------------------
  Filled cells: 81/81

[ORACLE] Asking oracle...
[NO] Oracle: No (invalid assignment)
[SUPPORT] Supporting 26 constraints
  [UPDATE] alldifferent(...): P=0.500 -> 0.790
  ...

============================================================
Refinement Complete
============================================================
Validated constraints: 27
Total queries: 45
Total time: 287.32s
```

## Theoretical Foundation

This implementation follows the HCAR methodology:

1. **Phase 1** (Implicit): All AllDifferent constraints start with P(c) = 0.5
2. **Phase 2** (COP Refinement): Iteratively validate/reject constraints
3. **Phase 3** (Not implemented): Would use MQuAcq for remaining constraints

### Bayesian Updates

The probability updates follow Bayes' theorem:

```
P(c ∈ C_T | E) = P(E | c ∈ C_T) * P(c ∈ C_T) / P(E)

where:
  - P(E | c ∈ C_T) = 1 - α (if E supports c)
  - P(E | c ∉ C_T) = α
```

This gives the simplified update formula used in the code.

## Sudoku Visualization

For Sudoku problems, the system automatically displays each violation query as a formatted 9x9 grid with box separators:

```
Violation Query Assignment
  -------------------------------------
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 5  3  4  | 8  2  6  | 9  1  7 |
  | 7  1  1  | 1  1  1  | 1  1  1 |
  -------------------------------------
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 6  1  1  | 9  1  1  | 1  1  2 |
  -------------------------------------
  | 1  1  1  | 1  1  1  | 1  1  1 |
  | 1  1  1  | 1  1  1  | 1  1  2 |
  | 1  1  1  | 1  1  1  | 1  1  2 |
  -------------------------------------
  Filled cells: 81/81
```

This helps visualize which constraints are violated and understand the COP behavior.

## Performance

**Sudoku (9x9, 27 AllDifferent constraints):**
- COP solve time: ~10s per iteration (first iteration), ~1s later iterations
- Can learn all 27 constraints with sufficient budget
- No spurious constraints learned
- Visual grid display for all queries

## Limitations

1. **Only AllDifferent**: Doesn't handle Sum/Count constraints
2. **COP Timeout**: May not find queries if constraints conflict
3. **No Noise Handling**: Assumes perfect oracle (can be extended)
4. **No Phase 3**: Doesn't learn remaining binary constraints

## Future Extensions

1. Add Sum/Count constraint support
2. Implement Phase 3 (MQuAcq2 integration)
3. Add noise-robust disambiguation
4. Optimize COP solver settings
5. Parallel constraint testing

## References

- HCAR Methodology Specification (see CLAUDE.md)
- BayesianQuAcq implementation (bayesian_quacq.py)
- Enhanced PQGen (enhanced_bayesian_pqgen.py)

## License

Same as parent project.


