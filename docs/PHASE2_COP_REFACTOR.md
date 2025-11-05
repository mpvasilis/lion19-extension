# Phase 2 COP Model Refactoring

## Overview

Refactored the Phase 2 COP-based constraint refinement to use a **recursive, principled approach** for disambiguation instead of hard-coded engineering with BayesianQuAcq.

## Key Changes

### 1. **Replaced Hard-Coded Disambiguation**

**Before:** The `disambiguate_violated_constraints` function used BayesianQuAcq with a separate learning loop for each violated constraint individually. This was engineering-heavy and didn't follow the same principled approach as the main COP refinement.

**After:** Disambiguation now uses the **same COP-based approach recursively**. When multiple constraints are violated, we recursively call the COP refinement on the violated set (`Viol_e`) to determine which constraints are correct.

### 2. **New Recursive Function: `cop_refinement_recursive`**

This is the core refinement algorithm that can be called recursively:

```python
def cop_refinement_recursive(CG_cand, C_validated, oracle, probabilities, all_variables,
                             alpha, theta_max, theta_min, max_queries, timeout, 
                             recursion_depth=0, experiment_name=""):
    """
    Recursive COP-based constraint refinement.
    
    Returns:
        C_validated: List of validated constraints
        CG_remaining: Set of remaining uncertain constraints  
        probabilities: Updated probability dictionary
        queries_used: Number of queries consumed
    """
```

**Key Features:**
- **Recursion depth tracking** with proper indentation for debugging
- **Budget management** - allocates fraction of remaining budget to recursive calls
- **Timeout handling** - distributes time budget across recursive levels
- **Backtracking** - probabilities and validation decisions bubble up from recursive calls

### 3. **Algorithm Implementation**

Implements the algorithm from the paper:

```
Algorithm: COP-Based Interactive Refinement with Disambiguation

While CG_cand ≠ ∅ and queries < budget:
    1. Generate violation query (y, Viol_e)
    2. If UNSAT: accept high-confidence constraints, break
    3. Ask oracle: answer = O.ASK(y)
    4. If answer = TRUE: 
         - Remove all violated constraints (counterexample)
    5. Else (answer = FALSE):
         - If |Viol_e| = 1: validate single constraint
         - Else: Disambiguate(Viol_e) via RECURSIVE COP refinement
```

### 4. **Disambiguation Flow**

When oracle rejects a query violating multiple constraints:

1. **Decompose violated constraints** to get their scope: `S = get_variables(Viol_e.decompose())`
2. **Filter validated constraints** to only relevant ones: `C_v = get_con_subset(C_validated, S)`
   - Only includes constraints whose scope has ≥2 variables in S
   - Reduces problem size and focuses recursive COP
3. **Recursive COP call** with `Viol_e` as new candidate set and `C_v` as validated set
4. Recursive COP generates queries that violate **subsets** of `Viol_e`
5. Narrows down which constraints are correct through bisection-like search
6. Results (ToValidate, ToRemove, updated probabilities) return to parent
7. Parent applies results and continues main loop

**Why Filter Validated Constraints?**
- **Efficiency**: Recursive COP only needs to respect constraints that interact with variables being disambiguated
- **Focus**: Smaller constraint set makes the COP solver faster
- **Correctness**: Irrelevant constraints don't affect the disambiguation outcome
- **Example**: If disambiguating row constraints in Sudoku, only row/box constraints matter, not column constraints

### 5. **Benefits**

✅ **Principled approach** - Same COP methodology throughout, no mixed strategies  
✅ **Automatic bisection** - COP naturally finds optimal subsets to violate  
✅ **Probability-guided** - Uses probabilities to guide which constraints to test  
✅ **Recursive elegance** - Clean, mathematical formulation  
✅ **Efficient recursion** - Only passes relevant validated constraints to recursive calls  
✅ **Better tracking** - Depth-based indentation shows recursive structure  

### 6. **Removed Dependencies**

Cleaned up imports - no longer need:
- `bayesian_quacq.BayesianQuAcq`
- `bayesian_ca_env.BayesianActiveCAEnv`
- `enhanced_bayesian_pqgen.EnhancedBayesianPQGen`
- `pycona.ProblemInstance`

Added:
- `pycona.utils.get_con_subset` - for filtering relevant constraints

### 7. **Wrapper Function**

`cop_based_refinement` now simply calls the recursive function and formats output:

```python
def cop_based_refinement(experiment_name, oracle, candidate_constraints, 
                         initial_probabilities, variables, ...):
    """Wrapper for recursive COP-based refinement."""
    
    C_validated, CG_remaining, probabilities_final, queries_used = \
        cop_refinement_recursive(...)
    
    return C_validated, stats
```

## Usage

No changes to external API - same command-line interface:

```bash
python main_alldiff_cop.py \
    --experiment sudoku \
    --alpha 0.42 \
    --theta_max 0.9 \
    --theta_min 0.1 \
    --max_queries 500 \
    --timeout 600
```

## Example Output

The recursive structure is now visible in the output:

```
COP Refinement [Depth=0]
──────────────────────────────────────────────────
[Iter 1] 0 validated, 27 candidates, 0q used
[ORACLE] Oracle: NO (invalid) → Disambiguate 5 violated constraints
  [DISAMBIGUATE] Recursively refining 5 constraints...
  
  COP Refinement [Depth=1]
  ──────────────────────────────────────────────────
  [Iter 1] 0 validated, 5 candidates, 0q used
  [ORACLE] Oracle: YES (valid) → Remove 2 violated constraints
  [Iter 2] 0 validated, 3 candidates, 1q used
  ...
  
  Refinement [Depth=1] Complete
  Validated: 3, Remaining: 0, Queries: 8
  
[DISAMBIGUATE] Recursive call used 8q
[DISAMBIGUATE] Results: 3 validated, 0 removed
```

## Testing

All existing tests should pass. The algorithm is more principled and should achieve:
- **Same or better accuracy** (correct/missing/spurious constraints)
- **Potentially fewer queries** (COP is smarter than individual QuAcq runs)
- **Cleaner, more maintainable code**

