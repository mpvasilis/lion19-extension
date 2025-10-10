# HCAR Methodology Compliance Analysis

## Issues Found

### 🔴 CRITICAL: Bayesian Update Formula is WRONG

**Location:** `hcar_advanced.py:648-657`

**Specification (CLAUDE.md:231):**
```python
def UpdateBayesian(P_c, alpha):
    P_new = P_c + (1 - P_c) * (1 - alpha)
    # WHERE alpha = 0.1
```

**Current Implementation:**
```python
likelihood = 1.0 - alpha
new_prob = (likelihood * current_prob) / (
    likelihood * current_prob + alpha * (1 - current_prob)
)
new_prob = current_prob + 0.7 * (new_prob - current_prob)  # Extra unexplained dampening
```

**Problem:**
1. Uses full Bayesian formula instead of simplified update
2. Adds mysterious 0.7 dampening factor not in specification
3. Makes update dependent on complex likelihood ratio

**Impact:**
- Confidence updates will be **incorrect**
- May converge too slowly or too quickly
- Makes results non-reproducible vs paper

**Fix:**
```python
def update_confidence(...):
    if response == OracleResponse.INVALID:
        # Specification formula
        new_prob = current_prob + (1 - current_prob) * (1 - alpha)
    else:  # VALID
        new_prob = 0.0  # Hard refutation (correct)
    return np.clip(new_prob, 0.0, 1.0)
```

---

## ✅ What's CORRECT

### Phase 1: Passive Candidate Generation
- ✅ **CONSTRAINT 1 (Independence)**: B_fixed pruned only with E+, not B_globals (line 1209-1218)
- ✅ Extracts AllDifferent, Sum, Count patterns
- ✅ Groups variables by naming patterns
- ✅ Checks consistency across all examples

### Phase 2: Interactive Refinement
- ✅ **CONSTRAINT 2 (Ground Truth Pruning)**: `_prune_fixed_bias_with_solution` only uses oracle-verified queries (line 1431-1468)
- ✅ **CONSTRAINT 4 (Hard Refutation)**: Sets P(c) = 0 on Valid response (line 1361)
- ✅ **CONSTRAINT 5 (Data-Driven Subsets)**: Now uses culprit scores (after our fix)
- ✅ Uncertainty-based candidate selection (line 1415-1429)
- ✅ Budget inheritance for child constraints (line 1408)
- ✅ UNSAT handling (sets confidence to theta_max)
- ✅ TIMEOUT handling (slight confidence boost)

### Principled Pruning
- ✅ Stores confirmed solutions (line 822, 1364)
- ✅ Uses confirmed solutions to prune B_fixed (line 1365)
- ✅ Never prunes using unverified B_globals

---

## ⚠️ POTENTIAL ISSUES (Need Investigation)

### 1. Budget Allocation Algorithm

**Specification (CLAUDE.md:243):**
```
ALLOCATION: Budget(c) ∝ Entropy(P(c))
EFFECT: Spend queries where they provide most information gain
```

**Current Implementation (need to check):**
Where is entropy-based budget allocation? Need to verify `_allocate_uncertainty_budget()`

### 2. ML Prior Features

**Specification (CLAUDE.md:176-189):**
```python
FEATURES MUST INCLUDE:
- Constraint type (AllDifferent, Sum, Count)
- Arity (scope size)
- Dimensional structure (row/column/block patterns)
- Variable naming patterns (x_i_j suggests grid)
- Participation in other constraints
```

**Need to check:** Does current feature extraction include all of these?

### 3. Query Generator Completeness

**CONSTRAINT 3 (CLAUDE.md:44-49):**
```
RULE: For any false constraint c, the query generator MUST eventually find
a violating valid solution (if one exists) or prove none exists (UNSAT)
```

**Need to check:**
- Does query generator properly construct auxiliary CSP?
- Does it handle all constraint types?
- Does timeout get handled correctly?

### 4. Phase 3 MQuAcq Integration

**Specification (CLAUDE.md:258-266):**
```
1. Treat C'_G as part of the learned model (no further validation needed)
2. Generate queries to resolve status of all candidates in B'_fixed
3. Learn set C_L of fixed-arity constraints
```

**Need to check:**
- Are validated globals passed to MQuAcq correctly?
- Is refined B_fixed used as bias?
- Does MQuAcq count queries correctly?

---

## Summary

### Must Fix Immediately
1. **Bayesian Update Formula** - Critical bug affecting convergence

### Should Investigate
2. Budget allocation - verify entropy-based allocation
3. ML features - verify all required features present
4. Query generator completeness
5. Phase 3 integration correctness

### Already Correct
- All 5 methodological constraints (after subset explorer fix)
- Principled information flow
- Phase 1 bias independence
- Phase 2 refinement loop structure
