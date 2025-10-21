# The Cascading Validation Bug - Root Cause Analysis

## The Problem

The oracle correctly says "No" to a **VALID Sudoku**, causing a spurious constraint to accumulate supporting evidence and get validated.

### Evidence

**Query Assignment (Verified VALID):**
```
  | 9  4  7  | 8  1  3  | 2  5  6 |
  | 6  5  3  | 4  2  9  | 8  7  1 |
  | 1  8  2  | 7  6  5  | 4  3  9 |
  -------------------------------------
  | 8  7  9  | 5  4  1  | 6  2  3 |
  | 5  6  1  | 3  8  2  | 7  9  4 |
  | 3  2  4  | 6  9  7  | 1  8  5 |
  -------------------------------------
  | 2  1  5  | 9  7  4  | 3  6  8 |
  | 7  3  6  | 1  5  8  | 9  4  2 |
  | 4  9  8  | 2  3  6  | 5  1  7 |
  -------------------------------------
```

**Verification:**
- ✅ All 9 rows have 1-9 unique
- ✅ All 9 columns have 1-9 unique
- ✅ All 9 boxes have 1-9 unique
- **Result: This IS a valid Sudoku!**

**Oracle Response:** "No" (invalid) ❌

**Consequence:** 
- Constraint `alldifferent(grid[6,3],grid[0,7],grid[7,6],grid[8,2])` gets supporting evidence
- P(c) increases: 0.877 → 0.948
- Constraint gets ACCEPTED (P ≥ 0.9)
- **But this constraint is SPURIOUS!**

## Root Cause Analysis

### Why Oracle Said "No" to Valid Sudoku

The oracle checks if the assignment satisfies ALL target constraints (27 correct Sudoku constraints).

If the oracle said "No", it means the assignment violates at least one target constraint.

**BUT WE VERIFIED THE SUDOKU IS VALID!**

Therefore, one of two things must be true:
1. **Bug in Oracle:** The oracle is checking wrong constraints
2. **Bug in COP Model:** The COP model is contaminating the solution

### Investigation: What are the 2 Validated Constraints?

From the log:
```
Building COP model: 29 candidates, 2 validated, 81 variables
```

The COP model includes these 2 validated constraints (line 212 in `generate_violation_query`):
```python
for c in C_validated:
    model += c  # Force solution to satisfy validated constraints
```

**If either of these 2 validated constraints is SPURIOUS**, then:
1. COP forces solution to satisfy the spurious constraint
2. Solution violates TRUE Sudoku constraints
3. Oracle correctly says "No"
4. Spurious candidate gets wrongly supported

### The Cascading Effect

```
Iteration 1:
  ├─ Test constraint A (spurious)
  ├─ COP generates query (no validated constraints yet)
  ├─ Query happens to satisfy A by chance
  ├─ Oracle says "No" (for other reasons)
  └─ A gets supporting evidence → P(A) increases

Iteration 2:
  ├─ A reaches theta_max → VALIDATED ✓ (wrong!)
  └─ C_validated = [A]

Iteration 3:
  ├─ Test constraint B (spurious)
  ├─ COP generates query that SATISFIES A (forced!)
  ├─ Solution violates true constraints (to satisfy A)
  ├─ Oracle says "No" (correct - solution is invalid)
  └─ B gets supporting evidence → P(B) increases (wrong!)

Iteration 4:
  ├─ B reaches theta_max → VALIDATED ✓ (wrong!)
  └─ C_validated = [A, B]  # Both spurious!

... (cascade continues)
```

## The Fundamental Issue

**Supporting Evidence is Ambiguous**

When oracle says "No" to a query that violates constraint C:
- **Correct Interpretation:** "This assignment violates constraint C AND constraint C is indeed incorrect"
- **Wrong Interpretation (Current):** "This assignment is invalid THEREFORE constraint C must be correct"

The current system uses the WRONG interpretation!

### Why This Happens

The oracle doesn't tell us **WHY** the assignment is invalid. It could be invalid because:
1. It violates the candidate constraint being tested (supports the constraint)
2. It violates a DIFFERENT constraint (doesn't support the candidate)
3. It violates a spurious VALIDATED constraint (contaminates future queries)

## The Solution

### Approach 1: Test Constraints in Isolation Earlier

Don't accumulate validated constraints until they've been thoroughly tested:

```python
# CURRENT (Broken):
if P(c) >= theta_max:
    C_validated.append(c)  # Immediate validation

# FIXED:
if P(c) >= theta_max:
    # Test in isolation before validating
    if test_constraint_in_clean_environment(c):
        C_validated.append(c)
```

### Approach 2: Don't Use C_validated in COP Model

Generate queries WITHOUT assuming C_validated is correct:

```python
# CURRENT (Creates cascade):
for c in C_validated:
    model += c  # Assumes all validated constraints are correct

# OPTION 1 (Conservative):
# Don't add any C_validated to the model
# Let queries be completely independent

# OPTION 2 (Smart):
# Only add constraints with very high confidence (P > 0.99)
for c in C_validated:
    if probabilities[c] > 0.99:
        model += c
```

### Approach 3: Validate Only After All Testing

Don't validate constraints during Phase 2 - just rank them:

```python
# Phase 2: Rank constraints by probability
# Phase 3: Validate top-ranked constraints through clean testing
# Phase 4: Use validated constraints in final model
```

## Recommended Fix

**Combine Approach 1 + Approach 2:**

1. **Higher Validation Threshold:**
   ```python
   theta_max = 0.95  # Instead of 0.9
   ```

2. **Clean Environment Testing Before Validation:**
   ```python
   if probabilities[c] >= theta_max:
       # Test constraint independently before validating
       result = test_constraints_individually([c], oracle, ...)
       if result shows high confidence:
           C_validated.append(c)
   ```

3. **Conservative COP Model:**
   ```python
   # Only use VERY high confidence validated constraints in COP
   for c in C_validated:
       if probabilities[c] >= 0.98:  # Extra conservative
           model += c
   ```

## Expected Impact

| Approach | Pros | Cons |
|----------|------|------|
| Higher theta_max | Simple, reduces false validations | Might reject some true constraints |
| Clean testing before validation | Catches spurious early | More queries needed |
| Conservative COP | Prevents cascade | Slower convergence |
| **Combined** | **Best of all** | **Some query overhead** |

## Implementation Priority

1. **Immediate (Critical):** Add clean environment testing before validation
2. **Short-term:** Increase theta_max to 0.95
3. **Medium-term:** Make COP model more conservative
4. **Long-term:** Redesign Phase 2 to separate ranking from validation

## Testing Plan

1. Run Sudoku with fix
2. Check: No spurious constraints validated early
3. Verify: S-Rec = 100%
4. Measure: Query count increase (acceptable if model is correct)
5. Test other benchmarks to ensure no regression

