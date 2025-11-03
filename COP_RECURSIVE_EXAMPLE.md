# Recursive COP Disambiguation - Example Walkthrough

## Scenario: Sudoku with 27 AllDifferent Constraints

Let's walk through how the recursive COP approach disambiguates violated constraints.

### Initial State

```
Candidate Constraints (CG): [c1, c2, c3, ..., c27]  (27 AllDifferent constraints)
Validated (CV): []
Probabilities: All P(c) = 0.5
```

---

## Iteration 1 (Depth=0)

### Step 1: Generate Violation Query

COP minimizes: `Σ P(ci) * γi` where `γi = 1` if constraint `ci` is violated

Result: Finds assignment **y1** that violates `[c3, c7, c12, c15, c20]` (5 constraints)

```
COP selects these 5 because:
- They have relatively low probabilities
- Violating them allows a valid assignment to other constraints
```

### Step 2: Ask Oracle

```
Oracle.ASK(y1) → FALSE (invalid assignment)
```

**Interpretation:** At least one of `[c3, c7, c12, c15, c20]` is CORRECT (caused rejection).

### Step 3: Disambiguate via Recursion

**Recursive Call:**
```
cop_refinement_recursive(
    CG_cand = [c3, c7, c12, c15, c20],  ← Only the violated set!
    C_validated = [],
    probabilities = {c3: 0.5, c7: 0.5, ...},
    max_queries = 250,  ← Half of remaining budget
    recursion_depth = 1  ← Increase depth
)
```

---

## Iteration 1.1 (Depth=1) - Inside Recursive Call

### Step 1: Generate Violation Query

Now COP works only with `[c3, c7, c12, c15, c20]`:

Result: Finds assignment **y2** that violates `[c7, c15]` (2 constraints)

```
COP strategy: Try to narrow down by violating subset
```

### Step 2: Ask Oracle

```
Oracle.ASK(y2) → TRUE (valid assignment!)
```

**Interpretation:** Both `c7` and `c15` are INCORRECT (they were violated, but oracle accepted).

### Step 3: Remove Incorrect Constraints

```
Remove c7, c15 from candidates
Update: P(c7) *= 0.42, P(c15) *= 0.42
Remaining in recursive call: [c3, c12, c20]
```

---

## Iteration 1.2 (Depth=1) - Continue Recursive Call

### Step 1: Generate Violation Query

Now COP works with `[c3, c12, c20]`:

Result: Finds assignment **y3** that violates `[c3, c12]` (2 constraints)

### Step 2: Ask Oracle

```
Oracle.ASK(y3) → FALSE (invalid assignment)
```

**Interpretation:** At least one of `[c3, c12]` is CORRECT.

But we could go deeper... or continue iterating at this level.

---

## Iteration 1.3 (Depth=1)

### Step 1: Generate Violation Query

COP works with `[c3, c12, c20]`, tries different subset:

Result: Finds assignment **y4** that violates only `[c3]` (1 constraint)

### Step 2: Ask Oracle

```
Oracle.ASK(y4) → FALSE (invalid assignment)
```

**Interpretation:** Since only `c3` is violated and oracle rejected, `c3` MUST be CORRECT.

### Step 3: Validate Constraint

```
Validate c3 → Move to C_validated
Update: P(c3) = 0.5 + (1 - 0.5) * (1 - 0.42) = 0.79
Remaining: [c12, c20]
```

---

## Iteration 1.4 (Depth=1)

### Step 1: Generate Violation Query

COP works with `[c12, c20]`:

Result: Finds assignment **y5** that violates `[c12]` (1 constraint)

### Step 2: Ask Oracle

```
Oracle.ASK(y5) → FALSE (invalid)
```

Validate `c12`.

---

## Iteration 1.5 (Depth=1)

Only `c20` remains. Similar process validates `c20`.

---

## Return from Recursion (Back to Depth=0)

Recursive call returns:

```python
C_validated_recursive = [c3, c12, c20]
CG_remaining_recursive = []
probabilities_updated = {
    c3: 0.79,
    c7: 0.21,   # Rejected
    c12: 0.79,
    c15: 0.21,  # Rejected
    c20: 0.79
}
queries_used = 5
```

### Apply Results

```
ToValidate = [c3, c12, c20]  ← Add to main C_validated
ToRemove = [c7, c15]          ← Remove from main CG
```

Main loop state:
```
CG = [c1, c2, c4, c5, c6, c8, c9, c10, c11, c13, c14, c16, ..., c27]  (22 remaining)
C_validated = [c3, c12, c20]  (3 validated)
Queries used = 1 (main) + 5 (recursive) = 6
```

---

## Continue Main Loop (Depth=0)

Now the main loop continues with:
- 22 candidates in CG (removed c7, c15, and validated c3, c12, c20)
- 3 validated constraints
- Updated probabilities

Next iteration generates a new violation query from the remaining 22 candidates...

---

## Key Insights

### 1. **Bisection-Like Search**
The COP naturally performs a bisection-like search by selecting subsets of violated constraints to test.

### 2. **Probability-Guided**
Lower probability constraints are preferentially violated, focusing effort where uncertainty is highest.

### 3. **Automatic Subset Selection**
Unlike manual engineering, the COP solver automatically determines optimal subsets to violate based on:
- Probabilities
- Satisfiability of remaining constraints
- Objective minimization

### 4. **Recursive Structure**
Each level of recursion focuses on a smaller set of constraints, making the problem more tractable.

### 5. **Backtracking with Information**
When recursion returns, updated probabilities and validation decisions propagate up, informing the parent level.

---

## Comparison with Old Approach

### Old (BayesianQuAcq per constraint):
```
For each c in Viol_e:
    Run separate QuAcq(bias=[c], ...)
    Isolate learning for individual constraint
    Many queries per constraint
```

**Problems:**
- Each constraint learned in isolation
- No leveraging of relationships between constraints
- Fixed query strategy
- Hard-coded engineering

### New (Recursive COP):
```
COP_Refine(Viol_e):
    Find optimal subset to violate
    Ask oracle
    If multiple violated and rejected:
        Recursively COP_Refine(new_violated_subset)
    Update probabilities
    Return results
```

**Benefits:**
- Leverages constraint relationships
- Adaptive query strategy (COP decides optimal subsets)
- Principled, mathematical approach
- Same methodology throughout (no mixed strategies)
- Potentially fewer queries via intelligent subset selection

---

## Visualization

```
Depth 0: CG = [c1, c2, c3, ..., c27] (27 constraints)
         │
         ├─ Query violates [c3, c7, c12, c15, c20]
         ├─ Oracle: FALSE → Must disambiguate
         │
         └─► Depth 1: CG = [c3, c7, c12, c15, c20] (5 constraints)
                      │
                      ├─ Query violates [c7, c15]
                      ├─ Oracle: TRUE → Remove both
                      │
                      ├─ Query violates [c3]
                      ├─ Oracle: FALSE → Validate c3
                      │
                      ├─ Query violates [c12]
                      ├─ Oracle: FALSE → Validate c12
                      │
                      ├─ Query violates [c20]
                      ├─ Oracle: FALSE → Validate c20
                      │
                      └─► Return: Validated=[c3,c12,c20], Removed=[c7,c15]
         
         Continue with CG = [c1, c2, c4, ..., c27] \ {c3,c7,c12,c15,c20}
```

This recursive structure naturally handles arbitrarily complex disambiguation scenarios!

