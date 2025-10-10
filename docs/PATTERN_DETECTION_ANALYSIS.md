# Pattern Detection Analysis - Nurse Rostering

## Summary

Detailed logging reveals **why constraints are missing** and **why S-Recall = 0%**.

## Ground Truth (21 constraints)

**AllDifferent constraints (13 total):**
- 7 per-day constraints: `AllDifferent(day_X)` for X ∈ [0,6]
- 6 consecutive-day constraints: `AllDifferent(day_X_last_shift, day_X+1_first_shift)`

**Count constraints (8 total):**
- `Count(all_vars, nurse_id) <= 6` for nurse_id ∈ [1,8]
- Ensures each nurse works at most 6 days per week

## What Was Detected (12 candidates)

### AllDifferent: 7 candidates (PERFECT MATCH)
```
row_0, row_1, row_2, row_3, row_4, row_5, row_6
```
These correctly correspond to the 7 per-day constraints.

**Status:** All 7 detected correctly

### Count: 5 candidates (PARTIAL - 5 out of 8)
```
Count(all_vars, 2) <= 6
Count(all_vars, 4) == 6  ← WRONG! Should be <= 6
Count(all_vars, 5) <= 6
Count(all_vars, 6) <= 5  ← WRONG! Should be <= 6
Count(all_vars, 7) <= 6
```

**Detected:** Nurses 2, 4, 5, 6, 7 (5 out of 8)
**Missing:** Nurses 1, 3, 8 (3 out of 8)

**Problems:**
1. **Nurse 4:** Detected as `== 6` instead of `<= 6` (over-fitted)
2. **Nurse 6:** Detected as `<= 5` instead of `<= 6` (under-fitted)
3. **Nurses 1, 3, 8:** Not detected at all

### Missing: 6 consecutive-day constraints
```
AllDifferent(day_0_last_shift, day_1_first_shift)
AllDifferent(day_1_last_shift, day_2_first_shift)
...
AllDifferent(day_5_last_shift, day_6_first_shift)
```

**Status:** NONE detected (0 out of 6)

**Why:** These span across day boundaries with specific shift positions. The pattern grouping logic only detects:
- Row patterns (all variables with same first index)
- Column patterns (all variables with same second index)
- Block patterns (for Sudoku-like structures)

It does NOT detect cross-row patterns with specific positions like:
`[day_X, shift_2, pos_0], [day_X, shift_2, pos_1], [day_X+1, shift_0, pos_0], [day_X+1, shift_0, pos_1]`

## Root Causes

### Issue 1: Consecutive-Day Constraints Not Detected

**Pattern grouping (`_group_variables_by_pattern`):**
```python
numbers = re.findall(r'\d+', var_name)  # Extracts [day, shift, position]
if len(numbers) >= 2:
    row, col = int(numbers[0]), int(numbers[1])  # Uses only first 2 indices
```

**Problem:** Only groups by (day, shift), not by custom patterns.

**Consecutive-day constraints require:**
- Variables from TWO different days
- Specific shifts (last shift of day X, first shift of day X+1)
- Specific positions (both positions for each shift)

**Current logic cannot detect this pattern.**

### Issue 2: Count Constraints Partially Detected

**Why missing nurses 1, 3, 8?**

Looking at the counts in examples:
```
Value 1: counts = [6, 4, 4, 3, 3] (min=3, max=6)
  REJECTED: Count varies (3-6) but max not frequent

Value 3: counts = [6, 5, 5, 5, 5] (min=5, max=6)
  REJECTED: Count varies (5-6) but max not frequent

Value 8: counts = [4, 5, 5, 5, 6] (min=4, max=6)
  REJECTED: Count varies (4-6) but max not frequent
```

**Detection heuristic:**
```python
if count_values.count(max_count) >= 2:  # Max must appear in >= 2 examples
    accept Count(...) <= max_count
```

**Problem:**
- Nurse 1: max=6 appears only once → rejected
- Nurse 3: max=6 appears only once → rejected
- Nurse 8: max=6 appears only once → rejected

**This heuristic is too strict!** The bound `<= 6` is correct, but because different examples have different counts (natural variation), the pattern detector rejects them.

### Issue 3: Incorrect Bounds for Detected Counts

**Nurse 4:**
```
counts = [6, 6, 6, 6, 6] (min=6, max=6)
Detected: Count(all_vars, 4) == 6
Correct:  Count(all_vars, 4) <= 6
```

**Problem:** Pattern detector assumes if all examples have the same count, it must be `==` (equality).

**Why wrong:** The 5 examples happened to all assign nurse 4 exactly 6 times. But the constraint allows nurse 4 to work FEWER than 6 days. This is **over-fitting**.

**Nurse 6:**
```
counts = [4, 4, 4, 5, 5] (min=4, max=5)
Detected: Count(all_vars, 6) <= 5
Correct:  Count(all_vars, 6) <= 6
```

**Problem:** Pattern detector uses `max(counts)` as the upper bound.

**Why wrong:** The 5 examples happened to never assign nurse 6 more than 5 times. But the constraint allows up to 6. This is **under-fitting** (learned model is MORE restrictive than target).

## Why S-Recall = 0%

**Learned model constraints:**
- 7 AllDifferent (days 0-6) ✓ Correct
- 5 Count constraints with WRONG bounds ✗
- 0 consecutive-day AllDifferent ✗
- 171 fixed binary constraints (from Phase 3)

**Target model constraints:**
- 7 AllDifferent (days 0-6) ✓
- 6 AllDifferent (consecutive days) ✓
- 8 Count constraints (all correct) ✓

**Incompatibility:**

1. **Missing consecutive-day constraints:** Learned model allows solutions where the same nurse works last shift of day X and first shift of day X+1, which target model forbids.

2. **Wrong Count bounds:**
   - `Count(nurse_4) == 6` forces nurse 4 to work ALL 6 allowed days
   - `Count(nurse_6) <= 5` forbids nurse 6 from working 6 days
   - Missing nurses 1, 3, 8 means they're unconstrained

3. **Incorrect fixed constraints:** Phase 3 learned 171 binary constraints to "compensate" for missing globals, but these are likely:
   - Over-fitted to the specific 5 examples
   - Conflicting with target model
   - Rejecting all valid target solutions

**Result:** The learned model is **solution-incompatible** with the target model.

- **S-Precision = 100%:** Learned solutions happen to satisfy target (luck or very restrictive constraints)
- **S-Recall = 0%:** Learned model rejects ALL target solutions

## Learned Model is UNSAT

From the experiment output:
```
WARNING - Could not generate learned solutions - learned model is UNSAT (over-constrained)
```

**This confirms:** The 171 fixed constraints create CONFLICTS, making the entire learned model unsatisfiable.

**When model is UNSAT:**
- `_sample_solutions(learned_model)` returns empty list
- S-Precision = 100% (vacuously true - no solutions to be wrong)
- S-Recall = 0% (rejects all target solutions)

## Recommendations

### Fix 1: Detect Consecutive-Day Patterns

Add specialized pattern detection for cross-boundary constraints:

```python
def _detect_consecutive_patterns(variables: List[str]) -> List[List[str]]:
    """Detect patterns that span across row/day boundaries."""
    import re

    # Parse variable indices
    var_indices = {}
    for var in variables:
        nums = re.findall(r'\d+', var)
        if len(nums) >= 3:
            day, shift, pos = int(nums[0]), int(nums[1]), int(nums[2])
            var_indices[var] = (day, shift, pos)

    # Detect consecutive-day patterns
    max_day = max(idx[0] for idx in var_indices.values())
    max_shift = max(idx[1] for idx in var_indices.values())
    patterns = []

    for day in range(max_day):
        # Last shift of day, first shift of next day
        last_shift_vars = [v for v, (d, s, p) in var_indices.items()
                          if d == day and s == max_shift]
        first_shift_vars = [v for v, (d, s, p) in var_indices.items()
                           if d == day+1 and s == 0]

        if last_shift_vars and first_shift_vars:
            patterns.append(last_shift_vars + first_shift_vars)

    return patterns
```

### Fix 2: Improve Count Heuristic

**Change from:**
```python
if count_values.count(max_count) >= 2:
    accept Count(...) <= max_count
```

**To:**
```python
# Always use <= with max observed count
# Equality constraint only if ALL examples have same count AND count == capacity
if min_count == max_count:
    # Check if this looks like a hard constraint (e.g., capacity limit)
    if is_likely_capacity(max_count, domain_size):
        accept Count(...) <= max_count  # Use <=, NOT ==
    else:
        accept Count(...) == max_count
else:
    # Variable counts → upper bound
    accept Count(...) <= max_count
```

**Rationale:** Bounded constraints (`<=`) are more common than exact constraints (`==`) in scheduling problems.

### Fix 3: Prevent Phase 3 Over-fitting

**Problem:** Phase 3 (MQuAcq-2) learns 171 binary constraints that create conflicts.

**Solution:** Increase Phase 2 budget to ensure ALL global constraints are validated before Phase 3.

```python
config.total_budget = 1000  # Instead of 500
config.base_budget_per_constraint = 20  # Instead of 10
```

**Effect:** More queries in Phase 2 → better validation of globals → fewer compensatory binary constraints in Phase 3.

### Fix 4: Add UNSAT Detection

```python
# After Phase 1
if not self._is_satisfiable(self.B_globals):
    logger.warning("Candidate globals are UNSAT - contains conflicts")
    # Remove conflicting candidates

# After learning
if not self._is_satisfiable(final_model):
    logger.error("Learned model is UNSAT!")
    raise ValueError("Cannot learn an unsatisfiable model")
```

## Summary

**Root causes identified:**

1. **Pattern grouping is too simple:** Only detects row/column/block patterns, misses custom patterns
2. **Count heuristic is too strict:** Rejects valid bounds due to natural variation in examples
3. **Count equality over-fitting:** Assumes same count across examples means equality constraint
4. **Phase 3 over-compensation:** Learns 171 conflicting binary constraints

**Impact:**
- Missing 6 consecutive-day constraints (29% of target)
- Missing 3 Count constraints (14% of target)
- Wrong bounds on 2 Count constraints
- 171 incorrect binary constraints causing UNSAT

**Result:** S-Recall = 0% (learned model rejects all valid solutions)

**Fixes available:**
1. Add consecutive-pattern detection
2. Fix Count heuristic to use `<=` by default
3. Increase Phase 2 budget
4. Add UNSAT detection and constraint conflict resolution
