# Nurse Rostering S-Rec = 0% Investigation

## Problem Statement

Nurse_Rostering benchmark shows **S-Precision = 100%** but **S-Recall = 0%** across all HCAR variants (Advanced, Heuristic, NoRefine). This indicates a systematic issue with the learned model.

## Investigation Steps

### 1. Benchmark Verification

**Result: PASS**
- Target model is SATISFIABLE
- All 5 positive examples are VALID
- Benchmark is correctly configured (max_workdays = 6)

### 2. Learned Model Satisfiability

**Result: SAT but INCOMPLETE**
- Learned model IS satisfiable (produces solutions)
- Sample learned solution: **INVALID** according to oracle
- Diagnosis: Learned model is missing critical constraints

### 3. Constraint Coverage Analysis

**Target Model** (21 constraints):
```
- 7 AllDifferent constraints (one per day)
- 6 AllDifferent constraints (consecutive days)
- 8 Count constraints (max_workdays per nurse)
```

**Learned Model** (48 constraints):
```
- 3 AllDifferent constraints (MISSING 4 out of 7 days)
- 0 Count constraints (MISSING all 8)
- 0 consecutive-day constraints (MISSING all 6)
- 45 fixed binary constraints
```

**Critical Gaps:**
1. Missing AllDifferent for days 1, 2, 3, 6
2. Missing ALL Count constraints for nurse workload limits
3. Missing ALL consecutive-day constraints

### 4. Root Cause: Blocking Clause Bug

**FOUND BUG in hcar_advanced.py:2746-2753**

The code to generate diverse solutions was COMMENTED OUT:

```python
# Add blocking clause to get different solution
# blocking = []
# for var in variables.values():
#     val = var.value()
#     if val is not None:
#         blocking.append(var != val)
# if blocking:
#     cpm_model += sum(blocking) > 0
```

**Impact:**
- `_sample_solutions()` generated 100 IDENTICAL solutions
- S-Precision/S-Recall calculated on duplicate samples
- Metrics were unreliable

**Fix Applied:** Uncommented blocking clause code (hcar_advanced.py:2746-2753)

### 5. Post-Fix Results

After fixing blocking clause bug:
- S-Precision: 100% (unchanged)
- S-Recall: 0% (unchanged)

**Interpretation:** The bug is fixed but metrics remain at 0% because the LEARNED MODEL itself is incomplete.

### 6. Metrics Interpretation

**S-Precision = 100%:**
- All solutions from learned model satisfy target constraints
- This is possible if learned model is a SUBSET of target
- OR learned constraints happen to be compatible with target

**S-Recall = 0%:**
- ZERO target solutions are accepted by learned model
- Learned model is REJECTING all valid solutions
- Indicates learned model has WRONG/CONFLICTING constraints

**Paradox:** How can learned model be both under-constrained (missing constraints) AND over-restrictive (S-Rec = 0%)?

**Answer:** The 45 fixed binary constraints are likely INCORRECT or CONFLICTING, causing the learned model to reject all valid target solutions while still being satisfiable.

## Deep Dive: Why Are Constraints Missing?

### Pattern Detection Analysis

Variable names: `var[day, shift, position]` where:
- day ∈ [0, 6] (7 days)
- shift ∈ [0, 2] (3 shifts per day)
- position ∈ [0, 1] (2 nurses per shift)

**Pattern grouping logic (`_group_variables_by_pattern`):**
```python
numbers = re.findall(r'\d+', var_name)  # Extracts [day, shift, position]
if len(numbers) >= 2:
    row, col = int(numbers[0]), int(numbers[1])  # Uses only day, shift
```

**Issue:** Grouping only uses first 2 indices (day, shift), ignoring position.

**Impact:**
- Each day creates groups: row_0, row_1, ..., row_6
- Each row_X should contain all 6 variables for that day
- AllDifferent pattern should be detected if values are all different

**Why missing constraints?**

Possible causes:
1. **Pattern detection failure:** Some days don't have AllDifferent in examples
2. **Count constraint detection failure:** `_extract_count_patterns` not finding nurse workload patterns
3. **Budget exhaustion:** Phase 2 ran out of budget before validating all candidates
4. **Query generation failure:** Could not generate refuting queries for some candidates

### Count Constraint Detection

Count constraints should detect: "Each nurse works at most 6 days"

This requires:
- Identifying that values 1-8 represent nurses
- Detecting that each nurse appears in bounded quantity across all variables
- Generating `Count(roster_matrix, nurse_id) <= 6` for each nurse

**Hypothesis:** Count pattern detection is NOT working for this 3D variable structure.

## Remaining Issues

### Issue 1: Missing Global Constraints

**Why:** Pattern detection or validation is failing to identify or confirm:
- AllDifferent for all 7 days
- Count constraints for all 8 nurses
- Consecutive-day constraints

**Investigation needed:**
- Add logging to `_extract_alldifferent_patterns` to see which patterns are detected
- Add logging to `_extract_count_patterns` to see if any Count patterns are found
- Check if examples actually satisfy ALL AllDifferent constraints

### Issue 2: S-Recall = 0% Despite Learned Model Being SAT

**Why:** The 45 fixed binary constraints contain INCORRECT constraints that reject valid solutions.

**Problem:** Binary constraints learned in Phase 3 (MQuAcq-2) may be:
- Over-fitted to the learned global constraints
- Conflicting with true target constraints
- Generated from queries that happened to be invalid

**Investigation needed:**
- Examine the 45 fixed constraints to identify which are incorrect
- Check if they conflict with target model
- Verify Phase 3 (MQuAcq-2) is using correct learned globals

### Issue 3: Why Don't Metrics Match Table 1?

Expected (from Table 1):
```
Nurse_Rostering | HCAR-Advanced | S-Prec: 100% | S-Rec: 0%
```

Actual:
```
Nurse_Rostering | HCAR-Advanced | S-Prec: 100% | S-Rec: 0%
```

**They match!** This suggests the 0% S-Recall is EXPECTED for this benchmark in the current implementation.

**But why?** Looking at the methodology, Phase 2 should:
1. Validate or refute all global constraint candidates
2. Prune fixed-arity bias based on confirmed solutions
3. Ensure learned model converges to target

**Hypothesis:** The current implementation may have bugs in:
- Phase 1: Not detecting all patterns
- Phase 2: Not generating enough validation queries
- Phase 3: Learning incorrect fixed constraints

## Recommendations

### Immediate Actions

1. **Add verbose logging** to pattern detection methods
   - Log all detected patterns
   - Log which patterns are discarded and why
   - Log which constraints hold in examples

2. **Verify examples** satisfy ALL target constraints
   - Print all 5 examples with detailed variable assignments
   - Check AllDifferent for each day
   - Check Count for each nurse
   - Check consecutive-day constraints

3. **Inspect learned fixed constraints**
   - Print all 45 binary constraints
   - Check which ones conflict with target
   - Identify source of incorrect constraints

4. **Increase Phase 2 budget**
   - Current budget may be too low for 21 target constraints
   - Try budget = 1000 to ensure full validation

### Long-term Fixes

1. **Improve pattern detection for 3D structures**
   - Current logic assumes 2D grids (row, col)
   - Nurse rostering has 3D structure (day, shift, position)
   - Need to detect patterns across all dimensions

2. **Add Count constraint detection**
   - Scan for value frequency patterns
   - Detect when specific values appear bounded number of times
   - Generate Count constraints automatically

3. **Fix bias pruning logic**
   - Ensure ground truth solutions prune bias correctly
   - Verify no information loss during pruning
   - Check CONSTRAINT 2 compliance (ground truth only pruning)

4. **Add consistency checks**
   - After learning, verify sol(C_learned) ⊆ sol(C_target)
   - If S-Rec = 0%, flag as critical error
   - Add assertions to catch bugs early

## Summary

**Bug Found:** Blocking clause code was commented out (FIXED)

**Remaining Issues:**
1. Pattern detection missing most AllDifferent constraints
2. Count constraints not detected at all
3. Learned fixed constraints are incorrect/conflicting
4. S-Recall = 0% indicates learned model rejects all valid solutions

**Next Steps:**
1. Add logging to pattern detection
2. Verify examples satisfy all target constraints
3. Inspect learned fixed constraints
4. Improve pattern detection for 3D structures
5. Implement Count constraint detection

**Status:** Blocking clause bug fixed, but learned model remains incomplete. Further investigation needed to understand why pattern detection and constraint learning are failing for Nurse_Rostering.
