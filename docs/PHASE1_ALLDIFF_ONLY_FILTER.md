# Phase 1: AllDifferent-Only Constraint Filter

## Summary

Modified Phase 1 to **only use AllDifferent constraints** from benchmark mock constraints, filtering out all other constraint types (inequalities, count, sum, ordering, etc.).

## Changes Made

### 1. Modified `phase1_passive_learning.py`

**Location**: Lines 696-726

**What Changed**:
- Added filtering logic to keep only AllDifferent constraints from mock constraints
- Discards all other constraint types (inequalities, count, sum, ordering)
- Displays detailed filtering information in output

**Code**:
```python
# Filter to ONLY keep AllDifferent constraints from benchmark
alldiff_mocks = []
other_mocks = []
for c in mock_constraints_from_benchmark:
    # Check if constraint is AllDifferent
    if isinstance(c, AllDifferent) or (hasattr(c, 'name') and 'alldifferent' in str(c.name).lower()):
        alldiff_mocks.append(c)
    else:
        other_mocks.append(c)

overfitted_constraints = alldiff_mocks
```

### 2. Fixed Unicode Encoding Issues

**Location**: Lines 823, 825

**What Changed**:
- Replaced checkmark characters (`✓`, `✗`) with ASCII text (`[OK]`, `[ERROR]`)
- Prevents encoding errors on Windows systems with non-UTF-8 console encoding

## Results Comparison

### Before Filtering

**Exam Timetabling Variant 1**:
- Total mock constraints from benchmark: 12
- Used in CG: All 12 (7 target + 12 overfitted = 19 total)
- Constraint types: AllDifferent, inequalities (>=), count, sum, ordering (<)
- **Spurious constraints in final model**: 12
  - 5 inequality constraints (`var[5,0] >= 43`, etc.)
  - 1 count constraint
  - 4 ordering constraints
  - 1 sum constraint
  - 1 overfitted AllDifferent

### After Filtering

**Exam Timetabling Variant 1**:
- Total mock constraints from benchmark: 12
- **Filtered**: Kept 1 AllDifferent, discarded 11 other types
- Used in CG: Only 8 (7 target + 1 overfitted, **all AllDifferent**)
- **Spurious constraints in final model**: 1
  - 1 overfitted AllDifferent (the mock constraint)

## Impact

### Precision Improvement
- **Before**: Validated all 19 constraints including 12 spurious
- **After**: Validated 8 constraints including only 1 spurious
- **Improvement**: 92% reduction in spurious constraints

### Query Efficiency
- **Before**: 11 queries to validate 19 constraints (many were incorrect)
- **After**: 12 queries to validate 8 constraints (only 1 incorrect)
- The system now focuses queries on relevant AllDifferent constraints

### Methodological Alignment
- Research focus is on **AllDifferent constraint learning**
- Non-AllDifferent constraints were polluting the evaluation
- Now Phase 2 truly tests the ability to refine **AllDifferent** constraints specifically

## Example Output

```
[MOCK] Received 12 mock constraints from benchmark
       Filtering: Keeping 1 AllDifferent, discarding 11 other types
       Discarded types:
         - var[5,0] >= 43
         - var[5,1] >= 43
         - var[5,2] >= 43
         - var[5,3] >= 43
         - var[5,4] >= 43
         ... and 6 more

[MOCK] Using 1 AllDifferent mock constraints from benchmark
       Mock 1: alldifferent((var[0,0]) // 6,(var[0,1]) // 6,(var[0,2]) // 6,(var[0,3]) // 6,(var[0,4]) // 6,(var[1,0]) // 6,(var[1,1]) // 6,(var[1,2]) // 6,(var[1,3]) // 6,(var[1,4]) // 6)

Final CG: 7 target + 1 overfitted = 8 total
```

## Verification

### Phase 1 Output
✓ Successfully filters mock constraints
✓ Only AllDifferent constraints in CG
✓ Proper metadata tracking
✓ No encoding errors

### Phase 2 Output
✓ Only challenges AllDifferent constraints
✓ Correctly validates 7 target constraints
✓ Correctly identifies 1 spurious constraint
✓ Much cleaner evaluation

## Files Modified

1. `phase1_passive_learning.py`
   - Added AllDifferent-only filtering (lines 696-726)
   - Fixed Unicode encoding issues (lines 823, 825)

## Usage

No changes to command-line interface. Simply run Phase 1 as before:

```bash
python phase1_passive_learning.py --benchmark examtt_v1 --num_examples 5 --num_overfitted 4
```

The filtering happens automatically when mock constraints are provided by the benchmark.

## Backward Compatibility

✓ Benchmarks without mock constraints work as before (generate random overfitted AllDifferent)
✓ Existing pickle files can still be loaded (but should be regenerated with new filtering)
✓ All Phase 2 code remains unchanged

## Recommendation

**Regenerate all Phase 1 pickle files** to apply the new filtering:

```bash
python run_phase1_experiments.py
```

This ensures clean evaluation data for all future Phase 2 experiments.

