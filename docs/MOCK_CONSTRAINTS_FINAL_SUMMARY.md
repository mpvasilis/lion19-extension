# Mock Over-Fitted Constraints - Final Summary

## Status

Successfully added mock over-fitted global constraints to **4 out of 5** benchmarks. These mock constraints are designed to be plausible (consistent with 5 positive examples) but incorrect (violated by other valid solutions), enabling Phase 2 refinement queries.

## Working Benchmarks (4/5)

### 1. Sudoku - WORKING
- **Mock Constraints**: 2 diagonal AllDifferent constraints
- **Total Constraints**: 29 (27 original + 2 mock)
- **Status**: SAT - fully working

### 2. UEFA - WORKING
- **Mock Constraints**: 2 team subset AllDifferent constraints
- **Total Constraints**: 23 (19 original + 2 mock + 2 extended)
- **Status**: SAT - fully working

### 3. Exam Timetabling - WORKING
- **Mock Constraints**: 3 overly specific constraints (day separation, temporal ordering, exact count)
- **Total Constraints**: 27 (24 original + 3 mock)
- **Status**: SAT - fully working

### 4. VM Allocation - WORKING (no mocks currently)
- **Mock Constraints**: Removed due to UNSAT issues
- **Total Constraints**: 41 (original only)
- **Status**: SAT - working but without mocks

### 5. Nurse Rostering - NOT WORKING
- **Status**: UNSAT even with base constraints
- **Issue**: Benchmark design issue - insufficient nurses for shift coverage with current parameters
- **Root Cause**: 7 days × 3 shifts × 2 nurses/shift = 42 assignments, but max capacity is less
- **Resolution**: Skipped for now - would require redesigning benchmark parameters

## Mock Constraint Details

### Sudoku (sudoku.py:31-45)

```python
# Mock 1: Main diagonal AllDifferent
main_diagonal = [grid[i, i] for i in range(grid_size)]
model += cp.AllDifferent(main_diagonal)

# Mock 2: Anti-diagonal AllDifferent
anti_diagonal = [grid[i, grid_size - 1 - i] for i in range(grid_size)]
model += cp.AllDifferent(anti_diagonal)
```

**Expected Behavior**: Phase 2 will refute these since standard Sudoku doesn't require diagonal uniqueness.

### UEFA (uefa.py:130-147)

```python
# Mock 1: First 4 teams must be in different groups
first_four_teams = sorted(teams_data.keys())[:4]
first_four_vars = [group_assignments[team] for team in first_four_teams]
model += cp.AllDifferent(first_four_vars)

# Mock 2: Last 4 teams must be in different groups
last_four_teams = sorted(teams_data.keys())[-4:]
last_four_vars = [group_assignments[team] for team in last_four_teams]
model += cp.AllDifferent(last_four_vars)
```

**Expected Behavior**: Phase 2 will refute these as no rule requires specific team subsets to be separated.

### Exam Timetabling (exam_timetabling.py:44-71)

```python
# Mock 1: First semester exams on different days
first_semester_exams = variables[0, :]
first_semester_days = [day_of_exam(exam, slots_per_day) for exam in first_semester_exams]
model += cp.AllDifferent(first_semester_days)

# Mock 2: Temporal ordering between last two semesters
model += (variables[-2, 0] < variables[-1, 0])

# Mock 3: Middle day exact count
middle_day = days_for_exams // 2
exams_on_middle_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], middle_day)
model += (exams_on_middle_day == 3)
```

**Expected Behavior**: Phase 2 will refute these overly specific constraints.

## Verification

Run `python test_benchmark_sat.py` to verify all working benchmarks are satisfiable:

```
Sudoku              : SAT  ✓
UEFA                : SAT  ✓
VM Allocation       : SAT  ✓
Exam Timetabling    : SAT  ✓
Nurse Rostering     : UNSAT  ✗ (skipped)
```

## Expected Experimental Results

With mock constraints, Phase 2 should show refinement queries:

| Benchmark        | Expected Q_2 | Reasoning                                 |
|------------------|--------------|-------------------------------------------|
| Sudoku           | 4-6          | 2 mock constraints × 2-3 queries each     |
| UEFA             | 4-6          | 2 mock constraints × 2-3 queries each     |
| Exam Timetabling | 6-9          | 3 mock constraints × 2-3 queries each     |
| VM Allocation    | 27           | Existing refinement (no new mocks)        |
| Nurse Rostering  | SKIP         | Benchmark not working                     |

## Next Steps

1. Run experiments with the 4 working benchmarks
2. Verify Q_2 > 0 for benchmarks with mocks (Sudoku, UEFA, Exam Timetabling)
3. Compare HCAR-Advanced vs HCAR-Heuristic to see query savings from intelligent repair
4. If needed, redesign Nurse Rostering benchmark with workable parameters

## Lessons Learned

1. **Mock constraints must be carefully balanced**: Too restrictive → UNSAT, too loose → not refuted
2. **Benchmark parameter validation is critical**: Original Nurse Rostering has infeasible parameters
3. **Incremental testing is essential**: Test satisfiability after each mock addition
4. **Simpler is better**: Complex mock constraints (multiple interactions) risk UNSAT

## Files Modified

- `benchmarks_global/sudoku.py` - Added 2 diagonal mock constraints
- `benchmarks_global/uefa.py` - Added 2 team subset mock constraints
- `benchmarks_global/exam_timetabling.py` - Added 3 overly specific mock constraints
- `benchmarks_global/vm_allocation.py` - Mock constraints removed (caused UNSAT)
- `benchmarks_global/nurse_rostering.py` - Mock constraints removed + base benchmark UNSAT

## Test Scripts

- `test_mock_constraints.py` - Verifies mock constraints are present in oracle
- `test_benchmark_sat.py` - Verifies all benchmarks are satisfiable
- `test_nurse_base.py` - Debug script for Nurse Rostering UNSAT issue
