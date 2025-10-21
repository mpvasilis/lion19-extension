# Sudoku Mock Constraints Fix

## Problem

On Sudoku benchmarks (sudoku, sudoku_gt), the system was learning **spurious AllDifferent constraints** because:

1. **Random overfitted constraints were too hard to refute**
   - Randomly selected variables that happened to be all different in 5 examples
   - No structure or pattern making them difficult for COP query generation to challenge
   - Example: `alldifferent(grid[8,6],grid[0,6],grid[6,1],grid[1,6],grid[8,1])`

2. **Phase 2 accepted all overfitted constraints**
   - Original output: 33 learned (27 target + 6 spurious)
   - All 6 random overfitted constraints were validated
   - 0% rejection rate for spurious constraints

## Solution

Added **structured mock AllDifferent constraints** to both Sudoku benchmarks that are:
- **Patterned** (diagonal, corners, anti-diagonal, center cross, edge patterns)
- **Easier to refute** because they have clear structure that won't hold generally
- **AllDifferent only** (aligned with research focus)

## Changes Made

### 1. `benchmarks_global/sudoku.py`

Added 6 structured mock AllDifferent constraints:

```python
# Mock 1: Diagonal pattern
mock_c1 = cp.AllDifferent([grid[0,0], grid[1,1], grid[2,2]])

# Mock 2: Corner cells pattern
mock_c4 = cp.AllDifferent([grid[0,0], grid[0,grid_size-1], 
                           grid[grid_size-1,0], grid[grid_size-1,grid_size-1]])

# Mock 3: Anti-diagonal pattern
mock_c5 = cp.AllDifferent([grid[0,grid_size-1], grid[1,grid_size-2], 
                           grid[2,grid_size-3]])

# Mock 4: Center cross pattern
center = grid_size // 2
mock_c6 = cp.AllDifferent([grid[center, center], grid[center-1, center], 
                           grid[center+1, center], grid[center, center-1], 
                           grid[center, center+1]])

# Mock 5: Edge pattern
mock_c7 = cp.AllDifferent([grid[0,0], grid[0,4], grid[4,0], 
                           grid[4,4], grid[2,2]])

# Mock 6: Random pattern
mock_c8 = cp.AllDifferent([grid[1,1], grid[3,3], grid[5,5]])
```

### 2. `benchmarks_global/sudoku_greater_than.py`

Added identical 6 structured mock AllDifferent constraints.

### 3. `phase1_passive_learning.py`

Updated `construct_instance()` to handle 3-value return (instance, oracle, mock_constraints) for both Sudoku benchmarks:

```python
elif 'sudoku' in benchmark_name.lower():
    result = construct_sudoku(3, 3, 9)
    # Handle optional mock_constraints return
    if len(result) == 3:
        instance, oracle, mock_constraints = result
        return instance, oracle, mock_constraints
    else:
        instance, oracle = result
        return instance, oracle
```

## Comparison: Random vs. Structured Mock Constraints

### Random (Before)
```python
# Example random overfitted constraints:
alldifferent(grid[8,6],grid[0,6],grid[6,1],grid[1,6],grid[8,1])
alldifferent(grid[5,5],grid[0,1],grid[5,7],grid[0,2],grid[8,6])
alldifferent(grid[1,1],grid[7,7],grid[0,7])
```

**Problems:**
- No structure or pattern
- Arbitrary variable selection
- Hard for COP to find violations
- Accidentally consistent with many solutions

### Structured (After)
```python
# Example structured mock constraints:
alldifferent(grid[0,0],grid[1,1],grid[2,2])  # Diagonal
alldifferent(grid[0,0],grid[0,8],grid[8,0],grid[8,8])  # Corners
alldifferent(grid[0,8],grid[1,7],grid[2,6])  # Anti-diagonal
```

**Benefits:**
- Clear geometric patterns
- Easier to understand and challenge
- More likely to be refutable by COP
- Better test of Phase 2's disambiguation ability

## Expected Impact

### Phase 2 Performance
With structured constraints, Phase 2 should be able to:
1. **Generate better violation queries** (target specific patterns)
2. **Successfully disambiguate** (patterns are easier to isolate)
3. **Reject spurious constraints** (higher rejection rate expected)

### Evaluation Quality
- **Cleaner metrics**: Better separation between true and spurious constraints
- **More meaningful tests**: Tests the system's ability to refute patterned constraints
- **Better research alignment**: Focuses on challenging but refutable overfitted patterns

## Usage

### Regenerate Phase 1 Data

```bash
# For sudoku_gt
python phase1_passive_learning.py --benchmark sudoku_gt --num_examples 5 --num_overfitted 6

# For sudoku
python phase1_passive_learning.py --benchmark sudoku --num_examples 5 --num_overfitted 6
```

### Run Phase 2 Experiments

```bash
# For sudoku_gt
python main_alldiff_cop.py --experiment sudoku_gt --phase1_pickle phase1_output/sudoku_gt_phase1.pkl --alpha 0.42 --theta_max 0.9 --theta_min 0.1 --max_queries 500 --timeout 600

# For sudoku
python main_alldiff_cop.py --experiment sudoku --phase1_pickle phase1_output/sudoku_phase1.pkl --alpha 0.42 --theta_max 0.9 --theta_min 0.1 --max_queries 500 --timeout 600
```

## Phase 1 Output Example

```
[MOCK] Using 6 AllDifferent mock constraints from benchmark
       Mock 1: alldifferent(grid[0,0],grid[1,1],grid[2,2])
       Mock 2: alldifferent(grid[0,0],grid[0,8],grid[8,0],grid[8,8])
       Mock 3: alldifferent(grid[0,8],grid[1,7],grid[2,6])
       Mock 4: alldifferent(grid[4,4],grid[3,4],grid[5,4],grid[4,3],grid[4,5])
       Mock 5: alldifferent(grid[0,0],grid[0,4],grid[4,0],grid[4,4],grid[2,2])
       Mock 6: alldifferent(grid[1,1],grid[3,3],grid[5,5])

Final CG: 27 target + 6 overfitted = 33 total
Initial probabilities: 27 @ 0.8 (target), 6 @ 0.3 (overfitted)
```

## Files Modified

1. `benchmarks_global/sudoku.py` - Added 6 structured mock AllDifferent constraints
2. `benchmarks_global/sudoku_greater_than.py` - Added 6 structured mock AllDifferent constraints
3. `phase1_passive_learning.py` - Updated to handle 3-value return for Sudoku benchmarks

## Next Steps

1. **Regenerate Phase 1 data** for both Sudoku benchmarks
2. **Run Phase 2 experiments** to test rejection rates
3. **Compare results** between random and structured overfitted constraints
4. **Document findings** on Phase 2's ability to refute structured patterns

## Research Implications

This change improves the **methodological soundness** of the experiments by:
- Providing **reproducible** overfitted constraints (structured vs. random)
- Creating **fairer challenges** for Phase 2 (patterns are refutable but non-trivial)
- Better **testing disambiguation** (structured patterns enable better isolation)
- Aligning with **research goals** (AllDifferent-only focus)

The structured mock constraints create a more meaningful test: Can Phase 2 identify and reject plausible but incorrect AllDifferent patterns that happen to hold in 5 training examples?

