# Phase 1 Passive Learning System

This module implements the passive learning phase (Phase 1) of the HCAR methodology.

## Overview

Phase 1 performs passive constraint acquisition from positive examples and prepares data for Phase 2 interactive refinement.

### What Phase 1 Does

1. **Generates Positive Examples**: Solves the target model to generate 5 diverse valid solutions
2. **Detects AllDifferent Patterns**: Finds variable subsets where values are all different in every example
3. **Adds Overfitted Constraints**: Generates 3-5 synthetic AllDifferent constraints that satisfy examples but aren't in the target
4. **Generates Binary Bias**: Creates all pairwise constraints using the language (==, !=, <, >, <=, >=)
5. **Prunes Bias**: Removes constraints inconsistent with positive examples
6. **Saves to Pickle**: Stores all data for Phase 2

### Output Format

The pickle file contains:
```python
{
    'CG': [AllDifferent constraints],           # Detected + overfitted
    'B_fixed': [pruned binary constraints],     # Consistent with E+
    'E+': [{var_name: value, ...}, ...],       # Positive examples
    'variables': [CPMpy variables],             # Problem variables
    'metadata': {
        'benchmark': 'sudoku',
        'num_examples': 5,
        'num_detected_alldiffs': 27,
        'num_overfitted_alldiffs': 4,
        'num_bias_initial': 12960,
        'num_bias_pruned': 6480,
        'target_alldiff_count': 27
    }
}
```

## Usage

### Running Phase 1

Generate Phase 1 data for a benchmark:

```bash
# Sudoku
python phase1_passive_learning.py --benchmark sudoku

# Exam Timetabling
python phase1_passive_learning.py --benchmark examtt

# Nurse Rostering
python phase1_passive_learning.py --benchmark nurse

# UEFA Champions League
python phase1_passive_learning.py --benchmark uefa
```

### Optional Arguments

```bash
python phase1_passive_learning.py \
    --benchmark sudoku \
    --output_dir phase1_output \
    --num_examples 5 \
    --num_overfitted 4
```

Options:
- `--benchmark`: Benchmark name (required) - sudoku, examtt, nurse, uefa
- `--output_dir`: Output directory for pickle files (default: phase1_output)
- `--num_examples`: Number of positive examples to generate (default: 5)
- `--num_overfitted`: Number of overfitted constraints to add (default: 4)

### Output Location

Phase 1 saves pickle files to:
```
phase1_output/
  ├── sudoku_phase1.pkl
  ├── examtt_phase1.pkl
  ├── nurse_phase1.pkl
  └── uefa_phase1.pkl
```

## Using Phase 1 Output in Phase 2

### Integration with main_alldiff_cop.py

Run Phase 2 refinement using Phase 1 data:

```bash
python main_alldiff_cop.py \
    --experiment sudoku \
    --phase1_pickle phase1_output/sudoku_phase1.pkl \
    --alpha 0.42 \
    --theta_max 0.9 \
    --theta_min 0.1 \
    --max_queries 500
```

When `--phase1_pickle` is provided:
- Loads CG from pickle (detected + overfitted constraints)
- Skips extraction from oracle
- Proceeds directly to Phase 2 refinement

### Without Phase 1 Pickle

You can still run Phase 2 standalone (original behavior):

```bash
python main_alldiff_cop.py --experiment sudoku
```

This extracts AllDifferent constraints directly from the oracle (no overfitted constraints).

## Complete Workflow Example

### 1. Run Phase 1 for all benchmarks

```bash
for benchmark in sudoku examtt nurse uefa; do
    python phase1_passive_learning.py --benchmark $benchmark
done
```

### 2. Run Phase 2 refinement

```bash
# Sudoku with Phase 1 data
python main_alldiff_cop.py \
    --experiment sudoku \
    --phase1_pickle phase1_output/sudoku_phase1.pkl \
    --max_queries 500

# Exam Timetabling with Phase 1 data
python main_alldiff_cop.py \
    --experiment examtt \
    --phase1_pickle phase1_output/examtt_phase1.pkl \
    --max_queries 500
```

## Implementation Details

### Pattern Detection Strategy

The system uses **two complementary strategies** for detecting AllDifferent constraints:

#### 1. Pattern-Based Detection (Default, Efficient)

Detects structured patterns by analyzing variable naming conventions:

**For Sudoku-style grids** (`grid[i,j]`):
- **Rows**: All cells in same row (e.g., `grid[0,0]`, `grid[0,1]`, ..., `grid[0,8]`)
- **Columns**: All cells in same column (e.g., `grid[0,0]`, `grid[1,0]`, ..., `grid[8,0]`)
- **Blocks**: 3×3 sub-grids (e.g., `grid[0,0]` through `grid[2,2]`)

**For Multi-dimensional problems** (`var[day,shift,nurse]`):
- Infers dimensions by parsing all variable names
- Detects slices across each dimension
- Validates with positive examples

**Benefits**:
- Fast: O(n) instead of O(2^n)
- Targeted: Finds expected constraint patterns
- Complete: Captures all standard row/column/block constraints

#### 2. Combinatorial Search (Optional, Expensive)

Enumerates all variable subsets of specified scope:
1. Try all subsets of size min_scope to max_scope
2. For each subset, check if values are all different in every example
3. Constraints that satisfy this property are detected patterns

**Usage**:
```python
detected = detect_alldifferent_patterns(
    variables, 
    positive_examples,
    use_structured=True,      # Enable pattern-based (recommended)
    use_combinatorial=False,  # Enable combinatorial (expensive)
    min_scope=7,              # Minimum subset size
    max_scope=11              # Maximum subset size
)
```

**Default behavior**: Only pattern-based detection is enabled for efficiency

### Overfitting Generation

Overfitted constraints are generated by:
1. Randomly sampling variable subsets (size 3-7)
2. Checking if they satisfy AllDifferent in all examples
3. Ensuring they are NOT in the target model (using variable name comparison)
4. These constraints are consistent with examples but are spurious

**Purpose**: Test Phase 2's ability to identify and reject over-fitted constraints

### Bias Pruning

The binary bias is pruned by:
1. Generating all pairwise constraints using the language
2. Evaluating each constraint on all positive examples
3. Removing constraints violated by any example
4. Only keeping constraints consistent with all examples

**Efficiency**: For Sudoku (81 vars), generates ~12k binary constraints, prunes to ~6k

## Key Design Principles

Following HCAR methodology constraints:

### CONSTRAINT 1: Independence of Biases
✓ B_globals (CG) and B_fixed maintained as independent sets

### CONSTRAINT 2: Ground Truth Only Pruning
✓ B_fixed pruned ONLY using E+ (positive examples)
✓ No oracle queries used in Phase 1

### Passive Learning Philosophy
- Phase 1 is purely passive (no oracle queries)
- Over-fitting is expected and intentional (will be corrected in Phase 2)
- Positive examples are the only ground truth for pruning

## Troubleshooting

### "Could not generate any positive examples"
- The target model is UNSAT
- Check benchmark construction functions

### "Only generated X overfitted constraints"
- Not enough variable combinations satisfy AllDifferent in examples
- Reduce `--num_overfitted` or check example diversity

### Large bias size
- For problems with many variables, the binary bias can be large
- This is expected; Phase 2 will refine it
- Consider filtering by scope if memory is an issue

## Files

- `phase1_passive_learning.py` - Main implementation
- `main_alldiff_cop.py` - Phase 2 refinement (modified to load Phase 1 data)
- `PHASE1_README.md` - This file
- `phase1_output/` - Output directory (created automatically)

## Next Steps

After running Phase 1:
1. Inspect pickle files to verify data
2. Run Phase 2 refinement with Phase 1 data
3. Compare results with/without overfitted constraints
4. Analyze Phase 2's ability to reject spurious constraints

