# Phase 3: Solution-Level Metrics and Final Model Persistence

## Overview

This document describes the enhancements made to Phase 3 to include **solution-space precision and recall** computation and **final model persistence** across all three phases.

## Changes Made

### 1. Solution-Space Metrics Computation

Added a new function `compute_solution_metrics()` that computes:

- **S-Precision**: `|Sol(Learned) ∩ Sol(Target)| / |Sol(Learned)|`
- **S-Recall**: `|Sol(Learned) ∩ Sol(Target)| / |Sol(Target)|`
- **S-F1**: Harmonic mean of S-Precision and S-Recall

#### Key Features

- **Solution Sampling**: Uses `solve()` with exclusion constraints for controlled sampling
- **Timeout Protection**: Configurable timeout (default: 300s per model)
- **Max Solutions**: Configurable limit (default: 100 solutions for sampling)
- **Completeness Tracking**: Reports whether full enumeration was achieved
- **Progress Reporting**: Prints progress every 100 solutions
- **Efficient Approach**: Avoids expensive full enumeration, suitable for large solution spaces

#### Implementation Details

The function uses an efficient sampling approach with solver restarts:

```python
def compute_solution_metrics(learned_constraints, target_constraints, variables, 
                             max_solutions=100, timeout_per_model=300):
    """
    Compute solution-space precision and recall using sampling.
    
    Algorithm:
    1. Create model with constraints
    2. While count < max_solutions:
       a. Solve the model
       b. If no solution found, break (complete enumeration)
       c. Extract and store solution
       d. Add exclusion constraint: any(v != v.value() for v in variables)
       e. Repeat (solver restart with new constraint)
    
    Returns:
        dict with:
        - s_precision: Solution-space precision
        - s_recall: Solution-space recall
        - s_f1: Solution-space F1 score
        - learned_solutions: Number of solutions sampled from learned model
        - target_solutions: Number of solutions sampled from target model
        - intersection_solutions: Number of common solutions
        - is_complete: Whether full enumeration was achieved
        - learned_incomplete: Whether learned model enumeration was incomplete
        - target_incomplete: Whether target model enumeration was incomplete
    """
```

### 2. Final Model Persistence

The complete learned model from all three phases is now saved to a pickle file:

#### File: `phase3_output/{experiment}_final_model.pkl`

Contains:
- **experiment**: Benchmark name
- **timestamp**: When the model was learned
- **phase1_data**: Complete Phase 1 data (E+, B_fixed, etc.)
- **C_validated**: Validated global constraints from Phase 2
- **final_constraints**: Complete learned model (Phase 1 + Phase 2 + Phase 3)
- **variables**: Problem variables
- **phase_stats**: Statistics from all three phases
  - Phase 1: queries, time, E+ size, B_fixed size
  - Phase 2: queries, time, validated globals
  - Phase 3: queries, time, final model size
  - Total: Combined queries and time
- **evaluation**: Both constraint-level and solution-level metrics

### 3. Enhanced Results Output

#### JSON Results: `phase3_output/{experiment}_phase3_results.json`

Now includes:
```json
{
  "evaluation": {
    "constraint_level": {
      "target_size": 324,
      "learned_size": 324,
      "correct": 324,
      "missing": 0,
      "spurious": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    },
    "solution_level": {
      "s_precision": 1.0,
      "s_recall": 1.0,
      "s_f1": 1.0,
      "learned_solutions": 6670903752021072936960,
      "target_solutions": 6670903752021072936960,
      "intersection_solutions": 6670903752021072936960,
      "is_complete": true,
      "learned_incomplete": false,
      "target_incomplete": false
    }
  }
}
```

## Usage

### Running Phase 3 with Solution Metrics

```bash
python run_phase3.py --experiment sudoku --phase2_pickle phase2_output/sudoku_phase2.pkl
```

### Running Complete Pipeline

```bash
python run_complete_pipeline.py
```

This will:
1. Run Phase 2 for all benchmarks
2. Run Phase 3 for all benchmarks (including solution metrics)
3. Save final models to pickle files
4. Generate comprehensive evaluation reports

## Output Files

For each experiment (e.g., `sudoku`):

1. **JSON Results**: `phase3_output/sudoku_phase3_results.json`
   - Human-readable JSON with all statistics
   - Includes both constraint-level and solution-level metrics

2. **Final Model Pickle**: `phase3_output/sudoku_final_model.pkl`
   - Complete learned model from all three phases
   - Can be loaded for deployment or further analysis

## Loading the Final Model

```python
import pickle

# Load the complete learned model
with open('phase3_output/sudoku_final_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Access components
phase1_data = model_data['phase1_data']
validated_globals = model_data['C_validated']
final_constraints = model_data['final_constraints']
variables = model_data['variables']
evaluation = model_data['evaluation']

print(f"Learned {len(final_constraints)} constraints")
print(f"S-Precision: {evaluation['solution_level']['s_precision']:.2%}")
print(f"S-Recall: {evaluation['solution_level']['s_recall']:.2%}")
```

## Interpretation of Metrics

### Perfect Learning (Expected for HCAR)

```
Constraint-Level Metrics:
  Precision: 100%    (no spurious constraints)
  Recall: 100%       (no missing constraints)

Solution-Space Metrics:
  S-Precision: 100%  (all learned solutions are valid)
  S-Recall: 100%     (all target solutions are found)
```

This indicates:
- ✅ The learned model is **solution-equivalent** to the target
- ✅ No over-fitting (S-Recall = 100%)
- ✅ No under-fitting (S-Precision = 100%)

### Over-Fitting Detection

If Phase 2 refinement is disabled (HCAR-NoRefine):

```
Constraint-Level Metrics:
  Precision: 80%
  Recall: 100%

Solution-Space Metrics:
  S-Precision: 100%
  S-Recall: 39%      ← Critical metric! Over-fitting detected
```

The low S-Recall indicates the learned model is **too restrictive** (over-fitted), which is exactly what Phase 2 is designed to fix.

## Performance Considerations

### Large Solution Spaces

For benchmarks with very large solution spaces (e.g., Sudoku: 6.67×10²¹ solutions):

- **Sampling approach**: Uses exclusion constraints to sample diverse solutions
- **Metrics are approximate**: Based on sampled solutions (default: 100)
- **Efficient**: Avoids expensive full enumeration
- **Warning message**: Will indicate if enumeration was incomplete

### Configuration

Adjust for your benchmark characteristics:

```python
# For quick validation (default - recommended)
solution_metrics = compute_solution_metrics(
    learned_constraints, target_constraints, variables,
    max_solutions=100,      # Sample 100 solutions
    timeout_per_model=300   # 5 minutes per model
)

# For more thorough sampling
solution_metrics = compute_solution_metrics(
    learned_constraints, target_constraints, variables,
    max_solutions=500,      # Sample 500 solutions
    timeout_per_model=600   # 10 minutes per model
)

# For small problems (can enumerate all)
solution_metrics = compute_solution_metrics(
    learned_constraints, target_constraints, variables,
    max_solutions=10000,    # Will stop when no more solutions
    timeout_per_model=300
)

# For very large problems (minimal sampling)
solution_metrics = compute_solution_metrics(
    learned_constraints, target_constraints, variables,
    max_solutions=50,       # Quick check with 50 solutions
    timeout_per_model=180   # 3 minutes per model
)
```

**Note**: The algorithm will automatically stop if it exhausts all solutions before reaching `max_solutions`, making it suitable for both small and large problem instances.

## Validation Checks

The system now performs comprehensive validation:

1. **Constraint-Level Check**: Ensures learned constraints match target
2. **Solution-Level Check**: Ensures solution spaces are equivalent
3. **Completeness Check**: Reports if metrics are exact or approximate

## Success Criteria

For a complete and correct HCAR implementation:

| Metric | Expected Value | Importance |
|--------|---------------|------------|
| S-Precision | 100% | ⭐⭐⭐ Critical |
| S-Recall | 100% | ⭐⭐⭐ Critical |
| C-Precision | 100% | ⭐⭐ Important |
| C-Recall | 100% | ⭐⭐ Important |
| Queries | < 500 | ⭐ Efficiency |

## Next Steps

1. **Run complete pipeline** for all benchmarks:
   ```bash
   python run_complete_pipeline.py
   ```

2. **Analyze results** from pickle files:
   ```bash
   python -c "
   import pickle
   import json
   
   with open('phase3_output/sudoku_final_model.pkl', 'rb') as f:
       data = pickle.load(f)
   
   print(json.dumps(data['evaluation'], indent=2))
   "
   ```

3. **Compare variants** (HCAR-Advanced vs HCAR-NoRefine):
   - Check solution-level metrics to validate Phase 2's importance
   - Confirm that HCAR-NoRefine has degraded S-Recall

## References

- **HCAR Methodology**: See `CLAUDE.md` for complete specification
- **Phase 1**: See `docs/PHASE1_README.md`
- **Phase 2**: See `docs/main_alldiff_cop_README.md`
- **Resilient FindC**: See `docs/RESILIENT_FINDC.md`

