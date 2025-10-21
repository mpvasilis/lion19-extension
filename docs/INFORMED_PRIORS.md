# Informed Priors for Phase 1 → Phase 2 Integration

## Overview

The system now uses **informed priors** based on how constraints were generated in Phase 1, rather than uniform priors for all constraints.

## Motivation

**Problem with uniform priors:**
- All constraints start with P(c) = 0.5
- Detected constraints (scope 9, pattern-based) treated same as overfitted constraints (scope 3-5, synthetic)
- No differentiation based on generation method

**Solution with informed priors:**
- Detected constraints → P(c) = 0.8 (high confidence)
- Overfitted constraints → P(c) = 0.3 (low confidence)
- Reflects the reliability of the generation method

## Implementation

### Phase 1 (passive_learning.py)

When creating the pickle file:

```python
# Create informed priors
initial_probabilities = {}
for c in detected_alldiffs:
    initial_probabilities[c] = 0.8  # High prior for detected constraints
for c in overfitted_alldiffs:
    initial_probabilities[c] = 0.3  # Low prior for overfitted constraints

# Save in pickle
output_data = {
    'CG': CG,
    'B_fixed': B_fixed_pruned,
    'E+': positive_examples,
    'variables': instance.X,
    'initial_probabilities': initial_probabilities,  # ← New!
    'metadata': {
        ...
        'prior_detected': 0.8,
        'prior_overfitted': 0.3
    }
}
```

### Phase 2 (main_alldiff_cop.py)

When loading from pickle:

```python
if args.phase1_pickle:
    phase1_data = load_phase1_data(args.phase1_pickle)
    CG = phase1_data['CG']
    
    # Load informed priors if available
    if 'initial_probabilities' in phase1_data:
        probabilities = phase1_data['initial_probabilities']
        print("Using informed priors from Phase 1:")
        print("  Detected: P=0.8, Overfitted: P=0.3")
    else:
        # Fallback to uniform
        probabilities = initialize_probabilities(CG, prior=args.prior)
```

## How It Works

### Example: Sudoku with 27 detected + 4 overfitted

**Initial probabilities:**
```
Constraint                                      | Type       | Prior | Reason
------------------------------------------------|------------|-------|------------------
alldifferent(grid[0,0],...,grid[0,8]) (9 vars) | Detected   | 0.8   | Pattern-based
alldifferent(grid[1,0],...,grid[1,8]) (9 vars) | Detected   | 0.8   | Pattern-based
...27 detected constraints...                   |            | 0.8   |
alldifferent(grid[3,5],grid[4,4],...) (4 vars) | Overfitted | 0.3   | Synthetic
alldifferent(grid[8,2],grid[8,5],...) (5 vars) | Overfitted | 0.3   | Synthetic
...4 overfitted constraints...                  |            | 0.3   |
```

### COP Behavior with Informed Priors

**Weighted violation objective:** `maximize sum((1 - P(c)) · γc)`

| Constraint Type | P(c) | Weight (1-P(c)) | COP Behavior |
|----------------|------|-----------------|--------------|
| Detected | 0.8 | 0.2 | Avoid violating (trusted) |
| Overfitted | 0.3 | 0.7 | Prefer violating (suspicious) |

**Result:** COP intelligently targets overfitted constraints for testing!

## Benefits

### 1. Query Efficiency
- COP preferentially violates overfitted constraints (P=0.3, weight=0.7)
- Detected constraints protected (P=0.8, weight=0.2)
- Faster convergence to correct model

### 2. Methodologically Sound
- Priors reflect generation reliability
- Pattern-based detection is more reliable than synthetic generation
- Aligns with HCAR principles

### 3. Fewer Queries Needed
- Overfitted constraints rejected faster (start at 0.3)
- Detected constraints accepted faster (start at 0.8)
- Less back-and-forth needed

## Example Scenario

**Without informed priors (uniform 0.5):**
```
Iteration 1: Violates random constraint (P=0.5)
Oracle: No
Update: P=0.5 → 0.79
Iteration 2: Still unsure...
Oracle: No
Update: P=0.79 → 0.91 → ACCEPT
Total: 2 queries per constraint
```

**With informed priors:**

**Detected constraint (P=0.8):**
```
Iteration 1: Avoided by COP (high P, low weight)
Iteration 2: Forced to test
Oracle: No
Update: P=0.8 → 0.92 → ACCEPT
Total: 1 query
```

**Overfitted constraint (P=0.3):**
```
Iteration 1: Targeted by COP (low P, high weight)
Oracle: Yes (valid violates it)
Disambiguation → REJECT
Total: 1-2 queries
```

## Backward Compatibility

If `initial_probabilities` is not in the pickle file:
- Falls back to `--prior` argument (default 0.5)
- Works with old Phase 1 pickles
- No breaking changes

## Usage

### Generate Phase 1 with informed priors:
```bash
python phase1_passive_learning.py --benchmark sudoku --output_dir phase1_output
```

This automatically creates informed priors (0.8 for detected, 0.3 for overfitted).

### Use in Phase 2:
```bash
python main_alldiff_cop.py \
    --experiment sudoku \
    --phase1_pickle phase1_output/sudoku_phase1.pkl \
    --max_queries 100
```

System automatically uses the informed priors from the pickle!

## Impact on Results

**Expected improvements:**
- ✅ Faster convergence (fewer iterations)
- ✅ More efficient query generation (targets suspicious constraints)
- ✅ Higher precision (overfitted constraints rejected quickly)
- ✅ 100% recall maintained (detected constraints preserved)

## Theory

This follows Bayesian principles:
- **Prior P(c)** reflects our initial belief based on generation method
- **Evidence** updates this belief through oracle responses
- **Posterior P(c | E)** converges to truth regardless of prior (with enough evidence)
- **Informed priors** just make convergence faster!

## Files Modified

1. `phase1_passive_learning.py`:
   - Added `initial_probabilities` dictionary
   - Saved in pickle output
   - Added metadata about priors

2. `main_alldiff_cop.py`:
   - Loads `initial_probabilities` from pickle
   - Falls back to uniform prior if not available
   - Shows prior distribution

## Related Documentation

- `PHASE1_README.md`: Phase 1 passive learning
- `main_alldiff_cop_README.md`: Phase 2 COP refinement
- `HCAR_METHODOLOGY.md`: Theoretical foundation

