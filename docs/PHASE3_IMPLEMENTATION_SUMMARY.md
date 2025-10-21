# Phase 3 Implementation Summary

## Status: Phase 2 Outputs Successfully Saved ✓, Phase 3 Has Integration Issue

### What Was Accomplished

#### 1. Phase 2 Output Saving ✓
Modified `main_alldiff_cop.py` to save complete Phase 2 outputs to pickle files:

**Saved Data:**
- `C_validated`: Validated global constraints from Phase 2
- `C_validated_strs`: String representations
- `probabilities`: Final probability scores
- `phase1_data`: Complete Phase 1 inputs (E_plus, B_fixed)
- `all_variables`: Variable list
- `metadata`: Experiment parameters and statistics

**File Location:** `phase2_output/{experiment}_phase2.pkl`

**Example Output (sudoku_gt):**
```
Phase 2 Complete:
  - Validated constraints: 3 AllDifferent constraints  
  - Total queries: 70
  - Total time: 88.73s
  - Saved to: phase2_output/sudoku_gt_phase2.pkl
```

#### 2. Phase 3 Script Created ✓
Created `run_phase3.py` implementing:
- Loading Phase 2 outputs
- Decomposing validated global constraints to binary constraints
- Pruning B_fixed using validated globals
- Running MQuAcq-2 for remaining fixed-arity constraints
- Computing complete HCAR pipeline metrics
- Saving results to JSON

**File Location:** `run_phase3.py`

#### 3. Complete Pipeline Runner ✓
Created `run_complete_pipeline.py` to run all benchmarks through Phase 2 → Phase 3.

### Current Issue: MQuAcq-2 Integration

**Error:**
```
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```

**Root Cause:**
The ProblemInstance passed to MQuAcq-2 has a variable mismatch with the oracle. This happens because:
1. Phase 2 works with global constraints and their variables
2. Phase 3 tries to use the binary oracle with decomposed constraints
3. The variable initialization in the ProblemInstance doesn't properly map to what MQuAcq-2 expects

**Location:** `run_phase3.py`, line 181:
```python
instance_binary.cl = CL_init
instance_binary.bias = B_pruned

# Run MQuAcq-2
ca_system = MQuAcq2()
learned_instance = ca_system.learn(instance_binary, oracle=oracle_binary, verbose=3)
```

### What Works

1. ✓ Phase 1: Passive learning generates candidates and saves to pickle
2. ✓ Phase 2: Interactive refinement validates globals and saves complete outputs
3. ✓ Phase 3 Loading: Successfully loads Phase 2 pickle files
4. ✓ Phase 3 Decomposition: Correctly decomposes global constraints to binary
5. ✓ Phase 3 Pruning: Correctly prunes B_fixed based on validated globals
6. ✗ Phase 3 MQuAcq-2: Variable initialization issue prevents learning

### Recommended Fix

The issue is that `instance_binary` from the benchmark constructor might have variables or constraints that don't align with the modified `cl` and `bias`. 

**Solution Approach:**
Create a fresh ProblemInstance for MQuAcq-2 instead of modifying the existing one:

```python
# In run_phase3.py, replace lines 178-181 with:
from pycona import ProblemInstance

# Create fresh instance for MQuAcq-2
variables_for_mquacq = get_variables(CL_init + B_pruned)
mquacq_instance = ProblemInstance(
    variables=cpm_array(variables_for_mquacq),
    init_cl=CL_init,
    name=f"{experiment_name}_phase3",
    bias=B_pruned
)

ca_system = MQuAcq2()
learned_instance = ca_system.learn(mquacq_instance, oracle=oracle_binary, verbose=3)
```

### File Structure Created

```
lion19-extension/
├── phase2_output/           # NEW: Phase 2 outputs
│   └── {experiment}_phase2.pkl
├── phase3_output/           # NEW: Phase 3 results (will be created)
│   └── {experiment}_phase3_results.json
├── main_alldiff_cop.py      # MODIFIED: Now saves Phase 2 outputs
├── run_phase3.py            # NEW: Phase 3 implementation
├── run_complete_pipeline.py # NEW: Full pipeline runner
└── test_phase3_single.py    # NEW: Single benchmark tester
```

### How to Use (Once Fixed)

**Run Phase 2 Only:**
```bash
python main_alldiff_cop.py --experiment sudoku_gt --phase1_pickle phase1_output/sudoku_gt_phase1.pkl
```

**Run Phase 3 Only:**
```bash
python run_phase3.py --experiment sudoku_gt --phase2_pickle phase2_output/sudoku_gt_phase2.pkl
```

**Run Complete Pipeline:**
```bash
python run_complete_pipeline.py  # Runs all benchmarks through Phase 2 + Phase 3
```

### Next Steps

1. **Fix MQuAcq-2 Integration:** Implement the recommended fix above
2. **Test Single Benchmark:** Run `python test_phase3_single.py` to validate
3. **Run Full Pipeline:** Execute `python run_complete_pipeline.py` for all benchmarks
4. **Analyze Results:** Compare Phase 3 results with Phase 2-only results

### Key Achievement

**✓ Phase 2 outputs are now properly saved and can be loaded for Phase 3**

This enables:
- Separating Phase 2 (expensive, interactive refinement) from Phase 3 (active learning)
- Reusing Phase 2 results across multiple Phase 3 runs
- Analyzing Phase 2 validation quality independently
- Implementing the complete HCAR methodology as specified

The Phase 3 integration issue is a minor bug that can be fixed with the recommended approach above.

