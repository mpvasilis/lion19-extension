# Benchmark Loading Fix - Summary

## Problem
The `run_hcar_experiments.py` script was trying to call a non-existent `create_problem()` method on benchmark modules, causing failures when loading benchmarks.

## Solution
Fixed the benchmark loading system to properly interface with the actual structure of the benchmark modules.

## Changes Made

### 1. Updated Imports
- Imported both `benchmarks_global/` (with global constraints) and `benchmarks/` (fixed-arity only) modules separately
- Added import for `ConstraintOracle` from PyConA

### 2. Rewrote `BenchmarkConfig` Class
The class now:
- Takes a `constructor_func` parameter instead of a module
- Accepts `**constructor_params` to pass the right parameters to each benchmark
- Properly interfaces with PyConA's `ProblemInstance` and `ConstraintOracle` objects
- Extracts variables, domains, and constraints from the returned instances
- Generates positive examples using CPMpy solver
- Creates a compatible oracle function for HCAR

### 3. Updated `get_benchmark_configs()` Function
Now:
- Accepts `use_global_constraints` parameter to select between benchmark types
- Returns properly configured `BenchmarkConfig` objects with correct parameters for each benchmark:
  - **Sudoku**: `block_size_row`, `block_size_col`, `grid_size`
  - **UEFA**: `teams_data`, `n_groups`, `teams_per_group`
  - **VM_Allocation**: `pm_data`, `vm_data` (with capacity and demand dictionaries)
  - **Exam_Timetabling**: `nsemesters`, `courses_per_semester`, `slots_per_day`, `days_for_exams`
  - **Nurse_Rostering**: `shifts_per_day`, `num_days`, `num_nurses`, `nurses_per_shift`, `max_workdays`

### 4. Added Command-Line Options
- `--use-global` (default): Use benchmarks with global constraints
- `--use-binary`: Use benchmarks with only fixed-arity constraints

### 5. Updated All Experiment Functions
- `run_single_experiment()`: Now accepts `use_global_constraints` parameter
- `run_full_comparison()`: Now accepts `use_global_constraints` parameter
- `main()`: Determines which benchmark type to use based on command-line arguments

## Usage

### Run with global constraints (default):
```bash
python run_hcar_experiments.py --benchmark Sudoku --method HCAR-Advanced
```

### Run with fixed-arity constraints only:
```bash
python run_hcar_experiments.py --benchmark Sudoku --method HCAR-Advanced --use-binary
```

### Run full comparison with global constraints:
```bash
python run_hcar_experiments.py --all --runs 3
```

### Run full comparison with fixed-arity constraints:
```bash
python run_hcar_experiments.py --all --runs 3 --use-binary
```

## Benchmark Configurations

### Global Constraints (`benchmarks_global/`)
These benchmarks use high-level constraints like `AllDifferent`, `Count`, and `Sum` directly:
- More compact model representation
- Suitable for testing HCAR's ability to learn global constraints
- Matches the paper's methodology

### Fixed-Arity Only (`benchmarks/`)
These benchmarks decompose global constraints into binary/ternary constraints:
- Pure active learning baseline
- All constraints are of fixed arity (typically binary)
- Useful for comparison with traditional CA approaches

## Key Interface Details

### What `BenchmarkConfig.load()` Returns
```python
{
    'variables': Dict[str, CPMPyVariable],     # name -> variable mapping
    'domains': Dict[str, Tuple[int, int]],     # name -> (lb, ub) mapping
    'target_model': List[CPMPyConstraint],     # ground truth constraints
    'oracle_func': Callable[[Dict], OracleResponse],  # validation function
    'positive_examples': List[Dict[str, int]]   # initial E+
}
```

### Oracle Function Interface
The oracle function:
- Takes an assignment dictionary: `{var_name: value, ...}`
- Returns `OracleResponse.VALID` or `OracleResponse.INVALID`
- Uses PyConA's `ConstraintOracle.ask_query()` under the hood

### Positive Examples Generation
- Generates diverse solutions by solving the target model multiple times
- Adds exclusion constraints after each solution to ensure diversity
- Defaults to 5 examples (as specified in HCAR methodology)

## Troubleshooting

### Issue: "Could not import benchmarks"
**Solution**: Make sure both `benchmarks/` and `benchmarks_global/` folders are in your path and PyConA is installed.

### Issue: "Error loading benchmark X"
**Solution**: Check that the benchmark's constructor function exists and has the correct signature. Look at the logs for detailed error messages.

### Issue: "Only generated N/5 examples"
**Solution**: This means the problem constraints are very restrictive. The system will continue with fewer examples, but results may be less reliable.

### Issue: "TypeError: 'int' object is not callable" when accessing `var.lb()` or `var.ub()`
**Solution**: ✅ **FIXED** - CPMpy variables have `lb` and `ub` as properties, not methods. The code now handles both property and method access patterns.

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'hcar_results/full_comparison.json'"
**Solution**: ✅ **FIXED** - The script now automatically creates the `hcar_results/` directory before writing results.

### Issue: "AttributeError: 'NoneType' object has no attribute 'is_bool'" during example generation
**Solution**: ✅ **FIXED** - Improved example generation with:
- Better error handling and validation
- Checks for `None` values before creating constraints
- Graceful fallback if exclusion constraints fail
- Verification that benchmark is satisfiable before generating examples
- Better logging of example generation progress

### Issue: Nurse Rostering benchmark is UNSAT
**Solution**: ✅ **FIXED** - Updated nurse rostering parameters to use `num_nurses=8` instead of 5. The problem requires at least `shifts_per_day * nurses_per_shift = 6` nurses per day, so having only 5 nurses made the problem unsatisfiable.

## Notes for HCAR Methodology

According to the HCAR specification:
- **Phase 1** (Passive Learning): Extracts global constraint candidates from 5 positive examples
- **Phase 2** (Interactive Refinement): Uses the oracle to validate/refute candidates
- **Phase 3** (Active Completion): Uses MQuAcq-2 for remaining fixed-arity constraints

This fix ensures that all three phases can access the proper benchmark data structure.

