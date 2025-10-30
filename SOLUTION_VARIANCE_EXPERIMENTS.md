# Solution Variance Experiments

## Overview

This experiment script (`run_solution_variance_experiments.py`) tests how the number of solutions and mock constraints affect the learning pipeline performance.

## Key Features

### 1. **Inverse Relationship**
As the number of solutions increases, mock constraints decrease:

| Solutions | Mock Constraints |
|-----------|------------------|
| 2         | 50               |
| 5         | 20               |
| 10        | 10               |
| 50        | 2                |

**Rationale**: More solutions provide better coverage of the constraint space, reducing the need for overfitted (mock) constraints.

### 2. **Phase 1 Timing**
The script precisely measures and logs the time taken for Phase 1 (passive learning) for each configuration.

### 3. **Complete Pipeline**
For each configuration, the script runs:
- **Phase 1**: Passive learning with timing
- **Phase 2**: Active constraint acquisition
- **Phase 3**: Final model refinement

### 4. **Comprehensive Reporting**
Generates two output files:
- `variance_experiment_report.txt` - Human-readable summary
- `variance_experiment_results.json` - Machine-readable data

## Benchmarks Tested

The script tests 8 benchmarks:
1. `sudoku` - Standard 9x9 Sudoku
2. `sudoku_gt` - Sudoku with greater-than constraints
3. `jsudoku` - Jigsaw Sudoku
4. `latin_square` - 9x9 Latin Square
5. `graph_coloring_register` - Register allocation
6. `examtt_v1` - Exam timetabling variant 1
7. `examtt_v2` - Exam timetabling variant 2
8. `nurse` - Nurse rostering

## Usage

### Run All Experiments
```bash
python run_solution_variance_experiments.py
```

This will:
1. Run all 8 benchmarks
2. Test each with 4 solution configurations (2, 5, 10, 50)
3. Total: 32 complete pipeline runs

### Expected Runtime
- Each configuration: 5-30 minutes (depending on benchmark complexity)
- Total experiment: 3-16 hours (estimate)

## Output Structure

```
solution_variance_output/
├── variance_results.txt                # Formatted results (human-readable)
├── variance_results.csv                # CSV format for analysis
├── variance_experiment_detailed.json   # Complete JSON data
└── <benchmark>_sol<N>_mock<M>/         # Per-configuration outputs
    ├── <benchmark>_phase1.pkl
    └── (intermediate files)
```

## Example Output Format

The output follows the standard HCAR pipeline format:

```
Prob.           Sols   StartC   InvC   CT    Bias   ViolQ  MQuQ   TQ    P1T(s)  VT(s)   MQuT(s)  TT(s)
================================================================================
Sudoku          2      37       10     27    2291   120    1500   1620  8.45    320.15  145.30   473.90
Sudoku          5      37       10     27    2291   101    1232   1333  15.32   317.69  141.14   474.15
Sudoku          10     37       10     27    2291   95     1100   1195  25.67   310.22  135.80   471.69
Sudoku          50     37       10     27    2291   88     980    1068  78.12   298.45  128.33   504.90
```

### Column Descriptions

- **Prob.**: Problem/Benchmark name
- **Sols**: Number of solutions (2, 5, 10, or 50)
- **StartC**: Starting candidate constraints from Phase 1
- **InvC**: Invalid constraints removed
- **CT**: Target constraint count
- **Bias**: Size of generated bias
- **ViolQ**: Violation queries (Phase 2)
- **MQuQ**: MQuAcq queries (Phase 3)
- **TQ**: Total queries
- **P1T(s)**: **Phase 1 passive learning time (NEW!)**
- **VT(s)**: Phase 2 violation time
- **MQuT(s)**: Phase 3 active learning time
- **TT(s)**: Total time (P1T + VT + MQuT)

## Understanding the Results

### Phase 1 Time Analysis (P1T)
- **Lower solution count** → Faster Phase 1 (fewer examples to generate)
- **Higher solution count** → Slower Phase 1 (more examples to generate and validate)
- Look for: How much does P1T increase as solutions increase?

### Total Query Analysis (TQ = ViolQ + MQuQ)
- **Hypothesis**: More solutions → Better learning → Fewer queries needed
- Compare TQ across different solution counts for the same benchmark
- Lower TQ indicates more efficient constraint acquisition

### Time vs. Accuracy Tradeoff
- **P1T + VT + MQuT = TT**: Total pipeline time
- As solutions increase:
  - P1T increases (more examples to process)
  - VT and MQuT may decrease (better initial learning)
- Find optimal: Best TT or best TQ for acceptable accuracy

### Mock Constraint Impact
- **2 solutions + 50 mocks**: High noise, many invalid constraints (high InvC)
- **50 solutions + 2 mocks**: Clean learning, fewer invalid constraints (low InvC)
- Compare InvC/StartC ratio to measure noise level

## Customization

To modify the solution/mock mapping, edit the `calculate_mock_constraints()` function:

```python
def calculate_mock_constraints(num_solutions):
    mapping = {
        2: 50,   # Modify these values
        5: 20,
        10: 10,
        50: 2
    }
    return mapping.get(num_solutions, 10)
```

To test different solution counts, modify the `solution_configs` list:

```python
solution_configs = [2, 5, 10, 50]  # Add or remove values
```

## Notes

- Each configuration gets its own output directory to avoid conflicts
- Phase 2 and Phase 3 outputs go to standard directories (`phase2_output/`, `phase3_output/`)
- The script continues even if one configuration fails
- All timings are measured using `time.time()` for precision

## Interpretation Guide

### Hypothesis Testing
This experiment helps answer:
1. **Does more data lead to faster convergence?** 
   - Compare ViolQ and MQuQ across solution counts
   - Expected: Higher Sols → Lower TQ
   
2. **Do mock constraints hurt or help?** 
   - Compare InvC across configurations
   - High InvC indicates noisy/unhelpful mocks
   
3. **What's the optimal solution/mock ratio?** 
   - Find configuration with minimum TT(s) or TQ
   - Balance P1T cost vs. learning quality

### Key Metrics to Compare

**For each benchmark, compare across solution counts:**

| Metric | What to Look For | Interpretation |
|--------|------------------|----------------|
| P1T(s) | Linear increase? | Cost of more examples |
| InvC | Decreases with more Sols? | Mock quality improves |
| ViolQ | Decreases with more Sols? | Better Phase 1 learning |
| TQ | Minimum at which Sols? | Optimal data amount |
| TT(s) | Total cost | Best overall efficiency |

### Analysis Examples

**Example 1: P1T vs TQ Tradeoff**
```
Sudoku:  2 sols → P1T=8s, TQ=1620    (fast start, many queries)
Sudoku: 50 sols → P1T=78s, TQ=1068   (slow start, fewer queries)
Conclusion: 70s investment saves 552 queries
```

**Example 2: Mock Constraint Noise**
```
JSudoku:  2 sols, 50 mocks → InvC=25/37 (67% invalid!)
JSudoku: 50 sols,  2 mocks → InvC=5/37  (13% invalid)
Conclusion: More solutions drastically reduce noise
```

**Example 3: Finding the Sweet Spot**
```
Compare all 4 configurations for a benchmark:
 2 sols: TQ=1620, TT=473s
 5 sols: TQ=1333, TT=474s ← Best TQ with similar TT
10 sols: TQ=1195, TT=471s ← Best TT
50 sols: TQ=1068, TT=504s ← Diminishing returns
Conclusion: 5-10 solutions is optimal for this benchmark
```

