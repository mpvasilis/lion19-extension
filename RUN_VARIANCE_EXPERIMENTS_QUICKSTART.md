# Solution Variance Experiments - Quick Start

## Overview
Test how solution count affects learning performance with inverse mock constraint relationship.

## Configuration Matrix

| Solutions | Mock Constraints | Rationale |
|-----------|------------------|-----------|
| 2         | 50               | Low data → Need more exploration |
| 5         | 20               | Baseline (current pipeline default) |
| 10        | 10               | Balanced approach |
| 50        | 2                | High data → Minimal noise |

## Run Command

```bash
python run_solution_variance_experiments.py
```

**No arguments needed!** The script automatically:
- Tests 8 benchmarks × 4 configurations = 32 complete pipelines
- Measures Phase 1 timing precisely
- Collects all standard HCAR metrics
- Generates formatted results

## Output Files

All files saved to `solution_variance_output/`:

1. **`variance_results.txt`** - Human-readable table (HCAR format)
2. **`variance_results.csv`** - CSV for data analysis
3. **`variance_experiment_detailed.json`** - Complete raw data

## Result Format

```
Prob.           Sols   StartC   InvC   CT    Bias   ViolQ  MQuQ   TQ    P1T(s)  VT(s)   MQuT(s)  TT(s)
==================================================================================================
Sudoku          2      37       10     27    2291   120    1500   1620  8.45    320.15  145.30   473.90
Sudoku          5      37       10     27    2291   101    1232   1333  15.32   317.69  141.14   474.15
Sudoku          10     37       10     27    2291   95     1100   1195  25.67   310.22  135.80   471.69
Sudoku          50     37       10     27    2291   88     980    1068  78.12   298.45  128.33   504.90
```

## Key Columns Explained

| Column | Meaning |
|--------|---------|
| **Sols** | Number of solutions (2, 5, 10, 50) |
| **StartC** | Initial candidate constraints |
| **InvC** | Invalid constraints removed |
| **P1T(s)** | **Phase 1 time - THE NEW METRIC!** |
| **TQ** | Total queries (lower = better) |
| **TT(s)** | Total time (P1T + VT + MQuT) |

## What to Look For

### 1. Phase 1 Scaling
**Question**: How does P1T grow with more solutions?
```
2 sols  → P1T = ~8s
5 sols  → P1T = ~15s  
10 sols → P1T = ~25s  
50 sols → P1T = ~78s  
```
**Expected**: Roughly linear or sub-linear growth

### 2. Query Reduction
**Question**: Do more solutions reduce total queries?
```
Compare TQ across solution counts:
2 sols  → TQ = 1620  (baseline)
50 sols → TQ = 1068  (34% reduction!)
```
**Expected**: TQ decreases as Sols increases

### 3. Mock Constraint Quality
**Question**: Do more solutions reduce invalid constraints?
```
InvC / StartC ratio:
2 sols, 50 mocks  → High InvC (noise)
50 sols, 2 mocks  → Low InvC (clean)
```
**Expected**: InvC decreases as Sols increases

### 4. Optimal Configuration
**Question**: Which configuration is best?
- **Best TQ**: Likely 50 solutions (most data)
- **Best TT**: Likely 5-10 solutions (good balance)
- **Best efficiency**: Compare TQ/TT ratio

## Quick Analysis

After running, open `variance_results.csv` in Excel/Python and plot:

1. **P1T vs Sols** → Understand scaling cost
2. **TQ vs Sols** → Find query reduction trend
3. **InvC vs Sols** → Validate mock constraint hypothesis
4. **TT vs Sols** → Identify optimal configuration

## Estimated Runtime

- **Per configuration**: 5-30 minutes
- **Per benchmark**: 20-120 minutes (4 configs)
- **Total experiment**: 3-16 hours (8 benchmarks)

💡 **Tip**: Run overnight or on a server

## Troubleshooting

**If a configuration fails:**
- Check `phase2_output/*.log` for errors
- Script continues to next configuration
- Missing configs will show in final statistics

**If no metrics collected:**
- Ensure Phase 1, 2, and 3 all complete
- Check that `phase3_output/*_phase3_results.json` exists
- Verify pickle files in `solution_variance_output/<bench>_sol<N>_mock<M>/`

## Next Steps

After results are generated:
1. Review `variance_results.txt` for quick overview
2. Analyze `variance_results.csv` for detailed trends
3. Compare against baseline (5 solutions from original pipeline)
4. Identify optimal configuration for each benchmark
5. Consider running more targeted experiments on interesting ranges

## Full Documentation

See `SOLUTION_VARIANCE_EXPERIMENTS.md` for:
- Detailed methodology
- Comprehensive interpretation guide
- Analysis examples
- Customization options

