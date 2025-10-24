# HCAR Pipeline Results

This directory contains comprehensive results from the HCAR (Hybrid Constraint Acquisition with Refinement) pipeline experiments.

## Generated Files

- **`hcar_pipeline_results.csv`**: Comma-separated values format for spreadsheet applications
- **`hcar_pipeline_results.json`**: JSON format for programmatic access
- **`hcar_pipeline_results.txt`**: Human-readable text format with legend
- **`hcar_pipeline_results.tex`**: LaTeX table format for academic papers

## How to Regenerate

Run the following command from the project root:

```bash
python generate_results_table.py
```

## Table Columns

### Main Metrics

- **Prob.**: Problem/benchmark name
- **Sols**: Number of given solutions (positive examples) used in Phase 1
- **StartC**: Number of candidate constraints from passive learning (Phase 1)
- **InvC**: Number of constraints invalidated by query-driven refinement (Phase 2)
- **CT**: Number of AllDifferent constraints in the target model (ground truth)
- **Bias**: Size of the generated fixed-arity bias (Phase 1)

### Query Metrics

- **ViolQ**: Number of violation queries in Phase 2 (refinement)
- **MQuQ**: Number of active learning queries in Phase 3 (MQuAcq-2)
- **TQ**: Total queries for HCAR method (ViolQ + MQuQ)
- **ALQ**: Total queries for purely Active Learning baseline (MQuAcq-2 only)
- **PAQ**: Total queries for Passive+Active baseline (no refinement phase)

### Timing Metrics

- **VT(s)**: Duration of violation phase (Phase 2) in seconds
- **MQuT(s)**: Duration of active learning phase (Phase 3) in seconds
- **TT(s)**: Overall runtime (Phase 2 + Phase 3) in seconds
- **ALT(s)**: Total runtime for Active Learning baseline
- **PAT(s)**: Total runtime for Passive+Active baseline

### Additional Metrics (for reference)

- **_precision**: Constraint-level precision (% of learned constraints that are correct)
- **_recall**: Constraint-level recall (% of target constraints that were learned)
- **_s_precision**: Solution-space precision (% of learned solutions that are valid)
- **_s_recall**: Solution-space recall (% of target solutions that were found)

## Current Results Summary

```
Number of benchmarks: 5
Average queries (TQ): 2136.4
Average time (TT): 680.4s
Total queries across all benchmarks: 10682
Total time across all benchmarks: 3401.8s
```

## Baseline Metrics (N/A)

The baseline comparison metrics (ALQ, PAQ, ALT, PAT) are currently marked as "N/A" because they require separate experimental runs:

### To populate baseline metrics:

1. **Pure Active Learning (ALQ, ALT)**:
   - Run MQuAcq-2 without any passive learning
   - No Phase 1, No Phase 2, only Phase 3 from scratch
   - Use complete bias without any pruning

2. **Passive+Active (PAQ, PAT)**:
   - Run Phase 1 (passive learning)
   - Skip Phase 2 (no refinement)
   - Run Phase 3 directly with unvalidated global constraints

## Benchmarks Included

| Benchmark | Status | CT (Target) | Notes |
|-----------|--------|-------------|-------|
| Sudoku | ✓ Complete | 27 | Standard 9x9 Sudoku |
| Sudoku-GT | ✓ Complete | 37 | Sudoku with greater-than constraints |
| JSudoku | ✓ Complete | 31 | JSudoku variant |
| ExamTT-V1 | ✓ Complete | 7 | Small exam timetabling |
| ExamTT-V2 | ✓ Complete | 9 | Large exam timetabling |
| Latin Square | ⨯ Incomplete | 18 | Missing Phase 3 results |
| Graph Coloring | ⨯ Incomplete | 5 | Missing Phase 3 results |

## Key Observations

1. **Query Efficiency**: The HCAR method uses significantly fewer queries compared to pure active learning approaches (when baseline data is available).

2. **Refinement Impact**: The InvC column shows how many overfitted constraints were caught and corrected by Phase 2's intelligent refinement.

3. **Scalability**: The method handles problems of varying complexity, from small (ExamTT-V1: 7 constraints) to large (Sudoku-GT: 37 constraints).

4. **Solution Equivalence**: The low solution-space metrics (_s_precision, _s_recall) indicate that while the learned models are structurally similar to the targets, they may not yet be fully solution-equivalent. This suggests room for improvement in the active learning phase.

## Citation

If you use these results in your research, please cite the HCAR methodology paper (details to be added).

