# Passive+Active vs Active-Only Comparison

This document explains how to run the comparison experiments between two constraint acquisition approaches:

1. **Passive+Active (Hybrid)**: Uses passive learning (Phase 1) followed directly by active learning (Phase 3), **skipping the Phase 2 refinement step**
2. **Active-Only (Pure)**: Uses only active learning (Phase 3) with no passive initialization

## Overview

The comparison script `run_comparison_passive_active_vs_active_only.py` runs both approaches on all benchmarks and compares:
- Total queries required
- Learning time
- Model quality (precision, recall, F1-score)
- Correctness metrics

## Quick Start

### Prerequisites

Ensure Phase 1 has been run for all benchmarks:

```bash
python run_phase1_experiments.py
```

This will create Phase 1 output files in `phase1_output/` directory.

### Running the Comparison

**Run all benchmarks with MQuAcq-2:**
```bash
python run_comparison_passive_active_vs_active_only.py
```

**Run all benchmarks with GrowAcq:**
```bash
python run_comparison_passive_active_vs_active_only.py --algorithm growacq
```

**Run specific benchmarks:**
```bash
python run_comparison_passive_active_vs_active_only.py --benchmarks sudoku jsudoku latin_square
```

**Run with GrowAcq on specific benchmarks:**
```bash
python run_comparison_passive_active_vs_active_only.py --algorithm growacq --benchmarks sudoku sudoku_gt
```

## Available Benchmarks

1. `sudoku` - Regular 9x9 Sudoku
2. `sudoku_gt` - Sudoku with Greater-Than constraints
3. `jsudoku` - Jigsaw Sudoku (9x9 with irregular regions)
4. `latin_square` - Latin Square (9x9)
5. `graph_coloring_register` - Graph Coloring (Register Allocation)
6. `examtt_v1` - Exam Timetabling Variant 1 (Small)
7. `examtt_v2` - Exam Timetabling Variant 2 (Large)

## Output

### Console Output

The script provides detailed console output showing:
- Progress for each benchmark
- Real-time metrics for both approaches
- Comparison summary after each benchmark
- Final aggregate statistics

Example output:
```
================================================================================
COMPARISON SUMMARY: sudoku
================================================================================

Approach             Queries      Time (s)     F1-Score    
--------------------------------------------------------
Passive+Active       245          12.34        95.50%      
Active-Only          387          18.92        92.30%      
--------------------------------------------------------

Passive+Active vs Active-Only:
  Query difference: -142 (-36.7%)
  F1 difference: +3.20%
```

### Results File

Results are saved to `comparison_results/` directory:

```
comparison_results/
  └── passive_active_vs_active_only_mquacq2_20251118_143022.json
```

The JSON file contains:
- Timestamp
- Algorithm used
- Detailed results for each benchmark
- Evaluation metrics (precision, recall, F1)
- Query counts and timing information

### Results Structure

```json
{
  "timestamp": "2025-11-18T14:30:22",
  "algorithm": "MQUACQ2",
  "benchmarks": 7,
  "results": [
    {
      "experiment": "sudoku",
      "passive_active": {
        "approach": "Passive+Active",
        "phase1": {
          "bias_size": 1944
        },
        "phase3": {
          "queries": 245,
          "time": 12.34,
          "learned_size": 27
        },
        "total": {
          "queries": 245,
          "time": 12.34
        },
        "evaluation": {
          "precision": 0.9630,
          "recall": 0.9630,
          "f1": 0.9630
        }
      },
      "active_only": {
        "approach": "Active-Only",
        "phase3": {
          "queries": 387,
          "time": 18.92,
          "learned_size": 26
        },
        "total": {
          "queries": 387,
          "time": 18.92
        },
        "evaluation": {
          "precision": 0.9231,
          "recall": 0.9231,
          "f1": 0.9231
        }
      }
    },
    // ... more benchmarks
  ]
}
```

## Understanding the Approaches

### Passive+Active (Hybrid)

1. **Phase 1 (Passive Learning)**: 
   - Learns from positive examples (solved instances)
   - Generates initial bias constraints (B_fixed)
   - Detects patterns without querying
   - **Queries: 0**

2. **Phase 2 (Skipped)**:
   - Interactive refinement is **not used** in this comparison
   - Goes directly from Phase 1 to Phase 3

3. **Phase 3 (Active Learning)**:
   - Uses MQuAcq-2 or GrowAcq with pruned bias from Phase 1
   - Starts with empty CL (no validated globals since Phase 2 is skipped)
   - Actively queries to learn constraints

**Total Queries**: Phase 3 queries only (Phase 1 = 0 queries)

### Active-Only (Pure)

1. **Phase 3 (Active Learning)**:
   - Uses MQuAcq-2 or GrowAcq with full bias
   - Starts with empty CL
   - No passive initialization
   - Actively queries to learn all constraints from scratch

**Total Queries**: Phase 3 queries

## Key Differences

| Aspect | Passive+Active | Active-Only |
|--------|---------------|-------------|
| **Phase 1** | ✅ Used | ❌ Skipped |
| **Phase 2** | ❌ Skipped | ❌ Skipped |
| **Phase 3** | ✅ Used | ✅ Used |
| **Initial Bias** | Pruned from Phase 1 | Full bias |
| **Query Cost** | Phase 3 only | Phase 3 only |
| **Expected Queries** | Fewer (pruned bias) | More (full bias) |

## Interpreting Results

### Query Reduction

A **negative** query difference means Passive+Active required fewer queries:
```
Query difference: -142 (-36.7%)
```
This indicates passive learning helped reduce the search space by 36.7%.

### F1-Score Improvement

A **positive** F1 difference means Passive+Active achieved better model quality:
```
F1 difference: +3.20%
```

### When Passive+Active Wins

Passive+Active typically performs better when:
- Examples reveal useful patterns
- Bias can be significantly pruned
- Target constraints have recognizable structures

### When Active-Only Might Win

Active-Only might perform better when:
- Examples are misleading
- Passive learning introduces noise
- Problem structure is highly irregular

## Advanced Usage

### Custom Timeout and Limits

The script uses default values from the resilient components:
- FindC timeout: 1 second
- Query generation timeout: 2 seconds

To modify these, edit the `run_active_learning()` function in the script.

### Modifying Benchmarks

To add or remove benchmarks, edit the `benchmarks` list in the `main()` function:

```python
benchmarks = [
    {
        'name': 'sudoku',
        'phase1_pickle': 'phase1_output/sudoku_phase1.pkl'
    },
    # Add more benchmarks here
]
```

### Using Different Algorithms

The script supports two active learning algorithms:
- `mquacq2` (default): MQuAcq-2 algorithm
- `growacq`: GrowAcq algorithm with inner MQuAcq-2

```bash
python run_comparison_passive_active_vs_active_only.py --algorithm growacq
```

## Troubleshooting

### Error: Phase 1 pickle not found

```
[ERROR] Phase 1 pickle not found: phase1_output/sudoku_phase1.pkl
```

**Solution**: Run Phase 1 first:
```bash
python run_phase1_experiments.py
```

### Error: Missing benchmark constructor

If you see errors about missing constructors, ensure:
1. The benchmark name is correct
2. The benchmark constructors exist in both `benchmarks/` and `benchmarks_global/`

### Memory Issues

For large benchmarks (e.g., examtt_v2), you may encounter memory issues. Consider:
1. Running benchmarks individually
2. Reducing the number of variables in the problem
3. Increasing available RAM

## Notes on Phase 2 Exclusion

This comparison **excludes Phase 2** to focus on:
- The pure benefit of passive learning (Phase 1)
- Direct comparison with no refinement step
- Measuring the impact of bias pruning alone

If you want to include Phase 2 refinement:
1. Run the full HCAR pipeline: `python run_complete_pipeline.py`
2. Compare HCAR (Phase 1 + 2 + 3) vs Active-Only

## Comparison with Full HCAR Pipeline

| Approach | Phase 1 | Phase 2 | Phase 3 | Query Source |
|----------|---------|---------|---------|--------------|
| **Passive+Active** (this script) | ✅ | ❌ | ✅ | Phase 3 only |
| **Active-Only** (this script) | ❌ | ❌ | ✅ | Phase 3 only |
| **Full HCAR** | ✅ | ✅ | ✅ | Phase 2 + 3 |

## Related Scripts

- `run_phase1_experiments.py` - Run Phase 1 (passive learning) for all benchmarks
- `run_phase2_experiments.py` - Run Phase 2 (refinement) for all benchmarks
- `run_phase3.py` - Run Phase 3 (active learning) for a single benchmark
- `run_complete_pipeline.py` - Run full HCAR pipeline (Phase 1 + 2 + 3)

## Citation

If you use this comparison in your research, please cite the relevant papers:
- LION19 paper (for passive learning approach)
- QuAcq/MQuAcq-2 papers (for active learning)
- GrowAcq paper (if using GrowAcq algorithm)

## Questions or Issues?

If you encounter any issues or have questions about the comparison:
1. Check the console output for detailed error messages
2. Verify Phase 1 outputs exist
3. Check the results JSON file for partial results
4. Review the source code comments for implementation details

