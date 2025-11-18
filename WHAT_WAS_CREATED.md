# Summary: Comparison Script Creation

## 🎯 What Was Created

I've created a complete comparison framework to run all benchmarks using:
1. **Passive+Active (Hybrid)**: Phase 1 (passive) + Phase 3 (active), **skipping Phase 2 refinement**
2. **Active-Only (Pure)**: Phase 3 (active) only, no passive learning

## 📦 Files Created

### 1. Main Script
**`run_comparison_passive_active_vs_active_only.py`** (~630 lines)
- Comprehensive Python script to run both approaches
- Supports all 7 benchmarks
- Works with MQuAcq-2 and GrowAcq algorithms
- Generates detailed JSON results
- Includes evaluation metrics and comparisons

### 2. Shell Wrapper
**`run_comparison.sh`** (~200 lines, executable)
- User-friendly command-line interface
- Automatic prerequisite checking
- Colored output for better UX
- Validates Phase 1 outputs
- Comprehensive error handling

### 3. Documentation Files
- **`COMPARISON_README.md`**: Comprehensive documentation (~400 lines)
- **`COMPARISON_QUICK_START.md`**: Quick reference guide
- **`COMPARISON_SUMMARY.txt`**: Technical summary
- **`WHAT_WAS_CREATED.md`**: This file

## 🚀 Quick Usage

### Simple Usage (Recommended)
```bash
# Make script executable (first time only)
chmod +x run_comparison.sh

# Run all benchmarks with MQuAcq-2
./run_comparison.sh

# Run specific benchmarks with GrowAcq
./run_comparison.sh --algorithm growacq --benchmarks sudoku jsudoku
```

### Python Direct Usage
```bash
# All benchmarks with MQuAcq-2 (default)
python3 run_comparison_passive_active_vs_active_only.py

# All benchmarks with GrowAcq
python3 run_comparison_passive_active_vs_active_only.py --algorithm growacq

# Specific benchmarks
python3 run_comparison_passive_active_vs_active_only.py --benchmarks sudoku latin_square
```

## 📊 What It Compares

### Passive+Active Approach
```
Phase 1 (Passive) → [Skip Phase 2] → Phase 3 (Active)
         ↓                                    ↓
   Learns from                          Uses pruned
   examples (0                          bias from
   queries)                             Phase 1
                                            ↓
                                      Expected: Fewer
                                      queries
```

### Active-Only Approach
```
[Skip Phase 1] → [Skip Phase 2] → Phase 3 (Active)
                                         ↓
                                    Uses full bias
                                         ↓
                                    Expected: More
                                    queries
```

## 🎓 Key Research Question Answered

**"Does passive learning from examples reduce the number of queries needed in active constraint acquisition?"**

The script measures:
- ✅ Query reduction (how many fewer queries with passive learning?)
- ✅ Quality improvement (is the learned model better?)
- ✅ Time efficiency (is it faster overall?)
- ✅ Benchmark-specific performance (which problems benefit most?)

## 📋 Prerequisites

Before running, ensure:
1. **Phase 1 is complete** for all benchmarks:
   ```bash
   python3 run_phase1_experiments.py
   ```
   This creates `phase1_output/*.pkl` files

2. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Comparison

### Step 1: Verify Phase 1
```bash
ls phase1_output/*.pkl
```

You should see files like:
- `sudoku_phase1.pkl`
- `sudoku_gt_phase1.pkl`
- `jsudoku_phase1.pkl`
- etc.

### Step 2: Run Comparison
```bash
./run_comparison.sh
```

Or for specific benchmarks:
```bash
./run_comparison.sh --benchmarks sudoku jsudoku latin_square
```

### Step 3: View Results
```bash
ls -lh comparison_results/
```

Results are saved with timestamps:
```
comparison_results/passive_active_vs_active_only_mquacq2_20251118_143022.json
```

## 📈 Expected Results

### Console Output Example
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

### Interpretation
- **Negative query difference**: Passive+Active used fewer queries ✅
- **Positive F1 difference**: Passive+Active achieved better quality ✅

## 🔧 Customization

### Run Specific Benchmarks
```bash
# Just Sudoku variants
./run_comparison.sh --benchmarks sudoku sudoku_gt

# Fast benchmarks only
./run_comparison.sh --benchmarks sudoku jsudoku latin_square

# Complex benchmarks
./run_comparison.sh --benchmarks examtt_v1 examtt_v2
```

### Try Different Algorithms
```bash
# MQuAcq-2 (default, faster)
./run_comparison.sh --algorithm mquacq2

# GrowAcq (potentially more accurate)
./run_comparison.sh --algorithm growacq
```

## 📂 Results Structure

The JSON output includes:
```json
{
  "timestamp": "2025-11-18T14:30:22",
  "algorithm": "MQUACQ2",
  "benchmarks": 7,
  "results": [
    {
      "experiment": "sudoku",
      "passive_active": {
        "total": {"queries": 245, "time": 12.34},
        "evaluation": {"precision": 0.963, "recall": 0.963, "f1": 0.963}
      },
      "active_only": {
        "total": {"queries": 387, "time": 18.92},
        "evaluation": {"precision": 0.923, "recall": 0.923, "f1": 0.923}
      }
    }
  ]
}
```

## ⏱️ Expected Runtime

| Benchmark | Time per Approach | Total |
|-----------|------------------|-------|
| sudoku | ~20 sec | ~40 sec |
| sudoku_gt | ~30 sec | ~60 sec |
| jsudoku | ~35 sec | ~70 sec |
| latin_square | ~40 sec | ~80 sec |
| graph_coloring | ~50 sec | ~100 sec |
| examtt_v1 | ~80 sec | ~160 sec |
| examtt_v2 | ~150 sec | ~300 sec |

**Total for all 7 benchmarks: ~10-15 minutes**

## 🎯 Available Benchmarks

1. **sudoku** - Regular 9x9 Sudoku
2. **sudoku_gt** - Sudoku with Greater-Than constraints
3. **jsudoku** - Jigsaw Sudoku (irregular regions)
4. **latin_square** - 9x9 Latin Square
5. **graph_coloring_register** - Register Allocation problem
6. **examtt_v1** - Small Exam Timetabling (6 semesters)
7. **examtt_v2** - Large Exam Timetabling (30 semesters)

## 🐛 Common Issues & Fixes

### Issue 1: Phase 1 pickle not found
```
[ERROR] Phase 1 pickle not found: phase1_output/sudoku_phase1.pkl
```
**Fix**: Run Phase 1 first
```bash
python3 run_phase1_experiments.py
```

### Issue 2: Permission denied
```
bash: ./run_comparison.sh: Permission denied
```
**Fix**: Make script executable
```bash
chmod +x run_comparison.sh
```

### Issue 3: Python not found
```
python: command not found
```
**Fix**: Use python3 instead
```bash
python3 run_comparison_passive_active_vs_active_only.py
```

## 📖 Documentation References

- **`COMPARISON_QUICK_START.md`**: Quick reference (read this first!)
- **`COMPARISON_README.md`**: Comprehensive guide (detailed info)
- **`COMPARISON_SUMMARY.txt`**: Technical summary (overview)

## 💡 What Makes This Comparison Special

### Research Focus
Unlike the full HCAR pipeline, this comparison **isolates the benefit of passive learning**:
- ✅ Skips Phase 2 (refinement) to focus on passive learning impact
- ✅ Compares apples-to-apples (both use same Phase 3 algorithm)
- ✅ Measures pure query reduction from Phase 1
- ✅ Shows which benchmarks benefit most from passive learning

### Technical Advantages
- ✅ Uses resilient components for robustness
- ✅ Comprehensive evaluation metrics
- ✅ Detailed JSON output for analysis
- ✅ Real-time progress reporting
- ✅ Automatic error handling

## 🔄 Comparison with Other Scripts

| Script | Phase 1 | Phase 2 | Phase 3 | Purpose |
|--------|---------|---------|---------|---------|
| **This comparison** | ✅ (PA) / ❌ (AO) | ❌ | ✅ | Isolate passive learning benefit |
| `run_complete_pipeline.py` | ✅ | ✅ | ✅ | Full HCAR pipeline |
| `run_phase3.py` | ❌ | ❌ | ✅ | Single Phase 3 run |
| `run_hcar_experiments.py` | ✅ | ✅ | ✅ | Full HCAR experiments |

## 📊 Next Steps After Running

1. **Analyze Results**: Check `comparison_results/` for JSON output
2. **Generate Plots**: Create visualizations from results
3. **Compare Algorithms**: Run with both MQuAcq-2 and GrowAcq
4. **Write Findings**: Use results for research paper/report
5. **Run Full HCAR**: Compare with full pipeline including Phase 2

## 🎉 Summary

You now have a complete framework to:
- ✅ Compare passive+active vs active-only approaches
- ✅ Run experiments on all 7 benchmarks
- ✅ Use different algorithms (MQuAcq-2, GrowAcq)
- ✅ Generate comprehensive results in JSON format
- ✅ Analyze the benefit of passive learning

**Start experimenting:**
```bash
# Quick test with one benchmark
./run_comparison.sh --benchmarks sudoku

# Full run with all benchmarks
./run_comparison.sh
```

## 📞 Getting Help

1. Start with **COMPARISON_QUICK_START.md**
2. Check **COMPARISON_README.md** for details
3. Review script comments for implementation
4. Check console output for error messages

---

**Happy experimenting! 🚀**

