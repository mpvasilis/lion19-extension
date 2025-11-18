# Quick Start: Passive+Active vs Active-Only Comparison

This guide helps you quickly run comparison experiments between two constraint acquisition approaches.

## 📋 What You're Comparing

1. **Passive+Active (Hybrid)**: Phase 1 (passive) → Phase 3 (active), **skipping Phase 2**
2. **Active-Only (Pure)**: Phase 3 (active) only, no passive learning

## 🚀 Quick Start (3 Steps)

### Step 1: Ensure Phase 1 is complete

```bash
# Check if Phase 1 outputs exist
ls phase1_output/*.pkl

# If missing, run Phase 1
python run_phase1_experiments.py
```

### Step 2: Run the comparison

**Option A: Using the shell script (easiest)**
```bash
# Run all benchmarks with MQuAcq-2
./run_comparison.sh

# Run all benchmarks with GrowAcq
./run_comparison.sh --algorithm growacq

# Run specific benchmarks
./run_comparison.sh --benchmarks sudoku jsudoku latin_square
```

**Option B: Using Python directly**
```bash
# Run all benchmarks with MQuAcq-2
python run_comparison_passive_active_vs_active_only.py

# Run all benchmarks with GrowAcq
python run_comparison_passive_active_vs_active_only.py --algorithm growacq

# Run specific benchmarks
python run_comparison_passive_active_vs_active_only.py --benchmarks sudoku jsudoku
```

### Step 3: View results

Results are saved to `comparison_results/` directory:
```bash
# List result files
ls -lh comparison_results/

# View latest results (requires jq)
cat comparison_results/passive_active_vs_active_only_mquacq2_*.json | jq '.'
```

## 📊 Expected Results

The comparison will show:
- **Query counts** for both approaches
- **Learning time** for each approach
- **Model quality** metrics (precision, recall, F1)
- **Comparative analysis** showing which approach performed better

Example output:
```
COMPARISON SUMMARY: sudoku
================================================================================

Approach             Queries      Time (s)     F1-Score    
--------------------------------------------------------
Passive+Active       245          12.34        95.50%      
Active-Only          387          18.92        92.30%      
--------------------------------------------------------

Passive+Active vs Active-Only:
  Query difference: -142 (-36.7%)  ← Passive+Active used 36.7% fewer queries!
  F1 difference: +3.20%            ← Passive+Active achieved 3.20% better quality!
```

## 🎯 Understanding the Results

### Query Reduction (Negative = Better for Passive+Active)
```
Query difference: -142 (-36.7%)
```
✅ Negative value means Passive+Active required **fewer queries**

### Quality Improvement (Positive = Better for Passive+Active)
```
F1 difference: +3.20%
```
✅ Positive value means Passive+Active achieved **better model quality**

## 📁 Files Created

| File | Description |
|------|-------------|
| `run_comparison_passive_active_vs_active_only.py` | Main comparison script |
| `run_comparison.sh` | Convenient shell wrapper |
| `COMPARISON_README.md` | Detailed documentation |
| `COMPARISON_QUICK_START.md` | This file |

## 🏃 Running Individual Benchmarks

### All Benchmarks (Default)
```bash
./run_comparison.sh
```

Runs: sudoku, sudoku_gt, jsudoku, latin_square, graph_coloring_register, examtt_v1, examtt_v2

### Specific Benchmarks
```bash
# Fast benchmarks only
./run_comparison.sh --benchmarks sudoku jsudoku latin_square

# Just Sudoku variants
./run_comparison.sh --benchmarks sudoku sudoku_gt

# Large benchmarks only
./run_comparison.sh --benchmarks examtt_v2
```

## 🔧 Algorithms

### MQuAcq-2 (Default, Recommended)
```bash
./run_comparison.sh --algorithm mquacq2
```
- Faster
- More predictable
- Good for most benchmarks

### GrowAcq (Alternative)
```bash
./run_comparison.sh --algorithm growacq
```
- Potentially more accurate
- Slower
- Better for complex problems

## ⏱️ Expected Runtime

| Benchmark | Passive+Active | Active-Only | Total Time |
|-----------|---------------|-------------|------------|
| sudoku | ~15 sec | ~25 sec | ~40 sec |
| sudoku_gt | ~20 sec | ~35 sec | ~55 sec |
| jsudoku | ~25 sec | ~40 sec | ~65 sec |
| latin_square | ~30 sec | ~50 sec | ~80 sec |
| graph_coloring | ~40 sec | ~60 sec | ~100 sec |
| examtt_v1 | ~60 sec | ~90 sec | ~150 sec |
| examtt_v2 | ~120 sec | ~180 sec | ~300 sec |

**Total for all benchmarks**: ~10-15 minutes

## 🐛 Troubleshooting

### Error: Phase 1 pickle not found
```
[ERROR] Phase 1 pickle not found: phase1_output/sudoku_phase1.pkl
```
**Fix**: Run Phase 1 first
```bash
python run_phase1_experiments.py
```

### Error: Permission denied on run_comparison.sh
```
bash: ./run_comparison.sh: Permission denied
```
**Fix**: Make script executable
```bash
chmod +x run_comparison.sh
```

### Error: Module not found
```
ModuleNotFoundError: No module named 'cpmpy'
```
**Fix**: Install dependencies
```bash
pip install -r requirements.txt
```

## 📖 More Information

For detailed documentation, see:
- **COMPARISON_README.md** - Full documentation with examples and troubleshooting
- **Script source code** - Well-commented Python code in `run_comparison_passive_active_vs_active_only.py`

## 💡 Tips

1. **Start small**: Test with one benchmark first
   ```bash
   ./run_comparison.sh --benchmarks sudoku
   ```

2. **Use MQuAcq-2 first**: It's faster and good for initial experiments
   ```bash
   ./run_comparison.sh --algorithm mquacq2
   ```

3. **Check Phase 1**: Ensure Phase 1 ran successfully before comparison
   ```bash
   ls -lh phase1_output/*.pkl
   ```

4. **Monitor progress**: The script provides detailed console output

5. **Save results**: Results are automatically saved to `comparison_results/`

## 🎓 What This Tells You

This comparison helps answer:
- ✅ **Does passive learning help?** (Compare query counts)
- ✅ **Is the quality better?** (Compare F1 scores)
- ✅ **What's the trade-off?** (Compare time vs accuracy)
- ✅ **Which approach for which problem?** (Benchmark-specific analysis)

## 🚦 Next Steps

After running the comparison:
1. Analyze the results in `comparison_results/`
2. Compare with full HCAR pipeline (including Phase 2)
3. Try different algorithms (MQuAcq-2 vs GrowAcq)
4. Experiment with different benchmarks
5. Generate visualizations from the JSON results

## 📞 Need Help?

If you encounter issues:
1. Check Phase 1 outputs exist
2. Review console output for errors
3. Check `comparison_results/` for partial results
4. Consult COMPARISON_README.md for detailed troubleshooting
5. Review the Python script comments

---

**Happy experimenting! 🧪**

