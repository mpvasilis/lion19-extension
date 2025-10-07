# HCAR: Hybrid Constraint Acquisition with Intelligent Refinement

Implementation of the advanced HCAR framework from the paper:
**"A Principled Framework for Interactive Refinement in Hybrid Constraint Acquisition"**

## 📋 Overview

This implementation provides a complete, production-ready version of the HCAR methodology with:

- ✅ **Three-phase architecture** (Passive → Interactive Refinement → Active Learning)
- ✅ **Intelligent Subset Exploration** with data-driven culprit scores
- ✅ **ML-based prior estimation** using XGBoost
- ✅ **Principled information flow** preventing catastrophic information loss
- ✅ **Bayesian confidence updates** with noise handling
- ✅ **Comprehensive experimental framework** matching paper's evaluation

## 🗂️ File Structure

```
lion19-extension/
├── hcar_advanced.py          # Core HCAR framework implementation
├── run_hcar_experiments.py   # Experimental runner and evaluation
├── HCAR_README.md            # This file
├── benchmarks/               # Your existing benchmark problems
├── oracles.py                # Your existing oracle implementations
└── utils.py                  # Your existing utilities
```

## 🚀 Quick Start

### Installation

First, ensure you have the required dependencies:

```bash
pip install numpy scikit-learn xgboost cpmpy
```

Or add to your `requirements.txt`:

```txt
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
cpmpy>=0.9.0
```

### Basic Usage

```python
from hcar_advanced import HCARFramework, HCARConfig, OracleResponse

# 1. Configure HCAR
config = HCARConfig(
    total_budget=500,           # Total query budget
    max_time_seconds=1800,      # 30 minutes max
    theta_min=0.15,             # Rejection threshold
    theta_max=0.85,             # Acceptance threshold
    max_subset_depth=3,         # Subset exploration depth
    enable_ml_prior=True        # Use ML for prior estimation
)

# 2. Create HCAR instance
hcar = HCARFramework(config, problem_name="MyProblem")

# 3. Define oracle function
def my_oracle(assignment: dict) -> OracleResponse:
    """Check if assignment satisfies target model."""
    if is_valid_solution(assignment):
        return OracleResponse.VALID
    else:
        return OracleResponse.INVALID

# 4. Run HCAR
learned_model, metrics = hcar.run(
    positive_examples=initial_examples,  # List of 5 example solutions
    oracle_func=my_oracle,
    variables=problem_variables,         # CPMpy variables dict
    domains=variable_domains,            # Domain dict
    target_model=ground_truth            # Optional, for evaluation
)

# 5. Analyze results
print(f"Learned {len(learned_model)} constraints")
print(f"Used {metrics['queries_total']} queries")
print(f"Time: {metrics['time_seconds']:.2f}s")
```

## 🔬 Running Experiments (Paper Replication)

### Run Full Experimental Comparison

Replicate Table 1 from the paper:

```bash
# Run all benchmarks × all methods with 3 repetitions
python run_hcar_experiments.py --all --runs 3
```

This will:
- Run 5 benchmarks (Sudoku, UEFA, VM Allocation, Exam Timetabling, Nurse Rostering)
- Compare 3 method variants (HCAR-Advanced, HCAR-Heuristic, HCAR-NoRefine)
- Generate comprehensive results in `hcar_results/`
- Print comparison table matching paper format

### Run Single Experiment

```bash
# Test HCAR-Advanced on Sudoku
python run_hcar_experiments.py --benchmark Sudoku --method HCAR-Advanced

# Compare all methods on UEFA benchmark
python run_hcar_experiments.py --benchmark UEFA --compare --runs 3
```

### Method Variants

1. **HCAR-Advanced** (Proposed Method)
   - Full intelligent subset exploration
   - ML-based prior estimation
   - All advanced features enabled

2. **HCAR-Heuristic** (Baseline)
   - Simple positional heuristics for subset exploration
   - No ML priors
   - From original LION19 paper

3. **HCAR-NoRefine** (Ablation)
   - Skips Phase 2 entirely
   - Demonstrates impact of over-fitting
   - Shows necessity of refinement

4. **MQuAcq-2** (External Baseline)
   - Pure active learning
   - No passive phase
   - May timeout on complex problems

## 📊 Output and Metrics

### Results Structure

Each experiment produces:

```json
{
  "queries_phase2_mean": 95.0,
  "queries_phase3_mean": 87.0,
  "queries_total_mean": 182.0,
  "time_seconds_mean": 58.3,
  "s_precision_mean": 100.0,
  "s_recall_mean": 100.0,
  "num_global_constraints_mean": 15.0,
  "num_fixed_constraints_mean": 12.0,
  ...
}
```

### Evaluation Metrics

**Model Quality:**
- `s_precision`: Solution-space Precision (%)
- `s_recall`: Solution-space Recall (%)

**Efficiency:**
- `queries_phase2` (Q₂): Queries in refinement phase
- `queries_phase3` (Q₃): Queries in active learning phase
- `queries_total` (Q_Σ): Total oracle queries

**Performance:**
- `time_seconds`: Wall-clock time
- `num_global_constraints`: Learned global constraints
- `num_fixed_constraints`: Learned fixed-arity constraints

## 🔧 Integration with Existing Code

### Adapting Your Benchmarks

To use HCAR with your existing benchmarks, create a wrapper:

```python
# In run_hcar_experiments.py or a new file

from benchmarks.sudoku import SudokuBenchmark  # Your existing benchmark
from hcar_advanced import OracleResponse

def create_sudoku_hcar_setup():
    """Prepare Sudoku benchmark for HCAR."""
    
    # 1. Load your benchmark
    benchmark = SudokuBenchmark(size=9)
    
    # 2. Get problem components
    variables = benchmark.get_variables()      # CPMpy variables
    domains = benchmark.get_domains()          # Variable domains
    target_model = benchmark.get_constraints() # Ground truth
    
    # 3. Generate initial examples
    positive_examples = []
    for _ in range(5):
        solution = benchmark.generate_solution()
        positive_examples.append(solution)
    
    # 4. Create oracle function
    def oracle_func(assignment: dict) -> OracleResponse:
        is_valid = benchmark.check_solution(assignment)
        return OracleResponse.VALID if is_valid else OracleResponse.INVALID
    
    return {
        'variables': variables,
        'domains': domains,
        'target_model': target_model,
        'oracle_func': oracle_func,
        'positive_examples': positive_examples
    }
```

### Using Custom Constraint Extractors

To integrate your existing passive learning code:

```python
from hcar_advanced import HCARFramework
from feature_extraction import extract_global_constraints  # Your existing code

class CustomHCAR(HCARFramework):
    """HCAR with custom Phase 1 extraction."""
    
    def _phase1_passive_generation(self, positive_examples, variables, domains):
        """Override to use your existing extraction methods."""
        
        # Use your existing pattern detection
        self.B_globals = extract_global_constraints(
            positive_examples,
            variables,
            domains
        )
        
        # Use your existing bias generation
        self.B_fixed = generate_fixed_bias(
            variables,
            domains,
            positive_examples
        )
        
        # Continue with HCAR's ML and budget allocation
        if self.config.enable_ml_prior:
            self._initialize_ml_priors()
        self._allocate_uncertainty_budget()
```

## 🧪 Key Algorithm Components

### 1. Intelligent Subset Exploration

The core innovation - replaces heuristics with data-driven culprit scores:

```python
from hcar_advanced import IntelligentSubsetExplorer

explorer = IntelligentSubsetExplorer()

# When a constraint is refuted, generate informed subsets
new_candidates = explorer.generate_informed_subsets(
    rejected_constraint=over_fitted_constraint,
    positive_examples=confirmed_solutions,
    learned_globals=validated_constraints,
    config=hcar_config
)

# Returns constraints with most likely culprit variable removed
```

**Culprit Score Components:**
1. **Structural Isolation** (0.4 weight): Edit distance from other variables
2. **Weak Constraint Support** (0.3 weight): Low participation in other constraints
3. **Value Pattern Deviation** (0.3 weight): Statistical anomalies in examples

### 2. ML Prior Estimation

Uses XGBoost to estimate initial constraint probabilities:

```python
from hcar_advanced import MLPriorEstimator, FeatureExtractor

# Extract features
features = FeatureExtractor.extract_features(constraint, problem_context)

# Estimate prior
ml_prior = MLPriorEstimator(config)
prior_probability = ml_prior.estimate_prior(constraint)
```

**Features Used:**
- Arity (raw and normalized)
- Constraint type (AllDifferent, Sum, Count, Other)
- Variable name patterns (sequential, row, column, block)
- Subset level (derived vs. original)

### 3. Bayesian Confidence Updates

Principled probabilistic updates with noise handling:

```python
from hcar_advanced import BayesianUpdater

# Update after oracle feedback
new_confidence = BayesianUpdater.update_confidence(
    current_prob=0.5,
    query=generated_query,
    response=oracle_response,
    constraint=candidate,
    alpha=0.1  # Noise parameter
)
```

**Update Rules:**
- **Invalid response**: Positive evidence → Bayesian boost
- **Valid response**: Hard refutation → P(c) = 0

### 4. Query Generation

Targeted queries to stress-test constraints:

```python
from hcar_advanced import QueryGenerator

generator = QueryGenerator(config)

# Generate query that violates target but satisfies others
query, status = generator.generate_query(
    target_constraint=constraint_to_test,
    validated_constraints=accepted_constraints,
    candidate_constraints=other_candidates,
    variables=problem_variables,
    domains=variable_domains
)

# Status: SUCCESS, UNSAT, TIMEOUT, or ERROR
```

## 📈 Expected Results

Based on paper experiments (5 initial examples):

| Benchmark | Method | S-Prec | S-Rec | Q₂ | Q₃ | Q_Σ | T(s) |
|-----------|--------|--------|-------|-----|-----|-----|------|
| Sudoku | HCAR-Advanced | 100% | 100% | 38 | 153 | 191 | 45.2 |
| | HCAR-Heuristic | 100% | 100% | 45 | 153 | 198 | 48.9 |
| | HCAR-NoRefine | 100% | 81% | -- | 285 | 285 | 35.1 |
| | MQuAcq-2 | 100% | 100% | -- | -- | 6844 | 653.8 |
| UEFA | HCAR-Advanced | 100% | 100% | 55 | 18 | 73 | 21.5 |
| VM Alloc | HCAR-Advanced | 100% | 100% | 110 | 67 | 177 | 65.7 |
| Nurse | HCAR-Advanced | 100% | 100% | 95 | 87 | 182 | 58.3 |

**Key Observations:**
- HCAR-Advanced consistently outperforms HCAR-Heuristic (24-27% fewer queries on complex problems)
- HCAR-NoRefine has catastrophic recall (39-81%), confirming need for refinement
- Hybrid approach uses orders of magnitude fewer queries than pure active learning

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Missing dependencies
pip install numpy scikit-learn xgboost cpmpy

# CPMpy solver issues
pip install ortools  # OR-Tools backend
```

**2. Oracle Implementation**

Your oracle function must return `OracleResponse` enum:

```python
from hcar_advanced import OracleResponse

def my_oracle(assignment):
    # ❌ Wrong: return True/False
    # ✅ Correct:
    return OracleResponse.VALID if is_valid else OracleResponse.INVALID
```

**3. Variable/Domain Format**

Ensure CPMpy-compatible format:

```python
from cpmpy import intvar

# Create variables
variables = {
    'x_0': intvar(1, 9, name='x_0'),
    'x_1': intvar(1, 9, name='x_1'),
    ...
}

# Domains
domains = {
    'x_0': list(range(1, 10)),
    'x_1': list(range(1, 10)),
    ...
}
```

**4. Timeout Issues**

Adjust configuration for faster experiments:

```python
config = HCARConfig(
    total_budget=100,         # Reduce budget
    max_time_seconds=600,     # 10 minutes
    query_timeout=10.0,       # Faster per-query timeout
    max_subset_depth=2        # Less deep exploration
)
```

## 📚 Paper Citation

If you use this implementation, please cite:

```bibtex
@article{balafas2025hcar,
  title={A Principled Framework for Interactive Refinement in Hybrid Constraint Acquisition},
  author={Balafas, [Your Name]},
  journal={[Journal Name]},
  year={2025}
}
```

## 🔗 Related Files in Your Project

- `bayesian_ca_env.py` - Bayesian CA environment (may have overlapping components)
- `bayesian_quacq.py` - Bayesian QuAcq implementation
- `enhanced_bayesian_pqgen.py` - Enhanced query generation
- `feature_extraction.py` - Pattern-based constraint extraction (integrate with Phase 1)
- `oracles.py` - Oracle implementations (compatible with HCAR)
- `utils.py` - Utility functions (may be useful for HCAR)
- `benchmark_runner.py` - Existing experiment runner (can be extended)

## 🎯 Next Steps

1. **Test on Simple Benchmark**
   ```bash
   python run_hcar_experiments.py --benchmark Sudoku --method HCAR-Advanced
   ```

2. **Integrate Your Benchmarks**
   - Adapt `BenchmarkConfig` in `run_hcar_experiments.py`
   - Connect to your existing benchmark modules
   - Test oracle functions

3. **Run Full Comparison**
   ```bash
   python run_hcar_experiments.py --all --runs 3
   ```

4. **Analyze Results**
   - Results saved to `hcar_results/`
   - Generate LaTeX tables
   - Compare with paper results

5. **Customize for Your Domain**
   - Add domain-specific features to `FeatureExtractor`
   - Train ML model on your historical data
   - Adjust culprit score weights

## 💡 Advanced Customization

### Custom Culprit Score Metrics

Add domain-specific culprit scoring:

```python
from hcar_advanced import IntelligentSubsetExplorer

class CustomSubsetExplorer(IntelligentSubsetExplorer):
    
    @staticmethod
    def _calculate_culprit_score(variable, scope, constraint, examples, learned):
        # Start with base scores
        base_score = super()._calculate_culprit_score(
            variable, scope, constraint, examples, learned
        )
        
        # Add domain-specific scoring
        domain_score = my_custom_metric(variable, scope)
        
        return 0.7 * base_score + 0.3 * domain_score
```

### Custom ML Features

Extend feature extraction for your domain:

```python
from hcar_advanced import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    
    @staticmethod
    def extract_features(constraint, problem_context):
        # Get base features
        features = super().extract_features(constraint, problem_context)
        
        # Add custom features
        features['my_custom_feature'] = calculate_custom_feature(constraint)
        features['domain_specific_metric'] = get_domain_metric(constraint)
        
        return features
```

### Offline ML Training

Train the ML prior on your historical benchmark data:

```python
from hcar_advanced import MLPriorEstimator

# Prepare training data
training_data = []
for benchmark in historical_benchmarks:
    for constraint in benchmark.all_constraints:
        features = extract_features(constraint)
        is_valid = constraint in benchmark.target_model
        training_data.append((features, is_valid))

# Train model
ml_prior = MLPriorEstimator(config)
ml_prior.train_offline(training_data)

# Save model
import pickle
with open('trained_ml_prior.pkl', 'wb') as f:
    pickle.dump(ml_prior.model, f)
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review paper methodology (Section 4)
3. Examine example usage in `run_hcar_experiments.py`
4. Check existing implementations in your codebase

---

**Happy Constraint Acquisition! 🎉**

