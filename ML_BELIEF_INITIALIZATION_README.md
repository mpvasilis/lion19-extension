# Probabilistic Belief Initialization for Constraint Learning

This module implements a machine learning approach to initialize belief distributions over candidate `AllDifferent` constraints, as described in the paper's methodology.

## Overview

The system uses a Random Forest classifier trained on synthetic data from CSPLib benchmark problems to distinguish between **true constraints** (that belong to the actual problem model) and **overfitted patterns** (spurious patterns extracted from solutions).

### Key Features

- **Random Forest Classifier**: 100 trees, max depth 10, min samples per leaf 5
- **Probability Calibration**: Isotonic regression for well-calibrated probabilities
- **Rich Feature Extraction**: Structural patterns, positional statistics, regularities
- **Target Performance**: ~92% accuracy on held-out test set
- **Feature Importance**: Scope size (0.31), complete row/column (0.28), diagonal patterns (0.18)

## Architecture

### 1. Feature Extraction (`ConstraintFeatureExtractor`)

Extracts rich features from `AllDifferent` constraints:

- **Scope size**: Number of variables in the constraint
- **Complete row/column coverage**: Whether constraint covers a full row or column
- **Diagonal patterns**: Main diagonal, anti-diagonal, or any diagonal
- **Sliding window structure**: Consecutive patterns
- **Positional statistics**: Average row/column positions, spans, density
- **Regularity metrics**: Row-to-column ratio, spatial density

### 2. Synthetic Data Generation (`SyntheticDataGenerator`)

Generates training data from CSPLib benchmarks:

1. Load benchmark problems (Sudoku, Latin Square, Graph Coloring, etc.)
2. Generate 50-200 random valid solutions per benchmark
3. Extract candidate `AllDifferent` constraints from solutions
4. Label as **positive** (true constraint) or **negative** (overfitted pattern)
5. Extract features for each candidate

**Dataset Statistics**:
- ~5,000 labeled constraint instances
- 50 benchmark problems from CSPLib
- Class distribution: 35% positive, 65% negative

### 3. Model Training (`ConstraintClassifier`)

Trains a Random Forest with probability calibration:

1. **Training**: Random Forest with specified hyperparameters
2. **Cross-validation**: 5-fold stratified CV to validate generalization
3. **Calibration**: Isotonic regression on held-out validation set
4. **Evaluation**: Test on unseen benchmark families

### 4. Belief Scoring (`MLBeliefScorer`)

Provides convenient interface for using the trained model:

- Score individual constraints
- Batch score multiple candidates
- Rank candidates by belief probability
- Initialize belief distributions for HCAR pipeline
- Explain predictions with feature contributions

## Installation

### Dependencies

Ensure you have the required packages:

```bash
pip install numpy pandas scikit-learn cpmpy pycona
```

### Required Files

The system requires the benchmark modules in `benchmarks_global/`:
- `sudoku.py`
- `latin_square.py`
- `jsudoku.py`
- `graph_coloring.py`
- `nurse_rostering.py`
- `exam_timetabling.py`
- `uefa.py`
- `vm_allocation.py`

## Usage

### Step 1: Train the Model

Run the training script to generate synthetic data and train the classifier:

```bash
python probabilistic_belief_initialization.py
```

This will:
1. Generate ~5,000 training instances from 50 benchmarks
2. Split into train (60%), validation (20%), test (20%)
3. Train Random Forest with cross-validation
4. Apply probability calibration
5. Evaluate on test set (~92% accuracy expected)
6. Save trained model to `ml_models/constraint_classifier_calibrated.pkl`

**Output files**:
- `ml_training_data/training_data.csv` - Raw training data
- `ml_training_data/metadata.json` - Dataset metadata
- `ml_models/constraint_classifier_calibrated.pkl` - Trained model
- `ml_models/feature_importance.csv` - Feature importance ranking
- `ml_models/training_report.json` - Performance summary

**Expected runtime**: 10-30 minutes depending on hardware

### Step 2: Use the Trained Model

#### Basic Usage

```python
from ml_belief_scorer import MLBeliefScorer
from benchmarks_global.sudoku import construct_sudoku

# Load a problem
instance, oracle, overfitted = construct_sudoku(3, 3, 9)

# Initialize scorer
scorer = MLBeliefScorer(model_path='ml_models/constraint_classifier_calibrated.pkl')

# Score a single constraint
constraint = oracle.C_T[0]
probability = scorer.score_constraint(constraint, instance.variables)
print(f"Probability: {probability:.4f}")
```

#### Batch Scoring

```python
# Score multiple candidates
candidates = list(oracle.C_T) + overfitted
scores = scorer.score_candidates(candidates, instance.variables)

for constraint, prob in scores.items():
    print(f"Constraint: scope={len(constraint.args[0])}, P={prob:.4f}")
```

#### Ranking Candidates

```python
# Get ranked list of candidates (filtered by threshold)
ranked = scorer.get_ranked_candidates(
    candidates, 
    instance.variables, 
    threshold=0.5
)

for i, (constraint, prob) in enumerate(ranked[:10]):
    print(f"{i+1}. Probability: {prob:.4f}")
```

#### Initialize Belief Distribution

```python
# For HCAR pipeline: initialize P(c) for all candidates
beliefs = scorer.initialize_beliefs(candidates, instance.variables)
# beliefs is a numpy array of probabilities
```

#### Explain Predictions

```python
# Get detailed explanation for a constraint
explanation = scorer.explain_score(constraint, instance.variables)

print(f"Score: {explanation['score']:.4f}")
print(f"Prediction: {explanation['prediction']}")
print(f"Confidence: {explanation['confidence']:.4f}")
print("\nTop Features:")
for contrib in explanation['top_contributors'][:5]:
    print(f"  {contrib['feature']}: {contrib['value']:.2f}")
```

### Step 3: Test the System

Run the demonstration:

```bash
python ml_belief_scorer.py
```

This demonstrates:
- Loading a trained model
- Scoring true constraints (should get high probabilities ~0.8-0.9)
- Scoring overfitted constraints (should get low probabilities ~0.2-0.4)
- Ranking candidates
- Explaining predictions

## Integration with HCAR Pipeline

To integrate with the HCAR (Heuristic-based Constraint Acquisition with Ranking) pipeline:

```python
from ml_belief_scorer import MLBeliefScorer

# In your HCAR initialization
def initialize_beliefs_with_ml(candidates, variables):
    """Initialize belief distribution using ML model."""
    scorer = MLBeliefScorer()
    return scorer.initialize_beliefs(candidates, variables)

# Use in Phase 1
belief_probs = initialize_beliefs_with_ml(candidate_constraints, problem_variables)

# For each candidate c_i, set P(c_i) = belief_probs[i]
for i, candidate in enumerate(candidate_constraints):
    belief_distribution[candidate] = belief_probs[i]
```

## Model Performance

### Expected Metrics

Based on the paper's methodology:

- **Accuracy**: ~92% on held-out test set
- **Precision**: ~0.88-0.92 (few false positives)
- **Recall**: ~0.85-0.90 (captures most true constraints)
- **ROC AUC**: ~0.94-0.96 (excellent discrimination)

### Feature Importance

Top features (expected importance):

1. **scope_size** (0.31): Larger scopes often indicate overfitting
2. **is_complete_row/col** (0.28): True constraints often cover complete structures
3. **is_any_diagonal** (0.18): Diagonal patterns are characteristic
4. **normalized_scope** (0.12): Relative constraint size matters
5. **density** (0.08): Spatial compactness indicates structure

### Calibration Quality

The isotonic regression calibration ensures:
- Among all constraints assigned probability p, approximately p fraction are truly valid
- Enables meaningful probabilistic reasoning in Bayesian frameworks
- Critical for HCAR's belief update mechanism

## Dataset Details

### Training Data

Generated from 50 CSPLib benchmark families:

**Sudoku-like problems**:
- Standard Sudoku (4×4, 6×6, 9×9)
- Latin Squares (4×4, 6×6, 9×9)
- Jigsaw Sudoku (4×4, 6×6, 9×9)

**Graph Coloring**:
- Queen graphs (5×5, 6×6)
- Register allocation
- Course scheduling

**Scheduling Problems**:
- Nurse rostering (various sizes)
- Exam timetabling (v1, v2)
- UEFA tournament scheduling

**Other Problems**:
- VM allocation
- Additional CSPLib variants

### Labeling Strategy

**Positive examples** (true constraints):
- Extracted directly from problem oracle
- Known to be part of the true constraint model
- Examples: All rows in Sudoku, all edges in graph coloring

**Negative examples** (overfitted patterns):
- Hand-crafted overfitted patterns (per benchmark)
- Random patterns extracted from solutions
- Patterns that hold in some solutions but not all
- Examples: Diagonal constraints in standard Sudoku, non-adjacent row pairs

### Class Balance

- **35% positive** (true constraints)
- **65% negative** (overfitted patterns)
- Reflects realistic prevalence of overfitting in constraint acquisition
- Stratified sampling preserves balance in train/val/test splits

## File Structure

```
lion19-extension/
├── probabilistic_belief_initialization.py  # Main training script
├── ml_belief_scorer.py                     # Inference utilities
├── ML_BELIEF_INITIALIZATION_README.md      # This file
├── ml_training_data/                       # Generated training data
│   ├── training_data.csv
│   └── metadata.json
├── ml_models/                              # Trained models
│   ├── constraint_classifier_calibrated.pkl
│   ├── feature_importance.csv
│   └── training_report.json
└── benchmarks_global/                      # Benchmark problems
    ├── sudoku.py
    ├── latin_square.py
    ├── jsudoku.py
    ├── graph_coloring.py
    ├── nurse_rostering.py
    ├── exam_timetabling.py
    ├── uefa.py
    └── vm_allocation.py
```

## Troubleshooting

### Model Not Found

```
Error: Model file not found: ml_models/constraint_classifier_calibrated.pkl
```

**Solution**: Run the training script first:
```bash
python probabilistic_belief_initialization.py
```

### Import Errors

```
ModuleNotFoundError: No module named 'benchmarks_global'
```

**Solution**: Ensure you're in the correct directory and have the benchmark modules.

### Low Accuracy

If the model achieves < 85% accuracy:

1. **Check data quality**: Verify benchmark implementations
2. **Increase training data**: Generate more solutions per benchmark
3. **Adjust features**: Add domain-specific features
4. **Tune hyperparameters**: Try different Random Forest settings

### Memory Issues

If running out of memory during training:

1. **Reduce num_benchmarks**: Use fewer than 50 benchmarks
2. **Reduce num_solutions**: Generate fewer solutions per benchmark
3. **Use batch processing**: Process benchmarks in batches

## Advanced Usage

### Custom Benchmarks

To add custom benchmarks to the training set:

```python
# In SyntheticDataGenerator._get_benchmark_configs()
configs.append({
    'name': 'my_custom_problem',
    'func': my_benchmark_module.construct_my_problem,
    'params': {'size': 10},
    'num_solutions': 50
})
```

### Custom Features

To add custom features:

```python
# In ConstraintFeatureExtractor.extract_features()
def extract_features(self, constraint):
    features = super().extract_features(constraint)
    
    # Add custom feature
    features['my_custom_feature'] = self._compute_custom_feature(constraint)
    
    return features
```

### Hyperparameter Tuning

To tune Random Forest hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

## Citation

If you use this probabilistic belief initialization in your research, please cite:

```bibtex
@article{yourpaper2024,
  title={Your Paper Title},
  author={Your Names},
  journal={Conference/Journal Name},
  year={2024}
}
```

## References

1. CSPLib: A Problem Library for Constraints - http://www.csplib.org/
2. Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates.
3. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

## License

[Your License Here]

## Contact

For questions or issues, please contact [Your Contact Information]

