"""
Example: ML-Based Belief Initialization (Demonstration Without Trained Model)

This script demonstrates the concept of probabilistic belief initialization
without requiring a fully trained model. It shows:
1. Feature extraction from constraints
2. How features differ between true and overfitted constraints
3. The conceptual flow of the ML belief initialization system
"""

import numpy as np
import cpmpy as cp
from cpmpy.expressions.globalconstraints import AllDifferent

from probabilistic_belief_initialization import ConstraintFeatureExtractor
from benchmarks_global.sudoku import construct_sudoku
from benchmarks_global.graph_coloring import construct_graph_coloring


def demonstrate_feature_extraction():
    """Show how features are extracted from different types of constraints."""
    print("="*70)
    print("DEMONSTRATION: FEATURE EXTRACTION")
    print("="*70)
    
    # Create a simple 4x4 Sudoku
    print("\n1. Creating a 4x4 Sudoku problem...")
    instance, oracle, overfitted = construct_sudoku(2, 2, 4)
    
    # Get true constraints and overfitted constraints
    if hasattr(oracle, 'C_T'):
        true_constraints = oracle.C_T
    else:
        true_constraints = []
    
    true_alldiff = [c for c in true_constraints if isinstance(c, AllDifferent)]
    overfitted_alldiff = [c for c in overfitted if isinstance(c, AllDifferent)]
    
    print(f"   True AllDifferent constraints: {len(true_alldiff)}")
    print(f"   Overfitted constraints: {len(overfitted_alldiff)}")
    
    # Create feature extractor
    extractor = ConstraintFeatureExtractor(instance.variables)
    
    # Extract features for a true constraint (e.g., first row)
    print("\n2. Extracting features from a TRUE constraint (complete row)...")
    if true_alldiff:
        true_features = extractor.extract_features(true_alldiff[0])
        print(f"\n   Key features:")
        print(f"   - scope_size: {true_features['scope_size']}")
        print(f"   - is_complete_row: {true_features['is_complete_row']}")
        print(f"   - is_complete_col: {true_features['is_complete_col']}")
        print(f"   - is_any_diagonal: {true_features['is_any_diagonal']}")
        print(f"   - normalized_scope: {true_features['normalized_scope']:.3f}")
        print(f"   - density: {true_features['density']:.3f}")
    
    # Extract features for an overfitted constraint
    if overfitted_alldiff:
        print("\n3. Extracting features from an OVERFITTED constraint...")
        over_features = extractor.extract_features(overfitted_alldiff[0])
        print(f"\n   Key features:")
        print(f"   - scope_size: {over_features['scope_size']}")
        print(f"   - is_complete_row: {over_features['is_complete_row']}")
        print(f"   - is_complete_col: {over_features['is_complete_col']}")
        print(f"   - is_any_diagonal: {over_features['is_any_diagonal']}")
        print(f"   - normalized_scope: {over_features['normalized_scope']:.3f}")
        print(f"   - density: {over_features['density']:.3f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("True constraints typically have:")
    print("  ✓ Regular structure (complete rows/columns/blocks)")
    print("  ✓ High density (compact regions)")
    print("  ✓ Appropriate scope sizes for the problem")
    print("\nOverfitted constraints typically have:")
    print("  ✗ Irregular structure (scattered cells)")
    print("  ✗ Low density (sparse patterns)")
    print("  ✗ Unusual scope sizes")
    print("="*70 + "\n")


def demonstrate_feature_comparison():
    """Compare features across multiple constraints to show discriminability."""
    print("="*70)
    print("DEMONSTRATION: FEATURE COMPARISON")
    print("="*70)
    
    print("\n1. Loading 9x9 Sudoku...")
    instance, oracle, overfitted = construct_sudoku(3, 3, 9)
    
    if hasattr(oracle, 'C_T'):
        true_constraints = oracle.C_T
    else:
        true_constraints = []
    
    true_alldiff = [c for c in true_constraints if isinstance(c, AllDifferent)]
    overfitted_alldiff = [c for c in overfitted if isinstance(c, AllDifferent)]
    
    extractor = ConstraintFeatureExtractor(instance.variables)
    
    # Extract features for all constraints
    print("\n2. Extracting features from all constraints...")
    
    true_features_list = []
    for c in true_alldiff[:5]:  # Limit to first 5
        features = extractor.extract_features(c)
        true_features_list.append(features)
    
    over_features_list = []
    for c in overfitted_alldiff:
        features = extractor.extract_features(c)
        over_features_list.append(features)
    
    # Compute statistics
    print("\n3. Feature Statistics:")
    print("-" * 70)
    
    key_features = ['scope_size', 'is_complete_row', 'is_complete_col', 
                    'is_block', 'is_any_diagonal', 'density']
    
    for feature in key_features:
        if true_features_list:
            true_values = [f[feature] for f in true_features_list]
            true_mean = np.mean(true_values)
        else:
            true_mean = 0.0
        
        if over_features_list:
            over_values = [f[feature] for f in over_features_list]
            over_mean = np.mean(over_values)
        else:
            over_mean = 0.0
        
        print(f"\n{feature:20s}")
        print(f"  True constraints:      {true_mean:6.3f}")
        print(f"  Overfitted constraints:{over_mean:6.3f}")
        diff = abs(true_mean - over_mean)
        discriminative = "✓ Discriminative" if diff > 0.3 else "○ Less discriminative"
        print(f"  Difference:            {diff:6.3f}  {discriminative}")
    
    print("\n" + "="*70 + "\n")


def demonstrate_different_problems():
    """Show features across different problem types."""
    print("="*70)
    print("DEMONSTRATION: FEATURES ACROSS PROBLEM TYPES")
    print("="*70)
    
    problems = [
        ("Sudoku 4x4", lambda: construct_sudoku(2, 2, 4)),
        ("Graph Coloring (Queen 5x5)", lambda: construct_graph_coloring('queen_5x5')),
        ("Graph Coloring (Register)", lambda: construct_graph_coloring('register')),
    ]
    
    for prob_name, prob_func in problems:
        print(f"\n{prob_name}:")
        print("-" * 70)
        
        try:
            instance, oracle, overfitted = prob_func()
            
            if hasattr(oracle, 'C_T'):
                true_constraints = oracle.C_T
            else:
                true_constraints = []
            
            true_alldiff = [c for c in true_constraints if isinstance(c, AllDifferent)]
            
            if true_alldiff:
                extractor = ConstraintFeatureExtractor(instance.variables)
                features = extractor.extract_features(true_alldiff[0])
                
                print(f"  Variables shape: {instance.variables.shape if hasattr(instance.variables, 'shape') else 'N/A'}")
                print(f"  First constraint scope: {features['scope_size']}")
                print(f"  Is complete row: {features['is_complete_row']}")
                print(f"  Is complete col: {features['is_complete_col']}")
                print(f"  Density: {features['density']:.3f}")
            else:
                print("  No AllDifferent constraints found")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*70 + "\n")


def conceptual_ml_workflow():
    """Explain the conceptual ML workflow without training."""
    print("="*70)
    print("CONCEPTUAL ML WORKFLOW")
    print("="*70)
    
    print("""
The probabilistic belief initialization system works as follows:

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: TRAINING (offline, run once)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Generate synthetic data from CSPLib benchmarks             │
│     ├─ For each benchmark problem:                            │
│     │  ├─ Generate 50-200 random valid solutions              │
│     │  ├─ Extract candidate AllDifferent constraints          │
│     │  ├─ Label as TRUE (from oracle) or OVERFITTED           │
│     │  └─ Extract features (scope, structure, positions)      │
│     └─ Result: ~5,000 labeled constraint instances            │
│                                                                 │
│  2. Train Random Forest Classifier                             │
│     ├─ 100 trees, max depth 10, min samples leaf 5            │
│     ├─ 5-fold cross-validation                                │
│     └─ ~60% train, 20% validation, 20% test                   │
│                                                                 │
│  3. Apply Probability Calibration                              │
│     ├─ Isotonic regression on validation set                  │
│     └─ Ensures P(true|features) is well-calibrated            │
│                                                                 │
│  4. Evaluate and Save Model                                    │
│     └─ Target: ~92% accuracy on held-out test set             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: INFERENCE (online, during constraint acquisition)     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each candidate constraint c:                              │
│                                                                 │
│  1. Extract features(c)                                        │
│     └─ scope_size, is_row, is_col, diagonal, positions...     │
│                                                                 │
│  2. Query trained model                                        │
│     └─ probability = model.predict_proba(features)             │
│                                                                 │
│  3. Initialize belief                                          │
│     └─ P(c is true) = probability                              │
│                                                                 │
│  4. Use in HCAR pipeline                                       │
│     ├─ High P(c): Prioritize for early investigation          │
│     ├─ Low P(c): Deprioritize or filter out                   │
│     └─ Bayesian updates during learning                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

BENEFITS:
  ✓ Informed starting point (vs. uniform priors)
  ✓ Reduces queries on unlikely constraints
  ✓ Faster convergence to true constraint model
  ✓ Domain-independent (learns from multiple problem types)
  ✓ Probabilistic (enables Bayesian reasoning)

USAGE:
  # Train (once):
  python probabilistic_belief_initialization.py

  # Use (during constraint acquisition):
  from ml_belief_scorer import MLBeliefScorer
  scorer = MLBeliefScorer()
  probability = scorer.score_constraint(candidate, variables)
""")
    
    print("="*70 + "\n")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("ML-BASED BELIEF INITIALIZATION - CONCEPTUAL DEMONSTRATION")
    print("="*70 + "\n")
    
    try:
        demonstrate_feature_extraction()
        input("\nPress Enter to continue...")
        
        demonstrate_feature_comparison()
        input("\nPress Enter to continue...")
        
        demonstrate_different_problems()
        input("\nPress Enter to continue...")
        
        conceptual_ml_workflow()
        
        print("="*70)
        print("NEXT STEPS")
        print("="*70)
        print("""
To use the full ML belief initialization system:

1. Train the model:
   python probabilistic_belief_initialization.py
   (Takes 10-30 minutes)

2. Test the trained model:
   python ml_belief_scorer.py

3. Integrate with your constraint acquisition system:
   from ml_belief_scorer import MLBeliefScorer
   scorer = MLBeliefScorer()
   beliefs = scorer.initialize_beliefs(candidates, variables)

For more information, see ML_BELIEF_INITIALIZATION_README.md
""")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.\n")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

