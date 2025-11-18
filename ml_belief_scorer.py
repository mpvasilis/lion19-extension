"""
ML-Based Belief Scorer for Constraint Learning

This module provides utilities for using the trained Random Forest classifier
to initialize belief probabilities for candidate AllDifferent constraints.

Usage:
    from ml_belief_scorer import MLBeliefScorer
    
    # Initialize scorer with trained model
    scorer = MLBeliefScorer(model_path='ml_models/constraint_classifier_calibrated.pkl')
    
    # Score a candidate constraint
    probability = scorer.score_constraint(candidate_constraint, problem_variables)
    
    # Score multiple candidates
    scores = scorer.score_candidates(candidate_list, problem_variables)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Union

import cpmpy as cp
from cpmpy.expressions.globalconstraints import AllDifferent

from probabilistic_belief_initialization import ConstraintFeatureExtractor


class MLBeliefScorer:
    """
    Score candidate AllDifferent constraints using trained ML model.
    
    This class provides a convenient interface for using the trained Random Forest
    classifier to initialize belief probabilities for candidate constraints in the
    HCAR (Heuristic-based Constraint Acquisition with Ranking) pipeline.
    """
    
    def __init__(self, model_path: str = 'ml_models/constraint_classifier_calibrated.pkl'):
        """
        Initialize the belief scorer with a trained model.
        
        Args:
            model_path: Path to the pickled trained model
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please run probabilistic_belief_initialization.py first to train the model."
            )
        
        # Load the trained model
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.calibrated_model = model_data['calibrated_model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        
        print(f"ML Belief Scorer loaded from {model_path}")
        print(f"Model features: {len(self.feature_names)}")
        print(f"Ready to score constraints.")
    
    def score_constraint(self, constraint: AllDifferent, variables) -> float:
        """
        Score a single AllDifferent constraint.
        
        Args:
            constraint: The AllDifferent constraint to score
            variables: The problem variables (for feature extraction)
            
        Returns:
            Probability that the constraint is a true constraint (0.0 to 1.0)
            Higher values indicate higher belief that the constraint is valid.
        """
        if not isinstance(constraint, AllDifferent):
            raise ValueError("Constraint must be an AllDifferent constraint")
        
        # Extract features
        extractor = ConstraintFeatureExtractor(variables)
        features = extractor.extract_features(constraint)
        
        # Convert to array in correct order
        feature_array = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        
        # Predict probability
        probability = self.calibrated_model.predict_proba(feature_array)[0, 1]
        
        return float(probability)
    
    def score_candidates(self, candidates: List[AllDifferent], variables) -> Dict[AllDifferent, float]:
        """
        Score multiple candidate constraints.
        
        Args:
            candidates: List of AllDifferent constraints to score
            variables: The problem variables (for feature extraction)
            
        Returns:
            Dictionary mapping each constraint to its probability score
        """
        scores = {}
        extractor = ConstraintFeatureExtractor(variables)
        
        # Extract features for all candidates
        feature_list = []
        valid_candidates = []
        
        for candidate in candidates:
            if isinstance(candidate, AllDifferent):
                try:
                    features = extractor.extract_features(candidate)
                    feature_array = [features.get(f, 0.0) for f in self.feature_names]
                    feature_list.append(feature_array)
                    valid_candidates.append(candidate)
                except Exception as e:
                    print(f"Warning: Could not extract features for constraint: {e}")
                    scores[candidate] = 0.5  # Default neutral score
        
        # Batch prediction
        if feature_list:
            X = np.array(feature_list)
            probabilities = self.calibrated_model.predict_proba(X)[:, 1]
            
            for candidate, prob in zip(valid_candidates, probabilities):
                scores[candidate] = float(prob)
        
        return scores
    
    def get_ranked_candidates(self, candidates: List[AllDifferent], variables, 
                            threshold: float = 0.5) -> List[tuple]:
        """
        Get candidates ranked by belief probability, optionally filtered by threshold.
        
        Args:
            candidates: List of AllDifferent constraints to score
            variables: The problem variables (for feature extraction)
            threshold: Minimum probability threshold (default: 0.5)
            
        Returns:
            List of (constraint, probability) tuples, sorted by probability (descending)
        """
        scores = self.score_candidates(candidates, variables)
        
        # Filter and sort
        ranked = [(c, p) for c, p in scores.items() if p >= threshold]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def initialize_beliefs(self, candidates: List[AllDifferent], variables) -> np.ndarray:
        """
        Initialize belief distribution for HCAR pipeline.
        
        This method returns a numpy array of probabilities that can be directly
        used to initialize the belief distribution P(c) for each candidate c.
        
        Args:
            candidates: List of AllDifferent constraints
            variables: The problem variables
            
        Returns:
            Numpy array of probabilities, one per candidate
        """
        scores = self.score_candidates(candidates, variables)
        return np.array([scores.get(c, 0.5) for c in candidates])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return dict(zip(self.feature_names, self.feature_importance))
    
    def explain_score(self, constraint: AllDifferent, variables) -> Dict[str, any]:
        """
        Explain the score for a constraint by showing feature values and contributions.
        
        Args:
            constraint: The AllDifferent constraint to explain
            variables: The problem variables
            
        Returns:
            Dictionary with score, features, and top contributing features
        """
        # Extract features
        extractor = ConstraintFeatureExtractor(variables)
        features = extractor.extract_features(constraint)
        
        # Get score
        score = self.score_constraint(constraint, variables)
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Compute weighted contributions
        contributions = []
        for feat_name in self.feature_names:
            feat_value = features.get(feat_name, 0.0)
            feat_importance = importance.get(feat_name, 0.0)
            contribution = feat_value * feat_importance
            contributions.append({
                'feature': feat_name,
                'value': feat_value,
                'importance': feat_importance,
                'contribution': contribution
            })
        
        # Sort by contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'score': score,
            'prediction': 'TRUE CONSTRAINT' if score >= 0.5 else 'OVERFITTED',
            'confidence': abs(score - 0.5) * 2,  # 0 to 1
            'all_features': features,
            'top_contributors': contributions[:10]
        }


def demonstrate_usage():
    """Demonstrate usage of the MLBeliefScorer."""
    print("="*70)
    print("ML BELIEF SCORER DEMONSTRATION")
    print("="*70)
    
    # Import a benchmark
    from benchmarks_global.sudoku import construct_sudoku
    
    print("\nLoading benchmark: 4x4 Sudoku")
    instance, oracle, overfitted = construct_sudoku(2, 2, 4)
    
    print(f"Number of true constraints: {len(oracle.C_T)}")
    print(f"Number of overfitted constraints: {len(overfitted)}")
    
    # Initialize scorer
    print("\nInitializing ML Belief Scorer...")
    try:
        scorer = MLBeliefScorer()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run: python probabilistic_belief_initialization.py")
        return
    
    # Score true constraints
    print("\n" + "-"*70)
    print("SCORING TRUE CONSTRAINTS")
    print("-"*70)
    
    true_alldiff = [c for c in oracle.C_T if isinstance(c, AllDifferent)]
    for i, constraint in enumerate(true_alldiff[:5]):  # Show first 5
        score = scorer.score_constraint(constraint, instance.variables)
        print(f"\nTrue Constraint {i+1}:")
        print(f"  Scope size: {len(constraint.args[0])}")
        print(f"  ML Probability: {score:.4f}")
        print(f"  Prediction: {'TRUE CONSTRAINT ✓' if score >= 0.5 else 'OVERFITTED ✗'}")
    
    # Score overfitted constraints
    if overfitted:
        print("\n" + "-"*70)
        print("SCORING OVERFITTED CONSTRAINTS")
        print("-"*70)
        
        for i, constraint in enumerate(overfitted[:5]):  # Show first 5
            if isinstance(constraint, AllDifferent):
                score = scorer.score_constraint(constraint, instance.variables)
                print(f"\nOverfitted Constraint {i+1}:")
                print(f"  Scope size: {len(constraint.args[0])}")
                print(f"  ML Probability: {score:.4f}")
                print(f"  Prediction: {'TRUE CONSTRAINT ✓' if score >= 0.5 else 'OVERFITTED ✗'}")
    
    # Demonstrate ranking
    print("\n" + "-"*70)
    print("RANKED CANDIDATES (by belief probability)")
    print("-"*70)
    
    all_candidates = true_alldiff + [c for c in overfitted if isinstance(c, AllDifferent)]
    ranked = scorer.get_ranked_candidates(all_candidates, instance.variables)
    
    print(f"\nTop 10 candidates:")
    for i, (constraint, prob) in enumerate(ranked[:10]):
        is_true = any(set(constraint.args[0]) == set(tc.args[0]) for tc in true_alldiff)
        label = "TRUE" if is_true else "OVERFITTED"
        marker = "✓" if is_true else "✗"
        print(f"  {i+1}. Probability: {prob:.4f}  [{label} {marker}]  Scope: {len(constraint.args[0])}")
    
    # Demonstrate explanation
    print("\n" + "-"*70)
    print("DETAILED EXPLANATION FOR A CONSTRAINT")
    print("-"*70)
    
    if true_alldiff:
        explanation = scorer.explain_score(true_alldiff[0], instance.variables)
        print(f"\nScore: {explanation['score']:.4f}")
        print(f"Prediction: {explanation['prediction']}")
        print(f"Confidence: {explanation['confidence']:.4f}")
        print(f"\nTop Contributing Features:")
        for contrib in explanation['top_contributors'][:5]:
            print(f"  - {contrib['feature']:25s}: "
                  f"value={contrib['value']:6.2f}, "
                  f"importance={contrib['importance']:6.4f}, "
                  f"contribution={contrib['contribution']:8.4f}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    demonstrate_usage()

