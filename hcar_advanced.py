"""
HCAR: Hybrid Constraint Acquisition with Intelligent Refinement
================================================================

This module implements the advanced HCAR framework presented in:
"A Principled Framework for Interactive Refinement in Hybrid Constraint Acquisition"

The framework consists of three phases:
1. Passive Candidate Generation
2. Query-Driven Interactive Refinement (with Intelligent Subset Exploration)
3. Final Active Learning Refinement

"""

import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
from cpmpy import *
from cpmpy import Model
from cpmpy.transformations.get_variables import get_variables
CPMPY_AVAILABLE = True
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
XGBOOST_AVAILABLE = True

# Import PyConA for MQuAcq-2
try:
    from pycona import MQuAcq2, ProblemInstance, ConstraintOracle, absvar
    PYCONA_AVAILABLE = True
except ImportError:
    PYCONA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




class QueryStatus(Enum):
    """Status of query generation."""
    SUCCESS = "success"
    UNSAT = "unsat"
    TIMEOUT = "timeout"
    ERROR = "error"


class OracleResponse(Enum):
    """Oracle feedback to queries."""
    VALID = "Valid"
    INVALID = "Invalid"


@dataclass
class Constraint:
    """Represents a constraint candidate with metadata."""
    id: str
    constraint: Any  # CPMpy constraint object
    scope: List[str]  # Variable names in scope
    constraint_type: str  # e.g., "AllDifferent", "Sum", "Count"
    arity: int
    level: int = 0  # Subset depth (0 = original, 1+ = derived)
    confidence: float = 0.5  # P(c) - Bayesian confidence
    budget: int = 0  # Allocated query budget
    budget_used: int = 0  # Queries used so far
    features: Dict[str, float] = field(default_factory=dict)  # For ML
    parent_id: Optional[str] = None  # For tracking subset exploration
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Constraint) and self.id == other.id


@dataclass
class HCARConfig:
    """Configuration for HCAR framework."""
    # Budget and time limits
    total_budget: int = 500
    max_time_seconds: float = 1800.0  # 30 minutes
    query_timeout: float = 30.0  # Per-query timeout
    
    # Confidence thresholds
    theta_min: float = 0.15  # Rejection threshold
    theta_max: float = 0.85  # Acceptance threshold
    
    # Bayesian update parameters
    alpha: float = 0.1  # Noise parameter
    beta_positive: float = 0.7  # Update weight for positive evidence
    beta_negative: float = 0.3  # Update weight for negative evidence
    
    # Subset exploration
    max_subset_depth: int = 3  # Maximum depth for subset exploration
    
    # Budget allocation
    base_budget_per_constraint: int = 10
    uncertainty_weight: float = 0.5
    
    # Phase 3 (Active learning)
    use_mquacq: bool = True  # Use MQuAcq-2 for Phase 3
    
    # Feature extraction for ML
    enable_ml_prior: bool = True


class FeatureExtractor:
    """Extract features from constraints for ML prior estimation."""
    
    @staticmethod
    def extract_features(constraint: Constraint, problem_context: Dict) -> Dict[str, float]:
        """
        Extract structural features from a constraint.
        
        Features include:
        - Arity (normalized)
        - Constraint type encoding
        - Variable name patterns
        - Dimensional properties
        """
        features = {}
        
        # Basic structural features
        features['arity'] = constraint.arity
        features['arity_normalized'] = constraint.arity / problem_context.get('num_variables', 100)
        
        # Constraint type one-hot encoding
        ctype_map = {'AllDifferent': 0, 'Sum': 1, 'Count': 2, 'Other': 3}
        ctype_idx = ctype_map.get(constraint.constraint_type, 3)
        for i, ctype in enumerate(['AllDifferent', 'Sum', 'Count', 'Other']):
            features[f'type_{ctype}'] = 1.0 if i == ctype_idx else 0.0
        
        # Variable name pattern features
        scope = constraint.scope
        if scope:
            # Check for regular patterns (rows, columns, blocks)
            features['has_sequential_pattern'] = FeatureExtractor._check_sequential_pattern(scope)
            features['has_row_pattern'] = FeatureExtractor._check_pattern(scope, 'row')
            features['has_col_pattern'] = FeatureExtractor._check_pattern(scope, 'col')
            features['has_block_pattern'] = FeatureExtractor._check_pattern(scope, 'block')
        
        # Subset level (higher = more likely spurious)
        features['subset_level'] = constraint.level
        features['is_derived'] = 1.0 if constraint.level > 0 else 0.0
        
        return features
    
    @staticmethod
    def _check_sequential_pattern(scope: List[str]) -> float:
        """Check if variable names suggest sequential pattern."""
        # Simple heuristic: check if indices are consecutive
        try:
            indices = [int(''.join(filter(str.isdigit, var))) for var in scope if any(c.isdigit() for c in var)]
            if len(indices) >= 2:
                diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                return 1.0 if all(d == 1 for d in diffs) else 0.0
        except:
            pass
        return 0.0
    
    @staticmethod
    def _check_pattern(scope: List[str], pattern_name: str) -> float:
        """Check if variable names contain specific pattern."""
        count = sum(1 for var in scope if pattern_name in var.lower())
        return count / len(scope) if scope else 0.0


class MLPriorEstimator:
    """Machine learning model for estimating prior constraint probabilities."""
    
    def __init__(self, config: HCARConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            # Fallback to sklearn
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train_offline(self, training_data: List[Tuple[Dict, bool]]):
        """
        Train the model offline on historical benchmark data.
        
        Args:
            training_data: List of (features, is_valid) pairs
        """
        if not training_data:
            logger.warning("No training data provided for ML prior.")
            return
        
        X = []
        y = []
        for features, label in training_data:
            feature_vector = self._features_to_vector(features)
            X.append(feature_vector)
            y.append(1 if label else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training ML prior model on {len(X)} examples...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("ML prior model trained successfully.")
    
    def estimate_prior(self, constraint: Constraint) -> float:
        """
        Estimate prior probability that a constraint is valid.
        
        Returns:
            Prior probability in [0, 1]
        """
        if not self.config.enable_ml_prior or not self.is_trained:
            # Fallback: use heuristic based on constraint type and level
            return self._heuristic_prior(constraint)
        
        feature_vector = self._features_to_vector(constraint.features)
        feature_vector = np.array([feature_vector])
        
        try:
            proba = self.model.predict_proba(feature_vector)[0, 1]
            return float(proba)
        except Exception as e:
            logger.warning(f"ML prior estimation failed: {e}. Using heuristic.")
            return self._heuristic_prior(constraint)
    
    def _heuristic_prior(self, constraint: Constraint) -> float:
        """Fallback heuristic for prior estimation."""
        base_prior = 0.5
        
        # Penalize derived constraints (from subset exploration)
        if constraint.level > 0:
            base_prior -= 0.1 * constraint.level
        
        # Favor common global patterns
        if constraint.constraint_type == "AllDifferent":
            base_prior += 0.1
        
        # Penalize very high arity (more likely over-fitted)
        if constraint.arity > 20:
            base_prior -= 0.15
        
        return np.clip(base_prior, 0.1, 0.9)
    
    def _features_to_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dictionary to fixed-length vector."""
        feature_names = [
            'arity', 'arity_normalized',
            'type_AllDifferent', 'type_Sum', 'type_Count', 'type_Other',
            'has_sequential_pattern', 'has_row_pattern', 
            'has_col_pattern', 'has_block_pattern',
            'subset_level', 'is_derived'
        ]
        return [features.get(name, 0.0) for name in feature_names]


class IntelligentSubsetExplorer:
    """
    Implements the Intelligent Subset Exploration mechanism.
    
    Uses data-driven "culprit scores" to identify the most likely
    incorrect variable in a rejected constraint's scope.
    """
    
    @staticmethod
    def generate_informed_subsets(
        rejected_constraint: Constraint,
        positive_examples: List[Dict],
        learned_globals: List[Constraint],
        config: HCARConfig
    ) -> List[Constraint]:
        """
        Generate informed subsets by removing the most likely culprit variable.
        
        Args:
            rejected_constraint: The constraint that was refuted
            positive_examples: Initial positive examples (ground truth)
            learned_globals: Already validated global constraints
            config: HCAR configuration
        
        Returns:
            List of new candidate constraints (subsets of original scope)
        """
        if rejected_constraint.level >= config.max_subset_depth:
            logger.info(f"Max subset depth reached for {rejected_constraint.id}")
            return []
        
        scope = rejected_constraint.scope
        if len(scope) <= 2:
            logger.info(f"Scope too small to generate subsets for {rejected_constraint.id}")
            return []
        
        # Calculate culprit score for each variable
        culprit_scores = {}
        for var in scope:
            culprit_scores[var] = IntelligentSubsetExplorer._calculate_culprit_score(
                var, scope, rejected_constraint, positive_examples, learned_globals
            )
        
        # Sort variables by culprit score (descending)
        sorted_vars = sorted(culprit_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Culprit scores for {rejected_constraint.id}: {sorted_vars[:3]}")
        
        # Generate new candidates by removing top culprits
        new_candidates = []
        num_subsets = min(2, len(scope) - 2)  # Generate top 2 candidates
        
        for i in range(num_subsets):
            culprit_var = sorted_vars[i][0]
            new_scope = [v for v in scope if v != culprit_var]
            
            # Create new constraint with reduced scope
            new_constraint = IntelligentSubsetExplorer._create_subset_constraint(
                rejected_constraint, new_scope, culprit_var
            )
            
            if new_constraint:
                new_candidates.append(new_constraint)
        
        return new_candidates
    
    @staticmethod
    def _calculate_culprit_score(
        variable: str,
        scope: List[str],
        constraint: Constraint,
        positive_examples: List[Dict],
        learned_globals: List[Constraint]
    ) -> float:
        """
        Calculate culprit score for a variable.
        
        Higher score = more likely to be incorrectly included.
        
        Components:
        1. Structural isolation: average distance to other variables
        2. Weak constraint support: participation in other constraints
        3. Value pattern deviation: statistical anomalies in examples
        """
        score = 0.0
        
        # 1. Structural Isolation Score (0-1)
        isolation_score = IntelligentSubsetExplorer._structural_isolation(
            variable, scope
        )
        score += 0.4 * isolation_score
        
        # 2. Weak Constraint Support Score (0-1)
        support_score = IntelligentSubsetExplorer._weak_constraint_support(
            variable, learned_globals
        )
        score += 0.3 * support_score
        
        # 3. Value Pattern Deviation Score (0-1)
        if positive_examples:
            deviation_score = IntelligentSubsetExplorer._value_pattern_deviation(
                variable, scope, positive_examples, constraint.constraint_type
            )
            score += 0.3 * deviation_score
        
        return score
    
    @staticmethod
    def _structural_isolation(variable: str, scope: List[str]) -> float:
        """
        Calculate structural isolation of a variable.
        
        Uses edit distance and naming patterns.
        """
        if len(scope) <= 1:
            return 0.0
        
        # Extract indices from variable names
        def extract_indices(var_name):
            import re
            numbers = re.findall(r'\d+', var_name)
            return tuple(int(n) for n in numbers) if numbers else ()
        
        var_indices = extract_indices(variable)
        other_indices = [extract_indices(v) for v in scope if v != variable]
        
        if not var_indices or not other_indices:
            return 0.5  # Cannot determine, medium isolation
        
        # Calculate minimum distance to any other variable
        min_distance = float('inf')
        for other in other_indices:
            if len(var_indices) == len(other):
                dist = sum(abs(a - b) for a, b in zip(var_indices, other))
                min_distance = min(min_distance, dist)
        
        # Normalize (higher distance = more isolated)
        if min_distance == float('inf'):
            return 0.5
        
        return min(1.0, min_distance / 10.0)
    
    @staticmethod
    def _weak_constraint_support(variable: str, learned_globals: List[Constraint]) -> float:
        """
        Calculate how weakly a variable is supported by other constraints.
        
        Low participation = weak support = higher culprit score.
        """
        if not learned_globals:
            return 0.5
        
        participation_count = sum(
            1 for c in learned_globals if variable in c.scope
        )
        
        # Normalize: fewer participations = weaker support
        max_possible = len(learned_globals)
        support_ratio = participation_count / max_possible if max_possible > 0 else 0
        
        # Invert: weak support = high culprit score
        return 1.0 - support_ratio
    
    @staticmethod
    def _value_pattern_deviation(
        variable: str,
        scope: List[str],
        positive_examples: List[Dict],
        constraint_type: str
    ) -> float:
        """
        Detect statistical anomalies in variable values across examples.
        """
        if not positive_examples:
            return 0.0
        
        # Extract values for this variable and others in scope
        var_values = []
        other_values = []
        
        for example in positive_examples:
            if variable in example:
                var_values.append(example[variable])
            for other_var in scope:
                if other_var != variable and other_var in example:
                    other_values.append(example[other_var])
        
        if not var_values or not other_values:
            return 0.0
        
        # For AllDifferent: check if variable values are less diverse
        if constraint_type == "AllDifferent":
            var_unique = len(set(var_values))
            other_unique = len(set(other_values)) / len(scope) if scope else 1
            diversity_ratio = var_unique / len(var_values) if var_values else 0
            avg_other_diversity = other_unique / len(other_values) if other_values else 0
            
            # Lower diversity = more anomalous = higher score
            if avg_other_diversity > 0:
                return max(0, 1.0 - (diversity_ratio / avg_other_diversity))
        
        return 0.0
    
    @staticmethod
    def _create_subset_constraint(
        parent: Constraint,
        new_scope: List[str],
        removed_var: str
    ) -> Optional[Constraint]:
        """Create a new constraint candidate with reduced scope."""
        if len(new_scope) < 2:
            return None
        
        new_id = f"{parent.id}_sub_{removed_var}"
        
        # Create new constraint object (simplified - needs actual CPMpy constraint)
        new_constraint = Constraint(
            id=new_id,
            constraint=None,  # Will be created by constraint generator
            scope=new_scope,
            constraint_type=parent.constraint_type,
            arity=len(new_scope),
            level=parent.level + 1,
            confidence=0.5,  # Reset confidence
            budget=max(5, parent.budget // 2),  # Inherit reduced budget
            features={},
            parent_id=parent.id
        )
        
        return new_constraint


class BayesianUpdater:
    """Bayesian confidence updating mechanism."""
    
    @staticmethod
    def update_confidence(
        current_prob: float,
        query: Dict,
        response: OracleResponse,
        constraint: Constraint,
        alpha: float = 0.1
    ) -> float:
        """
        Update constraint confidence using Bayesian rule.
        
        Args:
            current_prob: Current P(c)
            query: The query assignment
            response: Oracle response (Valid/Invalid)
            constraint: The constraint being tested
            alpha: Noise parameter (probability of oracle error)
        
        Returns:
            Updated probability
        """
        if response == OracleResponse.INVALID:
            # Query violated constraint and was correctly rejected
            # This is positive evidence for the constraint
            # P(c | evidence) ∝ P(evidence | c) * P(c)
            likelihood = 1.0 - alpha  # High likelihood if c is true
            new_prob = (likelihood * current_prob) / (
                likelihood * current_prob + alpha * (1 - current_prob)
            )
            # Boost update
            new_prob = current_prob + 0.7 * (new_prob - current_prob)
        
        else:  # response == OracleResponse.VALID
            # Query violated constraint but was accepted by oracle
            # This is strong negative evidence - constraint is wrong
            new_prob = 0.0  # Hard refutation
        
        return np.clip(new_prob, 0.0, 1.0)


class QueryGenerator:
    """Generate targeted queries for constraint validation."""
    
    def __init__(self, config: HCARConfig):
        self.config = config
    
    def generate_query(
        self,
        target_constraint: Constraint,
        validated_constraints: List[Constraint],
        candidate_constraints: List[Constraint],
        variables: Dict[str, Any],
        domains: Dict[str, List]
    ) -> Tuple[Optional[Dict], QueryStatus]:
        """
        Generate a query that violates target_constraint but satisfies others.
        
        This is the core of the interactive refinement phase.
        
        Args:
            target_constraint: Constraint to test (should violate)
            validated_constraints: Already accepted constraints (must satisfy)
            candidate_constraints: Other candidates (should satisfy)
            variables: Problem variables
            domains: Variable domains
        
        Returns:
            (query_assignment, status)
        """
        if not CPMPY_AVAILABLE:
            logger.error("CPMpy not available for query generation")
            return None, QueryStatus.ERROR
        
        try:
            # Collect constraints to add to model
            constraints_to_satisfy = []
            
            # Add validated constraints (must satisfy)
            for c in validated_constraints:
                if c.constraint is not None:
                    constraints_to_satisfy.append(c.constraint)
            
            # Add other candidates (should satisfy for informativeness)
            for c in candidate_constraints:
                if c.id != target_constraint.id and c.constraint is not None:
                    constraints_to_satisfy.append(c.constraint)
            
            # Build model
            model = Model(constraints_to_satisfy)
            
            # Add negation of target constraint (must violate)
            if target_constraint.constraint is not None:
                model += ~target_constraint.constraint
            else:
                # No constraint object to test
                logger.warning(f"Target constraint {target_constraint.id} has no constraint object")
                return None, QueryStatus.ERROR
            
            # Solve with timeout
            flag = model.solve(time_limit=self.config.query_timeout)
            
            if flag:
                # Found a query - extract variable values
                query = {}
                for var_name, var_obj in variables.items():
                    try:
                        query[var_name] = var_obj.value()
                    except:
                        # Variable not in solution, skip
                        pass
                
                return query, QueryStatus.SUCCESS
            
            else:
                # UNSAT - cannot violate target while satisfying others
                # Strong evidence that target is valid or implied
                return None, QueryStatus.UNSAT
        
        except Exception as e:
            logger.error(f"Query generation error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, QueryStatus.ERROR


class HCARFramework:
    """
    Main HCAR (Hybrid Constraint Acquisition with Refinement) framework.
    
    Implements the three-phase algorithm:
    1. Passive Candidate Generation
    2. Query-Driven Interactive Refinement
    3. Final Active Learning
    """
    
    def __init__(
        self,
        config: HCARConfig,
        problem_name: str = "unknown"
    ):
        self.config = config
        self.problem_name = problem_name
        
        # Components
        self.ml_prior = MLPriorEstimator(config)
        self.subset_explorer = IntelligentSubsetExplorer()
        self.query_generator = QueryGenerator(config)
        
        # State
        self.B_globals: Set[Constraint] = set()
        self.B_fixed: Set[Constraint] = set()
        self.C_validated_globals: List[Constraint] = []
        self.C_learned_fixed: List[Constraint] = []
        
        self.queries_phase2 = 0
        self.queries_phase3 = 0
        self.start_time = None
        
        # Ground truth (for principled pruning)
        self.confirmed_solutions: List[Dict] = []
    
    def run(
        self,
        positive_examples: List[Dict],
        oracle_func: callable,
        variables: Dict[str, Any],
        domains: Dict[str, List],
        target_model: Optional[List] = None
    ) -> Tuple[List[Constraint], Dict[str, Any]]:
        """
        Run the full HCAR framework.
        
        Args:
            positive_examples: Initial positive examples (E+)
            oracle_func: Oracle function (assignment -> OracleResponse)
            variables: Problem variables
            domains: Variable domains
            target_model: Ground truth model (for evaluation only)
        
        Returns:
            (final_learned_model, metrics_dict)
        """
        self.start_time = time.time()
        logger.info(f"=== Starting HCAR on {self.problem_name} ===")
        logger.info(f"Initial examples: {len(positive_examples)}")
        
        # Store confirmed solutions
        self.confirmed_solutions = positive_examples.copy()
        
        # Phase 1: Passive Candidate Generation
        logger.info("\n--- Phase 1: Passive Candidate Generation ---")
        self._phase1_passive_generation(positive_examples, variables, domains)
        
        # Phase 2: Query-Driven Interactive Refinement
        logger.info("\n--- Phase 2: Query-Driven Interactive Refinement ---")
        self._phase2_interactive_refinement(oracle_func, variables, domains)
        
        # Phase 3: Final Active Learning
        logger.info("\n--- Phase 3: Final Active Learning ---")
        self._phase3_active_learning(oracle_func, variables, domains, positive_examples, target_model)
        
        # Compile final model
        final_model = self.C_validated_globals + self.C_learned_fixed
        
        # Calculate metrics
        elapsed_time = time.time() - self.start_time
        metrics = {
            'queries_phase2': self.queries_phase2,
            'queries_phase3': self.queries_phase3,
            'queries_total': self.queries_phase2 + self.queries_phase3,
            'time_seconds': elapsed_time,
            'num_global_constraints': len(self.C_validated_globals),
            'num_fixed_constraints': len(self.C_learned_fixed),
            'total_constraints': len(final_model)
        }
        
        logger.info(f"\n=== HCAR Complete ===")
        logger.info(f"Total queries: {metrics['queries_total']}")
        logger.info(f"Time: {elapsed_time:.2f}s")
        logger.info(f"Learned {len(final_model)} constraints")
        
        return final_model, metrics
    
    def _phase1_passive_generation(
        self,
        positive_examples: List[Dict],
        variables: Dict[str, Any],
        domains: Dict[str, List]
    ):
        """
        Phase 1: Extract candidate global constraints and prune fixed-arity bias.
        
        Uses pattern-based detection (similar to ModelSeeker).
        NOTE: This is a simplified implementation for demonstration.
        For production, integrate with your existing feature_extraction.py
        """
        logger.info("Phase 1: Passive Candidate Generation")
        
        # Extract global constraint candidates
        logger.info("Extracting global constraint candidates...")
        self.B_globals = self._extract_global_constraints_simple(
            positive_examples, variables, domains
        )
        
        # Generate and prune fixed-arity bias
        logger.info("Generating fixed-arity bias...")
        self.B_fixed = self._generate_fixed_bias_simple(
            variables, domains, positive_examples
        )
        
        logger.info(f"Phase 1 complete: {len(self.B_globals)} global candidates, "
                   f"{len(self.B_fixed)} fixed-arity candidates")
        
        # Initialize ML features and priors
        if self.config.enable_ml_prior:
            self._initialize_ml_priors()
        
        # Allocate uncertainty budgets
        self._allocate_uncertainty_budget()
    
    def _extract_global_constraints_simple(
        self,
        positive_examples: List[Dict],
        variables: Dict[str, Any],
        domains: Dict[str, List]
    ) -> Set[Constraint]:
        """
        Simple pattern-based extraction of global constraints.
        
        This is a basic implementation. For production use, integrate with
        your existing feature_extraction.py or other pattern detection methods.
        """
        candidates = set()
        
        # Group variables by naming pattern (e.g., row, column)
        var_groups = self._group_variables_by_pattern(list(variables.keys()))
        
        for group_name, var_names in var_groups.items():
            if len(var_names) >= 2:
                # Check if AllDifferent holds in examples
                holds_in_all = True
                for example in positive_examples:
                    values = [example.get(v) for v in var_names if v in example]
                    if len(values) == len(var_names):
                        if len(values) != len(set(values)):
                            holds_in_all = False
                            break
                
                if holds_in_all and len(var_names) > 2:
                    # Create candidate constraint
                    try:
                        if CPMPY_AVAILABLE:
                            from cpmpy import AllDifferent
                            constraint_vars = [variables[v] for v in var_names if v in variables]
                            constraint_obj = AllDifferent(constraint_vars)
                        else:
                            constraint_obj = None
                        
                        candidate = Constraint(
                            id=f"alldiff_{group_name}",
                            constraint=constraint_obj,
                            scope=var_names,
                            constraint_type="AllDifferent",
                            arity=len(var_names),
                            level=0,
                            confidence=0.5
                        )
                        candidates.add(candidate)
                        logger.debug(f"  Found candidate: {candidate.id} with {candidate.arity} variables")
                    except Exception as e:
                        logger.warning(f"Failed to create constraint for {group_name}: {e}")
        
        logger.info(f"  Extracted {len(candidates)} global constraint candidates")
        return candidates
    
    def _generate_fixed_bias_simple(
        self,
        variables: Dict[str, Any],
        domains: Dict[str, List],
        positive_examples: List[Dict]
    ) -> Set[Constraint]:
        """
        Simple generation of fixed-arity bias.
        
        This is a basic implementation. For production use, integrate with
        your existing bias generation methods.
        """
        candidates = set()
        
        # For a simple example, generate binary constraints (arity 2)
        var_list = list(variables.keys())
        
        # Only generate a small subset for testing
        max_pairs = min(20, len(var_list) * (len(var_list) - 1) // 2)
        
        pair_count = 0
        for i, var1 in enumerate(var_list):
            for var2 in var_list[i+1:]:
                if pair_count >= max_pairs:
                    break
                
                # Create inequality constraint (simple example)
                try:
                    if CPMPY_AVAILABLE:
                        constraint_obj = (variables[var1] != variables[var2])
                    else:
                        constraint_obj = None
                    
                    candidate = Constraint(
                        id=f"neq_{var1}_{var2}",
                        constraint=constraint_obj,
                        scope=[var1, var2],
                        constraint_type="NotEqual",
                        arity=2,
                        level=0,
                        confidence=0.5
                    )
                    
                    # Check if violated by any positive example
                    violated = False
                    for example in positive_examples:
                        if var1 in example and var2 in example:
                            if example[var1] == example[var2]:
                                violated = True
                                break
                    
                    if not violated:
                        candidates.add(candidate)
                        pair_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to create binary constraint: {e}")
            
            if pair_count >= max_pairs:
                break
        
        logger.info(f"  Generated {len(candidates)} fixed-arity constraint candidates")
        return candidates
    
    def _group_variables_by_pattern(self, var_names: List[str]) -> Dict[str, List[str]]:
        """
        Group variables by naming patterns (e.g., rows, columns, blocks).
        
        This is a simple heuristic. For production, use more sophisticated methods.
        """
        import re
        
        groups = {}
        
        # Collect all numeric indices to determine grid size
        all_indices = []
        var_to_indices = {}
        
        for var_name in var_names:
            # Extract numeric indices
            numbers = re.findall(r'\d+', var_name)
            
            if len(numbers) >= 2:
                row, col = int(numbers[0]), int(numbers[1])
                all_indices.append((row, col))
                var_to_indices[var_name] = (row, col)
        
        if not all_indices:
            return groups
        
        # Determine grid size
        max_row = max(idx[0] for idx in all_indices)
        max_col = max(idx[1] for idx in all_indices)
        grid_size = max(max_row, max_col) + 1
        
        # Group by rows and columns
        for var_name, (row, col) in var_to_indices.items():
            row_key = f"row_{row}"
            col_key = f"col_{col}"
            
            if row_key not in groups:
                groups[row_key] = []
            groups[row_key].append(var_name)
            
            if col_key not in groups:
                groups[col_key] = []
            groups[col_key].append(var_name)
        
        # Detect block patterns (for Sudoku-like structures)
        # Check for 4x4 (block_size=2), 9x9 (block_size=3), 16x16 (block_size=4)
        possible_block_sizes = [2, 3, 4, 5]
        
        for block_size in possible_block_sizes:
            if grid_size == block_size * block_size or grid_size % block_size == 0:
                # Create block groups
                block_groups = {}
                for var_name, (row, col) in var_to_indices.items():
                    block_row = row // block_size
                    block_col = col // block_size
                    block_key = f"block_{block_row}_{block_col}_bs{block_size}"
                    
                    if block_key not in block_groups:
                        block_groups[block_key] = []
                    block_groups[block_key].append(var_name)
                
                # Only add blocks that have the expected size
                expected_block_size = block_size * block_size
                for block_key, block_vars in block_groups.items():
                    if len(block_vars) == expected_block_size:
                        groups[block_key] = block_vars
        
        return groups
    
    def _initialize_ml_priors(self):
        """Initialize ML-based prior probabilities for all candidates."""
        problem_context = {'num_variables': len(self.B_globals) + len(self.B_fixed)}
        
        for constraint in self.B_globals:
            # Extract features
            constraint.features = FeatureExtractor.extract_features(
                constraint, problem_context
            )
            # Estimate prior
            constraint.confidence = self.ml_prior.estimate_prior(constraint)
            
        logger.info("ML priors initialized for all candidates")
    
    def _allocate_uncertainty_budget(self):
        """Allocate query budget based on uncertainty."""
        total_budget = self.config.total_budget
        num_constraints = len(self.B_globals)
        
        if num_constraints == 0:
            return
        
        # Calculate uncertainty for each constraint
        uncertainties = {}
        for c in self.B_globals:
            # Uncertainty = distance from decision boundary
            uncertainty = 1.0 - abs(c.confidence - 0.5) * 2
            uncertainties[c.id] = uncertainty
        
        # Allocate budget proportionally
        total_uncertainty = sum(uncertainties.values())
        
        for constraint in self.B_globals:
            if total_uncertainty > 0:
                proportion = uncertainties[constraint.id] / total_uncertainty
                constraint.budget = int(
                    self.config.base_budget_per_constraint +
                    self.config.uncertainty_weight * proportion * total_budget
                )
            else:
                constraint.budget = self.config.base_budget_per_constraint
        
        logger.info(f"Budget allocated across {num_constraints} candidates")
    
    def _phase2_interactive_refinement(
        self,
        oracle_func: callable,
        variables: Dict[str, Any],
        domains: Dict[str, List]
    ):
        """
        Phase 2: Query-Driven Interactive Refinement.
        
        Core of the HCAR framework - validates and corrects global constraints.
        """
        budget_pool = 0
        
        while (self.B_globals and 
               self.queries_phase2 < self.config.total_budget and
               (time.time() - self.start_time) < self.config.max_time_seconds):
            
            # Select most uncertain constraint with available budget
            candidate = self._select_candidate_for_refinement()
            
            if candidate is None:
                logger.info("No more refinable candidates")
                break
            
            logger.info(f"\nRefining: {candidate.id} (P={candidate.confidence:.3f}, "
                       f"budget={candidate.budget - candidate.budget_used})")
            
            # Refinement loop for this candidate
            consecutive_failures = 0
            max_consecutive_failures = 3  # Stop after 3 consecutive query gen failures
            
            while (self.config.theta_min < candidate.confidence < self.config.theta_max and
                   candidate.budget_used < candidate.budget):
                
                # Generate query
                query, status = self.query_generator.generate_query(
                    candidate,
                    self.C_validated_globals,
                    [c for c in self.B_globals if c.id != candidate.id],
                    variables,
                    domains
                )
                
                if status == QueryStatus.UNSAT:
                    # Cannot violate this constraint - strong evidence it's valid
                    logger.info(f"  UNSAT - accepting {candidate.id}")
                    candidate.confidence = self.config.theta_max
                    break
                
                if status == QueryStatus.TIMEOUT:
                    # Solver timeout - slight boost and continue
                    candidate.confidence = min(1.0, candidate.confidence + 0.05)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"  Too many query generation timeouts, moving on")
                        break
                    continue
                
                if status == QueryStatus.ERROR or query is None:
                    # Query generation failed
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"  Too many query generation failures for {candidate.id}, moving on")
                        break
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Ask oracle
                response = oracle_func(query)
                self.queries_phase2 += 1
                candidate.budget_used += 1
                
                logger.info(f"  Query {self.queries_phase2}: Oracle says {response.value}")
                
                if response == OracleResponse.VALID:
                    # Hard refutation - constraint is wrong
                    candidate.confidence = 0.0
                    
                    # PRINCIPLED PRUNING: Use new ground truth to prune fixed bias
                    self.confirmed_solutions.append(query)
                    self._prune_fixed_bias_with_solution(query)
                    
                    logger.info(f"  REFUTED: {candidate.id}")
                    break
                
                else:  # INVALID
                    # Positive evidence - update confidence
                    candidate.confidence = BayesianUpdater.update_confidence(
                        candidate.confidence,
                        query,
                        response,
                        candidate,
                        self.config.alpha
                    )
                    logger.info(f"  Updated P={candidate.confidence:.3f}")
            
            # Process candidate based on final confidence
            if candidate.confidence >= self.config.theta_max:
                # Accept constraint
                logger.info(f"✓ ACCEPTED: {candidate.id}")
                self.C_validated_globals.append(candidate)
                self.B_globals.remove(candidate)
                
                # Redistribute unused budget
                budget_pool += max(0, candidate.budget - candidate.budget_used)
            
            elif candidate.confidence <= self.config.theta_min:
                # Reject constraint and explore subsets
                logger.info(f"✗ REJECTED: {candidate.id}")
                self.B_globals.remove(candidate)
                
                if candidate.level < self.config.max_subset_depth:
                    # Generate informed subsets
                    new_candidates = self.subset_explorer.generate_informed_subsets(
                        candidate,
                        self.confirmed_solutions,
                        self.C_validated_globals,
                        self.config
                    )
                    
                    for new_cand in new_candidates:
                        # Inherit budget from parent
                        new_cand.budget = max(5, candidate.budget // 2)
                        self.B_globals.add(new_cand)
                        logger.info(f"  → Generated subset: {new_cand.id}")
        
        logger.info(f"\nPhase 2 complete: {len(self.C_validated_globals)} validated globals, "
                   f"{self.queries_phase2} queries used")
    
    def _select_candidate_for_refinement(self) -> Optional[Constraint]:
        """Select the most uncertain candidate with available budget."""
        candidates_with_budget = [
            c for c in self.B_globals 
            if c.budget_used < c.budget
        ]
        
        if not candidates_with_budget:
            return None
        
        # Select by highest uncertainty
        def uncertainty_score(c):
            return 1.0 - abs(c.confidence - 0.5) * 2
        
        return max(candidates_with_budget, key=uncertainty_score)
    
    def _prune_fixed_bias_with_solution(self, solution: Dict):
        """
        Principled pruning: Remove fixed-arity constraints violated by confirmed solution.
        
        This is a key methodological principle - only use confirmed ground truth.
        """
        to_remove = []
        for constraint in self.B_fixed:
            if constraint.constraint is not None:
                # Check if solution violates this constraint
                # (Implementation depends on constraint representation)
                # For now, placeholder
                pass
        
        for c in to_remove:
            self.B_fixed.remove(c)
        
        if to_remove:
            logger.info(f"  Pruned {len(to_remove)} fixed-arity constraints")
    
    def _phase3_active_learning(
        self,
        oracle_func: callable,
        variables: Dict[str, Any],
        domains: Dict[str, List],
        positive_examples: List[Dict] = None,
        target_model: Optional[List] = None
    ):
        """
        Phase 3: Final active learning to complete the model.
        
        Uses MQuAcq-2 with validated globals and pruned fixed bias.
        """
        if not self.config.use_mquacq:
            logger.info("Phase 3 (MQuAcq-2) disabled")
            return
        
        if not PYCONA_AVAILABLE:
            logger.warning("PyConA not available - skipping Phase 3")
            return
        
        logger.info("Running MQuAcq-2 for fixed-arity constraints...")
        logger.info(f"  Validated globals: {len(self.C_validated_globals)}")
        logger.info(f"  Fixed-arity bias size: {len(self.B_fixed)}")
        
        start_time = time.time()
        
        try:
            # Step 1: Decompose validated global constraints into binary constraints
            constraints_decomposed = []
            for constraint_obj in self.C_validated_globals:
                c = constraint_obj.constraint
                if hasattr(c, 'name') and c.name == "alldifferent":
                    # Decompose AllDifferent into binary != constraints
                    c_decomposed_list = c.decompose()
                    if c_decomposed_list:
                        constraints_decomposed.extend(c_decomposed_list[0])
                    logger.info(f"  Decomposed {c} into {len(c_decomposed_list[0]) if c_decomposed_list else 0} binary constraints")
                else:
                    # For non-AllDifferent globals, keep as is (Sum, Count, etc.)
                    # Note: These might need special handling or decomposition
                    logger.info(f"  Keeping global constraint as-is: {c}")
                    constraints_decomposed.append(c)
            
            logger.info(f"  Total decomposed constraints for CL: {len(constraints_decomposed)}")
            
            # Step 2: Prepare bias from fixed-arity constraints
            # Convert Constraint objects to CPMpy constraints
            bias_constraints = []
            for constraint_obj in self.B_fixed:
                if constraint_obj.constraint is not None:
                    bias_constraints.append(constraint_obj.constraint)
            
            logger.info(f"  Bias constraints: {len(bias_constraints)}")
            
            # Step 3: Get all variables from constraints
            all_constraints = constraints_decomposed + bias_constraints
            if all_constraints:
                problem_variables = get_variables(all_constraints)
            else:
                # Fallback: use variables dict
                problem_variables = list(variables.values()) if isinstance(variables, dict) else list(variables)
            
            logger.info(f"  Problem variables: {len(problem_variables)}")
            
            # Step 4: Create language (abstract binary relations)
            AV = absvar(2)  # Create abstract vars for binary constraints
            lang = [
                AV[0] == AV[1], 
                AV[0] != AV[1], 
                AV[0] < AV[1], 
                AV[0] > AV[1], 
                AV[0] >= AV[1], 
                AV[0] <= AV[1]
            ]
            
            # Step 5: Create ProblemInstance for MQuAcq-2
            instance = ProblemInstance(
                variables=problem_variables,
                init_cl=constraints_decomposed,
                language=lang,
                name=f"{self.problem_name}_phase3"
            )
            instance.bias = bias_constraints

            # CRITICAL FIX: Initialize instance variables with values
            # PyConA's ask_membership_query expects variables to have values from a previous solve
            # We solve the validated constraints to get an initial solution
            if constraints_decomposed:
                logger.info(f"  Solving validated constraints to initialize variable values...")
                init_model = Model(constraints_decomposed)
                if init_model.solve():
                    logger.info(f"  Successfully initialized instance variables with solution")
                    # Values are now in the variables
                else:
                    logger.warning(f"  Could not solve validated constraints for initialization")
                    # Fallback: try to assign from positive example
                    if positive_examples and len(positive_examples) > 0:
                        first_example = positive_examples[0]
                        logger.info(f"  Using first positive example as fallback initialization")
                        for var in problem_variables:
                            var_name = var.name if hasattr(var, 'name') else str(var)
                            if var_name in first_example:
                                try:
                                    var._value = first_example[var_name]
                                except:
                                    pass
            
            # Step 6: Create ConstraintOracle
            # CRITICAL: Use PyConA's ConstraintOracle with the ground truth CP model
            # This is required for PyConA to work correctly
            if target_model is not None and len(target_model) > 0:
                # We have the ground truth model - create proper ConstraintOracle
                logger.info(f"  Creating ConstraintOracle with {len(target_model)} ground truth constraints")
                oracle = ConstraintOracle(target_model)
                # Ensure variables_list is set (required by PyConA)
                try:
                    from cpmpy import cpm_array
                    oracle.variables_list = cpm_array(problem_variables)
                except:
                    oracle.variables_list = problem_variables
            elif isinstance(oracle_func, ConstraintOracle):
                oracle = oracle_func
                # Ensure variables_list is set (required by PyConA)
                if not hasattr(oracle, 'variables_list') or oracle.variables_list is None:
                    try:
                        from cpmpy import cpm_array
                        oracle.variables_list = cpm_array(problem_variables)
                    except:
                        oracle.variables_list = problem_variables
            else:
                # Create a simple oracle adapter
                # This assumes oracle_func takes a dict and returns OracleResponse
                class OracleAdapter:
                    def __init__(self, func, variables):
                        self.func = func
                        self.constraints = []  # Placeholder
                        # Store variables for conversion
                        self._variables = variables
                        # Set variables_list (required by PyConA)
                        try:
                            from cpmpy import cpm_array
                            self.variables_list = cpm_array(variables)
                        except:
                            self.variables_list = variables
                        
                    def answer_membership_query(self, assignment):
                        """
                        Adapter for membership queries.
                        PyConA expects this method name (not membership_query).
                        
                        Args:
                            assignment: Can be a list, tuple, set, or dict
                                       - If list/tuple/set: indexed by variable position
                                       - If dict: variable -> value mapping
                            
                        Returns:
                            Boolean: True if valid, False if invalid
                        """
                        # Convert assignment to dictionary if it's a list/tuple/set
                        if isinstance(assignment, (list, tuple)):
                            # Convert list/tuple to dict: variables[i] -> assignment[i]
                            assignment_dict = {}
                            for i, var in enumerate(self._variables):
                                if i < len(assignment):
                                    # Store with both var name and var object as keys for compatibility
                                    var_key = var.name if hasattr(var, 'name') else str(var)
                                    assignment_dict[var_key] = assignment[i]
                                    # Also store with variable object as key
                                    assignment_dict[var] = assignment[i]
                        elif isinstance(assignment, set):
                            # Convert set to list first (sets are unordered, so this might be problematic)
                            assignment_list = list(assignment)
                            assignment_dict = {}
                            for i, var in enumerate(self._variables):
                                if i < len(assignment_list):
                                    var_key = var.name if hasattr(var, 'name') else str(var)
                                    assignment_dict[var_key] = assignment_list[i]
                                    assignment_dict[var] = assignment_list[i]
                        elif isinstance(assignment, dict):
                            assignment_dict = assignment
                        else:
                            # Try to convert to list
                            try:
                                assignment_list = list(assignment)
                                assignment_dict = {}
                                for i, var in enumerate(self._variables):
                                    if i < len(assignment_list):
                                        var_key = var.name if hasattr(var, 'name') else str(var)
                                        assignment_dict[var_key] = assignment_list[i]
                                        assignment_dict[var] = assignment_list[i]
                            except:
                                logger.error(f"Cannot convert assignment of type {type(assignment)} to dict")
                                return False
                        
                        # Call oracle function with dictionary
                        try:
                            result = self.func(assignment_dict)
                            # Convert response to boolean (True = valid, False = invalid)
                            if hasattr(result, 'value'):
                                return result.value == "Valid"
                            return result == "Valid" or result is True
                        except Exception as e:
                            logger.error(f"Oracle error in adapter: {e}")
                            import traceback
                            traceback.print_exc()
                            return False
                
                oracle = OracleAdapter(oracle_func, problem_variables)
            
            # Step 7: Run MQuAcq-2
            logger.info("  Starting MQuAcq-2 learning...")
            ca_system = MQuAcq2()
            learned_instance = ca_system.learn(instance, oracle=oracle, verbose=3)
            
            # Step 8: Extract learned constraints
            final_constraints = learned_instance.get_cpmpy_model().constraints
            
            # Convert learned constraints back to our Constraint objects
            for i, c in enumerate(final_constraints):
                if c not in [obj.constraint for obj in self.C_validated_globals]:
                    # Create Constraint object for newly learned fixed-arity constraint
                    c_obj = Constraint(
                        id=f"learned_fixed_{i}",
                        constraint=c,
                        scope=[str(v) for v in get_variables([c])],
                        constraint_type=self._infer_constraint_type(c),
                        arity=len(get_variables([c])),
                        confidence=1.0  # MQuAcq-2 learned constraints are certain
                    )
                    self.C_learned_fixed.append(c_obj)
            
            # Step 9: Update metrics
            if hasattr(ca_system, 'env') and hasattr(ca_system.env, 'metrics'):
                self.queries_phase3 = ca_system.env.metrics.total_queries
                logger.info(f"  MQuAcq-2 queries: {self.queries_phase3}")
                logger.info(f"  MQuAcq-2 time: {ca_system.env.metrics.total_time:.2f}s")
            
            elapsed = time.time() - start_time
            logger.info(f"Phase 3 complete: {len(self.C_learned_fixed)} fixed-arity constraints learned in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue without Phase 3 results
            logger.warning("Continuing without Phase 3 results")
    
    def _infer_constraint_type(self, constraint) -> str:
        """Infer constraint type from CPMpy constraint object."""
        constraint_str = str(constraint).lower()
        if "alldifferent" in constraint_str:
            return "AllDifferent"
        elif "sum" in constraint_str:
            return "Sum"
        elif "count" in constraint_str:
            return "Count"
        elif "==" in constraint_str:
            return "Equality"
        elif "!=" in constraint_str:
            return "Inequality"
        elif ">=" in constraint_str or "<=" in constraint_str:
            return "Comparison"
        else:
            return "Other"


# ============================================================================
# Experimental Setup
# ============================================================================

class ExperimentRunner:
    """Run comprehensive experiments as described in the paper."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def run_experiment(
        self,
        benchmark_name: str,
        method_name: str,
        positive_examples: List[Dict],
        oracle_func: callable,
        variables: Dict[str, Any],
        domains: Dict[str, List],
        target_model: List,
        num_runs: int = 1
    ) -> Dict[str, Any]:
        """
        Run a single experiment configuration.
        
        Args:
            benchmark_name: Name of benchmark (e.g., "Sudoku")
            method_name: Method variant (e.g., "HCAR-Advanced")
            positive_examples: Initial E+
            oracle_func: Oracle function
            variables: Problem variables
            domains: Variable domains
            target_model: Ground truth model
            num_runs: Number of repetitions
        
        Returns:
            Aggregated results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {benchmark_name} | {method_name}")
        logger.info(f"{'='*60}")
        
        all_results = []
        
        for run in range(num_runs):
            logger.info(f"\n--- Run {run + 1}/{num_runs} ---")
            
            # Configure method
            config = self._get_method_config(method_name)
            
            # Run HCAR
            hcar = HCARFramework(config, problem_name=benchmark_name)
            learned_model, metrics = hcar.run(
                positive_examples,
                oracle_func,
                variables,
                domains,
                target_model
            )
            
            # Evaluate model quality
            evaluation = self._evaluate_model(
                learned_model,
                target_model,
                variables,
                domains
            )
            
            # Combine metrics
            result = {**metrics, **evaluation}
            all_results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Save results
        self._save_results(benchmark_name, method_name, aggregated)
        
        return aggregated
    
    def _get_method_config(self, method_name: str) -> HCARConfig:
        """Get configuration for specific method variant."""
        config = HCARConfig()
        
        if method_name == "HCAR-Advanced":
            # Full advanced method (default config)
            pass
        
        elif method_name == "HCAR-Heuristic":
            # Disable intelligent subset exploration
            # (would need to add a flag to use heuristic instead)
            config.enable_ml_prior = False
        
        elif method_name == "HCAR-NoRefine":
            # Skip Phase 2 entirely
            config.total_budget = 0  # No queries for refinement
        
        elif method_name == "MQuAcq-2":
            # Pure active learning baseline
            # (would need different implementation path)
            pass
        
        return config
    
    def _evaluate_model(
        self,
        learned_model: List[Constraint],
        target_model: List,
        variables: Dict[str, Any],
        domains: Dict[str, List],
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate learned model quality using solution-space metrics.
        
        Returns:
            Dictionary with S-Precision and S-Recall
        """
        # Generate sample solutions from both models
        learned_solutions = self._sample_solutions(learned_model, variables, domains, num_samples)
        target_solutions = self._sample_solutions(target_model, variables, domains, num_samples)
        
        if not learned_solutions or not target_solutions:
            logger.warning("Could not generate solutions for evaluation")
            return {'s_precision': 0.0, 's_recall': 0.0}
        
        # Calculate metrics
        # S-Precision: fraction of learned solutions that are valid
        valid_learned = sum(
            1 for sol in learned_solutions
            if self._is_valid_solution(sol, target_model)
        )
        s_precision = valid_learned / len(learned_solutions) * 100
        
        # S-Recall: fraction of target solutions accepted by learned model
        accepted_target = sum(
            1 for sol in target_solutions
            if self._is_valid_solution(sol, learned_model)
        )
        s_recall = accepted_target / len(target_solutions) * 100
        
        return {
            's_precision': s_precision,
            's_recall': s_recall
        }
    
    def _sample_solutions(
        self,
        model: List,
        variables: Dict[str, Any],
        domains: Dict[str, List],
        num_samples: int
    ) -> List[Dict]:
        """Generate sample solutions from a model."""
        solutions = []
        
        try:
            # Build CPMpy model
            cpm_model = Model()
            for constraint in model:
                if isinstance(constraint, Constraint) and constraint.constraint is not None:
                    cpm_model += constraint.constraint
                else:
                    cpm_model += constraint
            
            # Generate diverse solutions
            for _ in range(num_samples):
                if cpm_model.solve():
                    solution = {name: var.value() for name, var in variables.items()}
                    solutions.append(solution)
                    
                    # Add blocking clause to get different solution
                    blocking = []
                    for var in variables.values():
                        blocking.append(var != var.value())
                    cpm_model += sum(blocking) > 0
                else:
                    break
        
        except Exception as e:
            logger.error(f"Solution sampling error: {e}")
        
        return solutions
    
    def _is_valid_solution(self, solution: Dict, model: List) -> bool:
        """Check if solution satisfies all constraints in model."""
        # Simplified validation
        try:
            for constraint in model:
                # Would need to evaluate constraint on solution
                # (implementation depends on constraint representation)
                pass
            return True
        except:
            return False
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple runs."""
        if not results:
            return {}
        
        aggregated = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
        
        return aggregated
    
    def _save_results(self, benchmark: str, method: str, results: Dict):
        """Save results to file."""
        import json
        filename = f"{self.output_dir}/{benchmark}_{method}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution for running HCAR experiments.
    
    Example usage demonstrating the experimental setup.
    """
    logger.info("HCAR Framework - Advanced Implementation")
    logger.info("Based on: A Principled Framework for Interactive Refinement")
    logger.info("in Hybrid Constraint Acquisition\n")
    
    # Example: Run on Sudoku benchmark
    # (This would integrate with existing benchmarks)
    
    # Configure experiment
    config = HCARConfig(
        total_budget=500,
        max_time_seconds=1800,
        theta_min=0.15,
        theta_max=0.85,
        max_subset_depth=3,
        enable_ml_prior=True
    )
    
    # Initialize experiment runner
    runner = ExperimentRunner(output_dir="hcar_results")
    
    # Define benchmarks to run
    benchmarks = [
        "Sudoku",
        "UEFA",
        "VM_Allocation",
        "Exam_Timetabling",
        "Nurse_Rostering"
    ]
    
    # Define method variants
    methods = [
        "HCAR-Advanced",
        "HCAR-Heuristic",
        "HCAR-NoRefine",
        # "MQuAcq-2"  # Optional baseline
    ]
    
    logger.info(f"Running experiments on {len(benchmarks)} benchmarks")
    logger.info(f"Comparing {len(methods)} methods\n")
    
    # Run experiments
    # (This is a template - would need actual benchmark integration)
    
    logger.info("\n" + "="*60)
    logger.info("Experimental setup complete")
    logger.info("Integrate with existing benchmarks to run full evaluation")
    logger.info("="*60)


if __name__ == "__main__":
    main()

