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
    use_intelligent_subsets: bool = True  # Use intelligent culprit scores (False = positional heuristics)
    use_counterexample_repair: bool = True  # Use counterexample-driven minimal repair (most advanced)

    # Budget allocation
    base_budget_per_constraint: int = 10
    uncertainty_weight: float = 0.5
    
    # Phase 3 (Active learning)
    use_mquacq: bool = True  # Use MQuAcq-2 for Phase 3

    # Experimental validation
    inject_overfitted: bool = False  # Inject deliberate overfitted constraints for testing

    # Feature extraction for ML
    enable_ml_prior: bool = True

    # Domain expert hints (optional)
    domain_hints: Optional[Dict] = None
    enable_hint_normalization: bool = True


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


class CounterexampleRepair:
    """
    Implements counterexample-driven minimal repair mechanism.

    When a constraint is refuted by a counterexample, this class:
    1. Analyzes which variables cause the violation
    2. Generates minimal repair hypotheses (remove minimum variables)
    3. Filters repairs for consistency with positive examples
    4. Ranks repairs using ML prior and structural metrics

    This is more principled than heuristic culprit scores as it uses
    the actual counterexample to identify conflicts.
    """

    @staticmethod
    def repair_from_counterexample(
        rejected_constraint: Constraint,
        counterexample: Dict,
        positive_examples: List[Dict],
        learned_globals: List[Constraint],
        config: HCARConfig,
        variables: Dict[str, Any] = None,
        ml_prior: 'MLPriorEstimator' = None
    ) -> List[Constraint]:
        """
        Generate minimal repair hypotheses from counterexample.

        Args:
            rejected_constraint: The constraint refuted by counterexample
            counterexample: The valid solution that violates the constraint
            positive_examples: Original positive examples (for consistency check)
            learned_globals: Already validated constraints
            config: HCAR configuration
            variables: Variable objects dict
            ml_prior: ML prior estimator for ranking

        Returns:
            List of repair hypotheses, ranked by plausibility
        """
        if rejected_constraint.level >= config.max_subset_depth:
            logger.info(f"Max subset depth reached for {rejected_constraint.id}")
            return []

        scope = rejected_constraint.scope
        if len(scope) <= 2:
            logger.info(f"Scope too small for repair: {rejected_constraint.id}")
            return []

        # Step 1: Identify variables causing violation in counterexample
        violating_vars = CounterexampleRepair._identify_violating_variables(
            rejected_constraint, counterexample, variables
        )

        if not violating_vars:
            logger.warning(f"Could not identify violating variables for {rejected_constraint.id}")
            return []

        logger.info(f"Counterexample analysis: violating variables = {violating_vars}")

        # Step 2: Generate minimal repair hypotheses
        repair_hypotheses = CounterexampleRepair._generate_minimal_repairs(
            rejected_constraint, violating_vars, counterexample, variables
        )

        logger.info(f"Generated {len(repair_hypotheses)} minimal repair hypotheses")

        # Step 3: Filter for consistency with positive examples
        consistent_repairs = CounterexampleRepair._filter_consistent_repairs(
            repair_hypotheses, positive_examples
        )

        logger.info(f"Filtered to {len(consistent_repairs)} repairs consistent with E+")

        if not consistent_repairs:
            return []

        # Step 4: Rank repairs by plausibility
        ranked_repairs = CounterexampleRepair._rank_repairs(
            consistent_repairs, positive_examples, ml_prior
        )

        # Return top-k repairs (limit to avoid bias pollution)
        max_repairs = min(2, len(ranked_repairs))
        top_repairs = ranked_repairs[:max_repairs]

        for i, repair in enumerate(top_repairs):
            logger.info(f"  Repair {i+1}: {repair.id} (score={repair.confidence:.3f})")

        return top_repairs

    @staticmethod
    def _identify_violating_variables(
        constraint: Constraint,
        counterexample: Dict,
        variables: Dict[str, Any]
    ) -> List[str]:
        """
        Identify which variables in the scope cause the violation.

        For AllDifferent: find variables with duplicate values (precise)
        For Sum: use contribution analysis + statistical outliers (principled)
        For Count: use direct violation analysis (precise)
        """
        scope = constraint.scope
        constraint_type = constraint.constraint_type
        violating_vars = []

        # Extract values from counterexample
        values = {}
        for var_name in scope:
            if var_name in counterexample:
                values[var_name] = counterexample[var_name]

        if len(values) != len(scope):
            return []  # Incomplete assignment

        if constraint_type == "AllDifferent":
            # PRECISE: Find variables with duplicate values
            value_to_vars = {}
            for var, val in values.items():
                if val not in value_to_vars:
                    value_to_vars[val] = []
                value_to_vars[val].append(var)

            # Variables involved in duplicates are violating
            for val, vars_with_val in value_to_vars.items():
                if len(vars_with_val) > 1:
                    violating_vars.extend(vars_with_val)

        elif constraint_type == "Sum":
            # PRINCIPLED: Contribution analysis for Sum constraints
            violating_vars = CounterexampleRepair._identify_violating_vars_sum(
                constraint, values
            )

        elif constraint_type == "Count":
            # PRECISE: Direct violation analysis for Count constraints
            violating_vars = CounterexampleRepair._identify_violating_vars_count(
                constraint, values
            )

        return list(set(violating_vars))

    @staticmethod
    def _identify_violating_vars_sum(
        constraint: Constraint,
        values: Dict[str, int]
    ) -> List[str]:
        """
        Identify violating variables for Sum constraints using contribution analysis.

        Strategy:
        1. Calculate actual sum and violation amount
        2. Find variables whose removal would fix the violation (contribution analysis)
        3. Fallback to statistical outliers if contribution analysis fails
        """
        var_list = list(values.keys())
        val_list = [values[v] for v in var_list]

        if not val_list:
            return []

        actual_sum = sum(val_list)

        # Extract bound and operator from constraint
        # Try to parse CPMpy constraint object
        bound = None
        operator = None

        try:
            # CPMpy constraints have structure like: sum(vars) <= bound
            constraint_obj = constraint.constraint
            if hasattr(constraint_obj, 'args'):
                # Look for comparison operators
                if hasattr(constraint_obj, '__class__'):
                    op_name = constraint_obj.__class__.__name__
                    if 'LessEqual' in op_name or '<=' in str(constraint_obj):
                        operator = '<='
                        # Try to extract bound from args
                        for arg in constraint_obj.args:
                            if isinstance(arg, (int, float)):
                                bound = arg
                    elif 'Equal' in op_name or '==' in str(constraint_obj):
                        operator = '=='
                        for arg in constraint_obj.args:
                            if isinstance(arg, (int, float)):
                                bound = arg
        except:
            pass

        # If we have bound and operator, use contribution analysis
        if bound is not None:
            violation_amount = actual_sum - bound if operator == '<=' else abs(actual_sum - bound)

            if violation_amount > 0:
                # Find variables that could fix violation if removed
                candidates = []
                for var in var_list:
                    var_value = values[var]
                    # If removing this variable would fix or reduce violation significantly
                    if var_value >= violation_amount * 0.5:  # At least 50% of violation
                        candidates.append(var)

                if candidates:
                    return candidates

        # FALLBACK: Statistical outlier detection
        mean_val = sum(val_list) / len(val_list)
        std_val = (sum((v - mean_val)**2 for v in val_list) / len(val_list))**0.5

        outliers = []
        for var in var_list:
            if std_val > 0 and abs(values[var] - mean_val) > std_val:
                outliers.append(var)

        # If no outliers, take variables with extreme values
        if not outliers:
            max_val = max(val_list)
            min_val = min(val_list)
            for var in var_list:
                if values[var] == max_val or values[var] == min_val:
                    outliers.append(var)

        return outliers

    @staticmethod
    def _identify_violating_vars_count(
        constraint: Constraint,
        values: Dict[str, int]
    ) -> List[str]:
        """
        Identify violating variables for Count constraints using direct violation analysis.

        Strategy:
        1. Determine target value and expected count
        2. If actual_count > expected: return variables WITH target value
        3. If actual_count < expected: return variables WITHOUT target value
        """
        var_list = list(values.keys())

        # Extract target value and bound from constraint
        target_value = None
        bound = None
        operator = None

        try:
            # CPMpy Count constraints: Count(vars, value) op bound
            constraint_obj = constraint.constraint
            constraint_str = str(constraint_obj)

            # Try to parse from string representation
            # Example: "Count([x1, x2, x3], 5) == 2"
            if 'Count' in constraint_str:
                # Extract target value (number after variables list)
                # Need to handle nested brackets: Count([x1, x2], 5)
                import re
                match = re.search(r'Count\(.+?,\s*(\d+)\)', constraint_str)
                if match:
                    target_value = int(match.group(1))

                # Extract operator and bound
                if '<=' in constraint_str:
                    operator = '<='
                    match = re.search(r'<=\s*(\d+)', constraint_str)
                    if match:
                        bound = int(match.group(1))
                elif '==' in constraint_str:
                    operator = '=='
                    match = re.search(r'==\s*(\d+)', constraint_str)
                    if match:
                        bound = int(match.group(1))
        except:
            pass

        # If we couldn't extract parameters, fallback to statistical analysis
        if target_value is None or bound is None:
            # Use statistical outlier detection as fallback
            val_list = [values[v] for v in var_list]
            mean_val = sum(val_list) / len(val_list)
            std_val = (sum((v - mean_val)**2 for v in val_list) / len(val_list))**0.5

            outliers = []
            for var in var_list:
                if std_val > 0 and abs(values[var] - mean_val) > std_val:
                    outliers.append(var)

            return outliers if outliers else var_list[:len(var_list)//2]

        # DIRECT VIOLATION ANALYSIS
        # Count how many variables have the target value
        vars_with_value = [v for v in var_list if values[v] == target_value]
        vars_without_value = [v for v in var_list if values[v] != target_value]
        actual_count = len(vars_with_value)

        if operator == '==' or operator == '<=':
            if actual_count > bound:
                # Too many variables have the target value
                # Any of them could be removed
                return vars_with_value
            elif actual_count < bound and operator == '==':
                # Too few variables have the target value
                # Removing variables WITHOUT the value might help
                return vars_without_value

        # If analysis inconclusive, return all as candidates
        return var_list

    @staticmethod
    def _generate_minimal_repairs(
        parent: Constraint,
        violating_vars: List[str],
        counterexample: Dict,
        variables: Dict[str, Any]
    ) -> List[Constraint]:
        """
        Generate minimal repair hypotheses by removing violating variables.

        Strategy: For each violating variable, create a repair by removing it.
        These are minimal in the sense that we remove single variables.
        """
        repairs = []

        for var_to_remove in violating_vars:
            new_scope = [v for v in parent.scope if v != var_to_remove]

            if len(new_scope) < 2:
                continue

            # Create repair hypothesis
            repair = IntelligentSubsetExplorer._create_subset_constraint(
                parent, new_scope, var_to_remove, variables
            )

            if repair:
                repairs.append(repair)

        return repairs

    @staticmethod
    def _filter_consistent_repairs(
        repairs: List[Constraint],
        positive_examples: List[Dict]
    ) -> List[Constraint]:
        """
        Filter repairs: keep only those consistent with positive examples E+.
        """
        consistent = []

        for repair in repairs:
            is_consistent = True

            for example in positive_examples:
                # Check if repair holds on this example
                if not CounterexampleRepair._check_constraint_on_example(repair, example):
                    is_consistent = False
                    break

            if is_consistent:
                consistent.append(repair)

        return consistent

    @staticmethod
    def _check_constraint_on_example(constraint: Constraint, example: Dict) -> bool:
        """Check if constraint is satisfied by example."""
        scope = constraint.scope
        constraint_type = constraint.constraint_type

        # Extract values
        values = []
        for var in scope:
            if var in example:
                values.append(example[var])
            else:
                return True  # Cannot check, assume consistent

        if len(values) != len(scope):
            return True

        # Check based on type
        if constraint_type == "AllDifferent":
            return len(values) == len(set(values))

        elif constraint_type == "Sum":
            # Extract constant from constraint ID
            import re
            match = re.search(r'_(\d+)', constraint.id)
            if match:
                constant = int(match.group(1))
                actual_sum = sum(values)
                if 'leq' in constraint.id:
                    return actual_sum <= constant
                else:
                    return actual_sum == constant

        elif constraint_type == "Count":
            # Extract value and constant from constraint ID
            import re
            match = re.search(r'val(\d+)_cnt(\d+)', constraint.id)
            if match:
                target_value = int(match.group(1))
                constant = int(match.group(2))
                count = values.count(target_value)
                if 'leq' in constraint.id:
                    return count <= constant
                else:
                    return count == constant

        return True

    @staticmethod
    def _rank_repairs(
        repairs: List[Constraint],
        positive_examples: List[Dict],
        ml_prior: 'MLPriorEstimator' = None
    ) -> List[Constraint]:
        """
        Rank repair hypotheses by plausibility score.

        Score combines:
        1. ML prior (if available)
        2. Arity (prefer larger scopes - closer to original)
        3. Structural metrics
        4. Frequency analysis on E+ (how consistent removed variable is)
        """
        scored_repairs = []

        for repair in repairs:
            score = 0.0

            # 1. ML Prior (30% weight)
            if ml_prior:
                try:
                    prior = ml_prior.estimate_prior(repair)
                    score += 0.3 * prior
                except:
                    score += 0.3 * 0.5  # Default

            # 2. Arity preference (25% weight) - prefer larger scopes
            max_arity = max(r.arity for r in repairs) if repairs else 1
            if max_arity > 0:
                score += 0.25 * (repair.arity / max_arity)

            # 3. Structural coherence (25% weight)
            # Prefer constraints where variables have similar naming patterns
            structural_score = CounterexampleRepair._structural_coherence(repair.scope)
            score += 0.25 * structural_score

            # 4. Frequency consistency (20% weight)
            # Prefer repairs where removed variable shows inconsistent pattern in E+
            if positive_examples and hasattr(repair, 'parent_id'):
                # Find which variable was removed (compare with parent)
                parent_scope = None
                for other_repair in repairs:
                    if other_repair.parent_id and len(other_repair.scope) > len(repair.scope):
                        parent_scope = other_repair.scope
                        break

                if parent_scope:
                    removed_var = [v for v in parent_scope if v not in repair.scope]
                    if removed_var:
                        freq_score = CounterexampleRepair._frequency_consistency(
                            removed_var[0], repair, positive_examples
                        )
                        score += 0.2 * freq_score
                else:
                    score += 0.2 * 0.5  # Neutral if cannot determine

            repair.confidence = score
            scored_repairs.append((score, repair))

        # Sort by score (descending)
        scored_repairs.sort(key=lambda x: x[0], reverse=True)

        return [repair for score, repair in scored_repairs]

    @staticmethod
    def _frequency_consistency(
        removed_var: str,
        repair: Constraint,
        positive_examples: List[Dict]
    ) -> float:
        """
        Calculate frequency consistency score for removed variable.

        Higher score = variable shows inconsistent pattern in E+,
        indicating it was correctly identified as spurious.

        For Sum/Count: Check if removed variable has unusual values/behavior.
        For AllDifferent: Not applicable (already precisely identified).
        """
        if not positive_examples or removed_var not in positive_examples[0]:
            return 0.5  # Neutral

        constraint_type = repair.constraint_type

        if constraint_type == "AllDifferent":
            # AllDifferent already has precise identification
            return 0.5  # Neutral

        # Extract values of removed variable across E+
        removed_var_values = []
        for ex in positive_examples:
            if removed_var in ex:
                removed_var_values.append(ex[removed_var])

        if not removed_var_values:
            return 0.5

        # Extract values of remaining variables in repair scope
        remaining_values = []
        for var in repair.scope:
            for ex in positive_examples:
                if var in ex:
                    remaining_values.append(ex[var])

        if not remaining_values:
            return 0.5

        # Calculate statistics
        removed_mean = sum(removed_var_values) / len(removed_var_values)
        removed_std = (sum((v - removed_mean)**2 for v in removed_var_values) / len(removed_var_values))**0.5

        remaining_mean = sum(remaining_values) / len(remaining_values)
        remaining_std = (sum((v - remaining_mean)**2 for v in remaining_values) / len(remaining_values))**0.5

        # If removed variable has significantly different distribution, higher score
        if remaining_std > 0 and removed_std > 0:
            mean_diff = abs(removed_mean - remaining_mean) / max(remaining_mean, 1)
            std_ratio = abs(removed_std - remaining_std) / max(remaining_std, 1)

            # Normalize to [0, 1]
            consistency_score = min(1.0, (mean_diff + std_ratio) / 2)
            return consistency_score

        return 0.5

    @staticmethod
    def _structural_coherence(scope: List[str]) -> float:
        """
        Measure structural coherence of variable names in scope.
        Higher = more coherent (similar naming patterns).
        """
        if len(scope) <= 1:
            return 1.0

        import re

        # Extract naming patterns
        patterns = []
        for var in scope:
            # Extract prefix and indices
            match = re.match(r'^([a-zA-Z_]+?)[\[\(]?(\d+)', var)
            if match:
                patterns.append(match.group(1))

        if not patterns:
            return 0.5

        # Calculate coherence: fraction with same prefix
        from collections import Counter
        counts = Counter(patterns)
        most_common_count = counts.most_common(1)[0][1] if counts else 0

        return most_common_count / len(patterns)


class IntelligentSubsetExplorer:
    """
    Implements the Intelligent Subset Exploration mechanism.

    Uses data-driven "culprit scores" to identify the most likely
    incorrect variable in a rejected constraint's scope.

    NOTE: This is the original heuristic-based approach.
    CounterexampleRepair is the newer, more principled approach.
    """

    @staticmethod
    def generate_informed_subsets(
        rejected_constraint: Constraint,
        positive_examples: List[Dict],
        learned_globals: List[Constraint],
        config: HCARConfig,
        variables: Dict[str, Any] = None
    ) -> List[Constraint]:
        """
        Generate informed subsets by removing the most likely culprit variable.

        Args:
            rejected_constraint: The constraint that was refuted
            positive_examples: Initial positive examples (ground truth)
            learned_globals: Already validated global constraints
            config: HCAR configuration
            variables: Variable objects dict (for creating CPMpy constraints)

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
                rejected_constraint, new_scope, culprit_var, variables
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
        removed_var: str,
        variables: Dict[str, Any] = None
    ) -> Optional[Constraint]:
        """Create a new constraint candidate with reduced scope."""
        if len(new_scope) < 2:
            return None

        new_id = f"{parent.id}_sub_{removed_var}"

        # Create actual CPMpy constraint object with reduced scope
        constraint_obj = None
        if CPMPY_AVAILABLE and variables:
            try:
                # Get variable objects for new scope
                constraint_vars = [variables[v] for v in new_scope if v in variables]

                if len(constraint_vars) == len(new_scope):
                    # All variables found, create constraint based on type
                    if parent.constraint_type == "AllDifferent":
                        from cpmpy import AllDifferent
                        constraint_obj = AllDifferent(constraint_vars)

                    elif parent.constraint_type == "Sum":
                        # Extract constant from parent ID (e.g., "sum_eq_cpu_PM_11" -> 11)
                        import re
                        match = re.search(r'_(\d+)', parent.id)
                        if match:
                            constant = int(match.group(1))
                            if 'leq' in parent.id:
                                constraint_obj = (sum(constraint_vars) <= constant)
                            else:
                                constraint_obj = (sum(constraint_vars) == constant)

                    elif parent.constraint_type == "Count":
                        # Extract value and constant from parent ID
                        import re
                        match = re.search(r'val(\d+)_cnt(\d+)', parent.id)
                        if match:
                            target_value = int(match.group(1))
                            constant = int(match.group(2))
                            from cpmpy import Count
                            if 'leq' in parent.id:
                                constraint_obj = (Count(constraint_vars, target_value) <= constant)
                            else:
                                constraint_obj = (Count(constraint_vars, target_value) == constant)

            except Exception as e:
                logger.warning(f"Failed to create constraint object for subset {new_id}: {e}")

        new_constraint = Constraint(
            id=new_id,
            constraint=constraint_obj,
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


class HeuristicSubsetExplorer:
    """
    Implements the Heuristic Subset Exploration mechanism (baseline).

    Uses simple positional heuristics (first/middle/last) to identify
    which variable to remove from a rejected constraint's scope.

    This is the baseline approach to demonstrate that intelligent
    culprit scores outperform blind positional guessing.
    """

    @staticmethod
    def generate_informed_subsets(
        rejected_constraint: Constraint,
        positive_examples: List[Dict],
        learned_globals: List[Constraint],
        config: HCARConfig,
        variables: Dict[str, Any] = None
    ) -> List[Constraint]:
        """
        Generate subsets using positional heuristics.

        Tries removing variables from first, middle, and last positions
        without any data-driven analysis.

        Args:
            rejected_constraint: The constraint that was refuted
            positive_examples: Initial positive examples (unused in heuristic)
            learned_globals: Already validated global constraints (unused)
            config: HCAR configuration
            variables: Variable objects dict (for creating CPMpy constraints)

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

        # POSITIONAL HEURISTIC: Try removing first, middle, or last variable
        positions_to_try = []

        # First position
        positions_to_try.append(0)

        # Middle position (if scope is large enough)
        if len(scope) > 3:
            positions_to_try.append(len(scope) // 2)

        # Last position (only if we don't already have 2 positions)
        if len(positions_to_try) < 2 and len(scope) > 2:
            positions_to_try.append(len(scope) - 1)

        logger.info(f"Heuristic exploration for {rejected_constraint.id}: "
                   f"trying positions {positions_to_try} (first/middle/last)")

        # Generate new candidates by removing variables at these positions
        new_candidates = []

        for position in positions_to_try:
            removed_var = scope[position]
            new_scope = [v for i, v in enumerate(scope) if i != position]

            # Create new constraint with reduced scope
            new_constraint = IntelligentSubsetExplorer._create_subset_constraint(
                rejected_constraint, new_scope, removed_var, variables
            )

            if new_constraint:
                new_candidates.append(new_constraint)
                logger.info(f"  → Heuristic subset: removed {removed_var} "
                           f"(position {position})")

        return new_candidates


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
        Update constraint confidence using unified probabilistic belief update.

        ROBUST TO NOISY ORACLES: Both oracle responses trigger probabilistic updates
        rather than deterministic hard refutation. This makes the system resilient
        to occasional oracle errors.

        Formulas:
        - If response == INVALID (evidence supports constraint):
          P_new(c) = P_old(c) + (1 - P_old(c)) * (1 - alpha)
        - If response == VALID (evidence refutes constraint):
          P_new(c) = P_old(c) * alpha

        Args:
            current_prob: Current P(c)
            query: The query assignment (unused, kept for interface compatibility)
            response: Oracle response (Valid/Invalid)
            constraint: The constraint being tested (unused, kept for interface)
            alpha: Noise parameter (probability of oracle error), default 0.1

        Returns:
            Updated probability in [0, 1]
        """
        if response == OracleResponse.INVALID:
            # Query violated constraint and was correctly rejected
            # This is positive evidence for the constraint
            # Increase confidence multiplicatively
            new_prob = current_prob + (1 - current_prob) * (1 - alpha)

        else:  # response == OracleResponse.VALID
            # Query violated constraint but was accepted by oracle
            # This is negative evidence - the constraint is likely wrong
            # Decrease confidence multiplicatively (probabilistic, not hard refutation)
            new_prob = current_prob * alpha

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

        # Conditionally instantiate subset explorer based on config
        if config.use_intelligent_subsets:
            self.subset_explorer = IntelligentSubsetExplorer()
            logger.info("Using IntelligentSubsetExplorer (data-driven culprit scores)")
        else:
            self.subset_explorer = HeuristicSubsetExplorer()
            logger.info("Using HeuristicSubsetExplorer (positional heuristics)")

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

        # Log ground truth constraints if available
        if target_model is not None:
            self._log_ground_truth(target_model, variables)

        # Store confirmed solutions
        self.confirmed_solutions = positive_examples.copy()

        # Phase 1: Passive Candidate Generation
        logger.info("\n--- Phase 1: Passive Candidate Generation ---")
        self._phase1_passive_generation(positive_examples, variables, domains)
        
        # Phase 2: Query-Driven Interactive Refinement
        logger.info("\n--- Phase 2: Query-Driven Interactive Refinement ---")
        self._phase2_interactive_refinement(oracle_func, variables, domains)

        # HCAR-NoRefine: Accept all unvalidated globals without refinement
        if self.config.total_budget == 0 and self.B_globals:
            logger.info(f"NoRefine mode: Accepting all {len(self.B_globals)} global candidates without validation")
            logger.info("WARNING: This may include over-fitted constraints (expected to degrade recall)")
            self.C_validated_globals.extend(list(self.B_globals))
            self.B_globals.clear()

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
    
    def _log_ground_truth(self, target_model: List, variables: Dict[str, Any]):
        """Log ground truth constraints to file for analysis."""
        log_file = "ground_truth.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)

        logger.info("="*80)
        logger.info("GROUND TRUTH CONSTRAINTS")
        logger.info("="*80)
        logger.info(f"Total constraints: {len(target_model)}")
        logger.info("")

        # Analyze constraint types
        from collections import Counter
        constraint_types = Counter()

        for idx, c in enumerate(target_model):
            constraint_str = str(c)
            constraint_types[type(c).__name__] += 1

            logger.info(f"{idx+1}. {constraint_str}")

            # Try to extract more details
            if hasattr(c, 'name'):
                logger.info(f"   Name: {c.name}")
            if hasattr(c, 'args'):
                logger.info(f"   Args: {len(c.args)}")

        logger.info("")
        logger.info("Constraint Type Summary:")
        for ctype, count in sorted(constraint_types.items()):
            logger.info(f"  {ctype}: {count}")

        logger.info("="*80)
        logger.info(f"Ground truth log written to: {log_file}")

        logger.removeHandler(file_handler)
        file_handler.close()

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

        # Inject overfitted constraints for experimental validation
        if self.config.inject_overfitted:
            logger.info(f"Config.inject_overfitted is True, calling injection method...")
            self._inject_overfitted_constraints(positive_examples, variables, domains)

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
        Pattern-based extraction of global constraints.

        Detects three types of patterns:
        1. AllDifferent: All values in a group are distinct
        2. Sum: Sum of variables equals/bounded by a constant
        3. Count: Count of value occurrences equals/bounded by a constant
        """
        # Setup detailed logging to file
        log_file = "pattern_detection.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)

        logger.info("="*80)
        logger.info("PATTERN DETECTION - PHASE 1")
        logger.info("="*80)
        logger.info(f"Variables: {len(variables)}")
        logger.info(f"Examples: {len(positive_examples)}")
        logger.info("")

        candidates = set()

        # 1. Extract AllDifferent constraints (standard patterns)
        logger.info("--- Extracting AllDifferent Patterns ---")
        alldiff = self._extract_alldifferent_patterns(positive_examples, variables)
        candidates.update(alldiff)
        logger.info(f"Found {len(alldiff)} AllDifferent candidates (row/col/block patterns)")
        logger.info("")

        # 1b. Extract cross-boundary AllDifferent patterns
        logger.info("--- Extracting Cross-Boundary AllDifferent Patterns ---")
        cross_boundary = self._extract_cross_boundary_alldiff(positive_examples, variables)
        candidates.update(cross_boundary)
        logger.info(f"Found {len(cross_boundary)} cross-boundary AllDifferent candidates")
        logger.info("")

        # 2. Extract Sum constraints
        logger.info("--- Extracting Sum Patterns ---")
        sum_constraints = self._extract_sum_patterns(positive_examples, variables, domains)
        candidates.update(sum_constraints)
        logger.info(f"Found {len(sum_constraints)} Sum candidates")
        logger.info("")

        # 3. Extract Count constraints
        logger.info("--- Extracting Count Patterns ---")
        count_constraints = self._extract_count_patterns(positive_examples, variables, domains)
        candidates.update(count_constraints)
        logger.info(f"Found {len(count_constraints)} Count candidates")
        logger.info("")

        logger.info(f"TOTAL: {len(candidates)} global constraint candidates")
        logger.info("="*80)
        logger.info(f"Detailed log written to: {log_file}")

        # Remove file handler
        logger.removeHandler(file_handler)
        file_handler.close()

        return candidates

    def _extract_alldifferent_patterns(
        self,
        positive_examples: List[Dict],
        variables: Dict[str, Any]
    ) -> Set[Constraint]:
        """Extract AllDifferent constraint candidates."""
        candidates = set()

        # Group variables by naming pattern (e.g., row, column, block)
        logger.info("Grouping variables by pattern...")
        var_groups = self._group_variables_by_pattern(list(variables.keys()))
        logger.info(f"Found {len(var_groups)} variable groups:")
        for group_name in sorted(var_groups.keys()):
            logger.info(f"  {group_name}: {len(var_groups[group_name])} variables")

        logger.info("")
        logger.info("Checking AllDifferent property for each group:")

        for group_name, var_names in sorted(var_groups.items()):
            if len(var_names) >= 2:
                logger.info(f"\n  Group: {group_name} ({len(var_names)} vars)")
                logger.info(f"    Variables: {var_names[:10]}{'...' if len(var_names) > 10 else ''}")

                # Check if AllDifferent holds in all examples
                holds_in_all = True
                for ex_idx, example in enumerate(positive_examples):
                    values = [example.get(v) for v in var_names if v in example]
                    if len(values) == len(var_names):
                        unique_values = len(set(values))
                        is_alldiff = (len(values) == unique_values)

                        if ex_idx < 2:  # Log first 2 examples
                            logger.info(f"    Example {ex_idx+1}: {len(values)} vars, {unique_values} unique -> {'PASS' if is_alldiff else 'FAIL'}")
                            if not is_alldiff:
                                # Find duplicates
                                from collections import Counter
                                counts = Counter(values)
                                duplicates = {val: count for val, count in counts.items() if count > 1}
                                logger.info(f"      Duplicates: {duplicates}")

                        if not is_alldiff:
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
                        logger.info(f"    RESULT: ACCEPTED as candidate")
                    except Exception as e:
                        logger.warning(f"    RESULT: FAILED to create - {e}")
                else:
                    reason = "too small" if len(var_names) <= 2 else "not all different"
                    logger.info(f"    RESULT: REJECTED ({reason})")

        return candidates

    def _extract_cross_boundary_alldiff(
        self,
        positive_examples: List[Dict],
        variables: Dict[str, Any]
    ) -> Set[Constraint]:
        """
        Extract AllDifferent constraints that span across row/day boundaries.

        Examples:
        - Consecutive days: last shift of day X + first shift of day X+1
        - Cross-shift patterns: specific combinations across shifts

        This handles patterns that _group_variables_by_pattern cannot detect.
        """
        import re
        candidates = set()

        logger.info("Parsing variable indices...")

        # Parse variable indices
        var_indices = {}
        for var_name in variables.keys():
            nums = re.findall(r'\d+', var_name)
            if len(nums) >= 3:
                idx1, idx2, idx3 = int(nums[0]), int(nums[1]), int(nums[2])
                var_indices[var_name] = (idx1, idx2, idx3)
            elif len(nums) >= 2:
                idx1, idx2 = int(nums[0]), int(nums[1])
                var_indices[var_name] = (idx1, idx2, None)

        if not var_indices:
            logger.info("  No multi-dimensional variables found")
            return candidates

        # Determine dimensions
        max_idx1 = max(idx[0] for idx in var_indices.values())
        max_idx2 = max(idx[1] for idx in var_indices.values())

        logger.info(f"  Variable structure: {len(var_indices)} vars with dimensions [{max_idx1+1}, {max_idx2+1}, ...]")

        # Pattern 1: Consecutive "rows" or "days" with specific "columns" or "shifts"
        # Example: last column of row X + first column of row X+1
        logger.info("\n  Pattern: Consecutive rows/days with specific columns/shifts")

        for idx1 in range(max_idx1):
            # Try to find variables at "boundary" positions
            # Last position in idx1, first position in idx1+1

            # Collect variables with idx1 and last idx2 (max_idx2)
            last_vars = [v for v, (i1, i2, i3) in var_indices.items()
                        if i1 == idx1 and i2 == max_idx2]

            # Collect variables with idx1+1 and first idx2 (0)
            first_vars = [v for v, (i1, i2, i3) in var_indices.items()
                         if i1 == idx1+1 and i2 == 0]

            if not last_vars or not first_vars:
                continue

            combined_vars = last_vars + first_vars

            if len(combined_vars) < 3:  # Need at least 3 for AllDifferent
                continue

            logger.info(f"\n    Testing boundary between {idx1} and {idx1+1}:")
            logger.info(f"      Group: {len(combined_vars)} vars = {len(last_vars)} from [{idx1},{max_idx2}] + {len(first_vars)} from [{idx1+1},0]")

            # Check if AllDifferent holds in all examples
            holds_in_all = True
            for ex_idx, example in enumerate(positive_examples):
                values = [example.get(v) for v in combined_vars if v in example]
                if len(values) == len(combined_vars):
                    unique_values = len(set(values))
                    is_alldiff = (len(values) == unique_values)

                    if ex_idx < 2:  # Log first 2 examples
                        logger.info(f"      Example {ex_idx+1}: {len(values)} vars, {unique_values} unique -> {'PASS' if is_alldiff else 'FAIL'}")
                        if not is_alldiff:
                            from collections import Counter
                            counts = Counter(values)
                            duplicates = {val: count for val, count in counts.items() if count > 1}
                            logger.info(f"        Duplicates: {duplicates}")

                    if not is_alldiff:
                        holds_in_all = False
                        break

            if holds_in_all:
                try:
                    if CPMPY_AVAILABLE:
                        from cpmpy import AllDifferent
                        constraint_vars = [variables[v] for v in combined_vars if v in variables]
                        constraint_obj = AllDifferent(constraint_vars)
                    else:
                        constraint_obj = None

                    candidate = Constraint(
                        id=f"alldiff_boundary_{idx1}_{idx1+1}",
                        constraint=constraint_obj,
                        scope=combined_vars,
                        constraint_type="AllDifferent",
                        arity=len(combined_vars),
                        level=0,
                        confidence=0.5
                    )
                    candidates.add(candidate)
                    logger.info(f"      RESULT: ACCEPTED as candidate")
                except Exception as e:
                    logger.warning(f"      RESULT: FAILED to create - {e}")
            else:
                logger.info(f"      RESULT: REJECTED (not all different)")

        return candidates

    def _extract_sum_patterns(
        self,
        positive_examples: List[Dict],
        variables: Dict[str, Any],
        domains: Dict[str, List]
    ) -> Set[Constraint]:
        """
        Extract Sum constraint candidates.

        Looks for patterns like: sum(group) == constant or sum(group) <= constant
        Common in resource allocation, scheduling problems.
        """
        candidates = set()

        # Group variables by common prefix (e.g., "cpu_", "memory_", "assign_")
        var_groups = self._group_variables_by_prefix(list(variables.keys()))

        for group_name, var_names in var_groups.items():
            if len(var_names) < 2:
                continue

            # Check if sum is consistent across all examples
            sum_values = []
            valid = True

            for example in positive_examples:
                values = [example.get(v) for v in var_names if v in example]
                if len(values) == len(var_names):
                    sum_values.append(sum(values))
                else:
                    valid = False
                    break

            if not valid or not sum_values:
                continue

            # Check if sum is constant or bounded
            min_sum = min(sum_values)
            max_sum = max(sum_values)

            # Pattern 1: Sum equals constant (all examples have same sum)
            if min_sum == max_sum:
                constant = min_sum
                try:
                    if CPMPY_AVAILABLE:
                        constraint_vars = [variables[v] for v in var_names if v in variables]
                        constraint_obj = (sum(constraint_vars) == constant)
                    else:
                        constraint_obj = None

                    candidate = Constraint(
                        id=f"sum_eq_{group_name}_{constant}",
                        constraint=constraint_obj,
                        scope=var_names,
                        constraint_type="Sum",
                        arity=len(var_names),
                        level=0,
                        confidence=0.5
                    )
                    candidates.add(candidate)
                    logger.debug(f"  Found Sum==: {candidate.id}")
                except Exception as e:
                    logger.warning(f"Failed to create Sum constraint: {e}")

            # Pattern 2: Sum bounded by maximum (sum <= constant)
            # Only if there's variation in the sum values
            elif max_sum > min_sum:
                # Check if max_sum could be a capacity bound
                # Heuristic: if max_sum appears in multiple examples or seems like a round number
                if max_sum % 5 == 0 or max_sum % 10 == 0 or sum_values.count(max_sum) >= 2:
                    try:
                        if CPMPY_AVAILABLE:
                            constraint_vars = [variables[v] for v in var_names if v in variables]
                            constraint_obj = (sum(constraint_vars) <= max_sum)
                        else:
                            constraint_obj = None

                        candidate = Constraint(
                            id=f"sum_leq_{group_name}_{max_sum}",
                            constraint=constraint_obj,
                            scope=var_names,
                            constraint_type="Sum",
                            arity=len(var_names),
                            level=0,
                            confidence=0.3  # Lower confidence for bounded constraints
                        )
                        candidates.add(candidate)
                        logger.debug(f"  Found Sum<=: {candidate.id}")
                    except Exception as e:
                        logger.warning(f"Failed to create Sum<= constraint: {e}")

        return candidates

    def _extract_count_patterns(
        self,
        positive_examples: List[Dict],
        variables: Dict[str, Any],
        domains: Dict[str, List]
    ) -> Set[Constraint]:
        """
        Extract Count constraint candidates.

        Looks for patterns like: count(group, value) == constant or count(group, value) <= constant
        Common in scheduling, assignment problems.
        """
        candidates = set()

        logger.info("Detecting value frequency patterns...")

        # Strategy 1: Group variables by common prefix
        logger.info("\nStrategy 1: Grouping by common prefix")
        var_groups = self._group_variables_by_prefix(list(variables.keys()))
        logger.info(f"Found {len(var_groups)} prefix groups")

        # Strategy 2: Use ALL variables as one group (for global count constraints)
        logger.info("\nStrategy 2: Checking global count patterns")
        var_groups['all_vars'] = list(variables.keys())
        logger.info(f"Added 'all_vars' group with {len(var_groups['all_vars'])} variables")

        for group_name, var_names in sorted(var_groups.items()):
            if len(var_names) < 2:
                continue

            logger.info(f"\n  Checking group: {group_name} ({len(var_names)} vars)")

            # For each possible value, check if count is consistent
            # First, determine the range of values that appear
            all_values = set()
            for example in positive_examples:
                for v in var_names:
                    if v in example:
                        all_values.add(example[v])

            logger.info(f"    Values appearing in examples: {sorted(all_values)}")

            # For each value, check if count pattern exists
            for target_value in sorted(all_values):
                count_values = []
                valid = True

                for example in positive_examples:
                    values = [example.get(v) for v in var_names if v in example]
                    if len(values) == len(var_names):
                        count = values.count(target_value)
                        count_values.append(count)
                    else:
                        valid = False
                        break

                if not valid or not count_values:
                    continue

                min_count = min(count_values)
                max_count = max(count_values)

                logger.info(f"      Value {target_value}: counts = {count_values} (min={min_count}, max={max_count})")

                # IMPROVED HEURISTIC: Default to <= (upper bound)
                # Rationale: In scheduling/resource allocation, bounds are more common than exact constraints

                # Skip if count is always 0 (value doesn't appear significantly)
                if max_count == 0:
                    logger.info(f"        SKIPPED: Value never appears")
                    continue

                # Always generate <= constraint with max observed count
                # This is safer than assuming == when all counts are equal
                try:
                    if CPMPY_AVAILABLE:
                        from cpmpy import Count
                        constraint_vars = [variables[v] for v in var_names if v in variables]
                        constraint_obj = (Count(constraint_vars, target_value) <= max_count)
                    else:
                        constraint_obj = None

                    constraint_type_str = "==" if min_count == max_count else "<="

                    candidate = Constraint(
                        id=f"count_leq_{group_name}_val{target_value}_cnt{max_count}",
                        constraint=constraint_obj,
                        scope=var_names,
                        constraint_type="Count",
                        arity=len(var_names),
                        level=0,
                        confidence=0.6 if min_count == max_count else 0.5  # Higher confidence if constant
                    )
                    candidates.add(candidate)

                    if min_count == max_count:
                        logger.info(f"        ACCEPTED: Count({group_name[:20]}, {target_value}) <= {max_count} (constant in examples)")
                    else:
                        logger.info(f"        ACCEPTED: Count({group_name[:20]}, {target_value}) <= {max_count} (varies {min_count}-{max_count})")

                except Exception as e:
                    logger.warning(f"        FAILED: Count constraint - {e}")

        # Apply domain expert hints if enabled
        if self.config.enable_hint_normalization and candidates:
            candidates = self._apply_parameter_normalization(candidates, "Count", variables)

        return candidates

    def _apply_parameter_normalization(
        self,
        candidates: Set[Constraint],
        constraint_type: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Set[Constraint]:
        """
        Apply parameter normalization to handle under-fitting from sparse examples.

        Strategy: For Count constraints, normalize outlier bounds to match majority.
        This handles cases where some values never reach their true capacity in examples.

        Example:
          Detected: Count(X, v1) <= 6, Count(X, v2) <= 6, Count(X, v3) <= 5, Count(X, v4) <= 6
          Normalized: All use <= 6 (majority bound)
        """
        if constraint_type != "Count" or not candidates:
            return candidates

        logger.info("\n--- Applying Parameter Normalization ---")
        logger.info("Strategy: Normalize Count constraint bounds using majority voting")

        import re
        from collections import Counter

        # Group constraints by scope
        scope_groups = {}
        for c in candidates:
            if c.constraint_type != "Count":
                continue

            # Use scope as grouping key
            scope_key = tuple(sorted(c.scope))
            if scope_key not in scope_groups:
                scope_groups[scope_key] = []
            scope_groups[scope_key].append(c)

        normalized_candidates = set()

        for scope_key, group in scope_groups.items():
            if len(group) < 2:
                # Single constraint, no normalization needed
                normalized_candidates.update(group)
                continue

            # Extract bounds from all constraints in group
            constraint_bounds = []
            for c in group:
                match = re.search(r'cnt(\d+)', c.id)
                if match:
                    bound = int(match.group(1))
                    constraint_bounds.append((c, bound))

            if not constraint_bounds:
                normalized_candidates.update(group)
                continue

            bounds = [b for _, b in constraint_bounds]
            bound_counts = Counter(bounds)

            # Find majority bound
            majority_bound, majority_count = bound_counts.most_common(1)[0]

            # If no clear majority (< 60%), use max bound as safety
            if majority_count < len(bounds) * 0.6:
                normalized_bound = max(bounds)
                reason = "max (no clear majority)"
            else:
                normalized_bound = majority_bound
                reason = f"majority ({majority_count}/{len(bounds)})"

            logger.info(f"\n  Group with {len(group)} Count constraints:")
            logger.info(f"    Bounds detected: {bounds}")
            logger.info(f"    Majority bound: {majority_bound} (appears {majority_count} times)")
            logger.info(f"    Normalized bound: {normalized_bound} ({reason})")

            # Recreate constraints with normalized bound
            for c, old_bound in constraint_bounds:
                if old_bound == normalized_bound:
                    # No change needed
                    normalized_candidates.add(c)
                    logger.info(f"      {c.id}: UNCHANGED")
                else:
                    # Extract target value from constraint ID
                    match = re.search(r'val(\d+)', c.id)
                    target_value = int(match.group(1)) if match else None

                    if target_value is None:
                        normalized_candidates.add(c)
                        logger.info(f"      {c.id}: FAILED to extract value")
                        continue

                    # Create normalized constraint
                    if CPMPY_AVAILABLE and variables:
                        from cpmpy import Count
                        constraint_vars = [variables.get(v) for v in c.scope]
                        constraint_vars = [v for v in constraint_vars if v is not None]
                        if constraint_vars:
                            constraint_obj = (Count(constraint_vars, target_value) <= normalized_bound)
                        else:
                            constraint_obj = None
                    else:
                        constraint_obj = None

                    normalized_c = Constraint(
                        id=c.id.replace(f"cnt{old_bound}", f"cnt{normalized_bound}"),
                        constraint=constraint_obj,
                        scope=c.scope,
                        constraint_type=c.constraint_type,
                        arity=c.arity,
                        level=c.level,
                        confidence=min(c.confidence * 1.15, 0.9)  # Boost confidence slightly
                    )
                    normalized_candidates.add(normalized_c)
                    logger.info(f"      {c.id}: NORMALIZED from {old_bound} -> {normalized_bound}")

        logger.info(f"\nNormalization complete: {len(normalized_candidates)} constraints")
        return normalized_candidates

    def _generate_fixed_bias_simple(
        self,
        variables: Dict[str, Any],
        domains: Dict[str, List],
        positive_examples: List[Dict]
    ) -> Set[Constraint]:
        """
        Generate complete fixed-arity bias using CPMpy constraints.

        This generates all binary constraints of all types (==, !=, <, >, <=, >=)
        and prunes them according to positive examples E+ (CONSTRAINT 2).

        Args:
            variables: Dict mapping variable names to CPMpy variables
            domains: Dict mapping variable names to domain tuples (lb, ub)
            positive_examples: List of positive examples (trusted E+)

        Returns:
            Set of Constraint objects forming the initial fixed-arity bias B_fixed
        """
        if not CPMPY_AVAILABLE:
            logger.warning("CPMpy not available - cannot generate bias")
            return set()

        logger.info(f"  Generating complete binary constraint bias...")
        logger.info(f"  Variables: {len(variables)}")
        logger.info(f"  Positive examples for pruning: {len(positive_examples)}")

        candidates = set()
        var_names = list(variables.keys())

        # Define all binary constraint types
        constraint_types = [
            ('==', 'Equal'),
            ('!=', 'NotEqual'),
            ('<', 'LessThan'),
            ('>', 'GreaterThan'),
            ('<=', 'LessThanOrEqual'),
            ('>=', 'GreaterThanOrEqual')
        ]

        # Statistics for reporting
        stats = {
            'generated': 0,
            'pruned': 0,
            'kept': 0
        }

        # Generate all pairs of variables
        total_pairs = len(var_names) * (len(var_names) - 1) // 2
        logger.info(f"  Generating constraints for {total_pairs} variable pairs...")

        for i, var1_name in enumerate(var_names):
            var1 = variables[var1_name]

            for var2_name in var_names[i+1:]:
                var2 = variables[var2_name]

                # Generate all constraint types for this pair
                for op, ctype_name in constraint_types:
                    try:
                        # Create CPMpy constraint
                        if op == '==':
                            cpm_constraint = (var1 == var2)
                        elif op == '!=':
                            cpm_constraint = (var1 != var2)
                        elif op == '<':
                            cpm_constraint = (var1 < var2)
                        elif op == '>':
                            cpm_constraint = (var1 > var2)
                        elif op == '<=':
                            cpm_constraint = (var1 <= var2)
                        elif op == '>=':
                            cpm_constraint = (var1 >= var2)
                        else:
                            continue

                        stats['generated'] += 1

                        # Create Constraint wrapper
                        constraint_id = f"{ctype_name}_{var1_name}_{var2_name}"
                        candidate = Constraint(
                            id=constraint_id,
                            constraint=cpm_constraint,
                            scope=[var1_name, var2_name],
                            constraint_type=ctype_name,
                            arity=2,
                            level=0,
                            confidence=0.5
                        )

                        # CONSTRAINT 2: Prune using ONLY positive examples E+
                        # Check if this constraint is violated by any positive example
                        violated = self._is_violated_by_examples(
                            candidate, positive_examples, variables
                        )

                        if violated:
                            stats['pruned'] += 1
                        else:
                            candidates.add(candidate)
                            stats['kept'] += 1

                    except Exception as e:
                        logger.warning(f"Failed to create {ctype_name} constraint for {var1_name}, {var2_name}: {e}")

        # Report statistics
        logger.info(f"  Bias generation complete:")
        logger.info(f"    Total generated: {stats['generated']}")
        logger.info(f"    Pruned by E+: {stats['pruned']}")
        logger.info(f"    Kept in B_fixed: {stats['kept']}")
        logger.info(f"  Final B_fixed size: {len(candidates)} constraints")

        return candidates

    def _is_violated_by_examples(
        self,
        candidate: Constraint,
        positive_examples: List[Dict],
        variables: Dict[str, Any]
    ) -> bool:
        """
        Check if a constraint is violated by any positive example.

        This is the ONLY pruning mechanism allowed in Phase 1 (CONSTRAINT 2).

        Args:
            candidate: The constraint to check
            positive_examples: Trusted positive examples E+
            variables: Variable mapping

        Returns:
            True if violated by at least one example, False otherwise
        """
        if not CPMPY_AVAILABLE:
            return False

        try:
            # Get the CPMpy constraint
            cpm_constraint = candidate.constraint
            scope = candidate.scope

            # Check each positive example
            for example in positive_examples:
                # Check if all variables in scope are present in example
                if not all(var_name in example for var_name in scope):
                    continue

                # Create a model with just this constraint and the example values
                model = Model([cpm_constraint])

                # Add constraints fixing variables to example values
                for var_name in scope:
                    var = variables[var_name]
                    value = example[var_name]
                    model += (var == value)

                # If the model is UNSAT, the constraint is violated by this example
                if not model.solve():
                    return True  # Constraint violated - should be pruned

            # Constraint holds on all positive examples
            return False

        except Exception as e:
            logger.warning(f"Error checking constraint {candidate.id}: {e}")
            # Conservative: keep constraint if we can't verify
            return False
    
    def _group_variables_by_prefix(self, var_names: List[str]) -> Dict[str, List[str]]:
        """
        Group variables by common prefix (for Sum/Count pattern detection).

        Examples:
        - "cpu_PM1", "cpu_PM2" -> group "cpu"
        - "assign_VM1", "assign_VM2" -> group "assign"
        - "memory_PM1", "memory_PM2" -> group "memory"
        """
        import re

        groups = {}

        for var_name in var_names:
            # Extract prefix (text before first number or underscore+number)
            match = re.match(r'^([a-zA-Z_]+?)(?:_?\d|$)', var_name)
            if match:
                prefix = match.group(1).rstrip('_')
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(var_name)

        # Filter out groups with only 1 variable
        groups = {k: v for k, v in groups.items() if len(v) > 1}

        return groups

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

    def _inject_overfitted_constraints(
        self,
        positive_examples: List[Dict],
        variables: Dict[str, Any],
        domains: Dict[str, List]
    ):
        """
        Inject deliberately overfitted global constraints for experimental validation.

        These constraints:
        1. Hold on all 5 positive examples (so they pass passive validation)
        2. Are over-scoped (include too many variables)
        3. Should be rejected during Phase 2 refinement
        4. Test if intelligent subset exploration can correct them

        This validates Hypothesis 1 from CLAUDE.md:
        "Passive learning from sparse data (5 examples) produces over-fitted
        global constraints that cause catastrophic accuracy loss"
        """
        if not CPMPY_AVAILABLE:
            return

        logger.info("EXPERIMENTAL: Injecting overfitted constraints for validation")
        injected_count = 0

        # SIMPLE STRATEGY: Merge two existing constraints of the same type
        # This guarantees they hold on examples (both originals do) but creates an over-scoped constraint

        # Strategy 1: Merge AllDifferent constraints
        alldiff_candidates = [c for c in self.B_globals if c.constraint_type == "AllDifferent"]

        if len(alldiff_candidates) >= 2:
            # Take two different AllDifferent constraints and merge their scopes
            c1, c2 = alldiff_candidates[0], alldiff_candidates[1]

            if len(set(c1.scope) & set(c2.scope)) == 0:  # No overlap
                merged_scope = c1.scope + c2.scope

                # This merged constraint is likely overfitted (too many variables)
                # but holds on examples since both components hold
                try:
                    from cpmpy import AllDifferent
                    constraint_vars = [variables[v] for v in merged_scope if v in variables]
                    constraint_obj = AllDifferent(constraint_vars)

                    overfitted = Constraint(
                        id=f"overfitted_merged_alldiff_{len(merged_scope)}vars",
                        constraint=constraint_obj,
                        scope=merged_scope,
                        constraint_type="AllDifferent",
                        arity=len(merged_scope),
                        level=0,
                        confidence=0.5
                    )

                    self.B_globals.add(overfitted)
                    injected_count += 1
                    logger.info(f"  → Injected overfitted AllDifferent by merging "
                              f"{len(c1.scope)}+{len(c2.scope)} vars = {len(merged_scope)} total")
                except Exception as e:
                    logger.debug(f"Failed to inject merged AllDifferent: {e}")

        # Strategy 2: For Sum constraints, add extra variables
        sum_candidates = [c for c in self.B_globals if c.constraint_type == "Sum"]

        for candidate in sum_candidates[:2]:  # Take first 2
            scope = candidate.scope
            if len(scope) >= 2:
                var_list = list(variables.keys())
                extra_vars = [v for v in var_list if v not in scope]

                if extra_vars:
                    # Find a variable that's always 0 in examples (won't change sum)
                    always_zero = None
                    for var in extra_vars[:5]:  # Check first 5
                        values = [example.get(var, None) for example in positive_examples]
                        if all(v == 0 for v in values if v is not None):
                            always_zero = var
                            break

                    if always_zero:
                        overfitted_scope = scope + [always_zero]

                        # Check sum is consistent
                        sum_values = []
                        for example in positive_examples:
                            values = [example.get(v, 0) for v in overfitted_scope if v in example]
                            if len(values) == len(overfitted_scope):
                                sum_values.append(sum(values))

                        if sum_values and min(sum_values) == max(sum_values):
                            constant = sum_values[0]
                            try:
                                constraint_vars = [variables[v] for v in overfitted_scope if v in variables]
                                constraint_obj = (sum(constraint_vars) == constant)

                                overfitted = Constraint(
                                    id=f"overfitted_sum_{len(overfitted_scope)}vars_{constant}",
                                    constraint=constraint_obj,
                                    scope=overfitted_scope,
                                    constraint_type="Sum",
                                    arity=len(overfitted_scope),
                                    level=0,
                                    confidence=0.5
                                )

                                self.B_globals.add(overfitted)
                                injected_count += 1
                                logger.info(f"  → Injected overfitted Sum with {len(overfitted_scope)} vars "
                                          f"(original: {len(scope)})")
                            except Exception as e:
                                logger.debug(f"Failed to inject overfitted Sum: {e}")

        logger.info(f"EXPERIMENTAL: Injected {injected_count} overfitted constraints (should be rejected in Phase 2)")

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
        # Store variables for subset generation
        self.variables = variables
        self.domains = domains
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

                # UNIFIED PROBABILISTIC UPDATE: Both responses update confidence
                # No more hard refutation - makes system robust to noisy oracles
                candidate.confidence = BayesianUpdater.update_confidence(
                    candidate.confidence,
                    query,
                    response,
                    candidate,
                    self.config.alpha
                )

                logger.info(f"  Updated P={candidate.confidence:.3f}")

                # Store counterexample if this is negative evidence (for later repair)
                if response == OracleResponse.VALID:
                    candidate.counterexample = query
                    # NOTE: We do NOT prune B_fixed here anymore
                    # Pruning will happen when constraint is accepted (P >= theta_max)
            
            # Process candidate based on final confidence
            if candidate.confidence >= self.config.theta_max:
                # Accept constraint
                logger.info(f"✓ ACCEPTED: {candidate.id}")
                self.C_validated_globals.append(candidate)
                self.B_globals.remove(candidate)

                # NEW LOGIC: Prune B_fixed using decomposition of accepted global
                # This is the correct trigger point for noisy oracle robustness
                logger.info(f"  Pruning B_fixed using accepted global constraint...")
                self._prune_fixed_bias_with_accepted_global(candidate)

                # Redistribute unused budget
                budget_pool += max(0, candidate.budget - candidate.budget_used)
            
            elif candidate.confidence <= self.config.theta_min:
                # Reject constraint and explore subsets
                logger.info(f"✗ REJECTED: {candidate.id}")
                self.B_globals.remove(candidate)

                if candidate.level < self.config.max_subset_depth:
                    # Choose repair mechanism based on configuration
                    if self.config.use_counterexample_repair and hasattr(candidate, 'counterexample'):
                        # Use counterexample-driven minimal repair (most advanced)
                        logger.info("Using counterexample-driven minimal repair")
                        new_candidates = CounterexampleRepair.repair_from_counterexample(
                            candidate,
                            candidate.counterexample,
                            self.confirmed_solutions,
                            self.C_validated_globals,
                            self.config,
                            self.variables,
                            self.ml_prior
                        )
                    else:
                        # Fallback to subset explorer (original approach)
                        logger.info("Using culprit score-based subset exploration")
                        new_candidates = self.subset_explorer.generate_informed_subsets(
                            candidate,
                            self.confirmed_solutions,
                            self.C_validated_globals,
                            self.config,
                            self.variables
                        )

                    for new_cand in new_candidates:
                        # Inherit budget from parent
                        new_cand.budget = max(5, candidate.budget // 2)
                        self.B_globals.add(new_cand)
                        logger.info(f"  → Generated repair: {new_cand.id}")
        
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
        According to CONSTRAINT 2 in CLAUDE.md: "Bias pruning MUST only use confirmed
        solutions (E+ and oracle-verified queries)".
        """
        if not CPMPY_AVAILABLE or not hasattr(self, 'variables'):
            return

        to_remove = []
        for constraint in self.B_fixed:
            if constraint.constraint is not None:
                try:
                    # Create a model with the constraint and the solution
                    test_model = Model()
                    test_model += constraint.constraint

                    # Add solution as constraints
                    for var_name, value in solution.items():
                        if var_name in self.variables:
                            var_obj = self.variables[var_name]
                            test_model += (var_obj == value)

                    # If UNSAT, the solution violates this constraint -> remove it
                    if not test_model.solve():
                        to_remove.append(constraint)
                        logger.debug(f"    Removing {constraint.id} (violated by solution)")

                except Exception as e:
                    logger.debug(f"Error checking constraint {constraint.id}: {e}")

        for c in to_remove:
            self.B_fixed.remove(c)

        if to_remove:
            logger.info(f"  Pruned {len(to_remove)} fixed-arity constraints using confirmed solution")

    def _prune_fixed_bias_with_accepted_global(self, accepted_constraint: Constraint):
        """
        Decomposition-based pruning: Remove B_fixed constraints that contradict accepted global.

        NEW LOGIC for noisy oracle robustness:
        When a global constraint is accepted (P >= theta_max), it becomes a "confirmed rule".
        We decompose it and prune contradictory constraints from B_fixed.

        Example: If AllDifferent({x1, x2, x3}) is accepted, then:
        - Binary inequalities x1 != x2, x1 != x3, x2 != x3 are now ground truth
        - Remove contradictory constraints: x1 == x2, x1 == x3, x2 == x3 from B_fixed

        Args:
            accepted_constraint: The global constraint that was just accepted
        """
        if not CPMPY_AVAILABLE or not hasattr(self, 'variables'):
            return

        to_remove = []

        # Decompose the accepted global constraint
        decomposed_constraints = []
        c = accepted_constraint.constraint

        if hasattr(c, 'name') and c.name == "alldifferent":
            # Decompose AllDifferent into binary != constraints
            c_decomposed_list = c.decompose()
            if c_decomposed_list:
                decomposed_constraints.extend(c_decomposed_list[0])
                logger.info(f"  Decomposed {accepted_constraint.id} into {len(c_decomposed_list[0])} binary inequalities")
        else:
            # For other constraint types (Sum, Count), we can't easily decompose
            # Just use the constraint itself
            decomposed_constraints.append(c)
            logger.info(f"  Using {accepted_constraint.id} directly for pruning (non-decomposable)")

        # Now check each B_fixed constraint for contradiction
        for b_constraint in self.B_fixed:
            if b_constraint.constraint is not None:
                try:
                    # Create a model with the B_fixed constraint and all decomposed constraints
                    test_model = Model()
                    test_model += b_constraint.constraint
                    test_model += decomposed_constraints

                    # If UNSAT, they contradict -> remove B_fixed constraint
                    if not test_model.solve():
                        to_remove.append(b_constraint)
                        logger.debug(f"    Removing {b_constraint.id} (contradicts accepted global)")

                except Exception as e:
                    logger.debug(f"Error checking contradiction for {b_constraint.id}: {e}")

        for c in to_remove:
            self.B_fixed.remove(c)

        if to_remove:
            logger.info(f"  Pruned {len(to_remove)} B_fixed constraints that contradict accepted global")
    
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
        num_runs: int = 1,
        config: HCARConfig = None
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

            # Configure method (use passed config or create one)
            if config is None:
                run_config = self._get_method_config(method_name)
            else:
                run_config = config
            
            # Run HCAR
            hcar = HCARFramework(run_config, problem_name=benchmark_name)
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
                domains,
                hcar  # Pass HCAR instance for variable access
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
            # - Intelligent subset exploration with culprit scores
            # - ML prior estimation
            pass

        elif method_name == "HCAR-Heuristic":
            # Use positional heuristics for subset exploration (baseline)
            config.use_intelligent_subsets = False

        elif method_name == "HCAR-NoRefine":
            # Skip Phase 2 entirely (ablation study)
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
        hcar_instance = None,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate learned model quality using solution-space metrics.

        Returns:
            Dictionary with S-Precision and S-Recall
        """
        # Store variables for validation
        self.variables = variables if hcar_instance is None else hcar_instance.variables
        # Generate sample solutions from both models
        learned_solutions = self._sample_solutions(learned_model, variables, domains, num_samples)
        target_solutions = self._sample_solutions(target_model, variables, domains, num_samples)

        if not target_solutions:
            logger.error("Could not generate target solutions - target model may be UNSAT")
            return {'s_precision': 0.0, 's_recall': 0.0}

        if not learned_solutions:
            logger.warning("Could not generate learned solutions - learned model is UNSAT (over-constrained)")
            # UNSAT model: rejects all solutions (S-Recall = 0%), but has no false positives (S-Precision = N/A)
            # Convention: report S-Precision as 100% (vacuously true - no solutions to be invalid)
            return {'s_precision': 100.0, 's_recall': 0.0}
        
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
                    # Check if all variables have valid values
                    solution = {}
                    valid = True
                    for name, var in variables.items():
                        val = var.value()
                        if val is None:
                            logger.warning(f"Variable {name} has None value after solve()")
                            valid = False
                            break
                        solution[name] = val

                    if not valid:
                        break

                    solutions.append(solution)

                    # Add blocking clause to get different solution
                    blocking = []
                    for var in variables.values():
                        val = var.value()
                        if val is not None:
                            blocking.append(var != val)

                    if blocking:
                        cpm_model += sum(blocking) > 0
                else:
                    break
        
        except Exception as e:
            logger.error(f"Solution sampling error: {e}")
        
        return solutions
    
    def _is_valid_solution(self, solution: Dict, model: List) -> bool:
        """
        Check if solution satisfies all constraints in model.

        Args:
            solution: Variable assignment dict
            model: List of constraints (either Constraint objects or CPMpy constraints)

        Returns:
            True if solution satisfies all constraints, False otherwise
        """
        if not CPMPY_AVAILABLE:
            return True  # Cannot validate without CPMpy

        try:
            # Build a model with all constraints
            cpm_model = Model()

            for constraint in model:
                if isinstance(constraint, Constraint):
                    if constraint.constraint is not None:
                        cpm_model += constraint.constraint
                else:
                    # Direct CPMpy constraint
                    cpm_model += constraint

            # Check if the specific solution satisfies the model
            # We do this by adding the solution as constraints and checking satisfiability
            temp_constraints = []
            for var_name, value in solution.items():
                if var_name in self.variables:
                    var_obj = self.variables[var_name]
                    temp_constraints.append(var_obj == value)

            # Create temporary model with solution constraints
            test_model = Model(cpm_model.constraints + temp_constraints)

            # If SAT, the solution is valid; if UNSAT, it violates some constraint
            result = test_model.solve()
            return result

        except Exception as e:
            logger.debug(f"Error validating solution: {e}")
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
        enable_ml_prior=True,
        use_intelligent_subsets=True
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

