# HCAR Methodology Specification (Noise-Robust Version)

## Research Objectives

### Primary Goal
Develop a hybrid constraint acquisition system that learns accurate constraint models from **sparse data** (5 positive examples) by intelligently correcting over-fitted global constraints through a **probabilistic interactive refinement** that is robust to oracle noise.

### What We Are Achieving
1.  **Correctness**: Learn models that are solution-equivalent to the target (100% precision and recall) under noisy conditions.
2.  **Efficiency**: Minimize oracle queries through intelligent mechanisms.
3.  **Robustness**: Handle over-fitting systematically and gracefully recover from occasional oracle errors.
4.  **Scalability**: Work on complex problems where pure active learning is infeasible.

### Research Question
**How can a hybrid CA framework be evolved from a heuristic-based system into a methodologically sound, intelligent, and theoretically grounded paradigm for overcoming over-fitting, even in the presence of oracle noise?**

## Core Hypothesis

*   **Hypothesis 1**: Passive learning from sparse data (5 examples) produces over-fitted global constraints that cause catastrophic accuracy loss if not validated.
*   **Hypothesis 2**: Intelligent model repair using counterexample-driven minimal relaxations outperforms positional heuristics in correcting over-scoped constraints.
*   **Hypothesis 3**: A unified probabilistic belief update for all oracle responses makes the learning process resilient to a noisy oracle, a scenario where hard refutation would fail.

## Methodological Constraints (Non-Negotiable Rules)

### CONSTRAINT 1: Independence of Biases
```
RULE: B_globals and B_fixed MUST be maintained as independent hypothesis sets.
WHY: Prevents irreversible information loss.
VIOLATION: Using unverified B_globals to prune B_fixed before its confidence is confirmed.
```

### CONSTRAINT 2: Ground Truth Only Pruning
```
RULE: Irreversible bias pruning MUST only use the initial, trusted set of examples E+.
WHY: Oracle responses are now treated as probabilistic evidence, not infallible ground truth, and cannot be used for irreversible actions.
IMPLEMENTATION: B_fixed is pruned ONLY once using E+ in Phase 1. It is NOT pruned further in Phase 2 based on oracle queries.
```

### CONSTRAINT 3: Complete Query Generation
```
RULE: For any false constraint c, the query generator MUST eventually find a violating valid solution (if one exists) or prove none exists (UNSAT).
WHY: Ensures all incorrect constraints can eventually be challenged.
TIMEOUT: If solver times out, increase confidence slightly (0.05) but do not make a strong update.
```

### CONSTRAINT 4: Unified Probabilistic Update```
RULE: ALL oracle feedback, "Valid" or "Invalid", MUST be integrated probabilistically. Hard refutation is forbidden.
WHY: This is the core mechanism for handling oracle noise. A single incorrect oracle response should not catastrophically derail the learning process.
IMPLEMENTATION: Both "Valid" and "Invalid" responses trigger a probabilistic confidence update, pushing the score up or down without setting it to 0 or 1.
```

### CONSTRAINT 5: Counterexample-Driven Model Repair
```
RULE: When correcting rejected constraints, MUST use the query that led to the lowest confidence score to generate minimal repair hypotheses.
WHY: This query is the strongest piece of refuting evidence; leveraging it is more systematic than heuristics.
APPROACH: Generate minimal relaxations of the rejected constraint that are consistent with this refuting query, then rank by plausibility.
```

## The Three-Phase Methodology

### Phase 1: Passive Candidate Generation

**OBJECTIVE**: Extract initial constraint hypotheses from sparse examples while maintaining two independent biases.

**INPUTS**:
*   `E+` = {5 positive examples}
*   Variable set `X` and domains `D`
*   Constraint language `Γ` (AllDifferent, Sum, Count)

**PROCESS**:
1.  **Pattern Detection**: Generate global constraint candidates from patterns consistent across all examples and store in `B_globals`.
2.  **Fixed-Arity Bias Initialization**: Create `B_fixed` as the complete bias of low-arity constraints.
3.  **Independent Pruning**: Prune `B_fixed` using **ONLY** `E+`. `B_globals` is not used for pruning.

**OUTPUTS**:
*   `B_globals`: Over-fitted candidate global constraints.
*   `B_fixed`: Pruned fixed-arity bias (refined ONLY with E+).

### Phase 2: Query-Driven Interactive Refinement

**OBJECTIVE**: Systematically validate or refute every global constraint candidate through a unified probabilistic framework, correcting over-fitting while being resilient to oracle noise.

**... (Inputs section remains the same) ...**

**REFINEMENT LOOP** (Algorithm 1, lines 9-27):

**WHILE** B_globals ≠ ∅ AND queries_used < Budget_total AND time < T_max:

1.  **SELECT**: Choose candidate `c` with highest uncertainty.
2.  **QUERY GENERATION**: Generate query `Y` that violates `c`.
3.  **ORACLE QUERY**: `R ← Ask(O, Y)`
4.  **UNIFIED BELIEF UPDATE (CONSTRAINT 4)**:

    **If R = "Invalid"** (Supporting Evidence):
    -   `P(c) ← P(c) + (1 - P(c)) * (1 - α)` (increase confidence).

    **If R = "Valid"** (Refuting Evidence):
    -   `P(c) ← P(c) * α` (drastically decrease confidence).
    -   Store `Y` as the latest `Y_counterexample` for `c`.

5.  **DECISION** (After inner query loop for `c` completes):

    **If P(c) ≥ θ_max**:
    -   **1. Accept `c`**: Move `c` to `C'_G`.
    -   **2. NEW: PRINCIPLED PRUNING (CONSTRAINT 2)**:
        ```python
        # The accepted constraint c is now a confirmed rule.
        # Use its implications to prune B_fixed.
        
        # Get the set of binary constraints implied by c's decomposition
        # e.g., for AllDifferent({x1,x2}), this is { (x1, '!=', x2) }
        implied_constraints = get_decomposition(c)
        
        constraints_to_remove = set()
        for implied_c in implied_constraints:
            # Find the direct contradiction (e.g., for '!=', find '==')
            contradiction = get_contradiction(implied_c)
            if contradiction in B_fixed:
                constraints_to_remove.add(contradiction)
        
        # Atomically remove all contradictory constraints from B_fixed
        B_fixed.difference_update(constraints_to_remove)
        ```
    -   **3. Redistribute Budget**: Redistribute `c`'s unused budget.

    **If P(c) ≤ θ_min**:
    -   Reject `c`.
    -   **INTELLIGENT REPAIR (CONSTRAINT 5)** using `Y_counterexample`.
    
    **Otherwise**: `c` remains in `B_globals` for potential future queries.

#### Key Mechanisms:

**A. ML Prior Estimation**
```python
def MLPrior(c):
    """
    OBJECTIVE: Initialize confidence using structural features.
    MODEL: XGBoost trained on diverse benchmarks offline.
    OUTPUT: P(c) ∈ [0, 1] representing likelihood c is correct.
    PURPOSE: Focus budget on structurally ambiguous constraints.
    """
```

**B. Intelligent Model Repair via Counterexample Analysis**
```python
def RepairConstraintFromCounterexample(c_rejected, Y_counterexample):
    """
    OBJECTIVE: Generate minimal repair hypotheses using the query that refuted c.

    PRINCIPLE: Y_counterexample is the strongest evidence of why c is wrong.
               Find the smallest changes to c that make it consistent with Y.

    STEP 1: GENERATE MINIMAL REPAIR HYPOTHESES
    # For AllDifferent: find variables with duplicate values in Y and generate
    # hypotheses by removing one of them.
    # For Sum/Count: generate single-variable removal hypotheses.

    STEP 2: FILTER AND RANK HYPOTHESES
    # Filter: Keep only hypotheses consistent with all original examples in E+.
    # Rank: Use a plausibility score (ML Prior + structural metrics).
    # Return top-ranked hypothesis.
    """
```

**C. Unified Probabilistic Confidence Update**
```python
def UpdateConfidence(P_c, response, alpha):
    """
    OBJECTIVE: Update confidence based on oracle response using a unified model.
    
    RULE: Both responses are treated as probabilistic evidence.
    
    IF response == "Invalid": # Supporting Evidence
        # Increase confidence, moving towards 1.0
        P_new = P_c + (1 - P_c) * (1 - alpha)
    ELSE: # Refuting Evidence ("Valid")
        # Decrease confidence, moving towards 0.0
        P_new = P_c * alpha
    
    RETURN P_new
    ```

### Phase 3: Active Learning Completion

**OBJECTIVE**: Learn remaining fixed-arity constraints, benefiting from validated globals and the refined bias.

**INPUTS**:
*   `C'_G`: Validated global constraints from Phase 2.
*   `B_fixed`: The fixed-arity bias from Phase 1.
*   Oracle `O`

**PROCESS**: Use MQuAcq-2 algorithm.
1.  Treat `C'_G` as part of the learned model.
2.  Generate queries to resolve the status of all candidates in `B_fixed`.

**OUTPUT**:
*   `C_final = C'_G ∪ C_L`
*   `Q_3`: Query count for this phase.

## Critical Principles (Must Be Enforced)

### Principle 1: Irreversible Actions Require Infallible Evidence
```
WHAT: Removing constraints from bias is irreversible.
THEREFORE: Only prune bias using the initial trusted examples E+.
NEVER: Prune bias using noisy oracle responses.
```

### Principle 2: Over-fitting is the Default
```
WHAT: 5 examples are insufficient for passive learning.
EXPECT: B_globals will contain spurious constraints.
THEREFORE: Phase 2 is not optional—it's essential for correctness.
```

### Principle 3: Intelligence Over Heuristics
```
WHAT: Correcting over-scoped constraints requires systematic reasoning.
BAD: Heuristics (remove first/middle/last variable) are blind guesses.
GOOD: Counterexample-driven minimal repair uses evidence from the refuting query.
```

### Principle 4: Resilience to Noise
```
WHAT: Oracles can be wrong.
BAD: Hard refutation is brittle and fails on a single error.
GOOD: Unified probabilistic updates are resilient, requiring a weight of evidence to make a decision.
```

## Success Criteria

### Primary Metrics (MUST Achieve)

*   **Model Accuracy**:
    *   **Solution-space Precision (S-Prec)**: 100%
    *   **Solution-space Recall (S-Rec)**: 100% (This is the critical metric that detects over-fitting)
*   **Efficiency**: `Q_Σ` (Total Queries) should be significantly lower than baselines.

### Expected Results (from Experiments)

| Benchmark       | HCAR-Advanced (Noisy Oracle) | HCAR-Heuristic (Perfect Oracle) | HCAR-NoRefine | MQuAcq-2       |
| --------------- | ---------------------------- | ------------------------------- | ------------- | -------------- |
| **S-Rec**       | 100%                         | 100%                            | 39-81%        | 100% / TIMEOUT |
| **Q_Σ** (approx)  | ~200-400                     | ~200-400                        | ~150-300      | >5,000         |
| **Robustness**  | High (recovers from errors)  | None (fails on first error)     | N/A           | None           |

### Validation Checks (MUST Pass)

1.  **Over-fitting Detection**: `HCAR-NoRefine` MUST show degraded S-Rec.
2.  **Intelligent Advantage**: `HCAR-Advanced` MUST outperform `HCAR-Heuristic` in query efficiency.
3.  **Hybrid Efficiency**: `HCAR-Advanced` MUST vastly outperform `MQuAcq-2`.
4.  **Noise Resilience**: `HCAR-Advanced` MUST converge to the correct model even when the oracle is simulated to give incorrect answers 5-10% of the time.

## Theoretical Foundations (What Guarantees Hold)

### Assumptions
*   **A1 (Perfect Oracle - for theoretical proof)**: Oracle always returns the correct answer. The *system design*, however, is robust to violations of this assumption.
*   **A2 (Complete Bias)**: The target model is expressible in the language.
*   **A3 (Complete Query Generator)**: The query generator can challenge any false constraint.

### Guarantees (Under Perfect Oracle Assumption)
*   **Guarantee 1: Soundness**: HCAR will not learn incorrect constraints.
*   **Guarantee 2: Completeness**: HCAR will not incorrectly discard true constraints.
*   **Guarantee 3: Convergence**: HCAR converges to a solution-equivalent model.

### Practical Guarantee (Under Noisy Oracle)
*   **Robustness**: The system's probabilistic nature ensures that it will converge to the correct model **in expectation**, provided the oracle's error rate (`α`) is not excessively high. A single error will not cause divergence.