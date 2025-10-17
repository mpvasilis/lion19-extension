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

### High-Level Overview

```
Phase 1: Passive Learning
  ├─ Input: E+ (positive examples)
  ├─ Generate candidate AllDifferent constraints (patterns)
  ├─ Generate candidate Sum/Count constraints (patterns)
  ├─ Prune inconsistent AllDifferent constraints using E+
  ├─ Generate complete fixed-arity bias (all binary constraints)
  ├─ Prune inconsistent fixed-arity constraints using E+
  └─ Output: CG (global candidates), B (pruned bias)

Phase 2: Query-Driven Refinement with Disambiguation
  ├─ ML Classifier: Assign P(c) to each c ∈ CG
  ├─ COP Generation: Find violation assignment e violating multiple constraints
  │   └─ Collect Viol(e) = constraints violated by e
  ├─ Oracle Query: ASK(e) → Yes/No
  ├─ If "No" (e invalid): All constraints in Viol(e) supported
  │   ├─ Increase P(c) for all c ∈ Viol(e)
  │   └─ Accept constraints with P(c) ≥ θ_max, prune B
  ├─ If "Yes" (e valid): Disambiguation phase
  │   ├─ For each c ∈ Viol(e):
  │   │   ├─ Try isolation: Generate e_test violating ONLY c (within Viol(e))
  │   │   ├─ If isolated successfully:
  │   │   │   ├─ ASK(e_test)
  │   │   │   │   ├─ "Yes" → c definitively FALSE (P(c) *= α)
  │   │   │   │   └─ "No" → Ambiguous (prior-weighted update)
  │   │   └─ Cannot isolate → Use prior P(c) for decision
  │   │       └─ Prior-weighted confidence update
  │   └─ Reject constraints with P(c) ≤ θ_min, attempt repair
  └─ Output: C'G (refined globals), pruned B

Phase 3: Active Learning (MQuAcq-2)
  ├─ Decompose C'G to binary constraints
  ├─ Initialize CL with decomposed constraints
  ├─ Run MQuAcq-2 with pruned B
  └─ Output: Final model C'G ∪ CL
```

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

### Phase 2: Query-Driven Interactive Refinement with Disambiguation

**OBJECTIVE**: Systematically validate or refute every global constraint candidate through disambiguation-based interactive refinement, efficiently identifying false constraints while being resilient to over-fitting.

**INPUTS**:
*   `B_globals` = Candidate global constraints from Phase 1
*   `B_fixed` = Pruned fixed-arity bias from Phase 1
*   `E+` = Initial positive examples (for consistency checking)
*   Oracle `O`
*   ML Classifier for P(c) estimation
*   Budget `Q_max`

**REFINEMENT LOOP** (Algorithm 2):

**WHILE** B_globals ≠ ∅ AND queries_used < Q_max AND time < T_max:

1.  **COP GENERATION**: Generate violation query `e` that violates at least one constraint in `B_globals`
    -   Build constraint optimization problem (COP)
    -   `e` must satisfy all validated constraints in `C'_G`
    -   `e` must violate at least one constraint in `B_globals` (prioritize high uncertainty)
    -   Collect `Viol(e)` = set of constraints violated by `e`

2.  **ORACLE QUERY**: `R ← Ask(O, e)`

3.  **RESPONSE HANDLING**:

    **If R = "No"** (e is invalid - Supporting Evidence):
    -   All constraints in `Viol(e)` are SUPPORTED
    -   For each `c ∈ Viol(e)`:
        ```python
        # Increase confidence
        P(c) ← P(c) + (1 - P(c)) * (1 - α)

        # If P(c) ≥ θ_max:
        #   - Accept c, move to C'_G
        #   - Prune B_fixed using c's decomposition
        ```

    **If R = "Yes"** (e is valid - Refuting Evidence):
    -   Some constraints in `Viol(e)` are FALSE
    -   **Start DISAMBIGUATION**:
        ```python
        for c in Viol(e):
            # Attempt Isolation
            e_test = GenerateIsolationQuery(c, Viol(e))

            if e_test exists:  # c can be isolated
                R_test ← Ask(O, e_test)

                if R_test == "Yes":
                    # e_test is valid but violates only c
                    # → c is DEFINITIVELY FALSE
                    P(c) ← P(c) * α  # Drastically decrease

                    if P(c) ≤ θ_min:
                        Reject c
                        Generate repair hypotheses (subset exploration)

                else:  # R_test == "No"
                    # Isolation query rejected - c is ambiguous
                    # Use prior-weighted update
                    update_factor = α + (1 - α) * P(c)
                    P(c) ← P(c) * update_factor

            else:  # Cannot isolate c
                # c cannot be distinguished within Viol(e)
                # Use ML prior P(c) for decision
                # Apply prior-weighted confidence update
                update_factor = α + (1 - α) * P(c)
                P(c) ← P(c) * update_factor

                if P(c) ≤ θ_min:
                    Reject c (low prior suggests false)
                    Generate repair hypotheses
        ```

4.  **PRUNING**: When constraint `c` is accepted (P(c) ≥ θ_max):
    ```python
    # Decompose c into binary constraints
    implied_constraints = get_decomposition(c)

    # Remove contradictory constraints from B_fixed
    for implied_c in implied_constraints:
        contradiction = get_contradiction(implied_c)
        if contradiction in B_fixed:
            B_fixed.remove(contradiction)
    ```

**OUTPUT**:
*   `C'_G`: Validated global constraints
*   `B_fixed`: Pruned fixed-arity bias
*   `Q_2`: Query count for Phase 2

#### Key Mechanisms:

**A. ML Prior Estimation**
```python
def MLPrior(c):
    """
    OBJECTIVE: Initialize confidence using structural features.
    MODEL: XGBoost trained on diverse benchmarks offline.
    OUTPUT: P(c) ∈ [0, 1] representing likelihood c is correct.
    PURPOSE:
    - Focus budget on structurally ambiguous constraints
    - Guide disambiguation decisions when isolation is impossible
    - Weight confidence updates based on structural plausibility
    """
```

**B. Violation Query Generation**
```python
def GenerateViolationQuery(B_globals, C'_G):
    """
    OBJECTIVE: Generate a query that violates multiple candidate constraints.

    STRATEGY:
    1. Build COP with validated constraints C'_G (must satisfy)
    2. Select top-k uncertain constraints from B_globals
    3. Add disjunction: must violate at least one selected constraint
       (c1.violated OR c2.violated OR ... OR ck.violated)
    4. Solve to get assignment e
    5. Check which constraints are actually violated by e

    RETURNS: (e, Viol(e), status)

    BENEFIT: Tests multiple constraints with a single query
    """
```

**C. Constraint Isolation**
```python
def GenerateIsolationQuery(c_target, Viol_e, C'_G):
    """
    OBJECTIVE: Generate query that violates ONLY c_target (within Viol_e).

    STRATEGY:
    1. Build COP with validated constraints C'_G (must satisfy)
    2. Add OTHER constraints from Viol_e EXCEPT c_target (must satisfy)
    3. Add negation of c_target (must violate)
    4. Solve to get e_test

    RESULT:
    - If SAT: e_test isolates c_target for disambiguation
    - If UNSAT: c_target cannot be isolated (implied by other Viol_e constraints)

    BENEFIT: Enables definitive identification of false constraints
    """
```

**D. Disambiguation with Prior-Weighted Updates**
```python
def Disambiguate(Viol_e, oracle, P_priors):
    """
    OBJECTIVE: Identify which constraints in Viol_e are false.

    STRATEGY:
    1. For each c in Viol_e:
       a. Attempt isolation
       b. If isolated:
          - ASK(e_test)
          - "Yes" → Definitive FALSE: P(c) *= α
          - "No" → Ambiguous: P(c) *= (α + (1-α)*P(c))
       c. If cannot isolate:
          - Use ML prior for decision
          - P(c) *= (α + (1-α)*P(c))

    2. Reject constraints with P(c) ≤ θ_min
    3. Generate repair hypotheses for rejected constraints

    BENEFIT: Combines definitive evidence with probabilistic reasoning
    """
```

**E. Intelligent Model Repair via Counterexample Analysis**
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

**F. Confidence Update Formulas**
```python
def UpdateConfidence(P_c, response, alpha, use_prior_weighting=False):
    """
    OBJECTIVE: Update confidence based on oracle response.

    FORMULAS:

    1. Supporting Evidence (response == "Invalid"):
       P_new = P_c + (1 - P_c) * (1 - α)
       # Increase confidence towards 1.0

    2. Definitive Refutation (isolated "Yes" response):
       P_new = P_c * α
       # Drastically decrease towards 0.0

    3. Ambiguous Refutation (cannot isolate, use prior):
       update_factor = α + (1 - α) * P_c
       P_new = P_c * update_factor
       # Softer decrease, weighted by prior belief
       # High prior (P_c ≈ 1) → smaller decrease
       # Low prior (P_c ≈ 0) → larger decrease

    RETURN P_new ∈ [0, 1]
    """
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