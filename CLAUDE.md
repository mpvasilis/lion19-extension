# HCAR Methodology Specification

## Research Objectives

### Primary Goal
Develop a hybrid constraint acquisition system that learns accurate constraint models from **sparse data** (5 positive examples) by intelligently correcting over-fitted global constraints through principled interactive refinement.

### What We Are Achieving
1. **Correctness**: Learn models that are solution-equivalent to the target (100% precision and recall)
2. **Efficiency**: Minimize oracle queries through intelligent mechanisms (orders of magnitude fewer than pure active learning)
3. **Robustness**: Handle over-fitting systematically rather than heuristically
4. **Scalability**: Work on complex problems where pure active learning fails (timeouts on VM Allocation, Nurse Rostering)

### Research Question
**How can a hybrid CA framework be evolved from a heuristic-based system into a methodologically sound, intelligent, and theoretically grounded paradigm for overcoming over-fitting?**

## Core Hypothesis

**Hypothesis 1**: Passive learning from sparse data (5 examples) produces over-fitted global constraints that cause catastrophic accuracy loss (39-81% recall) if accepted without validation.

**Hypothesis 2**: Intelligent subset exploration using data-driven "culprit scores" outperforms positional heuristics (first/middle/last) in correcting over-scoped constraints, saving 10-35% of refinement queries.

**Hypothesis 3**: A principled information flow that prunes bias only based on confirmed ground truth prevents information loss and ensures soundness.

## Methodological Constraints (Non-Negotiable Rules)

### CONSTRAINT 1: Independence of Biases
```
RULE: B_globals and B_fixed MUST be maintained as independent hypothesis sets
WHY: Prevents irreversible information loss
VIOLATION: Using unverified B_globals to prune B_fixed before oracle confirmation
```

### CONSTRAINT 2: Ground Truth Only Pruning
```
RULE: Bias pruning MUST only use confirmed solutions (E+ and oracle-verified queries)
WHY: Hypotheses may be refuted; premature pruning is irreversible
IMPLEMENTATION: When oracle returns "Valid" for query Y:
  - Y is NOW confirmed ground truth
  - Remove all constraints from B_fixed violated by Y
  - NEVER prune based on unverified constraints from B_globals
```

### CONSTRAINT 3: Complete Query Generation
```
RULE: For any false constraint c, the query generator MUST eventually find a violating valid solution (if one exists) or prove none exists (UNSAT)
WHY: Ensures all incorrect constraints can be refuted
TIMEOUT: If solver times out, increase confidence slightly (0.05) but do not refute
```

### CONSTRAINT 4: Hard Refutation on Counterexample
```
RULE: If oracle returns "Valid" for a query that violates constraint c, immediately set P(c) = 0
WHY: This is definitive proof that c is incorrect
IMPLEMENTATION: No gradual confidence decay - hard rejection
```

### CONSTRAINT 5: Data-Driven Subset Generation
```
RULE: When correcting rejected constraints, MUST use culprit scores, NOT positional heuristics
WHY: Positional heuristics (remove first/middle/last) are blind guesses; intelligent approach saves queries
METRICS: Structural isolation + weak constraint support + value diversity
```

## The Three-Phase Methodology

### Phase 1: Passive Candidate Generation

**OBJECTIVE**: Extract initial constraint hypotheses from sparse examples while maintaining two independent biases.

**INPUTS**:
- E+ = {5 positive examples}
- Variable set X and domains D
- Constraint language Γ (AllDifferent, Sum, Count)

**PROCESS** (MUST follow this order):
1. **Pattern Detection**:
   - Scan E+ for structural regularities (rows, columns, blocks, custom structures)
   - For each pattern consistent across ALL 5 examples, generate global constraint candidate
   - Store in B_globals with scope and parameters

2. **Fixed-Arity Bias Initialization**:
   - Generate all syntactically valid binary/ternary constraints from Γ
   - Create B_fixed as complete bias up to max_arity (typically 3)

3. **Independent Pruning**:
   - Prune B_fixed using ONLY E+ (remove constraints violated by any example)
   - Do NOT prune B_fixed using B_globals (violates CONSTRAINT 1)
   - Do NOT prune B_globals at this stage

**OUTPUTS**:
- B_globals: Over-fitted candidate global constraints (HIGH FALSE POSITIVE RATE)
- B_fixed: Pruned fixed-arity bias (partially refined, but incomplete)

**CRITICAL**: Phase 1 output is EXPECTED to be over-fitted. This is by design - Phase 2 will correct it.

### Phase 2: Query-Driven Interactive Refinement

**OBJECTIVE**: Systematically validate or refute every global constraint candidate through intelligent targeted queries, correcting over-fitting while minimizing oracle burden.

**INPUTS**:
- B_globals from Phase 1
- B_fixed from Phase 1
- Oracle O (perfect, returns Valid/Invalid)
- Budget_total (total queries allocated)
- Thresholds: θ_min = 0.2, θ_max = 0.8

**INITIALIZATION** (MUST execute before loop):
1. For each c ∈ B_globals: P(c) ← MLPrior(c)
2. Allocate uncertainty budgets proportional to entropy H(P(c))
3. Initialize C'_G = ∅ (validated globals)
4. Set Level(c) = 0 for all candidates

**REFINEMENT LOOP** (Algorithm 1, lines 9-27):

**WHILE** B_globals ≠ ∅ AND queries_used < Budget_total AND time < T_max:

1. **SELECT**: Choose candidate c with highest uncertainty (P(c) near 0.5) and available budget

2. **QUERY GENERATION**: Generate Y such that:
   - Y ⊭ c (violates candidate being tested)
   - Y ⊨ C'_G (satisfies all validated globals)
   - Y ⊨ (B_globals \ {c}) (satisfies all other candidates)
   
   **If UNSAT**: No violating solution exists → P(c) ← θ_max (strong evidence c is correct)
   
   **If TIMEOUT**: Solver cannot decide → P(c) ← P(c) + 0.05 (slight confidence increase)

3. **ORACLE QUERY**: R ← Ask(O, Y)

4. **BELIEF UPDATE**:
   
   **If R = "Invalid"**: 
   - Y is not a valid solution, evidence supports c
   - P(c) ← UpdateBayesian(P(c), α=0.1) (increase confidence)
   
   **If R = "Valid"**: 
   - **HARD REFUTATION**: P(c) ← 0 (counterexample found)
   - **PRINCIPLED PRUNING** (CONSTRAINT 2):
     ```
     FOR each c_f ∈ B_fixed:
         IF Y violates c_f:
             Remove c_f from B_fixed  // Y is now ground truth
     ```

5. **DECISION**:
   
   **If P(c) ≥ θ_max**:
   - Accept c into C'_G
   - Redistribute unused budget to other candidates
   
   **If P(c) ≤ θ_min**:
   - Reject c from B_globals
   - **INTELLIGENT CORRECTION** (CONSTRAINT 5):
     ```
     IF Level(c) < d_max:
         S ← GenerateInformedSubsets(c, E+)  // Use culprit scores
         Add new candidates from S to B_globals
         Inherit portion of parent budget
         Increment Level for child candidates
     ```
   
   **Otherwise**: Continue querying c

**OUTPUTS**:
- C'_G: Validated global constraints (HIGH CONFIDENCE, LOW FALSE POSITIVE RATE)
- B'_fixed: Further refined bias (pruned by all validated queries)
- Q_2: Query count for this phase

**KEY MECHANISMS**:

**A. ML Prior Estimation**
```python
def MLPrior(c):
    """
    OBJECTIVE: Initialize confidence using structural features
    
    FEATURES MUST INCLUDE:
    - Constraint type (AllDifferent, Sum, Count)
    - Arity (scope size)
    - Dimensional structure (row/column/block patterns)
    - Variable naming patterns (x_i_j suggests grid)
    - Participation in other constraints
    
    MODEL: XGBoost trained on diverse benchmarks offline
    OUTPUT: P(c) ∈ [0, 1] representing likelihood c is correct
    
    PURPOSE: Focus budget on structurally ambiguous constraints
    """
```

**B. Intelligent Subset Exploration**
```python
def GenerateInformedSubsets(c_rejected, E+):
    """
    OBJECTIVE: Intelligently identify which variable to remove from over-scoped c
    
    CONSTRAINT 5: MUST use data-driven culprit scores, NOT heuristics
    
    FOR each variable v_i in scope(c):
        score(v_i) = w1 * StructuralIsolation(v_i) 
                   + w2 * WeakConstraintSupport(v_i)
                   + w3 * ValueDiversity(v_i, E+)
    
    culprit = argmax(score)  // Variable most likely incorrect
    
    c_new = c with scope(c) \ {culprit}
    
    RETURN [c_new]  // New candidate for testing
    
    METRICS EXPLAINED:
    - StructuralIsolation: Average distance to other scope variables
      (edge/corner variables score higher)
    
    - WeakConstraintSupport: Inverse of # constraints containing v_i
      (rarely constrained variables score higher)
    
    - ValueDiversity: Low correlation with constraint relation
      (for Sum: high variance; for Count: unpredictable occurrence)
    """
```

**C. Bayesian Confidence Update**
```python
def UpdateBayesian(P_c, alpha):
    """
    OBJECTIVE: Update confidence after "Invalid" oracle response
    
    RULE: Invalid response is evidence FOR the constraint
    
    P_new = P_c + (1 - P_c) * (1 - alpha)
    
    WHERE:
    - alpha = 0.1 (noise parameter, accounts for rare oracle errors)
    - Update is multiplicative increase
    
    CONSTRAINT 4: If response is "Valid", do NOT update - set P_c = 0 (hard refutation)
    """
```

### Phase 3: Active Learning Completion

**OBJECTIVE**: Learn remaining fixed-arity constraints using standard active learning, benefiting from validated globals and refined bias.

**INPUTS**:
- C'_G: Validated global constraints from Phase 2 (treated as KNOWN)
- B'_fixed: Refined fixed-arity bias (smaller than original B_fixed due to principled pruning)
- Oracle O

**PROCESS**: Use MQuAcq-2 algorithm
1. Treat C'_G as part of the learned model (no further validation needed)
2. Generate queries to resolve status of all candidates in B'_fixed
3. Learn set C_L of fixed-arity constraints

**OUTPUT**: 
- C_final = C'_G ∪ C_L
- Q_3: Query count for this phase

**EFFICIENCY GAIN**: Because B'_fixed is smaller (due to Phase 2 pruning) and C'_G provides structure, Phase 3 converges faster than if MQuAcq-2 ran from scratch.

## Critical Principles (Must Be Enforced)

### Principle 1: Irreversible Actions Require Ground Truth
```
WHAT: Removing constraints from bias is irreversible
THEREFORE: Only prune when solution is confirmed by oracle
NEVER: Prune based on unverified hypotheses
```

### Principle 2: Over-fitting is the Default
```
WHAT: 5 examples are insufficient for passive learning
EXPECT: B_globals will contain spurious constraints
THEREFORE: Phase 2 is not optional - it's essential
EVIDENCE: HCAR-NoRefine achieves only 39-81% recall
```

### Principle 3: Intelligence Over Heuristics
```
WHAT: Correcting over-scoped constraints is a search problem
BAD: Heuristics (remove first/middle/last variable)
GOOD: Data-driven culprit scores
EVIDENCE: Saves 10-35% queries on complex benchmarks
```

### Principle 4: Budget Focus on Uncertainty
```
WHAT: Not all candidates are equally uncertain
ALLOCATION: Budget(c) ∝ Entropy(P(c))
EFFECT: Spend queries where they provide most information gain
AVOID: Uniform budget distribution (wastes queries on obvious cases)
```

### Principle 5: Soundness and Completeness
```
SOUNDNESS: System MUST NOT learn incorrect constraints
GUARANTEE: Under perfect oracle, all learned constraints ∈ C_T

COMPLETENESS: System MUST NOT incorrectly discard true constraints  
GUARANTEE: Under perfect oracle, all constraints in C_T are learned

PROOF: See Section 4 (Theoretical Analysis) of paper
```

## Success Criteria

### Primary Metrics (MUST Achieve)

**Model Accuracy**:
- **Solution-space Precision (S-Prec)**: 100%
  - Measures false positives: |sol(C_final) ∩ sol(C_T)| / |sol(C_final)|
  - All learned solutions must be valid

- **Solution-space Recall (S-Rec)**: 100%
  - Measures false negatives: |sol(C_final) ∩ sol(C_T)| / |sol(C_T)|
  - All valid solutions must be accepted
  - **CRITICAL**: This detects over-fitting

**Efficiency** (Compared to Baselines):
- Q_Σ (Total Queries): 
  - **vs HCAR-Heuristic**: 10-35% fewer queries (demonstrates intelligent > heuristic)
  - **vs MQuAcq-2**: Orders of magnitude fewer (demonstrates hybrid > pure active)

### Expected Results (from Experiments)

| Benchmark | HCAR-Advanced | HCAR-Heuristic | HCAR-NoRefine | MQuAcq-2 |
|-----------|---------------|----------------|---------------|----------|
| **Sudoku** |
| S-Rec | 100% | 100% | 81% | 100% |
| Q_Σ | 191 | 198 (+3.7%) | 285 | 6,844 |
| **UEFA** |
| S-Rec | 100% | 100% | 74% | 100% |
| Q_Σ | 73 | 86 (+17.8%) | 61 | 2,150 |
| **VM Allocation** |
| S-Rec | 100% | 100% | 39% | TIMEOUT |
| Q_Σ | 177 | 212 (+19.8%) | 145 | >10,000 |
| **Nurse Rostering** |
| S-Rec | 100% | 95% | 42% | TIMEOUT |
| Q_Σ | 182 | 228 (+25.3%) | 189 | >10,000 |

### Validation Checks (MUST Pass)

1. **Over-fitting Detection**: HCAR-NoRefine MUST show degraded S-Rec (proves over-fitting exists)
2. **Intelligent Advantage**: HCAR-Advanced MUST outperform HCAR-Heuristic (proves culprit scores work)
3. **Hybrid Efficiency**: HCAR-Advanced MUST vastly outperform MQuAcq-2 (proves hybrid approach works)
4. **Convergence**: All methods MUST achieve 100% S-Prec (proves no false positives learned)

## Implementation Constraints

### Technology Requirements

**MUST USE**:
- **CPMpy**: Constraint modeling layer
- **Google OR-Tools CP-SAT**: Constraint solver backend
- **XGBoost**: ML prior estimation
- **PyConA**: MQuAcq-2 implementation
- **NumPy/Pandas**: Data processing

**SOLVER CONFIGURATION** (MUST SET):
```python
TIMEOUT_PER_QUERY = 300  # seconds
MAX_WORKERS = 1          # deterministic behavior
```

### Experimental Protocol (MUST FOLLOW)

**Initialization**:
```python
N_POSITIVE_EXAMPLES = 5  # Always start with 5
RANDOM_SEED = 42         # For reproducibility
SAMPLE_SIZE_METRICS = 100  # For S-Prec/S-Rec estimation
```

**Benchmarks** (MUST TEST ALL FIVE):
1. Sudoku (9×9): 27 AllDifferent constraints
2. UEFA Scheduling: 19 mixed constraints
3. VM Allocation: 72 Sum constraints
4. Exam Timetabling: 24 mixed constraints
5. Nurse Rostering: 21 Count/Sum constraints

**Variants** (MUST IMPLEMENT ALL FOUR):
1. HCAR-Advanced (proposed - intelligent subset exploration)
2. HCAR-Heuristic (baseline - positional heuristic)
3. HCAR-NoRefine (ablation - skip Phase 2)
4. MQuAcq-2 (external baseline - pure active)

## What the System Must Know

### About Constraint Acquisition
- **CSP**: Triple (X, D, C) - variables, domains, constraints
- **Global Constraints**: AllDifferent, Sum, Count with arbitrary arity
- **Bias**: Set of all possible constraints (B_globals + B_fixed)
- **Target**: Unknown model C_T that we're learning
- **Goal**: Learn C_final such that sol(C_final) = sol(C_T)

### About Over-fitting
- **Cause**: Passive learning from sparse data (5 examples)
- **Manifestation**: Spurious constraints that hold on examples but not generally
- **Example**: AllDifferent([x1, x2, x3, x4]) when only AllDifferent([x1, x2, x3]) is true
- **Impact**: Overly restrictive model (low recall)
- **Solution**: Interactive refinement with targeted queries

### About Intelligent Correction
- **Problem**: How to fix over-scoped constraint AllDifferent([v1, v2, v3, v4])?
- **Bad Approach**: Remove first/middle/last (33% chance of being right)
- **Good Approach**: Calculate which variable is most likely wrong using data
- **Culprit Score**: Combines structural, support, and diversity metrics
- **Result**: Higher probability of finding correct subset on first try

### About Principled Information Flow
- **Hypothesis**: Unverified constraint from passive learning
- **Ground Truth**: Initial examples + oracle-verified queries
- **Rule**: Can only prune bias using ground truth
- **Reason**: Hypotheses may be refuted; pruning is irreversible
- **Violation Example**: Using AllDifferent(row1) to remove constraints from B_fixed, then discovering AllDifferent(row1) is wrong

## Development Workflow

### Phase 1: Build Foundation
1. Implement CSP data structures (Variable, Domain, Constraint classes)
2. Build passive pattern detector (scan for AllDifferent, Sum, Count)
3. Create bias manager (generate and prune B_fixed)
4. Validate: Can extract correct constraints from perfect data?

### Phase 2: Build Intelligent Refinement
1. Implement query generator (auxiliary CSP for violating assignments)
2. Build oracle interface (membership query simulation)
3. Implement Bayesian updater (confidence score management)
4. **CRITICAL**: Implement intelligent subset explorer with culprit scores
5. Implement principled pruner (ground truth only)
6. Validate: Can correct over-fitted constraints from 5 examples?

### Phase 3: Build Active Completion
1. Integrate PyConA's MQuAcq-2
2. Pass validated globals and refined bias
3. Validate: Can complete model efficiently?

### Phase 4: Benchmark Implementation
1. Implement all 5 benchmarks with ground truth models
2. Create solution generators (for S-Prec/S-Rec estimation)
3. Validate: Do benchmarks match specifications?

### Phase 5: Experimental Evaluation
1. Run all 4 variants on all 5 benchmarks
2. Collect metrics: S-Prec, S-Rec, Q_2, Q_3, Q_Σ, Time
3. Validate results against expected outcomes (Table 1)
4. Statistical analysis: Compute mean query savings

## Theoretical Foundations (What Guarantees Hold)

### Assumption A1: Perfect Oracle
Oracle always returns correct answer: Valid iff assignment ∈ sol(C_T)

### Assumption A2: Complete Bias
Target model is expressible: C_T ⊆ (B_globals ∪ B_fixed)

### Assumption A3: Complete Query Generator
For any false constraint c ∉ C_T, generator can find violating valid solution (or prove none exists)

### Guarantee 1: Soundness (Proposition 1)
HCAR will NOT learn incorrect constraints (no false positives)
**Proof**: Any wrong constraint will eventually be refuted by a counterexample query

### Guarantee 2: Completeness (Proposition 2)
HCAR will NOT discard correct constraints (no false negatives)
**Proof**: True constraints cannot be refuted; principled pruning protects them

### Guarantee 3: Convergence
Under A1-A3, HCAR converges to solution-equivalent model: sol(C_final) = sol(C_T)

### Complexity Analysis
- **Query Complexity**: Q_Σ = O(|B_globals| × b_max) + O(|B'_fixed|)
  - Phase 2 dominates when many over-fitted globals
  - Intelligent mechanisms reduce practical queries below worst case
- **Sample Complexity**: Inverse relationship between |E+| and Q_Σ
  - More examples → less over-fitting → fewer refinement queries
  - Design: Effective with minimal |E+| = 5

