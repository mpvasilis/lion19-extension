# Domain Expert Hints System

## Motivation

**Problem:** Passive learning from sparse examples (e.g., 5 solutions) cannot always infer correct constraint parameters, especially for:
- Upper/lower bounds that are never reached in examples
- Symmetric parameters that should be uniform across similar entities
- Capacity constraints that examples don't fully saturate

**Example from Nurse_Rostering:**
- **Detected:** `Count(roster, nurse_6) <= 5`
- **Correct:** `Count(roster, nurse_6) <= 6`
- **Why wrong:** Examples never showed nurse 6 working 6 days (under-sampling)

## Solution: Domain Expert Hints

Allow domain experts to provide **minimal hints** about constraint structure without specifying the full model.

### Design Principles

1. **Minimal Information:** Hints should be high-level properties, not full constraints
2. **Optional:** System works without hints (falls back to passive learning)
3. **Non-invasive:** Hints guide but don't override learner
4. **Verifiable:** Hints can be validated against examples and oracle

### Types of Hints

#### Type 1: Parameter Symmetry
"All entities of the same type should have the same bound"

**Example:**
```python
hint = {
    "type": "parameter_symmetry",
    "constraint_type": "Count",
    "scope_pattern": "all_vars",
    "values": "all",  # Apply to all target values
    "property": "upper_bound_uniform"
}
```

**Effect:** If Count(X, nurse_1) <= 6 is detected, enforce Count(X, nurse_i) <= 6 for all nurses.

#### Type 2: Bound Normalization
"The bound should be a round number or common threshold"

**Example:**
```python
hint = {
    "type": "bound_normalization",
    "constraint_type": "Count",
    "normalization": "round_up_to_common"
}
```

**Effect:** If bounds are detected as [6, 6, 6, 5, 6, 6, 6, 6], normalize to [6, 6, 6, 6, 6, 6, 6, 6].

#### Type 3: Capacity Hint
"There is a global capacity that applies to all entities"

**Example:**
```python
hint = {
    "type": "capacity",
    "constraint_type": "Count",
    "capacity": 6,  # Or "infer_from_majority"
    "applies_to": "all_values"
}
```

**Effect:** All Count constraints use the same upper bound.

#### Type 4: Domain Knowledge
"Based on problem domain, certain parameter patterns are expected"

**Example:**
```python
hint = {
    "type": "domain_knowledge",
    "domain": "scheduling",
    "knowledge": {
        "work_limit_uniform": True,  # All workers have same max shifts
        "bound_likely_multiple_of": 1  # Bound is likely a whole number
    }
}
```

## Implementation

### 1. Hint Configuration

Add hints to HCARConfig:

```python
class HCARConfig:
    def __init__(self):
        # ... existing config ...

        # Domain expert hints (optional)
        self.domain_hints: Optional[Dict] = None

        # Enable hint-based parameter normalization
        self.enable_hint_normalization: bool = True
```

### 2. Hint Application During Pattern Detection

Modify `_extract_count_patterns()` to apply hints:

```python
def _extract_count_patterns(...):
    # ... existing detection code ...

    candidates = set()

    # Collect all Count candidates first
    for group, value in detected_patterns:
        candidate = create_count_constraint(group, value, max_count_observed)
        candidates.add(candidate)

    # Apply domain hints if provided
    if self.config.domain_hints:
        candidates = self._apply_domain_hints(candidates, "Count")

    return candidates
```

### 3. Hint Application Logic

```python
def _apply_domain_hints(
    self,
    candidates: Set[Constraint],
    constraint_type: str
) -> Set[Constraint]:
    """Apply domain expert hints to normalize constraint parameters."""

    if not self.config.domain_hints:
        return candidates

    hints = self.config.domain_hints.get(constraint_type, [])
    if not hints:
        return candidates

    logger.info(f"\nApplying domain expert hints for {constraint_type} constraints...")

    for hint in hints:
        if hint["type"] == "parameter_symmetry":
            candidates = self._apply_parameter_symmetry(candidates, hint)

        elif hint["type"] == "bound_normalization":
            candidates = self._apply_bound_normalization(candidates, hint)

        elif hint["type"] == "capacity":
            candidates = self._apply_capacity_hint(candidates, hint)

    return candidates
```

### 4. Parameter Symmetry Implementation

```python
def _apply_parameter_symmetry(
    self,
    candidates: Set[Constraint],
    hint: Dict
) -> Set[Constraint]:
    """
    Enforce symmetric parameters across similar constraints.

    Example: If Count(X, nurse_1) <= 6, then Count(X, nurse_i) <= 6 for all nurses.
    """
    logger.info(f"  Applying parameter symmetry hint: {hint['property']}")

    # Group candidates by constraint structure (same scope pattern)
    groups = {}
    for c in candidates:
        if c.constraint_type != hint.get("constraint_type"):
            continue

        # Group by scope (ignoring the specific value being counted)
        scope_key = tuple(sorted(c.scope))
        if scope_key not in groups:
            groups[scope_key] = []
        groups[scope_key].append(c)

    normalized_candidates = set()

    for scope_key, group in groups.items():
        if len(group) <= 1:
            normalized_candidates.update(group)
            continue

        # Extract bounds from all constraints in group
        bounds = []
        for c in group:
            # Parse bound from constraint (e.g., "count_leq_..._cnt6" -> 6)
            import re
            match = re.search(r'cnt(\d+)', c.id)
            if match:
                bounds.append(int(match.group(1)))

        if not bounds:
            normalized_candidates.update(group)
            continue

        # Use majority or max bound as the normalized value
        from collections import Counter
        bound_counts = Counter(bounds)

        # Strategy 1: Use most common bound
        normalized_bound = bound_counts.most_common(1)[0][0]

        # Strategy 2: If most common is not dominant, use max
        if bound_counts[normalized_bound] < len(bounds) * 0.7:
            normalized_bound = max(bounds)

        logger.info(f"    Group with {len(group)} constraints: bounds = {bounds}")
        logger.info(f"    Normalized to: {normalized_bound}")

        # Recreate constraints with normalized bound
        for c in group:
            # Extract value being counted
            match = re.search(r'val(\d+)', c.id)
            target_value = int(match.group(1)) if match else None

            if target_value is None:
                normalized_candidates.add(c)
                continue

            # Create normalized constraint
            if CPMPY_AVAILABLE:
                from cpmpy import Count
                constraint_vars = [self.variables[v] for v in c.scope if v in self.variables]
                constraint_obj = (Count(constraint_vars, target_value) <= normalized_bound)
            else:
                constraint_obj = None

            normalized_c = Constraint(
                id=c.id.replace(f"cnt{bounds[group.index(c)]}", f"cnt{normalized_bound}"),
                constraint=constraint_obj,
                scope=c.scope,
                constraint_type=c.constraint_type,
                arity=c.arity,
                level=c.level,
                confidence=c.confidence * 1.1  # Boost confidence for normalized constraints
            )
            normalized_candidates.add(normalized_c)

    return normalized_candidates
```

### 5. Capacity Hint Implementation

```python
def _apply_capacity_hint(
    self,
    candidates: Set[Constraint],
    hint: Dict
) -> Set[Constraint]:
    """
    Apply a global capacity bound to all Count constraints.

    Example: max_workdays = 6 for all nurses.
    """
    capacity = hint.get("capacity")

    if capacity == "infer_from_majority":
        # Extract all bounds and use majority
        bounds = []
        for c in candidates:
            if c.constraint_type == "Count":
                match = re.search(r'cnt(\d+)', c.id)
                if match:
                    bounds.append(int(match.group(1)))

        from collections import Counter
        if bounds:
            capacity = Counter(bounds).most_common(1)[0][0]
        else:
            return candidates

    logger.info(f"  Applying capacity hint: {capacity} for all Count constraints")

    normalized_candidates = set()

    for c in candidates:
        if c.constraint_type != "Count":
            normalized_candidates.add(c)
            continue

        # Extract target value
        match = re.search(r'val(\d+)', c.id)
        target_value = int(match.group(1)) if match else None

        if target_value is None:
            normalized_candidates.add(c)
            continue

        # Recreate with capacity bound
        if CPMPY_AVAILABLE:
            from cpmpy import Count
            constraint_vars = [self.variables[v] for v in c.scope if v in self.variables]
            constraint_obj = (Count(constraint_vars, target_value) <= capacity)
        else:
            constraint_obj = None

        normalized_c = Constraint(
            id=f"count_leq_{c.scope[0][:20]}_val{target_value}_cnt{capacity}",
            constraint=constraint_obj,
            scope=c.scope,
            constraint_type=c.constraint_type,
            arity=c.arity,
            level=c.level,
            confidence=0.7  # Moderate confidence for hint-based constraints
        )
        normalized_candidates.add(normalized_c)

    return normalized_candidates
```

## Usage Example

### Nurse Rostering with Hints

```python
# Define domain hints
domain_hints = {
    "Count": [
        {
            "type": "capacity",
            "capacity": "infer_from_majority",  # Use most common bound
            "applies_to": "all_values"
        },
        {
            "type": "parameter_symmetry",
            "property": "upper_bound_uniform",
            "reason": "All nurses have equal max workdays"
        }
    ]
}

# Create config with hints
config = HCARConfig(
    domain_hints=domain_hints,
    enable_hint_normalization=True
)

# Run HCAR
framework = HCARFramework(config=config)
learned_model, metrics = framework.run(
    positive_examples=examples,
    oracle_func=oracle,
    variables=variables,
    domains=domains,
    target_model=target
)
```

**Effect:**
- Passive learning detects: `[Count(X, 1) <= 6, ..., Count(X, 6) <= 5, ..., Count(X, 8) <= 6]`
- Hint system identifies: Majority bound is 6, outlier is 5
- Normalized output: All Count constraints use `<= 6`

## Benefits

1. **Corrects under-fitting:** Examples don't need to demonstrate full capacity
2. **Minimal expert input:** One hint fixes multiple constraints
3. **Principled approach:** Uses statistical analysis + domain knowledge
4. **Transparent:** Logs all normalizations for verification
5. **Validated:** Phase 2 refinement can still correct if hint is wrong

## Integration with Existing System

**Phase 1 (Passive Learning):**
1. Extract constraints from examples (as before)
2. **NEW:** Apply domain hints to normalize parameters
3. Continue with bias generation

**Phase 2 (Interactive Refinement):**
- Hint-normalized constraints are treated like other candidates
- If hint was wrong, queries will refute the constraint
- System remains sound (won't accept incorrect constraints)

**Phase 3 (Active Learning):**
- Proceeds as before with refined constraints

## Evaluation

**Without hints:**
- Nurse 6: `Count(roster, 6) <= 5` (under-fitted)
- S-Recall: 0% (model rejects valid solutions)

**With hints:**
- Nurse 6: `Count(roster, 6) <= 6` (corrected)
- S-Recall: Expected to improve (model matches target)

## Future Extensions

1. **Learned hints:** System learns which hints work across multiple problems
2. **Active hint elicitation:** Ask expert targeted questions when ambiguity detected
3. **Hint confidence:** Track which hints are reliable vs. uncertain
4. **Hint validation:** Verify hints against examples before applying
