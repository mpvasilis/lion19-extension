# Resilient FindC Implementation

## Problem

During Phase 3 (MQuAcq-2 active learning), the standard FindC algorithm from the `pycona` library would **collapse** when it couldn't find a constraint in the bias for a given scope. This would raise an exception:

```
Exception: Collapse, the constraint we seek is not in B: []
```

This collapse behavior is problematic for HCAR because:

1. **Imperfect Bias**: The bias might have been over-pruned in Phase 1 or Phase 2
2. **No Robustness**: A single missing constraint would stop the entire learning process
3. **Violates HCAR Principles**: HCAR is designed to be resilient to imperfect inputs and noisy conditions

## Solution: ResilientFindC

We created a custom `ResilientFindC` class that **extends** the standard FindC with graceful failure handling:

### Key Features

1. **No Collapse**: When no constraints are found in the bias for a scope, instead of raising an exception:
   - Logs a warning with details about the missing constraint
   - Returns `None` to signal no constraint was found
   - Allows MQuAcq-2 to continue learning other constraints

2. **Resilience Tracking**: Maintains detailed statistics about resilience events:
   - Number of collapse warnings avoided
   - List of unresolved scopes
   - Target constraints that couldn't be learned (for analysis)

3. **Transparent Reporting**: Provides a `get_resilience_report()` method that returns:
   ```python
   {
       'collapse_warnings': int,  # Number of times collapse was avoided
       'unresolved_scopes': int,  # Number of scopes that couldn't be resolved
       'unresolved_details': [    # Detailed information for each unresolved scope
           {
               'scope': str,      # Variable scope as string
               'target': str      # Target constraint on that scope
           }
       ]
   }
   ```

## Implementation Details

### File: `resilient_findc.py`

The `ResilientFindC` class inherits from `pycona.find_constraint.findc.FindC` and overrides the `run()` method:

```python
class ResilientFindC(FindC):
    def run(self, scope):
        # ... standard FindC logic ...
        
        if len(delta) == 0:
            # INSTEAD OF: raise Exception("Collapse...")
            # DO THIS:
            self.collapse_warnings += 1
            self.unresolved_scopes.append((scope, target))
            print(f"[WARNING] No candidates in bias for scope {scope}")
            return None  # Graceful failure
```

### Integration: `run_phase3.py`

The ResilientFindC is injected into MQuAcq-2 through the `ActiveCAEnv`:

```python
# Create custom environment with ResilientFindC
resilient_findc = ResilientFindC()
custom_env = ActiveCAEnv(findc=resilient_findc)
ca_system = MQuAcq2(ca_env=custom_env)

# Run learning with resilient behavior
learned_instance = ca_system.learn(mquacq_instance, oracle=oracle_binary, verbose=3)

# Get resilience report after learning
report = resilient_findc.get_resilience_report()
```

## Alignment with HCAR Methodology

This implementation aligns with several core HCAR principles:

### 1. **Robustness (Constraint 4)**

> "ALL oracle feedback must be integrated probabilistically. Hard refutation is forbidden."

While this isn't directly about oracle responses, the principle extends to **all sources of uncertainty**. If the bias is imperfect (due to over-pruning or noise), the system should handle it gracefully rather than failing catastrophically.

### 2. **Independence of Biases (Constraint 1)**

> "B_globals and B_fixed MUST be maintained as independent hypothesis sets to prevent irreversible information loss."

By being resilient to missing constraints in B_fixed during Phase 3, we avoid situations where over-pruning of B_fixed (even if done correctly) causes complete system failure.

### 3. **Practical Guarantee (Under Noisy Oracle)**

> "The system's probabilistic nature ensures it will converge to the correct model in expectation, provided the oracle's error rate is not excessively high."

Similarly, if the bias error rate (missing constraints) is not excessively high, the system should still learn a reasonable model rather than collapsing entirely.

## Expected Behavior

### Without ResilientFindC (Original)
```
[ERROR] MQuAcq-2 failed: Collapse, the constraint we seek is not in B: []
Traceback (most recent call last):
  ...
Exception: Collapse, the constraint we seek is not in B: []
[SYSTEM EXIT]
```

### With ResilientFindC (New)
```
[INFO] Using ResilientFindC to handle imperfect bias gracefully

[WARNING] FindC: No candidates in bias for scope [x1, x2]
          Target constraint on this scope: (x1 != x2)
          Continuing learning (resilient mode)...

Phase 3 Results
===============
MQuAcq-2 queries: 245
MQuAcq-2 time: 12.34s
Final model constraints: 48

Resilience Report:
  Collapse warnings avoided: 3
  Unresolved scopes: 3
  Unresolved scope details:
    - [x1, x2]: target = (x1 != x2)
    - [x3, x4]: target = (x3 != x4)
    - [x5, x6]: target = (x5 != x6)
```

## Trade-offs

### Advantages
1. ✅ **No catastrophic failures**: System continues learning despite imperfect bias
2. ✅ **Detailed diagnostics**: Tracks exactly which constraints couldn't be learned
3. ✅ **Methodologically sound**: Aligns with HCAR's robustness principles
4. ✅ **Non-invasive**: Extends pycona's FindC without modifying library code

### Limitations
1. ⚠️ **Incomplete models**: If too many constraints are missing from bias, the learned model may be incomplete
2. ⚠️ **Precision still matters**: Missing constraints mean lower recall (though precision remains high)
3. ⚠️ **Requires analysis**: Need to examine resilience report to understand what was missed

## Usage

### Basic Usage
```python
from resilient_findc import ResilientFindC
from pycona.ca_environment import ActiveCAEnv
from pycona import MQuAcq2

# Create resilient FindC
resilient_findc = ResilientFindC()

# Inject into MQuAcq-2 via environment
env = ActiveCAEnv(findc=resilient_findc)
ca_system = MQuAcq2(ca_env=env)

# Run learning
learned = ca_system.learn(instance, oracle=oracle, verbose=3)

# Check resilience report
report = resilient_findc.get_resilience_report()
if report['collapse_warnings'] > 0:
    print(f"⚠️  {report['collapse_warnings']} collapse(s) avoided")
    print(f"   {report['unresolved_scopes']} scope(s) couldn't be resolved")
```

### In run_phase3.py
The resilient FindC is automatically used when running Phase 3:

```bash
python run_phase3.py --experiment sudoku --phase2_pickle phase2_output/sudoku_phase2.pkl
```

## Future Enhancements

1. **Adaptive Bias Expansion**: When a scope is unresolved, dynamically generate candidate constraints for that scope
2. **Confidence Thresholding**: Only skip scopes if confidence in missing constraint is below threshold
3. **Iterative Refinement**: After Phase 3, revisit unresolved scopes with targeted queries
4. **Statistical Analysis**: Correlate unresolved scopes with Phase 2 decisions to improve pruning

## Conclusion

The ResilientFindC implementation makes HCAR Phase 3 **robust to imperfect bias**, preventing catastrophic failures and enabling the system to learn as much as possible even when some constraints are missing. This aligns perfectly with HCAR's core philosophy of **intelligent resilience** over **brittle precision**.

