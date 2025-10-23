"""
FindC Implementation for HCAR Phase 3

This module provides a custom FindC implementation that doesn't collapse when
a constraint is not found in the bias. Instead, it logs a warning and returns
None, allowing the learning process to continue.

This aligns with HCAR's principle of robustness: the bias might be imperfect
due to over-pruning or noise, and the system should gracefully handle such cases.
"""

from pycona.find_constraint.findc import FindC
from pycona.utils import get_con_subset, check_value, get_kappa, restore_scope_values


class ResilientFindC(FindC):
    """
    A resilient version of FindC that doesn't collapse when a constraint
    is not found in the bias.
    
    Key differences from standard FindC:
    1. When delta is empty (no candidates in bias), logs warning and returns None
    2. Allows MQuAcq-2 to continue learning even with imperfect bias
    3. Tracks and reports scopes that couldn't be resolved
    """
    
    def __init__(self, ca_env=None, time_limit=0.2, **kwargs):
        super().__init__(ca_env, time_limit, **kwargs)
        self.unresolved_scopes = []  # Track scopes we couldn't resolve
        self.collapse_warnings = 0
    
    def run(self, scope):
        """
        Run the FindC algorithm with resilient behavior.
        
        Instead of raising an exception when delta is empty, this version:
        - Logs a warning
        - Records the unresolved scope
        - Returns None to signal no constraint was found
        
        :param scope: The scope in which we search for a constraint.
        :return: The constraint found, or None if no candidates exist.
        """
        assert self.ca is not None
        
        # Initialize delta
        delta = get_con_subset(self.ca.instance.bias, scope)
        delta = [c for c in delta if check_value(c) is False]
        
        if len(delta) == 1:
            c = delta[0]
            return c
        
        if len(delta) == 0:
            # RESILIENT BEHAVIOR: Don't collapse, log and continue
            target_on_scope = get_kappa(self.ca.oracle.constraints, scope)
            scope_str = f"[{', '.join(str(v) for v in scope)}]"
            
            self.collapse_warnings += 1
            self.unresolved_scopes.append((scope, target_on_scope))
            
            print(f"\n[WARNING] FindC: No candidates in bias for scope {scope_str}")
            print(f"          Target constraint on this scope: {target_on_scope}")
            print(f"          Continuing learning (resilient mode)...")
            
            # Return None to signal no constraint found
            # The calling code should handle this gracefully
            return None
        
        # Standard FindC logic for when delta has multiple candidates
        sub_cl = get_con_subset(self.ca.instance.cl, scope)
        scope_values = [x.value() for x in scope]
        
        while True:
            flag = self.generate_findc_query(sub_cl, delta)
            
            if flag is False:
                if len(delta) == 0:
                    # RESILIENT BEHAVIOR: Return None instead of raising exception
                    scope_str = f"[{', '.join(str(v) for v in scope)}]"
                    print(f"[WARNING] FindC: Delta became empty during search on scope {scope_str}")
                    self.collapse_warnings += 1
                    self.unresolved_scopes.append((scope, "unknown"))
                    return None
                
                restore_scope_values(scope, scope_values)
                return delta[0]
            
            self.ca.metrics.increase_findc_queries()
            
            if self.ca.ask_membership_query(scope):
                [self.ca.remove_from_bias(c) for c in delta if check_value(c) is False]
                delta = [c for c in delta if check_value(c) is not False]
            else:
                [self.ca.remove_from_bias(c) for c in delta if check_value(c) is not False]
                delta = [c for c in delta if check_value(c) is False]
    
    def get_resilience_report(self):
        """
        Get a report on resilience events during learning.
        
        Returns:
            dict: Summary of unresolved scopes and warnings
        """
        return {
            'collapse_warnings': self.collapse_warnings,
            'unresolved_scopes': len(self.unresolved_scopes),
            'unresolved_details': [
                {
                    'scope': f"[{', '.join(str(v) for v in scope)}]",
                    'target': str(target)
                }
                for scope, target in self.unresolved_scopes
            ]
        }

