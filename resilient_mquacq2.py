"""
MQuAcq2 Implementation for HCAR Phase 3

This module provides a custom MQuAcq2 implementation that handles None returns from 
ResilientFindC gracefully, allowing MQuAcq-2 to continue learning even when
some constraints cannot be found in the bias.
"""

import time
import numpy as np
from pycona import MQuAcq2
from pycona.utils import get_kappa


class ResilientMQuAcq2(MQuAcq2):
    """
    A resilient version of MQuAcq2 that handles None returns from FindC.
    
    Key differences from standard MQuAcq2:
    1. Checks if FindC returns None before adding to CL
    2. Skips scopes where no constraint could be found
    3. Tracks skipped scopes for reporting
    4. Continues learning despite missing constraints in bias
    """
    
    def __init__(self, ca_env=None, **kwargs):
        super().__init__(ca_env, **kwargs)
        self.skipped_scopes = []  # Track scopes we couldn't learn
    
    def learn(self, instance, oracle=None, verbose=0, X=None, metrics=None):
        """
        Learn constraints using modified QuAcq with resilient FindC handling.
        
        This overrides the standard MQuAcq2.learn() to handle None returns from FindC.
        """
        from pycona import UserOracle
        
        if oracle is None:
            oracle = UserOracle()
        
        if X is None:
            X = instance.X
        assert isinstance(X, list) and set(X).issubset(set(instance.X)), \
            "When using .learn(), set parameter X must be a list of variables"
        
        self.env.init_state(instance, oracle, verbose, metrics)
        
        # Hash the variables
        self.hashX = [hash(x) for x in self.env.instance.X]
        self.cl_neighbours = np.zeros((len(self.env.instance.X), len(self.env.instance.X)), dtype=bool)
        
        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias()
        
        while True:
            gen_start = time.time()
            Y = self.env.run_query_generation(X)
            gen_end = time.time()
            self.env.metrics.increase_generation_time(gen_end - gen_start)
            
            if len(Y) == 0:
                # Converged
                self.env.metrics.finalize_statistics()
                if self.env.verbose >= 1:
                    print(f"\nLearned {self.env.metrics.cl} constraints in "
                          f"{self.env.metrics.membership_queries_count} queries.")
                    if len(self.skipped_scopes) > 0:
                        print(f"[RESILIENT] Skipped {len(self.skipped_scopes)} scope(s) due to missing constraints in bias")
                return self.env.instance
            
            self.env.metrics.increase_generated_queries()
            kappaB = get_kappa(self.env.instance.bias, Y)
            
            while len(kappaB) > 0:
                
                if self.env.verbose >= 3:
                    print("Size of CL: ", len(self.env.instance.cl))
                    print("Size of B: ", len(self.env.instance.bias))
                    print("Number of queries: ", self.env.metrics.total_queries)
                    print("MQuAcq-2 Queries: ", self.env.metrics.top_lvl_queries)
                    print("FindScope Queries: ", self.env.metrics.findscope_queries)
                    print("FindC Queries: ", self.env.metrics.findc_queries)
                
                self.env.metrics.increase_top_queries()
                
                if self.env.ask_membership_query(Y):
                    # It is a solution, so all candidates violated must go
                    self.env.remove_from_bias(kappaB)
                    kappaB = set()
                else:  # User says UNSAT
                    
                    scope = self.env.run_find_scope(Y)
                    c = self.env.run_findc(scope)
                    
                    # RESILIENT HANDLING: Check if FindC returned None
                    if c is None:
                        # FindC couldn't find a constraint for this scope
                        scope_str = f"[{', '.join(str(v) for v in scope)}]"
                        self.skipped_scopes.append(scope_str)
                        
                        # if self.env.verbose >= 2:
                        #     print(f"[RESILIENT] Skipping scope {scope_str} - no constraint found in bias")
                        
                        # Remove this scope from Y to avoid infinite loop
                        Y = [y for y in Y if y not in scope]
                        kappaB = get_kappa(self.env.instance.bias, Y)
                        continue
                    
                    # Standard MQuAcq2 logic: add constraint to CL
                    self.env.add_to_cl(c)
                    
                    NScopes = set()
                    NScopes.add(tuple(scope))
                    
                    if self.perform_analyzeAndLearn:
                        NScopes = NScopes.union(self.analyze_and_learn(Y))
                    
                    Y = [y2 for y2 in Y if not any(y2 in set(nscope) for nscope in NScopes)]
                    
                    kappaB = get_kappa(self.env.instance.bias, Y)
    
    @property
    def perform_analyzeAndLearn(self):
        """Getter for analyze and learn flag."""
        return self._perform_analyzeAndLearn
    
    def get_resilience_report(self):
        """Get report of resilience events."""
        return {
            'skipped_scopes_count': len(self.skipped_scopes),
            'skipped_scopes': self.skipped_scopes
        }

