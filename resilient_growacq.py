
import copy
import time
from pycona import GrowAcq
from pycona.problem_instance import ProblemInstance
from pycona.answering_queries import Oracle, UserOracle
from pycona import Metrics


class ResilientGrowAcq(GrowAcq):
    """
    Resilient wrapper for GrowAcq that handles collapses gracefully.
    Uses ResilientMQuAcq2 as the inner algorithm by default.
    """
    
    def __init__(self, ca_env=None, inner_algorithm=None, **kwargs):
        # If no inner algorithm provided, use ResilientMQuAcq2
        if inner_algorithm is None:
            from resilient_mquacq2 import ResilientMQuAcq2
            inner_algorithm = ResilientMQuAcq2(ca_env=ca_env)
        
        super().__init__(ca_env=ca_env, inner_algorithm=inner_algorithm, **kwargs)
        self.skipped_scopes = []
        self.invalid_cl_constraints = []
        self.collapse_warnings = 0
    
    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, X=None, metrics: Metrics = None):
        """
        Learn constraints by incrementally adding variables and using the inner algorithm to learn constraints
        for each added variable. Handles collapses gracefully.
        
        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param X: The set of variables to consider, default is None.
        :param metrics: statistics logger during learning
        :return: the learned instance
        """
        if X is None:
            X = instance.X
        assert isinstance(X, list) and set(X).issubset(set(instance.X)), \
            "When using .learn(), set parameter X must be a list of variables"

        self.env.init_state(instance, oracle, verbose, metrics)

        if verbose >= 1:
            print(f"Running ResilientGrowAcq with {self.inner_algorithm.__class__.__name__} as inner algorithm")

        Y = []
        n_vars = len(X)
        
        for x in X:
            try:
                # we 'grow' the inner bias by adding one extra variable at a time
                Y.append(x)
                
                # add the constraints involving x and other added variables
                if len(self.env.instance.bias) == 0:
                    self.env.instance.construct_bias_for_var(x, Y)
                    
                if verbose >= 3:
                    print(f"Added variable {x} in GrowAcq")
                    print("size of B in growacq: ", len(self.env.instance.bias))

                if verbose >= 2:
                    print(f"\nGrowAcq: calling inner_algorithm for {len(Y)}/{n_vars} variables")
                
                # Update inner algorithm's environment before each call (like original GrowAcq)
                self.inner_algorithm.env = copy.copy(self.env)
                
                # Call inner algorithm with resilience
                try:
                    self.env.instance = self.inner_algorithm.learn(
                        self.env.instance, 
                        oracle, 
                        verbose=verbose, 
                        X=Y, 
                        metrics=self.env.metrics
                    )
                except Exception as e:
                    # Handle collapses in inner algorithm
                    print(f"\n[WARNING] Inner algorithm failed for variable {x}: {e}")
                    self.collapse_warnings += 1
                    
                    # Try to continue with next variable
                    # Collect resilience info from inner algorithm if available
                    if hasattr(self.inner_algorithm, 'get_resilience_report'):
                        inner_report = self.inner_algorithm.get_resilience_report()
                        if inner_report.get('skipped_scopes_count', 0) > 0:
                            self.skipped_scopes.extend(inner_report.get('skipped_scopes', []))
                        if inner_report.get('invalid_cl_constraints_count', 0) > 0:
                            self.invalid_cl_constraints.extend(inner_report.get('invalid_cl_constraints', []))
                    
                    # Continue to next variable
                    continue

                if verbose >= 3:
                    print("C_L: ", len(self.env.instance.cl))
                    print("B: ", len(self.env.instance.bias))
                    print("Number of queries: ", self.env.metrics.membership_queries_count)
                    print("Top level Queries: ", self.env.metrics.top_lvl_queries)
                    print("FindScope Queries: ", self.env.metrics.findscope_queries)
                    print("FindC Queries: ", self.env.metrics.findc_queries)
                    
            except Exception as e:
                # Handle any other errors during variable addition
                print(f"\n[WARNING] Error processing variable {x} in GrowAcq: {e}")
                self.collapse_warnings += 1
                # Continue to next variable
                continue
        
        # Collect final resilience info from inner algorithm
        if hasattr(self.inner_algorithm, 'get_resilience_report'):
            inner_report = self.inner_algorithm.get_resilience_report()
            if inner_report.get('skipped_scopes_count', 0) > 0:
                self.skipped_scopes.extend(inner_report.get('skipped_scopes', []))
            if inner_report.get('invalid_cl_constraints_count', 0) > 0:
                self.invalid_cl_constraints.extend(inner_report.get('invalid_cl_constraints', []))

        if verbose >= 3:
            print("Number of queries: ", self.env.metrics.membership_queries_count)
            print("Number of recommendation queries: ", self.env.metrics.recommendation_queries_count)
            print("Number of generalization queries: ", self.env.metrics.generalization_queries_count)
            print("Top level Queries: ", self.env.metrics.top_lvl_queries)
            print("FindScope Queries: ", self.env.metrics.findscope_queries)
            print("FindC Queries: ", self.env.metrics.findc_queries)
        
        if verbose >= 1:
            if len(self.skipped_scopes) > 0:
                print(f"[RESILIENT] GrowAcq skipped {len(self.skipped_scopes)} scope(s) due to missing constraints in bias")
            if self.collapse_warnings > 0:
                print(f"[RESILIENT] GrowAcq handled {self.collapse_warnings} collapse(s) during learning")

        self.env.metrics.finalize_statistics()
        return self.env.instance
    
    def get_resilience_report(self):
        """
        Get resilience report including skipped scopes and invalid constraints.
        Also aggregates reports from inner algorithm if available.
        """
        report = {
            'skipped_scopes_count': len(self.skipped_scopes),
            'skipped_scopes': self.skipped_scopes,
            'invalid_cl_constraints_count': len(self.invalid_cl_constraints),
            'invalid_cl_constraints': [str(c) for c in self.invalid_cl_constraints],
            'collapse_warnings': self.collapse_warnings
        }
        
        # Aggregate inner algorithm resilience report if available
        if hasattr(self.inner_algorithm, 'get_resilience_report'):
            inner_report = self.inner_algorithm.get_resilience_report()
            report['inner_algorithm'] = inner_report
        
        return report

