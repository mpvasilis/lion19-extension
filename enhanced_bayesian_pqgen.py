import time
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from pycona.query_generation.pqgen import PQGen
from pycona.utils import get_con_subset, restore_scope_values


def weighted_violation_objective(B, ca_env):
    
    if not hasattr(ca_env, 'constraint_probs') or not ca_env.constraint_probs:

        return cp.sum([~c for c in B])

    if hasattr(ca_env, 'verbose') and ca_env.verbose >= 2:
        print("\nViolation Objective:")
        for c in B[:5]:  
            p = ca_env.constraint_probs.get(c, 0.5)
            weight = 1.0 - p
            print(f"  {c}: P={p:.3f} -> weight={weight:.3f}")
        if len(B) > 5:
            print(f"  ... and {len(B)-5} more constraints")



    weighted_sum = cp.sum([
        (1.0 - ca_env.constraint_probs.get(c, 0.5)) * (~c)
        for c in B
    ])

    return weighted_sum


class EnhancedBayesianPQGen(PQGen):

    def __init__(self, *args, **kwargs):

        if 'objective_function' not in kwargs:
            kwargs['objective_function'] = weighted_violation_objective
        super().__init__(*args, **kwargs)
        self.previous_assignments = {}
        
    def add_previous_assignment(self, constraint, assignment):

        constraint_str = str(constraint)
        if constraint_str not in self.previous_assignments:
            self.previous_assignments[constraint_str] = []
        
        self.previous_assignments[constraint_str].append(assignment)
    
    def get_previous_assignments(self, constraint):
        constraint_str = str(constraint)
        return self.previous_assignments.get(constraint_str, [])
    
    def get_assignment_stats(self):
        stats = {}
        for constraint_str, assignments in self.previous_assignments.items():
            stats[constraint_str] = len(assignments)
        return stats
    
    def print_assignment_history(self, constraint=None, max_to_show=3):
        if constraint:
            constraint_strs = [str(constraint)]
        else:
            constraint_strs = list(self.previous_assignments.keys())
            
        for c_str in constraint_strs:
            assignments = self.previous_assignments.get(c_str, [])
            print(f"Assignment history for {c_str} ({len(assignments)} total):")
            
            for i, assignment in enumerate(assignments[:max_to_show]):
                print(f"  Assignment {i+1}: {assignment}")
                
            if len(assignments) > max_to_show:
                print(f"  ... and {len(assignments) - max_to_show} more")
                
    def generate(self, Y=None):
        if Y is None:
            Y = self.env.instance.X
        assert isinstance(Y, list), "When generating a query, Y must be a list of variables"

        t0 = time.time()

        Y2 = frozenset(get_variables(self.env.instance.bias))

        if len(Y2) < len(Y):
            Y = Y2

        lY = list(Y)
        
        B = get_con_subset(self.env.instance.bias, Y)


        Cl = get_con_subset(self.env.instance.cl, Y)

        if len(B) == 0:
            return set()

        if len(Cl) == 0:

            Cl = self.env.instance.cl

        if not self.partial and len(B) > self.blimit:
            m = cp.Model(Cl)
            flag = m.solve() 

            if flag and not all([c.value() for c in B]):
                return lY
            else:
                self.partial = True
        
        all_previous_assignments = []
        for constraint in B:
            all_previous_assignments.extend(self.get_previous_assignments(constraint))
            
        m = cp.Model(Cl)


        m += ~cp.all(B)
        
        for prev_assignment in all_previous_assignments:
            if len(prev_assignment) == len(lY):
                different_assignment = cp.any([lY[i] != prev_assignment[i] for i in range(len(lY))])
                m += different_assignment

        flag = m.solve()

        t1 = time.time() - t0
        if not flag or (t1 > self.time_limit):

            return lY if flag else set()


        values = [x.value() for x in lY]

        valid_vars = []
        valid_values = []
        for i, (var, val) in enumerate(zip(lY, values)):
            if val is not None:
                valid_vars.append(var)
                valid_values.append(val)

        if valid_vars and valid_values:
            if hasattr(m, 'solution_hint'):
                m.solution_hint(valid_vars, valid_values)
            else:

                pass
        try:
            objective = self.obj(B=B, ca_env=self.env)
        except:
            raise NotImplementedError(f"Objective given not implemented in PQGen: {self.obj} - Please report an issue")



        try:
            weights_sum = 0.0
            if hasattr(self.env, 'constraint_probs') and self.env.constraint_probs:
                for c in B:
                    weights_sum += (1.0 - self.env.constraint_probs.get(c, 0.5))
            penalty_M = weights_sum + 1.0

            violations = [~c for c in B]
            all_violated = cp.boolvar(name="all_violated")
            m += (all_violated == (cp.sum(violations) == len(B)))

            final_objective = objective - penalty_M * all_violated
        except Exception:
            final_objective = objective

        m.maximize(final_objective)

        flag2 = m.solve(time_limit=(self.time_limit - t1))

        if flag2:

            new_values = [x.value() for x in lY]

            valid_new_values = []
            for val in new_values:
                if val is not None:
                    valid_new_values.append(val)
            
            for constraint in B:
                self.add_previous_assignment(constraint, valid_new_values if len(valid_new_values) == len(lY) else new_values)
            
            return lY
        else:


            valid_values_original = []
            for val in values:
                if val is not None:
                    valid_values_original.append(val)
                    
            for constraint in B:
                self.add_previous_assignment(constraint, valid_values_original if len(valid_values_original) == len(lY) else values)
                
            restore_scope_values(lY, values)
            return lY 