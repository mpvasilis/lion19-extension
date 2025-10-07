import time
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables
from pycona.query_generation.pqgen import PQGen
from pycona.utils import get_con_subset, restore_scope_values

class EnhancedBayesianPQGen(PQGen):

    def __init__(self, *args, **kwargs):
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

        # Project down to only vars in scope of B
        Y2 = frozenset(get_variables(self.env.instance.bias))

        if len(Y2) < len(Y):
            Y = Y2

        lY = list(Y)
        
        B = get_con_subset(self.env.instance.bias, Y)


        Cl = get_con_subset(self.env.instance.cl, Y)
        # if len(Cl)==0:
        #     print("CL total size", len(self.env.instance.cl))
        #     for c in self.env.instance.cl:
        #         print(c)
        #     print("CL subset total size", len(Cl))
        #     input("CL is empty")


        # If no constraints left in B, just return
        if len(B) == 0:
            return set()

        # If no constraints learned yet, start by just generating an example in all the variables in Y
        if len(Cl) == 0:
            # print(" WARNING: Empty CL in PQGen")
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

        # We want at least one constraint to be violated to assure that each answer 
        # will lead to new information
        m += ~cp.all(B)
        
        for prev_assignment in all_previous_assignments:
            if len(prev_assignment) == len(lY):
                different_assignment = cp.any([lY[i] != prev_assignment[i] for i in range(len(lY))])
                m += different_assignment

        # Solve first without objective (to find at least one solution)
        flag = m.solve()

        t1 = time.time() - t0
        if not flag or (t1 > self.time_limit):
            # UNSAT or already above time_limit, stop here --- cannot optimize
            return lY if flag else set()

        # Next solve will change the values of the variables in lY
        # so we need to return them to the original ones to continue if we don't find a solution next
        values = [x.value() for x in lY]
        
        # Filter out None values before passing to solution_hint
        valid_vars = []
        valid_values = []
        for i, (var, val) in enumerate(zip(lY, values)):
            if val is not None:
                valid_vars.append(var)
                valid_values.append(val)
            else:
                print(f"⚠️ Skipping variable {var} with None value in solution_hint")

        # So a solution was found, try to find a better one now
        if valid_vars and valid_values:
            m.solution_hint(valid_vars, valid_values)
        else:
            print("⚠️ No valid variables for solution_hint")
        try:
            objective = self.obj(B=B, ca_env=self.env)
        except:
            raise NotImplementedError(f"Objective given not implemented in PQGen: {self.obj} - Please report an issue")

        # Run with the objective
        m.maximize(objective)

        flag2 = m.solve(time_limit=(self.time_limit - t1))

        if flag2:
            # Store this assignment for each constraint in B
            new_values = [x.value() for x in lY]
            
            # Filter out None values before storing
            valid_new_values = []
            for val in new_values:
                if val is not None:
                    valid_new_values.append(val)
                else:
                    print(f"⚠️ Found None value in optimization result")
            
            for constraint in B:
                self.add_previous_assignment(constraint, valid_new_values if len(valid_new_values) == len(lY) else new_values)
            
            return lY
        else:
            # Store the original solution if optimization failed
            # Filter out None values before storing
            valid_values_original = []
            for val in values:
                if val is not None:
                    valid_values_original.append(val)
                else:
                    print(f"⚠️ Found None value in original solution")
                    
            for constraint in B:
                self.add_previous_assignment(constraint, valid_values_original if len(valid_values_original) == len(lY) else values)
                
            restore_scope_values(lY, values)
            return lY 