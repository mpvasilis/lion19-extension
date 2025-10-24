import builtins
import math
import os
import pickle
import random
import sys
import numpy as np
import pandas as pd
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables
from numpy.f2py.auxfuncs import throw_error
from pycona.utils import get_scope
from cpmpy.expressions.core import Comparison
from cpmpy.expressions.globalconstraints import AllDifferent
from cpmpy.expressions.python_builtins import all, any, sum
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl
from sklearn.cluster import AgglomerativeClustering
import time
from itertools import combinations
from benchmarks_global import construct_nurse_rostering as nr_global
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks import construct_examtt_simple, construct_nurse_rostering, construct_sudoku_binary
from pycona import *
from cpmpy import AllDifferent, sum as cpmsum
from benchmarks_global import construct_sudoku
from bayesian_ca_env import BayesianActiveCAEnv
from bayesian_quacq import BayesianQuAcq
from pycona.query_generation import PQGen
from feature_extraction import extract_constraint_features
from enhanced_bayesian_pqgen import EnhancedBayesianPQGen
from bayesian_ca_env import BayesianActiveCAEnv
from bayesian_quacq import BayesianQuAcq

def active_learning_system(
    experiment, global_constraints, model, variables, n, oracle, clusters,
    total_query_budget=1500, initial_budget_per_constraint=30, use_fuzzy_rl=False, use_rl_bayesian=False,
    use_standard_pqgen=False, global_timeout=600,
    max_consecutive_failures=10,
    min_probability_threshold=0.2,
    initial_valid_constraints=None,
    initial_bias_constraints=None,
    max_synthetic_constraints=1,
    phase2_only=False
):
    start_time = time.time()
    total_queries = 0
    total_subsets_added = 0
    feature_evaluation_queries = 0  

    constraint_tree = {}  
    constraint_children = {}  
    constraint_level = {}  

    def register_subset(parent_constraint, subset_constraint):

        constraint_tree[subset_constraint] = parent_constraint

        if parent_constraint not in constraint_children:
            constraint_children[parent_constraint] = []

        constraint_children[parent_constraint].append(subset_constraint)

        if parent_constraint in constraint_level:
            constraint_level[subset_constraint] = constraint_level[parent_constraint] + 1
        else:
            constraint_level[parent_constraint] = 0
            constraint_level[subset_constraint] = 1
    
    def remove_constraint_family(constraint, reason="accepted"):

        constraints_to_remove = set()
        
        if reason == "accepted":

            if constraint in constraint_tree:
                parent = constraint_tree[constraint]

                siblings = constraint_children.get(parent, [])

                for sibling in siblings:
                    constraints_to_remove.add(sibling)

                    stack = [sibling]
                    while stack:
                        current = stack.pop()
                        if current in constraint_children:
                            children = constraint_children[current]
                            constraints_to_remove.update(children)
                            stack.extend(children)
        
        elif reason == "rejected":


            pass
        
        constraints_removed = 0
        for c in constraints_to_remove:
            if c in bias_constraints:
                bias_constraints.remove(c)
                if c in bias_proba:
                    bias_proba.pop(c)
                constraints_removed += 1
                print(f"Removed related constraint: {c} (level {constraint_level.get(c, '?')})")
        
        if constraints_removed > 0:
            print(f"Removed {constraints_removed} related constraints")
        
        return constraints_removed

    def calculate_constraint_weight(constraint):

        prob = bias_proba.get(constraint, 0.5)
        depth = constraint_level.get(constraint, 0)


        uncertainty_score = 1.0 - abs(prob - 0.5)  
        depth_penalty = max(0.5, 1.0 - 0.1 * depth)  
        
        return uncertainty_score * depth_penalty

    def allocate_budgets(constraints, total_budget):
        if not constraints:
            return {}

        weights = {c: calculate_constraint_weight(c) for c in constraints}
        total_weight = sum(weights.values())
        
        if total_weight <= 0 or total_budget <= 0:

            budget_per_constraint = max(1, total_budget // len(constraints))
            return {c: budget_per_constraint for c in constraints}

        allocations = {}
        for c in constraints:
            allocations[c] = max(1, int(total_budget * weights[c] / total_weight))

        leftover = total_budget - sum(allocations.values())
        if leftover > 0:

            sorted_constraints = sorted(constraints, key=lambda c: weights[c], reverse=True)
            for i in range(leftover):
                allocations[sorted_constraints[i % len(sorted_constraints)]] += 1
        
        return allocations

    learned_constraints =  []

    bias_constraints = []
    bias_proba = {}

    if initial_valid_constraints:
        print(f"\n=== Starting with {len(initial_valid_constraints)} constraints from passive learning ===")
        for constraint in initial_valid_constraints:
            if constraint is not None:  

                constraint_str = str(constraint).lower()
                is_simple_assignment = (
                    " == " in str(constraint) and 
                    "sum" not in constraint_str and 
                    "alldifferent" not in constraint_str and 
                    "count" not in constraint_str and
                    not any(op in str(constraint) for op in [" >= ", " <= ", " != ", " > ", " < "])
                )
                
                if not is_simple_assignment:
                    print(f"  Initial constraint: {constraint}")
                    learned_constraints.append(constraint)

        
        print(f"  Loaded {len(learned_constraints)} valid global constraints (filtered out simple assignments)")

        for constraint in learned_constraints:
            if constraint not in global_constraints:
                global_constraints.append(constraint)

    import joblib

    try:
        xgb_clf = joblib.load("constraint_classifier_xgb.joblib")
        import pandas as pd
        feature_cols_path = "xgb_feature_columns.txt"
        if not os.path.exists(feature_cols_path):
            print(f"ERROR: Feature columns file '{feature_cols_path}' not found. Please re-train the model and save the feature columns after get_dummies.")
           
        with open(feature_cols_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
    except Exception as e:
        xgb_clf = None
        feature_cols = None

    for gc in global_constraints:
        print(gc)
        constraint = gc
        if "sum" in str(constraint).lower() or "alldifferent" in str(constraint).lower() or "count" in str(constraint).lower():





            bias_constraints.append(constraint)
            prob = 0.5
            if xgb_clf is not None and feature_cols is not None:
                try:
                    feats = extract_constraint_features(constraint, variables, all_constraints=global_constraints)
                    feat_df = pd.DataFrame([feats])
                    cat_cols = feat_df.select_dtypes(include=['object']).columns.tolist()
                    if cat_cols:
                        feat_df = pd.get_dummies(feat_df, columns=cat_cols)
                    feat_df = feat_df.reindex(columns=feature_cols, fill_value=0)

                    prob = float(xgb_clf.predict_proba(feat_df)[0][1])
                    feature_evaluation_queries += 1  
                    if prob < 0.9:
                        prob = random.uniform(0.90, 0.95)
                except Exception as e:
                    print(f"Error predicting probability for constraint {constraint}: {e}")
                    prob = 0.99
            else:
                prob = 0.99
            bias_proba[constraint] = prob

            constraint_level[constraint] = 0

    if initial_bias_constraints:
        print(f"\n=== Adding {len(initial_bias_constraints)} bias constraints from passive learning ===")
        for constraint in initial_bias_constraints:
            if constraint is not None and constraint not in bias_constraints:
                bias_constraints.append(constraint)
                bias_proba[constraint] = 0.8 
                constraint_level[constraint] = 0
                print(f"  Added bias constraint: {constraint}")
            elif constraint is None:
                print(f"   Skipping None bias constraint from passive learning")
            else:
                print(f"   Bias constraint already exists: {constraint}")

    print("Total constraints in target model: ", len(bias_constraints))
    print(f"Total learned constraints from passive learning: {len(learned_constraints)}")

    print("\nAll constraints in bias_proba:")
    for constraint, probability in bias_proba.items():
        print(f"Constraint: {constraint}, Probability: {probability:.3f}")
    

    variables = get_variables(bias_constraints + learned_constraints)

    total_constraints_learned = len(learned_constraints)  
    total_constraints_removed = 0
    total_subsets_ignored_already_in_bias = 0
    remaining_global_budget = total_query_budget if total_query_budget is not None else float('inf')
    constraint_budgets = {}
    unused_budget = 0

    subset_exploration_constraints = set()
    subset_stats = {
        'satisfied': 0,
        'violated': 0
    }








    extended_constraints = []
    constraints_to_remove = []
    synthetic_constraints_count = 0
    for constraint in bias_constraints:
        constraint_str = str(constraint).lower()
        vars_in_constraint = get_variables(constraint)

        if len(variables) <= len(vars_in_constraint):
            continue

        available_vars = []
        for v in set(variables):
            if v not in set(vars_in_constraint):
                available_vars.append(v)

        if "alldifferent" in constraint_str:
            if len(available_vars) >= 1 and synthetic_constraints_count < max_synthetic_constraints:
                new_vars = vars_in_constraint + [available_vars[0]]
                new_constraint = AllDifferent(new_vars)
                extended_constraints.append(new_constraint)
                synthetic_constraints_count += 1
                prob = 0.1
                if xgb_clf is not None and feature_cols is not None:
                    try:
                        feats = extract_constraint_features(new_constraint, variables, all_constraints=global_constraints)
                        feat_df = pd.DataFrame([feats])
                        cat_cols = feat_df.select_dtypes(include=['object']).columns.tolist()
                        if cat_cols:
                            feat_df = pd.get_dummies(feat_df, columns=cat_cols)
                        feat_df = feat_df.reindex(columns=feature_cols, fill_value=0)
                        feat_df['is_column'] = 0
                        feat_df['is_row'] = 0
                        
                        prob = float(xgb_clf.predict_proba(feat_df)[0][1])
                        feature_evaluation_queries += 1  
                    except Exception as e:
                        print(f"Error predicting probability for extended constraint {new_constraint}: {e}")
                        prob = 0.1


                if prob > 0.9:
                    prob = random.uniform(0.5, 0.7)
                bias_proba[new_constraint] = prob
                constraints_to_remove.append(constraint)
                print(f"Generated synthetic AllDifferent constraint {synthetic_constraints_count}/{max_synthetic_constraints}: {new_constraint}")
            elif synthetic_constraints_count >= max_synthetic_constraints:
                print(f"Reached maximum synthetic constraints limit ({max_synthetic_constraints}). Skipping further AllDifferent constraint generation.")






            else:




                pass



        elif "sum" in constraint_str and "sum" in str(constraint.args[0]).lower():
            if len(available_vars) >= 1:
                sum_terms = []
                for v in vars_in_constraint:
                    sum_terms.append(v)
                sum_terms = sum_terms + [available_vars[0]]
                new_constraint = Comparison(constraint.name, sum(sum_terms),constraint.args[1])
                extended_constraints.append(new_constraint)
                prob = 0.1
                if xgb_clf is not None and feature_cols is not None:
                    try:
                        feats = extract_constraint_features(new_constraint, variables, all_constraints=global_constraints)
                        feat_df = pd.DataFrame([feats])
                        cat_cols = feat_df.select_dtypes(include=['object']).columns.tolist()
                        if cat_cols:
                            feat_df = pd.get_dummies(feat_df, columns=cat_cols)
                        feat_df = feat_df.reindex(columns=feature_cols, fill_value=0)
                        prob = float(xgb_clf.predict_proba(feat_df)[0][1])
                        feature_evaluation_queries += 1  
                    except Exception as e:
                        print(f"Error predicting probability for extended constraint {new_constraint}: {e}")
                        prob = 0.1


                bias_proba[new_constraint] = prob
                constraints_to_remove.append(constraint)
                register_subset(constraint, new_constraint)
                print("extended constraint: ", new_constraint)
        elif "count" in constraint_str:
            count_cstr_val = constraint.args[0].args[1]
            if len(available_vars) >= 1:
                sum_terms = []
                for v in vars_in_constraint:
                    sum_terms.append(v)
                sum_terms = sum_terms + [available_vars[0]]
                new_constraint = Comparison(constraint.name, Count(sum_terms,count_cstr_val),constraint.args[1])
                extended_constraints.append(new_constraint)
                prob = 0.1
                if xgb_clf is not None and feature_cols is not None:
                    try:
                        feats = extract_constraint_features(new_constraint, variables, all_constraints=global_constraints)
                        feat_df = pd.DataFrame([feats])
                        cat_cols = feat_df.select_dtypes(include=['object']).columns.tolist()
                        if cat_cols:
                            feat_df = pd.get_dummies(feat_df, columns=cat_cols)
                        feat_df = feat_df.reindex(columns=feature_cols, fill_value=0)
                        prob = float(xgb_clf.predict_proba(feat_df)[0][1])
                        feature_evaluation_queries += 1  
                    except Exception as e:
                        print(f"Error predicting probability for extended constraint {new_constraint}: {e}")
                        prob = 0.1


                bias_proba[new_constraint] = prob
                constraints_to_remove.append(constraint)
                register_subset(constraint, new_constraint)
                print("extended constraint: ", new_constraint)

            else:




                pass








    for constraint in extended_constraints:
        bias_constraints.append(constraint)

    print(f"Initial bias constraints ({len(bias_constraints)}):")
    for c in bias_constraints:
        print(f"Constraint: {c}")

    initial_bias_constraints = len(bias_constraints)
    initial_bias_constraints_set = set(bias_constraints)


    total_constraints = len(global_constraints)
  
    budget_per_constraint = total_query_budget // len(bias_constraints)
    constraint_budgets = {c: budget_per_constraint for c in bias_constraints}
    
    leftover = total_query_budget - (budget_per_constraint * len(bias_constraints))
    for i in range(leftover):
        constraint_budgets[list(bias_constraints)[i % len(bias_constraints)]] += 1
    
    print(f"Initial budget per constraint: {budget_per_constraint}")
   

    
    print("\n===== Initial Budget Allocation =====")
    total_allocated = sum(constraint_budgets.values())
    print(f"Total budget: {total_query_budget}, Allocated: {total_allocated}")
    for i, c in enumerate(bias_constraints):
        print(f"Constraint {i+1}/{len(bias_constraints)}: {c}")
        print(f"  Budget: {constraint_budgets[c]}")
        print(f"  Probability: {bias_proba.get(c, 0.5):.3f}")
        if c in constraint_level:
            print(f"  Level: {constraint_level[c]}")
    print("=====================================\n")

   

    constraint_learning_history = {c: {'queries': [], 'prob_changes': []} for c in bias_constraints}
    
    tree_stats = {
        'max_depth': 0,
        'constraints_by_level': {},
        'accepted_by_level': {},
        'rejected_by_level': {}
    }
    
    while True:
        if total_query_budget is not None and total_queries >= total_query_budget:
            print(f"Reached maximum query budget ({total_query_budget}). Terminating.")
            break
            
        if global_timeout and (time.time() - start_time) > global_timeout:
            print(f"Global timeout ({global_timeout}s) reached. Terminating.")
            break
        
        if len(bias_proba) == 0:
            break

        next_constraint = None
        min_prob = float('inf')
        
        for constraint, prob in bias_proba.items():
            remaining_budget = constraint_budgets.get(constraint, 0)
            if remaining_budget <= 0:  
                continue
                
            if prob < min_prob:
                min_prob = prob
                next_constraint = constraint
                
        if next_constraint:
            min_prob_constraint = next_constraint
        else:
            min_prob_constraint = min(bias_proba.items(), key=lambda x: x[1])[0]
        
        min_prob = bias_proba[min_prob_constraint]
            
        print("--------------------------------")
        constraint_level_str = f" (level {constraint_level.get(min_prob_constraint, 0)})"
        print(f"❗ Selected constraint with lowest probability: {min_prob_constraint} (p={min_prob:.3f}){constraint_level_str}")
        print(f"Budget allocated: {constraint_budgets.get(min_prob_constraint, 0)}")

        if min_prob_constraint in constraint_tree:
            parent = constraint_tree[min_prob_constraint]
            print(f"➡️ This is a subset of: {parent}")

        Y = get_variables(min_prob_constraint)
        init_cl = set()
        for c in set(learned_constraints):
            if str(c) != str(min_prob_constraint):
                init_cl.add(c)
        for c in set(bias_constraints):
            if str(c) != str(min_prob_constraint):
                init_cl.add(c)

        init_cl = list(init_cl)
        print("Total CL: ", len(init_cl))

        vars_for_instance= get_variables(init_cl+[min_prob_constraint])
        
        instance = ProblemInstance(
        variables=cpm_array(vars_for_instance),
        init_cl=init_cl,
        name="pqgen-query-generation",
        bias=[min_prob_constraint])

        alpha=0.42
        prior=min_prob
        env = BayesianActiveCAEnv(
            qgen=EnhancedBayesianPQGen(),
            theta_max=0.9, 
            theta_min=0.1,
            prior=prior,
            alpha=alpha
        )

        env.constraint_probs = {}
        for c in instance.bias:
            env.constraint_probs[c] = bias_proba.get(c, prior)
        
        ca_system = BayesianQuAcq(ca_env=env)
        learned_instance = None
        
        prev_prob = bias_proba.get(min_prob_constraint, prior)
        
        try:
            max_queries = min(constraint_budgets.get(min_prob_constraint, 20), remaining_global_budget)
            env.max_queries = max_queries
            
            print(f"Setting query budget: {max_queries} for constraint {min_prob_constraint}")
            
            learned_instance = ca_system.learn(instance, oracle=oracle, verbose=10)
            
            for c, prob in env.constraint_probs.items():
                bias_proba[c] = prob
                
            try:
                constraint_learning_history[min_prob_constraint]['queries'].append(env.metrics.total_queries)
                constraint_learning_history[min_prob_constraint]['prob_changes'].append(
                    abs(bias_proba.get(min_prob_constraint, 0) - prev_prob)
                )
            except Exception as e:
                print(f"Error recording learning progress for constraint {min_prob_constraint}: {e}")
            
        except Exception as e:
            learned_instance = None
            learned_constraints.append(min_prob_constraint)
            bias_constraints.remove(min_prob_constraint)
            bias_proba.pop(min_prob_constraint)
            env.metrics.total_queries=0
            print(f"Error learning constraint: {e}")
            import traceback
            traceback.print_exc()
            continue
    
        valid_constraints = set(learned_instance.cl)

        current_level = constraint_level.get(min_prob_constraint, 0)
        
        tree_stats['max_depth'] = max(tree_stats['max_depth'], current_level)
        if current_level not in tree_stats['constraints_by_level']:
            tree_stats['constraints_by_level'][current_level] = 0
            tree_stats['accepted_by_level'][current_level] = 0
            tree_stats['rejected_by_level'][current_level] = 0
        tree_stats['constraints_by_level'][current_level] += 1

        if min_prob_constraint in valid_constraints:
            if min_prob_constraint not in set(learned_constraints):
                print(f"✅ Learned constraint {min_prob_constraint} (scope len: {len(Y)})")
                learned_constraints.append(min_prob_constraint)
                total_constraints_learned += 1
                tree_stats['accepted_by_level'][current_level] += 1
                
                removed_count = remove_constraint_family(min_prob_constraint, "accepted")
                print(f"Removed {removed_count} related constraints after acceptance")
            else:
                print(f"⚠️ Constraint {min_prob_constraint} already learned")
        

            if min_prob_constraint in set(subset_exploration_constraints):
                subset_stats['satisfied'] += 1
                print(f"✅ Subset exploration constraint satisfied! (Total: {subset_stats['satisfied']})")
        
            if min_prob_constraint not in constraint_budgets:
                constraint_budgets[min_prob_constraint] = 20
            unused_budget += constraint_budgets[min_prob_constraint] - env.metrics.total_queries
            if min_prob_constraint in bias_proba:
                bias_proba.pop(min_prob_constraint)
            if min_prob_constraint in bias_constraints:
                bias_constraints.remove(min_prob_constraint)

        else:
            print(f"❌ Removed Constraint {min_prob_constraint} (scope len: {len(Y)}).")
            total_constraints_removed += 1
            tree_stats['rejected_by_level'][current_level] += 1

            if min_prob_constraint in set(oracle.constraints):
                input("Removed")

            if min_prob_constraint in set(subset_exploration_constraints):
                subset_stats['violated'] += 1
                print(f"❌ Subset exploration constraint violated! (Total: {subset_stats['violated']})")

            print("Starting to generate subsets")
            if min_prob_constraint in constraint_budgets:
                remaining_constraint_budget = constraint_budgets[min_prob_constraint] - env.metrics.total_queries
            else:
                constraint_budgets[min_prob_constraint] = 20
                remaining_constraint_budget = 20

            max_allowed_depth = 3
            if constraint_level.get(min_prob_constraint, 0) >= max_allowed_depth:
                print(f"⚠️ Maximum subset depth reached ({max_allowed_depth}). Not generating more subsets.")

                if min_prob_constraint in bias_constraints:
                    bias_constraints.remove(min_prob_constraint)
                if min_prob_constraint in bias_proba:
                    bias_proba.pop(min_prob_constraint)
                continue

            remaining_parent_budget = remaining_constraint_budget

            original_scope = get_variables(min_prob_constraint)
            subset_indices = []
            constraint_type = str(min_prob_constraint).lower()

            if "alldifferent" in constraint_type:

                num_possible_subsets = min(max(2, len(original_scope) // 2), 5)

                if len(original_scope) >= 3:
                    subset_indices = [0, len(original_scope) // 2, len(original_scope) - 1]

                    if num_possible_subsets > 3 and len(original_scope) > 4:
                        subset_indices.append(len(original_scope) // 3)
                    if num_possible_subsets > 4 and len(original_scope) > 5:
                        subset_indices.append(2 * len(original_scope) // 3)
            else:

                num_possible_subsets = 3
                if len(original_scope) >= 3:
                    subset_indices = [0, len(original_scope) // 2, len(original_scope) - 1]

            avg_queries_per_constraint = 0
            if min_prob_constraint in constraint_learning_history:
                if len(constraint_learning_history[min_prob_constraint]['queries']) > 0:
                    avg_queries = sum(constraint_learning_history[min_prob_constraint]['queries']) / len(constraint_learning_history[min_prob_constraint]['queries'])
                avg_prob_change = sum(constraint_learning_history[min_prob_constraint]['prob_changes']) / len(constraint_learning_history[min_prob_constraint]['prob_changes'])
                
                efficiency = avg_prob_change / (avg_queries + 1e-10)  
                scaling_factor = 1.0 + min(0.5, max(-0.5, (efficiency - 0.05) * 5))
            else:
                scaling_factor = 1.0

            base_subset_budget = int((remaining_parent_budget // max(1, num_possible_subsets)) * scaling_factor)


            subset_constraints = []
            
            for idx in subset_indices:
                if idx >= len(original_scope):
                    continue
                    
                subset_scope = original_scope[:idx] + original_scope[idx+1:]
                if len(subset_scope) < 2:
                    continue
                    
                try:
                    if 'sum' in str(min_prob_constraint).lower() and "sum" in str(min_prob_constraint.args[0]).lower():
                        subset_vars = original_scope[:idx] + original_scope[idx+1:]
                        subset_constraint = Comparison(min_prob_constraint.name, sum(subset_vars), min_prob_constraint.args[1])
                    elif 'count' in str(min_prob_constraint).lower():
                        count_cstr_val = min_prob_constraint.args[0].args[1]
                        subset_vars = original_scope[:idx] + original_scope[idx+1:]
                        subset_constraint = Comparison(min_prob_constraint.name, Count(subset_vars, count_cstr_val), min_prob_constraint.args[1])
                    else:
                        vars_all = get_variables(min_prob_constraint)
                        subset_vars = vars_all[:idx] + vars_all[idx+1:]
                        subset_constraint = AllDifferent(subset_vars)

                    if subset_constraint not in set(bias_proba) and subset_constraint not in set(learned_constraints) and len(get_variables(subset_constraint)) > 2:
                        subset_constraints.append(subset_constraint)

                        register_subset(min_prob_constraint, subset_constraint)
                        
                except Exception as e:
                    print(f"Failed to create subset constraint: {e}")

            if subset_constraints:

                total_subset_budget = remaining_constraint_budget

                if len(subset_constraints) > 0:

                    budget_per_subset = total_subset_budget // len(subset_constraints)
                    subset_budgets = {c: budget_per_subset for c in subset_constraints}

                    leftover = total_subset_budget - (budget_per_subset * len(subset_constraints))
                    subset_list = list(subset_constraints)
                    for i in range(leftover):
                        subset_budgets[subset_list[i % len(subset_list)]] += 1
                    
                    print(f"Distributing parent's remaining budget ({total_subset_budget}) equally to {len(subset_constraints)} subsets, {budget_per_subset} each")

                    print("\n----- Subset Budget Allocation -----")
                    print(f"Parent constraint: {min_prob_constraint}")
                    print(f"Parent's remaining budget: {total_subset_budget}")
                    print(f"Number of subsets: {len(subset_constraints)}")
                    print(f"Budget per subset: {budget_per_subset}")
                    for i, sc in enumerate(subset_constraints):
                        print(f"Subset {i+1}: {sc}")
                        print(f"  Budget: {subset_budgets[sc]}")
                    print("-----------------------------------\n")
                
                for subset_constraint in subset_constraints:
                    bias_constraints.append(subset_constraint)
                    bias_proba[subset_constraint] = 0.5
                    constraint_budgets[subset_constraint] = subset_budgets[subset_constraint]
                    subset_exploration_constraints.add(subset_constraint)
                    subset_level = constraint_level.get(subset_constraint, 0)
                    print(f"Added subset constraint: {subset_constraint} (level {subset_level})")
                    print(f"Allocated budget: {subset_budgets[subset_constraint]}")
                    total_subsets_added += 1
            else:
                print("No valid subsets could be generated.")

            removed_count = remove_constraint_family(min_prob_constraint, "rejected")

            if min_prob_constraint in bias_constraints:
                bias_constraints.remove(min_prob_constraint)
            if min_prob_constraint in bias_proba:
                bias_proba.pop(min_prob_constraint)

        total_queries += env.metrics.total_queries
        if total_query_budget is not None:
            remaining_global_budget = total_query_budget - total_queries

        if len(bias_constraints) > 0 and unused_budget > 0:

            extra_per_constraint = unused_budget // len(bias_constraints)
            
            if extra_per_constraint > 0:

                for c in bias_constraints:
                    constraint_budgets[c] += extra_per_constraint

                leftover = unused_budget - (extra_per_constraint * len(bias_constraints))
                bias_list = list(bias_constraints)
                for i in range(leftover):
                    constraint_budgets[bias_list[i % len(bias_list)]] += 1
                
                print(f"Redistributed {unused_budget} unused budget equally, {extra_per_constraint} per constraint")
                
            unused_budget = 0

        print(f"Total queries: {total_queries}")
        print(f"Remaining bias constraints: {len(bias_constraints)}")
        print("Total constraints learned: ", total_constraints_learned)
        print("Total constraints removed: ", total_constraints_removed)
        print("--------------------------------")


    print("\nAdding remaining constraints in bias_proba to learned_constraints:")
    for constraint in list(bias_proba.keys()):
        if constraint not in learned_constraints:
            learned_constraints.append(constraint)
            print(f"Added remaining constraint: {constraint}")
    
    bias_proba.clear()
    end_time = time.time()
    total_duration = end_time - start_time
    print("----------------------------------")

    print("Target model constraints:")
    for c in oracle.constraints:
        if "sum" in str(c).lower() or "alldifferent" in str(c).lower() or "count" in str(c).lower():
            print(f"Constraint: {c}")

    print("\nLearned Constraints Analysis:")
    target_model_constraints = []
    for gc in global_constraints:
        constraint = gc
        if "sum" in str(constraint).lower() or "alldifferent" in str(constraint).lower() or "count" in str(constraint).lower():
            target_model_constraints.append(str(constraint))

    matching_constraints = 0
    for learned in learned_constraints:
        learned_str = str(learned)
        if learned_str in target_model_constraints:
            matching_constraints += 1
            print(f"    ✓ {learned} (matches target model)")
        else:
            print(f"    ✗ {learned} (not in target model)")
    if len(learned_constraints) > 0:
        print(f"\nMatching constraints: {matching_constraints}/{len(learned_constraints)} ({matching_constraints/len(learned_constraints)*100:.1f}% of learned constraints)")
        print(f"Target model coverage: {matching_constraints}/{len(target_model_constraints)} ({matching_constraints/len(target_model_constraints)*100:.1f}% of target model)")
    else:
        print(f"No constraints learned")
    print("Total violation queries:", total_queries)
    print("Total subsets added: ", total_subsets_added)
    print("Total constraints learned: ", total_constraints_learned)
    print("Total constraints removed: ", total_constraints_removed)
    print("Initial bias constraints: ", initial_bias_constraints)
    print("Total subsets ignored already in bias: ", total_subsets_ignored_already_in_bias)
    print(f"Active Learning Duration: {total_duration:.2f} seconds")
    print("Total queries: ", total_queries)
    print("\nSubset Exploration Statistics:")
    print(f"    Total subset constraints generated: {len(subset_exploration_constraints)}")
    print(f"    Subset constraints satisfied: {subset_stats['satisfied']}")
    print(f"    Subset constraints violated: {subset_stats['violated']}")
    
    print("\nConstraint Tree Statistics:")
    print(f"    Maximum tree depth: {tree_stats['max_depth']}")
    print("    Constraints by level:")
    for level in sorted(tree_stats['constraints_by_level'].keys()):
        total = tree_stats['constraints_by_level'][level]
        accepted = tree_stats['accepted_by_level'].get(level, 0)
        rejected = tree_stats['rejected_by_level'].get(level, 0)
        print(f"        Level {level}: {total} constraints, {accepted} accepted, {rejected} rejected")
        
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    constraints_file = os.path.join(output_dir, f"{experiment}_learned_constraints.txt")
    with open(constraints_file, "w") as f:
        for constraint in learned_constraints:
            f.write(f"{constraint}\n")
    print(f"\nLearned constraints written to {constraints_file}")


    violated_constraints = [c for c in initial_bias_constraints_set if c not in set(learned_constraints)]

    return (
        learned_constraints,
        violated_constraints,
        total_queries,
        total_duration,
        matching_constraints,
        len(target_model_constraints),
        total_subsets_added,
        total_constraints_learned,
        total_constraints_removed,
        initial_bias_constraints,
        total_subsets_ignored_already_in_bias,
        len(subset_exploration_constraints),
        subset_stats['satisfied'],
        subset_stats['violated']
    )


def construct_instance(experiment_name):
    
    if 'sudoku' in experiment_name.lower():
        print("Constructing 9sudoku")
        n = 9
        instance_binary, oracle_binary = construct_sudoku_binary(3, 3, 9)
        result = construct_sudoku(3, 3, 9)

        instance_global, oracle_global = (result[0], result[1]) if len(result) == 3 else result
    elif 'examtt' in experiment_name.lower():
        print("Constructing examtt")
        n = 6
        result1 = construct_examtt_simple(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14)
        instance_binary, oracle_binary = (result1[0], result1[1]) if len(result1) == 3 else result1
        result2 = ces_global(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14)
        instance_global, oracle_global = (result2[0], result2[1]) if len(result2) == 3 else result2
    elif 'nurse' in experiment_name.lower():
        n = 0
        instance_binary, oracle_binary = construct_nurse_rostering()
        instance_global, oracle_global = nr_global()
    elif 'uefa' in experiment_name.lower():
        print("Constructing UEFA problem")
        from benchmarks.uefa import construct_uefa as construct_uefa_instance_binary
        from benchmarks_global.uefa import construct_uefa as construct_uefa_instance_global

        teams_data = {
            "RealMadrid": {"country": "ESP", "coefficient": 134000},
            "BayernMunich": {"country": "GER", "coefficient": 129000},
            "ManchesterCity": {"country": "ENG", "coefficient": 128000},
            "PSG": {"country": "FRA", "coefficient": 112000},
            "Liverpool": {"country": "ENG", "coefficient": 109000},
            "Barcelona": {"country": "ESP", "coefficient": 98000},
            "Juventus": {"country": "ITA", "coefficient": 95000},
            "AtleticoMadrid": {"country": "ESP", "coefficient": 94000},
            "ManchesterUnited": {"country": "ENG", "coefficient": 92000},
            "Chelsea": {"country": "ENG", "coefficient": 91000},
            "BorussiaDortmund": {"country": "GER", "coefficient": 88000},
            "Ajax": {"country": "NED", "coefficient": 82000},
            "RB Leipzig": {"country": "GER", "coefficient": 79000},
            "InterMilan": {"country": "ITA", "coefficient": 76000},
            "Sevilla": {"country": "ESP", "coefficient": 75000},
            "Napoli": {"country": "ITA", "coefficient": 74000},
            "Benfica": {"country": "POR", "coefficient": 73000},
            "Porto": {"country": "POR", "coefficient": 72000},
            "Arsenal": {"country": "ENG", "coefficient": 71000},
            "ACMilan": {"country": "ITA", "coefficient": 70000},
            "RedBullSalzburg": {"country": "AUT", "coefficient": 69000},
            "ShakhtarDonetsk": {"country": "UKR", "coefficient": 68000},
            "BayerLeverkusen": {"country": "GER", "coefficient": 67000},
            "Olympiacos": {"country": "GRE", "coefficient": 66000},
            "Celtic": {"country": "SCO", "coefficient": 65000},
            "Rangers": {"country": "SCO", "coefficient": 64000},
            "PSVEindhoven": {"country": "NED", "coefficient": 63000},
            "SportingCP": {"country": "POR", "coefficient": 62000},
            "Marseille": {"country": "FRA", "coefficient": 61000},
            "ClubBrugge": {"country": "BEL", "coefficient": 60000},
            "Galatasaray": {"country": "TUR", "coefficient": 59000},
            "Feyenoord": {"country": "NED", "coefficient": 58000}
        }

        instance_binary, oracle_binary = construct_uefa_instance_binary(teams_data)
        instance_global, oracle_global = construct_uefa_instance_global(teams_data)
        n = 8 
    elif 'vm_allocation' in experiment_name.lower():
        print("Constructing VM allocation problem")
        from benchmarks.vm_allocation import construct_vm_allocation as construct_vm_allocation_instance_binary
        from benchmarks_global.vm_allocation import construct_vm_allocation as construct_vm_allocation_instance_global
        from vm_allocation_model import PM_DATA, VM_DATA
        
        instance_binary, oracle_binary = construct_vm_allocation_instance_binary(PM_DATA, VM_DATA)
        instance_global, oracle_global = construct_vm_allocation_instance_global(PM_DATA, VM_DATA)
        n = len(instance_binary.X)  
    else:
        throw_error("Unknown experiment name")
        return None, None
    return instance_binary, oracle_binary, instance_global, oracle_global, n

def load_passive_learning_constraints(experiment_name, num_solutions=None, output_dir="output"):

    if num_solutions is not None:
        pickle_filename = os.path.join(output_dir, f"{experiment_name}_{num_solutions}_solutions_learned_constraints.pkl")
        if os.path.exists(pickle_filename):
            try:
                with open(pickle_filename, 'rb') as f:
                    constraint_data = pickle.load(f)
                print(f"Loaded passive learning constraints from: {pickle_filename}")
                print(f"  Learned constraints: {len(constraint_data.get('learned_constraints', []))}")
                print(f"  Bias constraints: {len(constraint_data.get('bias_constraints', []))}")
                print(f"  Invalid constraints: {len(constraint_data.get('invalid_constraints', []))}")
                return constraint_data
            except Exception as e:
                print(f"Error loading constraints from {pickle_filename}: {e}")
                return None
        else:
            print(f"Passive learning constraints file not found: {pickle_filename}")

    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith(f"{experiment_name}_") and filename.endswith("_learned_constraints.pkl"):
                pickle_filepath = os.path.join(output_dir, filename)
                try:
                    with open(pickle_filepath, 'rb') as f:
                        constraint_data = pickle.load(f)
                    print(f"Found and loaded passive learning constraints from: {pickle_filepath}")
                    print(f"  Learned constraints: {len(constraint_data.get('learned_constraints', []))}")
                    print(f"  Bias constraints: {len(constraint_data.get('bias_constraints', []))}")
                    print(f"  Invalid constraints: {len(constraint_data.get('invalid_constraints', []))}")
                    return constraint_data
                except Exception as e:
                    print(f"Error loading constraints from {pickle_filepath}: {e}")
                    continue
    
    print(f"No passive learning constraints found for experiment: {experiment_name}")
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run constraint acquisition with innovative techniques')
    parser.add_argument('--experiment', type=str, default="sudoku", help='Experiment name')
    parser.add_argument('--use_standard_pqgen', action='store_true', help='Use standard PQGen')
    parser.add_argument('--timeout', type=int, default=600, help='Global timeout in seconds (default: 600s/10min)')
    parser.add_argument('--use_bayesian', action='store_true', help='Use Bayesian constraint acquisition')
    parser.add_argument('--theta_max', type=float, default=0.9, help='Threshold to add constraints to CL (default: 0.9)')
    parser.add_argument('--theta_min', type=float, default=0.1, help='Threshold to remove constraints from bias (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Learning rate for Bayesian updates (default: 0.2)')
    parser.add_argument('--prior', type=float, default=0.5, help='Prior probability for constraints (default: 0.5)')
    parser.add_argument('--use_passive_constraints', action='store_true', help='Use constraints learned from passive learning')
    parser.add_argument('--passive_solutions', type=int, default=None, help='Number of solutions used in passive learning (if not specified, uses any available)')
    parser.add_argument('--passive_output_dir', type=str, default="output", help='Directory containing passive learning pickle files')
    parser.add_argument('--max_synthetic_constraints', type=int, default=1, help='Maximum number of synthetic overfitted constraints to generate (default: 1)')
    parser.add_argument('--phase2_only', action='store_true', help='Stop after phase 2 (constraint generation) and return results')
    args = parser.parse_args()
    
    experiment = args.experiment
    use_standard_pqgen = args.use_standard_pqgen
    global_timeout = args.timeout
    use_bayesian = args.use_bayesian
    use_passive_constraints = args.use_passive_constraints
    passive_solutions = args.passive_solutions
    passive_output_dir = args.passive_output_dir
    max_synthetic_constraints = args.max_synthetic_constraints
    phase2_only = args.phase2_only
     
    
    print(f"Running experiment: {experiment}")
    print(f"Global timeout: {global_timeout} seconds")
    print(f"Use passive learning constraints: {use_passive_constraints}")
    print(f"Max synthetic constraints: {max_synthetic_constraints}")
    print(f"Phase 2 only: {phase2_only}")
    
    data_dir = r"data/" + experiment

    instance_binary, oracle_binary, instance, oracle, n = construct_instance(experiment)
    oracle_binary.variables_list = cpm_array(instance_binary.X)
    oracle.variables_list = cpm_array(instance.X)

    global_constraints = oracle.constraints
    biases = []

    passive_constraint_data = None
    if use_passive_constraints:
        passive_constraint_data = load_passive_learning_constraints(
            experiment, passive_solutions, passive_output_dir
        )
        
        if passive_constraint_data:
            print("\n=== Using Passive Learning Constraints ===")
            passive_learned = passive_constraint_data.get('learned_constraints', [])
            passive_bias = passive_constraint_data.get('bias_constraints', [])
            
            print(f"Loaded {len(passive_learned)} learned constraints from passive learning")
            print(f"Loaded {len(passive_bias)} bias constraints from passive learning")
            
            initial_valid_constraints = passive_learned
            initial_bias_constraints = passive_bias
        else:
            print("Failed to load passive learning constraints, proceeding without them")
            initial_valid_constraints = []
            initial_bias_constraints = []
    else:
        print("Not using passive learning constraints")
        initial_valid_constraints = []
        initial_bias_constraints = []

    total_vars = len(instance.X)
    total_constraints = len(global_constraints)
    active_model = Model()
    
    grid_size = total_vars
    
    
    (valid_constraints, invalid_constraints,
         total_violation_queries, violation_time,
         matching_constraints, target_model_size,
         total_subsets_added, total_constraints_learned,
         total_constraints_removed, initial_bias_constraints,
         total_subsets_ignored, total_subset_constraints_generated,
         subset_satisfied, subset_violated) = active_learning_system(
        experiment, global_constraints, active_model, instance.X, grid_size, oracle, None,
        use_fuzzy_rl=False, use_rl_bayesian=False, use_standard_pqgen=use_standard_pqgen, global_timeout=global_timeout,
        initial_valid_constraints=initial_valid_constraints,
        initial_bias_constraints=initial_bias_constraints,
        max_synthetic_constraints=max_synthetic_constraints,
        phase2_only=phase2_only
    )
    
    print(f"Valid constraints after active learning: {len(valid_constraints)}")
    print(f"Invalid constraints after active learning: {len(invalid_constraints)}")
    print("Raw valid constraints:")
    for c in valid_constraints:
        print(f"  - {c}")

    if phase2_only:
        print("Exiting after phase 2 (constraint generation) as requested")
        sys.exit(0)
    
    if use_passive_constraints and passive_constraint_data:
        passive_learned = passive_constraint_data.get('learned_constraints', [])
        print(f"\n=== Combining Passive and Active Learning Results ===")

        
        filtered_passive_constraints = []
        for constraint in passive_learned:
            if constraint is not None:
                constraint_str = str(constraint).lower()
                is_simple_assignment = (
                    " == " in str(constraint) and 
                    "sum" not in constraint_str and 
                    "alldifferent" not in constraint_str and 
                    "count" not in constraint_str and
                    not any(op in str(constraint) for op in [" >= ", " <= ", " != ", " > ", " < "])
                )
                
                if not is_simple_assignment:
                    filtered_passive_constraints.append(constraint)
        
        
        combined_constraints = list(set(filtered_passive_constraints + valid_constraints))
        
        final_valid_constraints = combined_constraints
    else:
        final_valid_constraints = valid_constraints


    constraints_decomposed = []
    for c in final_valid_constraints:
        if c.name == "alldifferent":
            c_decomposed = c.decompose()[0]
            constraints_decomposed.extend(c_decomposed)


    for c in oracle_binary.constraints:
        if c not in set(constraints_decomposed) and c not in biases and "AllDifferent" not in str(c) and "Sum" not in str(c) and "Count" not in str(c):
            biases.append(c)

    print(f"Biases: {len(biases)}")
    instance.bias = biases
    instance.cl = constraints_decomposed
    instance.cl.append(final_valid_constraints)

    ca_system = MQuAcq2()


    learned_instance = ca_system.learn(instance, oracle=oracle_binary, verbose=3)

    final_constraints = learned_instance.get_cpmpy_model().constraints
    final_model = learned_instance.get_cpmpy_model()
    print(ca_system.env.metrics.short_statistics)
    print(f"Total Violation Queries: {total_violation_queries} Queries - Time: {violation_time:.2f} seconds")
    print(f"Total MQuAcq2 Queries: {ca_system.env.metrics.total_queries} Queries - Time: {ca_system.env.metrics.total_time:.2f} seconds")
    print(f"Total Queries: {total_violation_queries + ca_system.env.metrics.total_queries} Queries - Time: {violation_time + ca_system.env.metrics.total_time:.2f} seconds")
    print(f"Total Invalid Constraints: {len(invalid_constraints)} - Total Starting Constraints: {len(global_constraints)}")
    
    system_type = "Bayesian"
    if use_passive_constraints:
        system_type += "_with_Passive"
    
    results = {
        'problem_name': experiment,
        'system_type': system_type,
        'num_starting_constraints': len(global_constraints),
        'num_final_constraints': len(final_valid_constraints),
        'num_invalid_constraints': len(invalid_constraints),
        'total_violation_queries': total_violation_queries,
        'cl': len(constraints_decomposed),
        'bias': len(biases),
        'violation_time_seconds': round(violation_time,2),
        'total_MQuAcq2_queries': ca_system.env.metrics.total_queries,
        'MQuAcq2_time_seconds': round(ca_system.env.metrics.total_time,2),
        'total_queries': total_violation_queries + ca_system.env.metrics.total_queries,
        'total_time_seconds': round(violation_time + ca_system.env.metrics.total_time),
        'used_passive_constraints': use_passive_constraints,
        'passive_constraints_loaded': len(passive_constraint_data.get('learned_constraints', [])) if passive_constraint_data else 0
    }
    results_df = pd.DataFrame([results])
    csv_file = 'results.csv'
    if os.path.exists(csv_file):
        results_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_file, mode='w', header=True, index=False)

