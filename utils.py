import math
import shutil
import time
from cpmpy import intvar, boolvar, Model, all, sum
from sklearn.utils import class_weight
import numpy as np
import cpmpy
import re
from cpmpy.expressions.utils import all_pairs
from itertools import chain, combinations
import networkx as nx
from collections import defaultdict
import json
import matplotlib.pyplot as plt

import networkx as nx
import re


def build_instance_mapping(variables):
    """
    Build a mapping from variable names to CPMpy variables.
    Also infers the grid dimensions from the variable names.
    Handles VM allocation specific variables including:
    - assign_VM_X: VM assignment variables (1-5)
    - active_PM_X: PM activation variables (0-1)
    """
    import cpmpy as cp
    mapping = {}
    
    # First pass: find maximum VM and PM numbers from both variables and constraints
    max_vm = 0
    max_pm = 0
    
    # Check existing variables
    for var in variables:
        var_name = var.name
        if var_name.startswith('assign_VM_'):
            try:
                vm_num = int(var_name[len('assign_VM_'):])
                max_vm = max(max_vm, vm_num)
            except ValueError:
                continue
        elif var_name.startswith('active_PM_'):
            try:
                pm_num = int(var_name[len('active_PM_'):])
                max_pm = max(max_pm, pm_num)
            except ValueError:
                continue
    
    # Check learned constraints file for any additional variables
    try:
        with open('output/vm_allocation/learned_global_constraints.txt', 'r') as f:
            for line in f:
                # Look for VM assignments
                vm_matches = re.findall(r'assign_VM_(\d+)', line)
                for vm_num in vm_matches:
                    max_vm = max(max_vm, int(vm_num))
                
                # Look for PM activations
                pm_matches = re.findall(r'active_PM_(\d+)', line)
                for pm_num in pm_matches:
                    max_pm = max(max_pm, int(pm_num))
    except FileNotFoundError:
        pass  # File might not exist yet

    # Also check bias constraints file
    try:
        with open('output/vm_allocation/bias_constraints.txt', 'r') as f:
            for line in f:
                # Look for VM assignments
                vm_matches = re.findall(r'assign_VM_(\d+)', line)
                for vm_num in vm_matches:
                    max_vm = max(max_vm, int(vm_num))
                
                # Look for PM activations
                pm_matches = re.findall(r'active_PM_(\d+)', line)
                for pm_num in pm_matches:
                    max_pm = max(max_pm, int(pm_num))
    except FileNotFoundError:
        pass  # File might not exist yet
    
    # Ensure we have at least some minimum values
    max_vm = max(max_vm, 12)  # Based on the constraints we've seen
    max_pm = max(max_pm, 8)   # Based on the constraints we've seen
    
    # Second pass: create all VM assignment variables
    for i in range(1, max_vm + 1):
        var_name = f'assign_VM_{i}'
        if var_name not in mapping:
            var = cp.intvar(1, 5, name=var_name)  # VMs can be assigned to PMs 1-5
            mapping[var_name] = var
    
    # Create all PM activation variables
    for i in range(1, max_pm + 1):
        var_name = f'active_PM_{i}'
        if var_name not in mapping:
            var = cp.intvar(0, 1, name=var_name)  # PM active status is binary
            mapping[var_name] = var
    
    # Third pass: add any remaining variables from the input
    for var in variables:
        var_name = var.name
        if var_name not in mapping:
            mapping[var_name] = var
    
    # Grid dimensions are the maximum VM and PM numbers
    grid_dims = (max_vm, max_pm)
    return mapping, grid_dims

def parse_learned_constraint(constraint_str, instance_mapping=None):
    """Parse a learned constraint string into a CPMpy constraint."""
    import cpmpy as cp
    from cpmpy.expressions.variables import _IntVarImpl
    from cpmpy.expressions.globalconstraints import AllDifferent

    if instance_mapping is None:
        instance_mapping = {}

    def get_var(var_name):
        """Get or create a variable with appropriate bounds."""
        # First check if variable exists in mapping
        if var_name in instance_mapping:
            return instance_mapping[var_name]
        
        # For VM allocation, create variables with appropriate bounds
        if var_name.startswith('assign_VM_'):
            try:
                vm_num = int(var_name[len('assign_VM_'):])
                var = cp.intvar(1, 5, name=var_name)  # VMs can be assigned to PMs 1-5
                instance_mapping[var_name] = var
                return var
            except ValueError as e:
                print(f"Error parsing VM number from {var_name}: {e}")
                return None
        elif var_name.startswith('active_PM_'):
            try:
                pm_num = int(var_name[len('active_PM_'):])
                var = cp.intvar(0, 1, name=var_name)  # PM active status is binary
                instance_mapping[var_name] = var
                return var
            except ValueError as e:
                print(f"Error parsing PM number from {var_name}: {e}")
                return None
        
        # If we get here, it's an unknown variable type
        print(f"Warning: Creating default variable for unknown type: {var_name}")
        var = cp.intvar(0, 100, name=var_name)  # Default case
        instance_mapping[var_name] = var
        return var

    def extract_inside_sum(sum_str):
        """Extract the contents within Sum(...) call, ignoring any trailing comparison operators."""
        start_idx = sum_str.find('Sum(') + 4
        paren_count = 1
        end_idx = start_idx
        
        while paren_count > 0 and end_idx < len(sum_str):
            if sum_str[end_idx] == '(':
                    paren_count += 1
            elif sum_str[end_idx] == ')':
                    paren_count -= 1
            end_idx += 1
            
        return sum_str[start_idx:end_idx-1].strip()

    def parse_sum_expression(sum_str):
        """Parse a sum expression and return a weighted sum constraint."""
        if not sum_str:
            return None
        
        # Initialize variables
        terms = []
        current_term = ""
        paren_count = 0
        
        try:
            # Extract terms from the sum expression
            terms_str = sum_str.strip('sum([]).').strip()
            
            # Split terms while respecting nested parentheses
            for char in terms_str:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    if current_term.strip():
                        terms.append(current_term.strip())
                    current_term = ""
                else:
                    current_term += char
            
            if current_term.strip():
                terms.append(current_term.strip())

            coefficients = []
            conditions = []
            
            for term in terms:
                term = term.strip('() ')
                if '*' in term:
                    # Handle coefficient * variable
                    coeff_part, cond_part = term.split('*', 1)
                    
                    # Handle coefficient list
                    if coeff_part.startswith('[') and coeff_part.endswith(']'):
                        coefficients.extend([int(x.strip()) for x in coeff_part[1:-1].split(',')])
                    else:
                        coefficients.append(int(coeff_part.strip('() ')))
                    
                    # Handle condition list
                    if cond_part.startswith('[') and cond_part.endswith(']'):
                        conds = cond_part[1:-1].split(',')
                        for cond in conds:
                            var_name, value = [x.strip() for x in cond.split('==')]
                            var = get_var(var_name)
                            if var is not None:
                                value = int(value)
                                conditions.append(var == value)
                    else:
                        var_name = cond_part.split('==')[0].strip()
                        value = int(cond_part.split('==')[1].strip())
                        var = get_var(var_name)
                        if var is not None:
                            conditions.append(var == value)
                else:
                    # Handle simple condition
                    if '==' in term:
                        var_name, value = [x.strip() for x in term.split('==')]
                        var = get_var(var_name)
                        if var is not None:
                            value = int(value)
                            conditions.append(var == value)
                            coefficients.append(1)  # Default coefficient is 1

            if len(coefficients) == len(conditions):
                # Create weighted sum with initialized values
                return cp.wsum(coefficients, conditions)
            print(f"Warning: Mismatch in coefficients ({len(coefficients)}) and conditions ({len(conditions)})")
            return None
            
        except Exception as e:
            print(f"Error parsing sum expression: {e}")
            return None

    def extract_comparison(expr_str):
        """Extract comparison operator and bound from a string."""
        if '<=' in expr_str:
            parts = expr_str.split('<=')
            return '<=', int(parts[1].strip('() '))
        elif '>=' in expr_str:
            parts = expr_str.split('>=')
            return '>=', int(parts[1].strip('() '))
        elif '==' in expr_str:
            parts = expr_str.split('==')
            return '==', int(parts[1].strip('() '))
        elif '>' in expr_str:
            parts = expr_str.split('>')
            return '>', int(parts[1].strip('() '))
        return None, None

    # Handle AllDifferent constraints
    if constraint_str.startswith('AllDifferent('):
        vars_str = constraint_str[len('AllDifferent('):-1]
        var_names = [name.strip() for name in vars_str.split(',')]
        vars_list = []
        for name in var_names:
            var = get_var(name)
            if var is not None:
                vars_list.append(var)
        if vars_list:
            return ('ALLDIFFERENT', AllDifferent(vars_list), vars_list)

    # Handle Count constraints
    if constraint_str.startswith('Count('):
        count_expr = constraint_str[len('Count('):constraint_str.rindex(')')]
        conditions = [cond.strip('() ') for cond in count_expr.split(',')]
        vars_list = []
        count_conditions = []
        
        for cond in conditions:
            if '==' in cond:
                var_name, value = [x.strip() for x in cond.split('==')]
                var = get_var(var_name)
                if var is not None:
                    value = int(value)
                    vars_list.append(var)
                    count_conditions.append(var == value)
        
        if count_conditions:
            # Get the comparison operator and bound from the remaining part
            remaining = constraint_str[constraint_str.rindex(')')+1:].strip()
            op, bound = extract_comparison(remaining)
            
            if bound is not None:
                if op == '<=':
                    return ('COUNT', cp.sum(count_conditions) <= bound, vars_list)
                elif op == '>=':
                    return ('COUNT', cp.sum(count_conditions) >= bound, vars_list)
                elif op == '==':
                    return ('COUNT', cp.sum(count_conditions) == bound, vars_list)

    # Handle active PM constraints with Sum
    if constraint_str.startswith('active_PM_'):
        parts = constraint_str.split('==', 1)
        if len(parts) == 2:
            pm_var_name = parts[0].strip()
            rest = parts[1].strip('() ')
            pm_var = get_var(pm_var_name)
            
            if pm_var is not None and rest.startswith('(Sum('):
                sum_expr = parse_sum_expression(rest)
                if sum_expr is not None:
                    # Extract comparison operator and bound
                    remaining = rest[rest.rindex(')')+1:].strip('() ')
                    op, bound = extract_comparison(remaining)
                    
                    if op == '>':
                        return ('ACTIVE_PM', pm_var == (sum_expr > bound), [pm_var])
                    elif op == '<':
                        return ('ACTIVE_PM', pm_var == (sum_expr < bound), [pm_var])

    # Handle implications (=>)
    if '=>' in constraint_str:
        antecedent_str, consequent_str = [x.strip() for x in constraint_str.split('=>')]
        
        # Parse antecedent
        antecedent_str = antecedent_str.strip('() ')
        antecedent_var = None
        if '==' in antecedent_str:
            var_name, value = [x.strip() for x in antecedent_str.split('==')]
            antecedent_var = get_var(var_name)
            if antecedent_var is not None:
                antecedent_value = int(value)
                antecedent = (antecedent_var == antecedent_value)
        else:
            antecedent_var = get_var(antecedent_str)
            if antecedent_var is not None:
                antecedent = antecedent_var
        
        # Parse consequent if we have a valid antecedent
        if antecedent_var is not None and consequent_str.startswith('(Sum('):
            sum_expr = parse_sum_expression(consequent_str)
            if sum_expr is not None:
                op, bound = extract_comparison(consequent_str)
                
                if bound is not None:
                    if op == '>=':
                        consequent = (sum_expr >= bound)
                    elif op == '<=':
                        consequent = (sum_expr <= bound)
                    return ('IMPLICATION', cp.any([~antecedent, consequent]), [antecedent_var])

    # Handle Sum constraints
    if constraint_str.startswith('Sum('):
        try:
            sum_expr = parse_sum_expression(constraint_str)
            if sum_expr is not None:
                op, bound = extract_comparison(constraint_str)
                
                if bound is not None:
                    if op == '<=':
                        return ('SUM', sum_expr <= bound, [])
                    elif op == '>=':
                        return ('SUM', sum_expr >= bound, [])
                    elif op == '==':
                        return ('SUM', sum_expr == bound, [])
        except Exception as e:
            print(f"Error parsing sum constraint: {str(e)}")

    return None

def save_valid_constraints(global_constraints, invalid_constraints, output_file_path):
    """
    Saves all valid global constraints to a new model file, excluding the invalid constraints.

    Parameters:
    - global_constraints (list): List of tuples representing all global constraints.
    - invalid_constraints (list): List of tuples representing invalid constraints.
    - output_file_path (str): Path to the output model file.
    """
    with open(output_file_path, 'w') as f:
        for gc in global_constraints:
            # Create a key tuple for comparison
            key = tuple(gc[:2] + (tuple(var.name for var in gc[2]),))

            # Skip invalid constraints
            if key in invalid_constraints:
                continue

            constraint_type = gc[0]
            vars_in_scope = [var.name for var in gc[2]]

            if constraint_type == 'ALLDIFFERENT':
                # Format: ALLDIFFERENT\tvar0 var1 var2 var3
                line = f"ALLDIFFERENT\t{' '.join(vars_in_scope)}\n"
            elif constraint_type == 'SUM':
                # Format: SUM\tvar0 var1 var2 var3\t20
                # Assuming the target sum is stored in gc[3]
                if len(gc) >= 4:
                    total_sum = gc[3]
                    line = f"SUM\t{' '.join(vars_in_scope)}\t{total_sum}\n"
                else:
                    continue  # Skip if target sum is missing
            elif constraint_type == 'COUNT':
                # Format: COUNT\tvar0 var1 var2\t5\t3
                # Assuming count value and count times are stored in gc[3] and gc[4]
                if len(gc) >= 5:
                    count_value = gc[3]
                    count_times = gc[4]
                    line = f"COUNT\t{' '.join(vars_in_scope)}\t{count_value}\t{count_times}\n"
                else:
                    continue  # Skip if count details are missing
            elif constraint_type == 'ALTERNATING':
                # Format: ALTERNATING\tvar0 var1 var2
                line = f"ALTERNATING\t{' '.join(vars_in_scope)}\n"
            elif constraint_type == 'MINDAYSBETWEEN':
                # Format: MINDAYSBETWEEN\tvar0 var1 var2\t3
                if len(gc) >= 4:
                    days = gc[3]
                    line = f"MINDAYSBETWEEN\t{' '.join(vars_in_scope)}\t{days}\n"
                else:
                    continue  # Skip if days is missing
            else:
                continue  # Skip unknown constraint types

            f.write(line)


def visualize_dependency_graph(dependency_graph, output_path='constraint_dependency_graph.png'):
    """
    Visualizes the constraint dependency graph and saves it to a file.

    Args:
        dependency_graph (networkx.Graph): The constraint dependency graph.
        output_path (str): Path to save the visualization image.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(dependency_graph, k=0.15, iterations=20)
    nx.draw(dependency_graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold')
    plt.title("Constraint Dependency Graph")
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_constraint_dependency_graph(global_constraints, variables):
    """
    Builds a constraint dependency graph where nodes are constraints and edges indicate shared variables.

    Args:
        global_constraints (list): List of global constraints.
        variables (list): List of variables in the CP model.

    Returns:
        networkx.Graph: The constraint dependency graph.
    """
    G = nx.Graph()

    # Add nodes
    for idx, constraint_info in enumerate(global_constraints):
        constraint_type = constraint_info[0]
        vars_in_scope = constraint_info[2]
        key = tuple(constraint_info[:2] + (tuple(var.name for var in vars_in_scope),))
        G.add_node(key, type=constraint_type)

    # Add edges between constraints that share at least one variable
    for (c1, c2) in combinations(global_constraints, 2):
        key1 = tuple(c1[:2] + (tuple(var.name for var in c1[2]),))
        key2 = tuple(c2[:2] + (tuple(var.name for var in c2[2]),))
        vars1 = set(var.name for var in c1[2])
        vars2 = set(var.name for var in c2[2])
        if vars1.intersection(vars2):
            G.add_edge(key1, key2)

    return G

def compute_centrality_measures(dependency_graph):
    """
    Computes centrality measures for constraints in the dependency graph.

    Args:
        dependency_graph (networkx.Graph): The constraint dependency graph.

    Returns:
        dict: Dictionary mapping constraint keys to their centrality scores.
    """
    degree_centrality = nx.degree_centrality(dependency_graph)
    betweenness_centrality = nx.betweenness_centrality(dependency_graph)
    closeness_centrality = nx.closeness_centrality(dependency_graph)

    # Combine centrality measures (weighted sum)
    combined_centrality = {}
    for node in dependency_graph.nodes():
        combined_centrality[node] = (
            degree_centrality.get(node, 0) * 0.5 +
            betweenness_centrality.get(node, 0) * 0.3 +
            closeness_centrality.get(node, 0) * 0.2
        )

    return combined_centrality

def append_missing_vars(model_file, vars_file, default_value=0, backup=False):
    """
    Appends missing variables to the _vars file based on the _model file.

    Args:
        model_file (str): Path to the _model file.
        vars_file (str): Path to the _vars file.
        default_value (int, optional): Default value for new variables. Defaults to 0.
        backup (bool, optional): Whether to create a backup of the _vars file before modifying. Defaults to False.
    """
    # Step 1: Extract variable indices from the _model file
    model_vars = parse_model_file_vars(model_file)
    print(f"Variables in _model file: {sorted(model_vars)}")

    # Step 2: Extract existing variable indices from the _vars file
    existing_vars, total_vars = parse_vars_file_existing(vars_file)
    print(f"Existing variables in _vars file: {sorted(existing_vars)}")
    print(f"Total variables before update: {total_vars}")

    # Step 3: Identify missing variables
    missing_vars = model_vars - existing_vars
    if not missing_vars:
        #print("No missing variables found. The _vars file is up to date.")
        return
    print(f"Missing variables to append: {sorted(missing_vars)}")

    # Optional: Backup the _vars file
    if backup:
        backup_file = vars_file + ".bak"
        shutil.copy(vars_file, backup_file)
        print(f"Backup created at {backup_file}")

    # Step 4: Append missing variables to the _vars file
    with open(vars_file, 'a') as file:
        for var in sorted(missing_vars):
            file.write(f"{var} {default_value}\n")
            print(f"Appended variable {var} with default value {default_value}.")

    # Step 5: Update the total variable count at the top of the _vars file
    new_total_vars = total_vars + len(missing_vars)
    # Read the current content
    with open(vars_file, 'r') as file:
        lines = file.readlines()
    # Update the first line with the new total
    lines[0] = f"{new_total_vars}\n"
    # Write back the updated content
    with open(vars_file, 'w') as file:
        file.writelines(lines)
    print(f"Updated total variables in _vars file to: {new_total_vars}")


def parse_model_file_vars(model_file_path):
    """
    Parses the _model file to extract all unique variable indices.

    Args:
        model_file_path (str): Path to the _model file.

    Returns:
        set: A set of unique variable indices used in the model.
    """
    variable_indices = set()
    with open(model_file_path, 'r') as file:
        for line in file:
            # Extract all numbers within square brackets
            indices = re.findall(r'\[(.*?)\]', line)
            for group in indices:
                nums = re.findall(r'\d+', group)
                variable_indices.update(map(int, nums))
    return variable_indices


def parse_vars_file_existing(vars_file_path):
    """
    Parses the _vars file to extract existing variable indices.

    Args:
        vars_file_path (str): Path to the _vars file.

    Returns:
        tuple: A set of existing variable indices and the total number of variables.
    """
    existing_vars = set()
    with open(vars_file_path, 'r') as file:
        total_vars = int(file.readline().strip())
        for line in file:
            parts = line.strip().split()
            if parts:
                var_index = int(parts[0])
                existing_vars.add(var_index)
    return existing_vars, total_vars


def backup_file(original_path, backup_path):
    """
    Creates a backup of the original file.

    Args:
        original_path (str): Path to the original file.
        backup_path (str): Path to the backup file.
    """
    shutil.copy(original_path, backup_path)
    print(f"Backup created at {backup_path}")

def is_full_row(vars_in_scope):
    """
    Checks if the variables in vars_in_scope constitute a full row in a Sudoku grid.
    """
    indices = [int(var.name[3:]) for var in vars_in_scope]
    n = 9  # Grid size
    if len(indices) != n:
        return False
    rows = [idx // n for idx in indices]
    # All variables must be in the same row
    if len(set(rows)) != 1:
        return False
    cols = [idx % n for idx in indices]
    # Columns must cover all positions in the row
    if set(cols) == set(range(n)):
        return True
    return False

def is_full_column(vars_in_scope):
    """
    Checks if the variables in vars_in_scope constitute a full column in a Sudoku grid.
    """
    indices = [int(var.name[3:]) for var in vars_in_scope]
    n = 9  # Grid size
    if len(indices) != n:
        return False
    cols = [idx % n for idx in indices]
    # All variables must be in the same column
    if len(set(cols)) != 1:
        return False
    rows = [idx // n for idx in indices]
    # Rows must cover all positions in the column
    if set(rows) == set(range(n)):
        return True
    return False

import re

def parse_var_name_to_1d(var_name: str, cols: int = 9) -> int:
    """Convert a variable name into a 1D index.
    
    Handles the following variable patterns:
    - var[row,col] -> row * cols + col
    - var17 -> 17
    - group_TeamName -> index in sorted team list
    - match_day_N -> N
    - assign_VM_N -> N
    - active_PM_N -> N
    - resource_PM_N -> N (where resource is cpu/memory/disk)
    - home_R_G_M or away_R_G_M -> unique index based on round, group, match
    """
    # Try standard var[row,col] pattern
    match_2d = re.match(r"var\[(\d+),(\d+)\]", var_name)
    if match_2d:
        row = int(match_2d.group(1))
        col = int(match_2d.group(2))
        return row * cols + col

    # Try var17 pattern
    match_1d = re.match(r"var(\d+)", var_name)
    if match_1d:
        return int(match_1d.group(1))

    # Try UEFA match pattern (home_5_0_1 or away_5_0_1)
    match_game = re.match(r"(home|away)_(\d+)_(\d+)_(\d+)", var_name)
    if match_game:
        is_home = match_game.group(1) == "home"
        round_num = int(match_game.group(2))
        group_num = int(match_game.group(3))
        match_num = int(match_game.group(4))
        # Calculate unique index for match variables
        # Each round has 8 groups, each group has 2 matches (home/away)
        base_idx = 3000  # Start after other indices
        matches_per_round = 8 * 2  # 8 groups * 2 matches per group
        round_offset = round_num * matches_per_round
        group_offset = group_num * 2
        match_idx = match_num
        total_idx = base_idx + round_offset + group_offset + match_idx
        return total_idx

    # Try UEFA group assignment pattern
    if var_name.startswith("group_"):
        # We'll use the team name as is, since it will be mapped to an index elsewhere
        return var_name

    # Try UEFA match day pattern
    match_day = re.match(r"match_day_(\d+)", var_name)
    if match_day:
        return int(match_day.group(1))

    # Try VM allocation patterns
    vm_assign = re.match(r"assign_VM_(\d+)", var_name)
    if vm_assign:
        return int(vm_assign.group(1))

    pm_active = re.match(r"active_PM_(\d+)", var_name)
    if pm_active:
        return int(pm_active.group(1))

    resource = re.match(r"(cpu|memory|disk)_PM_(\d+)", var_name)
    if resource:
        pm_num = int(resource.group(2))
        resource_type = resource.group(1)
        # Map different resources to different ranges
        resource_offset = {"cpu": 0, "memory": 1000, "disk": 2000}
        return resource_offset[resource_type] + pm_num

    # For any other patterns, return the name as is
    # This allows the calling code to handle special cases
    return var_name

def constraint_to_vector(constraint_info, total_vars, cols=9):
    """
    Create a binary vector of length `total_vars`.
    Each variable in `vars_in_scope` is mapped to 1 in the appropriate index,
    and everything else is 0.
    """
    # Extract variables based on input type
    if hasattr(constraint_info, '_variables'):
        # For CPMpy constraints
        vars_in_scope = list(constraint_info._variables)
    elif isinstance(constraint_info, tuple) and len(constraint_info) >= 3:
        # For tuple format (type, constraint, variables)
        vars_in_scope = constraint_info[2]
    else:
        #print(f"Warning: Unsupported constraint type for vectorization: {type(constraint_info)}")
        return [0] * total_vars

    vector = [0] * total_vars
    
    # Map variables to indices
    for var in vars_in_scope:
        try:
            # Get variable name
            var_name = var.name if hasattr(var, 'name') else str(var)
            
            # Extract numeric part for VM allocation variables
            if var_name.startswith('assign_'):
                # Extract VM number from assign_VM_X
                vm_match = re.search(r'assign_VM_(\d+)', var_name)
                if vm_match:
                    idx = int(vm_match.group(1)) - 1
                else:
                    # For assign_X format
                    vm_num = var_name[len('assign_'):]
                    idx = int(vm_num) - 1
            elif var_name.startswith('active_'):
                # Extract PM number from active_PM_X
                pm_match = re.search(r'active_PM_(\d+)', var_name)
                if pm_match:
                    idx = int(pm_match.group(1)) - 1 + total_vars // 2  # Place active vars in second half
                else:
                    # For active_X format
                    pm_num = var_name[len('active_'):]
                    idx = int(pm_num) - 1 + total_vars // 2
            else:
                # Use parse_var_name_to_1d for other variables
                idx = parse_var_name_to_1d(var_name, cols=cols)
                if isinstance(idx, str):
                    print(f"Warning: Could not convert {var_name} to valid index")
                    continue
            
            if 0 <= idx < total_vars:
                vector[idx] = 1
            else:
                print(f"Warning: Index {idx} out of bounds for variable {var_name}")
        except Exception as e:
            print(f"Warning: Failed to process variable {var}: {e}")
            continue
    
    return vector

def is_full_block(vars_in_scope, n=9):
    """
    Checks if the variables in vars_in_scope constitute a full 3x3 block
    in a standard 9x9 Sudoku grid.

    :param vars_in_scope: A list of CPMPy variable objects with names like "var[4,0]" or "var17".
    :param n: Number of columns/rows in the Sudoku (default=9).
    :return: True if they form exactly one full 3x3 block, else False.
    """
    # Convert each var to its 1D index
    # (row * n + col) for 2D, or just the integer part if "var17"
    indices = []
    for var in vars_in_scope:
        idx = parse_var_name_to_1d(var.name, n)
        indices.append(idx)

    # Must have exactly n=9 variables to be a single block
    if len(indices) != n:
        return False

    # Compute each var's row and col in a 9×9
    rows = [idx // n for idx in indices]
    cols = [idx % n  for idx in indices]

    # All these variables must be in the same 3×3 region
    # (i.e. block_rows and block_cols are all identical)
    block_rows = [r // 3 for r in rows]  # integer division by 3
    block_cols = [c // 3 for c in cols]

    if len(set(block_rows)) != 1 or len(set(block_cols)) != 1:
        return False

    # Then verify they actually fill that entire 3×3 region
    br = block_rows[0]  # which block row?
    bc = block_cols[0]  # which block col?

    expected_indices = [
        r * n + c
        for r in range(br * 3, br * 3 + 3)
        for c in range(bc * 3, bc * 3 + 3)
    ]
    # If we've exactly matched those 9 positions => True
    return set(indices) == set(expected_indices)

def is_diagonal(vars_in_scope, n=9):
    """
    Checks if the variables in vars_in_scope constitute a diagonal in a Sudoku grid.
    """
    indices = [int(var.name[3:]) for var in vars_in_scope]

    if len(indices) != n:
        return False
    main_diagonal = [i * n + i for i in range(n)]
    anti_diagonal = [i * n + (n - 1 - i) for i in range(n)]
    if set(indices) == set(main_diagonal) or set(indices) == set(anti_diagonal):
        return True
    return False

import re


def parse_dom_file(file_path):
    domain_constraints = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 3:
                var_index = int(parts[0])
                lower_bound = int(parts[2])
                upper_bound = int(parts[-1])
                domain_constraints[var_index] = (lower_bound, upper_bound)
    return domain_constraints

def parse_con_file(file_path):
    biases = []

    with open(file_path, 'r') as file:
        for line in file:
            con_type, var1, var2 = map(int, line.strip().split())
            biases.append((con_type, var1, var2))

    return biases


def constraint_type_to_string(con_type):
    return {
        0: "!=",
        1: "==",
        2: ">",
        3: "<",
        4: ">=",
        5: "<="
    }.get(con_type, "Unknown")


def parse_vars_file(file_path):
    with open(file_path, 'r') as file:
        total_vars = int(file.readline().strip())
        vars_values = [0] * total_vars

        for i, line in enumerate(file):
            value, _ = map(int, line.split())
            vars_values[i] = value

    return vars_values


def parse_model_file(file_path):
    max_index = -1
    constraints = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            constraint_type, indices_part = parts[0], parts[1]
            indices = re.findall(r'\d+', indices_part)
            indices = [int(i) for i in indices]

            max_index_in_line = max(indices)
            if max_index_in_line > max_index:
                max_index = max_index_in_line

            if constraint_type == 'ALLDIFFERENT':
                constraints.append((constraint_type, indices))

    return constraints, max_index


def are_comparisons_equal(comp1, comp2):
    """
    Checks if two Comparison objects are equal.

    :param comp1: The first Comparison object.
    :param comp2: The second Comparison object.
    :return: True if the Comparisons are equal, False otherwise.
    """
    if comp1.name != comp2.name:
        return False

    if comp1.args[0] != comp2.args[0]:
        return False

    if comp1.args[1] != comp2.args[1]:
        return False

    return True


def check_value(c):
    return bool(c.value())


def get_con_subset(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y)]


def get_kappa(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is False]


def get_lambda(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is True]


def gen_pairwise(v1, v2):
    return [v1 == v2, v1 != v2, v1 < v2, v1 > v2]


# to create the binary oracle
def gen_pairwise_ineq(v1, v2):
    return [v1 != v2]


def alldiff_binary(grid):
    for v1, v2 in all_pairs(grid):
        for c in gen_pairwise_ineq(v1, v2):
            yield c


def gen_scoped_cons(grid):
    # rows
    for row in grid:
        for v1, v2 in all_pairs(row):
            for c in gen_pairwise_ineq(v1, v2):
                yield c
    # columns
    for col in grid.T:
        for v1, v2 in all_pairs(col):
            for c in gen_pairwise_ineq(v1, v2):
                yield c

    # DT: Some constraints are not added here, I will check it and fix  TODO
    # subsquares
    for i1 in range(0, 4, 2):
        for i2 in range(i1, i1 + 2):
            for j1 in range(0, 4, 2):
                for j2 in range(j1, j1 + 2):
                    if (i1 != i2 or j1 != j2):
                        for c in gen_pairwise_ineq(grid[i1, j1], grid[i2, j2]):
                            yield c


def gen_all_cons(grid):
    # all pairs...
    for v1, v2 in all_pairs(grid.flat):
        for c in gen_pairwise(v1, v2):
            yield c


def construct_bias(X, gamma):
    all_cons = []

    X = list(X)

    for relation in gamma:

        if relation.count("var") == 2:

            for v1, v2 in all_pairs(X):
                constraint = relation.replace("var1", "v1")
                constraint = constraint.replace("var2", "v2")
                constraint = eval(constraint)

                all_cons.append(constraint)

        elif relation.count("var") == 4:

            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    for x in range(j + 1, len(X) - 1):
                        for y in range(x + 1, len(X)):
                            if (y != i and x != j and x != i and y != j):
                                #            for v1, v2 in all_pairs(X):
                                #                for v3, v4 in all_pairs(X):
                                constraint = relation.replace("var1", "X[i]")
                                constraint = relation.replace("var2", "X[j]")
                                constraint = relation.replace("var3", "X[x]")
                                constraint = relation.replace("var4", "X[y]")
                                constraint = eval(constraint)

                                all_cons.append(constraint)

    return all_cons


def construct_bias_for_var(X, gamma, v1):
    all_cons = []

    for relation in gamma:
        if relation.count("var") == 2:
            for v2 in X:
                if not (v1 is v2):
                    constraint = relation.replace("var1", "v1")
                    constraint = constraint.replace("var2", "v2")
                    constraint = eval(constraint)

                    all_cons.append(constraint)

        elif relation.count("var") == 4:
            X = X.copy()
            X.reverse()
            print(X)
            for j in range(0, len(X)):
                for x in range(j + 1, len(X) - 1):
                    for y in range(x + 1, len(X)):
                        # if (y != i and x != j and x != i and y != j):
                        #            for v1, v2 in all_pairs(X):
                        #                for v3, v4 in all_pairs(X):
                        constraint = relation.replace("var1", "v1")
                        constraint = relation.replace("var2", "X[j]")
                        constraint = relation.replace("var3", "X[x]")
                        constraint = relation.replace("var4", "X[y]")
                        constraint = eval(constraint)

                        all_cons.append(constraint)

    return all_cons


def get_scopes_vars(C):
    return set([x for scope in [get_scope(c) for c in C] for x in scope])


def get_scopes(C):
    return list(set([tuple(get_scope(c)) for c in C]))


def get_scope(constraint):
    # this code is much more dangerous/too few cases then get_variables()
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                # non-recursive shortcut
                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


def get_arity(constraint):
    return len(get_scope(constraint))


def get_min_arity(C):
    if len(C) > 0:
        return min([get_arity(c) for c in C])
    return 0


def get_max_arity(C):
    if len(C) > 0:
        return max([get_arity(c) for c in C])
    return 0


def get_relation(c, gamma):
    scope = get_scope(c)

    for i in range(len(gamma)):
        relation = gamma[i]

        if relation.count("var") != len(scope):
            continue

        constraint = relation.replace("var1", "scope[0]")
        for j in range(1, len(scope)):
            constraint = constraint.replace("var" + str(j + 1), "scope[" + str(j) + "]")

        constraint = eval(constraint)

        if hash(constraint) == hash(c):
            return i

    return -1




def get_divisors(n):
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def join_con_net(C1, C2):
    C3 = [[c1 & c2 if c1 is not c2 else c1 for c2 in C2] for c1 in C1]
    C3 = list(chain.from_iterable(C3))
    C3 = remove_redundant_conj(C3)
    return C3


def remove_redundant_conj(C1):
    C2 = list()

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        flag_eq = False
        flag_neq = False
        flag_geq = False
        flag_leq = False
        flag_ge = False
        flag_le = False

        for c1 in conj_args:
            print(c1.name)
            # Tias is on 3.9, no 'match' please!
            if c1.name == "==":
                flag_eq = True
            elif c1.name == "!=":
                flag_neq = True
            elif c1.name == "<=":
                flag_leq = True
            elif c1.name == ">=":
                flag_geq = True
            elif c1.name == "<":
                flag_le = True
            elif c1.name == ">":
                flag_ge = True
            else:
                raise Exception("constraint name is not recognised")

            if not ((flag_eq and (flag_neq or flag_le or flag_ge)) or (
                    (flag_leq or flag_le) and ((flag_geq or flag_ge)))):
                C2.append(c)
    return C2


def get_max_conjunction_size(C1):
    max_conj_size = 0

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        max_conj_size = max(len(conj_args), max_conj_size)

    return max_conj_size


def get_delta_p(C1):
    max_conj_size = get_max_conjunction_size(C1)

    Delta_p = [[] for _ in range(max_conj_size)]

    for c in C1:

        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        Delta_p[len(conj_args) - 1].append(c)

    return Delta_p


def compute_sample_weights(Y):
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    sw = []

    for i in range(len(Y)):
        if Y[i] == False:
            sw.append(c_w[0])
        else:
            sw.append(c_w[1])

    return sw


class Metrics:

    def __init__(self):
        self.gen_queries_count = 0
        self.queries_count = 0
        self.top_lvl_queries = 0
        self.generated_queries = 0
        self.findscope_queries = 0
        self.findc_queries = 0

        self.average_size_queries = 0

        self.start_time_query = time.time()
        self.max_waiting_time = 0
        self.generation_time = 0

        self.converged = 1
        self.N_egativeQ = set()
        self.gen_no_answers = 0
        self.gen_yes_answers = 0

    def increase_gen_queries_count(self, amount=1):
        self.gen_queries_count += amount

    def increase_queries_count(self, amount=1):
        self.queries_count += amount

    def increase_top_queries(self, amount=1):
        self.top_lvl_queries += amount

    def increase_generated_queries(self, amount=1):
        self.generated_queries += amount

    def increase_findscope_queries(self, amount=1):
        self.findscope_queries += amount

    def increase_findc_queries(self, amount=1):
        self.findc_queries += amount

    def increase_generation_time(self, amount):
        self.generation_time += self.generation_time

    def increase_queries_size(self, amount):
        self.average_size_queries += 1

    def aggreagate_max_waiting_time(self, max2):
        if self.max_waiting_time < max2:
            self.max_waiting_time = max2

    def aggregate_convergence(self, converged2):
        if self.converged + converged2 < 2:
            self.converged = 0

    def __add__(self, other):

        new = self

        new.increase_queries_count(other.queries_count)
        new.increase_top_queries(other.top_lvl_queries)
        new.increase_generated_queries(other.generated_queries)
        new.increase_findscope_queries(other.findscope_queries)
        new.increase_findc_queries(other.findc_queries)
        new.increase_generation_time(other.generation_time)
        new.increase_queries_size(other.average_size_queries)

        new.aggreagate_max_waiting_time(other.max_waiting_time)
        new.aggregate_convergence(other.converged)

        return new


def find_suitable_vars_subset2(l, B, Y):
    if len(Y) <= get_min_arity(B) or len(B) < 1:
        return Y

    scope = get_scope(B[0])
    Y_prime = list(set(Y) - set(scope))

    l2 = int(l) - len(scope)

    if l2 > 0:
        Y1 = Y_prime[:l2]
    else:
        Y1 = []

    [Y1.append(y) for y in scope]

    return Y1


def calculate_modularity(G, communities):
    """
    Calculate the modularity of a given partition.
    """
    m = G.size(weight='weight')
    degrees = dict(G.degree(weight='weight'))
    Q = 0
    for community in communities:
        Lc = 0
        Dc = 0
        for u in community:
            Dc += degrees[u]
            for v in community:
                if G.has_edge(u, v):
                    Lc += G[u][v].get('weight', 1)
        Q += (Lc / (2 * m)) - (Dc / (2 * m)) ** 2
    return Q


def get_communities(partition):
    """
    Get the communities from the partition.
    """
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    return list(communities.values())


def optimize_modularity(G, max_iterations=1000, min_modularity_improvement=0.0001):
    partition = {node: i for i, node in enumerate(G.nodes())}
    best_modularity = -1
    best_partition = partition.copy()
    improvement = True
    iteration = 0

    while improvement and iteration < max_iterations:
        improvement = False
        current_modularity = calculate_modularity(G, get_communities(partition))

        for node in G.nodes():
            best_community = partition[node]
            best_increase = 0
            current_community = partition[node]

            # remove node from its current community
            partition[node] = -1

            for neighbor in G.neighbors(node):
                if partition[neighbor] != -1:
                    community = partition[neighbor]
                    partition[node] = community
                    new_modularity = calculate_modularity(G, get_communities(partition))
                    increase = new_modularity - current_modularity

                    if increase > best_increase:
                        best_increase = increase
                        best_community = community

                    partition[node] = -1

            partition[node] = best_community

            if best_increase > min_modularity_improvement:
                improvement = True

        new_modularity = calculate_modularity(G, get_communities(partition))

        if new_modularity > best_modularity:
            best_modularity = new_modularity
            best_partition = partition.copy()

        # aggregate the graph
        communities = get_communities(partition)
        new_G = nx.Graph()

        for i, community in enumerate(communities):
            new_node = i
            new_G.add_node(new_node)

            for node in community:
                for neighbor in G.neighbors(node):
                    if partition[neighbor] == partition[node]:
                        if new_G.has_edge(new_node, new_node):
                            new_G[new_node][new_node]['weight'] += G[node][neighbor].get('weight', 1)
                        else:
                            new_G.add_edge(new_node, new_node, weight=G[node][neighbor].get('weight', 1))
                    else:
                        neighbor_comm = partition[neighbor]

                        if new_G.has_edge(new_node, neighbor_comm):
                            new_G[new_node][neighbor_comm]['weight'] += G[node][neighbor].get('weight', 1)
                        else:
                            new_G.add_edge(new_node, neighbor_comm, weight=G[node][neighbor].get('weight', 1))

        G = new_G
        partition = {node: i for i, node in enumerate(G.nodes())}
        iteration += 1

    return get_communities(best_partition)

def transform_bias_constraints_pl_mapping(biases, instance_mapping, instance):
    """
    Transforms bias constraint strings into CPMpy constraints using the provided instance mapping.
    Handles VM allocation specific constraints.
    """
    import cpmpy as cp
    
    constraints = []
    
    # First ensure all required variables exist in the mapping
    max_vm = 0
    max_pm = 0
    
    # Scan biases to find max VM and PM numbers
    for bias in biases:
        bias = bias.strip()
        vm_matches = re.findall(r'assign_VM_(\d+)', bias)
        for vm_num in vm_matches:
            max_vm = max(max_vm, int(vm_num))
        
        pm_matches = re.findall(r'active_PM_(\d+)', bias)
        for pm_num in pm_matches:
            max_pm = max(max_pm, int(pm_num))
    
    # Create any missing variables
    for i in range(1, max_vm + 1):
        var_name = f'assign_VM_{i}'
        if var_name not in instance_mapping:
            var = cp.intvar(1, 5, name=var_name)  # VMs can be assigned to PMs 1-5
            instance_mapping[var_name] = var
    
    for i in range(1, max_pm + 1):
        var_name = f'active_PM_{i}'
        if var_name not in instance_mapping:
            var = cp.intvar(0, 1, name=var_name)  # PM active status is binary
            instance_mapping[var_name] = var
    
    # Process each bias constraint
    for bias in biases:
        bias = bias.strip()
        if bias.startswith('Sum('):
            # Handle Sum constraints
            try:
                # Extract the sum expression and the comparison
                sum_expr = bias[4:bias.rindex(')')+1]
                comparison = bias[bias.rindex(')')+1:].strip()
                
                # Parse the sum terms
                terms = []
                for term in sum_expr.strip('()').split(','):
                    term = term.strip()
                    if '*' in term:
                        # Handle coefficient * variable
                        coeff_part, var_part = term.split('*', 1)
                        coeff = int(coeff_part.strip('() '))
                        var_expr = var_part.strip('() ')
                        var_name = var_expr.split('==')[0].strip()
                        value = int(var_expr.split('==')[1].strip())
                        
                        if var_name in instance_mapping:
                            var = instance_mapping[var_name]
                            terms.append(coeff * (var == value))
                
                # Create the sum constraint
                sum_constraint = cp.sum(terms)
                
                # Add the comparison
                if '<=' in comparison:
                    bound = int(comparison.split('<=')[1].strip())
                    constraints.append(sum_constraint <= bound)
                elif '>=' in comparison:
                    bound = int(comparison.split('>=')[1].strip())
                    constraints.append(sum_constraint >= bound)
                elif '==' in comparison:
                    bound = int(comparison.split('==')[1].strip())
                    constraints.append(sum_constraint == bound)
            except Exception as e:
                print(f"Error parsing Sum constraint: {bias}")
                print(f"Error: {str(e)}")
                continue
        elif 'alldifferent' in bias.lower():
            # Handle AllDifferent constraints
            try:
                var_list = bias[bias.find('(')+1:bias.find(')')].split(',')
                vars_to_diff = []
                for var_name in var_list:
                    var_name = var_name.strip()
                    if var_name in instance_mapping:
                        vars_to_diff.append(instance_mapping[var_name])
                if vars_to_diff:
                    constraints.append(cp.AllDifferent(vars_to_diff))
            except Exception as e:
                print(f"Error parsing AllDifferent constraint: {bias}")
                print(f"Error: {str(e)}")
                continue
        else:
            # Handle simple comparison constraints
            tokens = bias.split()
            if len(tokens) != 3:
                continue
            
            var1_name = tokens[0]
            operator = tokens[1]
            var2_name = tokens[2]
            
            # Get the variables from the mapping
            if var1_name not in instance_mapping or var2_name not in instance_mapping:
                continue
            
            var1 = instance_mapping[var1_name]
            var2 = instance_mapping[var2_name]
            
            # Create the appropriate constraint
            if operator == '==':
                constraints.append(var1 == var2)
            elif operator == '!=':
                constraints.append(var1 != var2)
            elif operator == '<':
                constraints.append(var1 < var2)
            elif operator == '>':
                constraints.append(var1 > var2)
            elif operator == '<=':
                constraints.append(var1 <= var2)
            elif operator == '>=':
                constraints.append(var1 >= var2)
    
    return constraints

def get_var_name(var):
    """
    Get the name of a variable without its indices.

    :param var: The variable.
    :return: The name of the variable without its indices.
    """
    print(var.name)
    # Check if the name contains indices
    indices = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if indices:
        # If indices exist, remove them
        name = var.name.replace(indices[0], '')
    else:
        # If no indices, return the name as is
        name = var.name
    return name


def get_var_ndims(var):
    """
    Get the number of dimensions of a variable.

    :param var: The variable.
    :return: The number of dimensions of the variable.
    """
    dims = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if not dims:
        return 0  # Return 0 for non-indexed variables
    dims_str = "".join(dims)
    ndims = len(re.split(",", dims_str))
    return ndims


def get_var_dims(var):
    """
    Get the dimensions of a variable.

    :param var: The variable.
    :return: The dimensions of the variable.
    """
    dims = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if not dims:
        return []  # Return empty list for non-indexed variables
    dims_str = "".join(dims)
    dims = re.split(r"[\[\]]", dims_str)[1]
    dims = [int(dim) for dim in re.split(",", dims)]
    return dims

def extract_features(constraint):
    """
    Extracts features from a constraint as described in Table 2 of the paper.
    """
    scope = get_scope(constraint)
    arity = len(scope)
    
    # Basic features
    features = {
        'relation': constraint.name,
        'arity': arity,
        'has_constant': 0,
        'constant_value': 0
    }

    # Check for constants
    if hasattr(constraint, 'args'):
        for arg in constraint.args:
            if isinstance(arg, int):
                features['has_constant'] = 1
                features['constant_value'] = arg
                break

    # Variable name features
    var_names = [v.name for v in scope]
    base_names = [re.sub(r'\d+', '', name) for name in var_names]
    features['var_name_same'] = 1 if len(set(base_names)) == 1 else 0

    # Dimensionality features
    var_ndims = [get_var_ndims(v) for v in scope]
    features['var_ndims_same'] = 1 if len(set(var_ndims)) == 1 else 0
    features['var_ndims_max'] = max(var_ndims) if var_ndims else 0
    features['var_ndims_min'] = min(var_ndims) if var_ndims else 0

    # Per-dimension features
    if features['var_ndims_same'] and features['var_ndims_max'] > 0:
        for d in range(features['var_ndims_max']):
            dim_indices = []
            all_have_dim = True
            for v in scope:
                dims = get_var_dims(v)
                if len(dims) > d:
                    dim_indices.append(dims[d])
                else:
                    all_have_dim = False
                    break
            
            features[f'var_dim{d}_has'] = 1 if all_have_dim else 0
            if all_have_dim:
                features[f'var_dim{d}_same'] = 1 if len(set(dim_indices)) == 1 else 0
                features[f'var_dim{d}_max'] = np.max(dim_indices)
                features[f'var_dim{d}_min'] = np.min(dim_indices)
                features[f'var_dim{d}_avg'] = np.mean(dim_indices)
                features[f'var_dim{d}_spread'] = np.max(dim_indices) - np.min(dim_indices)
            else:
                features[f'var_dim{d}_same'] = 0
                features[f'var_dim{d}_max'] = 0
                features[f'var_dim{d}_min'] = 0
                features[f'var_dim{d}_avg'] = 0
                features[f'var_dim{d}_spread'] = 0

    # Flatten for XGBoost
    feature_vector = [
        features['relation'], features['arity'], features['has_constant'], features['constant_value'],
        features['var_name_same'], features['var_ndims_same'], features['var_ndims_max'], features['var_ndims_min']
    ]
    for d in range(features.get('var_ndims_max', 0)):
        feature_vector.extend([
            features.get(f'var_dim{d}_has', 0), features.get(f'var_dim{d}_same', 0),
            features.get(f'var_dim{d}_max', 0), features.get(f'var_dim{d}_min', 0),
            features.get(f'var_dim{d}_avg', 0), features.get(f'var_dim{d}_spread', 0)
        ])
        
    # The model expects a numerical array. We need to handle the categorical 'relation' feature.
    # A full implementation would use one-hot encoding based on a known vocabulary.
    # For now, we'll use a simple hash as a placeholder.
    feature_vector[0] = hash(features['relation']) % 1000  # Simple hashing for categorical feature
    
    # Ensure all features are numeric and handle missing dimension features
    final_vector = []
    for val in feature_vector:
        final_vector.append(float(val))

    # Pad the feature vector to a fixed size if necessary (e.g., for max dimensions)
    # The XGBoost model was trained with a fixed number of features. Let's assume it's 50 for now.
    # This needs to match the training configuration.
    expected_feature_count = 30 
    while len(final_vector) < expected_feature_count:
        final_vector.append(0.0)

    return final_vector[:expected_feature_count]

from pycona import ConstraintOracle

def get_positive_examples(problem_class, n_examples):
    return problem_class.get_positive_examples(n_examples)

def get_oracles(problem_class):
    return ConstraintOracle(problem_class.C_T)