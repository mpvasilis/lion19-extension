import numpy as np
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.core import Comparison
from cpmpy.expressions.globalconstraints import AllDifferent
import re

def _is_complete_row(var_indices, var_dims):
    if not var_dims or len(var_dims) != 2:
        return 0 # not a 2D matrix

    rows, cols = var_dims
    if not var_indices or not all(isinstance(idx, tuple) and len(idx) == 2 for idx in var_indices):
        return 0 # indices are not all 2D tuples

    if len(var_indices) != cols: # must have exactly 'cols' variables for a complete row
        return 0

    first_row_index = var_indices[0][0]
    if not all(idx[0] == first_row_index for idx in var_indices):
        return 0 # not all variables are in the same row

    # see if all column indices from 0 to cols-1 are present
    column_indices = {idx[1] for idx in var_indices}
    if column_indices == set(range(cols)):
        return 1
    return 0

def _is_complete_column(var_indices, var_dims):
    if not var_dims or len(var_dims) != 2:
        return 0 # not a 2D matrix

    rows, cols = var_dims
    if not var_indices or not all(isinstance(idx, tuple) and len(idx) == 2 for idx in var_indices):
        return 0 # indices are not all 2D tuples

    if len(var_indices) != rows: # must have exactly 'rows' variables for a complete column
        return 0

    first_col_index = var_indices[0][1]
    if not all(idx[1] == first_col_index for idx in var_indices):
        return 0 # not all variables are in the same column

    # check if all row indices from 0 to rows-1 are present
    row_indices = {idx[0] for idx in var_indices}
    if row_indices == set(range(rows)):
        return 1
    return 0

def extract_constraint_features(constraint, all_variables, all_constraints=None):
    features = {}
    
    # Determine constraint type (Relation feature)
    constraint_type = type(constraint).__name__
    if isinstance(constraint, Comparison) and hasattr(constraint.args[0], 'name'):
        if constraint.args[0].name == 'sum':
            constraint_type = 'Sum'
        elif constraint.args[0].name == 'count':
            constraint_type = 'Count'
    elif isinstance(constraint, AllDifferent):
        constraint_type = 'AllDifferent'
    
    # 1. Relation (String) - Original feature
    features['relation'] = constraint_type
    
    # Get variables in the constraint
    vars_in_constraint = get_variables(constraint)
    
    # 2. Arity (Int) - Original feature
    scope_size = len(vars_in_constraint)
    features['arity'] = scope_size
    
    # 3. Has constant (Bool) - Original feature
    # 4. Constant (Int) - Original feature
    features['has_constant'] = False
    features['constant_value'] = 0
    
    if isinstance(constraint, Comparison) and len(constraint.args) > 1:
        right_operand = constraint.args[1]
        if isinstance(right_operand, (int, float)) or (hasattr(right_operand, 'is_const') and right_operand.is_const()):
            features['has_constant'] = True
            try:
                const_value = int(right_operand) if isinstance(right_operand, (int, float)) else int(right_operand.value)
                features['constant_value'] = const_value
            except (ValueError, AttributeError):
                features['constant_value'] = 0
    
    # Check if all variables have the same base name
    var_base_names = set()
    var_indices = []
    var_dims = None
    has_parsed_indices = False
    dimensions_per_var = []
    
    # Initialize with at least one element to avoid empty list issues
    if not vars_in_constraint:
        dimensions_per_var = [0]
    
    for var in vars_in_constraint:
        if hasattr(var, 'name'):
            # Extract base name (without indices)
            base_name = var.name.split('[')[0] if '[' in var.name else var.name
            var_base_names.add(base_name)
            
            # Parse indices for dimensional analysis
            if '[' in var.name and ']' in var.name:
                try:
                    idx_str = var.name[var.name.find('[')+1:var.name.find(']')]
                    if ',' in idx_str:
                        indices = list(map(int, idx_str.split(',')))
                        var_indices.append(tuple(indices))
                        dimensions_per_var.append(len(indices))
                        matrix_vars = True
                        has_parsed_indices = True
                        if var_dims is None:
                            max_dims = [0] * len(indices)
                            for v in all_variables:
                                if hasattr(v, 'name') and '[' in v.name and ']' in v.name:
                                    try:
                                        idx_str = v.name[v.name.find('[')+1:v.name.find(']')]
                                        if ',' in idx_str:
                                            v_indices = list(map(int, idx_str.split(',')))
                                            for i, idx in enumerate(v_indices):
                                                if i < len(max_dims):
                                                    max_dims[i] = max(max_dims[i], idx)
                                    except:
                                        pass
                            var_dims = tuple(dim + 1 for dim in max_dims)
                    else:
                        # 1D index pattern like "var[i]"
                        try:
                            i = int(idx_str)
                            var_indices.append((i,))  # Store as a tuple for consistency
                            dimensions_per_var.append(1)
                            matrix_vars = False 
                            has_parsed_indices = True
                            if var_dims is None:
                                # Estimate max dimension
                                max_i = 0
                                for v in all_variables:
                                    if hasattr(v, 'name') and '[' in v.name and ']' in v.name:
                                        try:
                                            vi = int(v.name[v.name.find('[')+1:v.name.find(']')])
                                            max_i = max(max_i, vi)
                                        except:
                                            pass
                                var_dims = (max_i + 1,)
                        except ValueError:
                            var_indices.append(None)
                            dimensions_per_var.append(0)
                except Exception as e:
                    print(f"Warning: Error parsing index for variable {var.name}: {e}")
                    var_indices.append(None)
                    dimensions_per_var.append(0)
            else:
                var_indices.append(None)
                dimensions_per_var.append(0)
        else:
            # Handle variables without name attribute
            var_indices.append(None)
            dimensions_per_var.append(0)
    
    # 5. Var name same (Bool) - Original feature
    features['var_name_same'] = len(var_base_names) == 1 if var_base_names else False
    
    # Remove any None or invalid values from dimensions_per_var
    dimensions_per_var = [d for d in dimensions_per_var if isinstance(d, (int, float))]
    
    # 6. Var Ndims same (Bool) - Original feature
    features['var_ndims_same'] = len(set(dimensions_per_var)) == 1 if dimensions_per_var else True
    
    # 7-8. Var Ndims max/min (Int) - Original features
    # Ensure dimensions_per_var is not empty to avoid errors
    if dimensions_per_var:
        features['var_ndims_max'] = max(dimensions_per_var)
        features['var_ndims_min'] = min(dimensions_per_var)
    else:
        features['var_ndims_max'] = 0
        features['var_ndims_min'] = 0
    
    # Process dimensional features (9-14) - Original features
    # First, filter out None values from var_indices
    valid_indices = [idx for idx in var_indices if idx is not None]
    
    # Initialize dimension-specific features
    max_dims = features['var_ndims_max']
    for dim in range(max_dims):
        features[f'var_dim{dim}_has'] = False
        features[f'var_dim{dim}_same'] = False
        features[f'var_dim{dim}_max'] = 0
        features[f'var_dim{dim}_min'] = 0
        features[f'var_dim{dim}_avg'] = 0.0
        features[f'var_dim{dim}_spread'] = 0.0
    
    # Calculate dimensional features if we have valid indices
    if valid_indices:
        # Determine how many dimensions we need to analyze
        dims_to_analyze = max(len(idx) for idx in valid_indices)
        
        for dim in range(dims_to_analyze):
            # Extract values for this dimension across all variables that have it
            dim_values = [idx[dim] for idx in valid_indices if dim < len(idx)]
            
            if dim_values:
                # 9. Var dimi has (Bool) - Original feature
                features[f'var_dim{dim}_has'] = len(dim_values) == len(valid_indices)
                
                # 10. Var dimi same (Bool) - Original feature
                features[f'var_dim{dim}_same'] = len(set(dim_values)) == 1
                
                # 11-12. Var dimi max/min (Int) - Original features
                features[f'var_dim{dim}_max'] = max(dim_values)
                features[f'var_dim{dim}_min'] = min(dim_values)
                
                # 13. Var dimi avg (Float) - Original feature
                features[f'var_dim{dim}_avg'] = sum(dim_values) / len(dim_values)
                
                # 14. Var dimi spread (Float) - Original feature
                features[f'var_dim{dim}_spread'] = max(dim_values) - min(dim_values)
    
    # For compatibility with the rest of the codebase, keep attribute names 
    # but set to zeros or empty values
    features['constraint_type'] = constraint_type
    features['scope_size'] = scope_size
    features['scope_size_zscore'] = 0
    features['type_frequency'] = 0
    features['normalized_scope_size'] = 0
    features['is_row'] = 0
    features['is_column'] = 0
    features['is_block'] = 0
    features['is_diagonal'] = 0
    features['row_min'] = 0
    features['row_max'] = 0
    features['col_min'] = 0
    features['col_max'] = 0
    features['row_span'] = 0
    features['col_span'] = 0
    features['index_min'] = 0
    features['index_max'] = 0
    features['index_span'] = 0
    
    return features
