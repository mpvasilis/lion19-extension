import numpy as np
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.core import Comparison
from cpmpy.expressions.globalconstraints import AllDifferent
import re

def _is_complete_row(var_indices, var_dims):
    if not var_dims or len(var_dims) != 2:
        return 0 

    rows, cols = var_dims
    if not var_indices or not all(isinstance(idx, tuple) and len(idx) == 2 for idx in var_indices):
        return 0 

    if len(var_indices) != cols: 
        return 0

    first_row_index = var_indices[0][0]
    if not all(idx[0] == first_row_index for idx in var_indices):
        return 0 

    column_indices = {idx[1] for idx in var_indices}
    if column_indices == set(range(cols)):
        return 1
    return 0

def _is_complete_column(var_indices, var_dims):
    if not var_dims or len(var_dims) != 2:
        return 0 

    rows, cols = var_dims
    if not var_indices or not all(isinstance(idx, tuple) and len(idx) == 2 for idx in var_indices):
        return 0 

    if len(var_indices) != rows: 
        return 0

    first_col_index = var_indices[0][1]
    if not all(idx[1] == first_col_index for idx in var_indices):
        return 0 

    row_indices = {idx[0] for idx in var_indices}
    if row_indices == set(range(rows)):
        return 1
    return 0

def extract_constraint_features(constraint, all_variables, all_constraints=None):
    features = {}

    constraint_type = type(constraint).__name__
    if isinstance(constraint, Comparison) and hasattr(constraint.args[0], 'name'):
        if constraint.args[0].name == 'sum':
            constraint_type = 'Sum'
        elif constraint.args[0].name == 'count':
            constraint_type = 'Count'
    elif isinstance(constraint, AllDifferent):
        constraint_type = 'AllDifferent'

    features['relation'] = constraint_type

    vars_in_constraint = get_variables(constraint)

    scope_size = len(vars_in_constraint)
    features['arity'] = scope_size


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

    var_base_names = set()
    var_indices = []
    var_dims = None
    has_parsed_indices = False
    dimensions_per_var = []

    if not vars_in_constraint:
        dimensions_per_var = [0]
    
    for var in vars_in_constraint:
        if hasattr(var, 'name'):

            base_name = var.name.split('[')[0] if '[' in var.name else var.name
            var_base_names.add(base_name)

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

                        try:
                            i = int(idx_str)
                            var_indices.append((i,))  
                            dimensions_per_var.append(1)
                            matrix_vars = False 
                            has_parsed_indices = True
                            if var_dims is None:

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

            var_indices.append(None)
            dimensions_per_var.append(0)

    features['var_name_same'] = len(var_base_names) == 1 if var_base_names else False

    dimensions_per_var = [d for d in dimensions_per_var if isinstance(d, (int, float))]

    features['var_ndims_same'] = len(set(dimensions_per_var)) == 1 if dimensions_per_var else True


    if dimensions_per_var:
        features['var_ndims_max'] = max(dimensions_per_var)
        features['var_ndims_min'] = min(dimensions_per_var)
    else:
        features['var_ndims_max'] = 0
        features['var_ndims_min'] = 0


    valid_indices = [idx for idx in var_indices if idx is not None]

    max_dims = features['var_ndims_max']
    for dim in range(max_dims):
        features[f'var_dim{dim}_has'] = False
        features[f'var_dim{dim}_same'] = False
        features[f'var_dim{dim}_max'] = 0
        features[f'var_dim{dim}_min'] = 0
        features[f'var_dim{dim}_avg'] = 0.0
        features[f'var_dim{dim}_spread'] = 0.0

    if valid_indices:

        dims_to_analyze = max(len(idx) for idx in valid_indices)
        
        for dim in range(dims_to_analyze):

            dim_values = [idx[dim] for idx in valid_indices if dim < len(idx)]
            
            if dim_values:

                features[f'var_dim{dim}_has'] = len(dim_values) == len(valid_indices)

                features[f'var_dim{dim}_same'] = len(set(dim_values)) == 1

                features[f'var_dim{dim}_max'] = max(dim_values)
                features[f'var_dim{dim}_min'] = min(dim_values)

                features[f'var_dim{dim}_avg'] = sum(dim_values) / len(dim_values)

                features[f'var_dim{dim}_spread'] = max(dim_values) - min(dim_values)


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
