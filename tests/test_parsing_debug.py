import re

constraint_str = "Count([x1, x2, x3, x4], 5) == 2"

print(f"Testing: {constraint_str}")

# Extract target value
# Need to handle nested brackets: Count([x1, x2], 5)
match = re.search(r'Count\(.+?,\s*(\d+)\)', constraint_str)
if match:
    target_value = int(match.group(1))
    print(f"Target value: {target_value}")
else:
    print("Failed to extract target value")

# Extract operator and bound
if '<=' in constraint_str:
    operator = '<='
    match = re.search(r'<=\s*(\d+)', constraint_str)
    if match:
        bound = int(match.group(1))
        print(f"Operator: {operator}, Bound: {bound}")
elif '==' in constraint_str:
    operator = '=='
    match = re.search(r'==\s*(\d+)', constraint_str)
    if match:
        bound = int(match.group(1))
        print(f"Operator: {operator}, Bound: {bound}")

# Test with example
values = {"x1": 5, "x2": 5, "x3": 5, "x4": 3}
var_list = list(values.keys())

vars_with_value = [v for v in var_list if values[v] == target_value]
vars_without_value = [v for v in var_list if values[v] != target_value]
actual_count = len(vars_with_value)

print(f"\nVariables with value {target_value}: {vars_with_value}")
print(f"Actual count: {actual_count}, Expected (bound): {bound}")

if operator == '==' or operator == '<=':
    if actual_count > bound:
        print(f"Result: Too many! Return vars_with_value = {vars_with_value}")
    elif actual_count < bound and operator == '==':
        print(f"Result: Too few! Return vars_without_value = {vars_without_value}")
