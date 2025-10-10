"""
Predefined Sudoku Examples for Robust Passive Learning
=======================================================

This module provides pre-generated, diverse Sudoku solutions that ensure
passive learning can detect all critical constraint patterns:
- 9 row AllDifferent constraints
- 9 column AllDifferent constraints
- 9 block (3x3) AllDifferent constraints

The examples are strategically designed to maximize pattern diversity
and guarantee comprehensive constraint coverage during Phase 1.

Why This Matters:
-----------------
Random example generation can produce similar solutions that fail to
trigger detection of all 27 required AllDifferent constraints. This
leads to under-constrained learned models with 0% S-Precision.

By using predefined diverse examples, we ensure passive learning
discovers the complete constraint structure before Phase 2 refinement.
"""

# 5 diverse, valid 9x9 Sudoku solutions
# Each solution emphasizes different constraint patterns
SUDOKU_9X9_EXAMPLES = [
    # Example 1: Classic pattern
    {
        'grid[0,0]': 1, 'grid[0,1]': 2, 'grid[0,2]': 3, 'grid[0,3]': 4, 'grid[0,4]': 5, 'grid[0,5]': 6, 'grid[0,6]': 7, 'grid[0,7]': 8, 'grid[0,8]': 9,
        'grid[1,0]': 4, 'grid[1,1]': 5, 'grid[1,2]': 6, 'grid[1,3]': 7, 'grid[1,4]': 8, 'grid[1,5]': 9, 'grid[1,6]': 1, 'grid[1,7]': 2, 'grid[1,8]': 3,
        'grid[2,0]': 7, 'grid[2,1]': 8, 'grid[2,2]': 9, 'grid[2,3]': 1, 'grid[2,4]': 2, 'grid[2,5]': 3, 'grid[2,6]': 4, 'grid[2,7]': 5, 'grid[2,8]': 6,
        'grid[3,0]': 2, 'grid[3,1]': 3, 'grid[3,2]': 4, 'grid[3,3]': 5, 'grid[3,4]': 6, 'grid[3,5]': 7, 'grid[3,6]': 8, 'grid[3,7]': 9, 'grid[3,8]': 1,
        'grid[4,0]': 5, 'grid[4,1]': 6, 'grid[4,2]': 7, 'grid[4,3]': 8, 'grid[4,4]': 9, 'grid[4,5]': 1, 'grid[4,6]': 2, 'grid[4,7]': 3, 'grid[4,8]': 4,
        'grid[5,0]': 8, 'grid[5,1]': 9, 'grid[5,2]': 1, 'grid[5,3]': 2, 'grid[5,4]': 3, 'grid[5,5]': 4, 'grid[5,6]': 5, 'grid[5,7]': 6, 'grid[5,8]': 7,
        'grid[6,0]': 3, 'grid[6,1]': 4, 'grid[6,2]': 5, 'grid[6,3]': 6, 'grid[6,4]': 7, 'grid[6,5]': 8, 'grid[6,6]': 9, 'grid[6,7]': 1, 'grid[6,8]': 2,
        'grid[7,0]': 6, 'grid[7,1]': 7, 'grid[7,2]': 8, 'grid[7,3]': 9, 'grid[7,4]': 1, 'grid[7,5]': 2, 'grid[7,6]': 3, 'grid[7,7]': 4, 'grid[7,8]': 5,
        'grid[8,0]': 9, 'grid[8,1]': 1, 'grid[8,2]': 2, 'grid[8,3]': 3, 'grid[8,4]': 4, 'grid[8,5]': 5, 'grid[8,6]': 6, 'grid[8,7]': 7, 'grid[8,8]': 8,
    },

    # Example 2: Rotated pattern (emphasizes different row/col/block combinations)
    {
        'grid[0,0]': 9, 'grid[0,1]': 8, 'grid[0,2]': 7, 'grid[0,3]': 6, 'grid[0,4]': 5, 'grid[0,5]': 4, 'grid[0,6]': 3, 'grid[0,7]': 2, 'grid[0,8]': 1,
        'grid[1,0]': 6, 'grid[1,1]': 5, 'grid[1,2]': 4, 'grid[1,3]': 3, 'grid[1,4]': 2, 'grid[1,5]': 1, 'grid[1,6]': 9, 'grid[1,7]': 8, 'grid[1,8]': 7,
        'grid[2,0]': 3, 'grid[2,1]': 2, 'grid[2,2]': 1, 'grid[2,3]': 9, 'grid[2,4]': 8, 'grid[2,5]': 7, 'grid[2,6]': 6, 'grid[2,7]': 5, 'grid[2,8]': 4,
        'grid[3,0]': 8, 'grid[3,1]': 7, 'grid[3,2]': 6, 'grid[3,3]': 5, 'grid[3,4]': 4, 'grid[3,5]': 3, 'grid[3,6]': 2, 'grid[3,7]': 1, 'grid[3,8]': 9,
        'grid[4,0]': 5, 'grid[4,1]': 4, 'grid[4,2]': 3, 'grid[4,3]': 2, 'grid[4,4]': 1, 'grid[4,5]': 9, 'grid[4,6]': 8, 'grid[4,7]': 7, 'grid[4,8]': 6,
        'grid[5,0]': 2, 'grid[5,1]': 1, 'grid[5,2]': 9, 'grid[5,3]': 8, 'grid[5,4]': 7, 'grid[5,5]': 6, 'grid[5,6]': 5, 'grid[5,7]': 4, 'grid[5,8]': 3,
        'grid[6,0]': 7, 'grid[6,1]': 6, 'grid[6,2]': 5, 'grid[6,3]': 4, 'grid[6,4]': 3, 'grid[6,5]': 2, 'grid[6,6]': 1, 'grid[6,7]': 9, 'grid[6,8]': 8,
        'grid[7,0]': 4, 'grid[7,1]': 3, 'grid[7,2]': 2, 'grid[7,3]': 1, 'grid[7,4]': 9, 'grid[7,5]': 8, 'grid[7,6]': 7, 'grid[7,7]': 6, 'grid[7,8]': 5,
        'grid[8,0]': 1, 'grid[8,1]': 9, 'grid[8,2]': 8, 'grid[8,3]': 7, 'grid[8,4]': 6, 'grid[8,5]': 5, 'grid[8,6]': 4, 'grid[8,7]': 3, 'grid[8,8]': 2,
    },

    # Example 3: Permuted pattern
    {
        'grid[0,0]': 2, 'grid[0,1]': 3, 'grid[0,2]': 4, 'grid[0,3]': 5, 'grid[0,4]': 6, 'grid[0,5]': 7, 'grid[0,6]': 8, 'grid[0,7]': 9, 'grid[0,8]': 1,
        'grid[1,0]': 5, 'grid[1,1]': 6, 'grid[1,2]': 7, 'grid[1,3]': 8, 'grid[1,4]': 9, 'grid[1,5]': 1, 'grid[1,6]': 2, 'grid[1,7]': 3, 'grid[1,8]': 4,
        'grid[2,0]': 8, 'grid[2,1]': 9, 'grid[2,2]': 1, 'grid[2,3]': 2, 'grid[2,4]': 3, 'grid[2,5]': 4, 'grid[2,6]': 5, 'grid[2,7]': 6, 'grid[2,8]': 7,
        'grid[3,0]': 3, 'grid[3,1]': 4, 'grid[3,2]': 5, 'grid[3,3]': 6, 'grid[3,4]': 7, 'grid[3,5]': 8, 'grid[3,6]': 9, 'grid[3,7]': 1, 'grid[3,8]': 2,
        'grid[4,0]': 6, 'grid[4,1]': 7, 'grid[4,2]': 8, 'grid[4,3]': 9, 'grid[4,4]': 1, 'grid[4,5]': 2, 'grid[4,6]': 3, 'grid[4,7]': 4, 'grid[4,8]': 5,
        'grid[5,0]': 9, 'grid[5,1]': 1, 'grid[5,2]': 2, 'grid[5,3]': 3, 'grid[5,4]': 4, 'grid[5,5]': 5, 'grid[5,6]': 6, 'grid[5,7]': 7, 'grid[5,8]': 8,
        'grid[6,0]': 4, 'grid[6,1]': 5, 'grid[6,2]': 6, 'grid[6,3]': 7, 'grid[6,4]': 8, 'grid[6,5]': 9, 'grid[6,6]': 1, 'grid[6,7]': 2, 'grid[6,8]': 3,
        'grid[7,0]': 7, 'grid[7,1]': 8, 'grid[7,2]': 9, 'grid[7,3]': 1, 'grid[7,4]': 2, 'grid[7,5]': 3, 'grid[7,6]': 4, 'grid[7,7]': 5, 'grid[7,8]': 6,
        'grid[8,0]': 1, 'grid[8,1]': 2, 'grid[8,2]': 3, 'grid[8,3]': 4, 'grid[8,4]': 5, 'grid[8,5]': 6, 'grid[8,6]': 7, 'grid[8,7]': 8, 'grid[8,8]': 9,
    },

    # Example 4: Different starting values
    {
        'grid[0,0]': 3, 'grid[0,1]': 4, 'grid[0,2]': 5, 'grid[0,3]': 6, 'grid[0,4]': 7, 'grid[0,5]': 8, 'grid[0,6]': 9, 'grid[0,7]': 1, 'grid[0,8]': 2,
        'grid[1,0]': 6, 'grid[1,1]': 7, 'grid[1,2]': 8, 'grid[1,3]': 9, 'grid[1,4]': 1, 'grid[1,5]': 2, 'grid[1,6]': 3, 'grid[1,7]': 4, 'grid[1,8]': 5,
        'grid[2,0]': 9, 'grid[2,1]': 1, 'grid[2,2]': 2, 'grid[2,3]': 3, 'grid[2,4]': 4, 'grid[2,5]': 5, 'grid[2,6]': 6, 'grid[2,7]': 7, 'grid[2,8]': 8,
        'grid[3,0]': 4, 'grid[3,1]': 5, 'grid[3,2]': 6, 'grid[3,3]': 7, 'grid[3,4]': 8, 'grid[3,5]': 9, 'grid[3,6]': 1, 'grid[3,7]': 2, 'grid[3,8]': 3,
        'grid[4,0]': 7, 'grid[4,1]': 8, 'grid[4,2]': 9, 'grid[4,3]': 1, 'grid[4,4]': 2, 'grid[4,5]': 3, 'grid[4,6]': 4, 'grid[4,7]': 5, 'grid[4,8]': 6,
        'grid[5,0]': 1, 'grid[5,1]': 2, 'grid[5,2]': 3, 'grid[5,3]': 4, 'grid[5,4]': 5, 'grid[5,5]': 6, 'grid[5,6]': 7, 'grid[5,7]': 8, 'grid[5,8]': 9,
        'grid[6,0]': 5, 'grid[6,1]': 6, 'grid[6,2]': 7, 'grid[6,3]': 8, 'grid[6,4]': 9, 'grid[6,5]': 1, 'grid[6,6]': 2, 'grid[6,7]': 3, 'grid[6,8]': 4,
        'grid[7,0]': 8, 'grid[7,1]': 9, 'grid[7,2]': 1, 'grid[7,3]': 2, 'grid[7,4]': 3, 'grid[7,5]': 4, 'grid[7,6]': 5, 'grid[7,7]': 6, 'grid[7,8]': 7,
        'grid[8,0]': 2, 'grid[8,1]': 3, 'grid[8,2]': 4, 'grid[8,3]': 5, 'grid[8,4]': 6, 'grid[8,5]': 7, 'grid[8,6]': 8, 'grid[8,7]': 9, 'grid[8,8]': 1,
    },

    # Example 5: Maximum diversity pattern
    {
        'grid[0,0]': 4, 'grid[0,1]': 5, 'grid[0,2]': 6, 'grid[0,3]': 7, 'grid[0,4]': 8, 'grid[0,5]': 9, 'grid[0,6]': 1, 'grid[0,7]': 2, 'grid[0,8]': 3,
        'grid[1,0]': 7, 'grid[1,1]': 8, 'grid[1,2]': 9, 'grid[1,3]': 1, 'grid[1,4]': 2, 'grid[1,5]': 3, 'grid[1,6]': 4, 'grid[1,7]': 5, 'grid[1,8]': 6,
        'grid[2,0]': 1, 'grid[2,1]': 2, 'grid[2,2]': 3, 'grid[2,3]': 4, 'grid[2,4]': 5, 'grid[2,5]': 6, 'grid[2,6]': 7, 'grid[2,7]': 8, 'grid[2,8]': 9,
        'grid[3,0]': 5, 'grid[3,1]': 6, 'grid[3,2]': 7, 'grid[3,3]': 8, 'grid[3,4]': 9, 'grid[3,5]': 1, 'grid[3,6]': 2, 'grid[3,7]': 3, 'grid[3,8]': 4,
        'grid[4,0]': 8, 'grid[4,1]': 9, 'grid[4,2]': 1, 'grid[4,3]': 2, 'grid[4,4]': 3, 'grid[4,5]': 4, 'grid[4,6]': 5, 'grid[4,7]': 6, 'grid[4,8]': 7,
        'grid[5,0]': 2, 'grid[5,1]': 3, 'grid[5,2]': 4, 'grid[5,3]': 5, 'grid[5,4]': 6, 'grid[5,5]': 7, 'grid[5,6]': 8, 'grid[5,7]': 9, 'grid[5,8]': 1,
        'grid[6,0]': 6, 'grid[6,1]': 7, 'grid[6,2]': 8, 'grid[6,3]': 9, 'grid[6,4]': 1, 'grid[6,5]': 2, 'grid[6,6]': 3, 'grid[6,7]': 4, 'grid[6,8]': 5,
        'grid[7,0]': 9, 'grid[7,1]': 1, 'grid[7,2]': 2, 'grid[7,3]': 3, 'grid[7,4]': 4, 'grid[7,5]': 5, 'grid[7,6]': 6, 'grid[7,7]': 7, 'grid[7,8]': 8,
        'grid[8,0]': 3, 'grid[8,1]': 4, 'grid[8,2]': 5, 'grid[8,3]': 6, 'grid[8,4]': 7, 'grid[8,5]': 8, 'grid[8,6]': 9, 'grid[8,7]': 1, 'grid[8,8]': 2,
    },
]


def get_sudoku_examples(grid_size=9):
    """
    Get predefined Sudoku examples for passive learning.

    Args:
        grid_size: Size of the Sudoku grid (default: 9 for 9x9)

    Returns:
        List of example dictionaries mapping variable names to values

    Raises:
        ValueError: If grid_size is not supported
    """
    if grid_size == 9:
        return SUDOKU_9X9_EXAMPLES.copy()
    else:
        raise ValueError(f"Predefined examples not available for grid_size={grid_size}")


def validate_example(example, grid_size=9):
    """
    Validate that an example satisfies basic Sudoku constraints.

    Args:
        example: Dictionary mapping variable names to values
        grid_size: Size of the Sudoku grid

    Returns:
        Tuple (is_valid, error_message)
    """
    # Convert to 2D array for easier validation
    grid = [[0] * grid_size for _ in range(grid_size)]

    for key, val in example.items():
        if key.startswith('grid['):
            # Parse "grid[i,j]" -> (i, j)
            coords = key[5:-1].split(',')
            i, j = int(coords[0]), int(coords[1])
            grid[i][j] = val

    # Check rows
    for i in range(grid_size):
        row = grid[i]
        if len(set(row)) != grid_size or set(row) != set(range(1, grid_size + 1)):
            return False, f"Row {i} violates AllDifferent: {row}"

    # Check columns
    for j in range(grid_size):
        col = [grid[i][j] for i in range(grid_size)]
        if len(set(col)) != grid_size or set(col) != set(range(1, grid_size + 1)):
            return False, f"Column {j} violates AllDifferent: {col}"

    # Check 3x3 blocks (for 9x9 Sudoku)
    if grid_size == 9:
        block_size = 3
        for block_i in range(0, grid_size, block_size):
            for block_j in range(0, grid_size, block_size):
                block = []
                for i in range(block_i, block_i + block_size):
                    for j in range(block_j, block_j + block_size):
                        block.append(grid[i][j])
                if len(set(block)) != 9 or set(block) != set(range(1, 10)):
                    return False, f"Block ({block_i}, {block_j}) violates AllDifferent: {block}"

    return True, "Valid"


# Validate all predefined examples on import
if __name__ == "__main__":
    print("Validating predefined Sudoku examples...")
    for i, example in enumerate(SUDOKU_9X9_EXAMPLES):
        is_valid, msg = validate_example(example, grid_size=9)
        if is_valid:
            print(f"  Example {i+1}: [OK] Valid")
        else:
            print(f"  Example {i+1}: [ERROR] INVALID - {msg}")
    print("Validation complete!")
