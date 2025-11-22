import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_sudoku_4x4_gt_binary(block_size_row=2, block_size_col=2, grid_size=4):
    """
    Construct a 4x4 Sudoku with Greater-Than constraints (binary version).
    
    This version decomposes AllDifferent constraints into binary != constraints
    for use with binary constraint acquisition algorithms like MQuAcq-2.
    """

    parameters = {
        "block_size_row": block_size_row, 
        "block_size_col": block_size_col, 
        "grid_size": grid_size
    }

    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = cp.Model()

    # Row constraints: all different in each row (decomposed to binary)
    for row in grid:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                model += (row[i] != row[j])

    # Column constraints: all different in each column (decomposed to binary)
    for col in grid.T:
        for i in range(len(col)):
            for j in range(i + 1, len(col)):
                model += (col[i] != col[j])

    # Block constraints: all different in each 2x2 block (decomposed to binary)
    for block_row in range(0, grid_size, block_size_row):
        for block_col in range(0, grid_size, block_size_col):
            block = grid[block_row:block_row + block_size_row, block_col:block_col + block_size_col]
            block_flat = block.flatten()
            for i in range(len(block_flat)):
                for j in range(i + 1, len(block_flat)):
                    model += (block_flat[i] != block_flat[j])

    # Greater-than constraints for 4x4 grid
    # Use sparse greater-than constraints to keep problem solvable
    # These are carefully chosen to not over-constrain the problem
    
    # Horizontal greater-than constraints (left > right)
    true_horizontal_gt = [
        (0, 0, 0, 1),  # grid[0,0] > grid[0,1]
        (2, 1, 2, 2),  # grid[2,1] > grid[2,2]
    ]
    
    for r1, c1, r2, c2 in true_horizontal_gt:
        if r2 < grid_size and c2 < grid_size:
            model += (grid[r1, c1] > grid[r2, c2])

    # Vertical greater-than constraints (top > bottom)
    true_vertical_gt = [
        (1, 0, 2, 0),  # grid[1,0] > grid[2,0]
        (0, 3, 1, 3),  # grid[0,3] > grid[1,3]
    ]
    
    for r1, c1, r2, c2 in true_vertical_gt:
        if r1 < grid_size and r2 < grid_size and c1 < grid_size and c2 < grid_size:
            model += (grid[r1, c1] > grid[r2, c2])

    C_T = list(set(toplevel_list(model.constraints)))

    # Binary language for constraint acquisition
    AV = absvar(2)

    lang = [
        AV[0] == AV[1], 
        AV[0] != AV[1], 
        AV[0] < AV[1], 
        AV[0] > AV[1], 
        AV[0] >= AV[1], 
        AV[0] <= AV[1]
    ]

    instance = ProblemInstance(
        variables=grid, 
        params=parameters, 
        language=lang, 
        name="sudoku_4x4_gt"
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle

