import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_sudoku_greater_than(block_size_row, block_size_col, grid_size):
    """
    Sudoku with additional greater-than constraints between adjacent cells.
    This is similar to Futoshiki puzzle.
    """
    parameters = {
        "block_size_row": block_size_row, 
        "block_size_col": block_size_col, 
        "grid_size": grid_size
    }

    # Variables
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = cp.Model()

    # Standard Sudoku constraints
    # Constraints on rows
    for row in grid:
        model += cp.AllDifferent(row)

    # Constraints on columns
    for col in grid.T:
        model += cp.AllDifferent(col)

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col])

    # TRUE greater-than constraints (between adjacent cells in specific positions)
    # Add some strategically placed greater-than constraints to make problem interesting
    # These will be part of the target model
    
    # Horizontal greater-than constraints (some cells > right neighbor)
    true_horizontal_gt = [
        (0, 0, 0, 1),  # grid[0,0] > grid[0,1]
        (1, 1, 1, 2),  # grid[1,1] > grid[1,2]
        (2, 2, 2, 3),  # grid[2,2] > grid[2,3]
        (3, 3, 3, 4),  # grid[3,3] > grid[3,4]
        (4, 4, 4, 5),  # grid[4,4] > grid[4,5]
    ]
    
    for r1, c1, r2, c2 in true_horizontal_gt:
        if r2 < grid_size and c2 < grid_size:
            model += (grid[r1, c1] > grid[r2, c2])
    
    # Vertical greater-than constraints (some cells > bottom neighbor)
    true_vertical_gt = [
        (0, 2, 1, 2),  # grid[0,2] > grid[1,2]
        (1, 3, 2, 3),  # grid[1,3] > grid[2,3]
        (2, 4, 3, 4),  # grid[2,4] > grid[3,4]
        (3, 5, 4, 5),  # grid[3,5] > grid[4,5]
        (4, 6, 5, 6),  # grid[4,6] > grid[5,6]
    ]
    
    for r1, c1, r2, c2 in true_vertical_gt:
        if r1 < grid_size and r2 < grid_size and c1 < grid_size and c2 < grid_size:
            model += (grid[r1, c1] > grid[r2, c2])

    # Save TRUE constraints for oracle
    C_T = list(set(toplevel_list(model.constraints)))

    # MOCK OVER-FITTED CONSTRAINTS (AllDifferent only)
    # These are AllDifferent constraints that might hold in 5 examples but NOT generally valid
    mock_constraints = []
    
    # # Mock 1: 8 cells scattered across different regions
    # if grid_size >= 9:
    #     mock_c1 = cp.AllDifferent([
    #         grid[0,0], grid[0,4], grid[1,2], grid[2,7], 
    #         grid[3,1], grid[4,5], grid[5,8], grid[6,3]
    #     ])
    #     mock_constraints.append(mock_c1)
    #     model += mock_c1
    
    # # Mock 2: 8 different scattered cells
    # if grid_size >= 9:
    #     mock_c2 = cp.AllDifferent([
    #         grid[0,1], grid[1,5], grid[2,3], grid[3,8], 
    #         grid[4,0], grid[5,4], grid[6,7], grid[7,2]
    #     ])
    #     mock_constraints.append(mock_c2)
    #     model += mock_c2
    
    # # Mock 3: 8 cells from various positions
    # if grid_size >= 9:
    #     mock_c3 = cp.AllDifferent([
    #         grid[0,3], grid[1,7], grid[2,1], grid[3,5], 
    #         grid[4,2], grid[5,6], grid[6,0], grid[7,4]
    #     ])
    #     mock_constraints.append(mock_c3)
    #     model += mock_c3
    
    # # Mock 4: 8 scattered cells (different pattern)
    # if grid_size >= 9:
    #     mock_c4 = cp.AllDifferent([
    #         grid[0,2], grid[1,6], grid[2,4], grid[3,0], 
    #         grid[4,8], grid[5,1], grid[6,5], grid[7,3]
    #     ])
    #     mock_constraints.append(mock_c4)
    #     model += mock_c4
    
    # # Mock 5: 8 cells from mixed regions
    # if grid_size >= 9:
    #     mock_c5 = cp.AllDifferent([
    #         grid[0,5], grid[1,0], grid[2,8], grid[3,2], 
    #         grid[4,6], grid[5,3], grid[6,1], grid[7,7]
    #     ])
    #     mock_constraints.append(mock_c5)
    #     model += mock_c5
    
    # # Mock 6: 8 cells final pattern
    # if grid_size >= 9:
    #     mock_c6 = cp.AllDifferent([
    #         grid[0,6], grid[1,3], grid[2,0], grid[3,7], 
    #         grid[4,1], grid[5,5], grid[6,2], grid[7,8]
    #     ])
    #     mock_constraints.append(mock_c6)
    #     model += mock_c6
    # Create the language
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
        name="sudoku_greater_than"
    )

    oracle = ConstraintOracle(C_T)
    
    # Return mock constraints so Phase 1 can use them as overfitted constraints
    return instance, oracle, mock_constraints

