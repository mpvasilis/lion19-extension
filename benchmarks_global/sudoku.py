import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_sudoku(block_size_row, block_size_col, grid_size):

    parameters = {"block_size_row": block_size_row, "block_size_col": block_size_col, "grid_size": grid_size}

    # Variables
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = cp.Model()

    # Constraints on rows and columns
    for row in grid:
        model += cp.AllDifferent(row)

    for col in grid.T:
        model += cp.AllDifferent(col)

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col])


    C_T = list(set(toplevel_list(model.constraints)))

    # MOCK OVER-FITTED CONSTRAINTS (AllDifferent only)
    # These are AllDifferent constraints that might hold in 5 examples but NOT generally valid
    mock_constraints = []
    
    # # Mock 1: Diagonal pattern that might hold in examples but not generally
    # if grid_size >= 3:
    #     mock_c1 = cp.AllDifferent([grid[0,0], grid[1,1], grid[2,2]])
    #     mock_constraints.append(mock_c1)
    #     model += mock_c1
    
    # # Mock 2: Corner cells pattern (might hold accidentally)
    # if grid_size >= 4:
    #     corner_cells = [grid[0,0], grid[0,grid_size-1], grid[grid_size-1,0], grid[grid_size-1,grid_size-1]]
    #     mock_c4 = cp.AllDifferent(corner_cells)
    #     mock_constraints.append(mock_c4)
    #     model += mock_c4
    
    # # Mock 3: Anti-diagonal pattern
    # if grid_size >= 3:
    #     mock_c5 = cp.AllDifferent([grid[0,grid_size-1], grid[1,grid_size-2], grid[2,grid_size-3]])
    #     mock_constraints.append(mock_c5)
    #     model += mock_c5
    
    # # Mock 4: Center cross pattern (might hold accidentally)
    # if grid_size >= 5:
    #     center = grid_size // 2
    #     mock_c6 = cp.AllDifferent([grid[center, center], grid[center-1, center], 
    #                                grid[center+1, center], grid[center, center-1], 
    #                                grid[center, center+1]])
    #     mock_constraints.append(mock_c6)
    #     model += mock_c6
    
    # # Mock 5: Edge pattern
    # if grid_size >= 5:
    #     mock_c7 = cp.AllDifferent([grid[0,0], grid[0,4], grid[4,0], grid[4,4], grid[2,2]])
    #     mock_constraints.append(mock_c7)
    #     model += mock_c7
    
    # # Mock 6: Random pattern that might hold in examples
    # if grid_size >= 6:
    #     mock_c8 = cp.AllDifferent([grid[1,1], grid[3,3], grid[5,5]])
    #     mock_constraints.append(mock_c8)
    #     model += mock_c8

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="sudoku")

    oracle = ConstraintOracle(C_T)

    # Return mock constraints so Phase 1 can use them as overfitted constraints
    return instance, oracle, mock_constraints