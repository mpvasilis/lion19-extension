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
        # model += cp.AllDifferent(row).decompose()

    for col in grid.T:
        model += cp.AllDifferent(col)
        # model += cp.AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col])
            # model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col]).decompose()

    # MOCK OVER-FITTED CONSTRAINTS (will be consistent with 5 examples but NOT generally valid)
    # These should be detected and refined in Phase 2
    mock_constraints = []

    # Mock 1: Main diagonal AllDifferent (too restrictive - not part of standard Sudoku)
    main_diagonal = [grid[i, i] for i in range(grid_size)]
    mock_constraints.append(cp.AllDifferent(main_diagonal))

    # Mock 2: Anti-diagonal AllDifferent (too restrictive)
    anti_diagonal = [grid[i, grid_size - 1 - i] for i in range(grid_size)]
    mock_constraints.append(cp.AllDifferent(anti_diagonal))

    # Add mock constraints to model (they will be learned passively but should be refuted)
    for mock_c in mock_constraints:
        model += mock_c

    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="sudoku")

    oracle = ConstraintOracle(C_T)

    return instance, oracle