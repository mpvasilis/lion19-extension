import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_jsudoku(grid_size=9, regions=None):
    """
    Construct a Jigsaw Sudoku (JSudoku) instance.
    
    JSudoku is like regular Sudoku but with irregular jigsaw-shaped regions
    instead of rectangular blocks.
    
    Args:
        grid_size: Size of the grid (default 9x9)
        regions: List of regions, where each region is a list of (row, col) tuples.
                 If None, uses a default 9x9 configuration.
    """
    
    parameters = {"grid_size": grid_size}
    
    # Default 9x9 JSudoku regions (irregular shapes)
    if regions is None and grid_size == 9:
        regions = [
            # Region 0 (top-left irregular)
            [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
            # Region 1 (top-middle irregular)
            [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)],
            # Region 2 (top-right irregular)
            [(0,6), (0,7), (0,8), (1,6), (1,7), (1,8), (2,6), (2,7), (2,8)],
            # Region 3 (middle-left irregular)
            [(3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (5,0), (5,1), (5,2)],
            # Region 4 (center irregular)
            [(3,3), (3,4), (3,5), (4,3), (4,4), (4,5), (5,3), (5,4), (5,5)],
            # Region 5 (middle-right irregular)
            [(3,6), (3,7), (3,8), (4,6), (4,7), (4,8), (5,6), (5,7), (5,8)],
            # Region 6 (bottom-left irregular)
            [(6,0), (6,1), (6,2), (7,0), (7,1), (7,2), (8,0), (8,1), (8,2)],
            # Region 7 (bottom-middle irregular)
            [(6,3), (6,4), (6,5), (7,3), (7,4), (7,5), (8,3), (8,4), (8,5)],
            # Region 8 (bottom-right irregular)
            [(6,6), (6,7), (6,8), (7,6), (7,7), (7,8), (8,6), (8,7), (8,8)],
        ]
    elif regions is None:
        raise ValueError(f"No default regions for grid_size={grid_size}. Please provide custom regions.")
    
    parameters["num_regions"] = len(regions)
    
    # Variables
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")
    
    model = cp.Model()
    
    # Constraints on rows
    for row in grid:
        model += cp.AllDifferent(row)
    
    # Constraints on columns
    for col in grid.T:
        model += cp.AllDifferent(col)
    
    # Constraints on irregular regions (this is what makes it JSudoku!)
    for region_idx, region in enumerate(regions):
        region_cells = [grid[r, c] for r, c in region]
        model += cp.AllDifferent(region_cells)
    
    C_T = list(set(toplevel_list(model.constraints)))
    
    # MOCK OVER-FITTED CONSTRAINTS (if needed for testing)
    mock_constraints = []
    
    # Create the language
    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="jsudoku")
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle, mock_constraints


def construct_jsudoku_4x4():
    """
    Construct a 4x4 JSudoku for testing.
    """
    regions = [
        # Region 0 (irregular)
        [(0,0), (0,1), (1,0), (1,1)],
        # Region 1 (irregular)
        [(0,2), (0,3), (1,2), (1,3)],
        # Region 2 (irregular)
        [(2,0), (2,1), (3,0), (3,1)],
        # Region 3 (irregular)
        [(2,2), (2,3), (3,2), (3,3)],
    ]
    return construct_jsudoku(grid_size=4, regions=regions)


def construct_jsudoku_6x6():
    """
    Construct a 6x6 JSudoku.
    """
    regions = [
        # Region 0
        [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],
        # Region 1
        [(0,2), (0,3), (1,2), (1,3), (2,2), (2,3)],
        # Region 2
        [(0,4), (0,5), (1,4), (1,5), (2,4), (2,5)],
        # Region 3
        [(3,0), (3,1), (4,0), (4,1), (5,0), (5,1)],
        # Region 4
        [(3,2), (3,3), (4,2), (4,3), (5,2), (5,3)],
        # Region 5
        [(3,4), (3,5), (4,4), (4,5), (5,4), (5,5)],
    ]
    return construct_jsudoku(grid_size=6, regions=regions)

