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
    
    # MOCK OVER-FITTED CONSTRAINTS (AllDifferent only - Phase 1 filters others!)
    # These are designed to be consistent with 5 examples but NOT generally valid
    mock_constraints = []
    
    # Mock 1: AllDifferent on rows 0, 3, 6 first 3 cells each (non-contiguous rows)
    # Forces specific cells from non-adjacent rows to be unique
    if grid_size >= 9:
        non_adj_rows = []
        for row in [0, 3, 6]:
            for col in range(3):
                non_adj_rows.append(grid[row, col])
        mock_c1 = cp.AllDifferent(non_adj_rows)  # 9 cells
        mock_constraints.append(mock_c1)
        model += mock_c1
    
    # Mock 2: AllDifferent on columns 1, 4, 7 first 3 cells each (non-contiguous columns)
    if grid_size >= 9:
        non_adj_cols = []
        for col in [1, 4, 7]:
            for row in range(3):
                non_adj_cols.append(grid[row, col])
        mock_c2 = cp.AllDifferent(non_adj_cols)  # 9 cells
        mock_constraints.append(mock_c2)
        model += mock_c2
    
    # Mock 3: AllDifferent on diagonal elements (structural pattern)
    if grid_size >= 9:
        diagonal = [grid[i, i] for i in range(grid_size)]
        mock_c3 = cp.AllDifferent(diagonal)
        mock_constraints.append(mock_c3)
        model += mock_c3
    
    # Mock 4: AllDifferent on anti-diagonal elements
    if grid_size >= 9:
        anti_diag = [grid[i, grid_size - 1 - i] for i in range(grid_size)]
        mock_c4 = cp.AllDifferent(anti_diag)
        mock_constraints.append(mock_c4)
        model += mock_c4
    
    # Mock 5: AllDifferent on 3 cells from each corner region (12 cells total)
    if grid_size >= 9 and len(regions) >= 9:
        corner_cells = []
        # Take 3 cells from each corner region (0, 2, 6, 8)
        for region_idx in [0, 2, 6, 8]:
            if region_idx < len(regions):
                region_cells = [grid[r, c] for r, c in regions[region_idx][:3]]
                corner_cells.extend(region_cells)
        if len(corner_cells) >= 12:
            mock_c5 = cp.AllDifferent(corner_cells)
            mock_constraints.append(mock_c5)
            model += mock_c5
    
    # Mock 6: AllDifferent on border cells (top row + bottom row subset)
    if grid_size >= 9:
        border_cells = []
        # Top row first 6 cells
        border_cells.extend([grid[0, col] for col in range(6)])
        # Bottom row first 6 cells
        border_cells.extend([grid[8, col] for col in range(6)])
        mock_c6 = cp.AllDifferent(border_cells)  # 12 cells
        mock_constraints.append(mock_c6)
        model += mock_c6
    
    # Mock 7: AllDifferent on middle column + middle row intersection area
    if grid_size >= 9:
        middle_area = []
        # Middle column (column 4) - cells 2-6
        for row in range(2, 7):
            middle_area.append(grid[row, 4])
        # Middle row (row 4) - cells 2-6
        for col in range(2, 7):
            if col != 4:  # Avoid duplicate
                middle_area.append(grid[4, col])
        mock_c7 = cp.AllDifferent(middle_area)  # 9 cells
        mock_constraints.append(mock_c7)
        model += mock_c7
    
    # Mock 8: AllDifferent on L-shaped pattern (rows 0-2 col 0, rows 0 cols 1-3)
    if grid_size >= 9:
        l_shape = []
        # Vertical part: column 0, rows 0-2
        for row in range(3):
            l_shape.append(grid[row, 0])
        # Horizontal part: row 0, columns 1-5
        for col in range(1, 6):
            l_shape.append(grid[0, col])
        mock_c8 = cp.AllDifferent(l_shape)  # 8 cells
        mock_constraints.append(mock_c8)
        model += mock_c8
    
    # Mock 9: AllDifferent on corners + center (star pattern)
    if grid_size >= 9:
        star_pattern = [
            grid[0, 0], grid[0, grid_size-1],  # Top corners
            grid[grid_size-1, 0], grid[grid_size-1, grid_size-1],  # Bottom corners
            grid[grid_size//2, grid_size//2],  # Center
            grid[0, grid_size//2], grid[grid_size-1, grid_size//2],  # Middle edges
            grid[grid_size//2, 0], grid[grid_size//2, grid_size-1]  # Middle edges
        ]
        mock_c9 = cp.AllDifferent(star_pattern)
        mock_constraints.append(mock_c9)
        model += mock_c9
    
    # Mock 10: AllDifferent on Z-pattern (top row subset + diagonal + bottom row subset)
    if grid_size >= 9:
        z_pattern = []
        # Top row: columns 1-3
        for col in range(1, 4):
            z_pattern.append(grid[0, col])
        # Diagonal: 3 cells
        for i in range(3, 6):
            z_pattern.append(grid[i, i])
        # Bottom row: columns 5-7
        for col in range(5, 8):
            z_pattern.append(grid[8, col])
        mock_c10 = cp.AllDifferent(z_pattern)  # 9 cells
        mock_constraints.append(mock_c10)
        model += mock_c10
    
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

