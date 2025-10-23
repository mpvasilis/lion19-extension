import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_latin_square(n=9):
    """
    Construct a Latin Square instance.
    
    A Latin Square is an n×n grid filled with n different symbols such that:
    - Each symbol occurs exactly once in each row (AllDifferent)
    - Each symbol occurs exactly once in each column (AllDifferent)
    
    This is like Sudoku but WITHOUT the block constraints.
    
    Args:
        n: Size of the square (default 9)
    """
    
    parameters = {"n": n}
    
    # Variables
    grid = cp.intvar(1, n, shape=(n, n), name="grid")
    
    model = cp.Model()
    
    # Constraints on rows
    for row in grid:
        model += cp.AllDifferent(row)
    
    # Constraints on columns
    for col in grid.T:
        model += cp.AllDifferent(col)
    
    C_T = list(set(toplevel_list(model.constraints)))
    
    # MOCK OVER-FITTED CONSTRAINTS (optional for testing Phase 2)
    mock_constraints = []
    
    # Mock 1: Diagonal constraint (might hold in examples but not generally required)
    # if n >= 3:
    #     mock_c1 = cp.AllDifferent([grid[i,i] for i in range(n)])
    #     mock_constraints.append(mock_c1)
    #     model += mock_c1
    
    # Mock 2: Anti-diagonal constraint
    # if n >= 3:
    #     mock_c2 = cp.AllDifferent([grid[i,n-1-i] for i in range(n)])
    #     mock_constraints.append(mock_c2)
    #     model += mock_c2
    
    # Mock 3: Center cross pattern (for odd n)
    # if n >= 5 and n % 2 == 1:
    #     center = n // 2
    #     cross_cells = [grid[center, i] for i in range(n)]
    #     mock_c3 = cp.AllDifferent(cross_cells)
    #     mock_constraints.append(mock_c3)
    #     model += mock_c3
    
    # Create the language
    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="latin_square")
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle, mock_constraints


def construct_latin_square_4x4():
    """Construct a 4×4 Latin Square for quick testing."""
    return construct_latin_square(n=4)


def construct_latin_square_6x6():
    """Construct a 6×6 Latin Square."""
    return construct_latin_square(n=6)


def construct_latin_square_9x9():
    """Construct a 9×9 Latin Square."""
    return construct_latin_square(n=9)

