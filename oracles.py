import cpmpy as cp
import numpy as np
from itertools import combinations, product

# =============================================================================
# Models mainly based on AllDifferent constraints
# =============================================================================

def construct_quasigroup_existence(n=5):
    """
    prob003 Quasigroup Existence
    A Latin square model: an n×n table where each row and each column is all–different.
    """
    X = cp.intvar(1, n, shape=(n, n), name="var")
    model = cp.Model()
    for i in range(n):
        model += cp.AllDifferent(X[i, :])
    for j in range(n):
        model += cp.AllDifferent(X[:, j])
    return X, model


def construct_all_interval_series(n=8):
    """
    prob007 All-Interval Series (Matrix version)
    A 1×n matrix is used to represent the permutation of 0..n-1.
    The differences are modeled as a 1×(n-1) matrix.
    """
    # Represent the sequence as a 1×n matrix.
    s = cp.intvar(0, n-1, shape=(1, n), name="var")
    # Differences as a 1×(n-1) matrix.
    d = cp.intvar(1, n-1, shape=(1, n-1), name="d")
    model = cp.Model(cp.AllDifferent(s[0, :]), cp.AllDifferent(d[0, :]))
    for i in range(n-1):
        model += (d[0, i] == abs(s[0, i+1] - s[0, i]))
    model += (s[0, 0] == 0)  # symmetry breaking
    return s, model

def construct_nqueens(n=8):
    """
    prob054 N-Queens (Matrix boolean model)
    Instead of a vector, a boolean n×n matrix Q is used.
    Q[i,j] = 1 if a queen is placed at row i, column j.
    Constraints:
      - Exactly one queen per row and per column.
      - At most one queen per diagonal.
    """
    Q = cp.boolvar(shape=(n, n), name="var")
    model = cp.Model()
    # Row and column constraints.
    for i in range(n):
        model += (sum(Q[i, :]) == 1)
    for j in range(n):
        model += (sum(Q[:, j]) == 1)
    # Main diagonals.
    for d in range(-n+1, n):
        diag_vars = [Q[i, i-d] for i in range(n) if 0 <= i-d < n]
        if diag_vars:
            model += (sum(diag_vars) <= 1)
    # Anti-diagonals.
    for s in range(2*n - 1):
        diag_vars = [Q[i, s-i] for i in range(n) if 0 <= s-i < n]
        if diag_vars:
            model += (sum(diag_vars) <= 1)
    return Q, model


def construct_costas_arrays(n=6):
    """
    prob076 Costas Arrays (Matrix version)
    The permutation is represented as a 1×n matrix.
    """
    x = cp.intvar(0, n-1, shape=(1, n), name="var")
    model = cp.Model(cp.AllDifferent(x[0, :]))
    disp = []
    M = n  # scaling factor
    for i in range(n-1):
        for j in range(i+1, n):
            disp.append((j - i)*M + (x[0, j] - x[0, i]))
    model += cp.AllDifferent(disp)
    return x, model


def construct_killer_sudoku():
    """
    prob057 Killer Sudoku
    A 9x9 Sudoku with standard row, column, and block all–different constraints plus a couple
    of additional "cage" constraints that (here) also require all entries to be different.
    (In a full Killer Sudoku model, cages also have prescribed sums.)
    """
    grid = cp.intvar(1, 9, shape=(9,9), name="var")
    model = cp.Model()
    # Standard Sudoku constraints:
    for i in range(9):
        model += cp.AllDifferent(grid[i, :])
        model += cp.AllDifferent(grid[:, i])
    for i in range(0,9,3):
        for j in range(0,9,3):
            block = grid[i:i+3, j:j+3].flatten()
            model += cp.AllDifferent(block)
    # Two sample cages:
    cage1 = [grid[0,0], grid[0,1], grid[1,0]]
    model += (sum(cage1) == 15)
    model += cp.AllDifferent(cage1)
    cage2 = [grid[4,4], grid[4,5], grid[5,4], grid[5,5]]
    model += (sum(cage2) == 22)
    model += cp.AllDifferent(cage2)
    return grid, model


def construct_quasigroup_completion(n=5, preassigned=None):
    """
    prob067 Quasigroup Completion
    A Latin square (n×n) with some cells pre-assigned.
    preassigned should be a dict mapping (row,col) -> value.
    """
    X = cp.intvar(1, n, shape=(n, n), name="var")
    model = cp.Model()
    for i in range(n):
        model += cp.AllDifferent(X[i, :])
    for j in range(n):
        model += cp.AllDifferent(X[:, j])
    if preassigned is not None:
        for (i, j), val in preassigned.items():
            model += (X[i, j] == val)
    return X, model


def construct_costas_arrays(n=6):
    """
    prob076 Costas Arrays
    Model a Costas array as a permutation of 0..n-1 such that all “encoded” differences are distinct.
    Here we encode each pair’s displacement as: (j-i)*n + (x[j]-x[i]).
    """
    x = cp.intvar(0, n-1, shape=n, name="var")
    model = cp.Model(cp.AllDifferent(x))
    disp = []
    M = n  # scaling factor
    for i in range(n-1):
        for j in range(i+1, n):
            d = (j - i) * M + (x[j] - x[i])
            disp.append(d)
    model += cp.AllDifferent(disp)
    return x, model


def construct_nqueens_completion(n=8, preassigned=None):
    """
    prob079 n-Queens Completion (Matrix boolean model)
    Uses a boolean n×n matrix Q with one queen per row.
    For rows with a preassigned queen (dict mapping row → column), Q[i, col] is fixed to 1.
    Diagonal constraints are imposed as in the standard n-queens.
    """
    Q = cp.boolvar(shape=(n, n), name="var")
    model = cp.Model()
    # One queen per row.
    for i in range(n):
        model += (sum(Q[i, :]) == 1)
    # One queen per column.
    for j in range(n):
        model += (sum(Q[:, j]) == 1)
    # Diagonal constraints.
    for d in range(-n+1, n):
        diag_vars = [Q[i, i-d] for i in range(n) if 0 <= i-d < n]
        if diag_vars:
            model += (sum(diag_vars) <= 1)
    for s in range(2*n - 1):
        diag_vars = [Q[i, s-i] for i in range(n) if 0 <= s-i < n]
        if diag_vars:
            model += (sum(diag_vars) <= 1)
    if preassigned is not None:
        for i, col in preassigned.items():
            model += (Q[i, col] == 1)
    return Q, model


def construct_blocked_nqueens(n=8, blocked_positions=None):
    """
    prob080 Blocked n-Queens Problem (Matrix boolean model)
    Similar to standard n-Queens but with forbidden cells.
    For each (i, j) in blocked_positions, Q[i,j] is forced to 0.
    """
    Q = cp.boolvar(shape=(n, n), name="var")
    model = cp.Model()
    # One queen per row.
    for i in range(n):
        model += (sum(Q[i, :]) == 1)
    # One queen per column.
    for j in range(n):
        model += (sum(Q[:, j]) == 1)
    # Diagonals constraints.
    for d in range(-n+1, n):
        diag_vars = [Q[i, i-d] for i in range(n) if 0 <= i-d < n]
        if diag_vars:
            model += (sum(diag_vars) <= 1)
    for s in range(2*n - 1):
        diag_vars = [Q[i, s-i] for i in range(n) if 0 <= s-i < n]
        if diag_vars:
            model += (sum(diag_vars) <= 1)
    if blocked_positions is not None:
        for (i, j) in blocked_positions:
            model += (Q[i, j] == 0)
    return Q, model


def construct_peaceably_coexisting_armies_of_queens(n=8):
    """
    prob110 Peaceably Co-existing Armies of Queens
    Place n queens for each of two armies on an n×n board such that no queen of one army attacks
    a queen of the other. Here each army is modeled as a permutation (one queen per row).
    """
    white = cp.intvar(0, n-1, shape=n, name="white")
    black = cp.intvar(0, n-1, shape=n, name="black")
    model = cp.Model()
    # Each army: one queen per row, no conflicts internally.
    model += cp.AllDifferent(white)
    model += cp.AllDifferent(white + np.arange(n))
    model += cp.AllDifferent(white - np.arange(n))
    model += cp.AllDifferent(black)
    model += cp.AllDifferent(black + np.arange(n))
    model += cp.AllDifferent(black - np.arange(n))
    # Additional constraint: enemy queens do not attack.
    for i in range(n):
        model += (white[i] != black[i])
    for i in range(n):
        for j in range(n):
            if i != j:
                model += (abs(white[i] - black[j]) != abs(i - j))
    return (white, black), model


def construct_equidistant_frequency_permutation_arrays(k=3):
    """
    prob055 Equidistant Frequency Permutation Arrays (Matrix version)
    The sequence is modeled as a 1×(2*k) matrix.
    """
    n = 2 * k
    seq = cp.intvar(1, k, shape=(1, n), name="var")
    model = cp.Model()
    # Each number 1..k appears exactly twice.
    for i in range(1, k+1):
        model += (sum(seq[0, :] == i) == 2)
    # Auxiliary positions (still scalar variables).
    pos = [cp.intvar(0, n-1, name=f"pos_{i}") for i in range(1, k+1)]
    for i in range(1, k+1):
        model += (pos[i-1] + i + 1 < n)
        model += (cp.Element(seq[0, :], pos[i-1]) == i)
        model += (cp.Element(seq[0, :], pos[i-1] + i + 1) == i)
    model += cp.AllDifferent(pos)
    return seq, model

# =============================================================================
# Models mainly based on Sum constraints
# =============================================================================

def construct_magic_square(n=3):
    """
    prob019 Magic Squares and Sequences
    A magic square of order n with numbers 1..n^2 and all rows, columns, and both diagonals
    summing to the magic constant.
    """
    M = cp.intvar(1, n*n, shape=(n, n), name="var")
    magic_sum = n * (n*n + 1) // 2
    model = cp.Model()
    model += cp.AllDifferent(M)
    for i in range(n):
        model += (sum(M[i, :]) == magic_sum)
    for j in range(n):
        model += (sum(M[:, j]) == magic_sum)
    model += (sum(M[i, i] for i in range(n)) == magic_sum)
    model += (sum(M[i, n-1-i] for i in range(n)) == magic_sum)
    return M, model


def construct_darts_tournament():
    """
    prob020 Darts Tournament
    A simplified model in which 4 players score points over 3 rounds,
    and each round’s total score must equal a target value.
    """
    scores = cp.intvar(0, 20, shape=(4, 3), name="var")
    target = 30
    model = cp.Model()
    for j in range(3):
        model += (sum(scores[i, j] for i in range(4)) == target)
    return scores, model


def construct_magic_hexagon():
    """
    prob023 Magic Hexagon
    A simplified version of a magic hexagon of order 3.
    (The complete magic hexagon has a special geometric layout;
    here we use 19 variables and add a few sum constraints as an illustration.)
    """
    cells = cp.intvar(1, 19, shape=19, name="var")
    magic_sum = 38
    model = cp.Model(cp.AllDifferent(cells))
    # For illustration, we select a few “lines” (not the full set for a proper magic hexagon).
    lines = [
        [cells[0], cells[1], cells[2], cells[3]],
        [cells[4], cells[5], cells[6], cells[7]],
        [cells[8], cells[9], cells[10], cells[11]],
        [cells[12], cells[13], cells[14], cells[15]],
        [cells[2], cells[6], cells[10], cells[14]],
        [cells[3], cells[7], cells[11], cells[15]]
    ]
    for line in lines:
        model += (sum(line) == magic_sum)
    return cells, model


def construct_steel_mill_slab_design():
    """
    prob038 Steel Mill Slab Design
    A simplified model: assign 5 orders (with given lengths) to 3 slabs
    so that the sum of orders assigned to any slab does not exceed the slab capacity.
    """
    orders = [3, 4, 5, 2, 6]
    n_orders = len(orders)
    n_slabs = 3
    # x[i] is the slab (0..n_slabs-1) to which order i is assigned.
    x = cp.intvar(0, n_slabs-1, shape=n_orders, name="var")
    model = cp.Model()
    slab_capacity = 10
    for slab in range(n_slabs):
        # Here we “count” the total order size on slab by summing orders[i] for those i with x[i]==slab.
        # (Using an indicator expression: (x[i]==slab) evaluates to 1 if true, 0 otherwise.)
        model += (sum(orders[i] * (x[i] == slab) for i in range(n_orders)) <= slab_capacity)
    return x, model


def construct_nfractions():
    """
    prob041 n-Fractions Puzzle
    A simplified (and non-linear) model for the n-fractions puzzle.
    (Note: CPMPy’s core solvers handle integer constraints;
    non-linear division may require further transformation.)
    """
    digits = cp.intvar(1, 9, shape=9, name="var")
    model = cp.Model(cp.AllDifferent(digits))
    # Partition the 9 digits into three fractions:
    #   f1 = digits[0] / (digits[1]*digits[2])
    #   f2 = digits[3] / (digits[4]*digits[5])
    #   f3 = digits[6] / (digits[7]*digits[8])
    # and require f1 + f2 + f3 == 1.
    # (This is a symbolic illustration; in practice, one would need to clear denominators.)
    f1 = digits[0] / (digits[1]*digits[2])
    f2 = digits[3] / (digits[4]*digits[5])
    f3 = digits[6] / (digits[7]*digits[8])
    model += (f1 + f2 + f3 == 1)
    return digits, model


def construct_number_partitioning(numbers=[3, 1, 4, 2, 2]):
    """
    prob049 Number Partitioning
    Partition the given list of numbers into two subsets with equal total sum.
    """
    n = len(numbers)
    x = cp.intvar(0, 1, shape=n, name="var")  # x[i]==1 means the number goes to subset 1
    total = sum(numbers)
    model = cp.Model()
    model += (sum(x[i] * numbers[i] for i in range(n)) == total / 2)
    return x, model


def construct_discrete_lot_sizing():
    """
    prob058 Discrete Lot Sizing Problem
    A simplified production planning model over 5 periods.
    Production and inventory balance constraints are modeled via sum constraints.
    """
    periods = 5
    demand = [5, 7, 3, 8, 6]
    production = cp.intvar(0, 20, shape=periods, name="var")
    inventory = cp.intvar(0, 50, shape=periods, name="inv")
    model = cp.Model()
    model += (production[0] - demand[0] == inventory[0])
    for t in range(1, periods):
        model += (production[t] + inventory[t-1] - demand[t] == inventory[t])
    return (production, inventory), model


def construct_optimal_financial_portfolio():
    """
    prob065 Optimal Financial Portfolio Design
    A simplified portfolio selection problem modeled as a knapsack-like problem.
    """
    n_assets = 5
    costs   = [3, 4, 2, 5, 7]
    returns = [5, 6, 4, 8, 9]
    budget = 10
    x = cp.intvar(0, 1, shape=n_assets, name="var")
    model = cp.Model()
    model += (sum(costs[i] * x[i] for i in range(n_assets)) <= budget)
    # For illustration, require a minimum total return.
    model += (sum(returns[i] * x[i] for i in range(n_assets)) >= 15)
    return x, model


def construct_transshipment():
    """
    prob083 Transshipment Problem
    A very simplified network flow on 4 nodes with a few arcs.
    Supply/demand is specified at each node.
    """
    nodes = 4
    arcs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    supply = [10, 0, 0, -10]
    flow = {arc: cp.intvar(0, 10, name=f"f_{arc[0]}_{arc[1]}") for arc in arcs}
    model = cp.Model()
    for i in range(nodes):
        inflow = sum(flow[(j, i)] for (j, i) in arcs if (j, i) in flow)
        outflow = sum(flow[(i, j)] for (i, j) in arcs if (i, j) in flow)
        model += (inflow - outflow == supply[i])
    return flow, model


def construct_cvrp():
    """
    prob086 Capacitated Vehicle Routing Problem (CVRP)
    A very simplified CVRP where orders (with given sizes) are assigned to vehicles
    with given capacities.
    """
    orders = [2, 3, 4, 1]
    vehicle_capacity = [5, 5]
    n_orders = len(orders)
    n_vehicles = len(vehicle_capacity)
    x = cp.intvar(0, n_vehicles-1, shape=n_orders, name="var")
    model = cp.Model()
    for v in range(n_vehicles):
        model += (sum(orders[i] * (x[i] == v) for i in range(n_orders)) <= vehicle_capacity[v])
    return x, model


def construct_knapsack():
    """
    prob133 Knapsack Problem
    A standard knapsack model: select items subject to a weight limit.
    """
    weights = [2, 3, 4, 5]
    values  = [3, 4, 5, 6]
    capacity = 5
    n_items = len(weights)
    x = cp.intvar(0, 1, shape=n_items, name="var")
    model = cp.Model()
    model += (sum(weights[i] * x[i] for i in range(n_items)) <= capacity)
    # For illustration, require a minimum total value.
    model += (sum(values[i] * x[i] for i in range(n_items)) >= 7)
    return x, model

# =============================================================================
# Models mainly based on COUNT constraints
# =============================================================================

def construct_car_sequencing():
    """
    prob001 Car Sequencing
    In any window of 3 consecutive cars, at most 1 car may have a given option.
    """
    n_cars = 10
    option = cp.intvar(0, 1, shape=n_cars, name="var")
    model = cp.Model()
    window_size = 3
    for i in range(n_cars - window_size + 1):
        model += (sum(option[i:i+window_size]) <= 1)
    return option, model


def construct_nonogram():
    """
    prob012 Nonogram
    A simplified nonogram puzzle: a binary matrix with fixed row and column counts.
    """
    rows, cols = 5, 5
    grid = cp.intvar(0, 1, shape=(rows, cols), name="var")
    row_counts = [2, 3, 1, 2, 2]
    col_counts = [2, 2, 2, 2, 2]
    model = cp.Model()
    for i in range(rows):
        model += (sum(grid[i, :]) == row_counts[i])
    for j in range(cols):
        model += (sum(grid[:, j]) == col_counts[j])
    return grid, model


def construct_solitaire_battleships():
    """
    prob014 Solitaire Battleships
    A simplified battleships puzzle where the numbers given on rows/columns indicate
    how many ship parts are present.
    """
    rows, cols = 6, 6
    grid = cp.intvar(0, 1, shape=(rows, cols), name="var")
    row_counts = [1, 2, 1, 2, 1, 2]
    col_counts = [2, 1, 2, 1, 2, 1]
    model = cp.Model()
    for i in range(rows):
        model += (sum(grid[i, :]) == row_counts[i])
    for j in range(cols):
        model += (sum(grid[:, j]) == col_counts[j])
    return grid, model


def construct_bacp():
    """
    prob030 Balanced Academic Curriculum Problem (BACP)
    Assign n_courses to n_semesters such that (i) each semester gets a balanced number of courses,
    and (ii) the total credits in each semester lie between given bounds.
    """
    n_courses = 8
    n_semesters = 4
    credits = [3, 4, 2, 3, 5, 2, 4, 3]
    # x[i] is the semester (1..n_semesters) for course i.
    x = cp.intvar(1, n_semesters, shape=n_courses, name="var")
    model = cp.Model()
    min_courses = 2
    max_courses = 3
    for s in range(1, n_semesters+1):
        model += (sum(x == s) >= min_courses)
        model += (sum(x == s) <= max_courses)
        model += (sum(credits[i] * (x[i] == s) for i in range(n_courses)) >= 5)
        model += (sum(credits[i] * (x[i] == s) for i in range(n_courses)) <= 10)
    return x, model


def construct_covering_array():
    """
    prob045 The Covering Array Problem
    A simplified covering array for 3 factors, 2 levels each, and 4 rows.
    For every pair of factors, every combination of levels must appear in at least one row.
    """
    factors = 3
    levels = 2
    rows = 4
    CA = cp.intvar(0, levels-1, shape=(rows, factors), name="var")
    model = cp.Model()
    # For each pair of factors, cover all level combinations.
    for pair in combinations(range(factors), 2):
        for combo in product(range(levels), repeat=2):
            # Create a Boolean for each row: true if the row has the desired combination.
            bools = []
            for i in range(rows):
                b = cp.boolvar(name=f"cov_{pair}_{combo}_{i}")
                model += (b == ((CA[i, pair[0]] == combo[0]) & (CA[i, pair[1]] == combo[1])))
                bools.append(b)
            model += (sum(bools) >= 1)
    return CA, model


def construct_balanced_nursing_workload():
    """
    prob069 Balanced Nursing Workload Problem
    Assign 10 shifts to 3 nurses such that the number of shifts per nurse is as equal as possible.
    """
    n_shifts = 10
    n_nurses = 3
    x = cp.intvar(0, n_nurses-1, shape=n_shifts, name="var")
    model = cp.Model()
    counts = [sum(x == i) for i in range(n_nurses)]
    for i in range(n_nurses):
        for j in range(i+1, n_nurses):
            model += (abs(counts[i] - counts[j]) <= 1)
    return x, model


def construct_rotating_rostering():
    """
    prob087 Rotating Rostering Problem
    A simplified rostering: assign 7 days × 3 shifts to 4 workers, ensuring a balanced workload.
    """
    days, shifts, n_workers = 7, 3, 4
    roster = cp.intvar(0, n_workers-1, shape=(days, shifts), name="var")
    model = cp.Model()
    total_shifts = days * shifts
    avg = total_shifts // n_workers
    for w in range(n_workers):
        model += (sum(roster == w) >= avg)
        model += (sum(roster == w) <= avg + 1)
    return roster, model


def construct_bus_driver_scheduling():
    """
    prob022 Bus Driver Scheduling
    A simplified scheduling: for 7 days, exactly 2 out of 5 drivers work each day,
    and the total number of shifts assigned to any two drivers differs by at most 1.
    """
    days, n_drivers = 7, 5
    schedule = cp.intvar(0, 1, shape=(days, n_drivers), name="var")
    model = cp.Model()
    for d in range(days):
        model += (sum(schedule[d, :]) == 2)
    totals = [sum(schedule[:, i]) for i in range(n_drivers)]
    for i in range(n_drivers):
        for j in range(i+1, n_drivers):
            model += (abs(totals[i] - totals[j]) <= 1)
    return schedule, model

# =============================================================================
# Dictionary of models for easy lookup
# =============================================================================

# ------------------------------------------------------------------
# 1. VM to PM Allocation Problem
# ------------------------------------------------------------------
def construct_vm_pm_allocation():
    # Parameters
    num_vms = 10
    num_pms = 5
    # CPU requirements for each VM
    reqs = [2, 3, 4, 2, 3, 5, 1, 2, 3, 4]
    # CPU capacity for each PM (assume all PMs have equal capacity)
    capacities = [10, 10, 10, 10, 10]
    
    # Decision variables:
    # assign[i] indicates the PM (0..num_pms-1) to which VM i is allocated.
    assign = cp.intvar(0, num_pms-1, shape=num_vms, name="assign")
    # used[j] is 1 (True) if PM j is used, 0 (False) otherwise.
    used = cp.boolvar(shape=num_pms, name="used")
    
    # Create an indicator matrix: b[i,j] is 1 if assign[i] == j, 0 otherwise.
    b = cp.intvar(0, 1, shape=(num_vms, num_pms), name="b")
    
    model = cp.Model()
    
    # Link each b[i,j] with the assignment of VM i to PM j.
    # The reification constraint ensures: b[i,j] == 1  <==>  (assign[i] == j)
    for i in range(num_vms):
        for j in range(num_pms):
            # This constraint enforces that b[i,j] takes the same truth value as (assign[i] == j)
            model += (b[i, j] == (assign[i] == j))
    
    # For each PM, the total CPU requirement of VMs assigned to it must not exceed its capacity.
    for j in range(num_pms):
        model += (sum(reqs[i] * b[i, j] for i in range(num_vms)) <= capacities[j])
    
    # Link the 'used' variable with the assignments:
    # If any VM is assigned to PM j (i.e. some b[i,j] == 1) then used[j] must be 1,
    # and if used[j] is 0 then all b[i,j] must be 0.
    for j in range(num_pms):
        # If used[j]==0 then sum(b[:,j])==0; if used[j]==1 then sum(b[:,j]) can be positive.
        model += (sum(b[i, j] for i in range(num_vms)) <= num_vms * used[j])
        model += (sum(b[i, j] for i in range(num_vms)) >= used[j])
    
    # Objective: minimize the number of used PMs.
    model.minimize(sum(used))
    
    return (assign, used, b), model

# ------------------------------------------------------------------
# 2. Warehouse Location Problem
# ------------------------------------------------------------------
def construct_warehouse_location():
    # Parameters
    n_warehouses = 4
    n_customers = 8
    
    # Fixed opening costs for each warehouse.
    opening_cost = [100, 120, 90, 110]
    
    # Assignment cost matrix: cost for serving customer i from warehouse j.
    # (Here, we generate random costs between 10 and 50.)
    np.random.seed(0)  # for reproducibility
    assignment_cost = np.random.randint(10, 51, size=(n_customers, n_warehouses))
    
    # Decision variables:
    # x[i,j] = 1 if customer i is served by warehouse j, 0 otherwise.
    x = cp.intvar(0, 1, shape=(n_customers, n_warehouses), name="var")
    # y[j] = 1 if warehouse j is open, 0 otherwise.
    y = cp.boolvar(shape=n_warehouses, name="y")
    
    model = cp.Model()
    
    # Each customer must be served by exactly one warehouse.
    for i in range(n_customers):
        model += (sum(x[i, j] for j in range(n_warehouses)) == 1)
    
    # A customer can only be assigned to an open warehouse.
    for i in range(n_customers):
        for j in range(n_warehouses):
            model += (x[i, j] <= y[j])
    
    # # Objective: minimize total cost (opening costs + assignment costs).
    # total_cost = sum(opening_cost[j] * y[j] for j in range(n_warehouses)) \
    #              + sum(assignment_cost[i, j] * x[i, j] for i in range(n_customers) for j in range(n_warehouses))
    # model.minimize(total_cost)
    
    return (x, y, assignment_cost, opening_cost), model

CSP_LIBS = {
    # AllDifferent-based
    "prob003_quasigroup_existence": construct_quasigroup_existence,
    "prob007_all_interval_series": construct_all_interval_series,
    "prob054_nqueens": construct_nqueens,
    "prob057_killer_sudoku": construct_killer_sudoku,
    "prob067_quasigroup_completion": construct_quasigroup_completion,
    "prob076_costas_arrays": construct_costas_arrays,
    "prob079_nqueens_completion": construct_nqueens_completion,
    "prob080_blocked_nqueens": construct_blocked_nqueens,
    "prob110_peaceably_coexisting_armies_of_queens": construct_peaceably_coexisting_armies_of_queens,
    "prob055_equidistant_frequency_permutation_arrays": construct_equidistant_frequency_permutation_arrays,
    # Sum-based
    "prob019_magic_square": construct_magic_square,
    "prob020_darts_tournament": construct_darts_tournament,
    "prob023_magic_hexagon": construct_magic_hexagon,
    "prob038_steel_mill_slab_design": construct_steel_mill_slab_design,
    "prob041_nfractions": construct_nfractions,
    "prob049_number_partitioning": construct_number_partitioning,
    "prob058_discrete_lot_sizing": construct_discrete_lot_sizing,
    "prob065_optimal_financial_portfolio": construct_optimal_financial_portfolio,
    "prob083_transshipment": construct_transshipment,
    "prob086_cvrp": construct_cvrp,
    "prob133_knapsack": construct_knapsack,
    # Count-based
    "prob001_car_sequencing": construct_car_sequencing,
    "prob012_nonogram": construct_nonogram,
    "prob014_solitaire_battleships": construct_solitaire_battleships,
    "prob030_bacp": construct_bacp,
    "prob045_covering_array": construct_covering_array,
    "prob069_balanced_nursing_workload": construct_balanced_nursing_workload,
    "prob087_rotating_rostering": construct_rotating_rostering,
    "prob022_bus_driver_scheduling": construct_bus_driver_scheduling,
    "prob_vm_pm_allocation": construct_vm_pm_allocation,
    "prob_warehouse_location": construct_warehouse_location,
}



import numpy as np
import cpmpy as cp

def flatten_vars(v):
    """
    Recursively flatten a nested structure (list, tuple, or numpy array) of CPMPy variables.
    """
    if isinstance(v, (list, tuple, np.ndarray)):
        result = []
        for item in v:
            result.extend(flatten_vars(item))
        return result
    else:
        return [v]

def run_all_models():
    print("Running all CSPLib models:\n")
    for name, constructor in CSP_LIBS.items():
        print("="*60)
        print("Model:", name)
        try:
            instance = constructor()
        except Exception as e:
            print(f"Error constructing model {name}: {e}")
            continue

        if isinstance(instance, tuple):
            *var_parts, model = instance
            decision_vars = flatten_vars(var_parts)
        else:
            model = instance
            decision_vars = []

        print("Solving model...")
        if model.solve():
            print("Solution found for model", name)
            if decision_vars:
                for v in decision_vars:
                    try:
                        print(f"  {v.name} = {v.value()}")
                    except Exception as e:
                        #print(f"  Cannot display value for {v}: {e}")
                        pass
            else:
                print("  (No decision variables to display.)")
        else:
            print("No solution found for model", name)
        print("\n")

if __name__ == '__main__':
    run_all_models()
