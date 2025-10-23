import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_graph_coloring(graph_type="queen_5x5", num_colors=None):
    """
    Construct a Graph Coloring problem instance.
    
    Graph coloring is a classic CP problem where we assign colors to graph vertices
    such that no two adjacent vertices have the same color.
    
    Real-world applications:
    - Register allocation in compilers
    - Scheduling (time slots for exams, meetings)
    - Frequency assignment in mobile networks
    - Map coloring
    - Sudoku (special case of graph coloring)
    
    AllDifferent constraints: For each clique in the graph, all nodes in that
    clique must have different colors (AllDifferent).
    
    Args:
        graph_type: Type of graph to color
            - "queen_5x5": 5x5 Queen graph (25 nodes, like chess queens attacking)
            - "queen_6x6": 6x6 Queen graph (36 nodes)
            - "myciel3": Mycielski graph (chromatic number 4, 11 nodes)
            - "petersen": Petersen graph (chromatic number 3, 10 nodes)
            - "register": Register allocation graph (realistic compiler use case)
        num_colors: Number of colors available (if None, uses minimum chromatic number + 1)
    
    Returns:
        (instance, oracle, mock_constraints)
    """
    
    if graph_type == "queen_5x5":
        # 5x5 Queen graph: queens attacking each other on chessboard
        # Each queen attacks: same row, column, and diagonals
        # Chromatic number: 5
        n = 5
        num_nodes = n * n
        if num_colors is None:
            num_colors = 5
        
        # Create node variables (one for each position)
        nodes = cp.intvar(1, num_colors, shape=(n, n), name="color")
        
        model = cp.Model()
        
        # AllDifferent constraints for rows (queens in same row attack each other)
        for row in nodes:
            model += cp.AllDifferent(row)
        
        # AllDifferent constraints for columns
        for col in nodes.T:
            model += cp.AllDifferent(col)
        
        # AllDifferent constraints for diagonals
        # Main diagonals (top-left to bottom-right)
        for k in range(-(n-1), n):
            diag = [nodes[i, i-k] for i in range(n) if 0 <= i-k < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)
        
        # Anti-diagonals (top-right to bottom-left)
        for k in range(2*n-1):
            diag = [nodes[i, k-i] for i in range(n) if 0 <= k-i < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)
        
        variables = nodes
        graph_name = "queen_5x5"
        
    elif graph_type == "queen_6x6":
        # 6x6 Queen graph (larger, more complex)
        n = 6
        num_nodes = n * n
        if num_colors is None:
            num_colors = 6
        
        nodes = cp.intvar(1, num_colors, shape=(n, n), name="color")
        model = cp.Model()
        
        # Same structure as 5x5
        for row in nodes:
            model += cp.AllDifferent(row)
        for col in nodes.T:
            model += cp.AllDifferent(col)
        for k in range(-(n-1), n):
            diag = [nodes[i, i-k] for i in range(n) if 0 <= i-k < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)
        for k in range(2*n-1):
            diag = [nodes[i, k-i] for i in range(n) if 0 <= k-i < n]
            if len(diag) > 1:
                model += cp.AllDifferent(diag)
        
        variables = nodes
        graph_name = "queen_6x6"
        
    elif graph_type == "register":
        # Realistic register allocation problem
        # Simulates interference graph from a code block
        # Nodes = variables in code, edges = simultaneously live variables
        if num_colors is None:
            num_colors = 4  # 4 registers available
        
        num_nodes = 12
        nodes = cp.intvar(1, num_colors, shape=num_nodes, name="register")
        
        model = cp.Model()
        
        # Define interference cliques (variables live at the same time)
        # These represent AllDifferent constraints
        cliques = [
            [0, 1, 2],           # Variables in loop header
            [1, 2, 3, 4],        # Variables in loop body
            [3, 4, 5],           # Variables in first branch
            [4, 5, 6, 7],        # Variables crossing branches
            [6, 7, 8],           # Variables in second branch
            [8, 9, 10, 11],      # Variables in loop exit
            [0, 9, 10],          # Variables spanning loop
            [2, 5, 8, 11],       # Long-lived temporaries
        ]
        
        for clique in cliques:
            if len(clique) > 1:
                model += cp.AllDifferent([nodes[i] for i in clique])
        
        variables = nodes
        graph_name = "register_allocation"
        
    elif graph_type == "scheduling":
        # Course scheduling problem
        # Schedule courses into time slots such that conflicting courses
        # (students enrolled in both) are in different slots
        if num_colors is None:
            num_colors = 5  # 5 time slots per day
        
        num_courses = 15
        courses = cp.intvar(1, num_colors, shape=num_courses, name="timeslot")
        
        model = cp.Model()
        
        # Student enrollment conflicts (students taking multiple courses)
        # Each conflict group must be scheduled in different time slots
        conflict_groups = [
            [0, 1, 2, 3],        # Freshman core courses
            [2, 3, 4, 5],        # Math sequence
            [4, 5, 6, 7, 8],     # Science courses
            [6, 7, 9, 10],       # Engineering courses
            [9, 10, 11, 12],     # Senior electives
            [11, 12, 13, 14],    # Capstone sequence
            [0, 5, 10, 14],      # Popular cross-year courses
            [1, 6, 11],          # Lab courses
            [3, 8, 13],          # Theory courses
        ]
        
        for group in conflict_groups:
            if len(group) > 1:
                model += cp.AllDifferent([courses[i] for i in group])
        
        variables = courses
        graph_name = "course_scheduling"
        
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    C_T = list(set(toplevel_list(model.constraints)))
    
    # MOCK OVER-FITTED CONSTRAINTS
    # These might appear in 5 examples but aren't actually required
    mock_constraints = []
    
    # Example mock: "These specific nodes happen to all be different in examples"
    # but this isn't a real constraint of the problem
    # (Uncomment to test Phase 2 overfitting detection)
    
    # if graph_type == "register" and num_nodes >= 8:
    #     # Mock: Happens to be true in examples but not required
    #     mock_c1 = cp.AllDifferent([nodes[0], nodes[3], nodes[6]])
    #     mock_constraints.append(mock_c1)
    #     model += mock_c1
    
    # if graph_type == "queen_5x5":
    #     # Mock: Corner positions happen to differ in examples
    #     n = 5
    #     mock_c2 = cp.AllDifferent([nodes[0,0], nodes[0,n-1], nodes[n-1,0], nodes[n-1,n-1]])
    #     mock_constraints.append(mock_c2)
    #     model += mock_c2
    
    # Create the language
    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    parameters = {"graph_type": graph_type, "num_colors": num_colors}
    instance = ProblemInstance(variables=variables, params=parameters, language=lang, name=graph_name)
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle, mock_constraints


def construct_graph_coloring_queen5():
    """5x5 Queen graph - moderate size, good for testing."""
    return construct_graph_coloring("queen_5x5", num_colors=5)


def construct_graph_coloring_register():
    """Register allocation - realistic compiler use case."""
    return construct_graph_coloring("register", num_colors=4)


def construct_graph_coloring_scheduling():
    """Course scheduling - realistic timetabling problem."""
    return construct_graph_coloring("scheduling", num_colors=5)

