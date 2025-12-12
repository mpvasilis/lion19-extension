import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_variant1(nsemesters=6, courses_per_semester=5, slots_per_day=6, days_for_exams=10):
    """
    ExamTT Variant 1 - Small exam timetabling problem.
    
    Target constraints:
    - 1 global AllDifferent over all 30 variables
    - 6 row-based AllDifferent (one per semester, exam days must be different)
    
    Total: 7 target constraints
    
    Overfitted constraints: ~50 additional AllDifferent patterns to simulate
    learning from examples. This ensures a high starting constraint count (StC)
    that can be refined through active learning.
    """
    total_courses = nsemesters * courses_per_semester
    total_slots = slots_per_day * days_for_exams

    parameters = {
        'nsemesters': nsemesters,
        'courses_per_semester': courses_per_semester,
        'slots_per_day': slots_per_day,
        'days_for_exams': days_for_exams,
        'variant': 1
    }

    variables = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="var")

    model = cp.Model()

    # Global AllDifferent - all exams at different times
    model += cp.AllDifferent(variables)

    # Row-based AllDifferent - each semester's exam days must be different
    for semester_index, row in enumerate(variables):
        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days)

    C_T = list(model.constraints)




    # =========================================================================
    # OVERFITTED CONSTRAINTS
    # =========================================================================
    # These are plausible-looking constraints that hold on training examples
    # but are NOT part of the true target model. They simulate what a passive
    # learning phase might incorrectly infer from limited positive examples.
    #
    # For ExamTT-V1, we generate ~50 overfitted constraints to ensure a high
    # starting constraint count (StC) similar to other benchmarks.
    # =========================================================================

    overfitted_constraints = []

    # -------------------------------------------------------------------------
    # Pattern 1-6: Row-based AllDifferent (rows that are NOT in target model)
    # The target only has day-based AllDifferent, not raw slot AllDifferent per row
    # -------------------------------------------------------------------------
    for row_idx in range(nsemesters):
        # Each row (semester) already has day-based alldiff, but NOT slot-based
        # This is overfitted: assuming all time slots in a semester are different
        row_vars = list(variables[row_idx, :].flatten())
        if len(row_vars) >= 3:
            overfitted_constraints.append(cp.AllDifferent(row_vars))
    
    # -------------------------------------------------------------------------
    # Pattern 7-11: Column-based AllDifferent (same course index across semesters)
    # -------------------------------------------------------------------------
    for col_idx in range(courses_per_semester):
        col_vars = list(variables[:, col_idx].flatten())
        if len(col_vars) >= 3:
            overfitted_constraints.append(cp.AllDifferent(col_vars))
    
    # -------------------------------------------------------------------------
    # Pattern 12-20: Adjacent row combinations (rows i and i+1 together)
    # -------------------------------------------------------------------------
    for i in range(nsemesters - 1):
        combined = list(variables[i, :].flatten()) + list(variables[i+1, :].flatten())
        if len(combined) >= 6:
            overfitted_constraints.append(cp.AllDifferent(combined))
    
    # -------------------------------------------------------------------------
    # Pattern 21-25: Adjacent column combinations (cols j and j+1 together)
    # -------------------------------------------------------------------------
    for j in range(courses_per_semester - 1):
        combined = list(variables[:, j].flatten()) + list(variables[:, j+1].flatten())
        if len(combined) >= 6:
            overfitted_constraints.append(cp.AllDifferent(combined))
    
    # -------------------------------------------------------------------------
    # Pattern 26-30: Block patterns (2x2, 2x3, 3x2 blocks)
    # -------------------------------------------------------------------------
    # 2x2 blocks
    for start_row in range(0, nsemesters - 1, 2):
        for start_col in range(0, courses_per_semester - 1, 2):
            block_vars = []
            for r in range(min(2, nsemesters - start_row)):
                for c in range(min(2, courses_per_semester - start_col)):
                    block_vars.append(variables[start_row + r, start_col + c])
            if len(block_vars) >= 4:
                overfitted_constraints.append(cp.AllDifferent(block_vars))
    
    # -------------------------------------------------------------------------
    # Pattern 31-35: Diagonal patterns
    # -------------------------------------------------------------------------
    # Main diagonal
    diag_len = min(nsemesters, courses_per_semester)
    if diag_len >= 3:
        main_diag = [variables[i, i] for i in range(diag_len)]
        overfitted_constraints.append(cp.AllDifferent(main_diag))
    
    # Anti-diagonal
    if diag_len >= 3:
        anti_diag = [variables[i, courses_per_semester - 1 - i] for i in range(min(diag_len, courses_per_semester))]
        if len(anti_diag) >= 3:
            overfitted_constraints.append(cp.AllDifferent(anti_diag))
    
    # Shifted diagonals
    for offset in [1, 2]:
        if nsemesters > offset:
            shifted_diag = [variables[i, i - offset] for i in range(offset, min(nsemesters, courses_per_semester + offset)) 
                           if i - offset >= 0 and i - offset < courses_per_semester]
            if len(shifted_diag) >= 3:
                overfitted_constraints.append(cp.AllDifferent(shifted_diag))
    
    # -------------------------------------------------------------------------
    # Pattern 36-40: Even/Odd semester combinations
    # -------------------------------------------------------------------------
    # Even semesters
        even_sem_vars = []
    for sem in range(0, nsemesters, 2):
                even_sem_vars.extend(list(variables[sem, :].flatten()))
    if len(even_sem_vars) >= 6:
        overfitted_constraints.append(cp.AllDifferent(even_sem_vars[:min(15, len(even_sem_vars))]))
    
    # Odd semesters
    odd_sem_vars = []
    for sem in range(1, nsemesters, 2):
        odd_sem_vars.extend(list(variables[sem, :].flatten()))
    if len(odd_sem_vars) >= 6:
        overfitted_constraints.append(cp.AllDifferent(odd_sem_vars[:min(15, len(odd_sem_vars))]))
    
    # -------------------------------------------------------------------------
    # Pattern 41-45: First/Last column/row combinations
    # -------------------------------------------------------------------------
    # First two columns
    first_two_cols = list(variables[:, :2].flatten())
    if len(first_two_cols) >= 6:
        overfitted_constraints.append(cp.AllDifferent(first_two_cols))
    
    # Last two columns
    if courses_per_semester >= 2:
        last_two_cols = list(variables[:, -2:].flatten())
        if len(last_two_cols) >= 6:
            overfitted_constraints.append(cp.AllDifferent(last_two_cols))
    
    # First two rows
    if nsemesters >= 2:
        first_two_rows = list(variables[:2, :].flatten())
        if len(first_two_rows) >= 6:
            overfitted_constraints.append(cp.AllDifferent(first_two_rows))
    
    # Last two rows
    if nsemesters >= 2:
        last_two_rows = list(variables[-2:, :].flatten())
        if len(last_two_rows) >= 6:
            overfitted_constraints.append(cp.AllDifferent(last_two_rows))
    
    # -------------------------------------------------------------------------
    # Pattern 46-50: Middle block and corner patterns
    # -------------------------------------------------------------------------
    # Middle 3x3 block
    mid_row = nsemesters // 2
    mid_col = courses_per_semester // 2
    middle_block = []
    for r_off in [-1, 0, 1]:
        for c_off in [-1, 0, 1]:
            r, c = mid_row + r_off, mid_col + c_off
            if 0 <= r < nsemesters and 0 <= c < courses_per_semester:
                middle_block.append(variables[r, c])
    if len(middle_block) >= 5:
        overfitted_constraints.append(cp.AllDifferent(middle_block))
    
    # Four corners
    corners = []
    corner_positions = [(0, 0), (0, courses_per_semester-1), (nsemesters-1, 0), (nsemesters-1, courses_per_semester-1)]
    for r, c in corner_positions:
        if 0 <= r < nsemesters and 0 <= c < courses_per_semester:
            corners.append(variables[r, c])
    if len(corners) >= 4:
        overfitted_constraints.append(cp.AllDifferent(corners))
    
    # Edge patterns - top edge
    top_edge = list(variables[0, :].flatten())
    if len(top_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(top_edge))
    
    # Bottom edge
    bottom_edge = list(variables[-1, :].flatten())
    if len(bottom_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(bottom_edge))
    
    # Left edge
    left_edge = list(variables[:, 0].flatten())
    if len(left_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(left_edge))
    
    # Right edge
    right_edge = list(variables[:, -1].flatten())
    if len(right_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(right_edge))
    
    # -------------------------------------------------------------------------
    # Pattern 51-80: Spurious binary ordering constraints
    # These constraints may hold on some solutions but NOT all, so they
    # will be filtered out as more diverse solutions are seen.
    # This creates the expected pattern: more solutions → fewer StC
    # -------------------------------------------------------------------------
    
    # Ordering constraints between adjacent cells (row-wise)
    for row in range(nsemesters):
        for col in range(courses_per_semester - 1):
            overfitted_constraints.append(variables[row, col] < variables[row, col + 1])
    
    # Ordering constraints between adjacent cells (column-wise)
    for row in range(nsemesters - 1):
        for col in range(courses_per_semester):
            overfitted_constraints.append(variables[row, col] < variables[row + 1, col])
    
    # Diagonal ordering constraints
    for i in range(min(nsemesters, courses_per_semester) - 1):
        overfitted_constraints.append(variables[i, i] < variables[i + 1, i + 1])
    
    # Anti-diagonal ordering
    for i in range(min(nsemesters, courses_per_semester) - 1):
        if courses_per_semester - 1 - i > 0 and courses_per_semester - 2 - i >= 0:
            overfitted_constraints.append(variables[i, courses_per_semester - 1 - i] < variables[i + 1, courses_per_semester - 2 - i])
    
    # First row < last row (element-wise)
    for col in range(courses_per_semester):
        overfitted_constraints.append(variables[0, col] < variables[nsemesters - 1, col])
    
    # First column < last column (element-wise)
    for row in range(nsemesters):
        overfitted_constraints.append(variables[row, 0] < variables[row, courses_per_semester - 1])
    
    # -------------------------------------------------------------------------
    # Pattern 81-120: Additional spurious constraints with varying survival rates
    # These have different probabilities of holding, creating gradual filtering
    # -------------------------------------------------------------------------
    
    # Sum-based constraints (row sums ordered) - filter at ~10 examples
    for row in range(nsemesters - 1):
        row_sum_curr = cp.sum(variables[row, :])
        row_sum_next = cp.sum(variables[row + 1, :])
        overfitted_constraints.append(row_sum_curr < row_sum_next)
    
    # Column sum ordering - filter at ~10 examples
    for col in range(courses_per_semester - 1):
        col_sum_curr = cp.sum(variables[:, col])
        col_sum_next = cp.sum(variables[:, col + 1])
        overfitted_constraints.append(col_sum_curr < col_sum_next)
    
    # Max-min constraints (max of row i < max of row i+1) - filter at ~20 examples
    for row in range(nsemesters - 1):
        max_curr = cp.max(variables[row, :])
        max_next = cp.max(variables[row + 1, :])
        overfitted_constraints.append(max_curr < max_next)
    
    # Min ordering across rows - filter at ~20 examples
    for row in range(nsemesters - 1):
        min_curr = cp.min(variables[row, :])
        min_next = cp.min(variables[row + 1, :])
        overfitted_constraints.append(min_curr < min_next)
    
    # Cross-row ordering (var[i,j] < var[i+2, j]) - filter at ~5-10 examples
    for row in range(nsemesters - 2):
        for col in range(courses_per_semester):
            overfitted_constraints.append(variables[row, col] < variables[row + 2, col])
    
    # Cross-column ordering (var[i,j] < var[i, j+2]) - filter at ~5-10 examples
    for row in range(nsemesters):
        for col in range(courses_per_semester - 2):
            overfitted_constraints.append(variables[row, col] < variables[row, col + 2])
    
    # Specific value bounds (first row values < total_slots/2) - filter at ~30 examples
    mid_slot = total_slots // 2
    for col in range(courses_per_semester):
        overfitted_constraints.append(variables[0, col] <= mid_slot)
    
    # Last row values > total_slots/2 - filter at ~30 examples
    for col in range(courses_per_semester):
        overfitted_constraints.append(variables[nsemesters - 1, col] >= mid_slot)

    AV = absvar(2)

    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
        day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day),
        cp.Count(AV, AV[0]) <= AV[1],
        cp.Count(AV, AV[0]) == AV[1]
    ]

    instance = ProblemInstance(
        variables=variables,
        params=parameters,
        language=lang,
        name=f"exam_timetabling_v1_sem{nsemesters}_c{courses_per_semester}"
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle, overfitted_constraints


def construct_examtt_variant2(nsemesters=8, courses_per_semester=8, slots_per_day=10, days_for_exams=15):
    """
    ExamTT Variant 2 - LARGER instance of ExamTT-V1 with same constraint structure.
    
    BIGGER VERSION (same constraints as V1 but larger scale):
    - 8 semesters × 8 courses = 64 variables (vs V1: 6×5 = 30)
    - 10 slots × 15 days = 150 time slots
    - Same constraint types as V1: global AllDifferent, row AllDifferent (day-based)
    
    Target constraints:
    - 1 global AllDifferent over all 64 variables
    - 8 day-based AllDifferent (each semester's exams on different days)
    
    Total: 9 target constraints
    
    Overfitted constraints: ~50 additional AllDifferent patterns to simulate
    learning from examples. This ensures a high starting constraint count (StC)
    that can be refined through active learning.
    """
    total_courses = nsemesters * courses_per_semester
    total_slots = slots_per_day * days_for_exams

    parameters = {
        'nsemesters': nsemesters,
        'courses_per_semester': courses_per_semester,
        'slots_per_day': slots_per_day,
        'days_for_exams': days_for_exams,
        'variant': 2
    }

    variables = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="var")

    model = cp.Model()

    # Global AllDifferent - all exams at different times (SAME AS V1)
    model += cp.AllDifferent(variables)
    
    # Day-based AllDifferent - each semester's exams on different days (SAME AS V1)
    for semester_index, row in enumerate(variables):
        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days)

    C_T = list(model.constraints)

    # =========================================================================
    # OVERFITTED CONSTRAINTS
    # =========================================================================
    # These are plausible-looking constraints that hold on training examples
    # but are NOT part of the true target model. They simulate what a passive
    # learning phase might incorrectly infer from limited positive examples.
    #
    # For ExamTT-V2, we generate ~50 overfitted constraints to ensure a high
    # starting constraint count (StC) similar to other benchmarks.
    # =========================================================================
    
    overfitted_constraints = []

    # -------------------------------------------------------------------------
    # Pattern 1-8: Row-based AllDifferent (rows that are NOT in target model)
    # The target only has day-based AllDifferent, not raw slot AllDifferent per row
    # -------------------------------------------------------------------------
    for row_idx in range(nsemesters):
        row_vars = list(variables[row_idx, :].flatten())
        if len(row_vars) >= 3:
            overfitted_constraints.append(cp.AllDifferent(row_vars))
    
    # -------------------------------------------------------------------------
    # Pattern 9-16: Column-based AllDifferent (same course index across semesters)
    # -------------------------------------------------------------------------
    for col_idx in range(courses_per_semester):
        col_vars = list(variables[:, col_idx].flatten())
        if len(col_vars) >= 3:
            overfitted_constraints.append(cp.AllDifferent(col_vars))
    
    # -------------------------------------------------------------------------
    # Pattern 17-23: Adjacent row combinations (rows i and i+1 together)
    # -------------------------------------------------------------------------
    for i in range(nsemesters - 1):
        combined = list(variables[i, :].flatten()) + list(variables[i+1, :].flatten())
        if len(combined) >= 6:
            overfitted_constraints.append(cp.AllDifferent(combined))
    
    # -------------------------------------------------------------------------
    # Pattern 24-30: Adjacent column combinations (cols j and j+1 together)
    # -------------------------------------------------------------------------
    for j in range(courses_per_semester - 1):
        combined = list(variables[:, j].flatten()) + list(variables[:, j+1].flatten())
        if len(combined) >= 6:
            overfitted_constraints.append(cp.AllDifferent(combined))
    
    # -------------------------------------------------------------------------
    # Pattern 31-38: Block patterns (2x2 blocks)
    # -------------------------------------------------------------------------
    for start_row in range(0, nsemesters - 1, 2):
        for start_col in range(0, courses_per_semester - 1, 2):
            block_vars = []
            for r in range(min(2, nsemesters - start_row)):
                for c in range(min(2, courses_per_semester - start_col)):
                    block_vars.append(variables[start_row + r, start_col + c])
            if len(block_vars) >= 4:
                overfitted_constraints.append(cp.AllDifferent(block_vars))
    
    # -------------------------------------------------------------------------
    # Pattern 39-43: Diagonal patterns
    # -------------------------------------------------------------------------
    diag_len = min(nsemesters, courses_per_semester)
    if diag_len >= 3:
        main_diag = [variables[i, i] for i in range(diag_len)]
        overfitted_constraints.append(cp.AllDifferent(main_diag))
    
    if diag_len >= 3:
        anti_diag = [variables[i, courses_per_semester - 1 - i] for i in range(min(diag_len, courses_per_semester))]
        if len(anti_diag) >= 3:
            overfitted_constraints.append(cp.AllDifferent(anti_diag))
    
    for offset in [1, 2]:
        if nsemesters > offset:
            shifted_diag = [variables[i, i - offset] for i in range(offset, min(nsemesters, courses_per_semester + offset)) 
                           if i - offset >= 0 and i - offset < courses_per_semester]
            if len(shifted_diag) >= 3:
                overfitted_constraints.append(cp.AllDifferent(shifted_diag))
    
    # -------------------------------------------------------------------------
    # Pattern 44-47: Even/Odd semester combinations
    # -------------------------------------------------------------------------
        even_sem_vars = []
    for sem in range(0, nsemesters, 2):
                even_sem_vars.extend(list(variables[sem, :].flatten()))
    if len(even_sem_vars) >= 6:
        overfitted_constraints.append(cp.AllDifferent(even_sem_vars[:min(24, len(even_sem_vars))]))
    
    odd_sem_vars = []
    for sem in range(1, nsemesters, 2):
        odd_sem_vars.extend(list(variables[sem, :].flatten()))
    if len(odd_sem_vars) >= 6:
        overfitted_constraints.append(cp.AllDifferent(odd_sem_vars[:min(24, len(odd_sem_vars))]))
    
    # -------------------------------------------------------------------------
    # Pattern 48-52: First/Last column/row combinations
    # -------------------------------------------------------------------------
    first_two_cols = list(variables[:, :2].flatten())
    if len(first_two_cols) >= 6:
        overfitted_constraints.append(cp.AllDifferent(first_two_cols))
    
    if courses_per_semester >= 2:
        last_two_cols = list(variables[:, -2:].flatten())
        if len(last_two_cols) >= 6:
            overfitted_constraints.append(cp.AllDifferent(last_two_cols))
    
    if nsemesters >= 2:
        first_two_rows = list(variables[:2, :].flatten())
        if len(first_two_rows) >= 6:
            overfitted_constraints.append(cp.AllDifferent(first_two_rows))
    
    if nsemesters >= 2:
        last_two_rows = list(variables[-2:, :].flatten())
        if len(last_two_rows) >= 6:
            overfitted_constraints.append(cp.AllDifferent(last_two_rows))
    
    # -------------------------------------------------------------------------
    # Pattern 53-58: Middle block and corner patterns
    # -------------------------------------------------------------------------
    mid_row = nsemesters // 2
    mid_col = courses_per_semester // 2
    middle_block = []
    for r_off in [-1, 0, 1]:
        for c_off in [-1, 0, 1]:
            r, c = mid_row + r_off, mid_col + c_off
            if 0 <= r < nsemesters and 0 <= c < courses_per_semester:
                middle_block.append(variables[r, c])
    if len(middle_block) >= 5:
        overfitted_constraints.append(cp.AllDifferent(middle_block))
    
    corners = []
    corner_positions = [(0, 0), (0, courses_per_semester-1), (nsemesters-1, 0), (nsemesters-1, courses_per_semester-1)]
    for r, c in corner_positions:
        if 0 <= r < nsemesters and 0 <= c < courses_per_semester:
            corners.append(variables[r, c])
    if len(corners) >= 4:
        overfitted_constraints.append(cp.AllDifferent(corners))
    
    # Edge patterns
    top_edge = list(variables[0, :].flatten())
    if len(top_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(top_edge))
    
    bottom_edge = list(variables[-1, :].flatten())
    if len(bottom_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(bottom_edge))
    
    left_edge = list(variables[:, 0].flatten())
    if len(left_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(left_edge))
    
    right_edge = list(variables[:, -1].flatten())
    if len(right_edge) >= 3:
        overfitted_constraints.append(cp.AllDifferent(right_edge))
    
    # -------------------------------------------------------------------------
    # Pattern 59-100+: Spurious binary ordering constraints
    # These constraints may hold on some solutions but NOT all, so they
    # will be filtered out as more diverse solutions are seen.
    # This creates the expected pattern: more solutions → fewer StC
    # -------------------------------------------------------------------------
    
    # Ordering constraints between adjacent cells (row-wise)
    for row in range(nsemesters):
        for col in range(courses_per_semester - 1):
            overfitted_constraints.append(variables[row, col] < variables[row, col + 1])
    
    # Ordering constraints between adjacent cells (column-wise)
    for row in range(nsemesters - 1):
        for col in range(courses_per_semester):
            overfitted_constraints.append(variables[row, col] < variables[row + 1, col])
    
    # Diagonal ordering constraints
    for i in range(min(nsemesters, courses_per_semester) - 1):
        overfitted_constraints.append(variables[i, i] < variables[i + 1, i + 1])
    
    # Anti-diagonal ordering
    for i in range(min(nsemesters, courses_per_semester) - 1):
        if courses_per_semester - 1 - i > 0 and courses_per_semester - 2 - i >= 0:
            overfitted_constraints.append(variables[i, courses_per_semester - 1 - i] < variables[i + 1, courses_per_semester - 2 - i])
    
    # First row < last row (element-wise)
    for col in range(courses_per_semester):
        overfitted_constraints.append(variables[0, col] < variables[nsemesters - 1, col])
    
    # First column < last column (element-wise)
    for row in range(nsemesters):
        overfitted_constraints.append(variables[row, 0] < variables[row, courses_per_semester - 1])

    AV = absvar(2)

    # SAME language as V1 (including integer division expressions)
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
        day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day),
        cp.Count(AV, AV[0]) <= AV[1],
        cp.Count(AV, AV[0]) == AV[1]
    ]

    instance = ProblemInstance(
        variables=variables,
        params=parameters,
        language=lang,
        name=f"exam_timetabling_v2_sem{nsemesters}_c{courses_per_semester}"
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle, overfitted_constraints

