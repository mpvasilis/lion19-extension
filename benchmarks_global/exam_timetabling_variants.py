import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_variant1(nsemesters=6, courses_per_semester=5, slots_per_day=6, days_for_exams=10):
    """
    Exam Timetabling Variant 1: Smaller instance with specific mock constraints.
    Focus: More aggressive day-level constraints and specific slot patterns.
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
    
    # TRUE CONSTRAINTS (part of target model)
    # All exams must be scheduled in different timeslots
    model += cp.AllDifferent(variables)

    # Exams in the same semester must be on different days
    for semester_index, row in enumerate(variables):
        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days)
    
    # Limit the number of exams per day
    all_exams = variables.flatten()
    max_exams_per_day = (total_courses + days_for_exams - 1) // days_for_exams + 1
    for day in range(days_for_exams):
        exams_on_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], day)
        model += (exams_on_day <= max_exams_per_day)

    # Save TRUE constraints for oracle
    C_T = list(model.constraints)

    # MOCK OVER-FITTED CONSTRAINTS
    mock_constraints = []

    # # Mock 1: First two semesters must have all exams on different days (overly restrictive)
    # if nsemesters >= 2:
    #     first_two_semesters = variables[:2, :].flatten()
    #     first_two_days = [day_of_exam(exam, slots_per_day) for exam in first_two_semesters]
    #     mock_c1 = cp.AllDifferent(first_two_days)
    #     mock_constraints.append(mock_c1)
    #     model += mock_c1

    # # Mock 2: Last semester exams must be in last 3 days (might hold in examples)
    # if nsemesters >= 1 and days_for_exams >= 3:
    #     last_semester_exams = variables[nsemesters-1, :]
    #     for exam in last_semester_exams:
    #         # Require exam to be in one of last 3 days
    #         last_day_start = (days_for_exams - 3) * slots_per_day + 1
    #         mock_c2 = (exam >= last_day_start)
    #         mock_constraints.append(mock_c2)
    #         model += mock_c2

    # # Mock 3: Specific day must have exactly 2 exams (very specific)
    # if days_for_exams >= 5:
    #     target_day = 3  # Day 3
    #     exams_on_target_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], target_day)
    #     mock_c3 = (exams_on_target_day == 2)
    #     mock_constraints.append(mock_c3)
    #     model += mock_c3

    # # Mock 4: Middle semester courses must be in ascending order (overly restrictive)
    # if nsemesters >= 3 and courses_per_semester >= 3:
    #     middle_sem = nsemesters // 2
    #     middle_courses = variables[middle_sem, :]
    #     # Require courses to be scheduled in increasing slot order
    #     for i in range(len(middle_courses) - 1):
    #         mock_c4 = (middle_courses[i] < middle_courses[i+1])
    #         mock_constraints.append(mock_c4)
    #         model += mock_c4

    # # Mock 5: Sum constraint on first semester slots (arbitrary pattern)
    # if nsemesters >= 1:
    #     first_sem_sum = cp.sum(variables[0, :])
    #     # Constrain sum to be within a range that might hold in examples
    #     mock_c5 = (first_sem_sum <= 100)
    #     mock_constraints.append(mock_c5)
    #     model += mock_c5

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
    
    # Return mock constraints so Phase 1 can use them as overfitted constraints
    return instance, oracle, mock_constraints


def construct_examtt_variant2(nsemesters=8, courses_per_semester=7, slots_per_day=8, days_for_exams=12):
    """
    Exam Timetabling Variant 2: Larger instance with different mock constraints.
    Focus: Cross-semester patterns and ordering constraints.
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
    
    # TRUE CONSTRAINTS (part of target model)
    # All exams must be scheduled in different timeslots
    model += cp.AllDifferent(variables)

    # Exams in the same semester must be on different days
    for semester_index, row in enumerate(variables):
        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days)
    
    # Limit the number of exams per day
    all_exams = variables.flatten()
    max_exams_per_day = (total_courses + days_for_exams - 1) // days_for_exams + 1
    for day in range(days_for_exams):
        exams_on_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], day)
        model += (exams_on_day <= max_exams_per_day)

    # Save TRUE constraints for oracle
    C_T = list(model.constraints)

    # MOCK OVER-FITTED CONSTRAINTS
    mock_constraints = []

    # # Mock 1: First course of each semester must be ordered by semester index
    # if nsemesters >= 3:
    #     first_courses = [variables[sem, 0] for sem in range(min(4, nsemesters))]
    #     for i in range(len(first_courses) - 1):
    #         mock_c1 = (first_courses[i] < first_courses[i+1])
    #         mock_constraints.append(mock_c1)
    #         model += mock_c1

    # # Mock 2: Even semesters and odd semesters have disjoint day sets (might hold accidentally)
    # if nsemesters >= 4:
    #     even_sems = variables[::2, :].flatten()
    #     odd_sems = variables[1::2, :].flatten()
    #     even_days = [day_of_exam(exam, slots_per_day) for exam in even_sems]
    #     odd_days = [day_of_exam(exam, slots_per_day) for exam in odd_sems]
        
    #     # This is overly restrictive - no day can have both even and odd semester exams
    #     # We'll encode this as: for each even exam, its day must differ from all odd exam days
    #     # This is complex, so we'll use a simpler proxy: ensure days are partitioned
    #     # For simplicity, just add an AllDifferent on a subset
    #     if len(even_days) >= 3 and len(odd_days) >= 3:
    #         mock_c2 = cp.AllDifferent(even_days[:3] + odd_days[:3])
    #         mock_constraints.append(mock_c2)
    #         model += mock_c2

    # # Mock 3: Middle 3 days must have exactly 5 exams each
    # if days_for_exams >= 7:
    #     for day in range(days_for_exams // 2 - 1, days_for_exams // 2 + 2):
    #         exams_on_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], day)
    #         mock_c3 = (exams_on_day == 5)
    #         mock_constraints.append(mock_c3)
    #         model += mock_c3

    # # Mock 4: Diagonal pattern across semesters and courses
    # if nsemesters >= 4 and courses_per_semester >= 4:
    #     diagonal_exams = [variables[i, i] for i in range(min(4, nsemesters, courses_per_semester))]
    #     diagonal_days = [day_of_exam(exam, slots_per_day) for exam in diagonal_exams]
    #     mock_c4 = cp.AllDifferent(diagonal_days)
    #     mock_constraints.append(mock_c4)
    #     model += mock_c4

    # # Mock 5: Last 2 semesters must have all exams in first half of schedule
    # if nsemesters >= 2:
    #     last_two_sems = variables[-2:, :].flatten()
    #     midpoint = total_slots // 2
    #     for exam in last_two_sems:
    #         mock_c5 = (exam <= midpoint)
    #         mock_constraints.append(mock_c5)
    #         model += mock_c5

    # # Mock 6: Count constraint on specific day (overly specific)
    # if days_for_exams >= 8:
    #     specific_day = 6
    #     exams_on_specific_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], specific_day)
    #     mock_c6 = (exams_on_specific_day >= 3)
    #     mock_constraints.append(mock_c6)
    #     model += mock_c6

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
        name=f"exam_timetabling_v2_sem{nsemesters}_c{courses_per_semester}"
    )

    oracle = ConstraintOracle(C_T)
    
    # Return mock constraints so Phase 1 can use them as overfitted constraints
    return instance, oracle, mock_constraints

