import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_nurse_rostering(shifts_per_day=3, num_days=7, num_nurses=10, nurses_per_shift=2, max_workdays=6):

    parameters = {"shifts_per_day": shifts_per_day, "num_days": num_days, "num_nurses": num_nurses,
                  "nurses_per_shift": nurses_per_shift, "max_workdays": max_workdays}

    roster_matrix = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="var")

    model = cp.Model()

    # each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += cp.AllDifferent(roster_matrix[day, ...])
        # model += cp.AllDifferent(roster_matrix[day, ...]).decompose()

    # the last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        last_shift_nurses = roster_matrix[day, shifts_per_day - 1, :]
        first_shift_next_day = roster_matrix[day + 1, 0, :]
        combined = list(last_shift_nurses) + list(first_shift_next_day)
        model += cp.AllDifferent(combined)
        # model += cp.AllDifferent(combined).decompose()

    # each nurse works at most 'max_workdays' days per week (since a nurse appears at most once per day the inner sum is 0/1)
    for nurse in range(1, num_nurses + 1):
        model += cp.Count(roster_matrix, nurse) <= max_workdays

    # Save TRUE constraints for oracle (before adding mocks)
    C_T = list(model.constraints)

    # MOCK OVER-FITTED CONSTRAINTS (will be consistent with 5 examples but NOT generally valid)
    # These MUST be AllDifferent constraints (detectable by Phase 1 pattern detection)
    # Strategy: Add AllDifferent across positions that happen to differ in examples but aren't required
    # NOTE: Mocks are added to model for example generation but NOT to oracle (C_T)
    mock_constraints = []

    # Mock 1: First shift of day 0 and first shift of day 2 nurses are all different
    # (Not required, but might hold in 5 examples by chance)
    if num_days >= 3:
        mock_positions = list(roster_matrix[0, 0, :]) + list(roster_matrix[2, 0, :])
        mock_c1 = cp.AllDifferent(mock_positions)
        mock_constraints.append(mock_c1)
        # Don't add to model to allow violations
    
    # Mock 2: Middle shift of first and last day are all different
    # (Another spurious pattern that might appear in examples)
    if num_days >= 2 and shifts_per_day >= 2:
        mid_shift = shifts_per_day // 2
        mock_positions = list(roster_matrix[0, mid_shift, :]) + list(roster_matrix[num_days-1, mid_shift, :])
        mock_c2 = cp.AllDifferent(mock_positions)
        mock_constraints.append(mock_c2)
        # Don't add to model to allow violations
    
    # Mock 3: Specific non-adjacent shifts are all different
    # (Cross-day pattern that isn't actually required)
    if num_days >= 4 and shifts_per_day >= 2:
        mock_positions = list(roster_matrix[1, 0, :]) + list(roster_matrix[3, 1, :])
        mock_c3 = cp.AllDifferent(mock_positions)
        mock_constraints.append(mock_c3)
        # Don't add to model to allow violations

    AV = absvar(2)
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1]
    ]

    instance = ProblemInstance(variables=roster_matrix, params=parameters, language=lang,
                               name=f"nurse_rostering_nurses{num_nurses}_shifts{shifts_per_day}_days{num_days}_"
                                    f"nurses_per_shift{nurses_per_shift}_max_workdays{max_workdays}")

    oracle = ConstraintOracle(C_T)

    return instance, oracle, mock_constraints

