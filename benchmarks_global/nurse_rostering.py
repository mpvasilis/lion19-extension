import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_nurse_rostering(shifts_per_day=3, num_days=7, num_nurses=8, nurses_per_shift=2, max_workdays=5):

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
        model += cp.AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0])
        # model += cp.AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()

    # each nurse works at most 'max_workdays' days per week (since a nurse appears at most once per day the inner sum is 0/1)
    for nurse in range(1, num_nurses + 1):
        model += cp.Count(roster_matrix, nurse) <= max_workdays

    C_T = list(model.constraints)

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

    return instance, oracle

