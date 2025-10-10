"""
Deep analysis of Nurse Rostering UNSAT issue.
"""
import cpmpy as cp

# Parameters
num_days = 7
shifts_per_day = 3
nurses_per_shift = 2
num_nurses = 10
max_workdays = 6

print("=" * 70)
print("NURSE ROSTERING UNSAT ANALYSIS")
print("=" * 70)

print(f"\nProblem parameters:")
print(f"  Days: {num_days}")
print(f"  Shifts per day: {shifts_per_day}")
print(f"  Nurses per shift: {nurses_per_shift}")
print(f"  Total nurse-shift assignments needed: {num_days * shifts_per_day * nurses_per_shift}")
print(f"  Available nurses: {num_nurses}")
print(f"  Max workdays per nurse: {max_workdays}")
print(f"  Maximum capacity: {num_nurses * max_workdays} nurse-days")
print(f"  Nurses per day: {shifts_per_day * nurses_per_shift}")
print(f"  Total nurse-days needed: {num_days * shifts_per_day * nurses_per_shift}")

print("\n" + "=" * 70)
print("TESTING CONSTRAINT COMBINATIONS")
print("=" * 70)

# Test 1: Only daily AllDifferent
print("\nTest 1: Only daily AllDifferent")
roster1 = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="var")
model1 = cp.Model()
for day in range(num_days):
    model1 += cp.AllDifferent(roster1[day, ...])
result1 = model1.solve()
print(f"  Result: {'SAT' if result1 else 'UNSAT'}")

# Test 2: Daily AllDifferent + workday limits
print("\nTest 2: Daily AllDifferent + workday limits")
roster2 = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="var")
model2 = cp.Model()
for day in range(num_days):
    model2 += cp.AllDifferent(roster2[day, ...])
for nurse in range(1, num_nurses + 1):
    model2 += cp.Count(roster2, nurse) <= max_workdays
result2 = model2.solve()
print(f"  Result: {'SAT' if result2 else 'UNSAT'}")

# Test 3: Add consecutive shift constraint (CURRENT IMPLEMENTATION)
print("\nTest 3: Add consecutive shift constraint (AllDifferent of 4 nurses)")
roster3 = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="var")
model3 = cp.Model()
for day in range(num_days):
    model3 += cp.AllDifferent(roster3[day, ...])
for nurse in range(1, num_nurses + 1):
    model3 += cp.Count(roster3, nurse) <= max_workdays
# Current problematic constraint
for day in range(num_days - 1):
    last_shift = roster3[day, shifts_per_day - 1, :]
    first_shift_next = roster3[day + 1, 0, :]
    combined = list(last_shift) + list(first_shift_next)
    model3 += cp.AllDifferent(combined)
result3 = model3.solve()
print(f"  Result: {'SAT' if result3 else 'UNSAT'}")
print(f"  Analysis: AllDifferent([last_shift_2nurses, first_shift_next_2nurses])")
print(f"           Requires 4 different nurses across consecutive shifts")

# Test 4: Replace with binary != constraints (PROPOSED FIX)
print("\nTest 4: Replace with binary != constraints (PROPOSED FIX)")
roster4 = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="var")
model4 = cp.Model()
for day in range(num_days):
    model4 += cp.AllDifferent(roster4[day, ...])
for nurse in range(1, num_nurses + 1):
    model4 += cp.Count(roster4, nurse) <= max_workdays
# Better constraint: no nurse works both last shift of day D and first shift of day D+1
for day in range(num_days - 1):
    last_shift = roster4[day, shifts_per_day - 1, :]
    first_shift_next = roster4[day + 1, 0, :]
    for i in range(nurses_per_shift):
        for j in range(nurses_per_shift):
            model4 += last_shift[i] != first_shift_next[j]
result4 = model4.solve()
print(f"  Result: {'SAT' if result4 else 'UNSAT'}")
print(f"  Analysis: Binary != between each pair of nurses")
print(f"           More lenient: allows overlaps, just no same nurse in both")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if result4:
    print("SUCCESS: Binary != constraints make the problem SAT")
    print("This is the correct interpretation of 'no nurse works consecutive shifts'")
else:
    print("FAILURE: Even binary constraints cause UNSAT")
    print("Need to reconsider problem parameters")
